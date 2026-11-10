#!/usr/bin/env python3
"""Prepare archived ARM states on CPU, then measure confidence on an approved GPU.

Preparation only reads ZIP metadata; it neither imports torch nor loads tensors.
Scoring uses the pinned full-response SelectionARM prompt and grammar decoding.
"""
import argparse
import base64
from collections import Counter, OrderedDict
import hashlib
import io
import json
import math
import os
from pathlib import Path
import pickle
import re
import statistics
import sys
import time
import zipfile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from openwebrl.arm_inference import split_response

RUNTIME = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime')
MODEL = Path('/gpfs/scrubbed/zixianma/checkpoints/web/arm/selection/81b452d800d9f859687074f82680dd5257e02d89')
BASE = Path('/gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def drop_tensor(*args):
    return None


class StoragePlaceholder:
    pass


class MetadataUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ('collections', 'OrderedDict'):
            return OrderedDict
        if (module, name) == ('torch._utils', '_rebuild_tensor_v2'):
            return drop_tensor
        if module == 'torch' and name in ('FloatStorage', 'LongStorage'):
            return StoragePlaceholder
        raise ValueError(f'Unrecognized metadata class: {module}.{name}')

    def persistent_load(self, identity):
        return None


def archive_metadata(path):
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if n.endswith('/data.pkl')]
        if len(names) != 1 or z.getinfo(names[0]).file_size > 128 * 1024**2:
            raise ValueError('Missing or oversized archive metadata')
        return MetadataUnpickler(io.BytesIO(z.read(names[0]))).load()


def current_url(sample):
    messages = sample['metadata']['messages']
    users = [m for m in messages if m['role'] == 'user']
    content = users[-1]['content']
    text = content if isinstance(content, str) else '\n'.join(c.get('text', '') for c in content)
    matches = re.findall(r'^- Tab \d+ \(active\): ([^\n]+)', text, re.M)
    if len(matches) != 1:
        raise ValueError('Cannot recover exact current URL')
    return matches[0].split(' - ', 1)[0]


def image_bytes(sample, expected):
    images = (sample.get('multimodal_inputs') or {}).get('images', [])
    for image in reversed(images):
        if isinstance(image, str) and image.startswith('data:image/'):
            raw = base64.b64decode(image.split(',', 1)[1], validate=True)
            if sha(raw) == expected:
                return raw
    raise ValueError('No archived screenshot matches saved SHA256')


def prepare(panel_path, out):
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (768*1024**2, 768*1024**2))
    resource.setrlimit(resource.RLIMIT_CPU, (180, 185))
    out.mkdir(parents=True, exist_ok=True)
    (out/'images').mkdir(exist_ok=True)
    panel = json.loads(panel_path.read_text())
    archive_path, archive, failures, prepared = None, None, [], []
    started = time.monotonic()
    for row in sorted(panel, key=lambda r: (r['archive'], r['sample_index'])):
        try:
            if archive_path != row['archive']:
                archive = None
                archive = archive_metadata(row['archive'])
                archive_path = row['archive']
            raw = Path(row['label_file']).read_bytes()
            if sha(raw) != row['label_file_sha256']:
                raise ValueError('Label file changed after audit')
            matches = [r for r in json.loads(raw)['records'] if r['turn'] == row['record_turn']]
            if len(matches) != 1:
                raise ValueError('Ambiguous label join')
            r = matches[0]
            samples = archive['samples']
            matches = [s for s in samples if s['index'] == row['sample_index']]
            if len(matches) != 1:
                raise ValueError('Ambiguous sample join')
            s = matches[0]; meta = s['metadata']
            if (str(meta['trajectory_id']) != row['trajectory_id']
                    or row['task_id'] not in {str(meta['task_id']), str(meta.get('old_task_id'))}
                    or meta['turn_index'] != row['record_turn'] or s['response'] != r['candidates'][0]['response']):
                raise ValueError('Candidate/sample identity mismatch')
            if sha(r['prompt'].encode()) != r['prompt_sha256'] or s['prompt'] != r['prompt']:
                raise ValueError('Actor prompt identity mismatch')
            history_samples = sorted([x for x in samples if str(x['metadata'].get('trajectory_id')) == row['trajectory_id']
                                      and x['metadata']['turn_index'] < row['record_turn']],
                                     key=lambda x:x['metadata']['turn_index'])
            if [x['metadata']['turn_index'] for x in history_samples] != list(range(row['record_turn'])):
                raise ValueError('Incomplete archived history')
            history = [x['response'] for x in history_samples]
            url = current_url(s)
            if 'history' in r and history != r['history']:
                raise ValueError('History reconstruction differs from logged selector input')
            if 'url' in r and url != r['url']:
                raise ValueError('URL reconstruction differs from logged selector input')
            image = image_bytes(s, r['screenshot_sha256'])
            image_path = out/'images'/f'{r["screenshot_sha256"]}.png'
            image_path.write_bytes(image)
            prepared.append(dict(**row, task=meta['intent'], url=url, history=[split_response(x) for x in history],
                                 candidates=[split_response(c['response']) for c in r['candidates']],
                                 screenshot=str(image_path), screenshot_sha256=r['screenshot_sha256'],
                                 logged_permutation=r.get('permutation'), logged_selected_index=r.get('selected_index')))
        except (ValueError, KeyError, OSError) as e:
            failures.append({'id':row['id'], 'error':str(e)})
    with (out/'inputs.jsonl').open('w') as f:
        for r in sorted(prepared, key=lambda r:r['id']):
            f.write(json.dumps(r)+'\n')
    manifest = dict(states=len(prepared), strata=dict(Counter(r['stratum'] for r in prepared)),
                    splits=dict(Counter(r['split'] for r in prepared)), failures=failures,
                    inputs_sha256=sha((out/'inputs.jsonl').read_bytes()), panel_sha256=sha(panel_path.read_bytes()),
                    elapsed_seconds=time.monotonic()-started,
                    peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    (out/'prepared.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))
    if failures or len(prepared) != 90:
        raise ValueError('Panel incomplete; inspect failures before GPU launch')


def confidence_metrics(probabilities, classes):
    if len(probabilities) != 5 or abs(sum(probabilities)-1) > 1e-5:
        raise ValueError('Expected normalized five-candidate probabilities')
    mass = {}
    for p, c in zip(probabilities, classes):
        mass[c] = mass.get(c, 0.) + p
    counts = Counter(classes)
    averaged = {c:p/counts[c] for c,p in mass.items()}
    z = sum(averaged.values()); averaged = {c:p/z for c,p in averaged.items()}
    rank = sorted(probabilities, reverse=True)
    entropy = lambda values:-sum(p*math.log(p) for p in values if p > 0)
    winner = max(mass, key=mass.get)
    return dict(candidate_max=rank[0], candidate_margin=rank[0]-rank[1],
                candidate_entropy=entropy(probabilities)/math.log(5),
                action_mass=mass, action_mass_max=mass[winner],
                action_mass_above_multiplicity_baseline=mass[winner]-counts[winner]/5,
                action_mean_probability=averaged,
                action_mean_probability_max=max(averaged.values()),
                action_entropy=entropy(mass.values())/math.log(len(mass)) if len(mass)>1 else 0.,
                action_choice_informative=len(mass)>1)


def cases(rows):
    """Natural panel plus within-state controls that copy exact full responses."""
    for r in rows:
        yield r, 'natural', list(range(5))
    controls = sorted([r for r in rows if r['stratum']=='all_distinct_actions'], key=lambda r:sha(r['id'].encode()))[:20]
    for r in controls:
        # Uses the existing task split; never tune on the validation tasks.
        yield r, 'synthetic_three_copies', [0, 0, 0, 1, 2]
        yield r, 'synthetic_five_copies', [0, 0, 0, 0, 0]


def score(inputs_path, out):
    import torch
    from PIL import Image
    from tokenizers import Tokenizer
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor
    import xgrammar as xgr
    from xgrammar.contrib.hf import LogitsProcessor
    from openwebrl.arm_inference import load_selection_builder, selection_messages, selection_schema, parse_selection
    if not torch.cuda.is_available():
        raise RuntimeError('Scoring requires an explicitly approved GPU allocation')
    out.mkdir(parents=True, exist_ok=True)
    config = AutoConfig.from_pretrained(BASE, local_files_only=True)
    released = json.loads((MODEL/'config.json').read_text())
    for key in ('hidden_size','intermediate_size','num_hidden_layers','num_attention_heads','num_key_value_heads','vocab_size'):
        if released['text_config'][key] != getattr(config.text_config, key):
            raise ValueError(f'Model/base architecture differs: {key}')
    processor = AutoProcessor.from_pretrained(BASE, local_files_only=True, min_pixels=65536, max_pixels=262144)
    if Tokenizer.from_file(str(MODEL/'tokenizer.json')).get_vocab() != processor.tokenizer.get_vocab():
        raise ValueError('Model/base vocabularies differ')
    if (MODEL/'chat_template.jinja').read_bytes() != (BASE/'chat_template.jinja').read_bytes():
        raise ValueError('Model/base chat templates differ')
    digit_ids = [processor.tokenizer.encode(str(i), add_special_tokens=False) for i in range(1,6)]
    if any(len(x)!=1 for x in digit_ids) or len({x[0] for x in digit_ids})!=5:
        raise ValueError('Selection indices do not use distinct single tokens')
    digit_ids = [x[0] for x in digit_ids]
    model = AutoModelForImageTextToText.from_pretrained(MODEL, config=config, torch_dtype=torch.bfloat16,
                    device_map='cuda:0', attn_implementation='sdpa', local_files_only=True)
    model.eval().requires_grad_(False)
    builder = load_selection_builder(RUNTIME/'arm-reproduction/source')
    grammar = xgr.GrammarCompiler(xgr.TokenizerInfo.from_huggingface(processor.tokenizer,
                    vocab_size=config.text_config.vocab_size)).compile_json_schema(selection_schema(5))
    manifest = dict(inputs_sha256=sha(inputs_path.read_bytes()), model=str(MODEL), base=str(BASE),
                    permutations=5, max_pixels=262144, context_length=32768,
                    confidence='pre-grammar next-index scores, normalized across indices 1..5',
                    script_sha256=sha(Path(__file__).read_bytes()))
    if (out/'manifest.json').exists() and json.loads((out/'manifest.json').read_text()) != manifest:
        raise ValueError('Existing score output has different provenance')
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    result_path=out/'scores.jsonl'
    done = {r['pass_id'] for r in map(json.loads, result_path.read_text().splitlines())} if result_path.exists() else set()
    rows = list(map(json.loads, inputs_path.read_text().splitlines()))
    import wandb
    from dotenv import load_dotenv
    load_dotenv(REPO/'.env', override=False)
    wandb_args=dict(project='openwebrl-evals', group='arm-confidence-audit',
                    name=f'arm-confidence-{os.environ.get("SLURM_JOB_ID", "replay")}',
                    dir=str(out), config=manifest, settings=wandb.Settings(init_timeout=30))
    try:
        tracking=wandb.init(**wandb_args)
    except Exception as exc:
        print(f'Online W&B initialization failed ({type(exc).__name__}); preserving an offline run.', flush=True)
        tracking=wandb.init(**wandb_args, mode='offline')
    (out/'wandb.json').write_text(json.dumps(dict(id=tracking.id,url=tracking.url,project='openwebrl-evals'))+'\n')
    started = time.monotonic()
    for row, treatment, source in cases(rows):
        image_raw = Path(row['screenshot']).read_bytes()
        if sha(image_raw) != row['screenshot_sha256']:
            raise ValueError('Screenshot changed')
        img = Image.open(io.BytesIO(image_raw)).convert('RGB')
        order0 = row['logged_permutation'] or list(range(5))
        for rotation in range(5):
            pass_id = f'{row["id"]}:{treatment}:{rotation}'
            if pass_id in done:
                continue
            tick = time.monotonic()
            order = order0[rotation:]+order0[:rotation]
            mapped = [source[i] for i in order]
            messages = selection_messages(builder, row['task'], row['url'], row['history'],
                                           [row['candidates'][i] for i in mapped], image_raw)
            messages[1]['content'][1]={'type':'image'}
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[img], return_tensors='pt').to('cuda:0')
            if inputs.input_ids.shape[-1]+128>32768:
                raise ValueError('Selector context exceeds validated limit')
            trace=[]
            constraint=LogitsProcessor(grammar)
            def capture_then_constrain(input_ids, scores):
                # Capture before grammar can mask or modify the tensor in place.
                raw=scores[0].float()
                trace.append((raw[digit_ids].detach().cpu().tolist(), float(raw.logsumexp(-1))))
                return constraint(input_ids, scores)
            with torch.inference_mode():
                generated=model.generate(**inputs, do_sample=False, max_new_tokens=128,
                                         logits_processor=[capture_then_constrain])
            output_ids=generated[0,inputs.input_ids.shape[-1]:].tolist()
            raw=processor.tokenizer.decode(output_ids, skip_special_tokens=True)
            selected=parse_selection(raw,5)
            indices=[i for i,x in enumerate(output_ids) if x in digit_ids]
            if len(indices)!=1 or output_ids[indices[0]]!=digit_ids[selected]:
                raise ValueError('Cannot identify unique selection-token score')
            token_index=indices[0]; logits,log_z=trace[token_index]
            peak=max(logits); unnorm=[math.exp(x-peak) for x in logits]; den=sum(unnorm)
            probs=[x/den for x in unnorm]
            if probs[selected] < max(probs)-1e-5:
                raise ValueError('Grammar selection disagrees with captured index scores')
            # Store probabilities in original five-slot order, before permutation.
            p=[0.]*5
            for i,slot in enumerate(order):p[slot]=probs[i]
            classes=[row['action_class_ids'][i] for i in source]
            record=dict(pass_id=pass_id, state_id=row['id'], task_id=row['task_id'], run=row['run'],
                        iteration_index=row['iteration_index'], stratum=row['stratum'], split=row['split'],
                        treatment=treatment, source_indices=source, permutation=order, probabilities=p,
                        action_class_ids=classes, selected_slot=order[selected], selected_source=mapped[selected],
                        selected_action_class=classes[order[selected]], raw=raw,
                        selection_prefix=processor.tokenizer.decode(output_ids[:token_index]),
                        legal_index_mass=math.exp(peak+math.log(den)-log_z),
                        logged_selected_index=row['logged_selected_index'], input_tokens=inputs.input_ids.shape[-1],
                        seconds=time.monotonic()-tick,
                        reasoning_jaccard_distance=row['reasoning_jaccard_distance'],
                        coordinate_clusters_10=row['coordinate_clusters_10'],
                        coordinate_clusters_25=row['coordinate_clusters_25'],
                        response_tokens_std=row['response_tokens_std'], **confidence_metrics(p,classes))
            with result_path.open('a') as f:f.write(json.dumps(record)+'\n')
            print(json.dumps({'completed':len(done)+1,'pass_id':pass_id,'seconds':record['seconds'],
                              'confidence':record['candidate_max']}),flush=True)
            done.add(pass_id)
            tracking.log({'audit/completed_passes':len(done), 'audit/selector_seconds':record['seconds'],
                          'audit/candidate_confidence':record['candidate_max'],
                          'audit/candidate_entropy':record['candidate_entropy'],
                          'audit/legal_index_mass':record['legal_index_mass']})
    expected=sum(5 for _ in cases(rows))
    if len(done)!=expected:raise ValueError('Incomplete score panel')
    (out/'complete.json').write_text(json.dumps(dict(passes=len(done),elapsed_seconds=time.monotonic()-started))+'\n')
    tracking.finish()


def ranks(values):
    order=sorted(range(len(values)), key=values.__getitem__)
    result=[0.]*len(values);i=0
    while i<len(order):
        j=i+1
        while j<len(order) and values[order[j]]==values[order[i]]:j+=1
        for k in order[i:j]:result[k]=(i+j-1)/2
        i=j
    return result


def spearman(rows, left, right):
    if len(rows)<3:return None
    a=ranks([r[left] for r in rows]);b=ranks([r[right] for r in rows])
    am=statistics.mean(a);bm=statistics.mean(b)
    den=math.sqrt(sum((x-am)**2 for x in a)*sum((y-bm)**2 for y in b))
    return sum((x-am)*(y-bm) for x,y in zip(a,b))/den if den else None


def report(out):
    import html
    from collections import defaultdict
    if not (out/'complete.json').exists():raise ValueError('Scoring is not complete')
    passes=list(map(json.loads,(out/'scores.jsonl').read_text().splitlines()))
    groups=defaultdict(list)
    for r in passes:groups[(r['state_id'],r['treatment'])].append(r)
    states=[]
    for key,rows in groups.items():
        if len(rows)!=5:raise ValueError('Incomplete permutation block')
        r=rows[0];p=[statistics.mean(x['probabilities'][j] for x in rows) for j in range(5)]
        metrics=confidence_metrics(p,r['action_class_ids'])
        u=len(set(r['action_class_ids']))
        states.append(dict(state_id=key[0], treatment=key[1], stratum=r['stratum'], split=r['split'],
                           task_id=r['task_id'], run=r['run'], unique_actions=u,
                           candidate_stability=max(Counter(x['selected_slot'] for x in rows).values())/5,
                           action_stability=max(Counter(x['selected_action_class'] for x in rows).values())/5,
                           mean_raw_confidence=statistics.mean(x['candidate_max'] for x in rows),
                           class_excess_confidence=(metrics['action_mean_probability_max']-1/u)/(1-1/u) if u>1 else None,
                           reasoning_jaccard_distance=r['reasoning_jaccard_distance'],
                           coordinate_clusters_10=r['coordinate_clusters_10'],
                           coordinate_clusters_25=r['coordinate_clusters_25'],
                           response_tokens_std=r['response_tokens_std'],
                           input_tokens=statistics.mean(x['input_tokens'] for x in rows), **metrics))
    natural=[r for r in states if r['treatment']=='natural']
    summary={'states':len(states),'natural_states':len(natural),'passes':len(passes),
             'limitations':['Confidence is conditional model certainty, not calibrated correctness.',
                 'Permutation stability is reproducibility, not accuracy; it cannot validate a training threshold by itself.',
                 'Correlations are exploratory; the panel is balanced by diversity, not population representative.',
                 'Identical-action sets have no action choice; action confidence/stability are mechanically one.'],
             'strata':{}, 'threshold_stability_only':[]}
    fields=['candidate_max','candidate_entropy','action_mean_probability_max','action_mass_above_multiplicity_baseline',
            'candidate_stability','action_stability','reasoning_jaccard_distance','response_tokens_std','input_tokens']
    for name in sorted({r['stratum'] for r in natural}):
        rows=[r for r in natural if r['stratum']==name]
        summary['strata'][name]={'n':len(rows),**{k:statistics.mean(r[k] for r in rows) for k in fields}}
    corr=lambda rows:{feature:{metric:spearman(rows,feature,metric) for metric in ['candidate_max','candidate_entropy','candidate_stability']}
                      for feature in ['unique_actions','coordinate_clusters_10','coordinate_clusters_25',
                                      'reasoning_jaccard_distance','response_tokens_std','input_tokens']}
    summary['spearman_pooled']=corr(natural)
    summary['spearman_within_stratum']={s:corr([r for r in natural if r['stratum']==s]) for s in summary['strata']}
    summary['spearman_within_run']={s:corr([r for r in natural if r['run']==s]) for s in sorted({r['run'] for r in natural})}
    baseline={(r['state_id']):r for r in natural}
    summary['synthetic_paired_deltas']={}
    for t in ['synthetic_three_copies','synthetic_five_copies']:
        rows=[r for r in states if r['treatment']==t]
        summary['synthetic_paired_deltas'][t]={'n':len(rows),**{k:statistics.mean(r[k]-baseline[r['state_id']][k] for r in rows) for k in fields[:6]}}
    matches=[r['selected_source']==r['logged_selected_index'] for r in passes
             if r['treatment']=='natural' and r['pass_id'].endswith(':0') and r['logged_selected_index'] is not None]
    summary['logged_greedy_reproduction']={'n':len(matches),'matched':sum(matches)}
    for split in ['development','validation']:
        rows=[r for r in natural if r['split']==split and r['action_choice_informative']]
        for threshold in [0.,.25,.5,.75,.9]:
            keep=[r for r in rows if r['class_excess_confidence']>=threshold]
            summary['threshold_stability_only'].append(dict(split=split,threshold=threshold,n=len(keep),
                coverage=len(keep)/len(rows) if rows else 0,
                mean_action_stability=statistics.mean(r['action_stability'] for r in keep) if keep else None))
    # Deliberately no automatic "best" threshold: independent preference correctness is not labeled.
    (out/'confidence-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (out/'state-summary.jsonl').open('w') as f:
        for r in states:f.write(json.dumps(r)+'\n')
    page=['<!doctype html><meta charset="utf-8"><title>ARM confidence replay</title>',
          '<style>body{font:16px system-ui;margin:40px;max-width:1200px}table{border-collapse:collapse}td,th{padding:10px;border-bottom:1px solid #ddd}pre{white-space:pre-wrap}</style>',
          '<h1>ARM confidence vs candidate diversity</h1><p>Five orderings per state. Confidence describes model preference, not correctness.</p>',
          '<table><tr><th>Natural candidates</th><th>States</th><th>Candidate confidence</th><th>Candidate entropy</th><th>Order stability</th></tr>']
    for name,r in summary['strata'].items():
        page.append(f'<tr><td>{html.escape(name)}</td><td>{r["n"]}</td><td>{r["candidate_max"]:.3f}</td><td>{r["candidate_entropy"]:.3f}</td><td>{r["candidate_stability"]:.3f}</td></tr>')
    page+=['</table><h2>Correlations, paired controls, and threshold coverage</h2><pre>'+html.escape(json.dumps(summary,indent=2))+'</pre>']
    (out/'confidence-report.html').write_text('\n'.join(page))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command', required=True)
    a=sub.add_parser('prepare');a.add_argument('--panel',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    a=sub.add_parser('score');a.add_argument('--inputs',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    a=sub.add_parser('report');a.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.command=='prepare':prepare(a.panel,a.output)
    elif a.command=='score':score(a.inputs,a.output)
    else:report(a.output)
