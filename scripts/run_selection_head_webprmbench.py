#!/usr/bin/env python3
"""Evaluate a pinned deployed K=1 head on all released WEBPRMBench text pairs.

Preparation validates parsing and token bounds without loading actor weights.
Execution requires an exact, job-bound approval and the requested Slurm profile.
No training, provider calls, downloads, primary retries or label-based selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openwebrl.selection_head_webprmbench import (actor_prefix, choose, inference_record,
                                               pair_identity, parse_pair, prompt_key)
from openwebrl.webprm_static_bench import run_official

RESOURCES = dict(account='vlm', partition='gpu-h200', gpus=1, gpu_type='h200',
                 cpus=8, memory_gib=120, seconds=7200, maximum_allocations=1)
LIMITS = dict(api_usd=0, provider_calls=0, download_bytes=0, pairs=4600,
              states=1150, max_tokens=32768, scratch_bytes=32*1024**3,
              retained_bytes=2*1024**3, cleanup_seconds=120)


def reference(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''):
            h.update(block)
    return dict(path=str(Path(path).resolve()), sha256=h.hexdigest())


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write('\n')


def rows_and_groups(data):
    rows = [inference_record(json.loads(line)) for line in Path(data).open()]
    keys = [(r['source_name'], r['state_idx'], r['pair_idx']) for r in rows]
    states = {(a, b) for a, b, _ in keys}
    if len(rows) != 4600 or len(set(keys)) != 4600 or len(states) != 1150:
        raise ValueError('Expected the full distinct released pair inventory')
    groups = {}
    for row in rows:
        context, responses = parse_pair(row['prompt'])
        group = groups.setdefault(context, [])
        for response in responses:
            if response not in group:
                group.append(response)
    return rows, groups


def prepare(args):
    from transformers import AutoProcessor
    from openwebrl.selection_head_features import candidate_tokens
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    rows, groups = rows_and_groups(args.data)
    lengths = []
    for context, responses in groups.items():
        prefix = actor_prefix(processor.tokenizer, context)
        count = len(processor.tokenizer(prefix, add_special_tokens=False)['input_ids'])
        lengths.extend(count+len(candidate_tokens(processor.tokenizer, x)[0]) for x in responses)
    if max(lengths) > LIMITS['max_tokens']:
        raise ValueError('Context cap exceeded; truncation is not allowed')
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    actor_files = [args.model/'config.json']
    if (args.model/'model.safetensors.index.json').exists():
        actor_files.append(args.model/'model.safetensors.index.json')
    weights = sorted(args.model.glob('*.safetensors'))
    if not weights:
        raise ValueError('Local actor weights are absent')
    actor_files += weights
    actor_files += [p for p in sorted(args.model.iterdir()) if p.is_file() and
                    ('token' in p.name or 'preprocessor' in p.name or 'chat_template' in p.name)]
    manifest = dict(root=str(args.output.resolve()), model=str(args.model.resolve()),
        head=reference(args.head), data=str(args.data.resolve()), evaluator=str(args.evaluator.resolve()),
        resources=RESOURCES, limits=LIMITS, files=[reference(p) for p in dict.fromkeys(
            [args.head, args.data, args.evaluator, Path(__file__),
             Path(__file__).resolve().parents[1]/'openwebrl/selection_head_webprmbench.py',
             Path(__file__).resolve().parents[1]/'openwebrl/selection_head.py',
             Path(__file__).resolve().parents[1]/'openwebrl/selection_head_features.py',
             Path(__file__).resolve().parents[1]/'openwebrl/webprm_static_bench.py', *actor_files])],
        pairs=len(rows), states=1150, unique_contexts=len(groups),
        unique_context_response_encodings=sum(map(len, groups.values())),
        maximum_prefix_plus_response_tokens=max(lengths),
        protocol='k1-webprmbench-text-pair-v1',
        adaptation=dict(image=None, context='Released Intent through current URL; no judge criteria or labels',
            system='You are a helpful assistant.', candidates=2,
            response_format='THOUGHT/ACTION content mapped to actor think boundary',
            score='trained select logits; exact ties produce unparseable/error',
            weights='deployed all-fold teacher_k1 seed0; no benchmark tuning',
            caveat='Text-only two-candidate transfer from screenshot-conditioned five-candidate training'),
        new_allocations=0, new_provider_calls=0)
    write(args.output/'manifest.json', manifest)
    write(args.output/'resource-request.json', dict(status='prepared_not_approved',
        manifest=reference(args.output/'manifest.json'), resources=RESOURCES, limits=LIMITS,
        shared_gpu_hour_cap=100, additional_api_budget=0,
        additional_transfer_budget=0, additional_storage_allowance=0,
        purpose='All 4600 released pairs using pinned deployed student/RL K=1 head; original released metrics'))
    print(json.dumps({k: manifest[k] for k in ['pairs','states','unique_contexts',
        'unique_context_response_encodings','maximum_prefix_plus_response_tokens']}))


def execute(args):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from openwebrl.selection_head import CandidateSetHead, HeadConfig
    from openwebrl.selection_head_features import candidate_tokens, extract
    m = json.loads(args.manifest.read_text()); a = json.loads(args.approval.read_text())
    job = os.environ.get('SLURM_JOB_ID')
    if (a.get('status') != 'approved' or a.get('job_id') != job or not job
            or a.get('manifest') != reference(args.manifest)
            or a.get('resources') != RESOURCES or a.get('limits') != LIMITS):
        raise ValueError('Exact job-bound resource approval required')
    scheduler = subprocess.check_output(['scontrol','show','job','-o',job], text=True)
    if not all(x in scheduler.split() for x in ['Account=vlm','NumCPUs=8','NumNodes=1','JobState=RUNNING']):
        raise ValueError('Actual allocation profile differs')
    if 'mem=120G' not in scheduler or 'gres/gpu:h200=1' not in scheduler:
        raise ValueError('Actual allocation memory/GPU differs')
    if m['resources'] != RESOURCES or m['limits'] != LIMITS:
        raise ValueError('Frozen limits differ')
    for item in m['files']:
        if reference(item['path']) != item:
            raise ValueError('Frozen input changed: '+item['path'])
    left = subprocess.check_output(['squeue','-h','-j',job,'-o','%L'],text=True).strip()
    day, clock = left.split('-',1) if '-' in left else ('0',left)
    parts = [int(x) for x in clock.split(':')]
    if len(parts) == 2: parts.insert(0,0)
    seconds_left = int(day)*86400+parts[0]*3600+parts[1]*60+parts[2]
    if seconds_left <= LIMITS['cleanup_seconds'] or seconds_left > RESOURCES['seconds']:
        raise ValueError('No approved evaluation time remains')
    # Include every startup/verification attempt and retain cleanup reserve.
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Evaluation deadline')))
    signal.alarm(seconds_left-LIMITS['cleanup_seconds'])
    out = Path(m['root'])/('attempt-'+job);out.mkdir(mode=0o700,exist_ok=False)
    rows, groups = rows_and_groups(m['data'])
    processor = AutoProcessor.from_pretrained(m['model'],local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(m['model'],torch_dtype=torch.bfloat16,
        device_map='cuda:0',attn_implementation='sdpa',local_files_only=True).eval()
    model.requires_grad_(False)
    checkpoint = torch.load(m['head']['path'],map_location='cpu',weights_only=True)
    if checkpoint['source'] != 'teacher_k1' or checkpoint['seed'] != 0 or checkpoint['train_states'] != 1908:
        raise ValueError('Wrong deployed student/RL K=1 head')
    config = HeadConfig.from_json(checkpoint['config'])
    head = CandidateSetHead(config);head.load_state_dict(checkpoint['state_dict'],strict=True);head.eval().cuda()
    encoded = {}; predictions = []; checked = False; claimed = set()

    def generate(texts, params):
        nonlocal checked
        answers = []
        for text in texts:
            key = prompt_key(text)
            position = len(predictions)
            if position >= len(rows) or rows[position]['prompt'] != text:
                raise ValueError('Released evaluator pair order changed')
            identity = pair_identity(rows[position])
            if identity in claimed or len(claimed) >= 4600:
                raise ValueError('Primary pair already consumed or cap exceeded')
            claimed.add(identity)  # consume before any forward; keep duplicated text rows
            context, responses = parse_pair(text)
            if context not in encoded:
                prefix = processor(text=[actor_prefix(processor.tokenizer,context)],return_tensors='pt').to('cuda:0')
                candidates = [candidate_tokens(processor.tokenizer,x) for x in groups[context]]
                state, features = extract(model,prefix,candidates,config.layers,
                    pools=config.pools,batch_size=2,pad_id=processor.tokenizer.pad_token_id or 0)
                if not checked:
                    sr, fr = extract(model,prefix,candidates[:2],config.layers,
                        pools=config.pools,reuse_prefix=False,pad_id=processor.tokenizer.pad_token_id or 0)
                    rel = ((features[:2].float()-fr.float()).norm()/fr.float().norm()).item()
                    write(out/'feature-parity.json',dict(relative_error=rel,passed=rel<=.02))
                    if rel > .02:
                        raise ValueError('Cached/full feature parity failed')
                    checked = True
                encoded[context] = (state.float().cpu().flatten(),
                    {response:feat.float().cpu().flatten() for response,feat in zip(groups[context],features)})
            state, feats = encoded[context]
            batch = torch.stack([feats[x] for x in responses])[None].cuda()
            with torch.no_grad():
                logits = head(batch,state[None].cuda())[0][0].float().cpu().tolist()
                swapped = head(batch.flip(1),state[None].cuda())[0][0].flip(0).float().cpu().tolist()
            if max(abs(a-b) for a,b in zip(logits,swapped)) > 1e-4:
                raise ValueError('Pair order equivariance failed')
            pick = choose(logits)
            predictions.append(dict(identity=identity,prompt_sha256=key,scores=logits,swapped_scores=swapped,prediction=pick))
            with (out/'head-predictions.jsonl').open('a') as f:f.write(json.dumps(predictions[-1])+'\n')
            answers.append('<Answer>Response '+str(pick)+'</Answer>' if pick else 'TIE')
        print(json.dumps(dict(pairs=len(claimed),encoded_contexts=len(encoded))),flush=True)
        return answers

    metrics = run_official(m['evaluator'],model=m['head']['path'],data=m['data'],
        output=out/'official',generate=generate,batch_size=8)
    if len(claimed) != 4600 or len(predictions) != 4600:
        raise ValueError('Incomplete pair coverage')
    write(out/'complete.json',dict(job_id=job,pairs=4600,states=1150,metrics=metrics,
        head=m['head'],manifest=reference(args.manifest),provider_calls=0,downloads=0,
        caveat=m['adaptation']['caveat']))
    signal.alarm(0)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--prepare',action='store_true')
    for name in ['model','head','data','evaluator','output','manifest','approval']:
        ap.add_argument('--'+name,type=Path)
    args=ap.parse_args()
    if args.prepare:prepare(args)
    else:execute(args)


if __name__ == '__main__':main()
