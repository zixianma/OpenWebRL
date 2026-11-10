#!/usr/bin/env python3
"""Build the two approved serial SFT datasets from already audited source draws."""
import hashlib
import html
import json
import os
from pathlib import Path
import resource
from collections import Counter, defaultdict

from audit_arm_preference_pairs import parse_action, schemas
from report_arm_candidate_lengths import summarize
from train_arm_joint_sft import ACTOR, DATA, REPO, RUNTIME, digest, encoded, write_json
from openwebrl.serial_actions import augment_prompt, retained_indices, serialize, parse_serial

OUTPUT = RUNTIME / 'serial-data-v1'


def main():
    resource.setrlimit(resource.RLIMIT_CPU, (180, 185))
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(ACTOR/'tokenizer.json'))
    audit = json.loads((REPO/'openwebrl/docs/arm_results/serial_sft_feasibility.json').read_text())
    for name, checksum in audit['input_hashes'].items():
        if digest(DATA/name) != checksum: raise ValueError('Source manifest changed')
    excluded = {r['id'] for r in audit['exclusions']}
    OUTPUT.mkdir(exist_ok=False)
    stats = defaultdict(Counter); measures = defaultdict(lambda:defaultdict(list)); examples = []
    tasks = defaultdict(set); ids = defaultdict(list)
    hfpath = RUNTIME/'joint-data-inputs-0d83b48/openwebrl_actor/candidates_merged.jsonl'
    streams = { (v,s):(OUTPUT/f'{v}.{s}.jsonl').open('w') for v in ['all5','diverse3'] for s in ['train','validation'] }
    image_token = tok.token_to_id('<|image_pad|>')
    merge = json.loads((ACTOR/'preprocessor_config.json').read_text())['merge_size']
    with hfpath.open('rb') as hf:
        for split in ['train','validation']:
            for line in (DATA/f'joint_pairs.{split}.jsonl').open():
                row = json.loads(line)
                if row['id'] in excluded: continue
                if row['source'] == 'C2':
                    raw = Path(row['source_path']).read_bytes()
                    assert hashlib.sha256(raw).hexdigest() == row['source_sha256']
                    source = json.loads(raw); texts = [x['response'] for x in source['raw_candidates']]
                else:
                    hf.seek(row['candidate_file_offset']); raw = hf.readline()
                    assert hashlib.sha256(raw).hexdigest() == row['candidate_line_sha256']
                    source = json.loads(raw); texts = source['candidates']
                schema = schemas(row['prompt']); candidates=[]
                for text in texts:
                    actions,_ = parse_action(text,schema)
                    candidates.append(dict(reasoning=text.split('</think>')[0].removeprefix('<think>').strip(),actions=actions))
                assert len(candidates)==5 and candidates[row['chosen_index']]['actions']==row['chosen']['actions']
                tasks[split].add(row['task_group']); ids[split].append(row['id'])
                for variant in ['all5','diverse3']:
                    keep=retained_indices(candidates,row['chosen_index'],variant,row['id'])
                    target,bounds=serialize(candidates,keep,row['chosen_index'])
                    final,history,parsed=parse_serial(target,variant)
                    assert parsed['candidates'][parsed['selected_index']]['actions']==row['chosen']['actions']
                    assert parse_action(final,schema)[0]==row['chosen']['actions']
                    prompt=augment_prompt(row['prompt'],variant)
                    if not prompt.rstrip().endswith('<think>'): raise ValueError('Missing visible thinking prefix')
                    baseids=tok.encode(prompt,add_special_tokens=False).ids
                    assert baseids.count(image_token)==len(row['image_grid_thw'])==1
                    grid=row['image_grid_thw'][0]; image_count=grid[0]*grid[1]*grid[2]//merge**2
                    prefix=[t for t in baseids for _ in range(image_count if t==image_token else 1)]
                    enc=tok.encode(target,add_special_tokens=False)
                    segments=[0 if start<bounds[0] else 1 if start<bounds[1] else 2 for start,end in enc.offsets]
                    assert set(segments)=={0,1,2} and tok.decode(enc.ids,skip_special_tokens=False)==target
                    if len(prefix)+len(enc.ids)>32768:raise ValueError('New prompt exceeds context limit')
                    result={k:row[k] for k in ['id','source','task_group','images','image_grid_thw','task','turn','label_source']}
                    result.update(prompt=prompt,prompt_tokens=len(prefix),prompt_token_sha256=hashlib.sha256(encoded(prefix).encode()).hexdigest(),
                        chosen=dict(text=target,token_ids=enc.ids,tokens=len(enc.ids),segments=segments),
                        variant=variant,retained_original_indices=keep,selected_index=keep.index(row['chosen_index']),
                        original_selected_index=row['chosen_index'],original_prompt_sha256=row['prompt_sha256'],
                        source_draw_id=row['draw_id'],source_path=row['source_path'])
                    streams[variant,split].write(encoded(result)+'\n')
                    key=f'{variant}/{split}';stats[key]['states']+=1;stats[key][row['source']]+=1;stats[key][f'K{len(keep)}']+=1
                    for name,value in [('target',len(enc.ids)),('prompt',len(prefix)),('total',len(prefix)+len(enc.ids))]:measures[key][name].append(value)
                    if split=='train' and len(examples)<32 and (variant=='all5' or len(keep)<5):
                        examples.append(dict(id=row['id'],source=row['source'],variant=variant,keep=keep,winner=row['chosen_index'],target=target,image=row['images'][0]['path']))
    for stream in streams.values():stream.close()
    assert not tasks['train']&tasks['validation']
    assert len(ids['train'])==5437 and len(ids['validation'])==644
    report=dict(passed=True,counts=dict(stats),lengths={g:{k:summarize(v) for k,v in m.items()} for g,m in measures.items()},
        split_task_groups={s:len(t) for s,t in tasks.items()},task_overlap=0,
        data_hashes={p.name:digest(p) for p in OUTPUT.glob('*.jsonl')},
        source_hashes=audit['input_hashes'],audit_exclusions=len(excluded),
        diversity_rule='winner-preserving exact deduplication, typed max-min distance, at most three; no coordinate-only equivalence merges',
        threshold=0,near_coordinate_policy='Rank small jitter below functional differences, but preserve unless displaced by the cap; DOM identities unavailable.',
        limitation='Reuse reviewed image provenance and task decontamination; never include OM2W rollout traces. Full-set teacher label retained after subset pruning.')
    write_json(OUTPUT/'verification.json',report)
    write_json(REPO/'openwebrl/docs/arm_results/serial_sft_data.json',report)
    cards=[]
    for x in examples:
        cards.append('<article><h2>'+html.escape(f"{x['id']} / {x['variant']}")+'</h2><p>'+html.escape(f"Original indices: {x['keep']}; winner {x['winner']}")+'</p><pre>'+html.escape(x['target'])+'</pre></article>')
    (REPO/'openwebrl/docs/arm_results/serial_sft_review.html').write_text('<!doctype html><meta charset="utf-8"><title>Serial SFT review</title><style>body{max-width:1100px;margin:40px auto;font:16px system-ui}pre{white-space:pre-wrap;background:#f1f5f9;padding:20px}article{margin-bottom:50px}</style><h1>Serial SFT: exact targets and retained indices</h1>'+''.join(cards))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
