#!/usr/bin/env python3
"""Freeze outcome-blind baseline31--40 states for fresh baseline40 candidates."""
import gzip, json, resource, time
from collections import Counter, defaultdict
from pathlib import Path
from prepare_arm_forward_transfer import group_tasks, norm, sha, stable, rows, write
from replay_arm_confidence import current_url
from openwebrl.arm_inference import split_response

R=Path('/gpfs/scrubbed/zixianma/openwebrl-runtime')
C=R/'arm-turn-bonus-preparation/arm-refresh-20260924'
OUT=C/'baseline40'
REPO=Path(__file__).resolve().parents[1]

def main():
    started=time.monotonic();OUT.mkdir(exist_ok=True)
    if (OUT/'states-manifest.json').exists():
        raise ValueError('Frozen state manifest already exists; do not overwrite')
    benchmark=list(rows(REPO/'openwebrl/data/eval/online-mind2web.jsonl'))
    excluded_intents={norm(r['metadata']['intent']) for r in benchmark}
    excluded_ids={str(v) for r in benchmark for v in (r.get('task_id'),r['metadata'].get('task_id')) if v}
    inventory=[];counts=Counter()
    paths=[]
    for p in (R/'runs').glob('openwebrl-4b-reference-*/completed_rollout_archive/iteration_*'):
        if 31<=int(p.name.split('_')[1])<=40:paths.extend(sorted(p.glob('group*.json.gz')))
    for path in sorted(paths):
        with gzip.open(path,'rt') as f:g=json.load(f)
        counts['groups']+=1
        for ti,traj in enumerate(g['trajectories']):
            turns=traj['turns'];eligible=[]
            for si,s in enumerate(turns):
                m=s['metadata'];imgs=(s.get('multimodal_inputs') or {}).get('images') or []
                ids={str(m[k]) for k in ('task_id','old_task_id') if m.get(k) is not None}
                if norm(m.get('intent','')) in excluded_intents or ids&excluded_ids:
                    counts['benchmark_overlap_turns']+=1;continue
                if not imgs or not s.get('prompt') or not m.get('messages'):continue
                if [x['metadata']['turn_index'] for x in turns[:si+1]]!=list(range(m['turn_index']+1)):continue
                if any(not isinstance(im,dict) or 'archive_image' not in im or
                       not (path.parent.parent/im['archive_image']).is_file() for im in imgs):continue
                try:current_url(s)
                except ValueError:continue
                eligible.append(si)
            if not eligible:continue
            # Select one state per collected trajectory, independent of outcome.
            si=min(eligible,key=lambda k:stable(f'{path}:{ti}:{k}'))
            s=turns[si];m=s['metadata'];img=s['multimodal_inputs']['images'][-1]
            inventory.append(dict(id=sha(f'{path}:{ti}:{si}'.encode()),source_group=str(path),
                trajectory_offset=ti,turn_offset=si,turn=m['turn_index'],
                task_key=str(m.get('old_task_id') or m['task_id']),task_id=str(m['task_id']),
                task=m['intent'],prompt_sha256=sha(s['prompt'].encode()),screenshot_sha256=img['sha256'],
                iteration=int(path.parent.name.split('_')[1]),trajectory_id=str(m['trajectory_id'])))
    group_tasks(inventory)
    selected=defaultdict(list);used=Counter();states=set()
    # 85/15 split is by task alias group; future is the separate held-out OM2W panel.
    limits={'train':2600,'dev':400}
    for r in sorted(inventory,key=lambda x:stable(x['id'])):
        part='dev' if int(stable(r['task_group'])[:8],16)%100>=85 else 'train'
        ident=(r['prompt_sha256'],r['screenshot_sha256'])
        if len(selected[part])>=limits[part] or used[r['task_group']]>=5 or ident in states:continue
        selected[part].append(dict(r,split=part));used[r['task_group']]+=1;states.add(ident)
    if len(selected['train'])<2000 or len(selected['dev'])<250:
        write(OUT/'insufficient-states.json',dict(counts=counts,inventory=len(inventory),selected={k:len(v) for k,v in selected.items()}))
        raise ValueError('Insufficient task-disjoint states under the five-state cap')
    # Interleave dev states; reaching a quota never suppresses the other split.
    ordered=[]
    for i in range(max(map(len,selected.values()))):
        for part in ('train','dev'):
            if i<len(selected[part]):ordered.append(selected[part][i])
    byfile=defaultdict(list)
    for r in ordered:byfile[r['source_group']].append(r)
    for filename,rs in byfile.items():
        path=Path(filename)
        with gzip.open(path,'rt') as f:g=json.load(f)
        for r in rs:
            turns=g['trajectories'][r['trajectory_offset']]['turns'];s=turns[r['turn_offset']]
            images=[]
            for im in s['multimodal_inputs']['images']:
                p=path.parent.parent/im['archive_image']
                if sha(p.read_bytes())!=im['sha256']:raise ValueError('Archived image SHA mismatch')
                images.append(dict(path=str(p),sha256=im['sha256']))
            if sha(s['prompt'].encode())!=r['prompt_sha256']:raise ValueError('Archived state changed')
            r.update(actor_prompt=s['prompt'],actor_images=images,screenshot=images[-1]['path'],
                history=[split_response(t['response']) for t in turns[:r['turn_offset']]],url=current_url(s),
                state_source_policy='outcome-only-baseline',state_source_iteration=r['iteration'])
    dest=OUT/'states.jsonl';dest.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in ordered))
    manifest=dict(states=str(dest),states_sha256=sha(dest.read_bytes()),counts={p:len(v) for p,v in selected.items()},
        tasks={p:len({r['task_group'] for r in v}) for p,v in selected.items()},states_count=len(ordered),
        targets={'train':2000,'dev':250},max_candidate_sets=3000,source_iterations=[31,40],
        actor_checkpoint=str(R/'runs/openwebrl-4b-reference-287371-20260911T025909/iter_0000039'),
        split='task-alias union; stable hash 85% train / 15% dev; OM2W excluded',
        all_source_intents=sorted({norm(r['task']) for r in inventory}),groups=counts['groups'],
        inventory_trajectories=len(inventory),elapsed_seconds=time.monotonic()-started,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    write(OUT/'states-manifest.json',manifest)
    print(json.dumps({k:v for k,v in manifest.items() if k!='all_source_intents'},indent=2))

if __name__=='__main__':
    resource.setrlimit(resource.RLIMIT_AS,(1536*1024**2,1536*1024**2))
    resource.setrlimit(resource.RLIMIT_CPU,(180,185))
    main()
