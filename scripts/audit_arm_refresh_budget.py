#!/usr/bin/env python3
"""Bounded CPU audit for a baseline40 -> baseline90 ARM refresh budget."""
import gzip
import hashlib
import json
from pathlib import Path
import resource
import time
from collections import Counter

from replay_arm_confidence import archive_metadata,image_bytes,current_url,sha
from audit_arm_preference_pairs import canonical,parse_action,schemas

R=Path('/gpfs/scrubbed/zixianma/openwebrl-runtime')
C=R/'arm-turn-bonus-preparation/arm-refresh-20260924'


def audit():
    started=time.monotonic()
    root=R/'evaluations/arm-task-success-314664/baseline90'
    summary=json.loads((root/'success-summary.json').read_text())
    counts=Counter();eligible=[];tasks=Counter();call_gap=[]
    for outcome in summary['task_outcomes']:
        path=Path(outcome['arm']['artifact']);files=list((path/'selections').glob('*.json'))
        archived=archive_metadata(next(path.glob('*.pt')))
        turns=archived['turns']
        for file in files:
            record=json.loads(file.read_text());counts['saved_selection_records']+=1
            matches=[s for s in turns if s['metadata'].get('turn_index')==record['turn']
                     and sha(s['prompt'].encode())==record['prompt_sha256']]
            if len(matches)!=1:counts['no_unique_archived_state']+=1;continue
            sample=matches[0]
            try:
                if sample['response']!=record['responses'][record['selected_index']]:raise ValueError('executed_response_mismatch')
                image=image_bytes(sample,record['screenshot_sha256']);url=current_url(sample)
                schema=schemas(sample['prompt'])
                actions=[canonical(parse_action(c['action'],schema)[0]) for c in record['candidates']]
                if len(actions)!=5:raise ValueError('not_five')
                if len(set(actions))<2:raise ValueError('only_one_action')
                if any(f!='stop' for f in record['finish_types']):raise ValueError('not_stopped')
                history=sorted([s for s in turns if s['metadata'].get('turn_index',-1)<record['turn']],key=lambda s:s['metadata']['turn_index'])
                if [s['metadata']['turn_index'] for s in history]!=list(range(record['turn'])):raise ValueError('incomplete_history')
            except (KeyError,ValueError) as exc:
                counts['excluded:'+str(exc)]+=1;continue
            counts['fully_joined_eligible']+=1;tasks[record['task_id']]+=1
            eligible.append(dict(id=sha(str(file).encode()),task_id=record['task_id'],turn=record['turn'],
                record=str(file),record_sha256=sha(file.read_bytes()),archive=str(next(path.glob('*.pt'))),
                screenshot_sha256=record['screenshot_sha256'],prompt_sha256=record['prompt_sha256'],
                frozen_selected_index=record['selected_index'],frozen_permutation=record['permutation'],
                distinct_actions=len(set(actions)),task=sample['metadata']['intent'],url=url))
    selected=[];task_used=Counter()
    for row in sorted(eligible,key=lambda r:sha(('baseline90-panel-v1:'+r['id']).encode())):
        if task_used[row['task_id']]>=5:continue
        selected.append(row);task_used[row['task_id']]+=1
    (C/'baseline90-reusable-panel.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in selected))
    # Small deterministic archive sample to establish early-state availability
    # and candidate-token projections; this is not the final training manifest.
    dirs=[]
    for p in (R/'runs').glob('openwebrl-4b-reference-*/completed_rollout_archive/iteration_*'):
        try:n=int(p.name.split('_')[1])
        except ValueError:continue
        if 31<=n<=40:dirs.append(p)
    early=Counter();early_tasks=set();lengths=[]
    for directory in sorted(dirs):
        groups=sorted(directory.glob('group*.json.gz'),key=lambda p:sha(p.name.encode()))[:5]
        for path in groups:
            with gzip.open(path,'rt') as f:group=json.load(f)
            early['groups']+=1
            for trajectory in group['trajectories']:
                turns=trajectory['turns']
                if not turns:continue
                early['trajectories']+=1
                for turn in turns:
                    meta=turn['metadata'];early_tasks.add(str(meta['task_id']))
                    images=(turn.get('multimodal_inputs') or {}).get('images') or []
                    if not images:early['without_image']+=1;continue
                    image=images[-1]
                    if not isinstance(image,dict) or 'archive_image' not in image:early['unknown_image_format']+=1;continue
                    file=directory.parent/image['archive_image']
                    if not file.exists() or sha(file.read_bytes())!=image['sha256']:early['image_missing_or_changed']+=1;continue
                    if not turn['prompt'] or not meta.get('messages'):early['missing_prompt_messages']+=1;continue
                    early['turns_with_verified_current_image']+=1
                    lengths.append(turn['response_length'])
    report=dict(late=dict(counts=counts,eligible_tasks=len(tasks),panel_states=len(selected),panel_tasks=len(task_used),
            logged_selector_calls=summary['by_mode']['arm']['selector_calls'],
            note='Logged calls can exceed persisted records; only fully joined saved records are eligible.'),
        early=dict(sample_counts=early,tasks=len(early_tasks),archive_iterations=len(dirs),
            response_tokens_mean=sum(lengths)/len(lengths) if lengths else None,
            response_tokens_median=sorted(lengths)[len(lengths)//2] if lengths else None,
            note='Five groups per iteration; availability/length audit, not the final sampled training set.'),
        elapsed_seconds=time.monotonic()-started,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    (C/'baseline-budget-audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    resource.setrlimit(resource.RLIMIT_AS,(1536*1024**2,1536*1024**2))
    resource.setrlimit(resource.RLIMIT_CPU,(180,185))
    audit()
