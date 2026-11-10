#!/usr/bin/env python3
"""Test production candidate gates against a frozen CPU audit and saved GPU replay."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import resource
import statistics
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from openwebrl.arm_turn_bonus import eligible_actions,candidate_bonus,unit_bonus
from types import SimpleNamespace


def fingerprint(path):
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):digest.update(block)
    return dict(path=str(path.resolve()),sha256=digest.hexdigest())


def run(audit,replay,output):
    start=time.monotonic();last_path=None;labels={};counts=defaultdict(Counter)
    summary=json.loads((audit/'summary.json').read_text())
    for line in (audit/'state_metrics.jsonl').open():
        row=json.loads(line);path=row['label_file']
        if path!=last_path:
            raw=Path(path).read_bytes()
            if hashlib.sha256(raw).hexdigest()!=row['label_file_sha256']:raise ValueError('Archived labels changed')
            labels={r['turn']:r for r in json.loads(raw)['records']};last_path=path
        r=labels[row['record_turn']]
        outputs=[(c['response'],c['response_token_ids'],None,c['finish_type']) for c in r['candidates']]
        counts[row['run']]['candidate_sets']+=1
        for threshold,name in [(5,'distinct5'),(2,'min2')]:
            try:
                actions=eligible_actions(r['prompt'],outputs,threshold)
            except (ValueError,TypeError,KeyError) as e:
                counts[row['run']][name+'/rejected/'+str(e)[:80]]+=1
            else:
                counts[row['run']][name+'/passed']+=1
                if name=='min2':counts[row['run']]['min2/unique_actions/'+str(len(set(actions)))]+=1
    results={}
    for name,c in counts.items():
        old=c['distinct5/passed'];new=c['min2/passed'];base=summary['runs'][name]
        attempted=sum(base['sampled_turn_dispositions'].values());turns=sum(base['all_turn_dispositions'].values())
        results[name]=dict(counts=dict(c),sampled_turns=attempted,recorded_turns=turns,
            old_gate_passes=old,new_gate_passes=new,newly_eligible=new-old,
            gate_pass_multiplier=new/old,old_gate_fraction_of_attempts=old/attempted,
            new_gate_fraction_of_attempts=new/attempted)
    inputs={r['id']:r for r in map(json.loads,(replay.parent/'inputs.jsonl').read_text().splitlines())}
    scored=[];seen=set()
    for r in map(json.loads,(replay/'scores.jsonl').read_text().splitlines()):
        if r['treatment']!='natural':continue
        inp=inputs[r['state_id']]
        if inp['stratum']=='all_same_action':continue
        if r['pass_id'] in seen:raise ValueError('Duplicate saved replay pass')
        seen.add(r['pass_id'])
        # Classes were computed using the same strict parser; gate is independently retested above.
        label=dict(eligible=True,policy_id='offline-min2-check',candidate_gate='min2',
            executed_index=0,selected_index=r['selected_source'],
            **candidate_bonus(inp['action_class_ids'],r['selected_source'],'action_class'))
        sample=SimpleNamespace(metadata={'arm_turn_bonus':json.loads(json.dumps(label))},remove_sample=False)
        value=unit_bonus(sample,'offline-min2-check')
        index_label=dict(label,**candidate_bonus(inp['action_class_ids'],r['selected_source'],'response_index'))
        index_sample=SimpleNamespace(metadata={'arm_turn_bonus':json.loads(json.dumps(index_label))},remove_sample=False)
        index_value=unit_bonus(index_sample,'offline-min2-check')
        scored.append(dict(state_id=r['state_id'],stratum=inp['stratum'],rotation=int(r['pass_id'].rsplit(':',1)[1]),
            unit_bonus=value,scaled_bonus=.5*value,selected_index=r['selected_source'],
            response_index_unit_bonus=index_value,response_index_scaled_bonus=.5*index_value,
            duplicate_equivalent_positive=(r['selected_source']!=0 and label['selected_executed_action'])))
    model={}
    for stratum in ['all_distinct_actions','partially_duplicated_actions']:
        rows=[r for r in scored if r['stratum']==stratum]
        rotations=defaultdict(set)
        for r in rows:rotations[r['state_id']].add(r['rotation'])
        if len(rotations)!=30 or any(v!=set(range(5)) for v in rotations.values()):
            raise ValueError('Incomplete frozen replay stratum')
        model[stratum]=dict(states=len({r['state_id'] for r in rows}),passes=len(rows),
            positive_labels=sum(r['unit_bonus']>0 for r in rows),negative_labels=sum(r['unit_bonus']<0 for r in rows),
            equivalent_alternative_positive=sum(r['duplicate_equivalent_positive'] for r in rows),
            response_index_positive_labels=sum(r['response_index_unit_bonus']>0 for r in rows),
            response_index_bonus_rms=statistics.mean(r['response_index_scaled_bonus']**2 for r in rows)**.5,
            bonus_min=min(r['scaled_bonus'] for r in rows),bonus_max=max(r['scaled_bonus'] for r in rows),
            bonus_mean=statistics.mean(r['scaled_bonus'] for r in rows),
            bonus_rms=statistics.mean(r['scaled_bonus']**2 for r in rows)**.5)
    old=sum(r['old_gate_passes'] for r in results.values());new=sum(r['new_gate_passes'] for r in results.values())
    attempted=sum(r['sampled_turns'] for r in results.values())
    report=dict(runs=results,total=dict(old_gate_passes=old,new_gate_passes=new,newly_eligible=new-old,
        sampled_turns=attempted,old_gate_fraction=old/attempted,new_gate_fraction=new/attempted,multiplier=new/old),
        saved_gpu_replay=model,new_gpu_calls=0,new_actor_calls=0,
        candidate_gate='min2',credit_assignment='action_class',comparison_credit_assignment='response_index',
        beta=.5,q=.2,k=5,selector_calls_per_eligible_state=1,
        provenance={name:fingerprint(path) for name,path in dict(
            audit_summary=audit/'summary.json',audit_states=audit/'state_metrics.jsonl',
            replay_inputs=replay.parent/'inputs.jsonl',replay_scores=replay/'scores.jsonl',
            production_gate=Path(__file__).resolve().parents[1]/'openwebrl/arm_turn_bonus.py',
            audit_script=Path(__file__).resolve()).items()},
        limitation='Gate pass counts precede selector availability and group retention. Saved replay is balanced, not a population reward-scale estimate or evidence of RL gains.',
        wall_seconds=time.monotonic()-start,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    output.mkdir(parents=True,exist_ok=True)
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    (output/'replayed-bonuses.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in scored))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',type=Path,required=True);p.add_argument('--replay',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    resource.setrlimit(resource.RLIMIT_AS,(768*1024**2,768*1024**2))
    resource.setrlimit(resource.RLIMIT_CPU,(180,185))
    run(a.audit,a.replay,a.output)
