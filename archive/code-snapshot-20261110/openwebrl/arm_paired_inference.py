"""Fresh baseline and RL-task SelectionARM under the original ARM protocol."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import random

from openwebrl import arm_controlled_inference as core

MODES = ('arm', 'piotr', 'actor0')
MODEL = 'davidan03/openwebrl-4b-selectionarm-rltasks'
REVISION = 'a1f8d265855ddf1074eae91516c6e86f6f836177'
PIOTR_REVISION = '81b452d800d9f859687074f82680dd5257e02d89'
POLICY_SHA256 = '7028b29a14e6be1ff05e36a7ae708a89ab32529efad6cc47d2a388e531449aa9'
COHORT_REVISION = 'historical-system-policy-restored-v1'
PROTOCOL = dict(core.PROTOCOL, selector_model=MODEL, selector_revision=REVISION,
    piotr_selector_model='PTeterwak/OpenWebRL-4B-SelectionARM',piotr_selector_revision=PIOTR_REVISION,
    actor_revision='15e777db2ddba2e0e82080ebccd3ad8d215b7f0a', seed=42,
    sampling_request=dict(temperature=.7, top_p=.9, max_new_tokens=1024),
    top_k='omitted, original native generate endpoint default',
    design='one fresh baseline, Piotr-ARM and new-ARM episode per task',
    historical_reference='ARM_SUMMARY first table: 30.0% baseline and42.7% SelectionARM',
    actor_system_policy_sha256=POLICY_SHA256, cohort_revision=COHORT_REVISION,
    seed_realization='Request seeds recorded; FlashInfer deterministic inference remains disabled, as in historical collection.',
    comparability='Matched fresh pair; historical collection dates/runtime differ. Judge metering adds4096 output cap.')


def schedule(task_ids):
    if len(task_ids) != len(set(task_ids)):
        raise ValueError('Duplicate task IDs')
    ordered=sorted(task_ids,key=lambda t:hashlib.sha256(('42/task/'+t).encode()).hexdigest())
    rows=[]
    for index,task in enumerate(ordered):
        modes=list(MODES);random.Random('42/order/'+task).shuffle(modes)
        rows.append(dict(task_id=task,shard=index%2,attempts=[dict(mode=m,seed=42,
            key=hashlib.sha256((REVISION+'/'+COHORT_REVISION+'/'+task+'/'+m).encode()).hexdigest()) for m in modes]))
    return rows


def summarize(records, task_ids, require_complete=False):
    expected={(t,m) for t in task_ids for m in MODES}; indexed={}
    for row in records:
        key=(row['task_id'],row['mode'])
        if key not in expected or key in indexed:raise ValueError('Unexpected or duplicate result')
        indexed[key]=row
    complete=len(indexed)==len(expected)
    if require_complete and not complete:raise ValueError('Incomplete fresh pair')
    result=dict(complete=complete,scheduled_tasks=len(task_ids),scheduled_attempts=len(expected),
                completed_attempts=len(records),by_mode={})
    metrics=('actor_requests','actor_output_tokens','actor_input_tokens','actor_client_request_seconds',
        'browser_steps','browser_step_seconds','selector_calls','selector_input_tokens','selector_output_tokens',
        'selector_service_seconds','elapsed_seconds')
    yes=lambda r:int(r['valid'] and r['reward']==1)
    for mode in MODES:
        rows=[r for r in records if r['mode']==mode];valid=sum(r['valid'] for r in rows);wins=sum(map(yes,rows))
        result['by_mode'][mode]=dict(completed=len(rows),successes=wins,valid=valid,invalid=len(rows)-valid,
            overall=wins/len(task_ids) if complete else None,valid_only=wins/valid if valid else None,
            **{k:sum(r[k] for r in rows) for k in metrics})
    if complete:
        result['paired']={}
        for first,second in [('arm','actor0'),('piotr','actor0'),('arm','piotr')]:
            differences=[yes(indexed[t,first])-yes(indexed[t,second]) for t in task_ids]
            common=[t for t in task_ids if all(indexed[t,m]['valid'] for m in (first,second))]
            rng=random.Random(42)
            draws=sorted(sum(rng.choice(differences) for _ in differences)/len(differences) for _ in range(10000))
            result['paired'][first+'_minus_'+second]=dict(difference=sum(differences)/len(differences),
                bootstrap95=[draws[249],draws[9749]],first_only=differences.count(1),second_only=differences.count(-1),
                common_valid_tasks=len(common),common_valid_first_successes=sum(yes(indexed[t,first]) for t in common),
                common_valid_second_successes=sum(yes(indexed[t,second]) for t in common))
    result['cost_limitations']='Request latencies overlap; allocation GPU-hours and device logs are campaign totals, not per-arm GPU kernel measurements.'
    result['execution_phases']=execution_phases(records,task_ids)
    return result


def execution_phases(records, task_ids):
    """Expose logging repairs without discarding early outcomes or their cost.

    The common-phase view includes a task only when all three committed episodes
    used that phase. It is a sensitivity view, never a replacement denominator.
    """
    indexed={(r['task_id'],r['mode']):r for r in records}
    phase=lambda r:r.get('artifact_io_policy','synchronous_receipts_v0')
    phases=sorted({phase(r) for r in records})
    metrics=('elapsed_seconds','actor_client_request_seconds','actor_input_tokens',
             'actor_output_tokens','selector_input_tokens','selector_output_tokens')
    def describe(rows):
        return dict(episodes=len(rows),successes=sum(r['valid'] and r['reward']==1 for r in rows),
            invalid=sum(not r['valid'] for r in rows),
            mean={k:sum(r[k] for r in rows)/len(rows) if rows else None for k in metrics})
    result={}
    for name in phases:
        matched=[t for t in task_ids if all((t,m) in indexed and phase(indexed[t,m])==name for m in MODES)]
        result[name]=dict(by_mode={m:describe([r for r in records if r['mode']==m and phase(r)==name]) for m in MODES},
            common_phase_tasks=len(matched),common_phase_by_mode={m:describe([indexed[t,m] for t in matched]) for m in MODES})
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True,type=Path);a=p.parse_args()
    from openwebrl import run_evaluate  # Keep preserved event-loop initialization.
    asyncio.run(core.run(json.loads(a.config.read_text()),protocol=PROTOCOL,summary_fn=summarize))


if __name__=='__main__':main()
