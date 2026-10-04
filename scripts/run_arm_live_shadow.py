#!/usr/bin/env python3
"""Bounded real-browser executed-turn hook diagnostic; never trains the actor."""
import argparse
import hashlib
import json
from pathlib import Path

from run_arm_sol_gpu_diagnostic import training_plan
from run_arm_turn_bonus_cycles import SOURCE,REPO,RUNTIME,execute

IDS=('140424','40675','44555','120487')


def prepare():
    import pyarrow.parquet as pq
    from openwebrl.arm_turn_bonus import exclusion_sets,task_identity
    original=SOURCE/'openwebrl/data/webgym_filtered_popular_2102_cleaned.parquet'
    table=pq.read_table(original); rows=table.to_pylist()
    positions={str(row['metadata']['task_id']).split('/')[-1]:i for i,row in enumerate(rows)}
    selected=table.take([positions[key] for key in IDS])
    excluded_ids,excluded_intents=exclusion_sets(json.loads(x) for x in
        (REPO/'openwebrl/data/eval/online-mind2web.jsonl').read_text().splitlines())
    for row in selected.to_pylist():
        key,intent=task_identity(row)
        if key in excluded_ids or intent in excluded_intents: raise ValueError('Evaluation overlap in live diagnostic')
    path=RUNTIME/'arm-turn-bonus-preparation/live-shadow-fast4.parquet'
    pq.write_table(selected,path)
    return path


def plan(job,provider,minutes,attempt):
    if attempt<1: raise ValueError('Use a separate numbered attempt directory')
    path=prepare()
    p=training_plan(job,minutes,interactive=True,attempt=attempt)
    p.update(diagnostic_live_shadow=True,requested_iterations=1,selector_provider=provider,
        diagnostic_data=dict(task_ids=list(IDS),path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            selection='Four short, historically mixed-outcome training tasks from pilot 293194; not a benchmark sample'))
    p['arm_config'].update(shadow_only=True,train_after_calibration=False,synthetic_rewards_and_labels=False,
        minimum_cycle_seconds=600)
    if provider=='sol':
        p['arm_config'].update(selector_checkpoint='gpt-5.6-sol',selector_provider='openai-responses',reasoning_effort='medium')
    p['environment'].pop('ARM_GPU_DIAGNOSTIC',None)
    p['environment'].update(NUM_ROLLOUT='1',ROLLOUT_BATCH_SIZE='4',GLOBAL_BATCH_SIZE='1',TRAIN_DATA=str(path))
    for key,value in [('--num-rollout','1'),('--rollout-batch-size','4'),('--global-batch-size','1'),
            ('--prompt-data',str(path)),('--rollout-function-path','slime.rollout.sglang_rollout.generate_rollout')]:
        p['command'][p['command'].index(key)+1]=value
    p['limitation']='Real browser tasks and teacher labels, q=.2/K=5; shadow only. Small selected sample does not validate statistical calibration or policy quality.'
    return p


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-id',required=True); parser.add_argument('--provider',choices=['arm','sol'],default='sol')
    parser.add_argument('--minutes',type=int,default=20); parser.add_argument('--attempt',type=int,default=1)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args(); p=plan(args.job_id,args.provider,args.minutes,args.attempt)
    if args.execute: execute(p)
    else: print(json.dumps(p,indent=2))


if __name__=='__main__': main()
