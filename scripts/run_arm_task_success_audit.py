#!/usr/bin/env python3
"""Prepare/own two ARM-only fixed-100 task-success stages in one authorized allocation."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import sys
import time

import run_arm_quality_audit as quality
from prepare_arm_turn_bonus import copy_plain
from resume_baseline import allocation,validate_source,write_json

REPO,RUNTIME=quality.REPO,quality.RUNTIME
PARENT=quality.SOURCE
SOURCE=RUNTIME/'reference-arm-task-success-20260921-v2'
CONTROL=RUNTIME/'arm-turn-bonus-preparation/task-success-audit-20260921'
FILES=['openwebrl/arm_task_success.py','openwebrl/arm_terminal_budget.py','openwebrl/arm_rescue_yield.py']
SAMPLE=REPO/'openwebrl/docs/arm_c2_scaling_100.json'
TASKS=RUNTIME/'reference-arm-threeway-eval-20260914-v2/online_mind2web_monitor.jsonl'


def prepare():
    quality.prepare_source();validate_source(PARENT)
    CONTROL.mkdir(parents=True,exist_ok=True)
    tasks=TASKS
    rows=[json.loads(x) for x in tasks.read_text().splitlines()]
    sample=json.loads(SAMPLE.read_text())
    if len(rows)!=100 or [str(x['metadata']['task_id']) for x in rows]!=sample['task_ids']:
        raise ValueError('Reuse the original fixed100 cohort in exactly its original order')
    if hashlib.sha256(Path(sample['task_file']).read_bytes()).hexdigest()!=sample['task_file_sha256']:
        raise ValueError('Original benchmark task file changed')
    if not SOURCE.exists():
        shutil.copytree(PARENT,SOURCE,symlinks=True,copy_function=copy_plain,
            ignore=shutil.ignore_patterns('__pycache__','*.pyc','.git','.browser_use_sessions'))
        for name in FILES:copy_plain(REPO/name,SOURCE/name)
        copy_plain(tasks,SOURCE/'task_success_tasks.jsonl')
        config={'eval':{'defaults':dict(temperature=.8,top_p=1.,top_k=-1,max_response_len=4096),
            'datasets':[dict(name='arm-task-success',path=str(SOURCE/'task_success_tasks.jsonl'),
                input_key='prompt',metadata_key='metadata',n_samples_per_eval_prompt=1,
                custom_generate_function_path='openwebrl.arm_task_success.generate')]}}
        write_json(SOURCE/'task_success_eval.json',config)
        changed=FILES+['task_success_eval.json','task_success_tasks.jsonl']
        manifest=json.loads((SOURCE/'reference_manifest.json').read_text())
        manifest['recipe_files_sha256'].update({n:hashlib.sha256((SOURCE/n).read_bytes()).hexdigest() for n in changed})
        manifest['task_success_audit']=dict(parent=str(PARENT),changed_files=changed,optimizer_updates=0)
        write_json(SOURCE/'reference_manifest.json',manifest)
    validate_source(SOURCE)
    if any((SOURCE/n).read_bytes()!=(REPO/n).read_bytes() for n in FILES):
        raise ValueError('Freeze a new task-success source after code changes')
    if (SOURCE/'task_success_tasks.jsonl').read_bytes()!=tasks.read_bytes():raise ValueError('Cohort changed')
    return rows


def plan(job):
    rows=prepare();p=copy.deepcopy(quality.plan(job))
    p['stages']=[stage for stage in p['stages'] if stage['actor'] in ('baseline20','baseline90')]
    root=RUNTIME/f'evaluations/arm-task-success-{job}'
    p.update(output=str(root),requested_resources=dict(gpus=2,gpu_type='H200',hours=4,
        gpu_hours=8,cpus=16,memory_gib=480),teacher_max_calls=927,teacher_max_usd=30,
        audit_kind='arm_only_fixed100',browser_trajectories=206,primary_trajectories=200,
        expected_collection=dict(tasks=100,trajectories=100,smoke_trajectories=3,optimizer_updates=0))
    p['historical_controls']=json.loads((CONTROL/'historical-controls.json').read_text())
    p['historical_sft_selection']=json.loads((CONTROL/'historical-sft-selection-fixed100.json').read_text())
    p['cohort_manifest']=str(SAMPLE)
    p['cohort_sha256']=hashlib.sha256(TASKS.read_bytes()).hexdigest()
    ids=[str(x['metadata']['task_id']) for x in rows]
    for i,s in enumerate(p['stages']):
        old=s['output'];out=root/s['actor']
        s['source']=str(SOURCE);s['output']=str(out)
        s['command']=[x.replace(str(PARENT),str(SOURCE)).replace(old,str(out)) for x in s['command']]
        s['environment']={k:v.replace(str(PARENT),str(SOURCE)).replace(old,str(out)) for k,v in s['environment'].items()}
        cmd=s['command'];cmd[cmd.index('--eval-function-path')+1]='openwebrl.arm_task_success.generate_rollout'
        cmd[cmd.index('--eval-config')+1]=str(SOURCE/'task_success_eval.json')
        cmd[cmd.index('--wandb-group')+1]='ARM-task-success-fixed100'
        s.update(wandb_run_id=f'arm-task-success-{job}-{s["actor"]}',
            protocol='Fixed100 OM2W; ARM only K5 T0.8 response4096 horizon30; GPT4.1/action_history; historical controls')
        s['environment'].update(WANDB_RUN_ID=s['wandb_run_id'],BROWSER_CONCURRENCY='16',SGLANG_CONCURRENCY='16',
            OPENWEBRL_MULTIMODAL_STORAGE_DIR=str(RUNTIME/f'multimodal-scratch/task-success-{job}-{s["actor"]}'))
        s['quality_config']=dict(output=str(out),actor=s['actor'],actor_order=i,checkpoint=s['checkpoint'],
            policy_id=f'{s["actor"]}:completed{s["completed_training_iterations"]}',seed=20260921,
            tasks=str(SOURCE/'task_success_tasks.jsonl'),task_order=ids,max_steps=30,inference_timeout=120,budget=str(root/'terminal-budget'))
    return p


def summarize(p):
    root=Path(p['output']);stages={s['actor']:json.loads((Path(s['output'])/'success-summary.json').read_text()) for s in p['stages']}
    if any(s['tasks']!=100 for s in stages.values()):raise ValueError('Fixed100 stage incomplete')
    result=dict(complete=True,stages=stages,historical_controls=p['historical_controls'],historical_sft_selection=p['historical_sft_selection'],
        terminal_budget=json.loads((root/'terminal-budget/ledger.json').read_text()),
        limitation='Historical controls: different date and decoding; SFT judge mismatch. No paired causal ARM-lift estimate.')
    write_json(root/'task-success-summary.json',result)
    return result


def execute(p):
    job=p['job_id'];root=Path(p['output'])
    if os.getenv('SLURM_JOB_ID')!=job:raise ValueError('Use the authorized dedicated allocation')
    res=allocation(subprocess.check_output(['scontrol','show','job',job,'-o'],text=True),job,requested_gpus=2,maximum_hours=4)
    if res['cpus']<16 or res['allocated_memory_gib']<480:raise ValueError('Wrong allocation profile')
    root.mkdir(exist_ok=False);write_json(root/'audit-plan.json',p)
    deadline=time.monotonic()+res['maximum_seconds'];active=None
    try:
        for s in p['stages']:
            if deadline-time.monotonic()<1800:raise TimeoutError('Insufficient reserve for next actor')
            scratch=Path(s['environment']['OPENWEBRL_MULTIMODAL_STORAGE_DIR']);scratch.mkdir(parents=True,exist_ok=False)
            if shutil.disk_usage(scratch).free<2*1024**4:raise OSError('Require 2 TiB shared scratch headroom')
            command=['srun',f'--jobid={job}','--nodes=1','--ntasks=1','--cpus-per-task=16',
                '--gres=gpu:h200:2','--exact','--cpu-bind=none',sys.executable,str(Path(__file__).resolve()),
                '--job-id',job,'--worker',s['actor']]
            active=subprocess.Popen(command,start_new_session=True)
            while active.poll() is None:
                if time.monotonic()>deadline:raise TimeoutError('Allocation deadline')
                if (root/'terminal-budget/halt.json').exists():raise RuntimeError('Terminal judge budget halted')
                if (Path(s['output'])/'selector-failure.json').exists():raise RuntimeError('Selector failure; stop GPU stage')
                write_json(root/'status.json',dict(complete=False,stage=s['actor'],
                    primary_attempts=sum(len(list((Path(a['output'])/'compare_records').glob('*.json'))) for a in p['stages'])))
                time.sleep(10)
            if active.returncode:raise RuntimeError(f'{s["actor"]} audit failed; preserve all artifacts')
            active=None
        summarize(p)
        write_json(root/'status.json',dict(complete=True,stage='complete',primary_trajectories=200,
            smoke_trajectories=6,optimizer_updates=0))
    except BaseException as exc:
        write_json(root/'status.json',dict(complete=False,failed=True,error_type=type(exc).__name__,error=str(exc)[:300]))
        raise
    finally:quality.stop(active)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--job-id',default='PREPARE')
    parser.add_argument('--execute',action='store_true');parser.add_argument('--worker',choices=['baseline20','baseline90'])
    a=parser.parse_args()
    if a.worker:
        prepare();p=json.loads((RUNTIME/f'evaluations/arm-task-success-{a.job_id}/audit-plan.json').read_text())
        quality.SOURCE=SOURCE;quality.worker(p,a.worker)
    else:
        p=plan(a.job_id)
        if a.execute:execute(p)
        else:
            write_json(CONTROL/'launch-plan.json',p)
            print(json.dumps(dict(output=p['output'],resources=p['requested_resources'],
                trajectories=206,primary=200,teacher_max_usd=30),indent=2))
