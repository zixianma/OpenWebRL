#!/usr/bin/env python3
"""Prepare/own a matched frozen-versus-refreshed ARM browser comparison. Never submits."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import run_arm_quality_audit as quality
from prepare_arm_turn_bonus import copy_plain
from resume_baseline import allocation, validate_source, write_json
from check_storage_quota import check_storage_quota

REPO,RUNTIME=quality.REPO,quality.RUNTIME
PARENT=RUNTIME/'reference-arm-task-success-20260921-v2'
SOURCE=RUNTIME/'reference-arm-refresh-browser-20260926-v1'
CONTROL=RUNTIME/'arm-turn-bonus-preparation/arm-refresh-20260924/browser'
REFRESH=CONTROL.parent/'baseline40'
ADAPTER_SHA='c06bcad3ee6e64f58a121a63f93f61e30117936025edf451904fd474e0cf7d97'
FILES=['openwebrl/arm_task_success.py','openwebrl/arm_terminal_budget.py','scripts/serve_arm.py']
LIMITATION='Fresh frozen-versus-refreshed selector comparison on the same actor and task cohort. Browser/actor/judge stochasticity and sequential collection remain; no fresh actor-only or equal-compute control.'


def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def prepare():
    CONTROL.mkdir(parents=True,exist_ok=True)
    validate_source(PARENT)
    sample=read(REPO/'openwebrl/docs/arm_c2_scaling_100.json')
    task_file=Path(sample['task_file'])
    if sha(task_file)!=sample['task_file_sha256']:raise ValueError('Benchmark changed')
    original=[json.loads(x) for x in task_file.read_text().splitlines()]
    by_id={r['metadata']['task_id']:r for r in original}
    ids=sample['task_ids']+[r['metadata']['task_id'] for r in original if r['metadata']['task_id'] not in sample['task_ids']]
    if len(ids)!=300 or len(set(ids))!=300:raise ValueError('Expected original300 tasks')
    rows=[dict(prompt=[dict(role='user',content=by_id[t]['metadata']['intent'])],metadata=by_id[t]['metadata']) for t in ids]
    if sha(REFRESH/'adapter/adapter_model.safetensors')!=ADAPTER_SHA:raise ValueError('Final90-update ARM changed')
    if not SOURCE.exists():
        shutil.copytree(PARENT,SOURCE,symlinks=True,copy_function=copy_plain,
            ignore=shutil.ignore_patterns('__pycache__','*.pyc','.git','.browser_use_sessions'))
        for name in FILES:copy_plain(REPO/name,SOURCE/name)
        (SOURCE/'task_success_tasks.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        config=read(PARENT/'task_success_eval.json')
        config['eval']['datasets'][0]['path']=str(SOURCE/'task_success_tasks.jsonl')
        write_json(SOURCE/'task_success_eval.json',config)
        manifest=read(SOURCE/'reference_manifest.json')
        changed=FILES+['task_success_tasks.jsonl','task_success_eval.json']
        manifest['recipe_files_sha256'].update({n:sha(SOURCE/n) for n in changed})
        manifest['arm_refresh_browser']=dict(parent=str(PARENT),changed_files=changed,optimizer_updates=0)
        write_json(SOURCE/'reference_manifest.json',manifest)
    validate_source(SOURCE)
    if any(sha(REPO/n)!=sha(SOURCE/n) for n in FILES):raise ValueError('Freeze new source after changes')
    if [json.loads(x) for x in (SOURCE/'task_success_tasks.jsonl').read_text().splitlines()]!=rows:raise ValueError('Cohort mismatch')
    return ids


def plan(job='PREPARE'):
    ids=prepare();old=read(RUNTIME/'evaluations/arm-task-success-314664/audit-plan.json')
    template=next(s for s in old['stages'] if s['actor']=='baseline90')
    root=RUNTIME/f'evaluations/arm-refresh-browser-{job}'
    stages=[]
    for arm in ('frozen','adapted'):
        out=root/arm
        replacements=[(template['source'],str(SOURCE)),(template['output'],str(out)),('314664',str(job))]
        def rewrite(v):
            if isinstance(v,str):
                for a,b in replacements:v=v.replace(a,b)
                return v
            if isinstance(v,list):return [rewrite(x) for x in v]
            if isinstance(v,dict):return {k:rewrite(x) for k,x in v.items()}
            return v
        s=rewrite(copy.deepcopy(template))
        s.update(actor=arm,job_id=str(job),source=str(SOURCE),output=str(out),
            wandb_run_id=f'arm-refresh-browser-{job}-{arm}',
            protocol='300 OM2W, baseline90 actor, full-response K5 SelectionARM, T0.8 top_p1, horizon30, GPT4.1/action_history')
        s['environment'].update(WANDB_RUN_ID=s['wandb_run_id'],WANDB_RESUME='never',
            OPENWEBRL_MULTIMODAL_STORAGE_DIR=str(RUNTIME/f'multimodal-scratch/arm-refresh-browser-{job}-{arm}'))
        s['command'][s['command'].index('--wandb-group')+1]='ARM-refresh-browser-baseline90'
        # Same policy_id and seed ensure matched candidate seeds and permutations.
        s['quality_config'].update(output=str(out),actor='baseline90',actor_order=0,
            tasks=str(SOURCE/'task_success_tasks.jsonl'),task_order=ids,task_count=300,
            budget=str(root/'terminal-budget'),judge_max_calls=1818,judge_max_usd=30,
            comparison_limitation=LIMITATION)
        s['selector']=dict(python=str(RUNTIME/'arm-reproduction/venv-refresh/bin/python'),
            model=str(REFRESH/'model-compat'),base=str(REFRESH/'model-compat'),
            environment={'TRITON_CACHE_DIR':str(out/'triton-cache'),
                'LD_LIBRARY_PATH':str(RUNTIME/'arm-reproduction/refresh-torch28/torch/lib'),
                'OMP_NUM_THREADS':'1'})
        if arm=='adapted':s['selector'].update(adapter=str(REFRESH/'adapter'),adapter_sha256=ADAPTER_SHA)
        for name in ('common.pt','.metadata'):
            if not (Path(s['checkpoint'])/name).is_file():raise ValueError('Actor checkpoint missing')
        stages.append(quality.evaluation.configure_evaluation_tracking(s,'openwebrl-evals'))
    p=dict(job_id=str(job),output=str(root),stages=stages,
        requested_resources=dict(gpus=2,gpu_type='H200',hours=4,gpu_hours=8,cpus=16,memory_gib=480),
        primary_trajectories=600,smoke_trajectories=6,teacher_max_calls=1818,teacher_max_usd=30,
        expected_collection=dict(tasks=300,trajectories=300,smoke_trajectories=3,optimizer_updates=0),
        fixed100=ids[:100],cohort_sha256=sha(SOURCE/'task_success_tasks.jsonl'),
        adapter_sha256=ADAPTER_SHA,storage_reserve_bytes=300*1024**3,limitation=LIMITATION,
        approval_status='Approved on September26' if (CONTROL/'approval.json').exists() else
            'New GPU allocation and judge budget require explicit approval before submission')
    return p


def summarize(p):
    stages={s['actor']:read(Path(s['output'])/'success-summary.json') for s in p['stages']}
    if any(x['tasks']!=300 for x in stages.values()):raise ValueError('Incomplete full300')
    outcomes={name:{r['task_id']:r['arm'] for r in x['task_outcomes']} for name,x in stages.items()}
    ids=list(outcomes['frozen'])
    if set(ids)!=set(outcomes['adapted']):raise ValueError('Cohort mismatch')
    slices={}
    for name in stages:
        rows=[outcomes[name][i] for i in p['fixed100']]
        wins=sum(r['success'] for r in rows);valid=sum(r['valid'] for r in rows)
        slices[name]=dict(tasks=100,successes=wins,valid=valid,overall=wins/100,valid_only=wins/valid if valid else None)
    paired=dict(improved=sum(not outcomes['frozen'][i]['success'] and outcomes['adapted'][i]['success'] for i in ids),
        regressed=sum(outcomes['frozen'][i]['success'] and not outcomes['adapted'][i]['success'] for i in ids),
        jointly_valid=sum(outcomes['frozen'][i]['valid'] and outcomes['adapted'][i]['valid'] for i in ids))
    result=dict(complete=True,stages=stages,fixed100=slices,paired=paired,limitation=LIMITATION,
        terminal_budget=read(Path(p['output'])/'terminal-budget/ledger.json'))
    write_json(Path(p['output'])/'task-success-summary.json',result)
    return result


def execute(p):
    job=p['job_id'];root=Path(p['output'])
    if os.getenv('SLURM_JOB_ID')!=job:raise ValueError('Use the explicitly approved dedicated allocation')
    submission=read(CONTROL/'submission.json')
    approval=read(CONTROL/'approval.json')
    if not approval.get('approved') or submission['job_id']!=job:
        raise ValueError('No matching approved submission receipt')
    if sum(x['elapsed_seconds'] for x in submission.get('prior_attempts',[]))+submission['maximum_seconds']>14400:
        raise ValueError('Recovery exceeds the approved4h total')
    res=allocation(subprocess.check_output(['scontrol','show','job',job,'-o'],text=True),job,requested_gpus=2,maximum_hours=4)
    if res['cpus']<16 or res['allocated_memory_gib']<480:raise ValueError('Wrong allocation profile')
    check_storage_quota(p['storage_reserve_bytes'])
    root.mkdir(exist_ok=False);write_json(root/'audit-plan.json',p)
    deadline=time.monotonic()+res['maximum_seconds'];active=None
    try:
        for s in p['stages']:
            if deadline-time.monotonic()<1800:raise TimeoutError('Insufficient reserve for next stage')
            check_storage_quota(p['storage_reserve_bytes'])
            Path(s['environment']['OPENWEBRL_MULTIMODAL_STORAGE_DIR']).mkdir(parents=True,exist_ok=False)
            active=subprocess.Popen(['srun',f'--jobid={job}','--nodes=1','--ntasks=1','--cpus-per-task=16',
                '--gres=gpu:h200:2','--exact','--cpu-bind=none',sys.executable,str(Path(__file__).resolve()),
                '--job-id',job,'--worker',s['actor']],start_new_session=True)
            while active.poll() is None:
                if time.monotonic()>deadline:raise TimeoutError('Allocation deadline')
                for halt in (root/'terminal-budget/halt.json',Path(s['output'])/'selector-failure.json'):
                    if halt.exists():raise RuntimeError('Stop on judge/selector failure: '+str(halt))
                counts={a['actor']:len(list((Path(a['output'])/'compare_records').glob('*.json'))) for a in p['stages']}
                write_json(root/'status.json',dict(complete=False,stage=s['actor'],tasks_completed=counts))
                time.sleep(10)
            if active.returncode:raise RuntimeError(f'{s["actor"]} stage failed; artifacts preserved')
            active=None
        summarize(p);write_json(root/'status.json',dict(complete=True,stage='complete',primary_trajectories=600))
    except BaseException as exc:
        write_json(root/'status.json',dict(complete=False,failed=True,error_type=type(exc).__name__,error=str(exc)[:500]))
        raise
    finally:quality.stop(active)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--job-id',default='PREPARE')
    parser.add_argument('--execute',action='store_true');parser.add_argument('--worker',choices=['frozen','adapted'])
    a=parser.parse_args()
    if a.worker:
        prepare();p=read(RUNTIME/f'evaluations/arm-refresh-browser-{a.job_id}/audit-plan.json')
        quality.SOURCE=SOURCE;quality.worker(p,a.worker)
    else:
        p=plan(a.job_id)
        if a.execute:execute(p)
        else:
            write_json(CONTROL/'launch-plan.json',p)
            print(json.dumps({k:p[k] for k in ('requested_resources','primary_trajectories','teacher_max_usd','approval_status')},indent=2))

if __name__=='__main__':main()
