#!/usr/bin/env python3
"""Prepare the user-authorized beta1 replacement under normal scheduler policy."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

REPO=Path(__file__).resolve().parents[1]
BASE=REPO/'scripts/prepare_arm_ablation_to60.py'
CONTROL=REPO/'logs/arm-beta-replacement-20260927'
spec=importlib.util.spec_from_file_location('_beta_replacement_impl',BASE)
impl=importlib.util.module_from_spec(spec);spec.loader.exec_module(impl)
impl.CONTROL=CONTROL
impl.__file__=__file__
impl.PROFILES={'weight-recovery':dict(variant='weight',start=20,end=60,hours=14)}
original_plan=impl.plan
original_train=impl.train


def plan(job,profile,root,target):
    p=original_plan(job,profile,root,target)
    p['environment']['OPENWEBRL_CUDA_CACHE_LIMIT_GIB']='24'
    p['environment']['PYTHONDONTWRITEBYTECODE']='1'
    p['replacement_of']='332004'
    p['memory_recovery']=dict(cache_limit_gib=24,first_batch_gpu_validation_pending=True,
        objective_topology_optimizer_scheduler_cursor_unchanged=True)
    return p


def fingerprints():
    paths=[Path(__file__),BASE,CONTROL/'training-controller.py',CONTROL/'restore-controller.py',
        REPO/'scripts/arm_browser_startup_guard.py',REPO/'scripts/record_arm_gpu_memory.py',
        REPO/'scripts/resume_arm_turn_bonus.py',impl.SOURCES['weight']/'reference_manifest.json']
    return {str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def check_budget(_unused):
    approval=json.loads((CONTROL/'weight-recovery-approval.json').read_text())
    if not approval.get('approved') or approval.get('resources')!=impl.resources('weight-recovery'):
        raise ValueError('Exact user resource approval is required')
    raw=subprocess.check_output(['sacct','-X','-n','-P','-j','332004',
        '--format=JobID,ElapsedRaw,State'],text=True)
    rows=[x.split('|') for x in raw.splitlines() if x.startswith('332004|')]
    if len(rows)!=1 or int(rows[0][1])!=0 or not rows[0][2].startswith('CANCELLED'):
        raise ValueError('Replacement must consume only the unused14h allocation')
    return dict(replaces='332004',previous_used_seconds=0,requested_seconds=50400,
                approved_seconds=50400)


def train(job,profile,root,target):
    p=plan(job,profile,root,target)
    process=subprocess.Popen([sys.executable,str(REPO/'scripts/record_arm_gpu_memory.py'),
        '--root',p['output'],'--output',str(CONTROL/f'{job}-gpu-memory-to{target}.jsonl'),
        '--first-iteration',str(target-9),'--minutes','60'],
        stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:original_train(job,profile,root,target)
    finally:
        process.terminate()
        try:process.wait(timeout=10)
        except subprocess.TimeoutExpired:process.kill();process.wait()


impl.plan=plan
impl.fingerprints=fingerprints
impl.train=train
impl.check_recovery_budget=check_budget
validate_plan=impl.validate_plan


def prepare():
    from arm_browser_startup_guard import instrument_controller
    CONTROL.mkdir(parents=True,exist_ok=True)
    for old,new in [('run_arm_turn_bonus_cycles.py','training-controller.py'),
                    ('check_arm_gpu_checkpoint_restore.py','restore-controller.py')]:
        text=(REPO/'scripts'/old).read_text()
        anchor='from prepare_arm_failure_ablations import validate_plan'
        if text.count(anchor)!=1:raise ValueError('Unexpected validation boundary')
        text=text.replace(anchor,'from prepare_arm_beta_replacement import validate_plan')
        if new=='training-controller.py':text=instrument_controller(text)
        compile(text,new,'exec')
        path=CONTROL/new
        if path.exists() and path.read_text()!=text:raise ValueError('Prepared controller changed')
        path.write_text(text)
    impl.write_json(CONTROL/'readiness.json',dict(fingerprints=fingerprints(),
        native_gpu_restore_required=True,saved_iteration21_replay_required=True,
        normal_scheduler_submission=True,submitted=False))
    p=plan('PREPARE','weight-recovery',impl.ORIGINS['weight'],30)
    validate_plan(p);impl.write_json(CONTROL/'training-plan.json',p)
    for name,body in [('weight-recovery-approval.json',dict(approved=True,
        resources=impl.resources('weight-recovery'),
        basis='User requested replacement of unused8-H200 x14h job332004 on September27'))]:
        if not (CONTROL/name).exists():impl.write_json(CONTROL/name,body)
    impl.write_json(CONTROL/'submission-plan.json',dict(submitted=False,
        resources=impl.resources('weight-recovery'),resume_iteration=20,optimizer_updates=298,
        target_iteration=60,eval_iterations=[30,40,50,60],within_previously_approved_resource_budget=True,
        no_budget_extension=True,hold=False,dependency=None,account='zixianma',qos='normal',
        normal_scheduler_submission=True,first_saved_replay_gpu_validation_pending=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--job-id',default='PREPARE')
    parser.add_argument('--profile',default='weight-recovery',choices=impl.PROFILES)
    parser.add_argument('--root',type=Path,default=impl.ORIGINS['weight'])
    parser.add_argument('--worker',choices=['train','eval']);parser.add_argument('--target',type=int)
    args=parser.parse_args()
    if args.prepare:prepare()
    elif args.worker:
        if os.getenv('SLURM_JOB_ID')!=args.job_id:raise ValueError('Wrong worker allocation')
        (train if args.worker=='train' else impl.evaluate)(args.job_id,args.profile,args.root,args.target)
    else:
        check_budget(None)
        impl.execute(args.job_id,args.profile,args.root)
