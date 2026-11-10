#!/usr/bin/env python3
"""Resume beta1 with smaller TP4 shards inside the unused333402 budget."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

REPO=Path(__file__).resolve().parents[1]
BASE=REPO/'scripts/prepare_arm_ablation_to60.py'
CONTROL=REPO/'logs/arm-beta-tp4-recovery-20260927'
spec=importlib.util.spec_from_file_location('_beta_tp4_impl',BASE)
impl=importlib.util.module_from_spec(spec);spec.loader.exec_module(impl)
impl.CONTROL=CONTROL
impl.__file__=__file__
impl.PROFILES={'weight-recovery':dict(variant='weight',start=20,end=60,hours=48660/3600)}
original_plan=impl.plan
original_train=impl.train


def plan(job,profile,root,target):
    p=original_plan(job,profile,root,target)
    old=p['output'];new=old.replace('-tp2-', '-tp4-')
    def rewrite(v):
        if isinstance(v,str):return v.replace(old,new)
        if isinstance(v,list):return [rewrite(x) for x in v]
        if isinstance(v,dict):return {k:rewrite(x) for k,x in v.items()}
        return v
    p=rewrite(p)
    p['environment'].update(TP_SIZE='4',OPENWEBRL_CUDA_CACHE_LIMIT_GIB='24',
        PYTHONDONTWRITEBYTECODE='1',PYTHONFAULTHANDLER='1',RAY_DEDUP_LOGS='0',
        RAY_TMPDIR=f'/tmp/openwebrl-beta-ray-{job}')
    p['memory_recovery']=dict(previous_job='333402',cuda_allocation_exit_confirmed=True,
        tensor_parallel=4,data_parallel=2,microbatch=1,cache_limit_gib=24,
        objective_global_batch_optimizer_scheduler_cursor_unchanged=True,
        native_restore_and_full_saved_batch_validation_required=True)
    return p


def fingerprints():
    paths=[Path(__file__),BASE,CONTROL/'training-controller.py',CONTROL/'restore-controller.py',
        REPO/'scripts/arm_browser_startup_guard.py',REPO/'scripts/record_arm_gpu_memory.py',
        REPO/'scripts/resume_arm_turn_bonus.py',impl.SOURCES['weight']/'reference_manifest.json']
    return {str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def check_budget(_unused):
    a=json.loads((CONTROL/'weight-recovery-approval.json').read_text())
    if not a.get('approved') or a['resources']!=impl.resources('weight-recovery'):
        raise ValueError('Missing approved bounded beta recovery')
    raw=subprocess.check_output(['sacct','-X','-n','-P','-j','333402','--format=JobID,ElapsedRaw,State'],text=True)
    rows=[x.split('|') for x in raw.splitlines() if x.startswith('333402|')]
    if len(rows)!=1 or rows[0][2]!='FAILED' or int(rows[0][1])+48660>50400:
        raise ValueError('Recovery exceeds unused14h budget or predecessor is active')
    return dict(prior_job='333402',used_seconds=int(rows[0][1]),requested_seconds=48660,
                original_approved_seconds=50400)


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
        # Keep worker stderr from this job only, including native allocator exits.
        logroot=Path(p['environment']['RAY_TMPDIR'])/'ray/session_latest/logs'
        if logroot.is_dir():
            destination=Path(p['output'])/'preserved-ray-logs'
            destination.mkdir(exist_ok=True)
            for pattern in ('worker-*.err','raylet.err','raylet.out'):
                for path in logroot.glob(pattern):
                    shutil.copy2(path,destination/path.name)


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
        text=text.replace(anchor,'from recover_arm_beta_tp4 import validate_plan')
        if new=='training-controller.py':text=instrument_controller(text)
        compile(text,new,'exec');path=CONTROL/new
        if path.exists() and path.read_text()!=text:raise ValueError('Controller changed')
        path.write_text(text)
    impl.write_json(CONTROL/'readiness.json',dict(fingerprints=fingerprints(),
        gpu_restore_required=True,full_batch_validation_pending=True,submitted=False))
    p=plan('PREPARE','weight-recovery',impl.ORIGINS['weight'],30)
    validate_plan(p);impl.write_json(CONTROL/'training-plan.json',p)
    impl.write_json(CONTROL/'weight-recovery-approval.json',dict(approved=True,
        resources=impl.resources('weight-recovery'),
        basis='User approved fixing/relaunching beta1; 13h31m is inside unused8-H200 x14h approval.'))
    impl.write_json(CONTROL/'budget.json',check_budget(None))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare',action='store_true');p.add_argument('--job-id',default='PREPARE')
    p.add_argument('--profile',default='weight-recovery',choices=impl.PROFILES)
    p.add_argument('--root',type=Path,default=impl.ORIGINS['weight'])
    p.add_argument('--worker',choices=['train','eval']);p.add_argument('--target',type=int)
    args=p.parse_args()
    if args.prepare:prepare()
    elif args.worker:
        if os.getenv('SLURM_JOB_ID')!=args.job_id:raise ValueError('Wrong allocation')
        (train if args.worker=='train' else impl.evaluate)(args.job_id,args.profile,args.root,args.target)
    else:
        check_budget(None);impl.execute(args.job_id,args.profile,args.root)
