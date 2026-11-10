#!/usr/bin/env python3
"""Isolated beta recovery with tighter cache release and cumulative time accounting."""
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
CONTROL=REPO/'logs/arm-beta-memory-20260927'
spec=importlib.util.spec_from_file_location('_arm_beta_memory_impl',BASE)
impl=importlib.util.module_from_spec(spec);spec.loader.exec_module(impl)
impl.CONTROL=CONTROL
impl.__file__=__file__
impl.PROFILES={k:dict(v) for k,v in impl.PROFILES.items() if k.startswith('weight-')}
impl.PROFILES['weight-recovery']['hours']=13+10/60
original_plan=impl.plan
original_train=impl.train


def plan(job,profile,root,target):
    if profile not in impl.PROFILES:raise ValueError('Beta-only recovery')
    p=original_plan(job,profile,root,target)
    p['environment']['OPENWEBRL_CUDA_CACHE_LIMIT_GIB']='24'
    p['memory_recovery']=dict(previous_jobs=['330278','331995'],cache_limit_gib=24,
        objective_and_topology_unchanged=True,first_batch_gpu_validation_pending=True)
    return p


def fingerprints():
    paths=[Path(__file__).resolve(),BASE,CONTROL/'training-controller.py',CONTROL/'restore-controller.py',
        REPO/'scripts/resume_arm_turn_bonus.py',REPO/'scripts/record_arm_gpu_memory.py',
        *[s/'reference_manifest.json' for s in impl.SOURCES.values()]]
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def check_recovery_budget(accounting):
    rows={r[0]:r for r in (line.split('|') for line in accounting.splitlines()) if len(r)>=3}
    ids=['330278','331995']
    if any(j not in rows or rows[j][2]!='FAILED' for j in ids):
        raise ValueError('Every previous attempt must be terminal and accounted for')
    used=sum(int(rows[j][1]) for j in ids);requested=47400
    if used+requested>50400:raise ValueError('Recovery exceeds cumulative14h approval')
    return dict(previous_jobs=ids,used_seconds=used,requested_seconds=requested,approved_seconds=50400)


def train(job,profile,root,target):
    p=plan(job,profile,root,target)
    record=CONTROL/f'{job}-gpu-memory-to{target}.jsonl'
    process=subprocess.Popen([sys.executable,str(REPO/'scripts/record_arm_gpu_memory.py'),
        '--root',p['output'],'--output',str(record),'--first-iteration',str(target-9),'--minutes','60'],
        stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:original_train(job,profile,root,target)
    finally:
        process.terminate()
        try:process.wait(timeout=10)
        except subprocess.TimeoutExpired:process.kill();process.wait()


impl.plan=plan
impl.fingerprints=fingerprints
impl.train=train
impl.check_recovery_budget=check_recovery_budget
validate_plan=impl.validate_plan


def prepare():
    CONTROL.mkdir(parents=True,exist_ok=True)
    for old,new in [('run_arm_turn_bonus_cycles.py','training-controller.py'),
                    ('check_arm_gpu_checkpoint_restore.py','restore-controller.py')]:
        text=(REPO/'scripts'/old).read_text()
        before='from prepare_arm_failure_ablations import validate_plan'
        if text.count(before)!=1:raise ValueError('Unexpected validator boundary')
        text=text.replace(before,'from recover_arm_beta_memory import validate_plan')
        compile(text,new,'exec');path=CONTROL/new
        if path.exists() and path.read_text()!=text:raise ValueError('Frozen controller changed')
        path.write_text(text)
    impl.write_json(CONTROL/'readiness.json',dict(fingerprints=fingerprints(),gpu_restore_pending=True))
    p=plan('PREPARE','weight-recovery',impl.ORIGINS['weight'],30)
    validate_plan(p);impl.write_json(CONTROL/'training-plan.json',p)


def execute(job,profile,root):
    # The inherited worker queries one prior ID; supply complete cumulative
    # accounting explicitly before allowing its allocation checks to proceed.
    if profile=='weight-recovery':
        raw=subprocess.check_output(['sacct','-X','-n','-P','-j','330278,331995',
            '--format=JobID,ElapsedRaw,State'],text=True)
        budget=check_recovery_budget(raw)
        impl.write_json(CONTROL/f'{job}-budget.json',budget)
        impl.check_recovery_budget=lambda ignored:check_recovery_budget(raw)
    impl.execute(job,profile,root)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--job-id',default='PREPARE')
    parser.add_argument('--profile',choices=impl.PROFILES);parser.add_argument('--root',type=Path)
    parser.add_argument('--worker',choices=('train','eval'));parser.add_argument('--target',type=int)
    args=parser.parse_args()
    if args.prepare:prepare()
    elif args.worker:
        if os.getenv('SLURM_JOB_ID')!=args.job_id:raise ValueError('Wrong worker allocation')
        (train if args.worker=='train' else impl.evaluate)(args.job_id,args.profile,args.root,args.target)
    else:execute(args.job_id,args.profile,args.root)
