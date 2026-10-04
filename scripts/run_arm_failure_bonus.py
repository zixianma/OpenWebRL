#!/usr/bin/env python3
"""Prepare/check/run independent all-failure ARM RL; never request an allocation."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from evaluate_baseline_checkpoint import REPO, RUNTIME
from prepare_arm_turn_bonus import copy_plain
from resume_baseline import validate_source, write_json
import run_arm_turn_bonus_cycles as training

SOURCE = RUNTIME/'reference-arm-failure-bonus-20260913-v2'
RECEIPT = RUNTIME/'arm-turn-bonus-preparation/all-failure/readiness.json'
MODULES = ('openwebrl/arm_failure_bonus.py', 'openwebrl/arm_turn_bonus.py',
           'openwebrl/arm_turn_bonus_runtime.py')
FILES = (*MODULES, 'scripts/run_arm_failure_bonus.py', 'scripts/run_arm_failure_bonus_4gpu.sbatch',
         'scripts/run_arm_turn_bonus_cycles.py', 'scripts/check_arm_native_launch.py',
         'scripts/run_arm_turn_bonus_calibration.py', 'scripts/monitor_arm_turn_bonus.py',
         'scripts/inspect_training_checkpoint.py', 'scripts/resume_baseline.py',
         'scripts/audit_arm_failure_groups.py')


def prepare():
    """Freeze a new source; the earlier trained source remains immutable."""
    validate_source(training.SOURCE)
    if SOURCE.exists():
        validate_source(SOURCE)
        if any((SOURCE/f).read_bytes() != (REPO/f).read_bytes() for f in MODULES):
            raise ValueError('Prepared source differs from working modules; use a new source version')
        return
    shutil.copytree(training.SOURCE, SOURCE, symlinks=True, copy_function=copy_plain,
        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git', '.browser_use_sessions'))
    for f in MODULES:
        copy_plain(REPO/f, SOURCE/f)
    manifest=json.loads((SOURCE/'reference_manifest.json').read_text())
    hashes={f:hashlib.sha256((SOURCE/f).read_bytes()).hexdigest() for f in MODULES}
    manifest['recipe_files_sha256'].update(hashes)
    manifest['arm_failure_bonus']=dict(parent_source=str(training.SOURCE), changed_files_sha256=hashes,
        native_actor_loss_normalizer_unchanged=True, gpu_validated=False)
    write_json(SOURCE/'reference_manifest.json',manifest)
    validate_source(SOURCE)


def plan(job='UNAPPROVED_JOB', minutes=240):
    validate_source(SOURCE)
    p=training.plan(job,'arm',minutes)
    old_output=p['output']; output=RUNTIME/f'evaluations/arm-turn-bonus-fresh-allfailure-{job}'
    run_id=f'arm-allfailure-bonus-{job}'
    p['command']=[x.replace(old_output,str(output)).replace(str(training.SOURCE),str(SOURCE)) for x in p['command']]
    p['environment']={k:v.replace(old_output,str(output)).replace(str(training.SOURCE),str(SOURCE))
                      for k,v in p['environment'].items()}
    p['command'][p['command'].index('--wandb-group')+1]='executed-turn-bonus-all-failure'
    p['command'] += ['--dynamic-sampling-filter-path','openwebrl.arm_failure_bonus.filter_groups']
    p.update(experiment='arm-all-failure-bonus',output=str(output),source=str(SOURCE),wandb_run_id=run_id,
             compute_approved=False,readiness_receipt=str(RECEIPT))
    p['environment'].update(WANDB_RUN_ID=run_id,WANDB_RESUME='never',BROWSER_CONCURRENCY='32',SGLANG_CONCURRENCY='32',
        OPENWEBRL_MULTIMODAL_STORAGE_DIR=f'/tmp/arm-allfailure-{job}-multimodal')
    p['requested_resources']['browsers']=32
    p['browser_config']['browser_rollout_concurrency']=32
    p['arm_config'].update(admit_all_failure_groups=True,run_id=run_id,policy_id=f'{run_id}:uninitialized',
        output=str(output),run_output=str(output),minimum_cycle_seconds=4500)
    p['comparison']=dict(initialization='same original SFT weights; fresh optimizer, iteration zero',
        previous_run='arm-turn-bonus-fresh-294197', beta=.5, scored_fraction=.2,k=5,
        change='Admit valid five-failure actor groups with at least one usable local ARM label.',
        mixed_outcome_exposure='48 total accepted groups; report changed mixture and optimizer-update counts.',
        evaluation='Actor-only held-out OM2W under the baseline evaluation protocol; separate budget required.')
    return p


def fingerprint():
    validate_source(SOURCE)
    if any((SOURCE/f).read_bytes() != (REPO/f).read_bytes() for f in MODULES):
        raise ValueError('Working modules differ from the frozen all-failure source')
    paths=[REPO/f for f in FILES]
    paths += sorted((REPO/'tests').glob('test_arm_turn_bonus*.py'))
    paths += [REPO/'tests/test_arm_failure_bonus.py',SOURCE/'reference_manifest.json']
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def require_receipt(p):
    r=json.loads(Path(p['readiness_receipt']).read_text())
    if not r.get('passed') or r.get('fingerprint') != fingerprint():
        raise ValueError('All-failure CPU readiness receipt is missing or stale')
    if p != plan(p['job_id'],round(p['requested_resources']['hours']*60)):
        raise ValueError('All-failure launch plan changed after preparation')
    for path,sha in r['artifact_sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=sha:
            raise ValueError('Readiness evidence changed')
    return r


def check():
    prepare()
    start=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    folder=RECEIPT.parent/start; folder.mkdir(parents=True,exist_ok=False)
    before=fingerprint()
    write_json(RECEIPT,dict(passed=False,stage='checking',output=str(folder)))
    artifacts=[]
    for pattern in ('test_arm_failure_bonus.py','test_arm_turn_bonus*.py'):
        log=folder/(pattern.replace('*','all')+'.log');artifacts.append(log)
        with log.open('w') as out:
            subprocess.run([str(RUNTIME/'venv/bin/python'),'-m','unittest','discover','-s','tests','-p',pattern,'-v'],
                cwd=REPO,env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'),
                stdout=out,stderr=subprocess.STDOUT,check=True,timeout=180)
    subprocess.run(['bash','-n',str(REPO/'scripts/run_arm_failure_bonus_4gpu.sbatch')],check=True)
    prepared=plan();write_json(folder/'plan.json',prepared)
    with (folder/'native-check.log').open('w') as out:
        subprocess.run([str(RUNTIME/'venv/bin/python'),str(REPO/'scripts/check_arm_native_launch.py'),
            '--plan-file',str(folder/'plan.json'),'--output',str(folder/'native')],cwd=REPO,
            stdout=out,stderr=subprocess.STDOUT,check=True,timeout=180)
    native=folder/'native/arm-report.json'
    if not json.loads(native.read_text()).get('passed'):raise ValueError('Native parsing failed')
    artifacts += [native,folder/'plan.json']
    if fingerprint()!=before:raise ValueError('Sources changed while checks ran')
    result=dict(passed=True,checked_utc=start,fingerprint=before,plan=str(folder/'plan.json'),
        artifact_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts},
        gpu_validated=False,compute_approved=False,
        limitation='CPU admission, native normalization/backward, and native launch validation; live variant GPU gate pending.')
    write_json(RECEIPT,result);require_receipt(prepared)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare',action='store_true');mode.add_argument('--check',action='store_true')
    mode.add_argument('--execute',action='store_true')
    parser.add_argument('--job-id',default='UNAPPROVED_JOB');parser.add_argument('--minutes',type=int,default=240)
    parser.add_argument('--check-before-execute',action='store_true',help='Refresh readiness inside the authorized allocation before GPU loading')
    a=parser.parse_args()
    if a.prepare:prepare();print(SOURCE);return
    if a.check:
        try:print(json.dumps(check(),indent=2))
        except BaseException as exc:
            r=json.loads(RECEIPT.read_text()) if RECEIPT.exists() else {}
            r.update(passed=False,stage='failed',error_type=type(exc).__name__,error=str(exc)[:300])
            write_json(RECEIPT,r)
            raise
        return
    if a.check_before_execute:
        if not a.execute or os.environ.get('SLURM_JOB_ID') != a.job_id:
            raise ValueError('Pre-execution checks require the authorized allocation')
        check()
    p=plan(a.job_id,a.minutes)
    if not a.execute:print(json.dumps(p,indent=2));return
    require_receipt(p)
    # The allocation and exact resource budget must already be user-authorized.
    # This process owns and awaits both worker and monitor until completion.
    log_path=RUNTIME/f'logs/arm-allfailure-{a.job_id}-monitor.log'
    with log_path.open('w') as log:
        monitor=subprocess.Popen([str(RUNTIME/'venv/bin/python'),str(REPO/'scripts/monitor_arm_turn_bonus.py'),
            '--job-id',a.job_id,'--watch','--interval','900','--hours',str(a.minutes/60)],
            stdout=log,stderr=subprocess.STDOUT)
        try:training.execute(p)
        finally:
            monitor.terminate()
            try:monitor.wait(timeout=40)
            except subprocess.TimeoutExpired:monitor.kill();monitor.wait()
    return


if __name__=='__main__': main()
