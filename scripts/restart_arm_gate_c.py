#!/usr/bin/env python3
"""Prepare a fresh C first batch after its pre-update artifact-size failure.

No submission is performed. C's old crash happened before its cursor/auxiliary
manifest were saved; do not represent an inferred cursor as verified replay.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from unittest.mock import patch

import recover_arm_gate_ablation as recovery
import run_arm_gate_ablation as gate
from prepare_arm_turn_bonus import copy_plain, replace_once
from resume_baseline import validate_source, write_json

REPO,RUNTIME=gate.REPO,gate.RUNTIME
SOURCE=RUNTIME/'reference-arm-gate-c-restart-20260920-v1'
PREPARATION=RUNTIME/'arm-turn-bonus-preparation/gate-c-restart-20260920'
ORIGIN=RUNTIME/'evaluations/arm-failure-additive-309054'


def prepare():
    validate_source(recovery.SOURCE)
    if not SOURCE.exists():
        shutil.copytree(recovery.SOURCE,SOURCE,symlinks=True,copy_function=copy_plain,
            ignore=shutil.ignore_patterns('__pycache__','*.pyc','.git','.browser_use_sessions'))
        f='scripts/run_arm_turn_bonus_cycles.py';path=SOURCE/f
        path.write_text(replace_once(path.read_text(),"    if p.get('gate_recovery'):\n",
            "    if p.get('gate_c_restart'):\n        from restart_arm_gate_c import validate_plan\n        readiness=validate_plan(p)\n    elif p.get('gate_recovery'):\n"))
        manifest=json.loads((SOURCE/'reference_manifest.json').read_text())
        manifest['recipe_files_sha256'][f]=hashlib.sha256(path.read_bytes()).hexdigest()
        manifest['arm_gate_c_restart']=dict(parent=str(recovery.SOURCE),changed_files=[f],
            initialization='same original SFT; fresh first collection; zero prior optimizer updates')
        write_json(SOURCE/'reference_manifest.json',manifest)
    validate_source(SOURCE)


def plan(job):
    validate_source(SOURCE)
    old=json.loads((ORIGIN/'status.json').read_text())
    if (not old['failed'] or not old['no_optimizer_updates'] or old['completed_iterations']!=0
            or (ORIGIN/'runtime/latest_checkpointed_iteration.txt').exists()):
        raise ValueError('Restart requires C to have zero completed updates')
    p=gate.plan(job,'C')
    p['command']=[x.replace(str(gate.SOURCE),str(SOURCE)) for x in p['command']]
    p['environment']={k:v.replace(str(gate.SOURCE),str(SOURCE)) for k,v in p['environment'].items()}
    p['arm_config']={k:v.replace(str(gate.SOURCE),str(SOURCE)) if isinstance(v,str) else v for k,v in p['arm_config'].items()}
    run='arm-gate-c-309054'
    p.update(source=str(SOURCE),gate_c_restart=True,compute_approved=False,wandb_run_id=run)
    p['requested_resources'].update(hours=7,gpu_hours=28)
    p['environment'].update(WANDB_RUN_ID=run,WANDB_RESUME='must')
    p['arm_config'].update(run_id=run,policy_id=run+':uninitialized',gate_scale_guard='pretraining_coverage_audit_v1')
    p['comparison']['restart']='Fresh first collection at original SFT; preserve old failed archive without guessing missing manifest/cursor'
    p['budget_note']='Prepared, not approved: 4 H200 x 7h, 32 CPU, 480 GiB, 32 browsers; no extra eval allocation.'
    return p


def fingerprint():
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [
        REPO/'scripts/restart_arm_gate_c.py',REPO/'scripts/restart_arm_gate_c_4gpu.sbatch',
        SOURCE/'reference_manifest.json',gate.training.INITIAL/'config.json',gate.training.INITIAL/'tokenizer_config.json']}


def validate_plan(p):
    if p!=plan(p['job_id']):raise ValueError('C restart plan changed')
    receipt=json.loads((PREPARATION/'readiness.json').read_text())
    if not receipt['cpu_passed'] or not receipt['native_parse_passed'] or receipt['fingerprint']!=fingerprint():
        raise ValueError('C restart readiness stale')
    return receipt


def check():
    prepare();PREPARATION.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,PYTHONPATH=str(SOURCE),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    code="import sys,unittest;sys.path.insert(0,%r);sys.path.insert(1,%r);r=unittest.TextTestRunner(verbosity=2).run(unittest.TestLoader().loadTestsFromNames(['test_arm_gate_recovery','test_arm_min2_gate','test_arm_failure_additive']));raise SystemExit(not r.wasSuccessful())" % (str(REPO/'tests'),str(REPO/'scripts'))
    with (PREPARATION/'frozen-tests.log').open('w') as log:
        subprocess.run([sys.executable,'-c',code],cwd=SOURCE,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=90)
    subprocess.run(['bash','-n',str(REPO/'scripts/restart_arm_gate_c_4gpu.sbatch')],check=True)
    p=plan('PREPARE-C');write_json(PREPARATION/'plan.json',p)
    import run_arm_failure_additive as additive
    with patch.object(additive,'SOURCE',SOURCE):additive.native_check(p,PREPARATION)
    write_json(PREPARATION/'readiness.json',dict(cpu_passed=True,native_parse_passed=True,
        gpu_validated=False,compute_approved=False,fingerprint=fingerprint()))
    validate_plan(p)
    return dict(preparation=str(PREPARATION),submitted=False)


def execute(p):
    validate_plan(p)
    if os.environ.get('SLURM_JOB_ID')!=p['job_id'] or os.environ.get('ARM_GATE_C_RESTART_APPROVED')!='4gpu_7h':
        raise ValueError('Explicit C replacement-allocation approval required')
    spec=importlib.util.spec_from_file_location('gate_c_restart_controller',SOURCE/'scripts/run_arm_turn_bonus_cycles.py')
    controller=importlib.util.module_from_spec(spec);spec.loader.exec_module(controller)
    write_json(RUNTIME/f'logs/arm-gate-c-restart-{p["job_id"]}.json',p)
    with (RUNTIME/f'logs/arm-gate-c-restart-monitor-{p["job_id"]}.log').open('w') as log:
        monitor=subprocess.Popen([sys.executable,str(REPO/'scripts/monitor_arm_turn_bonus.py'),
            '--job-id',p['job_id'],'--watch','--interval','900','--hours','7'],stdout=log,stderr=subprocess.STDOUT)
        try:controller.execute(p)
        finally:
            monitor.terminate()
            try:monitor.wait(timeout=30)
            except subprocess.TimeoutExpired:monitor.kill();monitor.wait()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--job-id',default='UNAPPROVED')
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if args.check:print(json.dumps(check(),indent=2))
    else:
        p=plan(args.job_id)
        if args.execute:execute(p)
        else:print(json.dumps(p,indent=2))
