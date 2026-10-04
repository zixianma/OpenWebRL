#!/usr/bin/env python3
"""Launch approved B/C ablations from the original SFT actor at iteration zero."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import run_arm_turn_bonus_cycles as training
from prepare_arm_turn_bonus import copy_plain
from resume_baseline import validate_source, write_json

REPO, RUNTIME = training.REPO, training.RUNTIME
PARENT = RUNTIME/'reference-arm-failure-additive-20260914-v3'
SOURCE = RUNTIME/'reference-arm-gate-ablation-20260919-v2'
PREPARATION = RUNTIME/'arm-turn-bonus-preparation/gate-ablation-fromzero-20260919'
RULES = {'B': 'response_index', 'C': 'action_class'}


def replace_definitions(original, working, names):
    """Transplant only named definitions; keep frozen additive hooks intact."""
    def blocks(text):
        lines = text.splitlines(keepends=True)
        nodes = {n.name:n for n in ast.parse(text).body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
        return lines, nodes
    lines, old = blocks(original)
    new_lines, new = blocks(working)
    for name in sorted(names, key=lambda k:old[k].lineno, reverse=True):
        node, replacement = old[name], new[name]
        lines[node.lineno-1:node.end_lineno] = new_lines[replacement.lineno-1:replacement.end_lineno]
    return ''.join(lines)


def prepare():
    validate_source(PARENT)
    if SOURCE.exists():
        validate_source(SOURCE)
        return
    shutil.copytree(PARENT, SOURCE, symlinks=True, copy_function=copy_plain,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git', '.browser_use_sessions'))
    changed = ['openwebrl/arm_turn_bonus.py', 'openwebrl/arm_turn_bonus_runtime.py',
               'openwebrl/arm_failure_additive.py', 'openwebrl/arm_failure_aux.py']
    name = changed[0]
    working = (REPO/name).read_text()
    frozen = (PARENT/name).read_text()
    updated = replace_definitions(frozen, working, ['eligible_actions', 'ShadowSelector', 'generate', 'unit_bonus', 'panel'])
    nodes = {n.name:n for n in ast.parse(working).body if isinstance(n, ast.FunctionDef)}
    lines = working.splitlines(keepends=True)
    helpers = ''.join(''.join(lines[nodes[n].lineno-1:nodes[n].end_lineno])+'\n\n'
                      for n in ('candidate_minimum', 'credit_rule', 'action_class_bonus', 'candidate_bonus'))
    updated = updated.replace('def eligible_actions(', helpers+'def eligible_actions(', 1)
    assert 'write_auxiliary(args,samples,current,report)' in updated
    assert 'annotate_calibration(report' not in updated  # additive has a separate failure population
    (SOURCE/name).write_text(updated)
    name = changed[1]
    (SOURCE/name).write_text(replace_definitions((PARENT/name).read_text(), (REPO/name).read_text(), ['validate_config']))
    name = changed[2]
    (SOURCE/name).write_text(replace_definitions((PARENT/name).read_text(), (REPO/name).read_text(), ['write_auxiliary']))
    name = changed[3]
    working = (REPO/name).read_text()
    updated = replace_definitions((PARENT/name).read_text(),working,['prepare_auxiliary'])
    node = next(n for n in ast.parse(working).body if isinstance(n,ast.FunctionDef) and n.name=='validate_auxiliary_advantage')
    helper = ''.join(working.splitlines(keepends=True)[node.lineno-1:node.end_lineno])+'\n\n'
    (SOURCE/name).write_text(updated.replace('def prepare_auxiliary(',helper+'def prepare_auxiliary(',1))
    manifest = json.loads((SOURCE/'reference_manifest.json').read_text())
    manifest['recipe_files_sha256'].update({f:hashlib.sha256((SOURCE/f).read_bytes()).hexdigest() for f in changed})
    manifest['arm_gate_ablation'] = dict(parent=str(PARENT), changed_files=changed,
        gate='min2', independent_credit_rules=RULES, gpu_validated=False)
    write_json(SOURCE/'reference_manifest.json', manifest)
    validate_source(SOURCE)


def plan(job, variant):
    if variant not in RULES:
        raise ValueError('Expected variant B or C')
    validate_source(SOURCE)
    import run_arm_failure_additive as additive
    p = additive.plan(job, minutes=480)
    output = RUNTIME/f'evaluations/arm-failure-additive-{job}'
    run = f'arm-gate-{variant.lower()}-{job}'
    def rewrite(value):
        return value.replace(str(PARENT), str(SOURCE))
    p['command'] = [rewrite(x) for x in p['command']]
    p['environment'] = {k:rewrite(v) for k,v in p['environment'].items()}
    p['arm_config'] = {k:rewrite(v) if isinstance(v,str) else v for k,v in p['arm_config'].items()}
    p['arm_config'].pop('deadline_epoch_seconds', None)
    p.update(job_id=str(job), output=str(output), source=str(SOURCE), wandb_run_id=run,
        variant_continuation=False, gate_ablation=variant, initial_optimizer_updates=0, start_rollout_id=0,
        requested_iterations=10, target_completed_iterations=10, checkpoint=str(training.INITIAL),
        fresh_optimizer=True, compute_approved=True, evaluation_in_allocation=False)
    p['requested_resources'].update(gpus=4, hours=8, gpu_hours=32, cpus=32, memory_gib=480, browsers=32)
    p['environment'].update(NUM_GPUS='4', TP_SIZE='4', NUM_ROLLOUT='10', WANDB_RESUME='never', WANDB_RUN_ID=run,
        SLIME_LOAD_CHECKPOINT=str(training.INITIAL), BROWSER_CONCURRENCY='32', SGLANG_CONCURRENCY='32',
        OPENWEBRL_MULTIMODAL_STORAGE_DIR=f'/tmp/arm-gate-{variant.lower()}-{job}-multimodal')
    p['browser_config']['browser_rollout_concurrency'] = 32
    p['command'][p['command'].index('--wandb-group')+1] = 'arm-min2-independent-credit'
    p['selector_port'] = 31000+int(job)%20000 if str(job).isdigit() else 27511
    p['arm_config'].update(run_id=run, policy_id=run+':uninitialized', checkpoint=str(training.INITIAL),
        run_output=str(output), output=str(output), candidate_gate='min2', credit_assignment=RULES[variant],
        selector_endpoint=f"http://127.0.0.1:{p['selector_port']}", minimum_cycle_seconds=5400)
    p['comparison'] = dict(parent='original OpenWebRL-4B-SFT; fresh optimizer and initial task cursor; iteration zero',
        B='min2 gate, response-index credit', C='min2 gate, action-class credit',
        ordinary_groups=48, failure_group_cap=8, failure_loss_coefficient='N/48', beta=.5, q=.2,
        credit_scope='all retained labeled turns; full response loss including reasoning',
        control='existing unchanged additive trajectory is historical, not a fresh randomized control',
        target='at most 10 collections; 8-hour budget likely permits fewer; compare shared durable endpoints')
    p['budget_note'] = 'User-approved eight hours per variant, including actor startup and calibration; no separate evaluation budget.'
    return p


def fingerprint():
    paths = [SOURCE/'reference_manifest.json', *[REPO/f for f in (
        'scripts/run_arm_gate_ablation.py', 'scripts/run_arm_gate_ablation_4gpu.sbatch',
        'scripts/run_arm_turn_bonus_cycles.py')]]
    paths += [training.INITIAL/'config.json', training.INITIAL/'tokenizer_config.json']
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def validate_plan(p):
    if p != plan(p['job_id'], p['gate_ablation']):
        raise ValueError('Gate ablation plan or checkpoint changed')
    receipt = json.loads((PREPARATION/'readiness.json').read_text())
    if not receipt.get('cpu_passed') or not receipt.get('native_parse_passed') or receipt['fingerprint'] != fingerprint():
        raise ValueError('Gate ablation readiness missing/stale')
    if p.get('resume_from') or not p['fresh_optimizer'] or p['start_rollout_id'] != 0 or p['initial_optimizer_updates'] != 0:
        raise ValueError('B/C must start from SFT at iteration zero')
    return dict(passed=True, independent_knobs=True, starts_at_iteration_zero=True)


def check():
    prepare()
    PREPARATION.mkdir(parents=True, exist_ok=True)
    # Run against the frozen source, preserving its distinct additive reward hook.
    code = """import sys,unittest
sys.path.insert(0, %r)
sys.path.insert(1, %r)
suite=unittest.TestLoader().loadTestsFromNames(['test_arm_min2_gate','test_arm_failure_additive'])
result=unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(not result.wasSuccessful())
""" % (str(REPO/'tests'),str(REPO/'scripts'))
    env=dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONPATH=str(SOURCE))
    with (PREPARATION/'frozen-tests.log').open('w') as log:
        subprocess.run([sys.executable,'-c',code],cwd=SOURCE,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
    subprocess.run(['bash','-n',str(REPO/'scripts/run_arm_gate_ablation_4gpu.sbatch')],check=True)
    import run_arm_failure_additive as additive
    for variant in RULES:
        p=plan('PREPARE-'+variant,variant)
        out=PREPARATION/variant;out.mkdir(exist_ok=True)
        write_json(out/'plan.json',p)
        additive.native_check(p,out)
        if p['environment']['SLIME_LOAD_CHECKPOINT'] != str(training.INITIAL) or '--use-checkpoint-opt-param-scheduler' in p['command']:
            raise ValueError('Unexpected resumed initialization')
        log=(out/'native-parse.log').read_text()
        if str(training.INITIAL) not in log:
            raise ValueError('Native arguments lack starting SFT checkpoint')
    write_json(PREPARATION/'readiness.json',dict(cpu_passed=True,native_parse_passed=True,
        gpu_validated=False,compute_approved=True,fingerprint=fingerprint()))
    for variant in RULES:validate_plan(plan('PREPARE-'+variant,variant))
    return dict(preparation=str(PREPARATION),resource_request='two separate jobs, each 4 H200 x 8h, 32 CPU, 480 GiB, 32 browsers',submitted=False)


def execute(p):
    validate_plan(p)
    if os.environ.get('SLURM_JOB_ID') != p['job_id']:
        raise ValueError('Existing user-approved allocation required')
    manifest=RUNTIME/f'logs/arm-gate-{p["gate_ablation"]}-{p["job_id"]}.json'
    write_json(manifest,p)
    with (RUNTIME/f'logs/arm-gate-monitor-{p["job_id"]}.log').open('w') as log:
        monitor=subprocess.Popen([sys.executable,str(REPO/'scripts/monitor_arm_turn_bonus.py'),
            '--job-id',p['job_id'],'--watch','--interval','900','--hours','8'],stdout=log,stderr=subprocess.STDOUT)
        try:training.execute(p)
        finally:
            monitor.terminate()
            try:monitor.wait(timeout=30)
            except subprocess.TimeoutExpired:monitor.kill();monitor.wait()


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--variant',choices=RULES)
    parser.add_argument('--job-id',default='PREPARE')
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if args.check:print(json.dumps(check(),indent=2))
    else:
        if not args.variant:parser.error('--variant is required')
        p=plan(args.job_id,args.variant)
        if args.execute:execute(p)
        else:print(json.dumps(p,indent=2))
