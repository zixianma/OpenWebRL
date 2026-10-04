#!/usr/bin/env python3
"""CPU preparation and a single collection-only failure-coverage pilot; no sbatch."""
import argparse
import ast
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from prepare_arm_turn_bonus import copy_plain
from resume_baseline import source_command, validate_source, write_json
import resume_arm_turn_bonus as resume
import run_arm_turn_bonus_cycles as training
from run_arm_gate_ablation import replace_definitions

REPO, RUNTIME = training.REPO, training.RUNTIME
PARENT = RUNTIME/'reference-arm-additive-to100-20260921-v1'
SOURCE = RUNTIME/'reference-arm-failure-coverage-20260921-v1'
PREPARATION = RUNTIME/'arm-turn-bonus-preparation/failure-coverage-20260921'
DEFAULT_ORIGIN = RUNTIME/'evaluations/arm-failure-additive-311962'
MODULES = ['openwebrl/'+name+'.py' for name in (
    'arm_failure_recipe', 'arm_failure_coverage', 'arm_failure_additive',
    'arm_failure_aux', 'arm_failure_bonus', 'arm_gate_recovery')]


def prepare():
    validate_source(PARENT)
    if SOURCE.exists():
        validate_source(SOURCE)
        for name in MODULES:
            if (SOURCE/name).read_bytes() != (REPO/name).read_bytes():
                raise ValueError('Frozen preparation is stale; create a new source version')
        return
    shutil.copytree(PARENT, SOURCE, symlinks=True, copy_function=copy_plain,
        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git', '.browser_use_sessions'))
    for name in MODULES: copy_plain(REPO/name, SOURCE/name)
    name = 'openwebrl/arm_turn_bonus.py'
    working = (REPO/name).read_text()
    updated = replace_definitions((PARENT/name).read_text(), working,
        ['eligible_actions', 'ShadowSelector', 'generate', 'unit_bonus', 'panel'])
    nodes = {n.name:n for n in ast.parse(working).body if isinstance(n, ast.FunctionDef)}
    lines = working.splitlines(keepends=True)
    helpers = ''.join(''.join(lines[nodes[n].lineno-1:nodes[n].end_lineno])+'\n\n'
        for n in ('candidate_minimum', 'credit_rule', 'action_class_bonus', 'candidate_bonus'))
    updated = updated.replace('def eligible_actions(', helpers+'def eligible_actions(', 1)
    if 'write_auxiliary(args,samples,current,report)' not in updated:
        raise ValueError('Lost frozen additive serialization boundary')
    (SOURCE/name).write_text(updated)
    changed = [*MODULES, name]
    name = 'openwebrl/arm_turn_bonus_runtime.py'
    (SOURCE/name).write_text(replace_definitions((PARENT/name).read_text(),
        (REPO/name).read_text(), ['validate_config']))
    changed.append(name)
    name = 'slime/rollout/sglang_rollout.py'
    text = (PARENT/name).read_text()
    anchor = '    # There can be circumstances where users want to process all samples including filtered ones.\n'
    if text.count(anchor) != 1: raise ValueError('Deferred-label collection boundary changed')
    hook = ('    if os.environ.get("OPENWEBRL_ARM_TURN_BONUS_CONFIG"):\n'
            '        from openwebrl.arm_failure_coverage import finalize\n'
            '        await finalize(args, rollout_id)\n\n')
    (SOURCE/name).write_text(text.replace(anchor, hook+anchor, 1))
    changed.append(name)
    manifest = json.loads((SOURCE/'reference_manifest.json').read_text())
    manifest['recipe_files_sha256'].update({n:hashlib.sha256((SOURCE/n).read_bytes()).hexdigest() for n in changed})
    manifest['failure_coverage'] = dict(parent=str(PARENT), changed_files=changed,
        budget=4, max_groups=8, beta_mixed=.5, beta_failure=.5, gpu_validated=False)
    write_json(SOURCE/'reference_manifest.json', manifest)
    validate_source(SOURCE)


def plan(job='PREPARE', root=DEFAULT_ORIGIN):
    validate_source(SOURCE)
    initial = resume.origin(root)
    prior, report = initial['manifest'], initial['checkpoint_report']
    if (prior.get('experiment') != 'additive-all-failure' or prior.get('gate_ablation')
            or prior['arm_config'].get('candidate_gate', 'distinct5') != 'distinct5'):
        raise ValueError('Use an unchanged additive checkpoint, not B/C')
    start = report['iteration']+1
    out = RUNTIME/f'evaluations/arm-failure-coverage-pilot-{job}'
    run = f'arm-failure-coverage-pilot-{job}'
    p = deepcopy(prior)
    def rewrite(s): return s.replace(prior['output'], str(out)).replace(prior['source'], str(SOURCE))
    p['command'] = [rewrite(s) for s in prior['command']]
    p['environment'] = {k:rewrite(v) for k,v in prior['environment'].items()}
    p['arm_config'] = {k:rewrite(v) if isinstance(v,str) else v for k,v in prior['arm_config'].items()}
    p['arm_config'].pop('deadline_epoch_seconds', None)
    p.update(job_id=str(job), source=str(SOURCE), output=str(out), resume_from=initial['root'],
        resume_origin=initial, initial_optimizer_updates=report['completed_optimizer_updates'],
        checkpoint=report['checkpoint'], start_rollout_id=start, requested_iterations=1,
        target_completed_iterations=start+1, wandb_run_id=run, fresh_optimizer=False,
        failure_coverage_pilot=True, variant_continuation=False, diagnostic_only=True,
        diagnostic_live_shadow=True, evaluation_in_allocation=False, evaluation_reserve_seconds=0,
        compute_approved=False)
    p['requested_resources'] = dict(gpus=4, hours=3, gpu_hours=12, cpus=32, memory_gib=480, browsers=32)
    p['environment'].update(NUM_GPUS='4', TP_SIZE='4', NUM_ROLLOUT=str(start+1),
        WANDB_RESUME='never', WANDB_RUN_ID=run, WANDB_PROJECT='openwebrl-evals',
        SLIME_LOAD_CHECKPOINT=str(Path(initial['root'])/'runtime'),
        BROWSER_CONCURRENCY='32', SGLANG_CONCURRENCY='32',
        OPENWEBRL_MULTIMODAL_STORAGE_DIR=str(RUNTIME/f'multimodal-scratch/arm-variant-{job}'))
    p['multimodal_storage'] = dict(mode='shared', directory=p['environment']['OPENWEBRL_MULTIMODAL_STORAGE_DIR'],
        minimum_free_bytes=2*1024**4, retention='preserve; no automatic deletion')
    p['browser_config']['browser_rollout_concurrency'] = 32
    for key, value in [('--wandb-project', 'openwebrl-evals'), ('--wandb-group', 'failure-coverage-audit')]:
        p['command'][p['command'].index(key)+1] = value
    if '--use-checkpoint-opt-param-scheduler' not in p['command']:
        p['command'].append('--use-checkpoint-opt-param-scheduler')
    p['selector_port'] = 31000+int(job)%20000 if str(job).isdigit() else 27511
    p['arm_config'].update(run_id=run, run_output=str(out), output=str(out),
        checkpoint=report['checkpoint'], policy_id=run+':uninitialized',
        selector_endpoint=f'http://127.0.0.1:{p["selector_port"]}',
        shadow_only=True, train_after_calibration=False, minimum_cycle_seconds=5400,
        failure_ablation='coverage', failure_beta=.5, failure_turn_budget=4)
    p['gpu_restore_output'] = str(Path(initial['root'])/f'gpu-restore-coverage-{job}-4gpu')
    p['resume_restore_receipt'] = str(Path(p['gpu_restore_output'])/'result.json')
    p['comparison'] = dict(optimizer_updates=0, ordinary_groups=48,
        historical='20% turn sampling, at least one usable label to admit a failure group',
        treatment='up to8 raw valid failure groups, min(4,T) uniform pre-action states per trajectory',
        beta='mixed .5 and failure .5; failure-only beta1 is a separate future training treatment',
        source='same frozen actor, task collection and ordinary labels for paired coverage/cost report',
        maximum_new_candidate_responses=640, maximum_failure_labels=160,
        limitation='Pipeline and usable-label yield; no claim about improved task success')
    return p


def fingerprint():
    paths = [SOURCE/'reference_manifest.json', *[REPO/n for n in (
        'scripts/prepare_arm_failure_coverage.py', 'scripts/arm_failure_coverage_pilot.sbatch',
        'scripts/run_arm_turn_bonus_cycles.py', 'scripts/check_arm_gpu_checkpoint_restore.py',
        'tests/test_arm_failure_coverage.py')]]
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def validate_plan(p):
    resume.validate_resume_plan(p)
    if p != plan(p['job_id'], p['resume_from']): raise ValueError('Coverage pilot plan changed')
    receipt = json.loads((PREPARATION/'readiness.json').read_text())
    if not receipt.get('cpu_passed') or not receipt.get('native_parse_passed') or receipt['fingerprint'] != fingerprint():
        raise ValueError('Missing/stale coverage CPU readiness')
    return receipt


def native_check(p):
    env = dict(os.environ, **p['environment'])
    env.update(DRY_RUN='1', BROWSER_TRAIN_CONFIG=str(PREPARATION/'browser.json'),
        FLASHINFER_WORKSPACE_BASE='/tmp/arm-coverage-flashinfer', CUDA_VISIBLE_DEVICES='')
    write_json(PREPARATION/'browser.json', p['browser_config'])
    argv = shlex.split(subprocess.check_output(p['command'], env=env, text=True))
    env.pop('DRY_RUN')
    code = """from unittest.mock import patch
from slime.utils.arguments import parse_args
with patch('megatron.training.arguments.get_device_arch_version', return_value=9): a=parse_args()
expected={'actor_num_gpus_per_node':4,'tensor_model_parallel_size':4,'global_batch_size':256,
'micro_batch_size':1,'ppo_epochs':2,'rollout_batch_size':48,'browser_rollout_concurrency':32,
'judge_api_model':'gpt-4.1','judge_prompt_variant':'action_history',
'dynamic_sampling_filter_path':'openwebrl.arm_failure_additive.filter_groups',
'use_rollout_logprobs':True,'wandb_project':'openwebrl-evals','use_checkpoint_opt_param_scheduler':True}
assert all(getattr(a,k)==v for k,v in expected.items()),{k:getattr(a,k) for k in expected}
from openwebrl.arm_failure_aux import check_topology
check_topology(a)
print('FAILURE_COVERAGE_NATIVE_PARSE_OK')
"""
    with (PREPARATION/'native-parse.log').open('w') as log:
        subprocess.run(source_command(SOURCE, [sys.executable, '-c', code, *argv[2:]]),
            env=env, cwd=SOURCE, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)


def check(root=DEFAULT_ORIGIN):
    prepare(); PREPARATION.mkdir(parents=True, exist_ok=True)
    code = """import sys, unittest
sys.path.insert(0,%r)
sys.path.insert(1,%r)
names=['test_arm_failure_coverage','test_arm_failure_additive']
result=unittest.TextTestRunner(verbosity=2).run(unittest.TestLoader().loadTestsFromNames(names))
raise SystemExit(not result.wasSuccessful())
""" % (str(REPO/'tests'), str(REPO/'scripts'))
    env = dict(os.environ, PYTHONPATH=str(SOURCE), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        FLASHINFER_WORKSPACE_BASE='/tmp/arm-coverage-flashinfer', XDG_CACHE_HOME='/tmp/arm-coverage-cache')
    with (PREPARATION/'frozen-tests.log').open('w') as log:
        subprocess.run([sys.executable, '-c', code], cwd=SOURCE, env=env,
            stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
    subprocess.run(['bash', '-n', str(REPO/'scripts/arm_failure_coverage_pilot.sbatch')], check=True)
    p = plan(root=root)
    write_json(PREPARATION/'plan.json', p)
    native_check(p)
    receipt = dict(cpu_passed=True, native_parse_passed=True, gpu_validated=False,
        compute_approved=False, source=str(SOURCE), plan=str(PREPARATION/'plan.json'), fingerprint=fingerprint())
    write_json(PREPARATION/'readiness.json', receipt)
    validate_plan(p)
    return receipt


def execute(p):
    validate_plan(p)
    if os.environ.get('SLURM_JOB_ID') != p['job_id']:
        raise ValueError('An explicitly authorized existing allocation is required')
    from resume_arm_failure_variants import check_multimodal_storage
    check_multimodal_storage(p)
    manifest = RUNTIME/f'logs/arm-failure-coverage-pilot-{p["job_id"]}.json'
    write_json(manifest, p)
    subprocess.run(source_command(SOURCE, [sys.executable, str(REPO/'scripts/check_arm_gpu_checkpoint_restore.py'),
        '--training-root', p['resume_from'], '--job-id', p['job_id'], '--gpus', '4',
        '--output', p['gpu_restore_output'], '--continuation-plan', str(manifest), '--execute']), check=True)
    training.execute(p)  # owns and awaits actor, browsers and selector; no detached allocation workers


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--job-id', default='PREPARE')
    parser.add_argument('--resume-from', type=Path, default=DEFAULT_ORIGIN)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.check: print(json.dumps(check(args.resume_from), indent=2))
    else:
        prepare(); p = plan(args.job_id, args.resume_from)
        if args.execute: execute(p)
        else: print(json.dumps(p, indent=2))
