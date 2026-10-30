#!/usr/bin/env python3
"""Prepare or execute the 64-state audit. Never submits a Slurm allocation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import urllib.request

import evaluate_baseline_checkpoint as evaluation
from prepare_arm_turn_bonus import copy_plain
from resume_baseline import allocation, clean_environment, source_command, validate_source, write_json
from run_arm_turn_bonus_calibration import SELECTOR, selector_preflight
from runtime_ports import PORT_ENV, lease_ports

REPO, RUNTIME = evaluation.REPO, evaluation.RUNTIME
PARENT = RUNTIME/'reference-arm-rescue-yield-20260920-v2'
SOURCE = RUNTIME/'reference-arm-quality-audit-20260921-v2'
CONTROL = RUNTIME/'arm-turn-bonus-preparation/quality-audit-20260921-v2'


def prepare_source():
    validate_source(PARENT)
    changed = ['openwebrl/arm_quality_audit.py']
    if not SOURCE.exists():
        shutil.copytree(PARENT, SOURCE, symlinks=True, copy_function=copy_plain,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git', '.browser_use_sessions'))
        for name in changed:
            copy_plain(REPO/name, SOURCE/name)
        manifest = json.loads((SOURCE/'reference_manifest.json').read_text())
        manifest['recipe_files_sha256'].update({p: hashlib.sha256((SOURCE/p).read_bytes()).hexdigest() for p in changed})
        manifest['quality_audit'] = dict(parent=str(PARENT), changed_files=changed, optimizer_updates=0)
        write_json(SOURCE/'reference_manifest.json', manifest)
    validate_source(SOURCE)
    if any((SOURCE/p).read_bytes() != (REPO/p).read_bytes() for p in changed):
        raise ValueError('Freeze a new source version after editing audit code')


def plan(job):
    prepare_source()
    panel = json.loads((CONTROL/'comparison-plan.json').read_text())
    from openwebrl.arm_quality_audit import validate_inputs
    validate_inputs(panel)
    root = RUNTIME/f'evaluations/arm-quality-audit-{job}'
    stages = []
    for actor in panel['actors']:
        checkpoint = Path(actor['checkpoint'])
        # Reuse the tested eval-only topology; SFT explicitly starts from HF.
        base = Path(panel['actors'][-1]['checkpoint']) if actor['name'] == 'sft' else checkpoint
        p = evaluation.build_plan(SOURCE, base, root/actor['name'], job, gpus=2)
        p.update(checkpoint=str(checkpoint), completed_training_iterations=actor['completed_iterations'],
                 checkpoint_index=actor['completed_iterations']-1, actor=actor['name'],
                 wandb_run_id=f'arm-quality-{job}-{actor["name"]}',
                 protocol='offline fixed-state candidate/selector audit; not OM2W success')
        command = p['command']
        i = command.index('--save-debug-rollout-data'); del command[i:i+2]
        command += ['--eval-function-path', 'openwebrl.arm_quality_audit.generate_rollout',
                    '--sglang-mem-fraction-static', '.35']
        command[command.index('--wandb-group')+1] = 'ARM-quality-fixed-states'
        p['environment'].pop('OPENWEBRL_EVAL_ROLLOUT_DIR', None)
        p['environment'].update(WANDB_RUN_ID=p['wandb_run_id'], WANDB_RESUME='never',
            SGLANG_CONCURRENCY='24', BROWSER_CONCURRENCY='1',
            OPENWEBRL_ARM_QUALITY_CONFIG=str(root/actor['name']/'quality-config.json'))
        if actor['name'] == 'sft':
            p['environment'].pop('SLIME_CKPT_STEP', None)
            p['environment']['SLIME_LOAD_CHECKPOINT'] = str(checkpoint)
            command.remove('--use-checkpoint-opt-param-scheduler')
        p['quality_config'] = dict(states=panel['states'], states_sha256=panel['states_sha256'],
            actor=actor['name'], checkpoint=str(checkpoint), output=p['output'])
        stages.append(evaluation.configure_evaluation_tracking(p, 'openwebrl-evals'))
    return dict(job_id=str(job), output=str(root), stages=stages, panel=panel,
        requested_resources=dict(gpus=2, gpu_type='H200', cpus=16, memory_gib=480, hours=2, gpu_hours=4),
        teacher_max_calls=231, teacher_max_usd=20, optimizer_updates=0, browser_trajectories=0)


def environment():
    from dotenv import dotenv_values
    env = {k:v for k,v in clean_environment().items() if not k.startswith('OPENWEBRL_ARM_')}
    env.setdefault('FLASHINFER_WORKSPACE_BASE',str(RUNTIME))
    for k,v in dotenv_values(REPO/'.env').items():
        if v and k.startswith(('OPENAI_', 'JUDGE_', 'WANDB_', 'AZURE_')):
            env.setdefault(k,v)
    if env.get('OPENAI_API_KEY') and not env.get('JUDGE_API_BASE'):
        env.update(JUDGE_API_BASE='https://api.openai.com/v1', JUDGE_API_MODE='served')
    return env


def stop(child):
    if child is None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        child.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL); child.wait()


def worker(p, actor):
    job = p['job_id']
    if os.getenv('SLURM_JOB_ID') != job or f'/job_{job}/' not in Path('/proc/self/cgroup').read_text():
        raise ValueError('Run only in the dedicated authorized GPU job')
    stage = next(s for s in p['stages'] if s['actor'] == actor)
    root = Path(stage['output']); root.mkdir(parents=True, exist_ok=False)
    write_json(root/'evaluation_manifest.json', stage)
    env = environment(); env.update(stage['environment'])
    devices = env.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if len(devices) != 2 or not all(devices):
        raise ValueError('Exactly two assigned GPUs required')
    if actor != 'sft':
        ckpt = Path(stage['checkpoint']); view = root/'checkpoint-view'; view.mkdir()
        (view/ckpt.name).symlink_to(ckpt, target_is_directory=True)
        (view/'rollout').symlink_to(ckpt.parent/'rollout', target_is_directory=True)
        (view/'latest_checkpointed_iteration.txt').write_text(str(stage['checkpoint_index'])+'\n')
        subprocess.run(source_command(SOURCE, [sys.executable, str(REPO/'scripts/inspect_training_checkpoint.py'),
            str(view), '--expected-scheduler-offset-updates', '1', '--report', str(root/'checkpoint_validation.json')]),
            env=env, check=True, stdout=subprocess.DEVNULL)
    lease, base = lease_ports(job)
    env[PORT_ENV] = str(base)
    endpoint = f'http://127.0.0.1:{base+250}'
    stage['quality_config']['selector_endpoint'] = endpoint
    write_json(root/'quality-config.json', stage['quality_config'])
    selector = native = None
    handles = []
    try:
        handles.append((root/'selector.log').open('w'))
        selector_env = dict(env, CUDA_VISIBLE_DEVICES=devices[-1], PYTHONPATH=str(SOURCE),
                           CPATH=str(RUNTIME/'src/python-headers/Include')+':'+str(RUNTIME/'src/python-headers'))
        serving=stage.get('selector',{})
        selector_env.update(serving.get('environment',{}))
        selector_command=[serving.get('python',str(RUNTIME/'arm-reproduction/venv/bin/python')),
            str(SOURCE/'scripts/serve_arm.py'), '--mode','selection','--model',serving.get('model',str(SELECTOR)),
            '--port',str(base+250)]
        for flag in ('base','adapter','source-root'):
            if serving.get(flag):selector_command += ['--'+flag,serving[flag]]
        selector = subprocess.Popen(selector_command, env=selector_env, cwd=SOURCE, stdout=handles[-1],
            stderr=subprocess.STDOUT, start_new_session=True)
        limit = time.monotonic()+600
        while True:
            if selector.poll() is not None or time.monotonic()>limit:
                raise RuntimeError('Selector startup failed')
            try:
                with urllib.request.urlopen(endpoint+'/health', timeout=2) as response:
                    health=json.load(response)
                if serving and (health.get('model')!=serving['model'] or
                    health.get('adapter_sha256')!=serving.get('adapter_sha256')):
                    raise ValueError('Selector model/adapter identity changed')
                write_json(root/'selector-health.json',health)
                break
            except OSError: time.sleep(5)
        # Custom SelectionARM bases are checked against their declared path,
        # while historical callers retain the released-model default.
        selector_preflight(endpoint, root, expected_api_model=serving.get('model'))
        handles.append((root/'evaluation.log').open('w'))
        native = subprocess.Popen(source_command(SOURCE, stage['command']), env=env, cwd=SOURCE,
            stdout=handles[-1], stderr=subprocess.STDOUT, start_new_session=True)
        while native.poll() is None:
            if selector.poll() is not None: raise RuntimeError('Selector exited during audit')
            if actor != 'sft': evaluation.record_restore_evidence(stage)
            time.sleep(10)
        if native.returncode: raise RuntimeError(f'{actor} native worker failed: {native.returncode}')
        if actor != 'sft':
            evaluation.record_restore_evidence(stage)
            if not (root/'checkpoint_restore_evidence.json').exists():
                raise ValueError('Missing actual GPU checkpoint restoration evidence')
        saved = json.loads((root/'collection-complete.json').read_text())
        expected=p.get('expected_collection',dict(states=64,candidates=320))
        if any(saved.get(k)!=v for k,v in expected.items()):
            raise ValueError('Incomplete candidate collection')
        write_json(root/'status.json', saved)
    finally:
        stop(native); stop(selector)
        for handle in handles: handle.close()
        lease.close()


def execute(p):
    job = p['job_id']
    if os.getenv('SLURM_JOB_ID') != job:
        raise ValueError('An explicitly authorized allocation is required')
    resources = allocation(subprocess.check_output(['scontrol','show','job',job,'-o'], text=True),
                           job, requested_gpus=2, maximum_hours=2)
    if resources['cpus'] < 16 or resources['allocated_memory_gib'] < 480:
        raise ValueError('Allocation does not meet the approved profile')
    root = Path(p['output']); root.mkdir(exist_ok=False)
    write_json(root/'audit-plan.json', p)
    deadline = time.monotonic()+resources['maximum_seconds']
    teacher = None
    handle = (root/'teacher.log').open('w')
    active = None
    try:
        for stage in p['stages']:
            if deadline-time.monotonic()<600: raise TimeoutError('Insufficient time for another checkpoint')
            command = ['srun', f'--jobid={job}', '--nodes=1', '--ntasks=1', '--cpus-per-task=16',
                '--gres=gpu:h200:2', '--exact', '--cpu-bind=none', sys.executable, str(Path(__file__).resolve()),
                '--job-id',job,'--worker',stage['actor']]
            active = subprocess.Popen(command,start_new_session=True)
            while active.poll() is None:
                if time.monotonic()>=deadline:raise TimeoutError('Audit allocation deadline')
                if teacher is not None and teacher.poll() is not None:
                    raise RuntimeError('Teacher stopped; terminate generation promptly')
                write_json(root/'status.json',dict(complete=False,stage=stage['actor'],
                    saved_states=len(list((Path(stage['output'])/'records').glob('*.json')))))
                time.sleep(10)
            if active.returncode:raise RuntimeError(f'Native audit step failed: {active.returncode}')
            active = None
            if teacher is None:
                teacher = subprocess.Popen([sys.executable, str(REPO/'scripts/label_arm_quality.py'),
                    '--plan',str(root/'audit-plan.json'),'--execute'], env=environment(),
                    stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            elif teacher.poll() is not None:
                raise RuntimeError('Teacher stopped before all generation stages completed')
        write_json(root/'generation-complete.json', dict(complete=True))
        teacher.wait(timeout=max(1,deadline-time.monotonic()))
        if teacher.returncode: raise RuntimeError('Independent labeling failed; artifacts preserved')
        subprocess.run([sys.executable,str(REPO/'scripts/label_arm_quality.py'),
                        '--plan',str(root/'audit-plan.json'),'--report'],check=True)
        write_json(root/'status.json', dict(complete=True,states=64,candidate_sets=192,optimizer_updates=0))
    except BaseException as exc:
        write_json(root/'status.json',dict(complete=False,failed=True,error_type=type(exc).__name__,error=str(exc)[:300]))
        raise
    finally:
        stop(active);stop(teacher); handle.close()


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--job-id',default='PREPARE')
    a.add_argument('--execute',action='store_true')
    a.add_argument('--worker',choices=['sft','baseline20','baseline90'])
    args=a.parse_args()
    if args.worker:
        p=json.loads((RUNTIME/f'evaluations/arm-quality-audit-{args.job_id}/audit-plan.json').read_text())
        prepare_source(); worker(p,args.worker)
    else:
        p=plan(args.job_id)
        if args.execute: execute(p)
        else:
            write_json(CONTROL/'launch-plan.json',p)
            print(json.dumps(dict(output=p['output'],requested_resources=p['requested_resources'],
                                 stages=[s['actor'] for s in p['stages']],teacher_max_usd=20),indent=2))


if __name__=='__main__': main()
