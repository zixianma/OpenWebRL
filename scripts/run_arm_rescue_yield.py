#!/usr/bin/env python3
"""Prepare/execute a bounded rescue-yield pilot; never submit allocations."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
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
from runtime_ports import PORT_ENV, lease_ports, patch_rollout_ports

REPO, RUNTIME = evaluation.REPO, evaluation.RUNTIME
PARENT = RUNTIME/'reference-arm-eval80-additive-20260919-v1'
SOURCE = RUNTIME/'reference-arm-rescue-yield-20260920-v2'
CONTROL = RUNTIME/'arm-turn-bonus-preparation/rescue-yield-20260920'
CHECKPOINT = RUNTIME/'runs/openwebrl-4b-reference-294421-20260913T211532/iter_0000089'
TRAIN_TASKS = REPO/'openwebrl/data/webgym_filtered_popular_2102_cleaned.parquet'


def normalized(text):
    return ' '.join(str(text).casefold().split())


def selector_environment(env, device):
    return dict(env, CUDA_VISIBLE_DEVICES=device, PYTHONPATH=str(SOURCE),
        CPATH=str(RUNTIME/'src/python-headers/Include')+':'+str(RUNTIME/'src/python-headers'))


def prepare():
    validate_source(PARENT)
    CONTROL.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    from openwebrl.arm_rescue_yield import stable_seed
    evaluation_tasks = [json.loads(line) for line in (REPO/'openwebrl/data/eval/online-mind2web.jsonl').read_text().splitlines()]
    excluded_ids = {str(t.get('task_id', t.get('id'))).rsplit('/', 1)[-1] for t in evaluation_tasks}
    excluded_intents = {normalized(t.get('confirmed_task', t.get('intent', t.get('task', '')))) for t in evaluation_tasks}
    rows = []
    for row in pd.read_parquet(TRAIN_TASKS).to_dict('records'):
        meta = row['metadata']
        if (str(meta['task_id']).rsplit('/', 1)[-1] in excluded_ids
                or normalized(meta.get('task', meta.get('intent', ''))) in excluded_intents):
            continue
        rows.append(dict(prompt=list(row['prompt']), metadata=dict(meta)))
    rows.sort(key=lambda row: stable_seed(20260920, str(row['metadata']['task_id']), 'screen'))
    cohort = rows[:64]
    if len(cohort) != 64 or len({r['metadata']['task_id'] for r in cohort}) != 64:
        raise ValueError('Need 64 distinct training-only task IDs')
    serialized = ''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in cohort)
    files = ['openwebrl/arm_rescue_yield.py', 'openwebrl/arm_inference.py']
    if not SOURCE.exists():
        shutil.copytree(PARENT, SOURCE, symlinks=True, copy_function=copy_plain,
            ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git', '.browser_use_sessions'))
        for relative in files:
            copy_plain(REPO/relative, SOURCE/relative)
        port_file = SOURCE/'slime/ray/rollout.py'
        port_file.write_text(patch_rollout_ports(port_file.read_text()))
        files.append('slime/ray/rollout.py')
        (SOURCE/'rescue_yield_tasks.jsonl').write_text(serialized)
        config = {'eval': {'defaults': dict(n_samples_per_eval_prompt=5, temperature=.8,
            top_p=1., top_k=-1, max_response_len=1024), 'datasets': [dict(name='arm-rescue-screen',
                path=str(SOURCE/'rescue_yield_tasks.jsonl'), input_key='prompt', metadata_key='metadata',
                custom_generate_function_path='openwebrl.arm_rescue_yield.generate')]}}
        write_json(SOURCE/'rescue_yield_screen.json', config)
        manifest = json.loads((SOURCE/'reference_manifest.json').read_text())
        files += ['rescue_yield_tasks.jsonl', 'rescue_yield_screen.json']
        manifest['recipe_files_sha256'].update({f: hashlib.sha256((SOURCE/f).read_bytes()).hexdigest() for f in files})
        manifest['rescue_yield'] = dict(parent=str(PARENT), changed_files=files, optimizer_updates=0)
        manifest['runtime_port_leases'] = True
        write_json(SOURCE/'reference_manifest.json', manifest)
    validate_source(SOURCE)
    for relative in ['openwebrl/arm_rescue_yield.py', 'openwebrl/arm_inference.py']:
        if (SOURCE/relative).read_bytes() != (REPO/relative).read_bytes():
            raise ValueError('Use a new frozen source version after implementation changes')
    if (SOURCE/'rescue_yield_tasks.jsonl').read_text() != serialized:
        raise ValueError('Prespecified screening cohort changed')
    write_json(CONTROL/'cohort-audit.json', dict(training_tasks=len(rows), screening_tasks=64,
        train_file_sha256=hashlib.sha256(TRAIN_TASKS.read_bytes()).hexdigest(),
        cohort_sha256=hashlib.sha256(serialized.encode()).hexdigest(),
        task_ids=[r['metadata']['task_id'] for r in cohort], evaluation_ids_and_intents_excluded=True,
        seed=20260920, selection='Uniform hash order of eligible training tasks; no reward-based shortlist'))


def plan(job):
    prepare()
    output = RUNTIME/f'evaluations/arm-rescue-yield-{job}'
    p = evaluation.build_plan(SOURCE, CHECKPOINT, output, job, gpus=2)
    p.update(wandb_run_id=f'arm-rescue-yield-{job}', protocol='training-task rescue yield; not an OM2W benchmark',
        requested_resources=dict(gpus=2, hours=2, cpus=16, memory_gib=480), expected_screen_tasks=64)
    command = p['command']
    command[command.index('--wandb-group')+1] = 'ARM-rescue-yield-baseline-iter90'
    command[command.index('--eval-config')+1] = str(SOURCE/'rescue_yield_screen.json')
    index = command.index('--save-debug-rollout-data')
    del command[index:index+2]  # Every attempt has a durable archive; avoid duplicating a giant aggregate.
    command += ['--eval-function-path', 'openwebrl.arm_rescue_yield.generate_rollout',
                '--sglang-mem-fraction-static', '.35']
    p['selector_port'] = 31000 + int(job)%20000 if str(job).isdigit() else 27581
    p['environment'].update(WANDB_RUN_ID=p['wandb_run_id'], WANDB_RESUME='never',
        WANDB_CACHE_DIR=str(output/'wandb-cache'), BROWSER_CONCURRENCY='16', SGLANG_CONCURRENCY='16',
        BROWSER_TRAIN_CONFIG=str(output/'browser.json'), OPENWEBRL_EXPECTED_SCHEDULER_OFFSET='1',
        OPENWEBRL_RESCUE_YIELD_CONFIG=str(output/'rescue-config.json'),
        OPENWEBRL_MULTIMODAL_STORAGE_DIR=str(RUNTIME/f'multimodal-scratch/rescue-yield-{job}'))
    p['environment'].pop('OPENWEBRL_EVAL_ROLLOUT_DIR', None)
    p['rescue_config'] = dict(output=str(output), tasks=str(SOURCE/'rescue_yield_tasks.jsonl'),
        policy_id='qcq7i4ug:completed90:adam1016', checkpoint=str(CHECKPOINT),
        max_rescue_tasks=8, seed=20260920, selector_endpoint=f'http://127.0.0.1:{p["selector_port"]}')
    return evaluation.configure_evaluation_tracking(p, 'openwebrl-evals')


def execute(p):
    from dotenv import dotenv_values
    import yaml
    job = p['job_id']
    if os.getenv('SLURM_JOB_ID') != job or f'/job_{job}/' not in Path('/proc/self/cgroup').read_text():
        raise ValueError('Execute only inside the authorized GPU allocation')
    resources = allocation(subprocess.check_output(['scontrol', 'show', 'job', job, '-o'], text=True),
        job, requested_gpus=2, maximum_hours=2)
    if resources['cpus'] < 16 or resources['allocated_memory_gib'] < 480:
        raise ValueError('Prepared pilot requires 16 CPUs / 480 GiB RAM')
    steps = subprocess.check_output(['squeue', '--steps', f'--jobs={job}', '--noheader', '--format=%i'], text=True).split()
    allowed = {f'{job}.{s}' for s in ['batch', 'extern', os.getenv('SLURM_STEP_ID')]}
    if set(steps)-allowed:
        raise ValueError('Pilot requires a dedicated allocation')
    deadline = time.monotonic()+resources['maximum_seconds']
    output = Path(p['output']); output.mkdir(parents=True, exist_ok=False)
    write_json(output/'evaluation_manifest.json', p)
    write_json(output/'rescue-config.json', p['rescue_config'])
    browser = yaml.safe_load((SOURCE/'openwebrl/browser_training_config.yaml').read_text())
    browser['browser_rollout_concurrency'] = 16
    write_json(output/'browser.json', browser)
    scratch = Path(p['environment']['OPENWEBRL_MULTIMODAL_STORAGE_DIR'])
    scratch.mkdir(parents=True, exist_ok=False)
    if shutil.disk_usage(scratch).free < 2*1024**4:
        raise OSError('Require 2 TiB free shared scratch before starting GPU workers')
    view = output/'checkpoint-view'; view.mkdir()
    (view/CHECKPOINT.name).symlink_to(CHECKPOINT, target_is_directory=True)
    (view/'rollout').symlink_to(CHECKPOINT.parent/'rollout', target_is_directory=True)
    (view/'latest_checkpointed_iteration.txt').write_text('89\n')
    env = {k: v for k, v in clean_environment().items() if not k.startswith('OPENWEBRL_ARM_')}
    for k, v in dotenv_values(REPO/'.env').items():
        if v and k.startswith(('WANDB_', 'JUDGE_', 'OPENAI_', 'AZURE_')):
            env.setdefault(k, v)
    if env.get('OPENAI_API_KEY') and not env.get('JUDGE_API_BASE'):
        env.update(JUDGE_API_MODE='served', JUDGE_API_BASE='https://api.openai.com/v1')
    env.update(p['environment'])
    subprocess.run(source_command(SOURCE, [sys.executable, str(REPO/'scripts/inspect_training_checkpoint.py'),
        str(view), '--expected-updates', '1016', '--expected-scheduler-offset-updates', '1',
        '--report', str(output/'checkpoint_validation.json')]), env=env, check=True, stdout=subprocess.DEVNULL)
    children, logs = [], []
    port_lease, port_base = lease_ports(job)
    env[PORT_ENV] = str(port_base)
    write_json(output/'port-lease.json', dict(base=port_base, size=256, job_id=job))
    def stop(*unused):
        for child in reversed(children):
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
    old = [signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)]
    try:
        devices = env.get('CUDA_VISIBLE_DEVICES', '').split(',')
        if len(devices) != 2 or not all(devices):
            raise ValueError('Exactly two assigned GPU IDs required')
        logs.append((output/'selector.log').open('w'))
        selector = subprocess.Popen([str(RUNTIME/'arm-reproduction/venv/bin/python'),
            str(SOURCE/'scripts/serve_arm.py'), '--mode', 'selection', '--model', str(SELECTOR),
            '--port', str(p['selector_port'])], cwd=SOURCE, env=selector_environment(env, devices[-1]),
            stdout=logs[-1], stderr=subprocess.STDOUT, start_new_session=True)
        children.append(selector)
        startup = min(deadline-1800, time.monotonic()+600)
        while True:
            if selector.poll() is not None:
                raise RuntimeError('Selector exited at startup')
            try:
                with urllib.request.urlopen(p['rescue_config']['selector_endpoint']+'/health', timeout=2):
                    break
            except OSError:
                pass
            if time.monotonic() >= startup:
                raise TimeoutError('Selector startup exceeded time reserve')
            time.sleep(5)
        selector_preflight(p['rescue_config']['selector_endpoint'], output)
        remaining = int(deadline-time.monotonic())
        if remaining < 1800:
            raise TimeoutError('Insufficient remaining collection time')
        logs.append((output/'evaluation.log').open('w'))
        worker = subprocess.Popen(source_command(SOURCE, ['timeout', '--signal=INT', '--kill-after=90',
            str(remaining), *p['command']]), cwd=SOURCE, env=env,
            stdout=logs[-1], stderr=subprocess.STDOUT, start_new_session=True)
        children.append(worker)
        while worker.poll() is None:
            if selector.poll() is not None:
                raise RuntimeError('Selector exited during pilot')
            evaluation.record_restore_evidence(p)
            write_json(output/'status.json', dict(stage='collecting', complete=False, optimizer_updates=0,
                screen_trajectories=len(list((output/'screen_records').glob('*.json'))),
                retry_trajectories=len(list((output/'retry_records').glob('*.json'))), updated_epoch=time.time()))
            time.sleep(10)
        write_json(output/'launcher_exit.json', dict(returncode=worker.returncode))
        evaluation.record_restore_evidence(p)
        text = (output/'evaluation.log').read_text(errors='replace')
        if worker.returncode or re.search(r'train_one_step start|\[TrainMetrics\]', text):
            raise RuntimeError('Pilot failed or unexpectedly entered training; preserve all artifacts')
        if not (output/'checkpoint_restore_evidence.json').is_file() and not re.search(evaluation.restore_pattern(p), text):
            raise ValueError('Missing native GPU checkpoint-restore evidence')
        summary = json.loads((output/'yield-summary.json').read_text())
        records = [json.loads(f.read_text()) for phase in ['smoke', 'screen', 'retry']
                   for f in (output/f'{phase}_records').glob('*.json')]
        for record in records:
            archive = Path(record['artifact'])/(hashlib.sha256(record['task_id'].encode()).hexdigest()+'.pt')
            if not archive.is_file() or not archive.with_suffix('.json').is_file():
                raise ValueError('Missing attempt archive or judge sidecar')
        if not summary['complete'] or summary['screen_trajectories'] != 320 or summary['optimizer_updates'] != 0:
            raise ValueError('Incomplete prespecified pilot')
        write_json(output/'status.json', dict(stage='complete', complete=True, optimizer_updates=0,
            summary=summary, saved_trajectories=len(records), updated_epoch=time.time()))
    except BaseException as exc:
        write_json(output/'status.json', dict(stage='failed', complete=False, error_type=type(exc).__name__,
            error=str(exc)[:400], optimizer_updates=0, updated_epoch=time.time()))
        raise
    finally:
        stop()
        for child in children:
            try:
                child.wait(timeout=90)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL); child.wait()
        for handle in logs:
            handle.close()
        port_lease.close()
        for sig, handler in zip((signal.SIGINT, signal.SIGTERM), old):
            signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-id', default='PREPARE')
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    p = plan(args.job_id)
    if args.execute:
        execute(p)
    else:
        write_json(CONTROL/'plan.json', p)
        print(json.dumps(p, indent=2))


if __name__ == '__main__':
    main()
