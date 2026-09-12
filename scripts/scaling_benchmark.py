#!/usr/bin/env python3
"""Bounded scaling experiment in a separately approved eight-H200 allocation."""
import argparse
import ast
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from resume_baseline import RUNTIME, allocation, clean_environment, validate_source, write_json

REPO = Path(__file__).resolve().parents[1]
SOURCE = RUNTIME / 'benchmark-eightgpu-20260912'
ANCESTOR = RUNTIME / 'runs/openwebrl-4b-reference-288861-20260912T040027'


def validate_benchmark_source(source):
    validate_source(source)
    manifest = json.loads((source / 'reference_manifest.json').read_text())['scaling_benchmark']
    for section in ('changed_scripts_sha256', 'browser_configs_sha256'):
        if not manifest.get(section):
            raise ValueError(f'Missing benchmark provenance: {section}')
        for relative, expected in manifest[section].items():
            path = (source / relative).resolve()
            if not path.is_relative_to(source.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f'Benchmark source changed: {relative}')


def health_result(path, memory_gib=960):
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    if not rows:
        raise ValueError('Missing resource telemetry')
    def events(row):
        return dict(x.split() for x in row.get('memory.events', '').splitlines())
    first, last = events(rows[0]), events(rows[-1])
    oom = any(int(last.get(k, 0)) > int(first.get(k, 0)) for k in ('oom', 'oom_kill'))
    peak = max(int(x.get('memory.current', 0)) for x in rows) / 1024**3
    return {'sampled_peak_memory_gib': peak, 'new_host_oom': oom,
            'safe_to_increase_browsers': not oom and peak < memory_gib * 0.85}


def cpu_sample():
    path = Path('/sys/fs/cgroup') / Path('/proc/self/cgroup').read_text().strip().split('::', 1)[1].lstrip('/')
    path = next((p for p in (path, *path.parents) if p.name.startswith('job_')), path)
    return {'time': time.time(), **{name: (path / name).read_text() for name in
            ('cpu.stat', 'cpu.pressure', 'memory.pressure') if (path / name).exists()}}


def resources(info, job, now=None):
    base = allocation(info, job, now=now, requested_gpus=4)
    fields = dict(re.findall(r'(\w+)=(\S+)', info))
    tres = dict(x.split('=', 1) for x in fields['AllocTRES'].split(','))
    if int(tres.get('gres/gpu:h200', 0)) != 8 or base['cpus'] < 64 or base['allocated_memory_gib'] < 960:
        raise ValueError('Benchmark needs a dedicated eight-H200,64-CPU,960-GiB allocation')
    base['gpus'] = 8
    base['deadline'] = datetime.fromisoformat(base['end_time']).timestamp() - 180
    return base


def case_plan(job, output, name, tp, browsers, replay):
    if tp not in (1, 2, 4, 8) or browsers not in (64, 96, 128, 192, 256):
        raise ValueError('Unsupported benchmark candidate')
    output = Path(output).resolve()
    if not output.is_relative_to(RUNTIME):
        raise ValueError('Benchmark artifacts must remain in runtime storage')
    # Exact matching checkpoint/batch pair, shared across optimizer topology cases.
    index = 60 if replay else 61
    return {'job_id': str(job), 'name': name, 'output': str(output), 'source': str(SOURCE),
        'checkpoint': str(ANCESTOR / f'iter_{index:07d}'), 'checkpoint_index': index,
        'start_adam_updates': 720 if replay else 732, 'rollout_id': index + 1,
        'replay_batch': str(ANCESTOR / 'rollout_recovery/61.pt') if replay else None,
        'replay_consumed_groups': 144 if replay else None,
        'tensor_parallel': tp, 'data_parallel': 8 // tp, 'gpus': 8, 'browsers': browsers,
        'wandb_run_id': f'qcq7i4ug-scale-{job}-{name}', 'writes_main_run_pointer': False,
        'case_timeout_seconds': 1800 if replay else 2400}


def parse_result(text, plan):
    records = [ast.literal_eval(x) for x in re.findall(r'model.py:\d+ - step \d+: (\{.*\})', text)]
    unique = {x['train/step']: x for x in records}
    if len(unique) != len(records) or not records:
        raise ValueError('Missing or duplicate optimizer records')
    for record in records:
        if any(not math.isfinite(float(record[k])) for k in ['train/loss', 'train/grad_norm', 'train/ppo_kl']):
            raise ValueError('Nonfinite optimizer metric')
    if plan['replay_batch'] and len(records) != 12:
        raise ValueError('Replay must reproduce all twelve optimizer updates')
    timers = re.findall(r'Timer train end \(elapsed: ([\d.]+)s\)', text)
    if not timers:
        raise ValueError('Missing optimizer timer')
    train_seconds = max(map(float, timers))
    if not math.isfinite(train_seconds) or train_seconds <= 0:
        raise ValueError('Invalid optimizer timer')
    result = {'optimizer_updates': len(records), 'train_seconds': train_seconds,
              'seconds_per_update': train_seconds / len(records),
              'max_grad_norm': max(x['train/grad_norm'] for x in records)}
    if not plan['replay_batch']:
        n = plan['rollout_id'] + 1
        done = re.search(rf'rollout={n}/90 event=done groups=48/48 completed_groups=(\d+) pending_groups=(\d+) elapsed_secs=([\d.]+)', text)
        if not done:
            raise ValueError('No complete 48-group browser collection')
        result.update(collection_seconds=float(done[3]), completed_groups=int(done[1]), submitted_groups=int(done[1])+int(done[2]))
        times = {}
        pattern = rf'\[([\d-]+ [\d:]+)\] train.py:\d+ - \[TrainProgress\] rollout={n}/90 phase=(generate|recovery_checkpoint_complete)'
        for timestamp, phase in re.findall(pattern, text):
            times[phase] = datetime.fromisoformat(timestamp)
        if set(times) != {'generate', 'recovery_checkpoint_complete'}:
            raise ValueError('Missing full generation-to-checkpoint timing')
        result['iteration_seconds'] = (times['recovery_checkpoint_complete'] - times['generate']).total_seconds()
        if result['iteration_seconds'] <= 0:
            raise ValueError('Invalid full iteration timer')
        result['gpu_hours_per_iteration'] = 8 * result['iteration_seconds'] / 3600
    return result


def worker(plan):
    job = plan['job_id']
    if os.environ.get('SLURM_JOB_ID') != job or f'/job_{job}/' not in Path('/proc/self/cgroup').read_text():
        raise ValueError('Worker must execute inside its authorized Slurm step')
    info = subprocess.check_output(['scontrol', 'show', 'job', job, '-o'], text=True)
    budget = resources(info, job)
    source, out, checkpoint = map(Path, [plan['source'], plan['output'], plan['checkpoint']])
    validate_benchmark_source(source)
    view = out / 'checkpoint-view'
    view.mkdir(parents=True, exist_ok=False)
    (view / checkpoint.name).symlink_to(checkpoint, target_is_directory=True)
    (view / 'rollout').symlink_to(checkpoint.parent / 'rollout', target_is_directory=True)
    (view / 'latest_checkpointed_iteration.txt').write_text(str(plan['checkpoint_index'])+'\n')
    run = out / 'run'
    env = clean_environment()
    env.update(OPENWEBRL_BENCHMARK_OUTPUT=str(run), OPENWEBRL_STOP_AFTER_SAVED_ROLLOUT=str(plan['rollout_id']),
               RAY_ADDRESS='local', WANDB_RUN_ID=plan['wandb_run_id'])
    if plan['replay_batch']:
        env.update(OPENWEBRL_REPLAY_FIRST_BATCH=plan['replay_batch'], OPENWEBRL_REPLAY_ROLLOUT_ID=str(plan['rollout_id']),
                   OPENWEBRL_REPLAY_CONSUMED_GROUPS=str(plan['replay_consumed_groups']))
    seconds = min(plan['case_timeout_seconds'], int(budget['deadline']-time.time())-120)
    if seconds < 600:
        raise ValueError('Insufficient remaining benchmark time')
    argv = ['timeout', '--signal=INT', '--kill-after=90', str(seconds), sys.executable,
            str(source / 'scripts/run_small_baseline.py'), '--profile', 'reference', '--gpus', '8',
            '--tensor-parallel', str(plan['tensor_parallel']), '--browser-concurrency', str(plan['browsers']),
            '--resume-from', str(view), '--wandb-run-id', plan['wandb_run_id'], '--env-file', str(REPO / '.env')]
    start = time.monotonic()
    monitor = None
    with (out / 'launcher.log').open('w') as log, (out / 'monitor.log').open('w') as health, (out / 'cpu.jsonl').open('w', buffering=1) as cpu:
        proc = subprocess.Popen(argv, env=env, cwd=source, stdout=log, stderr=subprocess.STDOUT)
        next_cpu_sample = 0
        while proc.poll() is None:
            if time.monotonic() >= next_cpu_sample:
                cpu.write(json.dumps(cpu_sample()) + '\n')
                next_cpu_sample = time.monotonic() + 30
            if monitor is None and run.is_dir():
                monitor = subprocess.Popen([sys.executable, str(REPO / 'scripts/monitor_baseline.py'), str(run), '--seconds', str(seconds+120)], stdout=health, stderr=subprocess.STDOUT)
            time.sleep(5)
        write_json(run / 'exit_status.json', {'exit_code': proc.returncode}) if run.exists() else None
        if monitor is not None:
            monitor.wait(timeout=60)
    if proc.returncode:
        write_json(out / 'result.json', {'complete': False, 'returncode': proc.returncode, 'wall_seconds': time.monotonic()-start})
        return proc.returncode
    text = (run / 'training.log').read_text(errors='replace')
    if f'at iteration {plan["checkpoint_index"]}' not in text or 'successfully loaded checkpoint' not in text:
        raise ValueError('Missing GPU restore evidence')
    result = parse_result(text, plan)
    result.update(health_result(run / 'health.jsonl', budget['allocated_memory_gib']))
    if result['new_host_oom']:
        raise ValueError('Host OOM during the benchmark; candidate cannot be ranked')
    marker = int((run / 'latest_checkpointed_iteration.txt').read_text())
    if marker != plan['rollout_id']:
        raise ValueError('Expected completed benchmark checkpoint was not saved')
    inspection = [sys.executable, str(REPO / 'scripts/inspect_training_checkpoint.py'), str(run),
                  '--expected-updates', str(plan['start_adam_updates'] + result['optimizer_updates']),
                  '--expected-scheduler-offset-updates', '1', '--report', str(out / 'checkpoint_validation.json')]
    subprocess.run(inspection, check=True, stdout=subprocess.DEVNULL)
    result.update(complete=True, returncode=0, wall_seconds=time.monotonic()-start, **plan)
    write_json(out / 'result.json', result)
    print(json.dumps(result), flush=True)
    return 0


def controller(job, *, topology_only=False):
    info = subprocess.check_output(['scontrol', 'show', 'job', job, '-o'], text=True)
    budget = resources(info, job)
    root = RUNTIME / 'benchmarks' / f'qcq7i4ug-scaling-{job}'
    root.mkdir(parents=True, exist_ok=False)
    results = []

    def execute(name, tp, browsers, replay):
        if budget['deadline'] - time.time() < 1200:
            return None
        out = root / name
        out.mkdir()
        plan = case_plan(job, out, name, tp, browsers, replay)
        path = out / 'plan.json'; write_json(path, plan)
        command = ['srun', f'--jobid={job}', '--nodes=1', '--ntasks=1', '--cpus-per-task=64',
                   '--gres=gpu:h200:8', '--cpu-bind=none', '--exact',
                   'bash', '-c', f'source {SOURCE}/scripts/h200_env.sh\nexec python {REPO}/scripts/scaling_benchmark.py --worker {path}']
        print(json.dumps({'starting_case': plan}), flush=True)
        code = subprocess.run(command, env=clean_environment()).returncode
        report = out / 'result.json'
        result = json.loads(report.read_text()) if report.exists() else {'complete': False, 'returncode': code, **plan}
        results.append(result); write_json(root / 'results.json', results)
        return result if code == 0 and result.get('complete') else None

    topology = []
    for tp in (4, 2, 8, 1):
        result = execute(f'tp{tp}-replay', tp, 64, True)
        if result:
            topology.append(result)
    if not topology:
        raise SystemExit('No topology completed the optimizer correctness checks')
    winner = min(topology, key=lambda x: x['seconds_per_update'])['tensor_parallel']
    browser_results = []
    for count in (() if topology_only else (64, 96, 128, 192, 256)):
        result = execute(f'tp{winner}-b{count}', winner, count, False)
        if result:
            browser_results.append(result)
            if not result['safe_to_increase_browsers']:
                break
        elif count >= 128:
            break
        if len(browser_results) >= 2 and count >= 128:
            previous = min(x['iteration_seconds'] for x in browser_results[:-1])
            if result and result['iteration_seconds'] >= previous:
                break
    write_json(root / 'summary.json', {'topology_only': topology_only, 'topology_winner_tp': winner, 'results': results,
        'browser_winner': min(browser_results, key=lambda x: x['iteration_seconds'])['browsers'] if browser_results else None,
        'selection_is_provisional': True, 'notes': (
            'Saved-batch optimizer comparison only; browser throughput and full-cycle scaling remain unmeasured.'
            if topology_only else
            'Single live-web trials; compare failures and resource use before promoting. Startup-inclusive wall time is not steady-state iteration time.')})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--worker', type=Path)
    mode.add_argument('--controller', metavar='JOB_ID')
    p.add_argument('--topology-only', action='store_true', help='Replay saved batches only; do not start browser trials.')
    a = p.parse_args()
    if a.worker and a.topology_only:
        p.error('--topology-only is a controller option')
    if a.worker:
        raise SystemExit(worker(json.loads(a.worker.read_text())))
    controller(a.controller, topology_only=a.topology_only)
