#!/usr/bin/env python3
"""Allocation-owned servers and pilot; every child is awaited before exit."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from urllib.request import urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--browser-python', required=True)
    parser.add_argument('--engine-python', required=True)
    parser.add_argument('--env-file', required=True)
    parser.add_argument('--continuation-driver', type=Path)
    parser.add_argument('--closure-evidence', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    plan = json.loads((root/'plan.json').read_text())
    if plan.get('authorization_status') == 'pending':
        raise ValueError('New allocation proposal is not approved')
    models = json.loads((root/'models.json').read_text())
    processes, handles = [], []
    started = time.monotonic()
    allocation_seconds = int(os.environ.get('WEBPRM_ALLOCATION_SECONDS', '7200'))
    if not 0 < allocation_seconds <= plan['approval']['gpu_seconds_total']:
        raise ValueError('Allocation exceeds the approved total time')
    deadline = started + allocation_seconds - 180
    stopped = False

    def stop(signum, frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        # Own and await the real browser/action fixture before loading models.
        subprocess.run([args.browser_python, 'scripts/run_webprm_pilot.py', '--fixture',
                        '--output', str(root/'fixture'),
                        '--shepherd-source',str(root/'sources/shepherd'),
                        '--shepherd-actor-source',str(root/'sources/shepherd-demo.py'),
                        '--arbiter-source',str(root/'sources/arbiter')], check=True, timeout=120)
        for name, port, served in [('shepherd8b', 8011, 'WebShepherd8B'), ('arbiter7b', 8012, 'WebArbiter7B')]:
            if models[name]['status'] != 'downloaded':
                raise ValueError('Incomplete model download')
            log = (root/f'{name}-server.log').open('a')
            handles.append(log)
            command = [args.engine_python, '-m', 'sglang.launch_server',
                       '--model-path', models[name]['path'], '--served-model-name', served,
                       '--host', '127.0.0.1', '--port', str(port), '--tp', '1',
                       # Explicit caches prevent either server consuming the shared GPU.
                       # The fraction is a profiling ceiling, not a per-process partition.
                       '--mem-fraction-static', '0.70',
                       '--max-total-tokens', '65536',
                       # Native model ceilings; live OM2W states exceed the
                       # released Shepherd example's 10K serving restriction.
                       # Do not truncate scientific inputs or scale RoPE.
                       '--context-length', '40960' if name=='shepherd8b' else '32768',
                       '--random-seed','42',
                       '--max-running-requests', '4' if name=='shepherd8b' else '2', '--disable-cuda-graph']
            (root/f'{name}-server-command.json').write_text(json.dumps(command, indent=2))
            processes.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,start_new_session=True))
        ready = set()
        startup_deadline = min(deadline, time.monotonic()+900)
        while len(ready) < 2:
            if stopped or time.monotonic() >= startup_deadline:
                raise RuntimeError('Server startup stopped or timed out')
            for i, process in enumerate(processes):
                if process.poll() is not None:
                    raise RuntimeError('Server exited; inspect retained server log')
                try:
                    with urlopen(f'http://127.0.0.1:{8011+i}/v1/models', timeout=2) as response:
                        served = json.load(response)['data'][0]['id']
                    if served != ['WebShepherd8B', 'WebArbiter7B'][i]:
                        raise RuntimeError('Served model identity differs')
                    ready.add(i)
                except OSError:
                    pass
            time.sleep(3)
        subprocess.run([args.browser_python, 'scripts/preflight_webprm_models.py',
                        '--root', str(root)], check=True, timeout=min(480, max(1, int(deadline-time.monotonic()))))
        remaining = min(plan['approval']['gpu_seconds_total'], int(deadline-time.monotonic()))
        if remaining < 300:
            raise RuntimeError('Insufficient allocation time after startup')
        log = (root/'pilot-controller.log').open('a')
        handles.append(log)
        command = [args.browser_python, 'scripts/run_webprm_pilot.py',
                   '--output', str(root/'episodes'), '--tasks', str(root/'online-mind2web.jsonl'),
                   '--plan', str(root/'plan.json'), '--seconds', str(remaining),
                   '--quality-gate-tasks', str(plan.get('operational_settings', {}).get('quality_gate_tasks', 2)),
                   '--shepherd-source', str(root/'sources/shepherd'),
                   '--arbiter-source', str(root/'sources/arbiter'),
                   '--arbiter-benchmark', str(root/'WebPRMBench.jsonl'),
                   '--shepherd-actor-source',str(root/'sources/shepherd-demo.py'),
                   '--env-file',args.env_file]
        if args.continuation_driver:
            if not args.closure_evidence or not args.closure_evidence.is_file():
                raise ValueError('Reviewed continuation requires durable closure evidence')
            manifest = json.loads((root/'episodes/manifest.json').read_text())
            command = [args.browser_python, str(args.continuation_driver.resolve()),
                       '--runtime', str(root),
                       '--deadline', str(min(manifest['config']['deadline'], time.time()+remaining)),
                       '--allocation-deadline', str(time.time()+deadline-time.monotonic()),
                       '--closure-evidence', str(args.closure_evidence.resolve())]
        elif plan.get('shared_budget_path'):
            command.extend(['--budget-path', plan['shared_budget_path']])
        worker = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,start_new_session=True)
        processes.append(worker)
        while worker.poll() is None:
            if stopped or time.monotonic() >= deadline:
                raise RuntimeError('Allocation deadline or termination signal')
            if any(p.poll() is not None for p in processes[:-1]):
                raise RuntimeError('Reward server died during evaluation')
            time.sleep(3)
        if worker.returncode:
            raise RuntimeError('Pilot controller failed; retained artifacts require diagnosis')
    finally:
        for process in reversed(processes):
            try:
                os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError:
                pass
        for process in reversed(processes):
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL)
                process.wait()
        for handle in handles:
            handle.close()
        (root/'allocation-exit.json').write_text(json.dumps(dict(
            job_id=os.environ.get('SLURM_JOB_ID'), elapsed_seconds=time.monotonic()-started,
            child_returncodes=[p.returncode for p in processes], completed=(root/'episodes/complete.json').exists()), indent=2))


if __name__ == '__main__':
    main()
