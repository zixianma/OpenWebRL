#!/usr/bin/env python3
"""Freeze or own a bounded local harmful-states pilot; never submit Slurm jobs.

Preparation performs no model inference, browser work, or paid API calls. Running
requires a separate exact approval.json and a registered two-H200 allocation.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time
import urllib.request

REPO = Path(__file__).resolve().parents[1]
RUNTIME = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime')
DEFAULT_ROOT = RUNTIME / 'evaluations/harmful-states-pilot-20261010'
ACTOR = Path('/gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT')
SELECTOR = RUNTIME / 'arm-turn-bonus-preparation/arm-refresh-20260924/baseline40/model-compat'
RESOURCES = dict(gpus=2, gpu_type='H200', cpus=16, memory_gib=240,
                 total_seconds=3600, total_gpu_hours=2)
LIMITS = dict(paid_api_calls=0, paid_api_usd=0, campaigns=45, episodes=105,
              actor_requests=1650, selector_requests=150)
PROTOCOL = dict(seeds=[43, 44, 45], methods=['ordinary', 'retries5', 'selection5'],
                cases=5, max_steps=10, max_browser_operations_per_episode=30,
                episodes=5, candidates=5, temperature=0.7, top_p=0.9,
                top_k=-1, max_new_tokens=1024)
FINAL_STATES = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY',
                'NODE_FAIL', 'BOOT_FAIL', 'DEADLINE', 'REVOKED', 'PREEMPTED'}


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('w') as handle:
        json.dump(value, handle, indent=2)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def environment(source):
    """Do not read .env or forward API keys/tokens to local model workers."""
    allowed = {'HOME', 'USER', 'LOGNAME', 'PATH', 'SHELL', 'LANG', 'LC_ALL', 'TZ',
               'TMPDIR', 'CUDA_VISIBLE_DEVICES', 'SLURM_TMPDIR'}
    env = {key: value for key, value in os.environ.items()
           if key in allowed or key.startswith(('SLURM_', 'PMI_', 'PMIX_'))}
    env.update(PYTHONPATH=str(source), WANDB_PROJECT='openwebrl-evals',
               WANDB_MODE='disabled', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               HF_HUB_DISABLE_TELEMETRY='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               NO_PROXY='127.0.0.1,localhost,::1', no_proxy='127.0.0.1,localhost,::1',
               SLIME_BROWSER_ENV_MODE='local_process', FLASHINFER_WORKSPACE_BASE=str(RUNTIME))
    return env


def source_command(source, command):
    return ['bash', '-c', 'set -euo pipefail\nsource ' +
            shlex.quote(str(source / 'scripts/h200_env.sh')) + '\nexec ' + shlex.join(command)]


def source_files():
    files = {}
    for package in ('openwebrl', 'slime', 'slime_plugins'):
        for path in (REPO / package).rglob('*.py'):
            if not {'docs', 'data', '__pycache__'} & set(path.relative_to(REPO).parts):
                files[str(path.relative_to(REPO))] = path
    for path in (REPO / 'openwebrl/env/prompts').rglob('*'):
        if path.is_file() and path.suffix in {'.md', '.json', '.txt', '.yaml', '.yml'}:
            files[str(path.relative_to(REPO))] = path
    for name in ('scripts/run_harmful_states_pilot.py', 'scripts/serve_arm.py',
                 'scripts/h200_env.sh', 'scripts/runtime_ports.py',
                 'openwebrl/browser_training_config.yaml', 'openwebrl/env/config.yaml'):
        files[name] = REPO / name
    files['selector_source/inference/selection_prompt.py'] = (
        RUNTIME / 'arm-reproduction/source/inference/selection_prompt.py')
    for required in ('openwebrl/harmful_states.py', 'openwebrl/harmful_states_server.py',
                     'openwebrl/harmful_states_browser.py', 'openwebrl/harmful_states_worker.py'):
        if required not in files:
            raise FileNotFoundError(f'Pilot source is still being prepared: {required}')
    return files


def model_inventory(path):
    if not path.is_dir():
        raise FileNotFoundError(path)
    result = {}
    for file in sorted(path.iterdir()):
        if not file.is_file():
            continue
        stat = file.stat()
        row = dict(resolved_path=str(file.resolve()), size=stat.st_size, mtime_ns=stat.st_mtime_ns)
        if file.suffix in {'.json', '.jinja', '.txt'}:
            row['sha256'] = sha(file)
        result[file.name] = row
    if 'config.json' not in result or not any(name.endswith('.safetensors') for name in result):
        raise ValueError(f'Model files are incomplete: {path}')
    return result


def prepare(root):
    root = Path(root).resolve()
    files = source_files()
    hashes = {name: sha(path) for name, path in sorted(files.items())}
    source = root / ('source-' + digest(hashes)[:16])
    for name, original in files.items():
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        data = original.read_bytes()
        if target.exists():
            if target.read_bytes() != data:
                raise ValueError(f'Frozen source changed: {target}')
        else:
            with target.open('xb') as handle:
                handle.write(data)
    manifest = dict(schema=1, source=str(source), files_sha256=hashes,
                    contents='Python source, browser prompts, local config, pinned selector prompt; no .env or task payloads')
    manifest_path = source / 'pilot_source_manifest.json'
    if manifest_path.exists() and read(manifest_path) != manifest:
        raise ValueError('Frozen source manifest changed')
    if not manifest_path.exists():
        write(manifest_path, manifest)
    sys.path.insert(0, str(source))
    from openwebrl.harmful_states import list_cases
    cases = [case['case_id'] for case in list_cases()]
    if len(cases) != PROTOCOL['cases'] or len(set(cases)) != len(cases):
        raise ValueError('Pilot must have five distinct frozen fixtures')
    plan = dict(schema=1, root=str(root), source=str(source), resources=RESOURCES,
                limits=LIMITS, protocol=PROTOCOL, case_ids=cases, actor=str(ACTOR),
                selector_model=str(SELECTOR), source_manifest_sha256=sha(manifest_path),
                models={str(path): model_inventory(path) for path in (ACTOR, SELECTOR)},
                worker_module='openwebrl.harmful_states_worker',
                status='Prepared only; exact resource approval required before allocation',
                cost_semantics='Total allocation including model startup and every failed attempt; no paid API',
                scientific_status='Synthetic harness pilot, not validated safety benchmark')
    if (root / 'plan.json').exists() and read(root / 'plan.json') != plan:
        raise ValueError('Existing plan differs; prepare a new explicit root, preserving earlier attempts')
    write(root / 'plan.json', plan)
    write(root / 'approval-template.json', dict(approved=False, plan_sha256=digest(plan),
          resources=RESOURCES, limits=LIMITS, attempts=[]))
    # Compile without importing GPU/model code or writing bytecode into frozen source.
    for name in hashes:
        if name.endswith('.py'):
            compile((source / name).read_text(), str(source / name), 'exec')
    write(root / 'readiness.json', dict(source_compiles=True, source_files=len(hashes),
          plan_sha256=digest(plan), inference_calls=0, browser_calls=0, paid_api_calls=0,
          gpu_startup_verified=False, worker_runtime_import_verified=False))
    return dict(plan=str(root / 'plan.json'), plan_sha256=digest(plan), source=str(source),
                resources=RESOURCES, limits=LIMITS, status=plan['status'])


def validate(root):
    root = Path(root).resolve()
    plan = read(root / 'plan.json')
    if (plan['root'] != str(root) or plan['resources'] != RESOURCES or
            plan['limits'] != LIMITS or plan['protocol'] != PROTOCOL):
        raise ValueError('Frozen protocol or resource limits changed')
    source = Path(plan['source'])
    manifest = source / 'pilot_source_manifest.json'
    if sha(manifest) != plan['source_manifest_sha256']:
        raise ValueError('Source manifest changed')
    for name, expected in read(manifest)['files_sha256'].items():
        if sha(source / name) != expected:
            raise ValueError(f'Frozen source changed: {name}')
    for name, expected in plan['models'].items():
        if model_inventory(Path(name)) != expected:
            raise ValueError(f'Pinned local model files changed: {name}')
    if sha(__file__) != sha(source / 'scripts/run_harmful_states_pilot.py'):
        raise ValueError('Execute using the frozen controller recorded in plan.json')
    return plan


def approved_seconds(plan, approval, job):
    if (approval.get('approved') is not True or approval.get('plan_sha256') != digest(plan) or
            approval.get('resources') != RESOURCES or approval.get('limits') != LIMITS):
        raise ValueError('Exact resource and zero-API approval is required')
    attempts = approval.get('attempts', [])
    ids = [str(item['job_id']) for item in attempts]
    if len(ids) != len(set(ids)) or not all(value.isdecimal() for value in ids):
        raise ValueError('Duplicate or invalid allocation IDs')
    current = [item for item in attempts if str(item['job_id']) == job]
    if len(current) != 1:
        raise ValueError('Allocation is not registered in approval.json')
    used, accounting = 0, []
    for previous in ids:
        if previous == job:
            continue
        raw = subprocess.check_output(['sacct', '-X', '-n', '-P', '-j', previous,
                                      '--format=JobIDRaw,State,ElapsedRaw,AllocTRES'], text=True)
        rows = [line.split('|') for line in raw.splitlines() if line.split('|')[0] == previous]
        if len(rows) != 1 or rows[0][1].split()[0] not in FINAL_STATES:
            raise ValueError('Every previous allocation needs terminal scheduler accounting')
        elapsed = int(rows[0][2])
        if elapsed < 0:
            raise ValueError('Negative elapsed allocation time')
        used += elapsed
        accounting.append(dict(job_id=previous, elapsed_seconds=elapsed, sacct=raw))
    seconds = current[0]['time_limit_seconds']
    if type(seconds) is not int or seconds <= 0 or used + seconds > RESOURCES['total_seconds']:
        raise ValueError('All attempts together exceed the original one-hour allocation cap')
    return seconds, used, accounting


def allocation_deadline(job, seconds):
    info = subprocess.check_output(['scontrol', 'show', 'job', job, '-o'], text=True)
    fields = dict(re.findall(r'(\w+)=(\S+)', info))
    tres = dict(part.split('=', 1) for part in fields.get('AllocTRES', '').split(',') if '=' in part)
    if (fields.get('JobId') != job or fields.get('JobState') != 'RUNNING' or
            not fields.get('UserId', '').endswith(f'({os.getuid()})') or fields.get('NumNodes') != '1' or
            fields.get('NumCPUs') != '16' or tres.get('gres/gpu:h200') != '2' or
            tres.get('gres/gpu') != '2' or tres.get('mem') not in ('240G', '245760M')):
        raise ValueError('Allocation does not match the exact two-H200/16-CPU/240-GiB profile')
    limit = fields['TimeLimit']
    days, clock = limit.split('-', 1) if '-' in limit else ('0', limit)
    parts = [int(value) for value in clock.split(':')]
    if len(parts) == 2:
        parts.insert(0, 0)
    duration = int(days) * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]
    if duration > seconds:
        raise ValueError('Allocation time exceeds registered remaining approval')
    end = datetime.strptime(fields['EndTime'], '%Y-%m-%dT%H:%M:%S').timestamp()
    return min(time.time() + duration, end) - 120, info


def stop_all(children):
    # All children were started in private process groups by this controller.
    for child in reversed(children):
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 30
    for child in reversed(children):
        try:
            child.wait(timeout=max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait(timeout=10)
    # A server leader may exit before a worker descendant. Signal the complete
    # owned process group even when Popen has already reaped the leader.
    for child in reversed(children):
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def verify_completion(collection, plan):
    sentinel = read(collection / 'complete.json')
    if (sentinel.get('complete') is not True or sentinel.get('model_performance_result') is not True or
            sentinel.get('campaigns') != LIMITS['campaigns'] or sentinel.get('valid_campaigns') != 45 or
            sentinel.get('external_api_calls') != 0):
        raise ValueError('Worker did not verify all 45 valid local-only campaigns')
    for key in ('episodes', 'actor_requests', 'selector_requests'):
        value = sentinel.get(key)
        if type(value) is not int or not 0 <= value <= LIMITS[key]:
            raise ValueError('Missing or exceeded cumulative worker cap: ' + key)
    files = sentinel.get('record_files', [])
    paths = {str(path.resolve()) for path in (collection / 'records').glob('*.json')}
    if len(files) != 45 or {str(Path(row['path']).resolve()) for row in files} != paths or len(paths) != 45:
        raise ValueError('Completion does not bind exactly 45 durable campaign records')
    expected = {(case, seed, method) for case in plan['case_ids']
                for seed in PROTOCOL['seeds'] for method in PROTOCOL['methods']}
    actual, total_episodes = set(), 0
    for row in files:
        path = Path(row['path']).resolve()
        if path.parent != (collection / 'records').resolve() or sha(path) != row['sha256']:
            raise ValueError('Campaign record escaped its output root or changed')
        record = read(path)
        if record.get('valid') is not True or record.get('protocol_sha256') != sentinel['protocol_sha256']:
            raise ValueError('Incomplete campaign or mismatched protocol')
        actual.add((record['case_id'], record['seed'], record['method']))
        total_episodes += len(record['episodes'])
    if actual != expected or total_episodes != sentinel['episodes']:
        raise ValueError('Campaign schedule or episode accounting differs')
    return sentinel


def execute(root, job):
    if not job or not job.isdecimal() or os.environ.get('SLURM_JOB_ID') != job:
        raise ValueError('Run only inside the registered approved Slurm allocation')
    if f'/job_{job}/' not in Path('/proc/self/cgroup').read_text():
        raise ValueError('Controller must run inside a Slurm job step on its assigned node')
    root = Path(root).resolve()
    plan = validate(root)
    seconds, used, accounting = approved_seconds(plan, read(root / 'approval.json'), job)
    deadline, scheduler = allocation_deadline(job, seconds)
    if deadline - time.time() < 300:
        raise TimeoutError('Insufficient approved time remains for bounded startup')
    source = Path(plan['source'])
    env = environment(source)
    devices = env.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if len(devices) != 2 or len(set(devices)) != 2 or not all(devices):
        raise ValueError('Exactly two isolated CUDA devices are required')
    sys.path.insert(0, str(source / 'scripts'))
    from runtime_ports import lease_ports
    children, handles, lease = [], [], None
    collection = root / 'collection'
    journal = root / 'scheduler-attempts' / job
    completed = False

    def heartbeat(stage):
        write(root / 'current.json', dict(job_id=job, stage=stage, epoch=time.time(),
              prior_actual_seconds=used, current_attempt_limit_seconds=seconds,
              deadline_unix=deadline, verified_complete=False))
        if time.time() >= deadline:
            raise TimeoutError('Approved time ending; preserve partial campaign state')

    def spawn(command, label, child_env):
        handle = (journal / (label + '.log')).open('a')
        handles.append(handle)
        child = subprocess.Popen(command, cwd=source, env=child_env, stdout=handle,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        children.append(child)
        return child

    def interrupted(number, frame):
        raise SystemExit(128 + number)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    with (root / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        journal.mkdir(parents=True, exist_ok=False)
        write(journal / 'allocation.json', dict(job_id=job, scheduler=scheduler, resources=RESOURCES,
              prior_attempts=accounting, prior_actual_seconds=used, time_limit_seconds=seconds,
              plan_sha256=digest(plan)))
        try:
            lease, port = lease_ports(job)
            actor = spawn(source_command(source, [str(RUNTIME / 'venv/bin/python'), '-m', 'sglang.launch_server',
                  '--model-path', plan['actor'], '--host', '127.0.0.1', '--port', str(port),
                  '--dtype', 'bfloat16', '--tp', '1', '--mem-fraction-static', '.45',
                  '--context-length', '32768', '--max-running-requests', '8',
                  '--chunked-prefill-size', '4096', '--disable-cuda-graph']),
                  'actor', dict(env, CUDA_VISIBLE_DEVICES=devices[0]))
            selector_env = dict(env, CUDA_VISIBLE_DEVICES=devices[1],
                LD_LIBRARY_PATH=str(RUNTIME / 'arm-reproduction/refresh-torch28/torch/lib'),
                CPATH=str(RUNTIME / 'src/python-headers/Include') + ':' + str(RUNTIME / 'src/python-headers'),
                TRITON_CACHE_DIR=str(journal / 'triton'))
            selector = spawn([str(RUNTIME / 'arm-reproduction/venv-refresh/bin/python'),
                str(source / 'scripts/serve_arm.py'), '--mode', 'selection', '--model', plan['selector_model'],
                '--base', plan['selector_model'], '--source-root', str(source / 'selector_source'),
                '--port', str(port + 1)], 'selector', selector_env)
            pending = {'actor': f'http://127.0.0.1:{port}/get_model_info',
                       'selector': f'http://127.0.0.1:{port + 1}/health'}
            startup_deadline = min(deadline, time.time() + 900)
            while pending:
                heartbeat('model_startup')
                if actor.poll() is not None or selector.poll() is not None:
                    raise RuntimeError('A local model server exited during startup')
                if time.time() >= startup_deadline:
                    raise TimeoutError('Model startup timeout')
                for label, url in list(pending.items()):
                    try:
                        with urllib.request.urlopen(url, timeout=3) as response:
                            identity = json.load(response)
                        expected = plan['actor'] if label == 'actor' else plan['selector_model']
                        if identity.get('model_path' if label == 'actor' else 'model') != expected:
                            raise ValueError('Local model identity does not match frozen plan')
                        if label == 'selector' and (identity.get('adapter') or identity.get('mode') != 'selection'):
                            raise ValueError('Unexpected selector adapter or mode')
                        write(journal / (label + '-identity.json'), identity)
                        pending.pop(label)
                    except OSError:
                        pass
                if pending:
                    time.sleep(2)
            spawn(['nvidia-smi', '--query-gpu=timestamp,index,uuid,utilization.gpu,memory.used,power.draw',
                   '--format=csv', '-l', '10'], 'gpu', env)
            config = dict(output=str(collection), actor_port=port, actor=plan['actor'],
                selector_endpoint=f'http://127.0.0.1:{port + 1}', selector_model=plan['selector_model'],
                deadline_unix=deadline, seeds=PROTOCOL['seeds'], max_steps=PROTOCOL['max_steps'],
                max_browser_operations_per_episode=PROTOCOL['max_browser_operations_per_episode'],
                episodes=PROTOCOL['episodes'], candidates=PROTOCOL['candidates'],
                limits=LIMITS, plan_sha256=digest(plan))
            write(journal / 'worker-config.json', config)
            worker = spawn(source_command(source, [str(RUNTIME / 'venv/bin/python'), '-m',
                plan['worker_module'], '--config', str(journal / 'worker-config.json')]),
                'collection', dict(env, CUDA_VISIBLE_DEVICES=devices[0]))
            while worker.poll() is None:
                heartbeat('collecting')
                if actor.poll() is not None or selector.poll() is not None:
                    raise RuntimeError('A model server exited during collection')
                time.sleep(5)
            if worker.returncode:
                raise RuntimeError('Pilot worker failed; preserve every prior attempt and ledger')
            sentinel = verify_completion(collection, plan)
            write(root / 'completion.json', dict(complete=True, job_id=job, epoch=time.time(),
                  plan_sha256=digest(plan), worker_completion_sha256=sha(collection / 'complete.json'),
                  worker=sentinel, prior_actual_seconds=used,
                  final_allocation_accounting='Obtain terminal sacct elapsed after batch exits'))
            completed = True
        except BaseException as exc:
            write(root / 'current.json', dict(job_id=job, verified_complete=False, failed=True,
                  error_type=type(exc).__name__, error=str(exc), epoch=time.time()))
            raise
        finally:
            stop_all(children)
            for handle in handles:
                handle.close()
            if lease is not None:
                lease.close()
            if completed:
                write(root / 'current.json', dict(job_id=job, verified_complete=True, epoch=time.time(),
                      stage='complete', completion_sha256=sha(root / 'completion.json')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', '--dry-run', action='store_true', dest='prepare')
    mode.add_argument('--execute', action='store_true')
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--job-id')
    args = parser.parse_args()
    if args.prepare:
        print(json.dumps(prepare(args.root), indent=2))
    else:
        execute(args.root, args.job_id)


if __name__ == '__main__':
    main()
