#!/usr/bin/env python3
"""Own approved, independent GPU/CPU full-OM2W pools; never submit allocations."""
from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import urllib.request

SOURCE = Path(__file__).resolve().parents[1]
RUNTIME = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime')
PROJECT_ENV = Path('/gpfs/projects/krishna/zixianma/OpenWebRL/.env')
sys.path.insert(0, str(SOURCE))
from openwebrl.luna_qwen_metrics import digest, write_json
from openwebrl.luna_qwen_eval import verified_record
from openwebrl.luna_qwen_full_eval import validate_plan as validate_family_plan, verify_actor_files, verify_existing_records
from resume_baseline import source_command, validate_source
from run_arm_quality_audit import environment, stop
from runtime_ports import lease_ports

FINAL_STATES = frozenset(('COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT',
                         'OUT_OF_MEMORY', 'NODE_FAIL', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'))
PROFILES = {
    'gpu': dict(gpus=8, cpus=64, memory_gib=960, workers=8, families=['sft', 'qwen']),
    'cpu': dict(gpus=0, cpus=16, memory_gib=32, workers=4, families=['luna']),
}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def terminate(signum, frame):
    raise SystemExit(128 + signum)


def worker_environment():
    """Load allowed secrets at runtime; frozen sources never contain an .env."""
    from dotenv import dotenv_values
    env = environment()
    for key, value in dotenv_values(PROJECT_ENV).items():
        if value and key.startswith(('OPENAI_', 'JUDGE_', 'WANDB_', 'AZURE_')):
            env.setdefault(key, value)
    if env.get('OPENAI_API_KEY') and not env.get('JUDGE_API_BASE'):
        env.update(JUDGE_API_BASE='https://api.openai.com/v1', JUDGE_API_MODE='served')
    if not env.get('OPENAI_API_KEY'):
        raise ValueError('Required OPENAI_API_KEY is absent from inherited or authorized project environment')
    env.update(PYTHONPATH=str(SOURCE), WANDB_PROJECT='openwebrl-evals',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    return env


def check_approval(approval, plan, pool, job, used):
    """Require exact approval and charge every earlier attempt in this pool."""
    if pool not in PROFILES or type(used) is not int or used < 0:
        raise ValueError('Invalid pool or previous elapsed accounting')
    if (approval.get('approved') is not True or approval.get('plan_sha256') != digest(plan)
            or approval.get('limits') != plan['limits']
            or approval.get('pools') != {name: value['resources'] for name, value in plan['pools'].items()}):
        raise ValueError('Exact plan, resource and shared API budget approval required')
    attempts = approval.get('attempts', [])
    if len({a['job_id'] for a in attempts}) != len(attempts):
        raise ValueError('An allocation must occur exactly once in the shared attempt ledger')
    current = [a for a in attempts if a['job_id'] == job and a['pool'] == pool]
    if len(current) != 1:
        raise ValueError('Current allocation is absent from the approved pool')
    limit = current[0]['time_limit_seconds']
    if (type(limit) is not int or limit <= 0
            or used + limit > plan['pools'][pool]['resources']['total_seconds']):
        raise ValueError('Retry exceeds this pool total; previous runtime never resets')
    return current[0]


def previous_elapsed(approval, pool, job):
    total, receipts = 0, []
    for attempt in approval.get('attempts', []):
        if attempt['pool'] != pool or attempt['job_id'] == job:
            continue
        output = subprocess.check_output(['sacct', '-X', '-n', '-P', '-j', attempt['job_id'],
            '--format=JobIDRaw,State,ElapsedRaw,AllocTRES'], text=True)
        matches = [r.split('|') for r in output.splitlines() if r.split('|')[0] == attempt['job_id']]
        if len(matches) != 1 or matches[0][1].split()[0] not in FINAL_STATES:
            raise ValueError('Previous same-pool allocation lacks final scheduler accounting')
        elapsed = int(matches[0][2])
        if elapsed < 0:
            raise ValueError('Negative scheduler accounting')
        total += elapsed
        receipts.append(dict(job_id=attempt['job_id'], elapsed_seconds=elapsed, sacct=output))
    return total, receipts


def slurm_seconds(value):
    days, clock = value.split('-', 1) if '-' in value else ('0', value)
    parts = [int(v) for v in clock.split(':')]
    if len(parts) == 2:
        parts.insert(0, 0)
    if len(parts) != 3:
        raise ValueError('Unsupported Slurm time limit')
    return int(days) * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]


def validate_allocation(info, pool, job, time_limit, *, now=None, uid=None):
    fields = dict(re.findall(r'(\w+)=(\S+)', info))
    profile = PROFILES[pool]
    now = time.time() if now is None else now
    uid = os.getuid() if uid is None else uid
    if (fields.get('JobId') != job or fields.get('JobState') != 'RUNNING'
            or not fields.get('UserId', '').endswith(f'({uid})') or fields.get('NumNodes') != '1'):
        raise ValueError('Require the owned, running single-node allocation')
    tres = dict(part.split('=', 1) for part in fields.get('AllocTRES', '').split(',') if '=' in part)
    memory = re.fullmatch(r'([\d.]+)([KMGT])', tres.get('mem', ''))
    gib = float(memory[1]) * {'K': 2**-20, 'M': 2**-10, 'G': 1, 'T': 1024}[memory[2]] if memory else 0
    gpus = int(tres.get('gres/gpu', '0'))
    if (int(fields.get('NumCPUs', '0')) != profile['cpus'] or gib != profile['memory_gib']
            or gpus != profile['gpus']
            or (gpus and int(tres.get('gres/gpu:h200', '0')) != gpus)):
        raise ValueError('Allocation differs from the exact approved pool resources')
    if slurm_seconds(fields['TimeLimit']) > time_limit:
        raise ValueError('Scheduler time limit exceeds this approved attempt')
    end = datetime.strptime(fields['EndTime'], '%Y-%m-%dT%H:%M:%S').timestamp()
    remaining = min(time_limit, int(end - now)) - 180
    if remaining < 300:
        raise ValueError('Insufficient time after the three-minute shutdown margin')
    return dict(remaining_seconds=remaining, end_time=fields['EndTime'], resources=profile)


def validate_plans(root):
    root = Path(root).resolve()
    plan = read(root / 'plan.json')
    source = Path(plan['source']).resolve()
    if source != SOURCE.resolve() or plan['controller_sha256'] != sha(__file__):
        raise ValueError('Execute the exact frozen, prepared controller')
    validate_source(source)
    if sha(source / 'reference_manifest.json') != plan['source_manifest_sha256']:
        raise ValueError('Frozen source manifest changed')
    families = {}
    assigned = []
    for pool, expected in PROFILES.items():
        spec = plan['pools'][pool]
        if (spec['workers'] != expected['workers'] or spec['families'] != expected['families']
                or any(spec['resources'][k] != expected[k] for k in ('gpus', 'cpus', 'memory_gib'))
                or (pool == 'gpu' and spec['resources'].get('gpu_type') != 'H200')):
            raise ValueError('Pool topology differs from the prepared isolated worker layout')
        assigned.extend(spec['families'])
    if len(set(assigned)) != len(assigned) or set(assigned) != set(plan['families']):
        raise ValueError('Every family must have exactly one owning pool')
    for family, reference in plan['families'].items():
        family_path = Path(reference['plan'])
        fp = read(family_path)
        if digest(fp) != reference['plan_sha256'] or fp['family'] != family or fp['limits'] != plan['limits']:
            raise ValueError('Family plan or shared limits changed')
        if (fp['task_ids'] != plan['task_ids'] or fp['task_file_sha256'] != plan['task_file_sha256']
                or Path(fp['source']).resolve() != source
                or fp['source_manifest_sha256'] != plan['source_manifest_sha256']):
            raise ValueError('Family cohort or frozen source differs from the global plan')
        validate_family_plan(fp, dict(output=fp['output'], plan_sha256=reference['plan_sha256']))
        if not Path(fp['output']).resolve().is_relative_to(root):
            raise ValueError('Family outputs must stay inside this experiment root')
        if Path(fp['budget_root']).resolve() != root / 'budget':
            raise ValueError('All workers must use the same experiment API ledgers')
        if sha(fp['task_file']) != fp['task_file_sha256']:
            raise ValueError('Task file changed')
        verify_actor_files(fp)
        verify_existing_records(fp, reference['plan_sha256'])
        verify_records(fp)
        families[family] = fp
    return plan, families


def verify_records(plan):
    allowed = {(item['task_id'], mode) for item in plan['schedule'] for mode in item['modes']}
    count = 0
    for path in (Path(plan['output']) / 'records').glob('*.json'):
        record = verified_record(path, digest(plan['protocol']))
        key = (record['task_id'], record['mode'])
        if key not in allowed or path.name != digest(list(key))[:24] + '.json':
            raise ValueError('Unexpected or misnamed committed record')
        count += 1
    return count


def pending_blocks(output, schedule):
    return sum(any(not (Path(output) / 'records' / (digest([item['task_id'], mode])[:24] + '.json')).exists()
                   for mode in item['modes']) for item in schedule)


def choose_family(pending, active, initial=None, disabled=()):
    """Start both GPU families immediately; safely reuse slots as work ends."""
    order = ([initial] if initial else []) + list(pending)
    return next((family for family in dict.fromkeys(order)
                 if family not in disabled and pending[family] > active.get(family, 0)), None)


def worker_command(source, pool, job, config):
    cpus, memory = (8, 120) if pool == 'gpu' else (4, 8)
    gpu_flags = ['--gpus=h200:1', '--gres=gpu:h200:1'] if pool == 'gpu' else ['--gres=none']
    return ['srun', '--jobid=' + job, '--nodes=1', '--ntasks=1', f'--cpus-per-task={cpus}',
        *gpu_flags, f'--mem={memory}G', '--exact', '--cpu-bind=cores', sys.executable,
        str(Path(source) / 'scripts/run_luna_qwen_full300.py'), '--worker', str(config)]


def require_cgroup(job):
    if os.getenv('SLURM_JOB_ID') != job or f'/job_{job}/' not in Path('/proc/self/cgroup').read_text():
        raise ValueError('Workers must remain inside their approved allocation')


def stop_workers(children):
    """Signal every step immediately, then await them within one shared margin."""
    for entry in children.values():
        try:
            os.killpg(entry['process'].pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 60
    for entry in children.values():
        process = entry['process']
        try:
            process.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)


def worker(config):
    job, pool, family = config['job_id'], config['pool'], config['family']
    require_cgroup(job)
    signal.signal(signal.SIGTERM, terminate)
    root = Path(config['root'])
    plan, families = validate_plans(root)
    if config['global_plan_sha256'] != digest(plan) or family not in plan['pools'][pool]['families']:
        raise ValueError('Worker request differs from its prepared plan or pool')
    approval = read(root / 'approval.json')
    used, _ = previous_elapsed(approval, pool, job)
    check_approval(approval, plan, pool, job, used)
    fp = families[family]
    family_ref = plan['families'][family]
    logs = root / 'workers' / pool / job / str(config['worker_id'])
    logs.mkdir(parents=True, exist_ok=True)
    env = worker_environment()
    children, handles = [], []
    lease, port = lease_ports(job)

    def spawn(command, label, activate=False):
        handle = (logs / (label + '.log')).open('a')
        handles.append(handle)
        if activate:
            command = source_command(SOURCE, command)
        process = subprocess.Popen(command, cwd=SOURCE, env=env, stdout=handle,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        children.append(process)
        return process

    try:
        actor = None
        cfg = dict(output=fp['output'], plan=family_ref['plan'], plan_sha256=family_ref['plan_sha256'],
            worker_id=config['worker_id'], browser_port=port + 10, job_id=job, pool=pool,
            family=family, source=str(SOURCE))
        if pool == 'gpu':
            devices = env.get('CUDA_VISIBLE_DEVICES', '').split(',')
            if len(devices) != 1 or not devices[0]:
                raise ValueError('Expected one allocated H200 per local-model worker')
            uuid = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
            if not uuid.startswith('GPU-') or '\n' in uuid:
                raise ValueError('Expected exactly one cgroup-visible physical GPU')
            write_json(logs / 'device-assignment.json', dict(worker_id=config['worker_id'],
                family=family, gpu_uuid=uuid, cuda_visible_devices=devices,
                slurm_step_gpus=env.get('SLURM_STEP_GPUS')))
            cfg.update(actor_port=port, gpu_uuid=uuid)
            actor = spawn([str(RUNTIME / 'venv/bin/python'), '-m', 'sglang.launch_server',
                '--model-path', fp['actor_path'], '--host', '127.0.0.1', '--port', str(port),
                '--dtype', 'bfloat16', '--tp', '1', '--mem-fraction-static', '.60',
                '--context-length', '32768', '--max-running-requests', '10',
                '--chunked-prefill-size', '4096', '--disable-cuda-graph'], 'actor', True)
            deadline = time.monotonic() + 900
            while True:
                if actor.poll() is not None:
                    raise RuntimeError('Actor startup failed; preserve the logs for diagnosis')
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/get_model_info', timeout=5) as response:
                        json.load(response)
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError('Actor startup exceeded 15 minutes')
                    time.sleep(3)
            spawn(['nvidia-smi', '-i', uuid,
                '--query-gpu=timestamp,index,uuid,utilization.gpu,memory.used,power.draw',
                '--format=csv', '-l', '5'], 'devices')
        else:
            # A tokenizer/processor is allowed; no local model server or GPU is used.
            env['CUDA_VISIBLE_DEVICES'] = ''
        write_json(logs / 'worker-config.json', cfg)
        collection = spawn([str(RUNTIME / 'venv/bin/python'), '-m', 'openwebrl.luna_qwen_full_eval',
            '--config', str(logs / 'worker-config.json')], 'collection', True)
        while collection.poll() is None:
            if actor is not None and actor.poll() is not None:
                raise RuntimeError('Actor exited during collection')
            time.sleep(5)
        if collection.returncode:
            raise RuntimeError('Collection failed; do not retry the unchanged failure')
        status = read(Path(fp['output']) / 'workers' / str(config['worker_id']) / 'status.json')
        if not status.get('complete'):
            raise ValueError('Collection exited without a completion sentinel')
    finally:
        for process in reversed(children):
            stop(process)
        for handle in handles:
            handle.close()
        lease.close()


def execute(root, pool, job):
    root = Path(root).resolve()
    require_cgroup(job)
    signal.signal(signal.SIGTERM, terminate)
    plan, families = validate_plans(root)
    approval = read(root / 'approval.json')
    used, previous = previous_elapsed(approval, pool, job)
    attempt = check_approval(approval, plan, pool, job, used)
    # Presence validation happens before any model server, worker or API call.
    worker_environment()
    allocation_info = subprocess.check_output(['scontrol', 'show', 'job', job, '-o'], text=True)
    resources = validate_allocation(allocation_info, pool, job, attempt['time_limit_seconds'])
    pool_root = root / 'pools' / pool
    journal = pool_root / 'attempts' / job
    journal.mkdir(parents=True, exist_ok=True)
    write_json(journal / 'previous-accounting.json', dict(used_seconds=used, attempts=previous))
    spec = plan['pools'][pool]
    selected = {family: families[family] for family in spec['families']}
    children, history, failed = {}, [], {}
    generations = [0] * spec['workers']
    deadline = time.monotonic() + resources['remaining_seconds']

    def write_status(stage, **extra):
        write_json(pool_root / 'current.json', dict(job_id=job, pool=pool, stage=stage,
            complete=False, updated_unix=time.time(), failed_families=failed,
            final_scheduler_audit_required=True, **extra))

    with (pool_root / 'owner.lock').open('a+') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            while True:
                for slot, entry in list(children.items()):
                    code = entry['process'].poll()
                    if code is not None:
                        entry['receipt'].update(returncode=code, ended_epoch=time.time())
                        del children[slot]
                        if code:
                            failed[entry['receipt']['family']] = dict(worker_id=entry['receipt']['worker_id'],
                                returncode=code, failed_epoch=time.time())
                pending = {f: pending_blocks(p['output'], p['schedule']) for f, p in selected.items()}
                active = {f: sum(e['receipt']['family'] == f for e in children.values()) for f in selected}
                if time.monotonic() >= deadline:
                    raise TimeoutError('Approved allocation is ending; preserve every completed record and attempt')
                for slot in range(spec['workers']):
                    if slot in children:
                        continue
                    initial = ('sft' if slot < 4 else 'qwen') if pool == 'gpu' and generations[slot] == 0 else None
                    family = choose_family(pending, active, initial, failed)
                    if family is None:
                        continue
                    worker_id = f'{pool}-{job}-{slot}-{generations[slot]}'
                    generations[slot] += 1
                    config = journal / (worker_id + '.json')
                    write_json(config, dict(root=str(root), global_plan_sha256=digest(plan),
                        pool=pool, job_id=job, family=family, worker_id=worker_id))
                    command = worker_command(SOURCE, pool, job, config)
                    receipt = dict(slot=slot, family=family, worker_id=worker_id,
                                   started_epoch=time.time(), command=command)
                    history.append(receipt)
                    children[slot] = dict(process=subprocess.Popen(command, start_new_session=True), receipt=receipt)
                    active[family] += 1
                write_json(journal / 'workers.json', history)
                counts = {f: len(list((Path(p['output']) / 'records').glob('*.json'))) for f, p in selected.items()}
                write_status('running_with_failure' if failed else 'running', active_workers=active,
                    recorded_episodes=counts, remaining_task_blocks=pending,
                    remaining_seconds=max(0, deadline - time.monotonic()))
                if not children:
                    if failed:
                        raise RuntimeError('A family failed; healthy independent work was allowed to finish')
                    if any(pending.values()):
                        raise RuntimeError('Unfinished task blocks have no owning worker')
                    break
                if pool == 'gpu':
                    assigned = []
                    for entry in children.values():
                        path = root / 'workers' / pool / job / entry['receipt']['worker_id'] / 'device-assignment.json'
                        if path.exists():
                            assigned.append(read(path)['gpu_uuid'])
                    if len(set(assigned)) != len(assigned):
                        raise ValueError('Concurrent workers must own distinct physical GPUs')
                time.sleep(10)
            counts = {family: verify_records(fp) for family, fp in selected.items()}
            expected = {family: sum(len(item['modes']) for item in fp['schedule']) for family, fp in selected.items()}
            if counts != expected:
                raise ValueError('Pool endpoint does not match its complete task/arm cohort')
            write_json(pool_root / 'current.json', dict(job_id=job, pool=pool,
                stage='collection_complete', complete=True, recorded_episodes=counts,
                updated_unix=time.time(), final_scheduler_audit_required=True))
        except BaseException as exc:
            write_status('failed_or_interrupted', error_type=type(exc).__name__)
            raise
        finally:
            stop_workers(children)
            for entry in children.values():
                entry['receipt'].update(returncode=entry['process'].returncode, ended_epoch=time.time(), stopped_by_owner=True)
            write_json(journal / 'workers.json', history)
            result = subprocess.run(['sacct', '-X', '-n', '-P', '-j', job,
                '--format=JobIDRaw,State,ElapsedRaw,AllocTRES'], text=True, capture_output=True)
            write_json(root / 'accounting' / pool / (job + '.json'), dict(job_id=job, pool=pool,
                captured_unix=time.time(), final_scheduler_audit_required=True, sacct=result.stdout,
                sacct_returncode=result.returncode, approved_attempt_seconds=attempt['time_limit_seconds'],
                previous_actual_seconds=used))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--execute', nargs=2, metavar=('POOL', 'JOB_ID'))
    group.add_argument('--worker', type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(read(args.worker))
    elif args.root is None:
        parser.error('--root is required for --execute')
    else:
        execute(args.root, *args.execute)
