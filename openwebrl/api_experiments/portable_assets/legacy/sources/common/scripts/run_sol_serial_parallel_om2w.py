#!/usr/bin/env python3
"""Run an exactly approved paired100 OM2W pilot; never submit or retry jobs."""
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
import shlex
import subprocess
import sys
import time

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
from openwebrl.sol_serial_parallel_eval import LIMITS, RESOURCES, validate_plan
from openwebrl.luna_qwen_metrics import digest, write_json
from openwebrl.luna_qwen_full_eval import verify_actor_files
from openwebrl.sol_serial_parallel_lineage import verify_existing_records, accepted_plan_hashes
from openwebrl.luna_qwen_eval import verified_record
from runtime_ports import lease_ports


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(root, *, approved=False):
    root = Path(root).resolve()
    plan = read(root / 'plan.json')
    if Path(plan['source']).resolve() != SOURCE:
        raise ValueError('Run only the registered frozen source tree')
    validate_plan(plan, dict(output=plan['output'], plan_sha256=digest(plan)))
    required = {'openwebrl/sol_serial_parallel_eval.py', 'openwebrl/sol_serial_parallel_policy.py',
                'scripts/run_sol_serial_parallel_om2w.py', 'openwebrl/generate_browser.py',
                'openwebrl/outcome_validity.py', 'openwebrl/responses_actor_history.py'}
    if not required <= set(plan.get('source_hashes', {})):
        raise ValueError('Frozen source manifest missing required implementation hashes')
    for name, expected in plan['source_hashes'].items():
        path = (SOURCE / name).resolve()
        if not path.is_relative_to(SOURCE) or sha(path) != expected:
            raise ValueError('Frozen source changed: ' + name)
    if sha(plan['task_file']) != plan['task_file_sha256']:
        raise ValueError('Frozen task file changed')
    if Path(plan['output']).resolve() != root / 'results' or Path(plan['budget_root']).resolve() != root / 'budget':
        raise ValueError('Output or shared budget routing changed')
    if Path(plan['judge_budget_root']).resolve() != root / 'budget' / 'judge-budget':
        raise ValueError('Shared judge budget routing changed')
    verify_actor_files(plan)
    verify_existing_records(plan, digest(plan))
    approval = read(root / 'approval.json') if (root / 'approval.json').exists() else {}
    if approved and (approval.get('approved') is not True or approval.get('plan_sha256') != digest(plan)
                     or approval.get('resources') != RESOURCES or approval.get('limits') != LIMITS):
        raise ValueError('Exact resource/API approval bound to this frozen plan is required')
    return plan, approval


def prior_accounting(approval, job):
    total, receipts = 0, []
    attempts = approval.get('attempts', [])
    if len({a['job_id'] for a in attempts}) != len(attempts):
        raise ValueError('Duplicate allocation in all-attempt ledger')
    current = [a for a in attempts if a['job_id'] == job]
    if len(current) != 1:
        raise ValueError('Current allocation absent from bound attempt ledger')
    for attempt in attempts:
        if attempt['job_id'] == job:
            continue
        output = subprocess.check_output(['sacct', '-X', '-n', '-P', '-j', attempt['job_id'],
            '--format=JobIDRaw,State,ElapsedRaw,AllocTRES'], text=True)
        rows = [row.split('|') for row in output.splitlines() if row.split('|')[0] == attempt['job_id']]
        if len(rows) != 1 or rows[0][1].split()[0] not in {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT',
                'OUT_OF_MEMORY', 'NODE_FAIL', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}:
            raise ValueError('Prior allocation requires final scheduler accounting')
        elapsed = int(rows[0][2])
        if elapsed < 0:
            raise ValueError('Invalid elapsed accounting')
        total += elapsed
        receipts.append(dict(job_id=attempt['job_id'], elapsed_seconds=elapsed, sacct=output))
    limit = current[0]['time_limit_seconds']
    if type(limit) is not int or limit <= 0 or total + limit > RESOURCES['total_seconds']:
        raise ValueError('All-attempt12h compute cap exceeded; no budget reset')
    return current[0], total, receipts


def validate_allocation(info, job, approved_seconds):
    fields = dict(re.findall(r'(\w+)=(\S+)', info))
    if (fields.get('JobId') != job or fields.get('JobState') != 'RUNNING'
            or not fields.get('UserId', '').endswith(f'({os.getuid()})')
            or fields.get('NumNodes') != '1' or fields.get('NumCPUs') != '16'):
        raise ValueError('Require exactly the owned running16CPU single-node allocation')
    tres = dict(p.split('=', 1) for p in fields.get('AllocTRES', '').split(',') if '=' in p)
    if any(k.startswith('gres/gpu') and float(v) != 0 for k, v in tres.items()):
        raise ValueError('This pilot requests no GPUs')
    match = re.fullmatch(r'([\d.]+)([KMGT])', tres.get('mem', ''))
    memory = float(match[1]) * {'K': 2**-20, 'M': 2**-10, 'G': 1, 'T': 1024}[match[2]] if match else 0
    if memory != 64:
        raise ValueError('Require exactly64GiB memory')
    from run_luna_qwen_full300 import slurm_seconds
    if slurm_seconds(fields['TimeLimit']) > approved_seconds:
        raise ValueError('Scheduler time exceeds approved attempt limit')
    remaining = min(approved_seconds, int(datetime.strptime(fields['EndTime'], '%Y-%m-%dT%H:%M:%S').timestamp() - time.time())) - 180
    if remaining < 300:
        raise ValueError('Too little approved runtime after shutdown margin')
    return remaining


def runtime_environment(root):
    from dotenv import dotenv_values
    env = os.environ.copy()
    for name, value in dotenv_values('/gpfs/projects/krishna/zixianma/OpenWebRL/.env').items():
        if value and name.startswith(('OPENAI_', 'JUDGE_', 'WANDB_')):
            env.setdefault(name, value)
    if not env.get('OPENAI_API_KEY'):
        raise ValueError('Missing explicitly authorized API credential')
    cache = Path(root) / 'runtime-cache'
    cache.mkdir(parents=True, exist_ok=True)
    env.update(CUDA_VISIBLE_DEVICES='', WANDB_PROJECT='openwebrl-evals', PYTHONPATH=str(SOURCE),
        PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        HF_HOME=str(cache / 'huggingface'), HF_HUB_CACHE=str(cache / 'huggingface' / 'hub'),
        HUGGINGFACE_HUB_CACHE=str(cache / 'huggingface' / 'hub'),
        TRANSFORMERS_CACHE=str(cache / 'huggingface' / 'transformers'),
        HF_DATASETS_CACHE=str(cache / 'huggingface' / 'datasets'), TORCH_HOME=str(cache / 'torch'),
        NUMBA_CACHE_DIR=str(cache / 'numba'), TORCHINDUCTOR_CACHE_DIR=str(cache / 'inductor'),
        TRITON_CACHE_DIR=str(cache / 'triton'), WANDB_DIR=str(cache / 'wandb'),
        PIP_CACHE_DIR=str(cache / 'pip'), UV_CACHE_DIR=str(cache / 'uv'),
        PYTHONPYCACHEPREFIX=str(cache / 'pycache'), XDG_CACHE_HOME=str(cache / 'cache'),
        XDG_CONFIG_HOME=str(cache / 'config'), XDG_DATA_HOME=str(cache / 'data'),
        WANDB_CONFIG_DIR=str(cache / 'wandb-config'), WANDB_CACHE_DIR=str(cache / 'wandb-cache'),
        WANDB_DATA_DIR=str(cache / 'wandb-data'), MPLCONFIGDIR=str(cache / 'matplotlib'),
        TMPDIR=str(cache / 'tmp'), JUDGE_API_BASE='https://api.openai.com/v1', JUDGE_API_MODE='served')
    for name in ('HF_HUB_CACHE', 'HUGGINGFACE_HUB_CACHE', 'TRANSFORMERS_CACHE', 'HF_DATASETS_CACHE',
                 'TORCH_HOME', 'NUMBA_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR',
                 'WANDB_DIR', 'PIP_CACHE_DIR', 'UV_CACHE_DIR', 'PYTHONPYCACHEPREFIX', 'HF_HOME', 'XDG_CACHE_HOME', 'XDG_CONFIG_HOME', 'XDG_DATA_HOME', 'WANDB_CONFIG_DIR',
                 'WANDB_CACHE_DIR', 'WANDB_DATA_DIR', 'MPLCONFIGDIR', 'TMPDIR'):
        Path(env[name]).mkdir(parents=True, exist_ok=True)
    return env


def source_command(argv):
    # Runtime library paths come from the pinned repository environment; all
    # inherited cache destinations were already redirected to the approved root.
    shell = 'set -euo pipefail\nsource ' + shlex.quote(str(SOURCE / 'scripts/h200_env.sh'))
    shell += '\nexport OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=\nexec ' + shlex.join(argv)
    return ['bash', '-c', shell]


def stop(child):
    if child is None or child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
        child.wait(timeout=45)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()
    except ProcessLookupError:
        pass


def verify_records(plan):
    verify_existing_records(plan, digest(plan))
    rows = [verified_record(p, digest(plan['protocol'])) for p in (Path(plan['output']) / 'records').glob('*.json')]
    expected = {(t, m) for t in plan['task_ids'] for m in ('serial_high', 'parallel3_low')}
    if len(rows) != 200 or {(r['task_id'], r['mode']) for r in rows} != expected:
        raise ValueError('Require all200 frozen task/arm dispositions')
    return rows


def worker(config):
    from run_luna_qwen_full300 import require_cgroup
    job, root = config['job_id'], Path(config['root'])
    require_cgroup(job)
    plan, approval = validate(root, approved=True)
    attempt, _, _ = prior_accounting(approval, job)
    validate_allocation(subprocess.check_output(['scontrol', 'show', 'job', '-o', job], text=True),
                        job, attempt['time_limit_seconds'])
    lease, port = lease_ports(job)
    child = None
    try:
        worker_root = root / 'workers' / config['worker_id']
        worker_root.mkdir(parents=True, exist_ok=True)
        cfg = dict(output=plan['output'], plan=str(root / 'plan.json'), plan_sha256=digest(plan),
                   worker_id=config['worker_id'], browser_port=port + 10, job_id=job)
        write_json(worker_root / 'worker-config.json', cfg)
        command = source_command([plan['python'], '-m', 'openwebrl.sol_serial_parallel_eval',
                   '--config', str(worker_root / 'worker-config.json')])
        with (worker_root / 'collection.log').open('a') as out:
            child = subprocess.Popen(command, cwd=SOURCE, env=runtime_environment(root), stdout=out,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            code = child.wait()
        sentinel = Path(plan['output']) / 'workers' / config['worker_id'] / 'status.json'
        if code or not sentinel.exists() or not read(sentinel).get('complete'):
            raise RuntimeError('Worker failed or lacked collection sentinel; diagnose before retry')
    finally:
        stop(child)
        lease.close()
        if child is not None:
            write_json(root / 'workers' / config['worker_id'] / 'exit.json',
                       dict(returncode=child.returncode, controller_owned_process_group=True,
                            independent_browser_teardown_audit_required=True, ended_epoch=time.time()))


def execute(root, job):
    from run_luna_qwen_full300 import require_cgroup
    require_cgroup(job)
    root = Path(root).resolve()
    plan, approval = validate(root, approved=True)
    attempt, used, receipts = prior_accounting(approval, job)
    remaining = validate_allocation(subprocess.check_output(['scontrol', 'show', 'job', '-o', job], text=True),
                                    job, attempt['time_limit_seconds'])
    journal = root / 'attempts' / job
    journal.mkdir(parents=True, exist_ok=True)
    children, history = [], []
    signal.signal(signal.SIGTERM, lambda *args: (_ for _ in ()).throw(SystemExit(143)))
    deadline = time.monotonic() + remaining
    with (root / 'owner.lock').open('a+') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_json(journal / 'prior-accounting.json', dict(used_seconds=used, attempts=receipts))
        try:
            for slot in range(6):
                worker_id = f'{job}-{slot}'
                config = journal / f'worker-{slot}.json'
                write_json(config, dict(root=str(root), job_id=job, worker_id=worker_id))
                command = ['srun', '--jobid=' + job, '--nodes=1', '--ntasks=1', '--cpus-per-task=2',
                    '--mem=10G', '--gres=none', '--exact', '--cpu-bind=cores', plan['python'],
                    str(SOURCE / 'scripts/run_sol_serial_parallel_om2w.py'), '--worker', str(config)]
                children.append(subprocess.Popen(command, env=runtime_environment(root), start_new_session=True))
                history.append(dict(worker_id=worker_id, command=command, started_epoch=time.time()))
            while any(child.poll() is None for child in children):
                if any(child.poll() not in (None, 0) for child in children):
                    raise RuntimeError('Worker failure; no automatic unchanged retry')
                if time.monotonic() >= deadline:
                    raise TimeoutError('Remaining approved allocation ended')
                write_json(root / 'current.json', dict(stage='running', job_id=job, complete=False,
                    plan_sha256=digest(plan), updated_unix=time.time(),
                    recorded_episodes=len(list((Path(plan['output']) / 'records').glob('*.json')))))
                time.sleep(10)
            if any(child.returncode != 0 for child in children):
                raise RuntimeError('Worker failed at shutdown')
            rows = verify_records(plan)
            write_json(root / 'current.json', dict(stage='collection_complete_pending_independent_audit',
                job_id=job, complete=False, recorded_episodes=len(rows), updated_unix=time.time()))
        except BaseException as exc:
            write_json(root / 'current.json', dict(stage='failed_or_interrupted', complete=False, job_id=job,
                error_type=type(exc).__name__, error=str(exc), updated_unix=time.time()))
            raise
        finally:
            for child in children:
                stop(child)
            for entry, child in zip(history, children):
                entry.update(returncode=child.returncode, ended_epoch=time.time())
            write_json(journal / 'workers.json', history)
            output = subprocess.run(['sacct', '-X', '-n', '-P', '-j', job,
                '--format=JobIDRaw,State,ElapsedRaw,AllocTRES'], text=True, capture_output=True)
            write_json(journal / 'scheduler-accounting.json', dict(sacct=output.stdout,
                stderr=output.stderr, returncode=output.returncode, previous_elapsed_seconds=used,
                approved_attempt_seconds=attempt['time_limit_seconds'], final_scheduler_audit_required=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--dry-run', action='store_true')
    group.add_argument('--execute', metavar='JOB_ID')
    group.add_argument('--worker', type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(read(args.worker))
    elif args.execute:
        execute(args.root, args.execute)
    else:
        plan, approval = validate(args.root)
        print(json.dumps(dict(valid=True, approved=approval.get('approved') is True,
                              tasks=len(plan['task_ids']), episodes=200, resources=RESOURCES, limits=LIMITS)))


if __name__ == '__main__':
    main()
