#!/usr/bin/env python3
"""Own each separately approved CPU allocation and await every actor worker."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
import run_luna_qwen_full300 as base
from openwebrl.luna_qwen_metrics import digest, write_json
from openwebrl.luna_qwen_full_eval import verify_actor_files, verify_existing_records
from openwebrl.reasoning_actor_eval import validate_plan
from openwebrl.reasoning_actor_policy import MODELS
from resume_baseline import validate_source, source_command
from runtime_ports import lease_ports

RESOURCES = dict(gpus=0, cpus=16, memory_gib=32, total_seconds=21600)
ACTOR_LIMITS = {'luna_high': dict(max_usd=15., max_calls=9900),
                'sol61_high': dict(max_usd=100., max_calls=9900)}
JUDGE_LIMITS = dict(max_usd=5., max_calls=2640)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_approval(approval, proposal):
    if (approval.get('approved') is not True or approval.get('plan_sha256') != digest(proposal)
            or approval.get('pools') != {m: RESOURCES for m in MODELS}
            or approval.get('actor_limits') != ACTOR_LIMITS
            or approval.get('judge_limits') != JUDGE_LIMITS):
        raise ValueError('Exact approved protocol and separate resource/API totals required')


def check_attempt(approval, mode, job, used):
    attempts = approval.get('attempts', [])
    if mode not in MODELS or type(used) is not int or used < 0:
        raise ValueError('Invalid arm or elapsed accounting')
    if len({a['job_id'] for a in attempts}) != len(attempts):
        raise ValueError('Duplicate allocation identity')
    current = [a for a in attempts if a['pool'] == mode and a['job_id'] == job]
    if len(current) != 1:
        raise ValueError('Current allocation not registered for this arm')
    limit = current[0]['time_limit_seconds']
    if type(limit) is not int or limit <= 0 or used + limit > RESOURCES['total_seconds']:
        raise ValueError('Retry exceeds this arm six-hour total; no balance transfers')
    return current[0]


def validate(root):
    root = Path(root).resolve()
    proposal, approval, execution = [read(root / name) for name in ('plan.json', 'approval.json', 'execution.json')]
    validate_approval(approval, proposal)
    if (execution['proposal_sha256'] != digest(proposal)
            or approval.get('execution_sha256') != digest(execution)
            or Path(execution['source']).resolve() != SOURCE.resolve()
            or execution['controller_sha256'] != sha(__file__)
            or execution['source_manifest_sha256'] != sha(SOURCE / 'reference_manifest.json')):
        raise ValueError('Execute only the registered frozen implementation')
    validate_source(SOURCE)
    requested = {a['mode']: a for a in proposal['arms']}
    if set(execution['arms']) != set(MODELS):
        raise ValueError('Missing or unexpected actor arm')
    plans = {}
    for mode, ref in execution['arms'].items():
        plan = read(ref['plan'])
        if (digest(plan) != ref['plan_sha256'] or plan['protocol'] != requested[mode]['protocol']
                or plan['task_ids'] != proposal['task_ids']
                or plan['task_file_sha256'] != proposal['task_file_sha256']
                or sha(plan['task_file']) != plan['task_file_sha256']
                or plan['arm_mode'] != mode
                or Path(plan['output']).resolve() != root / mode / 'results'
                or Path(plan['budget_root']).resolve() != root / 'budget' / mode
                or Path(plan['judge_budget_root']).resolve() != root / 'budget' / 'judge-budget'
                or plan['limits'] != dict(actor_usd=ACTOR_LIMITS[mode]['max_usd'],
                    actor_calls=9900, judge_usd=5., judge_calls=2640)):
            raise ValueError('Worker protocol, cohort or budget routing changed')
        validate_plan(plan, dict(output=plan['output'], plan_sha256=ref['plan_sha256']))
        verify_actor_files(plan)
        verify_existing_records(plan, ref['plan_sha256'])
        plans[mode] = plan
    return proposal, approval, execution, plans


def worker_command(job, config):
    return ['srun', '--jobid=' + job, '--nodes=1', '--ntasks=1', '--cpus-per-task=4',
        '--gres=none', '--mem=8G', '--exact', '--cpu-bind=cores', sys.executable,
        str(SOURCE / 'scripts/run_reasoning_actors_full300.py'), '--worker', str(config)]


def worker(config):
    root, mode, job = Path(config['root']), config['mode'], config['job_id']
    base.require_cgroup(job)
    signal.signal(signal.SIGTERM, base.terminate)
    _, approval, execution, plans = validate(root)
    used, _ = base.previous_elapsed(approval, mode, job)
    check_attempt(approval, mode, job, used)
    if config['execution_sha256'] != digest(execution):
        raise ValueError('Worker execution identity changed')
    fp, ref = plans[mode], execution['arms'][mode]
    logs = root / 'workers' / mode / job / config['worker_id']
    logs.mkdir(parents=True, exist_ok=True)
    lease, port = lease_ports(job)
    child = None
    try:
        env = base.worker_environment()
        env['CUDA_VISIBLE_DEVICES'] = ''
        cfg = dict(output=fp['output'], plan=ref['plan'], plan_sha256=ref['plan_sha256'],
            worker_id=config['worker_id'], browser_port=port + 10, job_id=job)
        cfg_path = logs / 'worker-config.json'
        write_json(cfg_path, cfg)
        command = source_command(SOURCE, [str(base.RUNTIME / 'venv/bin/python'),
            '-m', 'openwebrl.reasoning_actor_eval', '--config', str(cfg_path)])
        with (logs / 'collection.log').open('a') as handle:
            child = subprocess.Popen(command, cwd=SOURCE, env=env, stdout=handle,
                stderr=subprocess.STDOUT, start_new_session=True)
            code = child.wait()
        if code or not read(Path(fp['output']) / 'workers' / config['worker_id'] / 'status.json').get('complete'):
            raise RuntimeError('Worker failed or omitted completion sentinel; inspect before retry')
    finally:
        base.stop(child)
        lease.close()


def execute(root, mode, job):
    root = Path(root).resolve()
    base.require_cgroup(job)
    signal.signal(signal.SIGTERM, base.terminate)
    _, approval, execution, plans = validate(root)
    used, earlier = base.previous_elapsed(approval, mode, job)
    attempt = check_attempt(approval, mode, job, used)
    info = subprocess.check_output(['scontrol', 'show', 'job', job, '-o'], text=True)
    allocation = base.validate_allocation(info, 'cpu', job, attempt['time_limit_seconds'])
    base.worker_environment()
    fp = plans[mode]
    pool = root / 'pools' / mode
    journal = pool / 'attempts' / job
    journal.mkdir(parents=True, exist_ok=True)
    write_json(journal / 'previous-accounting.json', dict(used_seconds=used, attempts=earlier))
    children, history = {}, []
    deadline = time.monotonic() + allocation['remaining_seconds']
    def status(stage, **extra):
        row = dict(job_id=job, pool=mode, stage=stage, complete=False,
            updated_unix=time.time(), final_scheduler_audit_required=True)
        row.update(extra)
        write_json(pool / 'current.json', row)
    with (pool / 'owner.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            for slot in range(4):
                worker_id = f'{mode}-{job}-{slot}'
                cfg_path = journal / (worker_id + '.json')
                write_json(cfg_path, dict(root=str(root), mode=mode, job_id=job,
                    worker_id=worker_id, execution_sha256=digest(execution)))
                command = worker_command(job, cfg_path)
                receipt = dict(worker_id=worker_id, started_epoch=time.time(), command=command)
                history.append(receipt)
                children[slot] = dict(process=subprocess.Popen(command, start_new_session=True), receipt=receipt)
            while children:
                failure = False
                for slot, entry in list(children.items()):
                    code = entry['process'].poll()
                    if code is not None:
                        entry['receipt'].update(returncode=code, ended_epoch=time.time())
                        del children[slot]
                        failure = failure or code != 0
                write_json(journal / 'workers.json', history)
                if failure:
                    raise RuntimeError('Worker failure; stop this arm, preserve every attempt, no automatic blind retry')
                if time.monotonic() >= deadline:
                    raise TimeoutError('Remaining approved allocation ended')
                status('running', active_workers=len(children),
                    recorded_episodes=len(list((Path(fp['output']) / 'records').glob('*.json'))))
                if children:
                    time.sleep(10)
            count = base.verify_records(fp)
            if count != 300:
                raise ValueError('Incomplete300-task endpoint')
            status('collection_complete', complete=True, recorded_episodes=count)
        except BaseException as exc:
            status('failed_or_interrupted', error_type=type(exc).__name__)
            raise
        finally:
            base.stop_workers(children)
            for entry in children.values():
                entry['receipt'].update(returncode=entry['process'].returncode,
                    ended_epoch=time.time(), stopped_by_owner=True)
            write_json(journal / 'workers.json', history)
            accounting = subprocess.run(['sacct', '-X', '-n', '-P', '-j', job,
                '--format=JobIDRaw,State,ElapsedRaw,AllocTRES'], text=True, capture_output=True)
            write_json(root / 'accounting' / mode / (job + '.json'), dict(job_id=job,
                captured_epoch=time.time(), sacct=accounting.stdout, previous_actual_seconds=used,
                approved_attempt_seconds=attempt['time_limit_seconds'], final_scheduler_audit_required=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--execute', nargs=2, metavar=('MODE', 'JOB_ID'))
    group.add_argument('--worker', type=Path)
    group.add_argument('--validate', action='store_true')
    args = parser.parse_args()
    if args.worker:
        worker(read(args.worker))
    elif args.validate:
        validate(args.root)
        print('Approved proposal, source, worker protocols, budgets and prior records validated')
    else:
        execute(args.root, *args.execute)
