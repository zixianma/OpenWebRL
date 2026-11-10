#!/usr/bin/env python3
"""Own the separately approved CPU-only local Jev cohort and all its attempts."""
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

SOURCE = Path(__file__).resolve().parents[1]
RUNTIME = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime')
sys.path.insert(0, str(SOURCE))
from openwebrl.controlled_jev_state import JevState
from openwebrl.controlled_sft_state import digest
from resume_baseline import validate_source, source_command
from run_controlled_sft_suite import check_approval, scheduler_accounting, process_group_members, stop_owned


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    with temp.open('w') as handle: json.dump(value, handle, indent=2); handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
    temp.replace(path)


def validate_frozen(root):
    root = Path(root).resolve(); plan = read(root/'plan.json')
    if Path(plan['source']).resolve() != SOURCE.resolve() or plan['controller_sha256'] != sha(__file__):
        raise ValueError('Use the exact frozen controller')
    validate_source(SOURCE)
    for path, expected in ((SOURCE/'reference_manifest.json', plan['source_manifest_sha256']),
        (Path(plan['task_file']), plan['task_file_sha256']), (root/'pricing-verification.json', plan['pricing_sha256'])):
        if sha(path) != expected: raise ValueError('Frozen cohort artifact changed: '+str(path))
    if (set(plan['conditions']) != {'L06'} or len(plan['tasks']) != 300
            or len(plan['schedule']) != 303 or len({r['item_id'] for r in plan['schedule']}) != 303):
        raise ValueError('Expected300 Jev primary tasks plus three distinct smoke attempts')
    for category, expected in (('smoke', 3), ('primary', 300)):
        rows = [r for r in plan['schedule'] if r['category'] == category]
        if len(rows) != expected or len({r['task_id'] for r in rows}) != expected:
            raise ValueError('Task schedule changed')
    browser = plan['worker_config']['browser_manifest']
    if sha(browser['binary_path']) != browser['binary_sha256']: raise ValueError('Pinned browser changed')
    return plan


def validate_allocation(info, job, seconds, *, uid=None, now=None):
    uid = os.getuid() if uid is None else uid
    fields = dict(re.findall(r'(\w+)=(\S+)', info))
    tres = dict(x.split('=', 1) for x in fields.get('AllocTRES', '').split(',') if '=' in x)
    if (fields.get('JobId') != str(job) or fields.get('JobState') != 'RUNNING'
            or not fields.get('UserId', '').endswith(f'({uid})') or fields.get('NumNodes') != '1'
            or fields.get('NumCPUs') != '8' or tres.get('mem') not in ('32G', '32768M')
            or any(k.startswith('gres/gpu') and v != '0' for k, v in tres.items())):
        raise ValueError('Require owned0GPU/8CPU/32GiB allocation')
    value = fields['TimeLimit']; days, clock = value.split('-', 1) if '-' in value else ('0', value)
    pieces = [int(x) for x in clock.split(':')]
    if len(pieces) == 2: pieces.insert(0, 0)
    duration = int(days)*86400+pieces[0]*3600+pieces[1]*60+pieces[2]
    if duration > seconds: raise ValueError('Scheduler limit exceeds remaining approved time')
    end = datetime.strptime(fields['EndTime'], '%Y-%m-%dT%H:%M:%S').timestamp()
    left = min(duration, int(end-(time.time() if now is None else now)))-180
    if left < 300: raise ValueError('Not enough allocation time after cleanup margin')
    return left


def environment():
    from dotenv import dotenv_values
    env = dict(os.environ)
    values = {**dotenv_values('/gpfs/projects/krishna/zixianma/OpenWebRL/.env'), **env}
    keys = dict(TYPESAFE_API_KEY=values.get('TYPESAFE_API_KEY') or values.get('JEV_API_KEY'),
        TEXT_MODEL_API_KEY=values.get('TEXT_MODEL_API_KEY') or values.get('OPENAI_API_KEY'),
        JUDGE_API_KEY=values.get('JUDGE_API_KEY') or values.get('OPENAI_API_KEY'))
    if not all(keys.values()): raise ValueError('Jev, typing helper, and judge credentials required')
    for key, value in values.items():
        if value and key.startswith('WANDB_'): env[key] = value
    env.update(keys)
    env.update(TYPESAFE_MODEL='jev-1.13.0', TEXT_MODEL='gpt-4.1-mini-2025-04-14', TEXT_MODEL_REASONING='none',
        TEXT_MODEL_BASE_URL='https://api.openai.com/v1', WANDB_PROJECT='openwebrl-evals',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        PYTHONPATH=str(SOURCE)+':'+str(RUNTIME/'src/Megatron-LM'))
    return env


def allocation_groups(job):
    """Every user-owned child group in this exclusive CPU job, including orphans."""
    groups = set()
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit(): continue
        try:
            if entry.stat().st_uid != os.getuid() or f'/job_{job}/' not in (entry/'cgroup').read_text(): continue
            parts = (entry/'stat').read_text().rsplit(')', 1)[1].split()
            pgid = int(parts[2])
            if pgid != os.getpgrp() and parts[0] != 'Z': groups.add(pgid)
        except (FileNotFoundError, ProcessLookupError): continue
    return groups


def stop_browser_descendants(job):
    # Playwright can create detached Chromium groups. Same-UID/current-job
    # validation is repeated immediately before any signal; peer allocations
    # and the controller's own group are never touched.
    groups = allocation_groups(job)
    for sig, delay in ((signal.SIGTERM, 10), (signal.SIGKILL, 5)):
        for pgid in groups:
            if process_group_members(pgid, job):
                try: os.killpg(pgid, sig)
                except ProcessLookupError: pass
        end = time.monotonic()+delay
        while time.monotonic() < end and any(process_group_members(g, job) for g in groups): time.sleep(.1)
    receipts = [dict(pgid=g, remaining=process_group_members(g, job)) for g in sorted(groups)]
    return receipts


def teardown_owned(children, job, wandb_run, exit_code):
    groups = []
    wandb_status = dict(finished=wandb_run is None, error_type=None)
    # W&B owns its own process group. Finish/sync it before sweeping orphaned
    # groups in this allocation; a browser sweep must not kill active logging.
    try:
        groups = stop_owned(children, job)
        if wandb_run is not None:
            try:
                wandb_run.finish(exit_code=exit_code)
                wandb_status['finished'] = True
            except Exception as exc:
                wandb_status['error_type'] = type(exc).__name__
    finally:
        # Also drain detached browsers when a shutdown signal interrupts W&B.
        browsers = stop_browser_descendants(job)
    clean = (all(not r['remaining'] for r in browsers) and all(r['process_gone'] for r in groups)
             and all(p.poll() is not None for p in children))
    return dict(verified=clean, direct_groups=groups, browser_groups=browsers, wandb=wandb_status)


def execute(root, job):
    root = Path(root).resolve(); plan = validate_frozen(root); approval = read(root/'approval.json')
    if os.getenv('SLURM_JOB_ID') != str(job) or f'/job_{job}/' not in Path('/proc/self/cgroup').read_text():
        raise ValueError('Must execute inside the registered approved allocation')
    used, old = scheduler_accounting(approval, job)
    seconds = check_approval(plan, approval, job, used)
    left = validate_allocation(subprocess.check_output(['scontrol', 'show', 'job', str(job), '-o'], text=True), job, seconds)
    deadline = time.monotonic()+left
    env = environment()
    journal = root/'scheduler-attempts'/str(job); journal.mkdir(parents=True, exist_ok=True)
    write(journal/'prior-accounting.json', dict(prior_seconds=used, attempts=old))
    children, logs, state, wandb_run, completed = [], [], None, None, False
    def signal_exit(signum, frame): raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM, signal_exit); signal.signal(signal.SIGINT, signal_exit)
    with (root/'controller.lock').open('a') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            write(journal/'allocation-validated.json', dict(job_id=str(job), plan_sha256=digest(plan),
                resources=plan['resources'], limits=plan['limits'], prior_actual_seconds=used,
                current_attempt_limit=seconds, work_deadline_unix=time.time()+left))
            state = JevState(root)
            initial = state.snapshot()
            if initial['halt'] or initial['queue'].get('active') or initial['queue'].get('interrupted'):
                raise RuntimeError('Preserved failed/interrupted work requires diagnosis before recovery')
            import wandb
            wandb_run = wandb.init(project='openwebrl-evals', id=plan['experiment_id'], resume='allow',
                config=dict(protocol=plan['protocol_name'], actor='Jev alone', plan_sha256=digest(plan),
                            resources=plan['resources'], limits=plan['limits']), dir=str(journal),
                settings=wandb.Settings(init_timeout=90, api_key=env.get('WANDB_API_KEY')))
            for phase in ('smoke', 'primary'):
                if phase == 'smoke' and state.snapshot()['phase'] == 'primary': continue
                if phase == 'primary' and state.snapshot()['phase'] == 'smoke': state.enable_primary()
                workers = []
                for i in range(plan['resources']['collectors']):
                    log = (journal/f'{phase}-worker-{i}.log').open('a'); logs.append(log)
                    argv = [str(RUNTIME/'venv/bin/python'), '-m', 'openwebrl.controlled_jev_eval', '--root', str(root),
                            '--worker-id', f'{job}-{phase}-{i}']
                    process = subprocess.Popen(argv, cwd=SOURCE, env=env,
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    children.append(process); workers.append(process)
                while any(p.poll() is None for p in workers):
                    snapshot = state.snapshot()
                    write(root/'current.json', dict(stage=phase, job_id=job, updated_unix=time.time(), **snapshot))
                    if snapshot['halt'] or any(p.poll() not in (None, 0) for p in workers):
                        raise RuntimeError('Jev worker/provider failed; preserve and diagnose')
                    if time.monotonic() > deadline: raise TimeoutError('Approved allocation time boundary')
                    wandb_run.log({'finished': snapshot['counts_by_condition'].get('L06', 0)})
                    time.sleep(5)
                if any(p.returncode for p in workers): raise RuntimeError('Jev worker failed')
            snapshot = state.snapshot()
            if snapshot['queue'] != {'finished': 303} or snapshot['counters']['active_browsers'] or snapshot['unsettled_reservations']:
                raise ValueError('Unresolved queue/budget evidence remains')
            with state.connect() as db:
                for row in db.execute("SELECT result_json FROM attempts WHERE status='finished'"):
                    result = json.loads(row['result_json'])
                    if sha(result['result_path']) != result['result_sha256']: raise ValueError('Saved result changed')
            completed = True
            write(root/'collection-complete.json', dict(collection_complete=True, verified_complete=False,
                primary_episodes=300, smoke_episodes=3, job_id=job, final_independent_audit_required=True))
        finally:
            try:
                teardown = teardown_owned(children, job, wandb_run, 0 if completed else 1)
                write(journal/'owned-process-teardown.json', teardown)
                if state is not None: state.mark_interrupted_after_teardown(teardown_verified=teardown['verified'], reason='Jev controller closeout')
            finally:
                for log in logs: log.close()
                raw = subprocess.run(['sacct', '-X', '-n', '-P', '-j', str(job), '--format=JobIDRaw,State,ElapsedRaw,AllocTRES'], capture_output=True, text=True)
                write(journal/'scheduler-accounting.json', dict(sacct=raw.stdout, prior_actual_seconds=used,
                    current_attempt_limit=seconds, final_scheduler_accounting_required=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dry-run', action='store_true'); mode.add_argument('--execute')
    args = parser.parse_args()
    if args.dry_run:
        plan = validate_frozen(args.root)
        print(json.dumps(dict(frozen_preflight_passed=True, launch_authorized=read(args.root/'approval.json')['approved'],
                             resources=plan['resources'], limits=plan['limits']), indent=2))
    else: execute(args.root, args.execute)


if __name__ == '__main__': main()
