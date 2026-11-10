#!/usr/bin/env python3
"""Wake the owning agent for the approved pilot; never mutate GPU jobs."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

ROOT = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/luna-qwen-inference-20261004')
SCHEDULER_STATES = {
    'BOOT_FAIL', 'CANCELLED', 'COMPLETED', 'CONFIGURING', 'COMPLETING', 'DEADLINE',
    'FAILED', 'NODE_FAIL', 'OUT_OF_MEMORY', 'PENDING', 'PREEMPTED', 'RUNNING',
    'REQUEUED', 'REQUEUE_FED', 'REQUEUE_HOLD', 'RESIZING', 'RESV_DEL_HOLD',
    'REVOKED', 'SIGNALING', 'SPECIAL_EXIT', 'STAGE_OUT', 'STOPPED', 'SUSPENDED', 'TIMEOUT',
}
TERMINAL_STATES = {'BOOT_FAIL', 'CANCELLED', 'COMPLETED', 'DEADLINE', 'FAILED',
                   'NODE_FAIL', 'OUT_OF_MEMORY', 'PREEMPTED', 'REVOKED', 'TIMEOUT'}
APPROVAL_BLOCKED_STAGES = {'awaiting_resource_profile_approval',
                           'awaiting_recovery_approval_and_luna_repair'}


def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.partial')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def scheduler(job):
    """Fall back to accounting when a terminal job has aged out of squeue."""
    job = str(job)
    if not re.fullmatch(r'[0-9]+', job):
        raise ValueError('Invalid registered Slurm job ID')
    commands = [
        ['squeue', '-h', '-j', job, '-o', '%i|%T|%M|%R'],
        ['sacct', '-X', '-n', '-P', '-j', job, '--format=JobIDRaw,State,Elapsed,ExitCode'],
    ]
    failures = []
    for command in commands:
        try:
            result = subprocess.run(command, text=True, capture_output=True, timeout=20)
        except (OSError, subprocess.TimeoutExpired) as exc:
            failures.append(f'{command[0]}: {type(exc).__name__}: {str(exc)[:240]}')
            continue
        if result.returncode:
            failures.append(f'{command[0]}: exit {result.returncode}: {result.stderr.strip()[:240]}')
            continue
        output = result.stdout.strip()
        rows = [line.strip().split('|') for line in output.splitlines()
                if line.split('|', 1)[0].strip() == job]
        if len(rows) != 1:
            failures.append(f'{command[0]}: expected one row for registered job, found {len(rows)}')
            continue
        row = rows[0]
        if len(row) == 5 and row[-1] == '':
            row.pop()  # Some accounting versions retain a trailing delimiter.
        if len(row) != 4 or not row[1].strip() or not row[2].strip():
            failures.append(f'{command[0]}: malformed registered-job row')
            continue
        state = row[1].split()[0].rstrip('+')
        if state not in SCHEDULER_STATES:
            failures.append(f'{command[0]}: unrecognized job state {state!r}')
            continue
        return dict(job_id=job, state=state, raw=output)
    raise RuntimeError(f'Cannot establish scheduler state for registered job {job}; ' + '; '.join(failures))


def snapshot():
    current = read(ROOT / 'current.json')
    approval = read(ROOT / 'approval.json')
    job = current.get('job_id') or approval['attempts'][-1]['job_id']
    scheduled = scheduler(job)
    job, live, state = scheduled['job_id'], scheduled['raw'], scheduled['state']
    addition = ROOT / 'sft-extension'
    extra_requested = (addition / 'request.json').exists()
    roots = (ROOT, addition) if extra_requested else (ROOT,)
    records = [read(p) for root in roots for p in (root / 'pilot/records').glob('*.json')]
    modes = {m:dict(episodes=sum(r.get('mode') == m for r in records),
                   successes=sum(r.get('mode') == m and r.get('valid') and r.get('reward') == 1 for r in records))
             for m in (('qwen','qwen_luna5','qwen_luna10','luna','sft_luna5') if extra_requested
                       else ('qwen','qwen_luna5','qwen_luna10','luna'))}
    activity = [p.stat().st_mtime for root in roots for p in (root / 'pilot').glob('attempts/*/*/requests/*.json')]
    halted = {r:read(ROOT / f'pilot/{r}-budget/halt.json') for r in ('luna','judge')}
    ledgers = {r:read(ROOT / f'pilot/{r}-budget/ledger.json') for r in ('luna','judge')}
    repair = read(ROOT / 'repairs/context-boundary-20261004/current.json')
    recovery_approval = ROOT / 'recovery-tail-20261005/approval.json'
    def worker_states(root):
        return {p.parent.name:dict(read(p), complete=read(p.with_name('status.json')).get('complete', False))
                for p in (root / 'pilot/workers').glob('*/heartbeat.json')}
    return dict(epoch=time.time(), pid=os.getpid(), job_id=job, scheduler=live, state=state,
        controller=current, modes=modes, completed=len(records), expected_episodes=50 if extra_requested else 40,
        sft_extension=read(addition / 'current.json') if extra_requested else {},
        context_repair=repair,
        recovery_approval_sha256=(hashlib.sha256(recovery_approval.read_bytes()).hexdigest()
                                  if recovery_approval.exists() else None),
        api_ledgers=ledgers, api_halts=halted,
        last_request_age_seconds=time.time()-max(activity) if activity else None,
        browser_workers=worker_states(ROOT),
        sft_browser_workers=worker_states(addition) if extra_requested else {},
        source='Active agent continuation via codex queue; this watcher performs no GPU mutations')


def dispatch_state(latest):
    return [latest['job_id'], latest['state'], bool(any(latest['api_halts'].values())),
            latest['sft_extension'].get('stage'), latest['controller'].get('stage'),
            latest['context_repair'].get('stage'), latest.get('recovery_approval_sha256')]


def review_interval(latest):
    # No worker is running in these explicitly approval-blocked terminal states.
    # Observe every minute and dispatch immediately on changes, but avoid asking
    # an agent to repeat an unchanged blocked review every fifteen minutes.
    blocked = (latest['state'] in TERMINAL_STATES and
               latest['controller'].get('stage') in APPROVAL_BLOCKED_STAGES)
    return 3600 if blocked else 900


def continuation_due(latest, state, review, now):
    return (dispatch_state(latest) != state.get('last_dispatched_state') or
            now - max(review.get('epoch', 0), state.get('queued_epoch', 0)) >= review_interval(latest))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--acknowledge', action='store_true')
    args = parser.parse_args()
    directory = ROOT / 'supervisor'
    directory.mkdir(parents=True, exist_ok=True)
    if args.acknowledge:
        latest = read(directory / 'latest.json')
        write(directory / 'review.json', dict(epoch=time.time(), job_id=latest.get('job_id'), state=latest.get('state')))
        print('Review acknowledged; continue diagnosis and verification before ending the turn.')
        return
    with (directory / 'owner.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        registry = read(directory / 'registry.json')
        while not read(directory / 'stop.json').get('verified_complete'):
            try:
                latest = snapshot()
                write(directory / 'latest.json', latest)
                state = read(directory / 'dispatch.json')
                review = read(directory / 'review.json')
                pending = state.get('queued_epoch', 0) > review.get('epoch', 0)
                change = dispatch_state(latest)
                due = continuation_due(latest, state, review, time.time())
                if due and not pending:
                    message = (
                        '[Automatic continuation of the approved Qwen/Luna pilot] '
                        f'Read {directory}/latest.json and {ROOT}/approval.json. '
                        'Run python3 scripts/watch_luna_qwen_inference.py --acknowledge. '
                        'Inspect current job, worker logs, usage receipts, browser progress and W&B. '
                        'Diagnose and repair failures; resume only within the original 4 H200 x90-minute total '
                        'and $15 Luna/$5 judge caps, preserving every attempt and updating current.json. '
                        'Include the requested official SFT N=5 arm: verify all50 episodes, final accounting and three plots; publish only aggregate '
                        'results/plots to arm. This reminder adds no budget. Routine summaries hourly; '
                        'alert sooner for failures, real blockers or verified completion.'
                    )
                    if len(message.encode()) > 1000:
                        raise ValueError('Continuation message too long')
                    result = subprocess.run(['codex', 'queue', '--thread', registry['thread_id'], '--message', message],
                        text=True, capture_output=True, timeout=45)
                    if result.returncode:
                        raise RuntimeError('Continuation failed: ' + result.stderr[:600])
                    write(directory / 'dispatch.json', dict(queued_epoch=time.time(),
                        last_dispatched_state=change, receipt=result.stdout.strip()))
                    print(json.dumps(dict(queued=True, receipt=result.stdout.strip())), flush=True)
            except Exception as exc:
                write(directory / 'error.json', dict(epoch=time.time(), error_type=type(exc).__name__, error=str(exc)))
                print(json.dumps(dict(error=type(exc).__name__, detail=str(exc))), flush=True)
            time.sleep(60)
        write(directory / 'finished.json', dict(epoch=time.time(), verified_complete=True))


if __name__ == '__main__':
    main()
