#!/usr/bin/env python3
"""Observe both approved full-set pools and wake their owning repair agent."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/luna-qwen-full300-20261004')
THREAD = '01a103a0-a243-7bb0-a194-6271293f0850'
MODES = {'qwen': 'qwen', 'qwen_luna5': 'qwen', 'sft_luna5': 'sft',
         'sft_luna10': 'sft', 'luna': 'luna'}
LIMITS = dict(luna_usd=100., luna_calls=39600, judge_usd=25., judge_calls=6600)
PEERS = {'jev': 'current_sft_jev_full300.json', 'kev-27b': 'current_sft_kev27b_full300.json'}
PEER_RESULT_CACHE = {}


def read(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.partial')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def mtime(path):
    try:
        return Path(path).stat().st_mtime
    except OSError:
        return 0


def scheduler(job):
    if job is None:
        return dict(job_id=None, state='UNSUBMITTED', raw='')
    if not str(job).isdecimal():
        raise ValueError('Invalid registered Slurm job ID')
    job = str(job)
    query = subprocess.run(['squeue', '-h', '-j', job, '-o', '%i|%T|%M|%R'],
                           text=True, capture_output=True, timeout=20)
    output = query.stdout.strip() if query.returncode == 0 else ''
    if not output:
        output = subprocess.check_output(['sacct', '-X', '-n', '-P', '-j', job,
            '--format=JobIDRaw,State,Elapsed,ExitCode'], text=True, timeout=20).strip()
    state = next((row.split('|')[1].split()[0] for row in output.splitlines()
                  if row.split('|')[0] == job), 'UNKNOWN')
    return dict(job_id=job, state=state, raw=output)


def worker_snapshot(root, family, plan, worker, job, now):
    output = Path(plan['output'])
    worker_id = str(worker['worker_id'])
    directory = output / 'workers' / worker_id
    heartbeat = read(directory / 'heartbeat.json')
    status = read(directory / 'status.json')
    pool = 'cpu' if family == 'luna' else 'gpu'
    # Historical workers keep their own allocation's logs after a retry.
    worker_job = str(status.get('job_id') or heartbeat.get('job_id') or job)
    if not worker_job.isascii() or not worker_job.isdecimal():
        raise ValueError('Invalid worker allocation identity')
    logs = root / 'workers' / pool / worker_job / worker_id
    requests = []
    if heartbeat.get('task_key'):
        requests = list((output / 'attempts' / heartbeat['task_key']).glob('*/requests/*.json'))
    latest_request = max((mtime(p) for p in requests), default=0)
    epoch = max(heartbeat.get('updated_unix', 0), latest_request, status.get('updated_unix', 0),
                worker.get('started_epoch', 0))
    active = 'returncode' not in worker
    startup_stalled = active and not heartbeat and now - epoch >= 1200
    progress_stalled = active and bool(heartbeat) and not status.get('complete') and now - epoch >= 900
    return dict(worker_id=worker_id, job_id=worker_job, family=family, active=active, heartbeat=heartbeat,
        status=status, last_activity_age_seconds=now - epoch if epoch else None,
        last_request_age_seconds=now - latest_request if latest_request else None,
        stalled=startup_stalled or progress_stalled,
        stall_kind='startup' if startup_stalled else ('progress' if progress_stalled else None),
        device=read(logs / 'device-assignment.json'),
        log_mtimes={name: mtime(logs / (name + '.log')) for name in ('actor', 'collection', 'devices')},
        wandb_run_id=plan.get('wandb_prefix', '') + '-w' + worker_id,
        wandb_log_mtime=mtime(directory / 'wandb/latest-run/logs/debug.log'))


def peer_accounting(directory, current_job):
    """Charge all registered attempts from fresh allocation rows, never pointers."""
    result = dict(job_id=str(current_job) if current_job is not None else None,
        state='UNKNOWN', raw='', consumed_scheduler_seconds=None,
        remaining_approved_seconds=None, approved_total_seconds=None,
        scheduler_attempts=[], accounting_error=None,
        accounting_source='sacct exact JobIDRaw rows for unique approval attempts')
    try:
        approval = read(Path(directory) / 'approval.json')
        total = approval.get('resources', {}).get('total_seconds')
        if approval.get('approved') is not True or type(total) is not int or total <= 0:
            raise ValueError('Missing or invalid peer approval/resource total')
        result['approved_total_seconds'] = total
        attempts = approval.get('attempts')
        if not isinstance(attempts, list) or not attempts:
            raise ValueError('Missing peer approval attempt registry')
        jobs = []
        for attempt in attempts:
            job = str(attempt.get('job_id')) if isinstance(attempt, dict) else ''
            if not job.isascii() or not job.isdecimal():
                raise ValueError('Invalid job ID in peer approval registry')
            if job not in jobs:
                jobs.append(job)
        if result['job_id'] not in jobs:
            raise ValueError('Current peer job is absent from its approval registry')
        output = subprocess.check_output(['sacct', '-X', '-n', '-P', '-j', ','.join(jobs),
            '--format=JobIDRaw,State,ElapsedRaw'], text=True, timeout=20)
        rows = {}
        for line in output.splitlines():
            fields = [part.strip() for part in line.split('|')]
            # Step rows such as JOB.batch must never add allocation time.
            if not fields or fields[0] not in jobs:
                continue
            if fields[0] in rows:
                raise ValueError('Duplicate exact allocation row in peer accounting')
            if len(fields) < 3 or not fields[1] or not fields[2].isascii() or not fields[2].isdecimal():
                raise ValueError('Invalid state or elapsed seconds in peer accounting')
            state = fields[1].split()[0].rstrip('+')
            if state not in {'BOOT_FAIL', 'CANCELLED', 'COMPLETED', 'CONFIGURING',
                    'COMPLETING', 'DEADLINE', 'FAILED', 'NODE_FAIL', 'OUT_OF_MEMORY',
                    'PENDING', 'PREEMPTED', 'REQUEUED', 'REQUEUE_FED', 'REQUEUE_HOLD',
                    'RESIZING', 'REVOKED', 'RUNNING', 'SIGNALING', 'SPECIAL_EXIT',
                    'STAGE_OUT', 'STOPPED', 'SUSPENDED', 'TIMEOUT'}:
                raise ValueError('Unknown allocation state in peer accounting')
            rows[fields[0]] = dict(job_id=fields[0], state=state, seconds=int(fields[2]))
            if fields[0] == result['job_id']:
                result.update(state=state, raw=line)
        if set(rows) != set(jobs):
            raise ValueError('Missing exact allocation rows in peer accounting')
        spent = sum(row['seconds'] for row in rows.values())
        result.update(scheduler_attempts=[rows[job] for job in jobs],
            consumed_scheduler_seconds=spent, remaining_approved_seconds=max(0, total - spent))
    except (OSError, subprocess.SubprocessError, ValueError, TypeError, AttributeError) as exc:
        result['accounting_error'] = f'{type(exc).__name__}: {exc}'
    return result


def peer_owner_review(directory, now):
    ack = read(Path(directory) / 'supervisor-ack.json')
    review = read(Path(directory) / 'agent-review-latest.json')

    def epoch(value):
        return value if type(value) in (int, float) and math.isfinite(value) and value > 0 else None

    ack_epoch = epoch(ack.get('epoch'))
    review_epoch = epoch(review.get('reviewed_unix')) or epoch(review.get('checked_unix'))
    stamps = {name: value for name, value in (
        ('supervisor-ack', ack_epoch), ('agent-review-latest', review_epoch)) if value is not None}
    source = max(stamps, key=stamps.get) if stamps else None
    latest = stamps.get(source)
    return dict(owner_last_review_epoch=latest,
        owner_last_review_age_seconds=max(0, now - latest) if latest is not None else None,
        owner_last_review_source=source, owner_acknowledge_epoch=ack_epoch,
        owner_review_document_epoch=review_epoch)


def peer_snapshot(runtime, mode, pointer_name, now):
    pointer = read(Path(runtime) / pointer_name)
    if not pointer.get('run_directory'):
        return dict(mode=mode, state='UNREGISTERED', job_id=None, episodes=0,
                    valid=0, successes=0, verified_complete=False, verified_failure=False)
    directory = Path(pointer['run_directory'])
    live = peer_accounting(directory, pointer.get('job_id'))
    heartbeat = read(directory / 'heartbeat.json')
    supervisor = read(directory / 'supervisor-latest.json')
    counts = dict(episodes=0, valid=0, successes=0)
    for path in (directory / mode / 'results').glob('*.json'):
        stamp = path.stat().st_mtime_ns
        cached = PEER_RESULT_CACHE.get(str(path))
        if cached is None or cached[0] != stamp:
            row = read(path)
            reward = row.get('reward')
            reward = reward.get('combined', reward.get('judge')) if isinstance(reward, dict) else reward
            cached = (stamp, bool(row.get('valid')), reward == 1)
            PEER_RESULT_CACHE[str(path)] = cached
        counts['episodes'] += 1
        counts['valid'] += int(cached[1])
        counts['successes'] += int(cached[1] and cached[2])
    halted = (directory / mode / 'selections/halt.json').exists()
    failed = live['state'] in ('FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'BOOT_FAIL')
    return dict(live, mode=mode, **counts, **peer_owner_review(directory, now),
        pointer_status=pointer.get('status'),
        verified_complete=bool(pointer.get('verified_complete')), verified_failure=failed or halted,
        selector_halted=halted, heartbeat_stage=heartbeat.get('stage'),
        heartbeat_age_seconds=now - heartbeat['updated_unix'] if heartbeat.get('updated_unix') else None,
        supervisor_age_seconds=now - supervisor['checked_unix'] if supervisor.get('checked_unix') else None,
        owning_agent_thread=pointer.get('agent_thread'), continuation_active=pointer.get('continuation_active'),
        continuation_active_source='owner_reported',
        approved_resources=pointer.get('approved_resources'),
        owner_reported_remaining_seconds=pointer.get('remaining_approved_seconds'),
        protocol_group='sft_jev_kev_full300', repair_ownership='Existing peer agent; coordinate before any repair')


def direct_actor_peer_snapshot(runtime, now):
    """Read the Kev actor owner's aggregate snapshot without adopting its budget."""
    pointer = read(Path(runtime) / 'current_kev27b_actor_full300.json')
    if not pointer.get('run_directory'):
        return dict(state='UNREGISTERED')
    directory = Path(pointer['run_directory'])
    live = peer_accounting(directory, pointer.get('job_id'))
    supervisor = read(directory / 'supervisor-latest.json')
    same_job = str(supervisor.get('job_id')) == str(pointer.get('job_id'))
    summary = supervisor.get('summary', {}) if same_job else {}
    # Use one coherent summary: the raw result count can be one record ahead
    # of its valid/success totals while the peer atomically commits an episode.
    counts = {name: summary.get(key) for name, key in (
        ('episodes', 'completed'), ('valid', 'valid'), ('successes', 'successes'))}
    coherent = (all(type(value) is int for value in counts.values()) and
                0 <= counts['successes'] <= counts['valid'] <= counts['episodes'] <= 300)
    verified = (coherent and counts['episodes'] == 300 and
                live['state'] == 'COMPLETED' and pointer.get('verified_complete') is True)
    heartbeat = supervisor.get('heartbeat', {}) if same_job else {}
    halted = bool(same_job and supervisor.get('selector_halted'))
    return dict(live, **counts, counts_available=coherent,
        verified_complete=verified,
        verified_failure=halted or live['state'] in (
            'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'BOOT_FAIL'),
        stale_heartbeat=bool(same_job and supervisor.get('stale_heartbeat')),
        heartbeat_age_seconds=now - heartbeat['updated_unix'] if heartbeat.get('updated_unix') else None,
        owner_reported_snapshot_epoch=supervisor.get('checked_unix') if same_job else None,
        owning_agent_thread=pointer.get('agent_thread'),
        continuation_active=pointer.get('continuation_active'),
        **peer_owner_review(directory, now),
        protocol_group='kev27b_direct_actor', matched_protocol=False,
        repair_ownership='Existing peer agent under its own protocol and budget')


def snapshot(root=ROOT):
    root, now = Path(root), time.time()
    plan, approval = read(root / 'plan.json'), read(root / 'approval.json')
    selected, pointers = {}, {}
    for pool in ('gpu', 'cpu'):
        pointers[pool] = read(root / 'pools' / pool / 'current.json')
        attempts = [a for a in approval.get('attempts', []) if a.get('pool') == pool]
        selected[pool] = str(attempts[-1]['job_id']) if attempts else pointers[pool].get('job_id')
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {pool: executor.submit(scheduler, job) for pool, job in selected.items()}
        peer_futures = {mode: executor.submit(peer_snapshot, root.parent, mode, pointer, now)
                        for mode, pointer in PEERS.items()}
        direct_future = executor.submit(direct_actor_peer_snapshot, root.parent, now)
        states = {pool: future.result() for pool, future in futures.items()}
        peers = {mode: future.result() for mode, future in peer_futures.items()}
        direct_peer = direct_future.result()
    modes = {mode: dict(episodes=0, valid=0, successes=0) for mode in MODES}
    workers, record_errors = {}, []
    family_plans = {family: read(reference['plan']) for family, reference in plan.get('families', {}).items()}
    for family, fp in family_plans.items():
        for path in (Path(fp['output']) / 'records').glob('*.json'):
            row = read(path)
            mode = row.get('mode')
            if mode not in modes or MODES[mode] != family:
                record_errors.append(str(path))
                continue
            modes[mode]['episodes'] += 1
            modes[mode]['valid'] += int(bool(row.get('valid')))
            modes[mode]['successes'] += int(bool(row.get('valid')) and row.get('reward') == 1)
        # Retain every historical worker heartbeat in the snapshot. Only the
        # current attempt's live journal entries are eligible for stall alerts.
        pool = 'cpu' if family == 'luna' else 'gpu'
        job = selected[pool]
        journal = read(root / 'pools' / pool / 'attempts' / str(job) / 'workers.json')
        journal = journal if isinstance(journal, list) else []
        active_entries = {str(w['worker_id']): w for w in journal if w.get('family') == family}
        known = {path.parent.name for path in (Path(fp['output']) / 'workers').glob('*/status.json')}
        for worker_id in sorted(known | set(active_entries)):
            entry = active_entries.get(worker_id, dict(worker_id=worker_id, returncode='historical'))
            workers[worker_id] = worker_snapshot(root, family, fp, entry, str(job), now)
    pools = {pool: dict(states[pool], controller=pointers[pool],
             pointer_stale=bool(pointers[pool].get('job_id') and pointers[pool]['job_id'] != selected[pool]))
             for pool in selected}
    ledgers = {role: read(root / 'budget' / (role + '-budget') / 'ledger.json') for role in ('luna', 'judge')}
    halts = {role: read(root / 'budget' / (role + '-budget') / 'halt.json') for role in ('luna', 'judge')}
    mismatch = (approval.get('approved') is not True or approval.get('limits') != LIMITS
                or approval.get('pools', {}).get('gpu', {}).get('total_seconds') != 57600
                or approval.get('pools', {}).get('cpu', {}).get('total_seconds') != 14400)
    pilot = read(root.parent / 'luna-qwen-inference-20261004/supervisor/latest.json')
    pilot_summary = {key: pilot.get(key) for key in ('job_id', 'state', 'completed', 'expected_episodes')}
    pilot_summary['supervisor_age_seconds'] = now - pilot['epoch'] if pilot.get('epoch') else None
    return dict(epoch=now, pid=os.getpid(), pools=pools, modes=modes, peers=peers,
        direct_actor_peer=direct_peer, pilot=pilot_summary,
        completed=sum(value['episodes'] for value in modes.values()), expected_episodes=1500,
        workers=workers, stalled_workers=sorted(k for k, w in workers.items() if w['stalled']),
        api_ledgers=ledgers, api_halts=halts, approval_mismatch=mismatch, record_errors=record_errors,
        source='Read-only job observation; codex queue brings back the owning agent for diagnosis and repair')


def cpu_verification_matches(root, pool, counts):
    """Promote collection only when the independent audit matches this job/cohort."""
    audit = read(Path(root) / 'supervisor/cpu-completion-readiness.json')
    return (pool['state'] == 'COMPLETED' and
            audit.get('complete') is True and audit.get('cpu_collection_verified') is True and
            audit.get('cpu_terminal_accounting') is True and
            audit.get('final_repair_worker_exits_successful') is True and
            audit.get('active_workers') == 0 and
            audit.get('integrity_errors') == [] and audit.get('completion_blockers') == [] and
            audit.get('wandb_reconciliation', {}).get('remote_API_called') is True and
            audit.get('scheduler', {}).get(str(pool['job_id']), {}).get('state') == 'COMPLETED' and
            audit.get('expected_episodes') == audit.get('committed_episodes') == counts['episodes'] == 300 and
            audit.get('valid') == counts['valid'] and audit.get('successes') == counts['successes'])


def update_tracker(latest, root=ROOT):
    # This runtime tracker contains private supervision metadata. Public exports
    # must whitelist aggregate result/protocol fields, stripping owner threads,
    # heartbeat metadata and internal paths rather than copying this JSON.
    root = Path(root)
    path = root / 'experiment-tracker.json'
    with (root / 'experiment-tracker.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        tracker = read(path)
        if not tracker.get('rows'):
            raise ValueError('Unified experiment tracker is missing; preserve its stable row identities')
        for row in tracker['rows']:
            mode = row.get('mode')
            if mode == 'kev27b_actor' and row.get('protocol_group') == 'kev27b_direct_actor':
                peer = latest.get('direct_actor_peer', {})
                if peer.get('state') == 'UNREGISTERED' or not peer:
                    continue
                for name in ('job_id', 'verified_complete', 'owning_agent_thread',
                             'repair_ownership', 'owner_reported_snapshot_epoch',
                             'heartbeat_age_seconds', 'owner_last_review_epoch',
                             'consumed_scheduler_seconds', 'remaining_approved_seconds',
                             'accounting_error', 'continuation_active'):
                    row[name] = peer.get(name)
                row['status'] = 'verified_complete' if peer['verified_complete'] else peer['state'].lower()
                if peer.get('counts_available'):
                    row.update(recorded_episodes=peer['episodes'], valid=peer['valid'], successes=peer['successes'])
                    if peer['episodes'] == 300 and not peer['verified_complete']:
                        row['status'] = 'collected_pending_final_verification'
                continue
            if mode in PEERS and row.get('protocol_group') == 'sft_jev_kev_full300':
                peer = latest.get('peers', {}).get(mode, {})
                if peer.get('state') == 'UNREGISTERED' or not peer:
                    continue
                status = 'verified_complete' if peer['verified_complete'] else peer['state'].lower()
                if peer['episodes'] == 300 and not peer['verified_complete']:
                    status = 'collected_pending_final_verification'
                row.update(recorded_episodes=peer['episodes'], successes=peer['successes'],
                    valid=peer['valid'], job_id=peer['job_id'], status=status,
                    verified_complete=peer['verified_complete'], owning_agent_thread=peer['owning_agent_thread'],
                    heartbeat_age_seconds=peer['heartbeat_age_seconds'],
                    supervisor_age_seconds=peer['supervisor_age_seconds'])
                for name in ('consumed_scheduler_seconds', 'remaining_approved_seconds',
                             'owner_reported_remaining_seconds', 'accounting_error',
                             'owner_last_review_epoch', 'owner_last_review_age_seconds',
                             'owner_last_review_source', 'owner_acknowledge_epoch',
                             'owner_review_document_epoch', 'continuation_active',
                             'continuation_active_source'):
                    row[name] = peer.get(name)
                continue
            if mode not in MODES or row.get('protocol_group') != 'luna_full300':
                continue
            pool = latest['pools']['cpu' if mode == 'luna' else 'gpu']
            counts = latest['modes'][mode]
            stage = pool['controller'].get('stage', pool['state'].lower())
            verified = mode == 'luna' and cpu_verification_matches(root, pool, counts)
            if counts['episodes'] == 300:
                stage = 'verified_complete' if verified else 'collected_pending_final_verification'
            elif pool['state'] not in ('RUNNING', 'PENDING', 'CONFIGURING'):
                stage = pool['state'].lower()
            elif pool['state'] == 'PENDING':
                stage = 'queued'
            row.update(recorded_episodes=counts['episodes'], successes=counts['successes'],
                       valid=counts['valid'], job_id=pool['job_id'], status=stage,
                       verified_complete=verified)
        tracker.update(snapshot_epoch=latest['epoch'], snapshot_utc=datetime.fromtimestamp(
            latest['epoch'], timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'))
        attempts = read(root / 'approval.json').get('attempts', [])
        for name in ('approved_luna_budgets', 'proposed_luna_budgets'):
            budget = tracker.get(name)
            if not isinstance(budget, dict):
                continue
            budget['approved'] = not latest['approval_mismatch']
            for pool_name in ('gpu', 'cpu'):
                pool = latest.get('pools', {}).get(pool_name)
                entry = budget.get(pool_name + '_pool')
                if not pool or not isinstance(entry, dict):
                    continue
                original = next((a['job_id'] for a in attempts if a.get('pool') == pool_name),
                                entry.get('job_id') or pool.get('job_id'))
                if original is not None:
                    entry.setdefault('original_job_id', original)
                # Refresh live metadata only; approved CPU/GPU/RAM/time totals
                # and shared API caps remain unchanged across every retry.
                entry.update(job_id=pool.get('job_id'), scheduler_state=pool['state'])
                entry.pop('scheduler_reason', None)
                if pool['state'] == 'PENDING':
                    row = next((line.split('|') for line in pool.get('raw', '').splitlines()
                                if line.split('|')[0] == str(pool.get('job_id'))), [])
                    if len(row) >= 4:
                        entry['scheduler_reason'] = row[3]
        write(path, tracker)


def dispatch_state(latest):
    return dict(pools={name: [p['job_id'], p['state'], p['controller'].get('stage'),
        sorted(p['controller'].get('failed_families', {})), p['pointer_stale']]
        for name, p in latest['pools'].items()}, halted=sorted(k for k, v in latest['api_halts'].items() if v),
        stalled=latest['stalled_workers'], approval_mismatch=latest['approval_mismatch'],
        record_errors=bool(latest['record_errors']), all_records=latest['completed'] == 1500,
        peers={mode: [peer['job_id'], peer['state'], peer['verified_complete'], peer['verified_failure'],
                      bool(peer.get('accounting_error'))]
               for mode, peer in latest.get('peers', {}).items()})


def continuation_message(root=ROOT):
    message = (
        '[Approved full300 Qwen/Luna continuation] '
        f'Read {root}/supervisor/latest.json and {root}/approval.json. '
        'Run python3 scripts/watch_luna_qwen_full300.py --acknowledge. '
        'Inspect both jobs, logs, receipts, browsers and W&B; diagnose and repair failures. '
        'Preserve separate totals across retries: 8 H200/64 CPU/960 GiB x16h and '
        '0 GPU/16 CPU/32 GiB x4h; shared $100 Luna/39600 calls and $25 judge/6600 calls. '
        'Keep independent work parallel, preserve every attempt, update pool pointers and unified tracker. '
        'Verify all1500 episodes, final scheduler/API accounting and three plots; publish aggregate results/plots only to arm. '
        'Track Jev/Kev peers under their own protocols, budgets and repair owners. '
        'This adds no budget. Routine summaries hourly; alert earlier for failure, blocker or verified completion.'
    )
    if len(message.encode()) >= 1000:
        raise ValueError('Continuation message must stay below 1000 bytes')
    return message


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--acknowledge', action='store_true')
    parser.add_argument('--once', action='store_true', help='Read-only snapshot to stdout; no queue or files changed')
    args = parser.parse_args()
    directory = args.root / 'supervisor'
    if args.once:
        print(json.dumps(snapshot(args.root), indent=2))
        return
    directory.mkdir(parents=True, exist_ok=True)
    if args.acknowledge:
        write(directory / 'review.json', dict(epoch=time.time()))
        print('Review acknowledged; inspect and repair both pools before ending the turn.')
        return
    with (directory / 'owner.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        registry = read(directory / 'registry.json')
        if registry.get('thread_id') != THREAD:
            raise ValueError('Watcher must dispatch to the registered owning thread')
        write(directory / 'pid.json', dict(pid=os.getpid(), epoch=time.time()))
        while not read(directory / 'stop.json').get('verified_complete'):
            try:
                latest = snapshot(args.root)
                write(directory / 'latest.json', latest)
                update_tracker(latest, args.root)
                state, review = read(directory / 'dispatch.json'), read(directory / 'review.json')
                pending = state.get('queued_epoch', 0) > review.get('epoch', 0)
                change = dispatch_state(latest)
                due = change != state.get('last_dispatched_state') or time.time() - max(
                    review.get('epoch', 0), state.get('queued_epoch', 0)) >= 900
                if due and not pending:
                    result = subprocess.run(['codex', 'queue', '--thread', THREAD,
                        '--message', continuation_message(args.root)], text=True, capture_output=True, timeout=45)
                    if result.returncode:
                        raise RuntimeError('Continuation dispatch failed: ' + result.stderr[:500])
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
