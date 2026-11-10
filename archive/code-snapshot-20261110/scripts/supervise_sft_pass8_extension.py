#!/usr/bin/env python3
"""Observe the approved SFT five-plus-three extension and queue its owning agent.

The queued agent owns diagnosis and repair. This process never launches jobs,
changes scientific settings, or expands resources. A scheduler snapshot alone
does not establish active repair: an acknowledged Codex queue receipt is required.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

DEFAULT_POINTER = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/current_sft_training_pass8.json')
TERMINAL = {'COMPLETED', 'FAILED', 'TIMEOUT', 'CANCELLED', 'OUT_OF_MEMORY', 'NODE_FAIL',
            'BOOT_FAIL', 'PREEMPTED', 'DEADLINE', 'REVOKED'}
RESOURCES = dict(jobs=4, gpus_per_job=1, gpu_type='H200', cpus_per_job=8,
                 memory_gib_per_job=120, total_seconds_per_shard=36000,
                 total_gpu_hours=40, max_concurrent_jobs=4)
LIMITS = dict(browsers_per_shard=1600, concurrent_browsers_per_shard=8,
              judge_http_calls_per_shard=4800, judge_usd_per_shard=12.5,
              browsers_total=6400, judge_http_calls_total=19200, judge_usd_total=50.,
              judge_id='gpt-4.1/action_history')


def read(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else {}


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f'.{os.getpid()}.partial')
    with temporary.open('w') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def redact(text):
    text = re.sub(r'\bsk-[A-Za-z0-9_-]{12,}\b', '[REDACTED]', text)
    return re.sub(r'(?i)((?:api[_-]?key|authorization|access_token)\s*[=:]\s*)[^\s,}]+',
                  r'\1[REDACTED]', text)


def file_status(path, now, tail_bytes=0):
    path = Path(path)
    try:
        stat = path.stat()
        info = dict(path=str(path), size_bytes=stat.st_size, modified_unix=stat.st_mtime,
                    age_seconds=max(0, now-stat.st_mtime))
        if tail_bytes and path.is_file():
            with path.open('rb') as handle:
                handle.seek(max(0, stat.st_size-tail_bytes))
                info['tail'] = redact(handle.read(tail_bytes).decode(errors='replace'))
        return info
    except FileNotFoundError:
        return dict(path=str(path), missing=True)


def scheduler_records(jobs, run=subprocess.run):
    jobs = sorted(set(map(str, jobs)))
    if not jobs or any(not job.isdecimal() for job in jobs):
        raise ValueError('Numeric registered scheduler identities are required')
    result = run(['/usr/bin/sacct', '-X', '-n', '-P', '-j', ','.join(jobs),
                  '--format=JobIDRaw,State,ElapsedRaw,ExitCode,NodeList,AllocTRES'],
                 capture_output=True, text=True, timeout=25)
    if result.returncode:
        raise RuntimeError('Scheduler query failed: '+redact(result.stderr[:300]))
    records = {}
    for line in result.stdout.splitlines():
        row = line.split('|')
        if row[0] not in jobs:
            continue
        if row[0] in records or len(row) < 6 or not row[1] or not row[2].isdigit():
            raise ValueError('Ambiguous or malformed scheduler accounting')
        records[row[0]] = dict(job_id=row[0], state=row[1].split()[0].rstrip('+'),
            elapsed_seconds=int(row[2]), exit_code=row[3], nodes=row[4], alloc_tres=row[5])
    return records


def validate_inputs(root, pointer, thread):
    current = read(pointer)
    if Path(current.get('root', '')).resolve() != root:
        raise ValueError('Campaign pointer root mismatch')
    if current.get('owning_thread_id', thread) != thread:
        raise ValueError('Continuation must target the owning root thread')
    plan, approval = read(root/'plan.json'), read(root/'approval.json')
    if (plan.get('root') != str(root) or plan.get('resources') != RESOURCES or
            plan.get('limits') != LIMITS or approval.get('approved') is not True or
            approval.get('plan_sha256') != digest(plan) or
            approval.get('resources') != RESOURCES or approval.get('limits') != LIMITS):
        raise ValueError('Exact plan/resource/API/browser approval is missing or changed')
    if current.get('plan_sha256', digest(plan)) != digest(plan):
        raise ValueError('Campaign pointer belongs to a different approved plan')
    historical, extension = plan.get('historical_task_order', []), plan.get('extension_task_order', [])
    if (len(historical) != 2000 or len(set(historical)) != 2000 or
            len(extension) != 707 or len(set(extension)) != 707 or
            [task for task in historical if task in set(extension)] != extension or
            plan.get('shards') != [extension[i::4] for i in range(4)] or
            plan.get('new_primary_episodes') != 2121):
        raise ValueError('Filtered extension cohort, partition or historical denominator changed')
    manifest = Path(plan['source'])/'reference_manifest.json'
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != plan['source_manifest_sha256']:
        raise ValueError('Frozen source manifest changed')
    attempts = approval.get('attempts', [])
    ids = [str(a.get('job_id', '')) for a in attempts]
    if not attempts or len(set(ids)) != len(ids):
        raise ValueError('Missing or duplicate registered scheduler attempts')
    for attempt in attempts:
        if (not str(attempt.get('job_id', '')).isdecimal() or type(attempt.get('shard')) is not int or
                attempt['shard'] not in range(4) or type(attempt.get('time_limit_seconds')) is not int or
                not 0 < attempt['time_limit_seconds'] <= 36000):
            raise ValueError('Invalid registered scheduler attempt or budget')
    shards = current.get('shards', [])
    if len(shards) != 4 or {s['shard'] for s in shards} != set(range(4)):
        raise ValueError('Expected four owned shards')
    for shard in shards:
        directory = root/f'shard-{shard["shard"]}'
        if Path(shard['root']).resolve() != directory:
            raise ValueError('Shard root differs from the approved campaign')
        registered = [str(a['job_id']) for a in attempts if a['shard'] == shard['shard']]
        if not registered:
            raise ValueError('Every shard needs a registered scheduler allocation')
        claimed = str(shard.get('job_id') or shard.get('current_job_id') or registered[-1])
        if claimed not in registered:
            raise ValueError('Current job is outside the approved attempt ledger')
    return current, plan, approval


def record_snapshot(directory, now, expected_primary_records):
    """Bounded JSON/sidecar read; deep Torch/screenshot audits remain agent work."""
    files = []
    record_dir = directory/'records'
    if record_dir.exists():
        for entry in os.scandir(record_dir):
            if entry.name.endswith('.json') and entry.is_file():
                files.append((entry.stat().st_mtime, Path(entry.path)))
    recent = []
    for _, path in sorted(files, reverse=True)[:3]:
        record = read(path)
        item = {k: record.get(k) for k in ('task_id', 'attempt', 'valid', 'reward', 'steps',
                  'seed', 'policy_id', 'judge_id', 'error_type', 'artifact')}
        item['record'] = str(path)
        task = str(record.get('task_id', ''))
        artifact = Path(record.get('artifact', ''))
        archive = artifact/(hashlib.sha256(task.encode()).hexdigest()+'.pt')
        sidecar = read(archive.with_suffix('.json'))
        item.update(archive=file_status(archive, now), terminal_status=sidecar.get('terminal_status'),
                    judge_verdict=sidecar.get('reward_metadata', {}).get('judge'),
                    archive_deep_validation='Not performed by poller; independent owner audit still required')
        recent.append(item)
    return dict(canonical_records=len(files), latest_record_unix=max((x[0] for x in files), default=None),
                latest_records=recent, expected_primary_records=expected_primary_records)


def wandb_snapshot(worker, now):
    directories = {p for pattern in ('wandb/run-*', 'wandb/offline-run-*',
                   'wandb/wandb/run-*', 'wandb/wandb/offline-run-*') for p in worker.glob(pattern)}
    result = []
    for path in sorted(directories)[-2:]:
        result.append(dict(run_directory=str(path), summary=file_status(path/'files/wandb-summary.json', now),
                           debug=file_status(path/'logs/debug.log', now, tail_bytes=700),
                           expected_project='openwebrl-evals'))
    return result


def continuation_prompt(root, pointer):
    prompt = (f'[Authorized SFT pass8 extension supervision] Read {pointer}; if verified_complete, silently ack and stop. '
        f'Read {root}/supervisor-latest.json; ack: python3 scripts/supervise_sft_pass8_extension.py --root {root} --ack. '
        'Follow all four jobs: logs/GPU, rollouts, images, GPT-4.1/action_history judges and W&B. '
        'Fix/relaunch only within each approved 1 H200/8 CPU/120 GiB/36000s TOTAL, '
        '1600 browsers,4800 judge calls/$12.5. Charge all attempts; preserve old five and new attempts 5-7. '
        'Update replacement pointers. Audit 2121 new records on 707 misses; join all 10000 old records before verified_complete. '
        'Report only meaningful changes, routine at most hourly; failures/completion promptly. No added budget.')
    if len(prompt.encode()) > 1000:
        raise ValueError('Continuation prompt exceeds the deployed 1000-byte queue limit')
    return prompt


def notification_decision(old, ack, snapshot, now):
    signature = [{k: s.get(k) for k in ('shard', 'job_id', 'state', 'stage', 'stale_heartbeat',
        'stalled_results', 'halt_sha256', 'collection_complete', 'errors')} for s in snapshot['shards']]
    def needs_urgent_review(s):
        return (s.get('state') in TERMINAL or s.get('stage') == 'failed' or
                s.get('stale_heartbeat') or s.get('stalled_results') or
                s.get('halt_sha256') or s.get('collection_complete') or s.get('errors'))
    previous = {s['shard']:s for s in old.get('signature',[]) if needs_urgent_review(s)}
    # A completed shard must not turn another shard's ordinary startup change
    # into an urgent notification. Only new/changed urgent evidence bypasses.
    urgent = any(needs_urgent_review(s) and s != previous.get(s['shard']) for s in signature)
    pending = old.get('queued_epoch', 0) > ack
    due = now-max(ack, old.get('queued_epoch', 0)) >= 900
    initial = not old.get('queued_epoch')
    return (urgent or (not pending and (initial or due))), signature, urgent, pending


def queue_review(root, pointer, thread, snapshot, old, ack, now, *, run=subprocess.run, codex_bin='codex'):
    dispatch, signature, urgent, pending = notification_decision(old, ack, snapshot, now)
    if not dispatch:
        return old
    result = run([codex_bin, 'queue', '--thread', thread, '--message', continuation_prompt(root, pointer)],
                 capture_output=True, text=True, timeout=45)
    receipt_text = result.stdout.strip()
    receipt_match = re.search(r'Queued message ([A-Za-z0-9-]+) for thread ([A-Za-z0-9-]+)\.', receipt_text)
    if result.returncode or not receipt_match or receipt_match.group(2) != thread:
        raise RuntimeError('Active-agent queue failed or lacks a verified receipt: '+redact(result.stderr[:300]))
    receipt = dict(queued_epoch=now, signature=signature, queue_receipt=receipt_text,
        queue_message_id=receipt_match.group(1), thread_id=thread,
        notification_kind='urgent' if urgent else 'routine', pending_callback_bypassed=bool(urgent and pending))
    with (root/'supervisor-notifications.jsonl').open('a') as handle:
        handle.write(json.dumps(receipt)+'\n')
        handle.flush()
        os.fsync(handle.fileno())
    write(root/'supervisor-state.json', receipt)
    write(root/'active-continuation.json', dict(active_agent_queue_verified=True,
        verified_unix=now, owning_thread_id=thread, queue_message_id=receipt_match.group(1),
        queue_receipt=receipt_text, mechanism='poller plus verified Codex queue acceptance for root-agent callback'))
    return receipt


def poll(root, pointer, thread, *, run=subprocess.run, now=None, codex_bin='codex'):
    root, pointer = Path(root).resolve(), Path(pointer)
    now = time.time() if now is None else now
    if read(pointer).get('verified_complete'):
        write(root/'supervisor-ack.json', dict(epoch=now, reason='silent stale-completion acknowledgement'))
        write(root/'supervisor-finished.json', dict(epoch=now, reason='verified campaign complete'))
        return False
    current, plan, approval = validate_inputs(root, pointer, thread)
    records = scheduler_records([a['job_id'] for a in approval['attempts']], run=run)
    snapshot = dict(checked_unix=now, owning_thread_id=thread, plan_sha256=digest(plan),
                    approval_sha256=digest(approval), verified_complete=False, shards=[])
    for shard in sorted(current['shards'], key=lambda row: row['shard']):
        index = shard['shard']
        directory = root/f'shard-{index}'
        attempts = [a for a in approval['attempts'] if a['shard'] == index]
        ids = [str(a['job_id']) for a in attempts]
        # Registration is append-only; the latest approved attempt supersedes
        # old pointer files even while its replacement is still pending.
        job = ids[-1]
        record = records.get(job, dict(job_id=job, state='UNKNOWN', elapsed_seconds=0))
        worker = directory/'workers'/job
        attempt_dir = directory/'scheduler-attempts'/job
        status = read(directory/'status.json')
        heartbeat = file_status(directory/'status.json', now)
        same_job = str(status.get('job_id')) == job
        running = record['state'] == 'RUNNING'
        expected_primary = 3*len(plan['shards'][index])
        recent = record_snapshot(directory, now, expected_primary)
        stale = running and record['elapsed_seconds'] > 300 and (
            not same_job or heartbeat.get('age_seconds', 1e20) > 240)
        last_result = recent['latest_record_unix']
        stalled = running and record['elapsed_seconds'] > 2700 and (
            last_result is None or now-last_result > 2700)
        accounted = sum(records[j]['elapsed_seconds'] for j in ids if j in records)
        missing = [j for j in ids if j not in records]
        errors = []
        if missing:
            errors.append('scheduler_accounting_missing:'+','.join(missing))
        if accounted > 36000:
            errors.append('all_attempt_scheduler_budget_exceeded')
        for identity in ids:
            allocation = records.get(identity, {})
            if allocation.get('elapsed_seconds', 0) and allocation.get('state') not in ('PENDING', 'CONFIGURING'):
                tres = dict(part.split('=', 1) for part in allocation.get('alloc_tres', '').split(',') if '=' in part)
                if (tres.get('cpu') != '8' or tres.get('mem') not in ('120G', '122880M') or
                        tres.get('gres/gpu') != '1' or tres.get('gres/gpu:h200') != '1'):
                    errors.append('unexpected_or_unverified_allocation:'+identity)
        if any(records.get(j, {}).get('state') in ('RUNNING', 'COMPLETING') for j in ids[:-1]):
            errors.append('prior_scheduler_attempt_still_active')
        halts = {name: read(directory/name) for name in ('halt.json', 'judge-budget/halt.json')
                 if (directory/name).exists()}
        budget = read(directory/'judge-budget/ledger.json')
        physical = directory/'physical-attempts'
        dispatches = len(list(physical.iterdir())) if physical.exists() else 0
        if dispatches > 1600:
            errors.append('physical_browser_dispatch_budget_exceeded')
        if budget.get('calls', 0) > 4800 or budget.get('charged_or_reserved_usd', 0.) > 12.5:
            errors.append('judge_budget_exceeded')
        if recent['canonical_records'] > expected_primary:
            errors.append('unexpected_extra_primary_records')
        completion = read(directory/'collection-complete.json')
        if completion.get('complete') is True and (
                completion.get('primary_attempts') != expected_primary or
                recent['canonical_records'] != expected_primary):
            errors.append('collection_receipt_does_not_cover_exact_filtered_shard')
        item = dict(shard=index, root=str(directory), job_id=job,
            **{k: v for k, v in record.items() if k != 'job_id'},
            stage=status.get('stage') if same_job else 'awaiting_current_controller',
            stale_heartbeat=stale, stalled_results=stalled, heartbeat=heartbeat,
            halt_sha256=digest(halts) if halts else None, halts=halts,
            collection_complete=completion.get('complete') is True,
            progress=read(directory/'progress.json'), results=recent,
            historical_join=read(directory/'joined-progress.json'),
            physical_dispatches_charged=dispatches, judge_budget=budget,
            scheduler_attempts=[records.get(j, dict(job_id=j, state='UNKNOWN')) for j in ids],
            all_attempt_scheduler_seconds=accounted, scheduler_accounting_complete=not missing,
            scheduler_budget_seconds=36000, remaining_scheduler_seconds=None if missing else max(0, 36000-accounted),
            gpu=file_status(attempt_dir/'gpu.log', now, tail_bytes=1500),
            worker_log=file_status(attempt_dir/'worker.log', now, tail_bytes=2000),
            evaluation_log=file_status(worker/'evaluation.log', now, tail_bytes=2500),
            wandb=wandb_snapshot(worker, now), errors=errors)
        snapshot['shards'].append(item)
    # Re-read mutable pointers and the attempt ledger before dispatching. A
    # recovery performed during this scan supersedes its old failure snapshot.
    fresh_pointer, fresh_approval = read(pointer), read(root/'approval.json')
    if (fresh_pointer.get('verified_complete') or fresh_pointer != current or fresh_approval != approval):
        return True
    write(root/'supervisor-latest.json', snapshot)
    queue_review(root, pointer, thread, snapshot, read(root/'supervisor-state.json'),
        read(root/'supervisor-ack.json').get('epoch', 0), now, run=run, codex_bin=codex_bin)
    return True


def loop(root, pointer, thread, codex_bin, once=False):
    with (root/'supervisor.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            try:
                if not poll(root, pointer, thread, codex_bin=codex_bin):
                    break
            except Exception as error:
                now = time.time()
                problem = dict(epoch=now, error_type=type(error).__name__, error=redact(str(error)[:500]))
                write(root/'supervisor-error.json', problem)
                snapshot = dict(shards=[dict(shard='supervisor', errors=[problem['error']])])
                try:
                    queue_review(root, pointer, thread, snapshot, read(root/'supervisor-state.json'),
                        read(root/'supervisor-ack.json').get('epoch', 0), now, codex_bin=codex_bin)
                except Exception as queue_error:
                    write(root/'supervisor-queue-error.json', dict(epoch=now, error=redact(str(queue_error)[:500])))
                if once:
                    raise
            if once:
                break
            time.sleep(60)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--pointer', type=Path, default=DEFAULT_POINTER)
    parser.add_argument('--thread')
    parser.add_argument('--codex-bin', default=shutil.which('codex') or 'codex')
    parser.add_argument('--ack', action='store_true')
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--daemon', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    if args.ack:
        write(root/'supervisor-ack.json', dict(epoch=time.time()))
        return
    if not args.thread:
        parser.error('--thread must identify the owning root agent')
    if args.daemon:
        root.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(Path(__file__).resolve()), '--root', str(root),
                   '--pointer', str(args.pointer), '--thread', args.thread, '--codex-bin', args.codex_bin]
        with (root/'supervisor-process.log').open('a') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, start_new_session=True)
        ticks = int(Path(f'/proc/{child.pid}/stat').read_text().rsplit(')', 1)[1].split()[19])
        write(root/'supervisor-service.json', dict(pid=child.pid, pgid=child.pid, proc_start_ticks=ticks,
            started_unix=time.time(), command=command, owning_thread_id=args.thread,
            mechanism='detached poller plus Codex root-agent queue',
            continuation_receipt=str(root/'active-continuation.json')))
        print(json.dumps(dict(supervisor_pid=child.pid, root=str(root),
                             active_agent_queue_verified=False, note='Check the first successful queue receipt')))
        return
    loop(root, args.pointer, args.thread, args.codex_bin, args.once)


if __name__ == '__main__':
    main()
