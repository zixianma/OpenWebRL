#!/usr/bin/env python3
"""Observe registered ARM jobs and queue an agent review in the owning thread.

This process never submits, cancels or modifies GPU jobs. The resumed agent
diagnoses failures and verifies fixes within the recorded remaining approval.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import time


RUNTIME = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime')
DEFAULT = RUNTIME/'arm-turn-bonus-preparation/mixed-reweight-20260927/supervisor'


def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def write(path, value):
    tmp = path.with_name(path.name + f'.{os.getpid()}.partial')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def needs_agent_completion_review(job):
    return (job.get('verified_complete')
            and job.get('require_agent_completion_review')
            and not job.get('agent_completion_reviewed'))


def active_jobs(registry):
    # Controllers may verify their artifacts before the agent has checked and
    # reported the final checkpoint and evaluation. Keep that handoff alive.
    return [j for j in registry['jobs']
            if not j.get('requires_user') and
            (not j.get('verified_complete') or needs_agent_completion_review(j))]


def observed_training_status(status, progress_tail):
    status = dict(status)
    phases = re.findall(r'\[TrainProgress\] rollout=\d+/\d+ phase=(\S+)', progress_tail)
    # Older live controllers call native phase "generate" training. Correct
    # the observation without modifying or interrupting the running process.
    if status.get('stage') == 'training' and phases and phases[-1] in {'generate','generate_rollout'}:
        status['stage'] = 'collection'
    return status


def startup_age_seconds(training, status, now):
    # A milestone resume starts another worker inside the same allocation.
    # The run-level launch manifest still dates from iteration0. Legacy
    # controllers write status.json once on entering actor-startup, so its
    # mtime is the stage boundary when an explicit stage timestamp is absent.
    started = status.get('stage_started_epoch_seconds')
    if status.get('stage') == 'actor-startup':
        if isinstance(started, (int, float)) and started > 0:
            return max(0, now - started)
        path = training/'status.json'
        if path.exists():
            return max(0, now - path.stat().st_mtime)
    manifest = training/'launch_manifest.json'
    return max(0, now - manifest.stat().st_mtime) if manifest.exists() else None


def inspect(registry):
    jobs = active_jobs(registry)
    ids = list(dict.fromkeys([j['job_id'] for j in jobs] +
                            [i for j in jobs for i in j.get('scheduled_continuations', [])]))
    if any(not j.isdecimal() for j in ids):
        raise ValueError('Expected explicit numeric Slurm IDs')
    if not ids:
        return []
    result = subprocess.run(['squeue', '-h', '-j', ','.join(ids), '-o', '%i|%T|%R|%M|%L'],
                            capture_output=True, text=True, timeout=30)
    live = {row.split('|')[0]:row for row in result.stdout.splitlines() if '|' in row}
    missing = [j for j in ids if j not in live]
    accounting = {}
    if missing:
        raw = subprocess.check_output(['sacct', '-X', '-n', '-P', '-j', ','.join(missing),
            '--format=JobID,State,Elapsed,ExitCode'], text=True, timeout=30)
        accounting = {row.split('|')[0]:row for row in raw.splitlines() if '|' in row}
    rows = []
    for job in jobs:
        controller, training = Path(job['controller_root']), Path(job['training_root'])
        # Multi-stage controllers switch their training directory at milestones.
        active = read(controller/'controller-plan.json').get('training_root')
        if active:
            training = Path(active)
        if any(not p.resolve().is_relative_to(RUNTIME) for p in (controller, training)):
            raise ValueError('Registered artifacts must be inside the project runtime')
        status, train = read(controller/'status.json'), read(training/'status.json')
        progress = training/'runtime/progress.log'
        tail = ''
        if progress.exists():
            with progress.open('rb') as handle:
                handle.seek(max(0, progress.stat().st_size-8192))
                tail = handle.read().decode(errors='replace')
        metrics = [line for line in tail.splitlines() if line.startswith('[TrainMetrics]')]
        train = observed_training_status(train, tail)
        progress_age = time.time()-progress.stat().st_mtime if progress.exists() else None
        startup_age = startup_age_seconds(training, train, time.time())
        evaluations = []
        for item in job.get('evaluations', []):
            root = Path(item['root'])
            if not root.resolve().is_relative_to(RUNTIME):
                raise ValueError('Evaluation artifacts must be inside the project runtime')
            if not root.exists():
                continue
            records = list((root/'rollouts').glob('*.json'))
            log = root/'evaluation.log'
            started = root/'evaluation_manifest.json'
            last = max((p.stat().st_mtime for p in records),
                       default=started.stat().st_mtime if started.exists() else time.time())
            evaluations.append(dict(iteration=item['iteration'], root=str(root),
                label=item.get('label'),
                saved_records=len(records), status=read(root/'status.json'),
                metrics=read(root/'metrics.json'), verified=item.get('verified', False),
                agent_reviewed=str(root) in job.get('agent_reviewed_evaluation_roots', []),
                last_record_age_seconds=time.time()-last,
                log_age_seconds=time.time()-log.stat().st_mtime if log.exists() else None))
        rows.append(dict(key=job['key'], job_id=job['job_id'],
            agent_completion_pending=bool(needs_agent_completion_review(job)),
            scheduler=live.get(job['job_id'], accounting.get(job['job_id'], 'UNKNOWN')),
            scheduled_continuations=[dict(job_id=i, scheduler=live.get(i, accounting.get(i, 'UNKNOWN')))
                for i in job.get('scheduled_continuations', []) if i != job['job_id']],
            controller=status, training=train, latest_optimizer_metric=metrics[-1] if metrics else None,
            progress_age_seconds=progress_age,
            startup_age_seconds=startup_age,
            browser_health=read(training/'browser-startup-health.json'),
            cache_policy=read(training/'cache-policy-48/status.json')
                if job.get('cache_policy_change') else {},
            cache_policy_verified=job.get('cache_policy_change', {}).get('verified', False),
            evaluations=evaluations))
    return rows


def urgent_issue(row):
    if row.get('agent_completion_pending'):
        return 'final_artifacts_ready'
    scheduler = row['scheduler'].split('|')
    state = scheduler[1] if len(scheduler) > 1 else 'UNKNOWN'
    if state not in ('PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING'):
        return 'scheduler:'+state
    if row['controller'].get('error') or row['training'].get('failed'):
        return 'worker_failure'
    if row.get('browser_health', {}).get('stop'):
        return 'browser_health_stop'
    for successor in row.get('scheduled_continuations', []):
        fields = successor['scheduler'].split('|')
        future_state = fields[1] if len(fields) > 1 else 'UNKNOWN'
        reason = fields[2] if len(fields) > 2 else ''
        if future_state not in ('PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING', 'COMPLETED'):
            return f"successor_{successor['job_id']}_{future_state}"
        if future_state == 'PENDING' and any(v in reason for v in ('JobHeld', 'DependencyNeverSatisfied')):
            return f"successor_{successor['job_id']}_blocked"
    for evaluation in row.get('evaluations', []):
        label = evaluation.get('label') or str(evaluation['iteration'])
        if evaluation['status'].get('returncode') not in (None, 0):
            return f"evaluation_{label}_failed"
        # Controllers verify artifacts before moving to the next stage. That
        # must not suppress the agent's independent audit and result report.
        if evaluation['status'].get('complete') and not evaluation.get('agent_reviewed'):
            return f"evaluation_{label}_artifacts_ready"
        if evaluation.get('verified'):
            continue
        if (row['controller'].get('stage') == 'evaluation'
                and row['controller'].get('iteration') == evaluation['iteration']
                and evaluation['last_record_age_seconds'] > 1800):
            return f"evaluation_{label}_progress_stale"
    policy = row.get('cache_policy', {})
    if policy and not row.get('cache_policy_verified'):
        if policy.get('stage') == 'failed':
            return 'cache_policy_change_failed'
        if policy.get('applied'):
            return 'cache_policy_receipts_ready_for_verification'
        heartbeat = policy.get('updated_epoch', policy.get('requested_epoch', 0))
        if time.time()-heartbeat > 180:
            return 'cache_policy_driver_stale'
    age = row.get('progress_age_seconds')
    stage = row['training'].get('stage')
    if (state == 'RUNNING' and stage == 'actor-startup'
            and (row.get('startup_age_seconds') or 0) > 900):
        return 'actor_startup_stale'
    if state == 'RUNNING' and age is not None:
        if stage == 'training' and age > 1200:
            return 'training_progress_stale'
        if stage == 'collection' and age > 1800:
            return 'collection_progress_stale'
    return None


def signature(rows):
    # Scalar optimizer steps are in latest.json, but wake the agent on major
    # state/checkpoint transitions rather than once per minibatch.
    return [[r['key'], r['job_id'], r['scheduler'].split('|')[1] if '|' in r['scheduler'] else 'UNKNOWN',
             r['controller'].get('stage'), r['training'].get('stage'),
             r['training'].get('iteration'), r['training'].get('completed_iterations'),
             r['training'].get('failed'), r['controller'].get('error'), urgent_issue(r)]
            for r in rows]


def notification_due(state, sig, reviewed, now, cadence, *, urgent=False, notify_on_change=True):
    pending = state.get('last_queued_epoch', 0) > reviewed
    changed = state.get('last_review_signature') != sig
    last_check = max(reviewed, state.get('last_queued_epoch', 0))
    # A phase transition while an agent is verifying the same urgent event
    # must not enqueue that event again. Distinguish new reasons from changes
    # elsewhere in the full progress signature.
    def reasons(value):
        result = set()
        for row in value or []:
            if len(row) < 10 or not row[-1]:
                continue
            reason = row[-1]
            # A worker failure followed by Slurm's terminal accounting is one
            # incident, not a second request while the agent is repairing it.
            if reason in ('worker_failure', 'browser_health_stop') or reason.startswith((
                    'scheduler:FAILED', 'scheduler:CANCELLED', 'scheduler:TIMEOUT',
                    'scheduler:OUT_OF_MEMORY', 'scheduler:NODE_FAIL', 'scheduler:PREEMPTED')):
                reason = 'job_failure'
            # The agent may acknowledge saved artifacts before Slurm reports
            # COMPLETED. That terminal transition is part of the same audit.
            # Keep each evaluation label too, so later milestones still wake
            # the agent even when one allocation owns several evaluations.
            if reason.startswith('evaluation_') and reason.endswith('_artifacts_ready'):
                result.add((row[0], row[1], 'completion_review'))
            elif reason in ('scheduler:COMPLETED', 'final_artifacts_ready'):
                reason = 'completion_review'
            result.add((row[0], row[1], reason))
        return result
    previous = state.get('last_review_signature', state.get('last_queued_signature'))
    new_urgent = urgent and bool(reasons(sig)-reasons(previous))
    return not pending and ((changed and notify_on_change) or new_urgent or now-last_check >= cadence)


def acknowledge(root):
    latest = read(root/'latest.json')
    write(root/'agent-review.json', dict(epoch=time.time(), signature=latest.get('signature'),
          note='Agent resumed; inspect live jobs and act before ending this turn.'))


def continuation_prompt(root):
    prompt = (
        '[Automatic continuation of authorized ARM supervision] '
        f'Read {root}/latest.json and registry.json. '
        'Run python3 scripts/arm_job_supervisor.py --acknowledge, then check live jobs, logs, '
        'W&B, checkpoints and results. Prioritize the expanded4102 outcome-only run to60 and full300 evals every10. '
        'Diagnose failures, test fixes and relaunch within remaining original approvals; '
        'preserve state and count all consumed time. Update registry/watchers for replacement IDs. '
        'Routine reports hourly; alert sooner for failures, stalls or completion. Mark verified_complete only after artifact checks; '
        'mark requires_user for a real approval blocker. This reminder adds no budget.'
    )
    if len(prompt.encode()) > 1000:
        raise ValueError('Continuation prompt exceeds the task-message limit')
    return prompt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=DEFAULT)
    p.add_argument('--acknowledge', action='store_true')
    args = p.parse_args(); root = args.root
    root.mkdir(parents=True, exist_ok=True)
    if args.acknowledge:
        acknowledge(root); print('Agent review acknowledged'); return
    lock = (root/'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = read(root/'dispatch-state.json')
    while True:
        registry = read(root/'registry.json')
        if not registry or not registry.get('thread_id'):
            raise ValueError('Explicit owning thread and job registry required')
        if registry.get('stop') or not active_jobs(registry):
            reason = 'Supervision explicitly stopped' if registry.get('stop') else 'Agent closed the registered work'
            now = time.time()
            rows = inspect(registry) if active_jobs(registry) else []
            write(root/'latest.json', dict(epoch=now, checked_utc=datetime.now(timezone.utc).isoformat(),
                pid=None, jobs=rows, signature=signature(rows), terminal_snapshot=True,
                monitoring_active=False, reason=reason))
            write(root/'finished.json', dict(epoch=now, reason=reason))
            return
        try:
            now = time.time(); rows = inspect(registry); sig = signature(rows)
            write(root/'latest.json', dict(epoch=now, checked_utc=datetime.now(timezone.utc).isoformat(),
                  pid=os.getpid(), jobs=rows, signature=sig,
                  behavior='Queue active-agent review; no unattended GPU mutations'))
            review = read(root/'agent-review.json')
            reviewed = review.get('epoch', registry['started_epoch'])
            if reviewed >= state.get('last_queued_epoch', 0):
                state['last_review_signature'] = review.get('signature', sig)
            if notification_due(state, sig, reviewed, now, registry.get('review_seconds', 900),
                                urgent=any(urgent_issue(r) for r in rows),
                                notify_on_change=registry.get('notify_on_state_change', True)):
                prompt = continuation_prompt(root)
                result = subprocess.run(['codex', 'queue', '--thread', registry['thread_id'], '--message', prompt],
                    capture_output=True, text=True, timeout=45)
                if result.returncode:
                    raise RuntimeError(f'Agent continuation rejected: {result.stderr[:600]}')
                state.update(last_queued_epoch=now, last_queued_signature=sig, queue_receipt=result.stdout.strip())
                with (root/'notifications.jsonl').open('a') as handle:
                    handle.write(json.dumps(dict(epoch=now,signature=sig,receipt=result.stdout.strip()))+'\n')
                print(json.dumps(dict(epoch=now,queued=True,receipt=result.stdout.strip())),flush=True)
            write(root/'dispatch-state.json', state)
        except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
            write(root/'dispatch-error.json', dict(epoch=time.time(), type=type(exc).__name__, error=str(exc)[:1000]))
            print(json.dumps(dict(dispatch_error=type(exc).__name__)),flush=True)
        time.sleep(max(30, registry.get('poll_seconds', 60)))


if __name__ == '__main__':
    main()
