#!/usr/bin/env python3
"""Observe the approved uniform-G8 RL pilot and queue its owning repair agent.

This supervisor is read-only with respect to training, scheduling, and budgets.
It records observations and verified Codex queue receipts. Completion requires
an independent owner audit; a terminal Slurm state never establishes completion.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

from supervise_sft_pass8_extension import (TERMINAL, digest, file_status, read,
                                          redact, scheduler_records, write)

RUNTIME = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime')
DEFAULT_POINTER = RUNTIME/'current_uniform8_rl_control.json'
RESOURCES = dict(jobs=1, gpus=8, gpu_type='H200', hours=16, gpu_hours=128,
                 cpus=64, memory_gib=960)
CAPS = dict(judge_cap_usd=100, judge_call_cap=8000,
            browser_dispatch_cap=12000, browser_concurrency=64)
TOTAL_SECONDS = 57600
RUN_ID = 'outcome-uniform8-original2102-20261006'


def validate_inputs(root, pointer, thread):
    current = read(pointer)
    if Path(current.get('root', '')).resolve() != root:
        raise ValueError('Campaign pointer root mismatch')
    if current.get('owning_thread_id', thread) != thread:
        raise ValueError('Continuation must target the owning root thread')
    approval = read(root/'approval-request.json')
    if (approval.get('approved') is not True or approval.get('resources') != RESOURCES or
            any(type(approval.get(k)) is bool or approval.get(k) != v for k, v in CAPS.items()) or
            approval.get('target_iteration') != 10 or approval.get('full300_evaluations') != [10] or
            approval.get('includes_all_startup_retries_and_evaluations') is not True or
            approval.get('unused_prior_budgets_added') is not False):
        raise ValueError('Exact resource/API/browser/endpoint approval is missing or changed')
    ledger = read(root/'attempts.json')
    if ledger.get('approved_total_seconds') != TOTAL_SECONDS:
        raise ValueError('Original total scheduler budget changed')
    attempts = ledger.get('attempts', [])
    identities = [str(a.get('job_id', '')) for a in attempts]
    if not attempts or len(set(identities)) != len(identities):
        raise ValueError('Missing or duplicate registered scheduler attempts')
    for attempt in attempts:
        if (not str(attempt.get('job_id', '')).isdecimal() or
                type(attempt.get('maximum_seconds')) is not int or
                not 0 < attempt['maximum_seconds'] <= TOTAL_SECONDS or
                type(attempt.get('charged_seconds', 0)) is not int or
                attempt.get('charged_seconds', 0) < 0):
            raise ValueError('Invalid registered scheduler identity or budget')
    claimed = str(current.get('job_id') or current.get('current_job_id') or identities[-1])
    if claimed not in identities:
        raise ValueError('Pointer job is outside the registered attempt ledger')
    prepared = read(root/'launcher-preparation.json')
    if (prepared.get('task_count') != 2102 or
            prepared.get('source_compilation_passed') is not True or
            prepared.get('native_launcher_dry_runs_passed') is not True):
        raise ValueError('Prepared scientific protocol is missing or changed')
    if current.get('source', prepared['source']) != prepared['source']:
        raise ValueError('Pointer and preparation source mismatch')
    return current, approval, ledger, prepared


def continuation_prompt(root, pointer):
    prompt = (f'[Authorized uniform G8 RL supervision] Read {pointer}; if verified_complete, silently ack and stop. '
        f'Read {root}/supervisor-latest.json; ack: python3 scripts/supervise_uniform8_rl_control.py --root {root} --ack. '
        'Follow the current job through iteration10 and full300eval10: logs/GPU, checkpoints, saved rollouts, '
        'images, GPT-4.1/action_history judges and W&B. Fix/test/relaunch only within approved '
        '8H200/64CPU/960GiB/57600s TOTAL,12000 browsers/64concurrent,8000 judge calls/$100 lifetime. '
        'Charge all attempts; preserve optimizer/scheduler/cursor and update replacement pointer. '
        'Audit final checkpoint and all300 eval records before verified_complete. Adaptive run stays paused. '
        'Report changes, routine hourly; failures/completion promptly. No added budget.')
    if len(prompt.encode()) > 1000:
        raise ValueError('Continuation prompt exceeds deployed 1000-byte queue limit')
    return prompt


def notification_decision(old, ack, snapshot, now):
    signature = {k: snapshot.get(k) for k in ('job_id', 'state', 'stage', 'failed',
        'stale_heartbeat', 'stalled_progress', 'halt_sha256', 'controller_complete', 'errors')}
    # Keep polling Slurm every minute, but an unchanged queue does not need a
    # repair-agent turn every fifteen minutes. Allocation startup does.
    previous = old.get('signature') or {}
    started = (signature.get('state') == 'RUNNING' and
        (previous.get('state') != 'RUNNING' or previous.get('job_id') != signature.get('job_id')))
    urgent_evidence = (started or signature.get('state') in TERMINAL or signature.get('failed') or
        signature.get('stale_heartbeat') or signature.get('stalled_progress') or
        signature.get('halt_sha256') or signature.get('controller_complete') or signature.get('errors'))
    urgent = bool(urgent_evidence and signature != old.get('signature'))
    pending = old.get('queued_epoch', 0) > ack
    interval = 3600 if signature.get('state') == 'PENDING' else 900
    due = now-max(ack, old.get('queued_epoch', 0)) >= interval
    initial = not old.get('queued_epoch')
    return urgent or (not pending and (initial or due)), signature, urgent, pending


def queue_review(root, pointer, thread, snapshot, old, ack, now, *, run=subprocess.run, codex_bin='codex'):
    dispatch, signature, urgent, pending = notification_decision(old, ack, snapshot, now)
    if not dispatch:
        return old
    result = run([codex_bin, 'queue', '--thread', thread, '--message', continuation_prompt(root, pointer)],
                 capture_output=True, text=True, timeout=45)
    receipt_text = result.stdout.strip()
    match = re.search(r'Queued message ([A-Za-z0-9-]+) for thread ([A-Za-z0-9-]+)\.', receipt_text)
    if result.returncode or not match or match.group(2) != thread:
        raise RuntimeError('Active-agent queue failed or lacks a verified receipt: '+redact(result.stderr[:300]))
    receipt = dict(queued_epoch=now, signature=signature, queue_receipt=receipt_text,
        queue_message_id=match.group(1), thread_id=thread,
        notification_kind='urgent' if urgent else 'routine', pending_callback_bypassed=bool(urgent and pending))
    with (root/'supervisor-notifications.jsonl').open('a') as handle:
        handle.write(json.dumps(receipt)+'\n')
        handle.flush()
        os.fsync(handle.fileno())
    write(root/'supervisor-state.json', receipt)
    write(root/'active-continuation.json', dict(active_agent_queue_verified=True,
        verified_unix=now, owning_thread_id=thread, queue_message_id=match.group(1),
        queue_receipt=receipt_text, mechanism='poller plus verified Codex root-agent queue acceptance'))
    return receipt


def wandb_snapshot(root, project, now):
    paths = {p for pattern in ('wandb/run-*', 'wandb/offline-run-*',
        'wandb/wandb/run-*', 'wandb/wandb/offline-run-*') for p in root.glob(pattern)}
    return [dict(run_directory=str(p), expected_project=project,
        summary=file_status(p/'files/wandb-summary.json', now),
        debug=file_status(p/'logs/debug.log', now, tail_bytes=600)) for p in sorted(paths)[-2:]]


def artifact_snapshot(run_root, eval_root, now):
    marker = run_root/'latest_checkpointed_iteration.txt'
    durable, errors = None, []
    if marker.exists():
        try:
            raw = marker.read_text().strip()
            if not re.fullmatch(r'\d+', raw):
                raise ValueError('Invalid checkpoint marker')
            durable = int(raw)+1
            if not 1 <= durable <= 10 or not (run_root/f'iter_{durable-1:07d}').is_dir():
                errors.append('checkpoint_marker_outside_endpoint_or_directory_missing')
        except ValueError:
            errors.append('malformed_checkpoint_marker')
    rollouts = eval_root/'rollouts'
    records = sorted(rollouts.glob('*.json'), key=lambda p: p.stat().st_mtime, reverse=True) if rollouts.exists() else []
    recent = []
    for path in records[:2]:
        record = read(path)
        recent.append(dict(record=str(path), task_id=record.get('task_id'),
            judge_model=record.get('judge_model'), judge_prompt_variant=record.get('judge_prompt_variant'),
            metrics=record.get('metrics'), rollout_file=record.get('rollout_file'),
            deep_validation='Owner must independently audit image, rollout and verdict'))
    if len(records) > 300:
        errors.append('unexpected_extra_evaluation_records')
    status = read(eval_root/'status.json')
    if status.get('complete') and (status.get('returncode') != 0 or len(records) != 300):
        errors.append('evaluation_completion_does_not_cover_exact_300')
    return dict(durable_iteration=durable, marker=file_status(marker, now),
        evaluation=dict(root=str(eval_root), status=status, canonical_records=len(records),
            latest_records=recent, controller_audit=read(eval_root/'controller-artifact-audit.json'),
            manifest=file_status(eval_root/'evaluation_manifest.json', now),
            restore=file_status(eval_root/'checkpoint_restore_evidence.json', now)), errors=errors)


def poll(root, pointer, thread, *, run=subprocess.run, now=None, codex_bin='codex'):
    root, pointer = Path(root).resolve(), Path(pointer)
    now = time.time() if now is None else now
    if read(pointer).get('verified_complete'):
        write(root/'supervisor-ack.json', dict(epoch=now, reason='silent stale-completion acknowledgement'))
        write(root/'supervisor-finished.json', dict(epoch=now, reason='verified campaign complete'))
        return False
    current, approval, ledger, prepared = validate_inputs(root, pointer, thread)
    attempts = ledger['attempts']
    ids = [str(a['job_id']) for a in attempts]
    records = scheduler_records(ids, run=run)
    job = ids[-1]
    record = records.get(job, dict(job_id=job, state='UNKNOWN', elapsed_seconds=0))
    errors, missing = [], [j for j in ids if j not in records]
    if missing:
        errors.append('scheduler_accounting_missing:'+','.join(missing))
    # Preserve previous accounting even if the scheduler is temporarily behind.
    charged = {str(a['job_id']):max(a.get('charged_seconds', 0),
        records.get(str(a['job_id']), {}).get('elapsed_seconds', 0)) for a in attempts}
    consumed = sum(charged.values())
    if consumed > TOTAL_SECONDS:
        errors.append('all_attempt_scheduler_budget_exceeded')
    if sum(charged[j] for j in ids[:-1])+attempts[-1]['maximum_seconds'] > TOTAL_SECONDS:
        errors.append('current_allocation_exceeds_remaining_original_budget')
    for identity in ids:
        allocation = records.get(identity, {})
        if allocation.get('elapsed_seconds', 0) and allocation.get('state') not in ('PENDING', 'CONFIGURING'):
            tres = dict(x.split('=', 1) for x in allocation.get('alloc_tres', '').split(',') if '=' in x)
            if (tres.get('cpu') != '64' or tres.get('mem') not in ('960G', '983040M') or
                    tres.get('gres/gpu') != '8' or tres.get('gres/gpu:h200') != '8'):
                errors.append('unexpected_or_unverified_allocation:'+identity)
    if any(records.get(j, {}).get('state') not in TERMINAL for j in ids[:-1] if j in records):
        errors.append('prior_scheduler_attempt_still_active_or_queued')
    run_root = Path(current.get('run_directory', current.get('run', str(RUNTIME/'runs'/RUN_ID))))
    eval_root = Path(current.get('evaluation_root', str(RUNTIME/'evaluations'/f'{RUN_ID}-iter10')))
    status = read(root/'status.json')
    same_job = str(status.get('job_id')) == job
    stage = status.get('stage') if same_job else 'awaiting_current_controller'
    budget = read(root/'judge-budget/ledger.json')
    if budget:
        if budget.get('schema_version') != 1 or budget.get('caps') != CAPS:
            errors.append('judge_browser_ledger_protocol_or_caps_changed')
        for key, cap in [('calls',CAPS['judge_call_cap']),
                         ('browser_dispatches',CAPS['browser_dispatch_cap']),
                         ('charged_or_reserved_usd',CAPS['judge_cap_usd'])]:
            value = budget.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or value > cap:
                errors.append('invalid_or_exceeded_budget_counter:'+key)
    else:
        if any((root/'judge-budget').glob('request-*.json')) or any((root/'judge-budget').glob('browser-*.json')):
            errors.append('missing_budget_ledger_with_existing_physical_receipts')
    halts = {name:read(root/name) for name in ('halt.json','judge-budget/halt.json') if (root/name).exists()}
    artifacts = artifact_snapshot(run_root, eval_root, now)
    artifacts['checkpoint_review'] = read(root/'checkpoint-review.json')
    errors.extend(artifacts.pop('errors'))
    logs = sorted(set(root.glob(f'training-{job}-*.log')) | set(root.glob(f'evaluation-{job}-*.log')) |
                  {root/f'slurm-{job}.out', run_root/'progress.log'}, key=str)
    evaluation_activity = []
    if same_job and stage == 'evaluation':
        # The controller's wrapper log is mostly silent: the worker writes its
        # live log and durable task records under the separate evaluation root.
        logs = sorted(set(logs) | {eval_root/'evaluation.log', eval_root/'runtime/progress.log'}, key=str)
        evaluation_activity = [file_status(Path(item['record']), now)
            for item in artifacts['evaluation']['latest_records']]
    activity = [file_status(path, now, tail_bytes=1400) for path in logs]
    heartbeat = file_status(root/'status.json', now)
    ages = [f['age_seconds'] for f in activity+[heartbeat]+evaluation_activity if not f.get('missing')]
    running = record['state'] == 'RUNNING'
    stale = running and record['elapsed_seconds'] > 900 and (not ages or min(ages) > 600)
    old = read(root/'supervisor-latest.json')
    progress_signature = dict(job_id=job, durable_iteration=artifacts['durable_iteration'],
        browser_dispatches=budget.get('browser_dispatches',0), judge_calls=budget.get('calls',0),
        evaluation_records=artifacts['evaluation']['canonical_records'],
        progress_bytes=file_status(run_root/'progress.log', now).get('size_bytes',0))
    progress_changed = old.get('progress_signature') != progress_signature
    last_progress = now if progress_changed else old.get('last_progress_change_unix',now)
    stalled = running and record['elapsed_seconds'] > 2700 and now-last_progress > 2700
    if status.get('complete') and same_job and (artifacts['durable_iteration'] != 10 or
            artifacts['evaluation']['canonical_records'] != 300):
        errors.append('controller_completion_before_checkpoint10_and_full300')
    snapshot = dict(checked_unix=now, owning_thread_id=thread, job_id=job,
        **{k:v for k,v in record.items() if k!='job_id'}, stage=stage,
        failed=bool(status.get('failed') and same_job), controller_complete=bool(status.get('complete') and same_job),
        verified_complete=False, status=status, heartbeat=heartbeat, stale_heartbeat=stale,
        stalled_progress=stalled, progress_signature=progress_signature,
        last_progress_change_unix=last_progress, all_attempt_scheduler_seconds=consumed,
        scheduler_accounting_complete=not missing, scheduler_budget_seconds=TOTAL_SECONDS,
        remaining_scheduler_seconds=None if missing else max(0,TOTAL_SECONDS-consumed),
        scheduler_attempts=[dict(records.get(j,dict(job_id=j,state='UNKNOWN')), charged_seconds=charged[j]) for j in ids],
        approval_sha256=digest(approval), attempt_ledger_sha256=digest(ledger), judge_budget=budget,
        halts=halts, halt_sha256=digest(halts) if halts else None, artifacts=artifacts,
        logs=activity, gpu=[file_status(p,now,tail_bytes=1500) for p in
            (root/'gpu.log',root/f'gpu-{job}.log',root/f'scheduler-attempts/{job}/gpu.log')],
        wandb_training=wandb_snapshot(run_root,'openwebrl',now),
        wandb_evaluation=wandb_snapshot(eval_root,'openwebrl-evals',now), errors=errors)
    # An owner recovery supersedes a stale failure collected during this scan.
    if read(pointer) != current or read(root/'approval-request.json') != approval or read(root/'attempts.json') != ledger:
        return True
    write(root/'supervisor-latest.json', snapshot)
    queue_review(root,pointer,thread,snapshot,read(root/'supervisor-state.json'),
        read(root/'supervisor-ack.json').get('epoch',0),now,run=run,codex_bin=codex_bin)
    return True


def loop(root, pointer, thread, codex_bin, once=False):
    with (root/'supervisor.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            try:
                if not poll(root,pointer,thread,codex_bin=codex_bin):
                    break
            except Exception as error:
                now=time.time()
                problem=dict(epoch=now,error_type=type(error).__name__,error=redact(str(error)[:500]))
                write(root/'supervisor-error.json',problem)
                try:
                    queue_review(root,pointer,thread,dict(errors=[problem['error']]),read(root/'supervisor-state.json'),
                        read(root/'supervisor-ack.json').get('epoch',0),now,codex_bin=codex_bin)
                except Exception as queue_error:
                    write(root/'supervisor-queue-error.json',dict(epoch=now,error=redact(str(queue_error)[:500])))
                if once:
                    raise
            if once:
                break
            time.sleep(60)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--pointer',type=Path,default=DEFAULT_POINTER)
    parser.add_argument('--thread')
    parser.add_argument('--codex-bin',default=shutil.which('codex') or 'codex')
    parser.add_argument('--ack',action='store_true')
    parser.add_argument('--once',action='store_true')
    parser.add_argument('--daemon',action='store_true')
    args=parser.parse_args()
    root=args.root.resolve()
    if args.ack:
        write(root/'supervisor-ack.json',dict(epoch=time.time()))
        return
    if not args.thread:
        parser.error('--thread must identify the owning root agent')
    if args.daemon:
        root.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,str(Path(__file__).resolve()),'--root',str(root),
            '--pointer',str(args.pointer),'--thread',args.thread,'--codex-bin',args.codex_bin]
        with (root/'supervisor-process.log').open('a') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,start_new_session=True)
        ticks=int(Path(f'/proc/{child.pid}/stat').read_text().rsplit(')',1)[1].split()[19])
        write(root/'supervisor-service.json',dict(pid=child.pid,pgid=child.pid,proc_start_ticks=ticks,
            started_unix=time.time(),command=command,owning_thread_id=args.thread,
            mechanism='detached poller plus Codex root-agent queue',
            continuation_receipt=str(root/'active-continuation.json')))
        print(json.dumps(dict(supervisor_pid=child.pid,root=str(root),active_agent_queue_verified=False,
            note='Confirm first successful queue receipt before claiming active repair continuation')))
        return
    loop(root,args.pointer,args.thread,args.codex_bin,args.once)


if __name__=='__main__':
    main()
