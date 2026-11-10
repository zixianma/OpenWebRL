#!/usr/bin/env python3
"""Observe approved local benchmark jobs and queue their owner for active repair."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import run_arm_localbench90 as suite
from supervise_controlled_sft_suite import scheduler_records
from watch_arm_branches import dispatch_due


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def original_snapshot(root, now):
    control = root/'original-continuation'
    current = read(control/'current.json')
    if not current or current.get('verified_complete'):
        return {}
    approval = read(control/'approval.json')
    expected = dict(gpus=4, gpu_type='H200', cpus=32, memory_gib=480,
                    total_all_attempt_seconds=64800, total_gpu_hours=72)
    if approval.get('approved') is not True or approval.get('resources') != expected:
        raise ValueError('Original continuation lacks exact separate approval')
    jobs = approval['approved_job_ids']
    records = scheduler_records(jobs)
    if set(records) != set(jobs) or current['job_id'] not in jobs:
        raise ValueError('Original accounting is incomplete')
    job = current['job_id']
    if any(v['state'] not in suite.TERMINAL for k,v in records.items() if k != job):
        raise ValueError('Original lineage has overlapping attempts')
    consumed = sum(v['elapsed_seconds'] for v in records.values())
    if consumed > 64800:
        raise ValueError('Original continuation total budget exceeded')
    attempt = control/'attempts'/job
    checkpoint = attempt/'runtime/latest_checkpointed_iteration.txt'
    evaluation = suite.RUNTIME/f'evaluations/arm-original-iter90-{job}'
    eval_records = list((evaluation/'rollouts').glob('*.json'))
    markers = [control/'restore'/job/'restore.log', attempt/'status.json', checkpoint,
               evaluation/'evaluation.log', suite.RUNTIME/f'logs/arm-original90-{job}.out']
    markers += list((attempt/'iterations').glob('*/groups/*.json'))
    newest = max((p.stat().st_mtime for p in markers if p.exists()), default=0)
    state = records[job]['state']
    alerts = []
    if state == 'RUNNING' and records[job]['elapsed_seconds'] > 1800 and (not newest or now-newest > 1800):
        alerts.append('Original has no fresh restore/training/checkpoint/evaluation evidence for30 minutes')
    return dict(job_id=job, scheduler=records[job], accounting=records,
                all_attempt_seconds=consumed, remaining_seconds=64800-consumed,
                native_checkpoint=checkpoint.read_text().strip() if checkpoint.exists() else None,
                restore=read(control/'restore'/job/'result.json'),
                training_status=read(attempt/'status.json'), eval_records=len(eval_records),
                completion=read(control/'completion.json'), alerts=alerts)


def snapshot(root, now):
    approved = read(root/'local-suite-approval.json')
    if (approved.get('approved') is not True or approved.get('resources') != suite.RESOURCES
            or approved.get('source_manifest_sha256') != suite.sha(suite.prepare.SOURCE/'reference_manifest.json')):
        raise ValueError('Approved resources or source changed')
    if root != suite.CONTROL:
        raise ValueError('Wrong campaign root')
    current = read(root/'local-suite-current.json')
    ledger = suite.ledger()
    jobs = [str(a['job_id']) for v in ledger['methods'].values() for a in v['attempts']]
    records = scheduler_records(jobs)
    if set(records) != set(jobs):
        raise ValueError('Missing scheduler accounting')
    rows = []
    for method, entry in ledger['methods'].items():
        attempts = entry['attempts']
        job = str(attempts[-1]['job_id'])
        previous = [records[str(a['job_id'])] for a in attempts[:-1]]
        if any(p['state'] not in suite.TERMINAL for p in previous):
            raise ValueError('Overlapping attempts: '+method)
        prior = sum(p['elapsed_seconds'] for p in previous)
        total = prior + records[job]['elapsed_seconds']
        if prior + attempts[-1]['maximum_seconds'] > suite.CAP_SECONDS or total > suite.CAP_SECONDS:
            raise ValueError('Cumulative time cap exceeded: '+method)
        controller = root/f'controller-local-{method}-{job}'
        status = read(controller/'status.json')
        stages = []
        for benchmark, expected in suite.prepare.COUNTS.items():
            output = suite.RUNTIME/f'evaluations/arm90-local-{method}-{benchmark}-r1-{job}'
            files = list((output/'rollouts').glob('*.json'))
            log = output/'evaluation.log'
            newest = max((f.stat().st_mtime for f in files), default=0)
            stages.append(dict(benchmark=benchmark, output=str(output), records=len(files),
                expected_records=expected, latest_record_epoch=newest,
                log_mtime=log.stat().st_mtime if log.exists() else 0,
                checkpoint_restore=read(output/'checkpoint_restore_evidence.json'),
                status=read(output/'status.json'), audit=read(output/'audit.json')))
        active = next((s for s in stages if s['benchmark'] == status.get('stage')), None)
        state = records[job]['state']
        issues = []
        disabled = method in approved.get('disabled_methods', [])
        held = method in approved.get('launch_paused_pending_scope_confirmation', [])
        if status.get('stage') == 'failed':
            issues.append('Controller failed: '+status.get('detail',''))
        if state == 'RUNNING' and records[job]['elapsed_seconds'] > 900 and not status:
            issues.append('Controller startup has no status after15 minutes')
        if state == 'RUNNING' and active:
            marker = active['latest_record_epoch'] or active['log_mtime']
            if marker and now-marker > 1800:
                issues.append('No useful progress for30 minutes')
        rows.append(dict(method=method, job_id=job, scheduler=records[job],
            disabled_by_user=disabled, held_for_scope_confirmation=held,
            all_attempt_seconds=total, remaining_seconds=suite.CAP_SECONDS-total,
            controller_status=status, stages=stages, alerts=issues))
    return dict(epoch=now, approved_resources=approved['resources'], methods=rows,
                scope_confirmation_pending=current.get('scope_confirmation_pending',False),
                original_training=original_snapshot(root, now),
                original_dependency=current.get('original'), verified_complete=False)


def poll(root, thread, codex):
    now = time.time()
    if read(root/'local-suite-current.json').get('verified_complete'):
        suite.atomic(root/'local-supervisor-finished.json',dict(epoch=now,verified_complete=True))
        return False
    try:
        snap = snapshot(root, now)
    except Exception as exc:
        snap = dict(epoch=now, error=str(exc), methods=[])
    suite.atomic(root/'local-supervisor-latest.json', snap)
    rows = [r for r in snap['methods'] if not r.get('disabled_by_user')]
    if snap.get('scope_confirmation_pending') and not snap.get('error'):
        # Every remaining job is held. Never turn intentional cancellation into
        # a recovery request while the owner's clarification is pending.
        if all(r.get('held_for_scope_confirmation') and
               r['scheduler']['state'] in ({'PENDING'} | suite.TERMINAL) for r in rows):
            return True
    signature = dict(job=','.join(r['job_id'] for r in rows),
        state=','.join(r['scheduler']['state'] for r in rows),
        gate=snap.get('error'), halt=[r['alerts'] for r in rows],
        stale=any(r['alerts'] for r in rows),
        stage='collected_pending_audit' if any(r['controller_status'].get('artifacts_ready') for r in rows) else 'active')
    original = snap.get('original_training', {})
    if original:
        signature['job'] += ','+original['job_id']
        signature['state'] += ','+original['scheduler']['state']
        signature['halt'].append(original['alerts'])
        signature['stale'] |= bool(original['alerts'])
        if original['completion'].get('complete'):
            signature['stage'] = 'collected_pending_audit'
    old = read(root/'local-supervisor-dispatch.json')
    ack = read(root/'local-supervisor-ack.json').get('epoch',0)
    urgent = bool(snap.get('error') or signature['stale'] or
        any(r['scheduler']['state'] in suite.TERMINAL for r in rows) or
        original.get('scheduler',{}).get('state') in suite.TERMINAL or
        signature['state'] != old.get('signature',{}).get('state'))
    if dispatch_due(signature,old,ack,now,urgent):
        message = (f'[Authorized local WebVoyager/DeepShop supervision] Read {root}/local-suite-current.json first; '
            'if verified_complete, silently ack and stop. '
            f'Read {root}/local-supervisor-latest.json and local-suite-approval.json. '
            f'Ack: python3 scripts/watch_arm_localbench90.py --root {root} --ack. '
            'Follow all registered jobs through WebVoyager595 then DeepShop150: native89 weights, sampling, local browsers, '
            'GPT-4o benchmark judges, logs/GPU/memory, task archives/screenshots/verdicts and W&B. '
            'Diagnose/test/fix/relaunch only within EACH approved1H200/8CPU/240GiB/43200s TOTAL; '
            'charge every attempt, preserve partial outputs and update replacement pointers. '
            'Eight browsers/job; repeat1 only. Also follow original-continuation/current.json: separately approved '
            '4H200/32CPU/480GiB/64800s TOTAL, preserved optimizer/scheduler/cursor from80 through90 and owned OM2W300eval90. '
            'Audit native89 then launch Original local595+150 within its already approved1GPU12h. No stealth or repeats2/3. '
            'Audit each595+150 cohort and accounting before verified_complete. Hourly meaningful reports; failures/completion promptly. No added budget.')
        if len(message.encode()) > 1800:
            raise ValueError('Callback exceeds message limit')
        result = subprocess.run([codex,'queue','--thread',thread,'--message',message],
                                text=True,capture_output=True,timeout=45)
        if result.returncode:
            raise RuntimeError('Owner dispatch failed: '+result.stderr[:300])
        receipt = dict(epoch=now,signature=signature,thread=thread,receipt=result.stdout.strip())
        suite.atomic(root/'local-supervisor-dispatch.json',receipt)
        with (root/'local-supervisor-dispatch-history.jsonl').open('a') as f:
            f.write(json.dumps(receipt)+'\n')
    return True


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=suite.CONTROL)
    p.add_argument('--thread');p.add_argument('--codex-bin',default='codex')
    p.add_argument('--ack',action='store_true');p.add_argument('--once',action='store_true');p.add_argument('--daemon',action='store_true')
    a=p.parse_args();root=a.root.resolve()
    if a.ack:
        suite.atomic(root/'local-supervisor-ack.json',dict(epoch=time.time()));return
    if not a.thread:p.error('--thread required')
    if a.daemon:
        command=[sys.executable,'-B',str(Path(__file__).resolve()),'--root',str(root),
                 '--thread',a.thread,'--codex-bin',a.codex_bin]
        with (root/'local-supervisor-process.log').open('a') as log:
            child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        suite.atomic(root/'local-supervisor-process.json',dict(pid=child.pid,started=time.time(),thread=a.thread,command=command))
        print(child.pid);return
    with suite.locked(root/'local-supervisor.lock',nonblocking=True):
        while True:
            try:
                if not poll(root,a.thread,a.codex_bin):break
            except Exception as exc:
                suite.atomic(root/'local-supervisor-error.json',dict(epoch=time.time(),error=str(exc)))
                if a.once:raise
            if a.once:break
            time.sleep(60)


if __name__=='__main__':main()
