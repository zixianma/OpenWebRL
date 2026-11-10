#!/usr/bin/env python3
"""Observe both approved API-actor pools and dispatch their repair owner."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from watch_luna_qwen_full300 import scheduler, read, write

ROOT = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/reasoning-actors-full300-20261005')
THREAD = '01a103a0-a243-7bb0-a194-6271293f0850'
MODES = {'luna_high': ('AS10', 'gpt-6-luna', 15.),
         'sol61_high': ('AS11', 'gpt-6.1-sol', 100.)}
RECEIPT_CACHE = {}


def receipt_health(paths, model, now):
    """Cache small metadata only; never retain images or request payloads."""
    counts = Counter(); errors = Counter(); violations = []; overdue = []
    for path in paths:
        stat = path.stat()
        stamp = (stat.st_mtime_ns,stat.st_size,stat.st_ino)
        cached = RECEIPT_CACHE.get(str(path))
        if cached is None or cached[0] != stamp:
            row = read(path); request = row.get('request', {}); response = row.get('response', {})
            correct = (request.get('model') == model and request.get('reasoning') == {'effort':'high'}
                and request.get('max_output_tokens') == 4096)
            if response:
                correct = correct and (response.get('model') == model or response.get('model','').startswith(model+'-'))
                correct = correct and response.get('reasoning',{}).get('effort') == 'high'
                correct = correct and response.get('status') == 'completed'
            cached = (stamp, dict(status=row.get('status','unreadable'), correct=correct,
                error_type=row.get('error_type'), http_status=row.get('http_status'),
                started_epoch=row.get('started_epoch',now)))
            RECEIPT_CACHE[str(path)] = cached
        row = cached[1]; counts[row['status']] += 1
        if row['error_type']: errors[str(row['http_status'] or row['error_type'])] += 1
        if not row['correct']: violations.append(path.name)
        if row['status'] == 'reserved' and now-row['started_epoch'] > 210: overdue.append(path.name)
    return dict(status_counts=dict(counts), errors=dict(errors), error_count=sum(errors.values()),
        protocol_violations=sorted(violations), overdue_inflight=sorted(overdue))


def snapshot(root=ROOT):
    now = time.time()
    approval = read(root / 'approval.json')
    pools, errors = {}, []
    if (not approval.get('approved') or approval.get('judge_limits') != dict(max_usd=5., max_calls=2640)
            or approval.get('pools') != {m: dict(gpus=0,cpus=16,memory_gib=32,total_seconds=21600) for m in MODES}
            or approval.get('actor_limits') != {m: dict(max_usd=v[2],max_calls=9900) for m,v in MODES.items()}):
        errors.append('approval_mismatch')
    with ThreadPoolExecutor(max_workers=2) as executor:
        jobs = {m: [a for a in approval.get('attempts',[]) if a['pool']==m][-1]['job_id'] for m in MODES}
        futures = {m: executor.submit(scheduler, j) for m,j in jobs.items()}
        states = {m:f.result() for m,f in futures.items()}
    for mode, (row_id,model,_) in MODES.items():
        output = root / mode / 'results'
        records = [read(p) for p in (output/'records').glob('*.json')]
        expected = read(root/mode/'plan.json')
        if any(r.get('mode')!=mode or r.get('actor_model')!=model for r in records):
            errors.append(mode+':record_identity')
        workers=[]
        journal=read(root/'pools'/mode/'attempts'/jobs[mode]/'workers.json') or []
        for entry in journal:
            worker_id=entry['worker_id'];w=output/'workers'/worker_id
            heart,status=read(w/'heartbeat.json'),read(w/'status.json')
            requests=list((output/'attempts'/str(heart.get('task_key','absent'))).glob('*/requests/*.json'))
            last=max([heart.get('updated_unix',0),status.get('updated_unix',0),entry.get('started_epoch',0)]
                +[p.stat().st_mtime for p in requests])
            active='returncode' not in entry and states[mode]['state']=='RUNNING'
            log=root/'workers'/mode/jobs[mode]/worker_id/'collection.log'
            workers.append(dict(worker_id=worker_id,active=active,returncode=entry.get('returncode'),
                status=status,heartbeat=heart,activity_age_seconds=now-last,
                stalled=active and now-last>900,
                log_mtime=log.stat().st_mtime if log.exists() else None,
                wandb_run_id=expected['wandb_prefix']+'-w'+worker_id))
        pools[mode]=dict(states[mode],controller=read(root/'pools'/mode/'current.json'),
            recorded_episodes=len(records),valid=sum(bool(r.get('valid')) for r in records),
            successes=sum(bool(r.get('valid')) and r.get('reward')==1 for r in records),workers=workers,
            ledger=read(root/'budget'/mode/'actor-budget/ledger.json'),
            halt=read(root/'budget'/mode/'actor-budget/halt.json'))
        pools[mode]['api_health']=receipt_health(output.glob('attempts/*/*/requests/actor-*.json'),model,now)
    return dict(epoch=now,pid=os.getpid(),pools=pools,errors=errors,
        judge_ledger=read(root/'budget/judge-budget/ledger.json'),judge_halt=read(root/'budget/judge-budget/halt.json'),
        episodes=sum(v['recorded_episodes'] for v in pools.values()),expected_episodes=600,
        verified_complete=read(root/'completion.json').get('verified_complete') is True)


def update_tracker(latest,root=ROOT):
    tracker_root=root.parent/'luna-qwen-full300-20261004'
    with (tracker_root/'experiment-tracker.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        path=tracker_root/'experiment-tracker.json';tracker=read(path)
        for row in tracker.get('rows',[]):
            mode=row.get('mode')
            if mode not in MODES:continue
            pool=latest['pools'][mode]
            stage=pool['state'].lower()
            if pool['recorded_episodes']==300:stage='collected_pending_final_verification'
            if latest['verified_complete']:stage='verified_complete'
            row.update(approved=True,job_id=pool['job_id'],status=stage,
                recorded_episodes=pool['recorded_episodes'],successes=pool['successes'],valid=pool['valid'],
                verified_complete=latest['verified_complete'],last_review_snapshot_epoch=latest['epoch'])
        tracker['reasoning_high_experiment']=dict(approved=True,root=str(root),expected_episodes=600,
            episodes=latest['episodes'],supervisor_pid=latest['pid'])
        if 'proposed_reasoning_high_experiment' in tracker:tracker['proposed_reasoning_high_experiment']['approved']=True
        tracker['comparison_rows']=len(tracker.get('rows',[]))
        tracker['new_reasoning_high_episodes']=600
        tracker['new_episodes_all_sessions']=sum(r['target_tasks'] for r in tracker.get('rows',[])
            if r.get('protocol_group')!='historical_sft')
        write(path,tracker)


def state_key(latest):
    return dict(pools={m:[v['job_id'],v['state'],v['controller'].get('stage'),bool(v['halt']),
        any(w['stalled'] for w in v['workers']),v['recorded_episodes']==300,
        v.get('api_health',{}).get('error_count',0),v.get('api_health',{}).get('protocol_violations',[]),
        v.get('api_health',{}).get('overdue_inflight',[])] for m,v in latest['pools'].items()},
        errors=latest['errors'],judge_halt=bool(latest['judge_halt']),verified_complete=latest['verified_complete'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--acknowledge',action='store_true');p.add_argument('--once',action='store_true');a=p.parse_args()
    d=ROOT/'supervisor';d.mkdir(exist_ok=True)
    if a.acknowledge:
        write(d/'review.json',dict(epoch=time.time()));print('Review acknowledged; inspect both approved actor pools.');return
    if a.once:
        print(json.dumps(snapshot(),indent=2));return
    with (d/'owner.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write(d/'pid.json',dict(pid=os.getpid(),epoch=time.time(),thread_id=THREAD))
        while True:
            try:
                latest=snapshot();write(d/'latest.json',latest);update_tracker(latest)
                states={v['state'] for v in latest['pools'].values()}
                stage=('verified_complete' if latest['verified_complete'] else
                       'running' if states=={'RUNNING'} else
                       'active' if states & {'RUNNING','PENDING','CONFIGURING'} else
                       'awaiting_final_verification' if latest['episodes']==600 else 'failed_or_partial')
                pointer=read(ROOT/'current.json');pointer.update(last_supervision_epoch=latest['epoch'],stage=stage,
                    completed_episodes=latest['episodes'],verified_complete=latest['verified_complete'],supervisor_pid=os.getpid())
                write(ROOT/'current.json',pointer)
                state,review=read(d/'dispatch.json'),read(d/'review.json')
                key=state_key(latest);pending=state.get('queued_epoch',0)>review.get('epoch',0)
                due=key!=state.get('state') or time.time()-max(state.get('queued_epoch',0),review.get('epoch',0))>=900
                if due and not pending:
                    message=('[Approved Luna-high/Sol6.1-high full300 continuation] Read '+str(ROOT)+'/supervisor/latest.json and approval.json. '
                        'Run python3 scripts/watch_reasoning_actors_full300.py --acknowledge. Inspect both jobs, worker/browser logs, exact model/high-effort receipts, W&B and per-model/shared ledgers; diagnose and repair failures. '
                        'Each arm retains its own0 GPU/16 CPU/32GiB x6h total across retries. Luna cap$15/9900 calls; Sol$100/9900; shared judge$5/2640. No transfers from prior studies. '
                        'Keep arms parallel, preserve all attempts, update pointers and unified tracker. Verify600 episodes, final accounting and three cost/latency/token plots; publish aggregate artifacts only to arm. '
                        'Continue original Qwen/Luna supervision under its separate approval. Hourly routine summaries, earlier failures/blockers/completion. Adds no budget.')
                    if len(message.encode())>=1000:raise ValueError('Continuation message exceeds queue limit')
                    result=subprocess.run(['codex','queue','--thread',THREAD,'--message',message],text=True,capture_output=True,timeout=45)
                    if result.returncode:raise RuntimeError('Agent dispatch failed: '+result.stderr[:300])
                    write(d/'dispatch.json',dict(queued_epoch=time.time(),state=key,receipt=result.stdout.strip()))
                if latest['verified_complete']:
                    write(d/'finished.json',dict(epoch=time.time(),verified_complete=True));break
            except Exception as exc:
                write(d/'error.json',dict(epoch=time.time(),error_type=type(exc).__name__,error=str(exc)))
                print(type(exc).__name__,str(exc),flush=True)
            time.sleep(60)


if __name__=='__main__':main()
