#!/usr/bin/env python3
"""Queue active-agent supervision for the approved historical-protocol comparison."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from supervise_controlled_sft_suite import read, write, digest, scheduler_records, TERMINAL
from watch_arm_branches import dispatch_due


RESOURCES = dict(gpus=2, gpu_type='H200', cpus=16, memory_gib=240, total_seconds=14400)
LIMITS = dict(judge_usd=10., judge_calls=1980)


def validate_approval(plan, approval):
    if (approval.get('approved') is not True or approval.get('plan_sha256') != digest(plan)
            or approval.get('resources') != plan.get('resources') or plan.get('resources') != RESOURCES
            or approval.get('limits') != plan.get('limits') or plan.get('limits') != LIMITS):
        raise ValueError('Exact independent shard approval required')
    jobs = [str(x['job_id']) for x in approval.get('attempts', [])]
    if not jobs or len(set(jobs)) != len(jobs):
        raise ValueError('Unique registered attempts required')
    return jobs


def shard_snapshot(root, now):
    plan, approval = read(root/'plan.json'), read(root/'approval.json')
    jobs = validate_approval(plan, approval)
    records = scheduler_records(jobs)
    if set(records) != set(jobs):
        raise ValueError('Scheduler accounting incomplete')
    current = read(root/'current.json'); job = str(current.get('job_id'))
    if job not in jobs:
        raise ValueError('Current attempt is not registered')
    scheduler = records[job]; prior = [r for j,r in records.items() if j != job]
    if any(r['state'] not in TERMINAL for r in prior):
        raise ValueError('Multiple active writers in one shard')
    previous = sum(r['elapsed_seconds'] for r in prior)
    attempt = next(a for a in approval['attempts'] if str(a['job_id']) == job)
    if previous + attempt['time_limit_seconds'] > RESOURCES['total_seconds']:
        raise ValueError('Retry reservation exceeds independent shard budget')
    total = previous + scheduler['elapsed_seconds']
    ledger = read(root/'collection/judge-budget/ledger.json')
    if (total > RESOURCES['total_seconds'] or ledger.get('calls',0) > LIMITS['judge_calls']
            or ledger.get('charged_or_reserved_usd',0) > LIMITS['judge_usd'] + 1e-8):
        raise ValueError('Allocation or API cap exceeded')
    collection = root/plan.get('collection_directory','collection')
    if collection.parent != root:
        raise ValueError('Collection must be an immediate shard subdirectory')
    files = list((collection/'records').glob('*.json'))
    heartbeat_age = now - (root/'current.json').stat().st_mtime
    progress_path = root/'scheduler-attempts'/job/'worker-config.json'
    progress_time = max([p.stat().st_mtime for p in files] +
                        ([progress_path.stat().st_mtime] if progress_path.exists() else [now]))
    running = scheduler['state'] == 'RUNNING'
    return dict(root=str(root),job_id=job,scheduler=records,current=current,
        records=len(files),expected_records=450,allocation_seconds=total,judge=ledger,
        collection_directory=str(collection),
        halt=read(root/'collection/judge-budget/halt.json'),
        stale=running and scheduler['elapsed_seconds'] > 300 and heartbeat_age > 240,
        stalled=running and current.get('stage') == 'collecting' and now-progress_time > 2100,
        current_state=scheduler['state'])


def poll(root, thread, codex):
    now=time.time()
    if read(root/'completion.json').get('verified_complete'):
        write(root/'supervisor-finished.json',dict(epoch=now,verified_complete=True)); return False
    shards=[]
    for i in range(2):
        try:
            shards.append(shard_snapshot(root/f'shard-{i}',now))
        except Exception as exc:
            shards.append(dict(root=str(root/f'shard-{i}'),error=str(exc)))
    write(root/'supervisor-latest.json',dict(epoch=now,shards=shards,verified_complete=False))
    signature=dict(job=','.join(s.get('job_id','?') for s in shards),
        state=','.join(s.get('current_state','ERROR') for s in shards),
        stage='collecting',halt=[s.get('halt') for s in shards],
        stale=any(s.get('stale') or s.get('stalled') for s in shards),
        gate=[s.get('error') for s in shards])
    old=read(root/'supervisor-dispatch.json'); ack=read(root/'supervisor-ack.json').get('epoch',0)
    urgent=(signature['stale'] or any(s.get('error') or s.get('halt') for s in shards)
        or any(s.get('current_state') in TERMINAL for s in shards)
        or ('RUNNING' in signature['state'] and signature['state'] != old.get('signature',{}).get('state')))
    if dispatch_due(signature,old,ack,now,urgent):
        message=(f'[Approved historical-protocol three-arm ARM inference] Read {root}/supervisor-latest.json '
            'and both shard approval.json files. Ack: python3 scripts/watch_arm_paired_inference.py --root '
            f'{root} --ack. Inspect scheduler, logs, actual models/decoding, browsers, judge receipts and W&B. '
            'Diagnose/test/repair within EACH shard2H200/16CPU/240GiB x4h total and $10/1980 judge calls; '
            'preserve all attempts, no transfers. Verify900 episodes, paired analysis, final accounting and plots; '
            'publish aggregates only to arm. Hourly routine reports, earlier failures/completion. Adds no budget.')
        if len(message.encode()) > 1000: raise ValueError('Queue message too long')
        result=subprocess.run([codex,'queue','--thread',thread,'--message',message],capture_output=True,text=True,timeout=45)
        if result.returncode: raise RuntimeError('Agent dispatch failed: '+result.stderr[:300])
        receipt=dict(epoch=now,signature=signature,thread=thread,receipt=result.stdout.strip())
        write(root/'supervisor-dispatch.json',receipt)
        with (root/'supervisor-dispatch-history.jsonl').open('a') as handle: handle.write(json.dumps(receipt)+'\n')
    return True


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--thread');p.add_argument('--codex-bin',default='codex')
    p.add_argument('--ack',action='store_true');p.add_argument('--once',action='store_true');p.add_argument('--daemon',action='store_true')
    a=p.parse_args();root=a.root.resolve()
    if a.ack: write(root/'supervisor-ack.json',dict(epoch=time.time()));return
    if not a.thread:p.error('--thread required')
    if a.daemon:
        command=[sys.executable,str(Path(__file__).resolve()),'--root',str(root),'--thread',a.thread,'--codex-bin',a.codex_bin]
        with (root/'supervisor-process.log').open('a') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        write(root/'supervisor-process.json',dict(pid=child.pid,started=time.time(),thread=a.thread,command=command,
            active_agent_queue_verified=False));print(child.pid);return
    with (root/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            try:
                if not poll(root,a.thread,a.codex_bin):break
            except Exception as exc:
                write(root/'supervisor-error.json',dict(epoch=time.time(),error=str(exc)))
                if a.once:raise
            if a.once:break
            time.sleep(60)


if __name__=='__main__':main()
