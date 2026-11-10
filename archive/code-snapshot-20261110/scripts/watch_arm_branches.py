#!/usr/bin/env python3
"""Observe branch work and queue its owning agent for diagnosis and repair."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

from supervise_controlled_sft_suite import read, write, digest, database_snapshot, scheduler_records, TERMINAL


def dispatch_due(signature, old, ack, now, urgent):
    """Coalesce ordinary batch stages; urgent changes bypass a pending callback."""
    def health_key(value):
        value = value or {}
        key = {k: value.get(k) for k in ('job','state','halt','stale','gate')}
        key['closeout'] = value.get('stage') if value.get('stage') in (
            'collected_pending_audit','partial_reconstruction_or_budget') else None
        return key
    changed = health_key(signature) != health_key(old.get('signature'))
    pending = old.get('epoch',0) > ack
    return bool((urgent and changed) or (not pending and (
        changed or now-max(ack,old.get('epoch',0)) >= 900)))


def poll(root, thread, codex='codex'):
    now=time.time()
    if read(root/'completion.json').get('verified_complete'):
        write(root/'supervisor-finished.json',dict(epoch=now,verified_complete=True)); return False
    plan,approval=read(root/'plan.json'),read(root/'approval.json')
    if (approval.get('approved') is not True or approval.get('plan_sha256')!=digest(plan)
            or approval.get('resources')!=plan['resources'] or approval.get('limits')!=plan['limits']):
        raise ValueError('Exact branch allocation approval is required')
    jobs=[str(x['job_id']) for x in approval['attempts']]
    if not jobs or len(set(jobs))!=len(jobs): raise ValueError('Registered attempts required')
    records=scheduler_records(jobs)
    current=read(root/'current.json'); job=str(current.get('job_id') or jobs[-1])
    if job not in jobs: raise ValueError('Pointer refers to an unregistered allocation')
    record=records.get(job,dict(state='UNKNOWN',error='scheduler_record_missing'))
    db=database_snapshot(root)
    age=now-(root/'current.json').stat().st_mtime if (root/'current.json').exists() else None
    snapshot=dict(epoch=now,job_id=job,scheduler=records,current=current,database=db,
        stale=record['state']=='RUNNING' and record.get('elapsed_seconds',0)>300 and (age is None or age>240),
        allocation_seconds=sum(x.get('elapsed_seconds',0) for x in records.values()),
        accepted_states=read(root/'accepted-states.json'),reconstruction_gate=read(root/'reconstruction-gate-failed.json'))
    write(root/'supervisor-latest.json',snapshot)
    signature=dict(job=job,state=record['state'],stage=current.get('stage'),halt=db.get('halt'),
                   stale=snapshot['stale'],gate=snapshot['reconstruction_gate'])
    old=read(root/'supervisor-dispatch.json'); ack=read(root/'supervisor-ack.json').get('epoch',0)
    urgent=record['state'] in TERMINAL or snapshot['stale'] or db.get('halt') or snapshot['reconstruction_gate']
    if dispatch_due(signature,old,ack,now,urgent):
        message=(f'[Approved ARM continuation branches] Read {root}/supervisor-latest.json and approval.json. '
            'Ack with scripts/watch_arm_branches.py --root set to that directory and --ack. '
            'Inspect jobs, browser/actor/judge receipts and W&B; diagnose/test/repair within the exact existing caps. '
            'Keep all attempts and remaining allocation time. Index-only Luna-high; no factored outputs. '
            f"Verify{plan['states_target']*plan['candidate_count']*plan['continuation_repetitions']}continuations/"
            f"{plan['states_target']}states, paired success/repeat-control analysis and final accounting; "
            'publish aggregates only to arm. Hourly routine reports, earlier failures/completion. Adds no budget.')
        if len(message.encode())>1000: raise ValueError('Continuation exceeds queue message cap')
        result=subprocess.run([codex,'queue','--thread',thread,'--message',message],capture_output=True,text=True,timeout=45)
        if result.returncode: raise RuntimeError('Agent queue failed: '+result.stderr[:300])
        receipt=dict(epoch=now,signature=signature,thread=thread,receipt=result.stdout.strip())
        write(root/'supervisor-dispatch.json',receipt)
        with (root/'supervisor-dispatch-history.jsonl').open('a') as handle:
            import json
            handle.write(json.dumps(receipt)+'\n')
    return True


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--thread'); parser.add_argument('--codex-bin',default='codex')
    parser.add_argument('--ack',action='store_true'); parser.add_argument('--once',action='store_true')
    parser.add_argument('--daemon',action='store_true'); args=parser.parse_args(); root=args.root.resolve()
    if args.ack: write(root/'supervisor-ack.json',dict(epoch=time.time())); return
    if not args.thread: parser.error('--thread must identify the owning agent')
    if args.daemon:
        command=[sys.executable,str(Path(__file__).resolve()),'--root',str(root),'--thread',args.thread,'--codex-bin',args.codex_bin]
        with (root/'supervisor-process.log').open('a') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        write(root/'supervisor-process.json',dict(pid=child.pid,started=time.time(),thread=args.thread,
            command=command,active_agent_queue_verified=False)); print(child.pid); return
    with (root/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            try:
                if not poll(root,args.thread,args.codex_bin): break
            except Exception as exc:
                write(root/'supervisor-error.json',dict(epoch=time.time(),error=str(exc)))
                if args.once: raise
            if args.once: break
            time.sleep(60)


if __name__=='__main__': main()
