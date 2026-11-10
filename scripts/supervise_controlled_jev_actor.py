#!/usr/bin/env python3
"""Queue active-agent supervision for the separately approved CPU-only Jev run.

No submissions, cancellations, model calls or changes to the SFT supervisor.
"""
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import time

from supervise_controlled_sft_suite import (
    read,write,digest,file_status,database_snapshot,scheduler_records,notification_decision)

POINTER=Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/current_jev_actor_local_full300.json')


def continuation_prompt(root,pointer):
    value=(f'[Authorized local Jev actor supervision] Read {pointer}; if verified_complete, silently ack and stop. '
        f'Read {root}/supervisor-latest.json; ack: python3 scripts/supervise_controlled_jev_actor.py --root {root} --ack. '
        'Follow current job: logs, CPU/memory, rollouts, final screenshots, canonical OM2W judges, W&B. '
        'Diagnose/test/fix/relaunch within approved 0GPU/8CPU/32GiB/14400s TOTAL; 330 browsers/4 concurrent,19800 Jev,19800 helper/$15,1320 judge/$8. '
        'Charge all attempts,preserve evidence,update replacement pointer. Audit300 primary+3smokes before verified_complete; update docs/private review. '
        'Only meaningful reports,hourly routine; failures/completion promptly. No added budget.')
    if len(value.encode())>1000:raise ValueError('Continuation exceeds queue limit')
    return value


def queue_review(root,pointer,thread,snapshot,now,*,run=subprocess.run,codex_bin='codex'):
    old=read(root/'supervisor-state.json');ack=read(root/'supervisor-ack.json').get('epoch',0)
    dispatch,signature,urgent,pending=notification_decision(old,ack,snapshot,now)
    if not dispatch:return
    result=run([codex_bin,'queue','--thread',thread,'--message',continuation_prompt(root,pointer)],
        capture_output=True,text=True,timeout=45)
    if result.returncode:raise RuntimeError('Active-agent continuation failed: '+result.stderr[:500])
    receipt=dict(queued_epoch=now,signature=signature,queue_receipt=result.stdout.strip(),thread_id=thread,
        notification_kind='urgent' if urgent else 'routine',pending_callback_bypassed=bool(pending and urgent))
    import json
    with (root/'supervisor-notifications.jsonl').open('a') as handle:
        handle.write(json.dumps(receipt)+'\n');handle.flush();os.fsync(handle.fileno())
    write(root/'supervisor-state.json',receipt)


def poll(root,pointer,thread,*,run=subprocess.run,codex_bin='codex',now=None):
    root=Path(root).resolve();now=time.time() if now is None else now
    current=read(pointer)
    if current.get('verified_complete'):
        write(root/'supervisor-ack.json',dict(epoch=now,reason='silent stale-completion acknowledgement'))
        write(root/'supervisor-finished.json',dict(epoch=now,reason='independently verified Jev completion'))
        return False
    if Path(current.get('root','')).resolve()!=root:raise ValueError('Jev pointer root mismatch')
    shards=current.get('shards',[])
    if len(shards)!=1 or shards[0].get('shard')!=0 or Path(shards[0]['root']).resolve()!=root:
        raise ValueError('Expected one independent Jev actor cohort')
    plan,approval=read(root/'plan.json'),read(root/'approval.json')
    expected=dict(gpus=0,cpus=8,memory_gib=32,total_seconds=14400,collectors=4)
    if (approval.get('approved') is not True or approval.get('plan_sha256')!=digest(plan)
            or approval.get('resources')!=expected or plan.get('resources')!=expected
            or approval.get('limits')!=plan['limits']):
        raise ValueError('Exact independent Jev allocation and API/browser approval required')
    ids=[str(a['job_id']) for a in approval.get('attempts',[])]
    job=str(shards[0].get('job_id') or current.get('job_id') or (ids[-1] if ids else ''))
    if not ids or len(set(ids))!=len(ids) or job not in ids:raise ValueError('Current job is outside the attempt ledger')
    records=scheduler_records(ids,run=run)
    record=records.get(job,dict(job_id=job,state='UNKNOWN',elapsed_seconds=0,error='scheduler_record_missing'))
    heart=read(root/'current.json');heartbeat=file_status(root/'current.json',now)
    directory=root/'scheduler-attempts'/job;db=database_snapshot(root)
    # Actor descriptors use api_attempts.judge rather than the SFT field.
    if db.get('present'):
        import json
        with sqlite3.connect((root/'state.sqlite3').as_uri()+'?mode=ro',uri=True,timeout=10) as connection:
            for value in db['results'].values():value['judge_http_attempts']=0
            for raw,category,condition in connection.execute("SELECT a.result_json,q.category,q.condition FROM attempts a JOIN queue q USING(item_id) WHERE a.status='finished'"):
                result=json.loads(raw)
                if category+'/'+condition in db['results']:
                    db['results'][category+'/'+condition]['judge_http_attempts']+=result.get('api_attempts',{}).get('judge',0)
    same=str(heart.get('job_id'))==job;running=record['state']=='RUNNING'
    total=sum(r['elapsed_seconds'] for r in records.values())
    item=dict(shard=0,root=str(root),**record,stage=heart.get('stage') if same else 'awaiting_current_controller',
        heartbeat=heartbeat,stale_heartbeat=running and record['elapsed_seconds']>300 and
            (not same or heartbeat.get('age_seconds',1e20)>180),
        stalled_events=running and heart.get('stage') in ('smoke','primary') and bool(db.get('last_event_unix')) and now-db['last_event_unix']>2700,
        halt_sha256=digest(db['halt']) if db.get('halt') else None,database=db,
        collection_complete=read(root/'collection-complete.json').get('collection_complete',False),
        scheduler_attempts=[records.get(j,dict(job_id=j,state='UNKNOWN')) for j in ids],
        all_attempt_scheduler_seconds=total,scheduler_budget_seconds=14400,remaining_scheduler_seconds=max(0,14400-total),
        slurm_log=file_status(root/f'slurm-{job}.log',now,tail_bytes=2500),
        worker_logs=[file_status(p,now,tail_bytes=750) for p in sorted(directory.glob('*worker-*.log'))],
        wandb=[file_status(p,now) for p in sorted((directory/'wandb').glob('run-*'))],
        owned_process_teardown=read(directory/'owned-process-teardown.json'),gpu_allocation=0)
    if len(records)!=len(ids):item['error']='one_or_more_scheduler_attempts_missing'
    snapshot=dict(checked_unix=now,owning_thread_id=thread,campaign_id=current.get('campaign_id'),
        primary_episodes=300,smoke_episodes=3,verified_complete=False,shards=[item])
    fresh=read(pointer)
    if fresh.get('verified_complete') or fresh.get('shards')!=shards or fresh.get('job_id')!=current.get('job_id'):return True
    write(root/'supervisor-latest.json',snapshot)
    queue_review(root,pointer,thread,snapshot,now,run=run,codex_bin=codex_bin)
    return True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--pointer',type=Path,default=POINTER)
    parser.add_argument('--thread');parser.add_argument('--codex-bin',default=shutil.which('codex') or 'codex')
    parser.add_argument('--once',action='store_true');parser.add_argument('--ack',action='store_true')
    args=parser.parse_args();root=args.root.resolve()
    if args.ack:write(root/'supervisor-ack.json',dict(epoch=time.time()));return
    if not args.thread:parser.error('--thread must identify the owning root agent')
    with (root/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            try:
                if not poll(root,args.pointer,args.thread,codex_bin=args.codex_bin):break
            except Exception as error:
                now=time.time();problem=dict(epoch=now,error_type=type(error).__name__,error=str(error)[:500])
                write(root/'supervisor-error.json',problem)
                latest=read(root/'supervisor-latest.json');latest['supervisor_error']=problem
                write(root/'supervisor-latest.json',latest)
                try:
                    queue_review(root,args.pointer,args.thread,dict(shards=[dict(shard='supervisor',error=digest(problem['error']))]),now,codex_bin=args.codex_bin)
                except Exception as failed:write(root/'supervisor-queue-error.json',dict(epoch=now,error=str(failed)[:500]))
                if args.once:raise
            if args.once:break
            time.sleep(60)


if __name__=='__main__':main()
