#!/usr/bin/env python3
"""Observe approved shards and queue the owning agent for active repair.

This process never submits/cancels jobs, changes scientific sources or adds a
budget. Agent callbacks, rather than this read-only poller, own diagnosis and
recovery. A successful queue receipt is required before claiming continuation.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time

DEFAULT_POINTER=Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/current_local_sft_selector_controlled.json')
TERMINAL={'COMPLETED','FAILED','TIMEOUT','CANCELLED','OUT_OF_MEMORY','NODE_FAIL','BOOT_FAIL','PREEMPTED','DEADLINE','REVOKED'}


def read(path):
    path=Path(path)
    return json.loads(path.read_text()) if path.exists() else {}


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.{os.getpid()}.partial')
    with temporary.open('w') as handle:
        json.dump(value,handle,indent=2);handle.write('\n');handle.flush();os.fsync(handle.fileno())
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def file_status(path,now,tail_bytes=0):
    path=Path(path)
    try:
        stat=path.stat()
        info=dict(path=str(path),size_bytes=stat.st_size,modified_unix=stat.st_mtime,age_seconds=max(0,now-stat.st_mtime))
        if tail_bytes:
            with path.open('rb') as handle:
                handle.seek(max(0,stat.st_size-tail_bytes));info['tail']=handle.read(tail_bytes).decode(errors='replace')
        return info
    except FileNotFoundError:return dict(path=str(path),missing=True)


def database_snapshot(root):
    """Never create/update a benchmark DB, even when collection has not started."""
    path=Path(root)/'state.sqlite3'
    if not path.exists():return dict(present=False)
    with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True,timeout=10) as db:
        db.row_factory=sqlite3.Row
        counts=[dict(r) for r in db.execute('SELECT category,condition,status,COUNT(*) AS count FROM queue GROUP BY category,condition,status')]
        active=[dict(r) for r in db.execute("SELECT attempt_id,worker_id,job_id,started,artifact_directory FROM attempts WHERE status='active'")]
        counters={r['key']:r['value'] for r in db.execute('SELECT * FROM counters')}
        halted=db.execute("SELECT value FROM metadata WHERE key='halt'").fetchone()
        last=db.execute('SELECT MAX(created) FROM events').fetchone()[0]
        finished=[]
        for r in db.execute("SELECT a.result_json,q.category,q.condition FROM attempts a JOIN queue q USING(item_id) WHERE a.status='finished'"):
            value=json.loads(r['result_json'])
            finished.append(dict(category=r['category'],condition=r['condition'],valid=value.get('valid'),
                reward=value.get('reward'),invalid_reason=value.get('invalid_reason'),
                screenshot=bool(value.get('final_screenshot')),judge_http_attempts=value.get('judge_http_attempts',0)))
        summary={}
        for row in finished:
            key=row['category']+'/'+row['condition']
            target=summary.setdefault(key,dict(finished=0,valid=0,success=0,screenshots=0,judge_http_attempts=0,invalid_reasons={}))
            target['finished']+=1;target['valid']+=row['valid'] is True
            target['success']+=row['valid'] is True and row['reward']==1
            target['screenshots']+=row['screenshot'];target['judge_http_attempts']+=row['judge_http_attempts']
            if row['valid'] is not True:
                reason=str(row['invalid_reason']);target['invalid_reasons'][reason]=target['invalid_reasons'].get(reason,0)+1
        return dict(present=True,counts=counts,active_attempts=active,counters=counters,
            halt=json.loads(halted['value']) if halted else None,last_event_unix=last,
            results=summary,unsettled_reservations=db.execute("SELECT COUNT(*) FROM reservations WHERE status='reserved'").fetchone()[0])


def scheduler_records(jobs,run=subprocess.run):
    if not jobs:return {}
    jobs=sorted(set(str(j) for j in jobs))
    if any(not j.isdecimal() for j in jobs):raise ValueError('Scheduler identities must be numeric')
    result=run(['/usr/bin/sacct','-X','-n','-P','-j',','.join(jobs),
        '--format=JobIDRaw,State,ElapsedRaw,ExitCode,NodeList'],capture_output=True,text=True,timeout=25)
    if result.returncode:raise RuntimeError('Scheduler query failed: '+result.stderr[:300])
    records={}
    for line in result.stdout.splitlines():
        row=line.split('|')
        if row[0] in jobs:
            if row[0] in records:raise ValueError('Ambiguous scheduler identity')
            records[row[0]]=dict(job_id=row[0],state=row[1].split()[0].rstrip('+'),elapsed_seconds=int(row[2]),
                exit_code=row[3],nodes=row[4])
    return records


def continuation_prompt(root,pointer):
    prompt=(f'[Authorized local SFT selector supervision] Read {pointer}; if verified_complete, silently ack and stop. '
        f'Read {root}/supervisor-latest.json; ack: python3 scripts/supervise_controlled_sft_suite.py --root {root} --ack. '
        'Follow both jobs: logs/GPU, rollouts, final screenshots, canonical judges,W&B. '
        'Diagnose/test/fix/relaunch within each approved2H200/16CPU/240GiB/28800s TOTAL and API/browser caps; charge all attempts,preserve evidence,update replacement pointer. '
        'Audit1200primary+24smokes before verified_complete; update docs/private review. '
        'Only meaningful reports,hourly routine; failures/completion promptly. No added budget.')
    if len(prompt.encode())>1000:raise ValueError('Continuation prompt exceeds1000-byte limit')
    return prompt


def notification_decision(old,ack,snapshot,now):
    signature=[{k:s.get(k) for k in ('shard','job_id','state','stage','stale_heartbeat','stalled_events','halt_sha256','collection_complete','error')}
               for s in snapshot['shards']]
    urgent=any(s.get('state') in TERMINAL or s.get('stale_heartbeat') or s.get('stalled_events')
               or s.get('halt_sha256') or s.get('collection_complete') or s.get('error') for s in snapshot['shards'])
    changed=signature!=old.get('signature')
    pending=old.get('queued_epoch',0)>ack
    due=now-max(ack,old.get('queued_epoch',0))>=900
    # A pending ordinary callback must never suppress a new failure/finish.
    return ((urgent and changed) or (not pending and (changed or due))),signature,urgent,pending


def queue_review(root,pointer,thread,snapshot,old,ack,now,run=subprocess.run,codex_bin='codex'):
    dispatch,signature,urgent,pending=notification_decision(old,ack,snapshot,now)
    if not dispatch:return old
    message=continuation_prompt(root,pointer)
    result=run([codex_bin,'queue','--thread',thread,'--message',message],capture_output=True,text=True,timeout=45)
    if result.returncode:raise RuntimeError('Active-agent queue failed: '+result.stderr[:500])
    receipt=dict(queued_epoch=now,signature=signature,queue_receipt=result.stdout.strip(),thread_id=thread,
        notification_kind='urgent' if urgent else 'routine',pending_callback_bypassed=bool(pending and urgent))
    with (root/'supervisor-notifications.jsonl').open('a') as handle:
        handle.write(json.dumps(receipt)+'\n');handle.flush();os.fsync(handle.fileno())
    write(root/'supervisor-state.json',receipt)
    return receipt


def poll(root,pointer,thread,*,run=subprocess.run,now=None,codex_bin='codex'):
    root=Path(root).resolve();pointer=Path(pointer);now=time.time() if now is None else now
    current=read(pointer)
    if current.get('verified_complete'):
        write(root/'supervisor-ack.json',dict(epoch=now,reason='silent stale-completion acknowledgement'))
        write(root/'supervisor-finished.json',dict(epoch=now,reason='independently verified campaign complete'))
        return False
    if Path(current.get('root','')).resolve()!=root:raise ValueError('Campaign pointer root mismatch')
    shards=current.get('shards',[])
    if len(shards)!=2 or {s['shard'] for s in shards}!={0,1}:raise ValueError('Expected exactly two owned shards')
    inputs=[];jobs=[]
    for shard in shards:
        directory=Path(shard['root']).resolve()
        if not directory.is_relative_to(root):raise ValueError('Shard root escapes campaign')
        plan,approval=read(directory/'plan.json'),read(directory/'approval.json')
        if (approval.get('approved') is not True or approval.get('plan_sha256')!=digest(plan)
                or approval.get('resources')!=plan['resources'] or approval.get('limits')!=plan['limits']):
            raise ValueError('Shard exact approval is missing or differs from frozen plan')
        attempts=approval.get('attempts',[])
        if not attempts:raise ValueError('No registered allocation for approved shard')
        ids=[str(a['job_id']) for a in attempts]
        if len(ids)!=len(set(ids)):raise ValueError('Duplicate allocation in shard ledger')
        job=str(shard.get('job_id') or shard.get('current_job_id') or ids[-1])
        if job not in ids:raise ValueError('Current job is outside the approved attempt ledger')
        jobs.extend(ids);inputs.append((shard,directory,plan,approval,job,ids))
    records=scheduler_records(jobs,run=run)
    snapshot=dict(checked_unix=now,campaign_id=current.get('campaign_id'),owning_thread_id=thread,
        verified_complete=False,shards=[])
    for shard,directory,plan,approval,job,ids in inputs:
        record=records.get(job,dict(job_id=job,state='UNKNOWN',elapsed_seconds=0,error='scheduler_record_missing'))
        heart=read(directory/'current.json');job_directory=directory/'scheduler-attempts'/job
        heartbeat=file_status(directory/'current.json',now)
        same_job=str(heart.get('job_id'))==job
        running=record['state']=='RUNNING'
        db=database_snapshot(directory)
        started=record['elapsed_seconds']
        stale=running and started>300 and (not same_job or heartbeat.get('age_seconds',1e20)>240)
        stalled=running and heart.get('stage') in ('smoke','primary') and bool(db.get('last_event_unix')) and now-db['last_event_unix']>2700
        total=sum(records[j]['elapsed_seconds'] for j in ids if j in records)
        item=dict(shard=shard['shard'],root=str(directory),job_id=job,**{k:v for k,v in record.items() if k!='job_id'},
            stage=heart.get('stage') if same_job else 'awaiting_current_controller',heartbeat=heartbeat,
            stale_heartbeat=stale,stalled_events=stalled,database=db,
            halt_sha256=digest(db['halt']) if db.get('halt') else None,
            collection_complete=read(directory/'collection-complete.json').get('collection_complete',False),
            scheduler_attempts=[records.get(j,dict(job_id=j,state='UNKNOWN')) for j in ids],
            all_attempt_scheduler_seconds=total,scheduler_budget_seconds=plan['resources']['total_seconds'],
            remaining_scheduler_seconds=max(0,plan['resources']['total_seconds']-total),
            gpu=file_status(job_directory/'gpu.log',now,tail_bytes=2000),
            slurm_log=file_status(directory/f'slurm-{job}.log',now,tail_bytes=2500),
            worker_logs=[file_status(p,now,tail_bytes=500) for p in sorted(job_directory.glob('*worker-*.log'))],
            wandb=[file_status(p,now) for p in sorted((job_directory/'wandb').glob('run-*'))],
            smoke_performance=read(job_directory/'smoke-performance.json'),
            owned_process_teardown=read(job_directory/'owned-process-teardown.json'))
        if len([j for j in ids if j in records])!=len(ids):item['error']='one_or_more_scheduler_attempts_missing'
        snapshot['shards'].append(item)
    # The owner can replace a job during this read-only scan. Never dispatch a
    # stale failure after the replacement pointer has already been written.
    fresh=read(pointer)
    previous_jobs=[(s['shard'],s.get('job_id'),s.get('current_job_id'),s['root']) for s in shards]
    fresh_jobs=[(s['shard'],s.get('job_id'),s.get('current_job_id'),s['root']) for s in fresh.get('shards',[])]
    if fresh.get('verified_complete') or fresh_jobs!=previous_jobs:return True
    write(root/'supervisor-latest.json',snapshot)
    old=read(root/'supervisor-state.json');ack=read(root/'supervisor-ack.json').get('epoch',0)
    queue_review(root,pointer,thread,snapshot,old,ack,now,run=run,codex_bin=codex_bin)
    return True


def loop(root,pointer,thread,codex_bin,once=False):
    with (root/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            try:
                if not poll(root,pointer,thread,codex_bin=codex_bin):break
            except Exception as error:
                now=time.time();problem=dict(epoch=now,error_type=type(error).__name__,error=str(error)[:500])
                write(root/'supervisor-error.json',problem)
                snapshot=dict(checked_unix=now,shards=[dict(shard='supervisor',error=digest(problem['error']))])
                try:
                    queue_review(root,pointer,thread,snapshot,read(root/'supervisor-state.json'),
                        read(root/'supervisor-ack.json').get('epoch',0),now,codex_bin=codex_bin)
                except Exception as queue_error:
                    write(root/'supervisor-queue-error.json',dict(epoch=now,error=str(queue_error)[:500]))
                if once:raise
            if once:break
            time.sleep(60)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--pointer',type=Path,default=DEFAULT_POINTER);p.add_argument('--thread')
    p.add_argument('--codex-bin',default=shutil.which('codex') or 'codex')
    p.add_argument('--ack',action='store_true');p.add_argument('--once',action='store_true');p.add_argument('--daemon',action='store_true')
    a=p.parse_args();a.root=a.root.resolve()
    if a.ack:
        write(a.root/'supervisor-ack.json',dict(epoch=time.time()));return
    if not a.thread:p.error('--thread must identify the owning root agent')
    if a.daemon:
        command=[sys.executable,str(Path(__file__).resolve()),'--root',str(a.root),'--pointer',str(a.pointer),
                 '--thread',a.thread,'--codex-bin',a.codex_bin]
        with (a.root/'supervisor-process.log').open('a') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        proc=Path(f'/proc/{child.pid}')
        write(a.root/'supervisor-service.json',dict(pid=child.pid,pgid=child.pid,
            proc_start_ticks=int((proc/'stat').read_text().rsplit(')',1)[1].split()[19]),
            started_unix=time.time(),command=command,thread_id=a.thread,mechanism='detached_process_plus_codex_queue',
            active_agent_queue_verified=False))
        print(json.dumps(dict(supervisor_pid=child.pid,root=str(a.root))));return
    loop(a.root,a.pointer,a.thread,a.codex_bin,a.once)


if __name__=='__main__':main()
