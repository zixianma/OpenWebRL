#!/usr/bin/env python3
"""Bring the owning agent back for diagnosis of an approved full300 run."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import time

RESULT_CACHE = {}

def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def write(path, value):
    temporary=path.with_suffix('.json.partial')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def poll(root, pointer, thread):
    current=read(pointer);plan=read(root/'plan.json');approval=read(root/'approval.json')
    if not approval.get('approved'):
        raise ValueError('Cannot supervise an unapproved run')
    if current.get('verified_complete'):
        write(root/'supervisor-finished.json',dict(epoch=time.time(),reason='verified completion'))
        return False
    job=current['job_id']
    if not job.isdecimal():raise ValueError('Invalid job ID')
    raw=subprocess.check_output(['sacct','-X','-n','-P','-j',job,
        '--format=JobIDRaw,State,ElapsedRaw,ExitCode'],text=True,timeout=20)
    record=next((line.split('|') for line in raw.splitlines() if line.split('|')[0]==job),None)
    if record is None:raise ValueError('Scheduler has no record for current attempt')
    mode=plan['modes'][0];heart=read(root/'heartbeat.json');now=time.time()
    stale=record[1]=='RUNNING' and now-heart.get('updated_unix',0)>180
    halted=(root/mode/'selections/halt.json').exists()
    results=list((root/mode/'results').glob('*.json'))
    invalid=0
    for path in results:
        stamp=path.stat().st_mtime_ns
        if str(path) not in RESULT_CACHE or RESULT_CACHE[str(path)][0]!=stamp:
            RESULT_CACHE[str(path)]=(stamp,not read(path).get('valid',False))
        invalid+=int(RESULT_CACHE[str(path)][1])
    snapshot=dict(checked_unix=now,job_id=job,state=record[1],elapsed_seconds=int(record[2]),
        heartbeat=heart,stale_heartbeat=stale,selector_halted=halted,
        completed_results=len(results),invalid_results=invalid,
        summary=read(root/mode/'summary.json'),usage=read(root/'budget/usage.json'),
        verified_complete=False)
    write(root/'supervisor-latest.json',snapshot)
    old=read(root/'supervisor-state.json');ack=read(root/'supervisor-ack.json').get('epoch',0)
    signature=[job,record[1],heart.get('stage'),stale,halted,invalid,bool(snapshot['summary'].get('complete'))]
    pending=old.get('queued_epoch',0)>ack
    if not pending and (signature!=old.get('signature') or now-max(ack,old.get('queued_epoch',0))>=900):
        prompt=(f'[Authorized SFT selector full300 supervision] Read {root}/plan.json, approval.json and supervisor-latest.json; '
            'ack with scripts/supervise_sft_selection_full300.py --root '+str(root)+' --ack. '
            f'Follow current {mode} job{job}; inspect logs,GPU use,rollouts,final screenshots,judge verdicts and W&B. '
            'Diagnose/test/fix/relaunch only within the exact approved total resources and API/browser caps, charging every attempt. '
            'Preserve evidence and update the run pointer for replacements. Audit all300 before verified_complete; update docs/private review. '
            'Hourly routine reports; alert failures/completion promptly. This continuation adds no budget.')
        if len(prompt.encode())>1000:raise ValueError('Continuation prompt exceeds queue limit')
        result=subprocess.run(['codex','queue','--thread',thread,'--message',prompt],capture_output=True,text=True,timeout=45)
        if result.returncode:
            raise RuntimeError('Continuation queue failed: '+result.stderr[:500])
        old.update(queued_epoch=now,signature=signature,queue_receipt=result.stdout.strip())
        with (root/'supervisor-notifications.jsonl').open('a') as log:log.write(json.dumps(old)+'\n')
    write(root/'supervisor-state.json',old)
    return True


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--pointer',type=Path)
    p.add_argument('--thread');p.add_argument('--ack',action='store_true');p.add_argument('--once',action='store_true')
    args=p.parse_args()
    if args.ack:
        write(args.root/'supervisor-ack.json',dict(epoch=time.time()))
    else:
        if not args.pointer or not args.thread:p.error('--pointer and --thread required for supervision')
        with (args.root/'supervisor.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            while True:
                try:
                    if not poll(args.root,args.pointer,args.thread):break
                except Exception as error:
                    write(args.root/'supervisor-error.json',dict(epoch=time.time(),error_type=type(error).__name__,error=str(error)[:500]))
                    if args.once:raise
                if args.once:break
                time.sleep(60)
