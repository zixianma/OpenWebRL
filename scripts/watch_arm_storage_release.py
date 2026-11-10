#!/usr/bin/env python3
"""Release only explicitly registered held recoveries once quota permits them."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from check_storage_quota import check_storage_quota
from recover_arm_unfinished import CONTROL
from resume_baseline import write_json


def live_jobs(ids):
    text=subprocess.check_output(['squeue','-h','-j',','.join(ids),'-o','%i|%T|%R'],text=True,timeout=30)
    return {row[0]:dict(state=row[1],reason=row[2]) for row in
            (line.split('|',2) for line in text.splitlines()) if len(row)==3}


def prerequisite_complete(item, jobs, live):
    """Do not reserve storage for a future stage before its predecessor succeeds."""
    dependency=item.get('dependency','')
    if not dependency.startswith('afterok:'):
        return True
    parent=dependency.removeprefix('afterok:')
    if not parent.isdigit() or parent in live:
        return False
    record=next((r for r in jobs if r['job_id']==parent),None)
    if record is None:
        return False
    try:
        status=json.loads((Path(record['controller_root'])/'status.json').read_text())
    except (FileNotFoundError,json.JSONDecodeError):
        return False
    if not status.get('complete'):
        return False
    raw=subprocess.check_output(['sacct','-X','-n','-P','-j',parent,
        '--format=JobID,State'],text=True,timeout=30)
    rows=[r.split('|') for r in raw.splitlines()]
    return any(len(r)>=2 and r[0]==parent and r[1]=='COMPLETED' for r in rows)


def tick():
    path=CONTROL/'submissions.json'
    jobs=json.loads(path.read_text())
    live=live_jobs([r['job_id'] for r in jobs])
    quota=check_storage_quota(0)
    # Reserve headroom for released jobs that may start together. This is
    # conservative: already-written bytes remain reflected in quota usage too.
    reserved=sum(r['storage_reserve_bytes'] for r in jobs
                 if r.get('released') and r['job_id'] in live)
    available=max(0,quota.get('launch_headroom_bytes',quota['safe_headroom_bytes'])-reserved)
    changed=False
    blocked={}
    for item in jobs:
        state=live.get(item['job_id'])
        if state and state['state']=='PENDING' and state['reason']=='(JobHeldAdmin)':
            blocked[item['job_id']]='Administrator hold; normal user release is not permitted'
            if item.get('released'):
                item['released']=False;changed=True
            continue
        if not state or item.get('released'): continue
        if state['state']!='PENDING' or state['reason']!='(JobHeldUser)':
            # A manually released job must also count against remaining space.
            item.update(released=True,release_mode='external',released_epoch=time.time())
            available=max(0,available-item['storage_reserve_bytes']);changed=True
            continue
        if not prerequisite_complete(item,jobs,live):
            blocked[item['job_id']]='Waiting for predecessor checkpoint/evaluations and successful batch completion'
            continue
        if item.get('kind')=='B90':
            from prepare_arm_b_to90 import prerequisite_ready
            if not prerequisite_ready():continue
        if available>=item['storage_reserve_bytes']:
            subprocess.run(['scontrol','release',item['job_id']],check=True,timeout=30)
            item.update(released=True,release_mode='quota-guard',released_epoch=time.time())
            available-=item['storage_reserve_bytes'];changed=True
        else:
            blocked[item['job_id']]='Insufficient storage headroom after earlier-job reservations'
    if changed: write_json(path,jobs)
    states={}
    for item in jobs:
        root=Path(item['controller_root'])
        try: status=json.loads((root/'status.json').read_text())
        except (FileNotFoundError,json.JSONDecodeError): status={}
        details={}
        try:
            plan=json.loads((root/'controller-plan.json').read_text())
            training=Path(plan['training_root'])
            if item['kind']!='beta':
                path=training/'status.json'
                if path.exists():details['training']=json.loads(path.read_text())
            evaluations={}
            for directory in root.parent.glob(f"*-iter*-{item['job_id']}"):
                if not (directory/'evaluation_manifest.json').is_file():continue
                evaluations[directory.name]=dict(verdicts=len(list((directory/'rollouts').glob('*.json'))),
                    rollouts=len(list((directory/'rollouts').glob('*.pt'))))
            details['evaluations']=evaluations
        except (KeyError,FileNotFoundError,json.JSONDecodeError): pass
        states[item['job_id']]=dict(slurm=live.get(item['job_id']),controller=status,
            released=item.get('released',False),details=details,hold_reason=blocked.get(item['job_id']))
    write_json(CONTROL/'watch-status.json',dict(updated_epoch=time.time(),quota=quota,jobs=states,
        mutations='Only release registered held jobs; no submissions, deletions, or budget changes'))
    return not live


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hours',type=float,default=72,help='Zero follows registered jobs until all leave the queue')
    parser.add_argument('--once',action='store_true')
    args=parser.parse_args()
    if args.hours<0:parser.error('hours must be nonnegative')
    lock=(CONTROL/'watch.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    write_json(CONTROL/'watch-process.json',dict(pid=os.getpid(),started_epoch=time.time(),hours=args.hours))
    end=time.monotonic()+args.hours*3600 if args.hours else None
    while True:
        try:
            if tick(): break
        except Exception as exc:
            write_json(CONTROL/'watch-error.json',dict(time=time.time(),error_type=type(exc).__name__,error=str(exc)[:400]))
        if args.once or (end is not None and time.monotonic()>=end): break
        time.sleep(60)


if __name__=='__main__': main()
