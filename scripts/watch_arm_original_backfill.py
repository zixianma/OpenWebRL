#!/usr/bin/env python3
"""Observe approved original-bonus evaluations and audit results; never submit jobs."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import time

from prepare_arm_original_backfill import CONTROL, ROOTS, RUNTIME, combined_audit, read
from resume_baseline import write_json


def tick():
    submissions = [read(CONTROL/f'original{i}-submission.json') for i in ROOTS]
    ids = [s['job_id'] for s in submissions]
    raw = subprocess.run(['squeue','-h','-j',','.join(ids),'-o','%i|%T|%M|%R'],
        capture_output=True,text=True,timeout=30)
    if raw.returncode and 'Invalid job id' not in raw.stderr:
        raise RuntimeError(raw.stderr[:400])
    live = {r[0]:dict(state=r[1],elapsed=r[2],reason=r[3])
        for r in (line.split('|',3) for line in raw.stdout.splitlines()) if len(r)==4}
    jobs = {}
    for item in submissions:
        job, iteration = item['job_id'], item['iteration']
        root = Path(item['evaluation_root'])
        records = [read(p) for p in (root/'rollouts').glob('*.json')]
        state = read(root/'status.json') if (root/'status.json').exists() else {}
        checkpoint = RUNTIME/'evaluations'/ROOTS[iteration]/'runtime'/f'iter_{iteration-1:07d}'
        restore = read(root/'checkpoint_restore_evidence.json') if (root/'checkpoint_restore_evidence.json').exists() else None
        result = dict(iteration=iteration,queue=live.get(job),new_tasks=len(records),
            expected_new_tasks=item['new_tasks'],reused_tasks=item['reused_tasks'],
            saved_rollouts=len(list((root/'rollouts').glob('*.pt'))),status=state,
            restored_checkpoint=restore.get('checkpoint') if restore else None,
            needs_attention=False)
        report = root/'full300-audit.json'
        receipt = CONTROL/f'original{iteration}-completion.json'
        if report.exists() and state.get('complete'):
            if not receipt.exists():
                directories = [root/'rollouts']
                if iteration == 40: directories.insert(0,CONTROL/'retained40/rollouts')
                audit = combined_audit(directories)
                if audit != read(report): raise ValueError('Worker and observer audits disagree')
                manifest = read(root/'evaluation_manifest.json')
                if (manifest['checkpoint'] != str(checkpoint) or not restore
                        or restore.get('checkpoint') != str(checkpoint) or restore.get('job_id') != job):
                    raise ValueError('Checkpoint restore identity mismatch')
                write_json(receipt,dict(job_id=job,iteration=iteration,verified_epoch=time.time(),
                    checkpoint=str(checkpoint),metrics=audit))
            result.update(complete=True,verified_result=read(receipt)['metrics'])
        elif job not in live or state.get('complete') is False:
            result['needs_attention']=True
            result['accounting']=subprocess.check_output(['sacct','-X','-n','-P','-j',job,
                '--format=JobID,State,ElapsedRaw,ExitCode'],text=True,timeout=30).strip()
        jobs[job]=result
    result=dict(updated_epoch=time.time(),pid=os.getpid(),hostname=socket.gethostname(),jobs=jobs,
        all_verified=all(v.get('complete') for v in jobs.values()),
        automatic_bug_repair=False,automatic_resubmission=False)
    write_json(CONTROL/'watch-status.json',result)
    return result,not live


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--once',action='store_true');a=ap.parse_args()
    with (CONTROL/'watch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write_json(CONTROL/'watch-process.json',dict(pid=os.getpid(),hostname=socket.gethostname(),
            started_epoch=time.time(),automatic_bug_repair=False,automatic_resubmission=False))
        while True:
            try:
                result,terminal=tick()
                if terminal or a.once:return
                delay=600 if all(j['new_tasks'] or j.get('complete') for j in result['jobs'].values()) else 60
            except Exception as exc:
                write_json(CONTROL/'watch-error.json',dict(updated_epoch=time.time(),
                    error_type=type(exc).__name__,error=str(exc)[:500]))
                if a.once:raise
                delay=60
            time.sleep(delay)


if __name__=='__main__':main()
