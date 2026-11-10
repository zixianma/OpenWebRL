#!/usr/bin/env python3
"""Observe the authorized ARM browser job and verify completion; never submit/retry."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time

from run_arm_refresh_browser import CONTROL,RUNTIME,read,write_json


def queue(job):
    result=subprocess.run(['squeue','-h','-j',job,'-o','%T|%M|%R'],capture_output=True,text=True,timeout=30)
    if result.returncode and 'Invalid job id' not in result.stderr:
        raise RuntimeError('Queue lookup failed: '+result.stderr[:250])
    return result.stdout.strip()


def stage_status(root):
    result={}
    for name in ('frozen','adapted'):
        stage=root/name
        rows=[read(p) for p in (stage/'compare_records').glob('*.json')]
        successes=sum(bool(r['valid'] and r['reward']==1) for r in rows)
        valid=sum(r['valid'] for r in rows)
        result[name]=dict(tasks=len(rows),successes=successes,valid=valid,
            smoke_passed=(stage/'smoke-gate.json').exists(),
            actor_restored=(stage/'checkpoint_restore_evidence.json').exists(),
            selector_ready=(stage/'selector-health.json').exists(),
            selector_failure=read(stage/'selector-failure.json') if (stage/'selector-failure.json').exists() else None)
    return result


def verify_complete(root):
    plan=read(root/'audit-plan.json');summary=read(root/'task-success-summary.json')
    if not summary.get('complete'):raise ValueError('Incomplete final report')
    for s in plan['stages']:
        folder=Path(s['output']);rows=[read(p) for p in (folder/'compare_records').glob('*.json')]
        if len(rows)!=300 or {r['task_id'] for r in rows}!=set(s['quality_config']['task_order']):
            raise ValueError('Incomplete or duplicate primary cohort')
        if len({r['task_id'] for r in rows})!=300:raise ValueError('Duplicate primary tasks')
        if not (folder/'smoke-gate.json').exists() or not (folder/'checkpoint_restore_evidence.json').exists():
            raise ValueError('Missing smoke or native checkpoint restoration')
        for r in rows:
            archive=Path(r['artifact'])/(hashlib.sha256(r['task_id'].encode()).hexdigest()+'.pt')
            if not archive.is_file() or not archive.stat().st_size or not archive.with_suffix('.json').is_file():
                raise ValueError('Missing rollout/verdict')
        m=summary['stages'][s['actor']]['by_mode']['arm']
        if m['tasks']!=300 or m['successes']!=sum(bool(r['valid'] and r['reward']==1) for r in rows):
            raise ValueError('Final metrics do not match per-task verdicts')
    return {k:v['by_mode']['arm'] for k,v in summary['stages'].items()}


def tick():
    submission=read(CONTROL/'submission.json');job=submission['job_id'];root=RUNTIME/f'evaluations/arm-refresh-browser-{job}'
    state=read(root/'status.json') if (root/'status.json').exists() else {}
    q=queue(job)
    result=dict(job_id=job,updated_epoch=time.time(),hostname=socket.gethostname(),pid=os.getpid(),
        queue=q,controller=state,stages=stage_status(root),automatic_bug_repair=False)
    if state.get('complete'):
        result.update(complete=True,verified_results=verify_complete(root))
    elif state.get('failed') or (root/'terminal-budget/halt.json').exists() or not q:
        result['needs_attention']=True
        result['accounting']=subprocess.check_output(['sacct','-j',job,'-X','-n','-P','-o','JobID,State,ElapsedRaw,ExitCode'],text=True,timeout=30).strip()
    budget=root/'terminal-budget/ledger.json'
    if budget.exists():result['judge_budget']=read(budget)
    write_json(CONTROL/'watch-status.json',result)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--once',action='store_true');a=ap.parse_args()
    with (CONTROL/'watch.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write_json(CONTROL/'watch-process.json',dict(pid=os.getpid(),hostname=socket.gethostname(),started_epoch=time.time(),expires_epoch=time.time()+72*3600,automatic_bug_repair=False))
        end=time.monotonic()+72*3600
        while time.monotonic()<end:
            try:
                status=tick()
                if status.get('complete') or a.once:return
                steady=all(v['smoke_passed'] for v in status['stages'].values()) or status.get('needs_attention')
            except Exception as exc:
                write_json(CONTROL/'watch-error.json',dict(updated_epoch=time.time(),error_type=type(exc).__name__,error=str(exc)[:500]))
                if a.once:raise
                steady=False
            for _ in range(10 if steady else 1):time.sleep(60)
        write_json(CONTROL/'watch-expired.json',dict(updated_epoch=time.time()))

if __name__=='__main__':main()
