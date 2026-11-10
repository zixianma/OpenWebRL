#!/usr/bin/env python3
"""Wait for an approved interactive allocation, then run the ARM continuation.

Never requests resources, submits jobs, extends time, cancels an allocation or
retries a failed worker. The interactive allocation survives worker failures.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import re
import subprocess
import time

from evaluate_baseline_checkpoint import REPO,RUNTIME
from resume_baseline import write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-id',required=True)
    parser.add_argument('--resume-from',type=Path,required=True)
    parser.add_argument('--minutes',type=int,required=True)
    args=parser.parse_args()
    if not re.fullmatch(r'\d+',args.job_id) or not 1<=args.minutes<=240:
        raise ValueError('Numeric existing allocation ID and bounded duration required')
    output=RUNTIME/'logs'/f'arm-interactive-{args.job_id}-autostart.json'
    if output.exists(): raise ValueError('An automatic starter already exists for this allocation')
    command=['srun','--overlap',f'--jobid={args.job_id}','--nodes=1','--ntasks=1',
        '--cpus-per-task=32','--gres=gpu:h200:4','--cpu-bind=none','--exact',
        str(RUNTIME/'venv/bin/python'),str(REPO/'scripts/resume_arm_turn_bonus.py'),
        '--job-id',args.job_id,'--resume-from',str(args.resume_from.resolve()),
        '--minutes',str(args.minutes),'--execute']
    def record(stage,**values):
        report=dict(job_id=args.job_id,stage=stage,checked_utc=datetime.now(timezone.utc).isoformat(),
            command=command,new_allocation_requested=False,**values)
        write_json(output,report);print(json.dumps(report),flush=True)
    record('waiting-for-allocation')
    while True:
        info=subprocess.check_output(['scontrol','show','job',args.job_id,'-o'],text=True,timeout=20)
        match=re.search(r'\bJobState=(\S+)',info)
        if not match: raise ValueError('No scheduler state for the specified allocation')
        state=match[1]
        if state=='RUNNING': break
        if state not in ('PENDING','CONFIGURING'):
            record('allocation-unavailable',scheduler_state=state);return
        time.sleep(30)
    record('worker-running')
    with (RUNTIME/'logs'/f'arm-interactive-{args.job_id}-worker.log').open('x') as log:
        result=subprocess.run(command,cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
    record('worker-finished' if result.returncode==0 else 'worker-failed',exit_code=result.returncode,
        note='Interactive allocation remains available until its deadline; inspect artifacts before any retry.')


if __name__=='__main__': main()
