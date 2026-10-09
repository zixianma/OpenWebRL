#!/usr/bin/env python3
"""Observe one authorized job through training/evaluation handoffs; no mutations."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time


def read(path):
    try:return json.loads(path.read_text())
    except (OSError,json.JSONDecodeError):return {}


def write(path,body):
    temp=path.with_suffix('.partial')
    temp.write_text(json.dumps(body,indent=2)+'\n');temp.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-id',required=True)
    parser.add_argument('--controller-root',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--hours',type=float,default=16,help='0 follows the job until terminal, including queue time')
    parser.add_argument('--interval',type=float,default=30)
    args=parser.parse_args()
    if not args.job_id.isdigit():raise ValueError('Numeric job ID required')
    if args.hours<0 or args.interval<10:raise ValueError('Invalid observation limits')
    args.output.mkdir(parents=True,exist_ok=True)
    deadline=time.monotonic()+args.hours*3600 if args.hours else float('inf')
    last=None;reported=0;terminal_checks=0
    while time.monotonic()<deadline:
        try:
            raw=subprocess.check_output(['squeue','-h','-j',args.job_id,'-o','%T|%R|%M|%L'],
                text=True,timeout=20).strip()
            active=bool(raw)
            if not active:
                raw=subprocess.check_output(['sacct','-X','-n','-P','-j',args.job_id,
                    '--format=JobID,State,Elapsed,ExitCode'],text=True,timeout=20).strip()
            controller=read(args.controller_root/'status.json')
            plan=read(args.controller_root/'controller-plan.json')
            root=Path(plan['training_root']) if plan.get('training_root') else None
            training=read(root/'status.json') if root else {}
            health=read(root/'browser-startup-health.json') if root else {}
            evaluations={}
            for path in args.controller_root.parent.glob(f'*-iter*-{args.job_id}'):
                if not (path/'evaluation_manifest.json').exists():continue
                evaluations[path.name]=dict(status=read(path/'status.json'),
                    rollouts=len(list((path/'rollouts').glob('*.pt'))),
                    verdicts=len(list((path/'rollouts').glob('*.json'))))
            report=dict(checked_utc=datetime.now(timezone.utc).isoformat(),job_id=args.job_id,
                scheduler=raw,active=active,controller=controller,training=training,
                browser_startup_health=health,evaluations=evaluations,
                monitor='read-only observation; controller owns workers and health stops')
            write(args.output/'latest.json',report)
            key=(active,controller.get('stage'),controller.get('current_target',controller.get('target')),
                 training.get('iteration'),training.get('completed_iterations'),training.get('failed'),
                 training.get('stage'))
            if key!=last or time.monotonic()-reported>=900:
                with (args.output/'history.jsonl').open('a') as out:out.write(json.dumps(report)+'\n')
                print(json.dumps(dict(job_id=args.job_id,scheduler=raw,stage=controller.get('stage'),
                    training_stage=training.get('stage'),iteration=training.get('iteration'),
                    failed=training.get('failed',False))),flush=True)
                last=key;reported=time.monotonic()
            terminal_states={'COMPLETED','FAILED','CANCELLED','TIMEOUT','NODE_FAIL',
                             'OUT_OF_MEMORY','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}
            accounting_states=[row.split('|')[1].split()[0] for row in raw.splitlines()
                               if row.startswith(args.job_id+'|')] if not active else []
            terminal_checks=terminal_checks+1 if accounting_states and all(
                state in terminal_states for state in accounting_states) else 0
            if terminal_checks>=2:break
        except (OSError,subprocess.SubprocessError) as exc:
            write(args.output/'error.json',dict(epoch=time.time(),error_type=type(exc).__name__))
        time.sleep(args.interval)
    write(args.output/'finished.json',dict(epoch=time.time(),reason='terminal job or observer time limit'))


if __name__=='__main__':main()
