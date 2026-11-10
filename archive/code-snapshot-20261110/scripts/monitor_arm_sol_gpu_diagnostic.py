#!/usr/bin/env python3
"""Read-only diagnostic monitoring: stage changes and 15-minute metric snapshots."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import time
import monitor_arm_turn_bonus as monitor
from resume_baseline import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job-id',required=True); p.add_argument('--watch',action='store_true')
    p.add_argument('--attempt',type=int,default=0)
    p.add_argument('--stop-on-training-end',action='store_true')
    a=p.parse_args(); job=a.job_id
    root=monitor.RUNTIME/f'evaluations/arm-sol-gpu-diagnostic-{job}'
    train=Path(str(root)+'-training'); infer=Path(str(root)+'-inference')
    if a.attempt: train=Path(str(train)+f'-r{a.attempt}')
    monitor.run_root=lambda _:train
    output=monitor.RUNTIME/f'arm-turn-bonus-preparation/monitor-{job}'
    if a.attempt: output=output/f'attempt-{a.attempt}'
    output.mkdir(parents=True,exist_ok=True)
    end=time.monotonic()+24*3600; next_full=0; next_scheduler=0; last=None; slurm={}
    while time.monotonic()<end:
        now=time.monotonic()
        if now>=next_scheduler:
            slurm=monitor.scheduler(job)
            next_scheduler=now+(60 if slurm['state'] in ('PENDING','CONFIGURING','UNKNOWN') else 180)
        states={k:monitor.read(path/'status.json') for k,path in [('job',root),('training',train),('inference',infer)]}
        key=(slurm.get('state'),tuple((k,v.get('stage'),v.get('failed'),v.get('passed'),v.get('iteration'),
            v.get('completed_iterations')) for k,v in states.items()))
        if key!=last or now>=next_full:
            report=dict(checked_utc=datetime.now(timezone.utc).isoformat(),slurm=slurm,states=states,
                training=monitor.snapshot(job,slurm,remote=False,compare_baseline=False),api=monitor.read(infer/'sol-api/usage.json'),
                evaluation=monitor.read(infer/'comparison.json'))
            if states['training']:
                report['wandb']=monitor.wandb_sample(job)
            active=states.get(states['job'].get('stage'),{}) or states['training']
            if slurm.get('state')=='RUNNING' and active and active.get('stage') not in ('complete',):
                report['resources']=monitor.resource_sample(job,gpus=2)
            write_json(output/'latest.json',report)
            with (output/'history.jsonl').open('a') as f: f.write(json.dumps(report)+'\n')
            print(json.dumps(dict(slurm=slurm,states=states,
                latest_train=report['training'].get('latest_train'),
                alerts=report['training'].get('alerts'),resources=report.get('resources'))),flush=True)
            last=key; next_full=now+900
        if slurm.get('state','').split()[0].rstrip('+') in monitor.TERMINAL or not a.watch: break
        if a.stop_on_training_end and (states['training'].get('training_complete')
                or states['training'].get('shadow_complete') or states['training'].get('failed')): break
        time.sleep(30)


if __name__=='__main__': main()
