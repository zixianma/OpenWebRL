#!/usr/bin/env python3
"""Read-only allocation, inference progress and API usage monitoring."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import time
from monitor_arm_turn_bonus import RUNTIME,read,scheduler,resource_sample,TERMINAL
from resume_baseline import write_json


def main():
    p=argparse.ArgumentParser(); p.add_argument('--job-id',required=True)
    p.add_argument('--watch',action='store_true'); p.add_argument('--hours',type=float,default=5)
    a=p.parse_args(); root=RUNTIME/f'evaluations/sol-selection300-{a.job_id}'
    output=RUNTIME/f'arm-turn-bonus-preparation/monitor-{a.job_id}'; output.mkdir(parents=True,exist_ok=True)
    end=time.monotonic()+a.hours*3600; next_full=0; next_scheduler=0; old=None; slurm={}
    while time.monotonic()<end:
        if time.monotonic()>=next_scheduler:
            slurm=scheduler(a.job_id); next_scheduler=time.monotonic()+(60 if slurm['state']=='PENDING' else 300)
        state=read(root/'status.json'); key=(slurm['state'],state.get('stage'),state.get('failed'))
        if key!=old or time.monotonic()>=next_full:
            rows=[json.loads(path.read_text()) for path in root.glob('*/results/*.json')]
            valid=[r for r in rows if r.get('valid')]; wins=sum(r.get('reward')==1 for r in valid)
            report=dict(checked_utc=datetime.now(timezone.utc).isoformat(),slurm=slurm,status=state,
                completed=len(rows),valid=len(valid),successes=wins,overall_so_far=wins/300,
                valid_only_so_far=wins/len(valid) if valid else None,api_usage=read(root/'sol-api/usage.json'),alerts=[])
            if state.get('failed'): report['alerts'].append('Inference controller failed; inspect retained results immediately')
            if slurm['state']=='RUNNING' and state: report['resources']=resource_sample(a.job_id,2)
            write_json(output/'latest.json',report)
            with (output/'history.jsonl').open('a') as f: f.write(json.dumps(report)+'\n')
            print(json.dumps(report),flush=True); old=key; next_full=time.monotonic()+900
        if slurm['state'].split()[0].rstrip('+') in TERMINAL or not a.watch: break
        time.sleep(30)


if __name__=='__main__': main()
