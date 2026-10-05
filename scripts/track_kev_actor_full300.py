#!/usr/bin/env python3
"""Sync aggregate metrics for a direct Kev eval; never upload task payloads."""
import argparse
import json
import os
from pathlib import Path
import signal
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--env-file',type=Path,required=True);args=p.parse_args()
    from dotenv import dotenv_values
    import wandb
    values=dotenv_values(args.env_file)
    if values.get('WANDB_API_KEY'):os.environ['WANDB_API_KEY']=values['WANDB_API_KEY']
    os.environ.update(WANDB_PROJECT='openwebrl-evals',WANDB_MODE='online',
        WANDB_CACHE_DIR=str(args.root/'wandb-cache'),WANDB_DATA_DIR=str(args.root/'wandb-data'))
    plan=json.loads((args.root/'plan.json').read_text())
    protocol=plan['protocol']
    config={k:protocol[k] for k in ('kev','harness_revision','text_model','text_sampling','text_max_tokens',
        'judge_model','judge_prompt_variant','judge_max_completion_tokens','max_steps','max_decisions',
        'task_timeout_seconds','viewport','actor_input','actor_sampling')}
    config.update(tasks=300,role='standalone browser policy',candidate_proposer=None,
        resources=plan['resources'],limits=plan['limits'])
    run=wandb.init(entity='zixianma',project='openwebrl-evals',id=plan['wandb_id'],resume='allow',
        name='Kev27B actor full300',group='direct-policy-full300-20261004',config=config,dir=str(args.root))
    ready=args.root/'wandb-ready.json';tmp=ready.with_suffix('.partial')
    tmp.write_text(json.dumps(dict(job_id=os.environ.get('SLURM_JOB_ID'),url=run.url,online=True,unix=time.time()))+'\n');tmp.replace(ready)
    stop=False;last=None;deadline=0
    def finish(*_):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGTERM,finish);signal.signal(signal.SIGINT,finish)
    while True:
        if stop or time.monotonic()>=deadline:
            path=args.root/'actor/summary.json'
            if path.exists():
                summary=json.loads(path.read_text())
                metrics={k:v for k,v in summary.items() if isinstance(v,(int,float,bool))}
                metrics.update({'api_attempts/'+k:v for k,v in summary.get('api_attempts',{}).items()})
                if metrics!=last:run.log(metrics);last=metrics
            deadline=time.monotonic()+30
        if stop:break
        time.sleep(1)
    run.finish()
    (args.root/'wandb-finished.json').write_text(json.dumps(dict(unix=time.time(),job_id=os.environ.get('SLURM_JOB_ID'),synced=True))+'\n')


if __name__=='__main__':main()
