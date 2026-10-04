#!/usr/bin/env python3
"""Own CPU readiness, TP8 restoration and training in the approved allocation."""
import os
import json
from pathlib import Path
import subprocess
import sys

repo=Path(__file__).resolve().parents[1]
job=os.environ['SLURM_JOB_ID']
gpus=os.environ.get('ARM_RESUME_GPUS','8')
# A shared atomic directory elects exactly one controller, even if both jobs
# start simultaneously. The loser exits before checks, model loading or W&B.
if os.environ.get('ARM_RESUME_RACE'):
    race=Path(os.environ['ARM_RESUME_RACE'])
    registered=json.loads((race/'jobs.json').read_text())
    if job not in registered['jobs']:
        raise ValueError('Job is not registered for this approved race')
    try:
        (race/'claimed').mkdir()
    except FileExistsError:
        print('Another allocation already claimed the continuation; exiting.',flush=True)
        sys.exit(0)
    (race/'claimed'/'winner.json').write_text(json.dumps({'job_id':job,'gpus':gpus}))
    for other in registered['jobs']:
        if other != job:
            subprocess.run(['scancel',other],check=True)
    print('Continuation claimed by '+job,flush=True)

receipt=Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-turn-bonus-preparation')/f'resume8-{job}'/'receipt.json'
os.environ['ARM_READINESS_FILE']=str(receipt)
subprocess.run([sys.executable,str(repo/'scripts/arm_launch_preflight.py'),
    '--check','--receipt',str(receipt)],check=True)
subprocess.run([sys.executable,str(repo/'scripts/check_arm_native_launch.py'),
    '--provider','arm','--resume-from',os.environ['ARM_RESUME_FROM'],
    '--resume-minutes',os.environ.get('ARM_TRAIN_MINUTES','480'),'--resume-gpus',gpus,
    '--output',str(receipt.parent/'native-resume')],check=True)
subprocess.run([sys.executable,str(repo/'scripts/resume_arm_turn_bonus.py'),
    '--execute','--job-id',job,'--gpus',gpus,'--resume-from',os.environ['ARM_RESUME_FROM'],
    '--minutes',os.environ.get('ARM_TRAIN_MINUTES','480')],check=True)
