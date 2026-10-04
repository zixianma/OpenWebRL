#!/usr/bin/env python3
"""Release exactly one approved, held evaluation when its checkpoint is durable.

Runs as a lightweight CPU watcher, never allocates GPUs or submits another job.
The released Slurm controller owns and awaits the complete evaluation worker.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import time

from run_arm_iteration80_eval import CONTROL, RUNTIME, VARIANTS, checkpoint_ready, prepare_source
from resume_baseline import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--held-job', required=True)
    parser.add_argument('--training-job', default='309490')
    args = parser.parse_args()
    if not args.held_job.isdigit() or args.training_job != '309490':
        raise ValueError('This watcher is restricted to the declared iteration-80 continuation')
    owner = CONTROL/'allfailure-watcher-owner'
    owner.mkdir(exist_ok=True)
    lock = (owner/'lock').open('a+')
    # A kernel lock survives normal tool/session boundaries and is released on
    # process exit, so a dead watcher cannot leave an unrecoverable owner marker.
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    write_json(owner/'pid.json', dict(pid=os.getpid(), held_job=args.held_job))
    target = CONTROL/'allfailure-watcher-status.json'
    started = time.time()
    while time.time()-started < 7*24*3600:
        record = subprocess.check_output(['scontrol','show','job',args.held_job,'-o'],text=True)
        fields = dict(re.findall(r'(\w+)=(\S+)',record))
        if fields.get('JobState') != 'PENDING' or fields.get('Reason') != 'JobHeldUser':
            write_json(target, dict(stage='no_longer_held',job=args.held_job,state=fields.get('JobState'),
                                   reason=fields.get('Reason'),updated_epoch=time.time()))
            return
        checkpoint = checkpoint_ready('allfailure')
        if checkpoint is not None:
            prepare_source('allfailure')
            receipt = CONTROL/'submission-allfailure.json'
            submitted = json.loads(receipt.read_text())
            if submitted['job_id'] != args.held_job or not submitted['compute_approved']:
                raise ValueError('Held evaluation does not match its approval/submission receipt')
            subprocess.run(['scontrol','release',args.held_job],check=True)
            write_json(target, dict(stage='released',job=args.held_job,checkpoint=str(checkpoint),
                                   updated_epoch=time.time()))
            return
        root = RUNTIME/'evaluations'/VARIANTS['allfailure'][1]
        status = json.loads((root/'status.json').read_text()) if (root/'status.json').is_file() else {}
        train_state = subprocess.check_output(['squeue','--noheader','--jobs',args.training_job,'--format=%T'],text=True).strip()
        if status.get('failed') or status.get('stage') == 'complete' or not train_state:
            write_json(target, dict(stage='checkpoint_unavailable',job=args.held_job,training_state=train_state,
                training_status=status,updated_epoch=time.time(),action='Held evaluation retained; requires supervision'))
            return
        write_json(target, dict(stage='waiting_for_checkpoint',job=args.held_job,training_job=args.training_job,
                               training_state=train_state,updated_epoch=time.time()))
        time.sleep(300)
    write_json(target, dict(stage='watcher_expired',job=args.held_job,updated_epoch=time.time()))


if __name__ == '__main__':
    main()
