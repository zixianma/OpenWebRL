#!/usr/bin/env python3
"""Release one approved held B/C evaluation after its iteration-20 save.

No GPU allocation or new Slurm submission is made by this watcher.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import time

from run_arm_gate_checkpoint_eval import checkpoint_ready, plan
from resume_baseline import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text())
    if (receipt.get('compute_approved') is not True or receipt['variant'] not in ('B', 'C')
            or not str(receipt['job_id']).isdigit()
            or receipt['resources'] != dict(gpus=2, hours=1, cpus=16, memory_gib=480)
            or receipt['completed_iteration'] != 20):
        raise ValueError('Requires the approved B/C iteration-20 submission receipt')
    job, variant = str(receipt['job_id']), receipt['variant']
    root = Path(receipt['training_root'])
    control = args.receipt.parent
    lock = (control/f'{variant}-watcher.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    write_json(control/f'{variant}-watcher-process.json', dict(pid=os.getpid(), job=job))
    status = control/f'{variant}-watcher-status.json'
    deadline = time.monotonic() + 7*24*3600
    while time.monotonic() < deadline:
        raw = subprocess.check_output(['scontrol', 'show', 'job', job, '-o'], text=True, timeout=30)
        fields = dict(re.findall(r'(\w+)=(\S+)', raw))
        if fields.get('JobState') != 'PENDING' or fields.get('Reason') != 'JobHeldUser':
            write_json(status, dict(stage='no_longer_held', job=job, state=fields.get('JobState'),
                                    updated_epoch=time.time()))
            return
        if checkpoint_ready(variant, root, 20) is not None:
            prepared = plan(variant, root, job, 20)
            write_json(control/f'{variant}-released-plan.json', prepared)
            subprocess.run(['scontrol', 'release', job], check=True, timeout=30)
            write_json(status, dict(stage='released', job=job, checkpoint=prepared['checkpoint'],
                                    updated_epoch=time.time()))
            return
        training = json.loads((root/'status.json').read_text()) if (root/'status.json').is_file() else {}
        stopped = training.get('stage') == 'complete' or training.get('failed')
        write_json(status, dict(stage='checkpoint_unavailable' if stopped else 'waiting_for_checkpoint',
                                job=job, training_root=str(root), updated_epoch=time.time()))
        if stopped:
            return  # Keep the evaluation held; never allocate GPUs without its checkpoint.
        time.sleep(300)
    write_json(status, dict(stage='expired', job=job, updated_epoch=time.time()))


if __name__ == '__main__':
    main()
