#!/usr/bin/env python3
"""Recover the zero-update action-DPO run inside its existing allocation."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def process_identity(pid, job, fragment):
    root = Path(f'/proc/{pid}')
    if not root.exists():
        return False
    state = (root / 'stat').read_text().rsplit(')', 1)[1].split()[0]
    if state == 'Z':
        return False
    command = (root / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
    if fragment not in command or f'/job_{job}/' not in (root / 'cgroup').read_text():
        raise ValueError(f'Process identity mismatch for {pid}')
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-id', required=True)
    parser.add_argument('--batch-owner-pid', required=True, type=int)
    parser.add_argument('--old-trainer-pid', required=True, type=int)
    parser.add_argument('--config', required=True, type=Path)
    args = parser.parse_args()
    if os.environ.get('SLURM_JOB_ID') != args.job_id:
        raise ValueError('Recovery must run inside the approved allocation')
    if not process_identity(args.batch_owner_pid, args.job_id, 'run_arm_joint_pipeline.py'):
        raise ValueError('Batch owner is no longer alive')
    config = json.loads(args.config.read_text())
    root = Path(config['output'])
    if config['objective'] != 'dpo_action' or (root / 'student/metrics.jsonl').exists():
        raise ValueError('Recovery target is not a fresh action-DPO run')
    old_status = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/runs/joint-v2-dpo-action-2gpu/training-status.json')
    status = json.loads(old_status.read_text())
    if status.get('updates') != 0 or 'wandb.util' not in status.get('error', ''):
        raise ValueError('Unexpected predecessor failure')
    os.kill(args.batch_owner_pid, signal.SIGSTOP)
    try:
        if process_identity(args.old_trainer_pid, args.job_id, 'torch.distributed.run'):
            os.killpg(args.old_trainer_pid, signal.SIGTERM)
        deadline = time.monotonic() + 180
        while process_identity(args.old_trainer_pid, args.job_id, 'torch.distributed.run'):
            if time.monotonic() > deadline:
                os.killpg(args.old_trainer_pid, signal.SIGKILL)
                break
            time.sleep(2)
        with (root / 'recovery-controller.log').open('a') as log:
            code = subprocess.call([
                '/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/venv/bin/python',
                '/gpfs/projects/krishna/zixianma/OpenWebRL/scripts/run_arm_joint_pipeline.py',
                '--objective', 'dpo_action', '--config', str(args.config.resolve()),
            ], stdout=log, stderr=subprocess.STDOUT,
               cwd='/gpfs/projects/krishna/zixianma/OpenWebRL')
        if code or not (root / 'complete.json').exists():
            raise RuntimeError(f'Replacement pipeline failed with code {code}')
    finally:
        if process_identity(args.batch_owner_pid, args.job_id, 'run_arm_joint_pipeline.py'):
            os.kill(args.batch_owner_pid, signal.SIGCONT)


if __name__ == '__main__':
    main()
