#!/usr/bin/env python3
"""Repair a stopped ARM supervisor inside its existing, already-paid allocation."""
import json
import os
from pathlib import Path
import signal
import time
import zipfile

from run_arm_turn_bonus_calibration import RUNTIME, execute, plan


def belongs(pid, job):
    p = Path('/proc')/str(pid)
    try:
        return f'/job_{job}/' in (p/'cgroup').read_text()
    except OSError:
        return False


def alive(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1][0] != 'Z'
    except OSError:
        return False


def main():
    job = os.environ['SLURM_JOB_ID']
    root = RUNTIME/f'evaluations/arm-turn-bonus-calibration-{job}'
    hold = json.loads((root/'supervisor-handoff-hold.json').read_text())
    supervisor = hold['held_supervisor_pid']
    if not belongs(os.getpid(),job) or not belongs(supervisor,job):
        raise ValueError('Existing-allocation process ownership mismatch')
    children = [int(x) for x in Path(f'/proc/{supervisor}/task/{supervisor}/children').read_text().split()]
    if not children or any(not belongs(pid,job) for pid in children):
        raise ValueError('Unexpected original supervisor children')
    worker = next((pid for pid in children if Path(f'/proc/{pid}/cmdline').read_bytes().startswith(b'timeout\0')),None)
    result = {'job_id':job, 'new_paid_allocation':False, 'first_stage':'ARM-exclusion bug; preserve as q=0 control'}
    try:
        print('Waiting for the original frozen-actor collection to save and exit.',flush=True)
        while worker is not None and alive(worker):
            time.sleep(10)
        recovery = root/'runtime/rollout_recovery/70.pt'
        with zipfile.ZipFile(recovery) as archive:
            if not any(n.endswith('/data.pkl') for n in archive.namelist()):
                raise ValueError('Original batch was not saved completely')
        print('Original batch saved; stopping its remaining services before the corrected collection.',flush=True)
        stopped = []
        for pid in children:
            if alive(pid) and belongs(pid,job):
                os.killpg(pid,signal.SIGTERM)
                stopped.append(pid)
        # A failed Ray driver can leave detached services. Scope cleanup to its
        # original Slurm step; leave the held supervisor and this repair step alone.
        for p in Path('/proc').glob('[0-9]*'):
            try:
                pid = int(p.name)
                cgroup = (p/'cgroup').read_text()
                cmd = (p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
                if pid == supervisor or f'/job_{job}/step_0/' not in cgroup:
                    continue
                if any(x in cmd for x in ['raylet','gcs_server','ray::','ray/dashboard','ray/_private','sglang','wandb-core','serve_arm.py']):
                    os.kill(pid,signal.SIGTERM)
                    stopped.append(pid)
            except (OSError,ProcessLookupError):
                pass
        result['stopped_original_service_pids'] = sorted(set(stopped))
        time.sleep(10)
        print('Starting corrected ARM collection in the remaining allocation; project=openwebrl.',flush=True)
        execute(plan(job,attempt=2))
        result['corrected_collection_complete'] = True
    except BaseException as exc:
        result.update(corrected_collection_complete=False,error=type(exc).__name__+': '+str(exc)[:300])
        raise
    finally:
        (root/'repair-handoff-status.json').write_text(json.dumps(result,indent=2)+'\n')
        if belongs(supervisor,job):
            os.kill(supervisor,signal.SIGCONT)


if __name__ == '__main__':
    main()
