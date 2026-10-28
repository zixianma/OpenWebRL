#!/usr/bin/env python3
"""Preview or submit the single exact approved Piotr WEBPRMBench allocation."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess

from openwebrl.webprm_native_release import reference
from scripts.run_piotr_webprmbench import (RESOURCES, LIMITS, allocation_seconds,
                                         check_bindings, safe_output, slurm_time, write_new)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--approval', required=True)
    ap.add_argument('--launch', action='store_true')
    args = ap.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    check_bindings(manifest)
    root = safe_output(manifest['root'])
    approval = json.loads(Path(args.approval).read_text())
    if (approval.get('status') != 'approved' or approval.get('job_id') is not None
            or approval.get('manifest_sha256') != reference(args.manifest)['sha256']
            or approval.get('resources') != RESOURCES or approval.get('limits') != LIMITS):
        raise ValueError('Exact user approval is required; a prepared resource request is not approval')
    seconds = allocation_seconds(approval)
    command = ['sbatch', '--hold', '--parsable', '--time=' + slurm_time(seconds),
               '--export=ALL,TMPDIR=/tmp', manifest['sbatch']]
    if not args.launch:
        print(json.dumps(dict(command=command, resources=RESOURCES, limits=LIMITS,
                              would_submit=False, manifest=reference(args.manifest)), indent=2))
        return
    with (root/'submission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (root/'submission.json').exists() or (root/'job-approval.json').exists() or list(root.glob('attempt-*')):
            raise ValueError('Single allocation already submitted/consumed; no duplicate submission')
        result = subprocess.check_output(command, text=True).strip()
        job = result.split(';')[0]
        if not job.isdigit():
            raise RuntimeError('Submission response could not be bound: ' + result)
        write_new(root/'submission.json', dict(job_id=job, command=command, response=result,
                  held_before_release=True, resources=RESOURCES, limits=LIMITS,
                  original_approval=reference(args.approval), manifest=reference(args.manifest)))
        write_new(root/'job-approval.json', dict(approval, job_id=job))
        subprocess.run(['scontrol', 'release', job], check=True)
        print(json.dumps(dict(submitted=True, job_id=job, released=True)))


if __name__ == '__main__':
    main()
