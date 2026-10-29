#!/usr/bin/env python3
"""Inner PID-namespace bootstrap: private procfs, confined reads, then Chrome exec."""
import argparse
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text())
    actual=os.readlink('/proc/self/ns/pid')
    mounts=Path('/proc/self/mountinfo').read_text().splitlines()
    proc=[line for line in mounts if line.split()[4]=='/proc' and ' - proc ' in line]
    if os.getpid()!=1 or os.getppid()!=0 or actual==cfg['expected_host_pid_namespace'] or not proc:
        raise RuntimeError('Physical browser requires independently mounted private PID namespace')
    visible=sorted(int(p.name) for p in Path('/proc').iterdir() if p.name.isdigit())
    if visible!=[1]:
        raise RuntimeError('Private procfs unexpectedly exposes other processes before Chrome exec')
    proof={'time':time.time(),'reservation_id':cfg['reservation_id'],'private_pid':os.getpid(),'private_ppid':os.getppid(),'pid_namespace_expected_host':cfg['expected_host_pid_namespace'],'pid_namespace_actual':actual,'private_proc_mount':proc[-1],'proc_mount_type':'proc','visible_pids_before_exec':visible,'physical_chromium':cfg['physical_chromium'],'argv':cfg['argv'],'read_scope':True,'actor_credentials_inherited':False}
    Path(cfg['namespace_proof']).write_text(json.dumps(proof,indent=2)+'\n')
    from confinement import restrict
    restrict(cfg['output'],cfg['private_temp'],cfg['physical_chromium'],read_scope=True,private_proc=True)
    os.execve(cfg['physical_chromium'],cfg['argv'],dict(os.environ))


if __name__=='__main__':
    raise SystemExit(main())
