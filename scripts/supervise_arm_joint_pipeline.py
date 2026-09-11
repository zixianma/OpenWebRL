#!/usr/bin/env python3
"""Keep a batch allocation owned through its pipeline and a bounded repair window."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from train_arm_joint_sft import REPO,write_json
from resume_arm_c2_training import allocation_deadline,parse_job_record


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True,type=Path)
    a=p.parse_args();config=json.loads(a.config.read_text());root=Path(config['output'])
    root.mkdir(parents=True,exist_ok=True);job=os.environ.get('SLURM_JOB_ID')
    if not job or f'/job_{job}/' not in Path('/proc/self/cgroup').read_text():
        raise ValueError('Supervisor requires the approved Slurm allocation')
    record=parse_job_record(subprocess.check_output(['scontrol','show','job',job,'-o'],text=True,timeout=20))
    deadline=allocation_deadline(record,margin_minutes=2).timestamp()
    stopping=False;child=None
    def stop(*_):
        nonlocal stopping
        stopping=True
        if child is not None and child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        while not stopping and time.time()<deadline:
            with (root/'pipeline.log').open('a') as log:
                child=subprocess.Popen([sys.executable,str(REPO/'scripts/run_arm_joint_pipeline.py'),
                      '--objective',config['objective'],'--config',str(a.config)],cwd=REPO,
                      stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                while child.poll() is None:
                    if time.time()>=deadline:stop()
                    status_path=root/'training-status.json'
                    training=json.loads(status_path.read_text()) if status_path.exists() else {}
                    write_json(root/'supervisor-status.json',dict(phase='running',job=job,
                        checked_utc=datetime.now(timezone.utc).isoformat(),child_pid=child.pid,training=training))
                    time.sleep(15)
                code=child.wait()
            if code==0 and (root/'complete.json').exists():
                write_json(root/'supervisor-status.json',dict(phase='complete',job=job));return
            if stopping:break
            until=min(time.time()+900,deadline)
            write_json(root/'supervisor-status.json',dict(phase='awaiting_repair',returncode=code,
                       repair_until_utc=datetime.fromtimestamp(until,timezone.utc).isoformat(),job=job))
            request=root/'retry-controller.request'
            while time.time()<until and not stopping:
                if request.exists():
                    request.rename(root/f'retry-controller.consumed-{int(time.time())}');break
                time.sleep(15)
            else:raise RuntimeError('Pipeline failed; bounded repair window expired')
        raise SystemExit(1)
    finally:
        stop()
        if child is not None and child.poll() is None:
            try:child.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);child.wait()


if __name__=='__main__':main()
