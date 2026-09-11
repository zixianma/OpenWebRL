#!/usr/bin/env python3
"""Recover the already-approved DPO allocation while its batch owner is paused.

Preserve immutable failed-run files, reuse only verified frozen-reference scores,
own the replacement controller to completion, then release the original owner.
Never submits or extends a job.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from train_arm_joint_sft import REPO,digest,write_json


def identity(pid,job,fragment):
    proc=Path(f'/proc/{pid}')
    if not proc.exists():return False
    if (proc/'stat').read_text().rsplit(')',1)[1].split()[0]=='Z':return False
    cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
    if fragment not in cmd or f'/job_{job}/' not in (proc/'cgroup').read_text():
        raise ValueError(f'PID identity mismatch: {pid}')
    return True


def main():
    p=argparse.ArgumentParser();p.add_argument('--job-id',required=True)
    p.add_argument('--parent-pid',required=True,type=int);p.add_argument('--old-trainer-pid',required=True,type=int)
    p.add_argument('--old-worker-pids',required=True,nargs=2,type=int)
    p.add_argument('--old-config',required=True,type=Path);p.add_argument('--new-config',required=True,type=Path)
    a=p.parse_args()
    if os.environ.get('SLURM_JOB_ID')!=a.job_id:raise ValueError('Outside authorized allocation')
    if not identity(a.parent_pid,a.job_id,'run_arm_joint_pipeline.py'):raise ValueError('Batch owner is gone')
    old=json.loads(a.old_config.read_text());new=json.loads(a.new_config.read_text())
    old_hash=digest(a.old_config);new_hash=digest(a.new_config)
    root=Path(new['output']);root.mkdir(parents=True,exist_ok=True)
    for field in ['actor','model_metadata_hashes','data_hashes','order_sha256','objective','beta']:
        if old[field]!=new[field]:raise ValueError(f'Cannot reuse reference cache across changed {field}')
    if new['reference_cache_parent_config_sha256']!=old_hash:raise ValueError('Wrong predecessor config')
    write_json(root/'predecessor-config.json',old)
    parent_fragment='run_arm_joint_pipeline.py'
    try:
        for pid in a.old_worker_pids:
            if identity(pid,a.job_id,'train_arm_joint_ddp.py'):os.kill(pid,signal.SIGTERM)
        start=time.monotonic()
        while identity(a.old_trainer_pid,a.job_id,'torch.distributed.run'):
            if time.monotonic()-start>150:raise RuntimeError('Old trainer did not finish graceful shutdown')
            time.sleep(2)
        old_root=Path(old['output']);state_path=old_root/'student/latest-checkpoint.json'
        if state_path.exists() and json.loads(state_path.read_text())['updates']!=0:
            raise ValueError('Recovery would discard actual optimizer updates')
        provenance=[]
        for rank in range(2):
            source=old_root/f'reference-rank-{rank}.jsonl';dest=root/source.name
            if dest.exists():raise ValueError('Refuse to overwrite migrated cache')
            count=0
            with source.open() as incoming,dest.open('x') as outgoing:
                for line in incoming:
                    item=json.loads(line)
                    if item['config_sha256']!=old_hash:raise ValueError('Old reference cache provenance mismatch')
                    item['config_sha256']=new_hash
                    outgoing.write(json.dumps(item,separators=(',',':'))+'\n');count+=1
            provenance.append(dict(source=str(source),source_sha256=digest(source),destination=str(dest),
                                   destination_sha256=digest(dest),cached_pairs=count))
        write_json(root/'cache-migration.json',dict(reason='Fix deterministic backward mode; base/reference/data unchanged',
                 old_config_sha256=old_hash,new_config_sha256=new_hash,caches=provenance))
        while True:
            write_json(root/'recovery-status.json',dict(phase='running_replacement',job=a.job_id,parent_pid=a.parent_pid))
            with (root/'recovery-controller.log').open('a') as log:
                code=subprocess.call([sys.executable,str(REPO/'scripts/run_arm_joint_pipeline.py'),
                    '--objective','dpo','--config',str(a.new_config)],cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
            if code==0 and (root/'complete.json').exists():break
            # Keep the already-paid allocation available briefly for an observed
            # repair; no automatic objective changes and no allocation extension.
            write_json(root/'recovery-status.json',dict(phase='awaiting_repair',returncode=code,
                      retry_sentinel=str(root/'retry-controller.request'),grace_seconds=900))
            until=time.monotonic()+900
            while time.monotonic()<until:
                request=root/'retry-controller.request'
                if request.exists():
                    request.rename(root/f'retry-controller.consumed-{int(time.time())}');break
                time.sleep(15)
            else:raise RuntimeError('Repair grace expired; preserving run artifacts')
        write_json(root/'recovery-status.json',dict(phase='complete',job=a.job_id))
    finally:
        # No batch-owner exit can cancel the replacement before its work ends.
        if identity(a.parent_pid,a.job_id,parent_fragment):os.kill(a.parent_pid,signal.SIGCONT)


if __name__=='__main__':main()
