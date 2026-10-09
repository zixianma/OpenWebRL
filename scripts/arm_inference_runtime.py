"""Shared approved-allocation checks extracted unchanged from validated controllers."""
from datetime import datetime
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
from openwebrl.controlled_sft_state import digest
from resume_baseline import clean_environment
REPO=Path('/gpfs/projects/krishna/zixianma/OpenWebRL')
RUNTIME=Path('/gpfs/scrubbed/zixianma/openwebrl-runtime')
FINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','BOOT_FAIL','DEADLINE','REVOKED','PREEMPTED'}


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with tmp.open('w') as f: json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    tmp.replace(path)

def check_approval(plan,approval,job,prior_seconds):
    if (approval.get('approved') is not True or approval.get('plan_sha256')!=digest(plan)
        or approval.get('resources')!=plan['resources'] or approval.get('limits')!=plan['limits']):
        raise ValueError('Exact new resources and API/browser caps have not been approved')
    attempts=approval.get('attempts',[])
    if len({str(a['job_id']) for a in attempts})!=len(attempts): raise ValueError('Duplicate scheduler attempt')
    current=[a for a in attempts if str(a['job_id'])==str(job)]
    if len(current)!=1: raise ValueError('Current allocation is not registered')
    seconds=current[0]['time_limit_seconds']
    if type(seconds) is not int or seconds<=0 or prior_seconds<0 or prior_seconds+seconds>plan['resources']['total_seconds']:
        raise ValueError('All-attempt scheduler cap exceeded; retries never reset it')
    return seconds

def scheduler_accounting(approval,job):
    total=0;rows=[]
    for attempt in approval.get('attempts',[]):
        old=str(attempt['job_id'])
        if old==str(job): continue
        raw=subprocess.check_output(['sacct','-X','-n','-P','-j',old,'--format=JobIDRaw,State,ElapsedRaw,AllocTRES'],text=True)
        matches=[r.split('|') for r in raw.splitlines() if r.split('|')[0]==old]
        if len(matches)!=1 or matches[0][1].split()[0] not in FINAL:
            raise ValueError('Every earlier allocation needs final scheduler accounting')
        elapsed=int(matches[0][2])
        if elapsed<0: raise ValueError('Negative scheduler time')
        total+=elapsed;rows.append(dict(job_id=old,elapsed_seconds=elapsed,sacct=raw))
    return total,rows

def stop_owned(children,job,grace=60):
    records=[]
    for p in children:
        members=process_group_members(p.pid,job)
        records.append(dict(pgid=p.pid,original_live_group_members=members))
        if members:
            try:os.killpg(p.pid,signal.SIGTERM)
            except ProcessLookupError:pass
    end=time.monotonic()+grace
    while time.monotonic()<end and any(process_group_members(p.pid,job) for p in children):
        time.sleep(.1)
    for p in children:
        if process_group_members(p.pid,job):
            try:os.killpg(p.pid,signal.SIGKILL)
            except ProcessLookupError:pass
        p.wait(timeout=10)
    end=time.monotonic()+10
    while time.monotonic()<end and any(process_group_members(p.pid,job) for p in children):time.sleep(.1)
    for record in records:
        record['remaining_live_group_members']=process_group_members(record['pgid'],job)
        record['process_gone']=not record['remaining_live_group_members']
    return records

def process_group_members(pgid,job,proc_root=Path('/proc')):
    members=[]
    for directory in proc_root.iterdir():
        if not directory.name.isdigit():continue
        try:
            parts=(directory/'stat').read_text().rsplit(')',1)[1].split()
            if int(parts[2])!=pgid:continue
            if directory.stat().st_uid!=os.getuid() or f'/job_{job}/' not in (directory/'cgroup').read_text():
                raise ValueError('Process group contains an unowned or different-allocation process')
            if parts[0]!='Z':members.append(int(directory.name))
        except (FileNotFoundError,ProcessLookupError):continue
    return members

def allocation_deadline(info, job, seconds, resources):
    fields = dict(re.findall(r'(\w+)=(\S+)', info))
    tres = dict(x.split('=',1) for x in fields.get('AllocTRES','').split(',') if '=' in x)
    if (fields.get('JobId') != str(job) or fields.get('JobState') != 'RUNNING'
            or not fields.get('UserId','').endswith(f'({os.getuid()})') or fields.get('NumNodes') != '1'
            or fields.get('NumCPUs') != str(resources['cpus'])
            or tres.get('gres/gpu:h200') != str(resources['gpus'])
            or tres.get('mem') not in (f"{resources['memory_gib']}G", f"{resources['memory_gib']*1024}M")):
        raise ValueError('Allocation differs from exact approved profile')
    value = fields['TimeLimit']; days, clock = value.split('-',1) if '-' in value else ('0',value)
    parts = [int(x) for x in clock.split(':')]
    if len(parts) == 2: parts.insert(0,0)
    duration = int(days)*86400 + parts[0]*3600 + parts[1]*60 + parts[2]
    if duration > seconds: raise ValueError('Allocation time exceeds remaining approved cap')
    end = datetime.strptime(fields['EndTime'], '%Y-%m-%dT%H:%M:%S').timestamp()
    return min(time.time()+duration, end)-180

def environment():
    from dotenv import dotenv_values
    env = {k:v for k,v in clean_environment().items() if not k.startswith('OPENWEBRL_ARM_')}
    env.setdefault('FLASHINFER_WORKSPACE_BASE',str(RUNTIME))
    for k,v in dotenv_values(REPO/'.env').items():
        if v and k.startswith(('OPENAI_', 'JUDGE_', 'WANDB_', 'AZURE_')):
            env.setdefault(k,v)
    if env.get('OPENAI_API_KEY') and not env.get('JUDGE_API_BASE'):
        env.update(JUDGE_API_BASE='https://api.openai.com/v1', JUDGE_API_MODE='served')
    return env
