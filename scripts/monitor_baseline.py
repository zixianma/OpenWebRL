"""Read-only baseline monitor: record process progress, GPU and allocation memory.

Does not submit allocations, restart runs, or stop processes. Exits after the
run's exit_status appears or the supplied monitoring duration expires.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('run',type=Path)
p.add_argument('--seconds',type=int,default=14400)
a=p.parse_args()
cgroup=Path('/sys/fs/cgroup')/Path('/proc/self/cgroup').read_text().strip().split('::',1)[1].lstrip('/')
while not (cgroup/'memory.max').exists() and cgroup.parent != cgroup:
    cgroup=cgroup.parent
# Slurm's budget is on the job ancestor, not always the leaf task group.
for candidate in [cgroup,*cgroup.parents]:
    if candidate.name.startswith('job_'):
        cgroup=candidate
        break
end=time.monotonic()+a.seconds
with (a.run/'health.jsonl').open('a',buffering=1) as out:
    while time.monotonic()<end:
        progress=a.run/'progress.log'
        lines=progress.read_text(errors='replace').splitlines() if progress.exists() else []
        row={'utc':datetime.now(timezone.utc).isoformat(),'progress':lines[-1] if lines else 'initializing'}
        for key in ['memory.current','memory.peak','memory.events']:
            f=cgroup/key
            if f.exists(): row[key]=f.read_text().strip()
        result=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
        row['gpu']=result.stdout.strip().splitlines()
        status=a.run/'exit_status.json'
        if status.exists(): row['exit_status']=json.loads(status.read_text())
        out.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
        if status.exists(): break
        time.sleep(30)
