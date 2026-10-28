#!/usr/bin/env python3
"""Prepare an accounted WebPRM continuation; default is read-only dry run.

This helper never submits jobs or makes API calls. --prepare archives metadata
and writes a reviewable batch script. --begin is a one-time hook inside that
actual allocation; it revises only the operational deadline and archives cells
proven to be unattempted. Interrupted/ambiguous cells require separate review.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time

DEFAULT_ROOT=Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-baselines-20261009')
ARMS=('shepherd_unguided','shepherd_random5','shepherd8b',
      'arbiter_unguided','arbiter_random5','arbiter7b')
ENGINE='/gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python'
ENV_FILE='/gpfs/projects/krishna/zixianma/OpenWebRL/.env'
CUDA='/gpfs/scrubbed/zixianma/openwebrl-runtime/cuda'
TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL',
          'PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}
ACCOUNT_FIELDS=('JobIDRaw','State','ElapsedRaw','Start','End','AllocTRES')
METADATA=('plan.json','allocation-ledger.json','submission.json','allocation-exit.json',
          'model-preflight/result.json','episodes/manifest.json','episodes/results.json',
          'episodes/status.json','episodes/partial.json','episodes/complete.json','episodes/budget.json')


def write_json(path, value):
    path=Path(path)
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w') as handle:
        json.dump(value,handle,indent=2)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    directory=os.open(str(path.parent),os.O_RDONLY)
    try:os.fsync(directory)
    finally:os.close(directory)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scheduler(command):
    env=dict(os.environ,TZ='UTC',SLURM_TIME_FORMAT='%Y-%m-%dT%H:%M:%S')
    return subprocess.check_output(command,text=True,env=env).strip()


def query_terminal_accounting(job_ids):
    joined=','.join(job_ids)
    # squeue --jobs exits nonzero after completed IDs leave controller memory.
    # Query the live queue, then select exact attempt IDs; sacct remains the
    # authoritative terminal accounting check below.
    queued=scheduler(['squeue','--noheader','--format=%i|%T'])
    live='\n'.join(line for line in queued.splitlines()
                   if line.split('|',1)[0] in job_ids)
    if live:
        raise ValueError('Prior attempt is still present in squeue; no recovery mutation allowed')
    command=['sacct','--noheader','--parsable2','--allocations','--jobs',joined,
             '--format='+','.join(ACCOUNT_FIELDS)]
    raw=scheduler(command)
    rows=[]
    for line in raw.splitlines():
        values=line.split('|')
        if len(values)!=len(ACCOUNT_FIELDS):
            raise ValueError('Unexpected Slurm accounting format')
        rows.append(dict(zip(ACCOUNT_FIELDS,values)))
    return rows,dict(command=command,raw=raw,squeue=live,queried_at=time.time())


def terminal_seconds(job_ids, rows):
    if len(job_ids)!=len(set(job_ids)) or not job_ids:
        raise ValueError('Each allocation attempt must have one unique job ID')
    indexed={}
    for row in rows:
        ident=row['JobIDRaw']
        if ident not in job_ids or ident in indexed:
            raise ValueError('Missing/duplicate/requeued allocation accounting is ambiguous')
        state=row['State'].split()[0].rstrip('+')
        if state not in TERMINAL:
            raise ValueError('Every prior allocation must be terminal')
        elapsed=int(row['ElapsedRaw'])
        if elapsed<0 or not row['End'] or row['End'] in ('Unknown','None'):
            raise ValueError('Allocation elapsed/end accounting is incomplete')
        if elapsed and (row['Start'] in ('Unknown','None','') or not row['AllocTRES']):
            raise ValueError('Used allocation lacks start/resource accounting')
        indexed[ident]=elapsed
    if set(indexed)!=set(job_ids):
        raise ValueError('Accounting does not cover every prior allocation')
    return sum(indexed.values())


def inspect_cells(root, identity, task_ids):
    """Preserve finalized attempts and reject every ambiguous interrupted cell."""
    rows={}
    missing=[]
    unattempted=[]
    for task_id in task_ids:
        if not re.fullmatch(r'[A-Za-z0-9_]+',task_id):
            raise ValueError('Unsafe task identifier')
        for arm in ARMS:
            relative=Path('episodes')/task_id/arm
            folder=root/relative
            if folder.is_symlink() or folder.resolve()!=folder:
                raise ValueError('Symlinked cell requires manual audit')
            result=folder/'result.json'
            if not result.exists():
                if folder.exists():
                    raise ValueError(f'Interrupted/ambiguous cell requires manual audit: {relative}')
                missing.append(str(relative))
                continue
            row=json.loads(result.read_text())
            if (row.get('task_id')!=task_id or row.get('arm')!=arm
                    or row.get('identity_sha256')!=identity or type(row.get('attempted')) is not bool
                    or type(row.get('valid')) is not bool or type(row.get('success')) is not int
                    or row['success'] not in (0,1) or row['success'] and not row['valid']):
                raise ValueError(f'Unverifiable finalized cell: {relative}')
            rows[str(relative)]=sha(result)
            if not row['attempted']:
                if row['valid'] or row['success'] or row.get('turns',0)!=0:
                    raise ValueError('Unattempted result has trajectory/outcome evidence')
                if set(p.name for p in folder.iterdir())!={'result.json'}:
                    raise ValueError('Unattempted placeholder has extra evidence; manual audit required')
                unattempted.append(str(relative))
                missing.append(str(relative))
    return dict(result_sha256=rows,missing_cells=missing,unattempted_cells=unattempted,
                preserved_attempted_cells=len(rows)-len(unattempted))


def inspect_runtime(root, current_job=None):
    root=Path(root).resolve()
    plan=json.loads((root/'plan.json').read_text())
    if (len(plan['pilot_task_ids'])!=12 or len(set(plan['pilot_task_ids']))!=12
            or sha(root/'online-mind2web.jsonl')!=plan['task_file_sha256']):
        raise ValueError('Fixed twelve-task cohort or canonical task file changed')
    ledger=json.loads((root/'allocation-ledger.json').read_text())
    if ledger['approved_total_seconds']!=plan['approval']['gpu_seconds_total']:
        raise ValueError('Allocation ledger and explicit approval total disagree')
    if ledger['api_cap_usd']!=plan['approval']['api_cap_usd']:
        raise ValueError('API ledger approval differs')
    jobs=[str(attempt['job_id']) for attempt in ledger['attempts'] if str(attempt['job_id'])!=current_job]
    version=plan.get('code_version','v2')
    if not re.fullmatch(r'v[0-9]+',version):
        raise ValueError('Unsupported frozen code version')
    freeze_path=root/f'code-freeze-{version}.json'
    freeze=json.loads(freeze_path.read_text())
    code=root/f'code-{version}'
    if Path(freeze['code_directory']).resolve()!=code:
        raise ValueError('Recovery must use the same absolute frozen code directory')
    for name,expected in freeze['code_sha256'].items():
        if sha(code/name)!=expected:
            raise ValueError(f'Frozen source changed: {name}')
    manifest_path=root/'episodes/manifest.json'
    manifest=json.loads(manifest_path.read_text())
    identity=manifest['identity']
    digest=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    if (digest!=manifest['identity_sha256'] or identity['task_ids']!=plan['pilot_task_ids']
            or identity['arms']!=list(ARMS) or identity['task_file_sha256']!=plan['task_file_sha256']
            or identity['api_usd_cap']!=ledger['api_cap_usd']):
        raise ValueError('Fixed scientific identity differs from plan')
    budget_path=Path(manifest['config'].get('budget_path',root/'episodes/budget.json'))
    budget=json.loads(budget_path.read_text())
    if (budget['cap']!=ledger['api_cap_usd'] or not math.isfinite(budget['charged'])
            or not 0<=budget['charged']<budget['cap'] or budget.get('halt_reason')):
        raise ValueError('Shared API ledger has no approved spend remaining')
    charges=[]
    for index,call in enumerate(budget['calls']):
        if call['id']!=index or call['state'] not in ('reserved','settled'):
            raise ValueError('API ledger reservations are ambiguous or overrun')
        amount=call['reserved'] if call['state']=='reserved' else call['charged']
        if not math.isfinite(amount) or amount<0:
            raise ValueError('Invalid API ledger charge')
        charges.append(amount)
    if not math.isclose(sum(charges),budget['charged'],rel_tol=1e-10,abs_tol=1e-8):
        raise ValueError('API reservations do not reconcile with charged total')
    cells=inspect_cells(root,digest,plan['pilot_task_ids'])
    if not cells['missing_cells']:
        raise ValueError('No missing cells; completed attempts must never be rerun')
    if (root/'episodes/complete.json').exists():
        raise ValueError('Completion marker conflicts with missing cells; manual audit required')
    return dict(root=str(root),code_directory=str(code),identity_sha256=digest,prior_job_ids=jobs,
        approved_total_seconds=ledger['approved_total_seconds'],api_cap_usd=ledger['api_cap_usd'],
        api_charged_usd=budget['charged'],budget_path=str(budget_path),budget_sha256=sha(budget_path),
        manifest_sha256=sha(manifest_path),freeze_sha256=sha(freeze_path),
        previous_deadline=manifest['config']['deadline'],**cells)


def build_plan(snapshot, accounting, proof, max_seconds=7200):
    consumed=terminal_seconds(snapshot['prior_job_ids'],accounting)
    remaining=snapshot['approved_total_seconds']-consumed
    if not 0<max_seconds<=snapshot['approved_total_seconds']:
        raise ValueError('Per-allocation ceiling must fit the explicitly approved total')
    requested=(min(max_seconds,remaining)//60)*60
    if requested<600:
        raise ValueError('Less than ten minutes remains for startup and useful continuation')
    return dict(schema_version=1,**snapshot,consumed_seconds=consumed,remaining_seconds=remaining,
        requested_seconds=requested,rounding_unrequested_seconds=remaining-requested,
        resources=dict(gpus=1,gpu_type='H200',cpus=8,memory_gib=120,seconds=requested),
        scheduler_accounting=accounting,scheduler_proof=proof,prepared_at=time.time(),
        helper_sha256=sha(__file__),status='prepared_not_submitted')


def prepare(root, destination, max_seconds=7200, current_job=None):
    if current_job is not None and current_job!=os.environ.get('SLURM_JOB_ID'):
        raise ValueError('Only the actual current allocation can be excluded from prior accounting')
    snapshot=inspect_runtime(root,current_job=current_job)
    rows,proof=query_terminal_accounting(snapshot['prior_job_ids'])
    plan=build_plan(snapshot,rows,proof,max_seconds=max_seconds)
    destination=Path(destination).resolve()
    destination.mkdir(parents=True,exist_ok=False)
    for relative in METADATA:
        source=Path(root)/relative
        if source.exists():
            target=destination/'archive'/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,target)
    shutil.copy2(__file__,destination/'resume_webprm_pilot.py')
    shutil.copy2(plan['budget_path'],destination/'archive/shared-budget.json')
    plan['metadata_sha256']={relative:sha(destination/'archive'/relative) for relative in METADATA
                             if (destination/'archive'/relative).exists()}
    write_json(destination/'recovery-plan.json',plan)
    seconds=plan['requested_seconds']
    duration=f'{seconds//3600:02}:{seconds%3600//60:02}:00'
    q=shlex.quote
    script=f'''#!/usr/bin/env bash
# Reviewed recovery only; this file is never submitted by the helper.
#SBATCH --job-name=webprm-recovery
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time={duration}
#SBATCH --signal=B:TERM@180
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/webprm-pilot-%j.out
set -euo pipefail
if [[ -f {q(str(Path(root)/'model-cache.env'))} ]]; then source {q(str(Path(root)/'model-cache.env'))}; fi
{q(ENGINE)} {q(str(destination/'resume_webprm_pilot.py'))} --root {q(str(root))} --begin --prepared {q(str(destination))}
source {q(str(destination/'active.env'))}
cd {q(plan['code_directory'])}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 WANDB_PROJECT=openwebrl-evals
export CUDA_HOME={q(CUDA)}
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${{LD_LIBRARY_PATH:-}}"
exec {q(ENGINE)} scripts/launch_webprm_pilot.py --root {q(str(root))} --browser-python {q(str(Path(root)/'browser-env/bin/python'))} --engine-python {q(ENGINE)} --env-file {q(ENV_FILE)}
'''
    (destination/'resume.sbatch').write_text(script)
    return plan


def parse_live_job(raw, job_id, expected_seconds, now):
    fields=dict(re.findall(r'(\w+)=([^\s]+)',raw))
    if fields.get('JobId')!=job_id or fields.get('JobState')!='RUNNING':
        raise ValueError('Begin hook requires its actual running Slurm allocation')
    def duration(name):
        value=fields[name]
        match=re.fullmatch(r'(?:(\d+)-)?(\d+):(\d{2}):(\d{2})',value)
        if not match:
            raise ValueError(f'Unrecognized Slurm duration: {name}={value}')
        days,hours,minutes,seconds=match.groups()
        return int(days or 0)*86400+int(hours)*3600+int(minutes)*60+int(seconds)
    limit,elapsed=duration('TimeLimit'),duration('RunTime')
    if limit!=expected_seconds or not 0<=elapsed<limit:
        raise ValueError('Current allocation duration differs from exact prepared request')
    # Use Slurm's enforced duration and elapsed runtime. Calendar rendering can
    # differ between scheduler clients; it must not reset an approved budget.
    # Five seconds cover query latency and whole-second runtime rounding.
    start=now-elapsed
    end=now+limit-elapsed-5
    tres=dict(part.split('=',1) for part in fields.get('AllocTRES','').split(',') if '=' in part)
    if (tres.get('cpu')!='8' or tres.get('node')!='1' or tres.get('gres/gpu')!='1'
            or tres.get('gres/gpu:h200')!='1' or tres.get('mem') not in ('120G','122880M')):
        raise ValueError('Actual allocation resources differ from 1 H200 / 8 CPU / 120 GiB')
    remaining=int(end-now)
    deadline=end-240
    if deadline-now<300:
        raise ValueError('Insufficient allocation time remaining for continuation')
    return dict(start=start,end=end,remaining_seconds=remaining,new_deadline=deadline,
                scheduler_time_limit_seconds=limit,scheduler_elapsed_seconds=elapsed,
                scheduler_calendar_start=fields.get('StartTime'),
                scheduler_calendar_end=fields.get('EndTime'))


def begin(root, prepared):
    prepared=Path(prepared).resolve()
    plan=json.loads((prepared/'recovery-plan.json').read_text())
    if sha(__file__)!=plan['helper_sha256']:
        raise ValueError('Recovery helper changed after preparation')
    current=os.environ.get('SLURM_JOB_ID','')
    if not current.isdigit() or current in plan['prior_job_ids']:
        raise ValueError('Begin hook must run inside a new, registered Slurm attempt')
    ledger=json.loads((Path(root)/'allocation-ledger.json').read_text())
    current_rows=[row for row in ledger['attempts'] if str(row['job_id'])==current]
    if len(current_rows)!=1 or current_rows[0]['requested_seconds']!=plan['requested_seconds']:
        raise ValueError('Root must register the exact new attempt in allocation-ledger.json first')
    snapshot=inspect_runtime(root,current_job=current)
    for key in ('root','identity_sha256','prior_job_ids','approved_total_seconds','api_cap_usd',
                'budget_sha256','manifest_sha256','freeze_sha256','result_sha256',
                'missing_cells','unattempted_cells'):
        if snapshot[key]!=plan[key]:
            raise ValueError(f'Prepared state changed before begin: {key}')
    rows,proof=query_terminal_accounting(plan['prior_job_ids'])
    consumed=terminal_seconds(plan['prior_job_ids'],rows)
    if consumed+plan['requested_seconds']>plan['approved_total_seconds']:
        raise ValueError('Updated accounting would exceed the approved total budget')
    raw=scheduler(['scontrol','show','job','--oneliner',current])
    live=parse_live_job(raw,current,plan['requested_seconds'],time.time())
    with (prepared/'begin.lock').open('x') as handle:
        handle.write(current+'\n')
        handle.flush()
        os.fsync(handle.fileno())
    archived=[]
    for relative in plan['unattempted_cells']:
        source=Path(root)/relative
        target=prepared/'archive/unattempted'/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        source.rename(target)
        archived.append(dict(path=relative,result_sha256=sha(target/'result.json')))
    path=Path(root)/'episodes/manifest.json'
    manifest=json.loads(path.read_text())
    manifest['config']['deadline']=live['new_deadline']
    write_json(path,manifest)
    receipt=dict(job_id=current,started_at=time.time(),identity_sha256=plan['identity_sha256'],
        old_manifest_sha256=plan['manifest_sha256'],new_manifest_sha256=sha(path),
        old_deadline=plan['previous_deadline'],**live,consumed_prior_seconds=consumed,
        accounting=rows,scheduler_proof=proof,scontrol=raw,archived_unattempted=archived,
        budget_sha256_before=plan['budget_sha256'],budget_sha256_after=sha(plan['budget_path']))
    write_json(prepared/'begin-receipt.json',receipt)
    (prepared/'active.env').write_text(f"export WEBPRM_ALLOCATION_SECONDS={live['remaining_seconds']}\n")
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--dry-run',action='store_true')
    modes.add_argument('--prepare',action='store_true')
    modes.add_argument('--begin',action='store_true')
    modes.add_argument('--prepare-and-begin',action='store_true',
                       help='Prepare at a dependency handoff inside the registered new allocation')
    parser.add_argument('--output',type=Path,help='New recovery directory, required for --prepare')
    parser.add_argument('--prepared',type=Path,help='Prepared recovery directory, required for --begin')
    parser.add_argument('--max-seconds',type=int,default=7200,
                        help='Additional per-allocation ceiling; never extends the approved total')
    args=parser.parse_args()
    if args.prepare_and_begin:
        if not args.output:parser.error('--prepare-and-begin requires --output')
        current=os.environ.get('SLURM_JOB_ID','')
        if not current.isdigit():parser.error('Dependency handoff requires a Slurm allocation')
        prepare(args.root.resolve(),args.output,max_seconds=args.max_seconds,current_job=current)
        result=begin(args.root.resolve(),args.output)
    elif args.prepare:
        if not args.output:parser.error('--prepare requires --output')
        result=prepare(args.root.resolve(),args.output,max_seconds=args.max_seconds)
    elif args.begin:
        if not args.prepared:parser.error('--begin requires --prepared')
        result=begin(args.root.resolve(),args.prepared)
    else:
        snapshot=inspect_runtime(args.root)
        rows,proof=query_terminal_accounting(snapshot['prior_job_ids'])
        result=build_plan(snapshot,rows,proof,max_seconds=args.max_seconds)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
