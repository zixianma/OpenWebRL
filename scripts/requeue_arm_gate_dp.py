#!/usr/bin/env python3
"""Requeue only B318934/C318935 at a durable boundary, retaining unused budget."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

from prepare_arm_gate_dp import CONTROL,JOBS,SOURCES
from resume_baseline import REPO,RUNTIME,write_json


def seconds(value):
    days=0
    if '-' in value:
        d,value=value.split('-');days=int(d)
    h,m,s=map(int,value.split(':'))
    return days*86400+h*3600+m*60+s


def unused_budget(info):
    remaining=seconds(info['TimeLimit'])-seconds(info['RunTime'])
    # Reserve two minutes for checkpoint-boundary dispatch and Slurm teardown.
    capped=(remaining-120)//60*60
    if capped<7200:raise ValueError('Less than two hours left; do not requeue automatically')
    return capped


def scheduler(job):
    raw=subprocess.check_output(['scontrol','show','job',job,'-o'],text=True,timeout=20)
    return dict(re.findall(r'(\w+)=(\S+)',raw))


def fingerprints():
    files=[REPO/'scripts'/name for name in (
        'prepare_arm_gate_dp.py','validate_arm_gate_dp.py','requeue_arm_gate_dp.py',
        'prepare_arm_gate_to60.py','resume_arm_failure_variants.py','resume_arm_turn_bonus.py',
        'check_arm_gpu_checkpoint_restore.py')]
    files += [REPO/'openwebrl/arm_failure_dp.py',REPO/'openwebrl/arm_dp_validation.py']
    files += [source/'reference_manifest.json' for source in SOURCES.values()]
    from validate_arm_gate_dp import VALIDATION_SOURCES
    files += [source/'reference_manifest.json' for source in VALIDATION_SOURCES.values()]
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def watch(job):
    if job not in JOBS:raise ValueError('Only explicitly authorized B/C jobs')
    ready=json.loads((CONTROL/'readiness.json').read_text())
    if not ready['cpu_passed'] or ready['fingerprints'] != fingerprints():
        raise ValueError('DP readiness changed before requeue')
    root=RUNTIME/f'evaluations/arm-failure-additive-{job}-iter40'
    start=json.loads((root/'status.json').read_text())
    previous=start.get('last_valid_checkpoint')
    initial=int(Path(previous).name.split('_')[-1]) if previous else 19
    path=CONTROL/f'{job}.json'
    if path.exists():raise ValueError('Existing migration request; inspect instead of duplicating')
    request=dict(job_id=job,variant=JOBS[job],stage='waiting-for-checkpoint',resume_from=str(root),
        initial_completed_iteration=initial+1,requested_topology='TP2/DP4/microbatch1',
        resource_request_unchanged=True,budget_extension=False,created_at=time.time())
    write_json(path,request)
    deadline=time.monotonic()+6*3600
    while time.monotonic()<deadline:
        info=scheduler(job)
        if info['JobState']!='RUNNING':raise ValueError('Original job stopped before migration boundary')
        state=json.loads((root/'status.json').read_text())
        checkpoint=state.get('last_valid_checkpoint')
        rid=int(Path(checkpoint).name.split('_')[-1]) if checkpoint else 19
        validation=root/'iterations'/f'{rid:04d}'/'checkpoint-validation.json'
        boundary_available = rid>initial or (checkpoint and state.get('stage') in ('collection','selector-startup','actor-startup'))
        if boundary_available and validation.exists():
            report=json.loads(validation.read_text())
            if report['completed_optimizer_updates'] != state['completed_optimizer_updates']:
                raise ValueError('Checkpoint validation and controller disagree')
            break
        time.sleep(15)
    else:raise TimeoutError('No durable checkpoint within six hours')
    if ready['fingerprints'] != fingerprints():raise ValueError('Readiness changed while waiting')
    remaining=unused_budget(info)
    request.update(stage='holding',remaining_seconds=remaining,checkpoint=checkpoint,
        completed_iteration=rid+1,completed_optimizer_updates=state['completed_optimizer_updates'],
        old_elapsed_seconds=seconds(info['RunTime']),old_limit_seconds=seconds(info['TimeLimit']),
        interrupted_phase=state.get('stage'),unfinished_collection_artifacts_preserved=True)
    write_json(path,request)
    subprocess.run(['scontrol','requeuehold',job],check=True,timeout=60)
    for _ in range(24):
        held=scheduler(job)
        if held['JobState']=='PENDING':break
        time.sleep(5)
    else:raise TimeoutError('Requeued job did not enter pending hold')
    # Preserve the previous Slurm log before its cached batch script starts again.
    log=RUNTIME/f'logs/slurm-arm-gate-to60-{job}.out'
    preserved=log.with_name(log.stem+'-before-tp2'+log.suffix)
    if log.exists():
        if preserved.exists():raise ValueError('Previous log archive already exists')
        log.rename(preserved)
    current=json.loads((root/'status.json').read_text())
    current.update(stage='complete',failed=False,has_saved_checkpoints=True,training_complete=False,
        last_valid_checkpoint=checkpoint,completed_optimizer_updates=report['completed_optimizer_updates'],
        stop_reason='User-authorized topology requeue at a validated durable checkpoint',
        topology_requeue=dict(job_id=job,completed_iteration=rid+1,checkpoint=checkpoint))
    write_json(root/'status-before-topology-requeue.json',state)
    write_json(root/'status.json',current)
    hours,rest=divmod(remaining,3600);minutes,secs=divmod(rest,60)
    limit=f'{hours:02d}:{minutes:02d}:{secs:02d}'
    subprocess.run(['scontrol','update',f'JobId={job}',f'TimeLimit={limit}'],check=True,timeout=30)
    checked=scheduler(job)
    if checked['JobState']!='PENDING' or seconds(checked['TimeLimit'])!=remaining:
        raise ValueError('Budget cap not confirmed; keep job held')
    # Write before release: a fast scheduler may start the batch immediately.
    request.update(stage='released',new_time_limit=limit,released_at=time.time())
    write_json(path,request)
    subprocess.run(['scontrol','release',job],check=True,timeout=30)
    print(json.dumps(request),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--job-id',choices=JOBS,required=True)
    args=p.parse_args()
    try:watch(args.job_id)
    except BaseException as exc:
        write_json(CONTROL/f'{args.job_id}-watch-error.json',dict(error_type=type(exc).__name__,error=str(exc),time=time.time()))
        raise
