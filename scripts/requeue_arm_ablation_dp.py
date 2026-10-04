#!/usr/bin/env python3
"""Requeue approved beta318949 at its next checkpoint with unused time only."""
import json
import os
from pathlib import Path
import subprocess
import time

from prepare_arm_ablation_dp import CONTROL,ORIGIN,require_ready,plan
from requeue_arm_gate_dp import scheduler,seconds,unused_budget
from resume_baseline import RUNTIME,write_json


def boundary_report(root,state,initial):
    """Controller validation must catch up to the native checkpoint pointer."""
    checkpoint=state['last_valid_checkpoint'];rid=int(Path(checkpoint).name.split('_')[-1])
    marker=int((root/'runtime/latest_checkpointed_iteration.txt').read_text())
    if marker!=rid:return None
    boundary=rid>initial or state['stage'] in ('collection','selector-startup','actor-startup')
    receipt=root/'iterations'/f'{rid:04d}'/'checkpoint-validation.json'
    if not boundary or not receipt.exists():return None
    report=json.loads(receipt.read_text())
    if report['iteration']!=rid or report['completed_optimizer_updates']!=state['completed_optimizer_updates']:
        raise ValueError('Checkpoint counters disagree')
    return report


def watch():
    job='318949';require_ready();path=CONTROL/f'{job}.json'
    if path.exists():raise ValueError('Migration already recorded; inspect it before retrying')
    state=json.loads((ORIGIN/'status.json').read_text())
    initial=int(Path(state['last_valid_checkpoint']).name.split('_')[-1])
    req=dict(job_id=job,variant='weight',stage='waiting-for-checkpoint',resume_from=str(ORIGIN),
        initial_completed_iteration=initial+1,requested_topology='TP2/DP4/microbatch1',
        resource_request_unchanged=True,budget_extension=False,created_at=time.time())
    write_json(path,req);end=time.monotonic()+7200
    while time.monotonic()<end:
        info=scheduler(job)
        if info['JobState']!='RUNNING':raise ValueError('Beta stopped before its migration checkpoint')
        state=json.loads((ORIGIN/'status.json').read_text())
        report=boundary_report(ORIGIN,state,initial)
        if report:
            checkpoint=report['checkpoint'];rid=report['iteration']
            break
        time.sleep(15)
    else:raise TimeoutError('No durable beta boundary within two hours')
    require_ready();remaining=unused_budget(info)
    req.update(stage='holding',remaining_seconds=remaining,checkpoint=checkpoint,
        completed_iteration=rid+1,completed_optimizer_updates=report['completed_optimizer_updates'],
        old_elapsed_seconds=seconds(info['RunTime']),old_limit_seconds=seconds(info['TimeLimit']),
        interrupted_phase=state['stage'],unfinished_collection_artifacts_preserved=True)
    write_json(path,req)
    subprocess.run(['scontrol','requeuehold',job],check=True,timeout=60)
    for _ in range(24):
        if scheduler(job)['JobState']=='PENDING':break
        time.sleep(5)
    else:raise TimeoutError('Requeued beta did not enter pending hold')
    os.environ.setdefault('FLASHINFER_WORKSPACE_BASE',str(RUNTIME))
    from inspect_training_checkpoint import inspect_checkpoint
    durable=inspect_checkpoint(ORIGIN/'runtime',expected_updates=report['completed_optimizer_updates'])
    if durable['iteration']!=rid:raise ValueError('Checkpoint advanced during shutdown; verify newest state before release')
    log=RUNTIME/f'logs/slurm-arm-failure-ablation-{job}.out';archive=log.with_name(log.stem+'-before-tp2'+log.suffix)
    if log.exists():
        if archive.exists():raise ValueError('Previous Slurm log archive already exists')
        log.rename(archive)
    current=json.loads((ORIGIN/'status.json').read_text())
    write_json(ORIGIN/'status-before-topology-requeue.json',state)
    current.update(stage='complete',failed=False,has_saved_checkpoints=True,training_complete=False,
        last_valid_checkpoint=checkpoint,completed_optimizer_updates=report['completed_optimizer_updates'],
        stop_reason='User-authorized TP2 topology requeue at validated checkpoint')
    write_json(ORIGIN/'status.json',current)
    # Validate actual durable state before releasing the held allocation.
    p=plan('weight',job,remaining/3600,ORIGIN)
    from prepare_arm_ablation_dp import validate_plan,native_parse
    validate_plan(p);native_parse(p,'weight-actual-resume')
    write_json(CONTROL/'weight-actual-resume-plan.json',p)
    h,rest=divmod(remaining,3600);m,s=divmod(rest,60);limit=f'{h:02d}:{m:02d}:{s:02d}'
    subprocess.run(['scontrol','update',f'JobId={job}',f'TimeLimit={limit}'],check=True,timeout=30)
    check=scheduler(job)
    if check['JobState']!='PENDING' or seconds(check['TimeLimit'])!=remaining:raise ValueError('Budget cap not confirmed')
    req.update(stage='released',new_time_limit=limit,released_at=time.time())
    write_json(path,req)
    subprocess.run(['scontrol','release',job],check=True,timeout=30)
    print(json.dumps(req),flush=True)


if __name__=='__main__':
    try:watch()
    except BaseException as exc:
        write_json(CONTROL/'318949-watch-error.json',dict(error_type=type(exc).__name__,error=str(exc),time=time.time()))
        raise
