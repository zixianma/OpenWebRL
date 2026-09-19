#!/usr/bin/env python3
"""Prepare/resume ARM RL in an existing allocation; never submits paid jobs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import run_arm_turn_bonus_cycles as training
from resume_baseline import write_json,source_command

REPO,RUNTIME=training.REPO,training.RUNTIME


def origin(root):
    root=Path(root).resolve()
    if not root.is_relative_to(RUNTIME/'evaluations'):
        raise ValueError('ARM continuation must use a recorded runtime experiment')
    manifest=json.loads((root/'launch_manifest.json').read_text())
    state=json.loads((root/'status.json').read_text())
    # A controller can fail during finalization after durable checkpoints have
    # already been written.  Permit continuation from that verified checkpoint
    # while still rejecting runs with no saved state or a failed diagnostic.
    durable_checkpoint = state.get('last_valid_checkpoint')
    has_durable_checkpoint = bool(state.get('has_saved_checkpoints') or
                                  (durable_checkpoint and Path(durable_checkpoint).exists()))
    durable_after_controller_failure = bool(state.get('failed') and has_durable_checkpoint)
    if (manifest.get('diagnostic_only') or manifest.get('selector_provider')!='arm'
            or (state.get('stage')!='complete' and not durable_after_controller_failure)
            or (state.get('failed') and not durable_after_controller_failure)
            or not has_durable_checkpoint):
        raise ValueError('Requires an inactive, completed real ARM training run')
    allowed_sources={training.SOURCE.resolve(),
        RUNTIME/'reference-arm-turn-bonus-cycles-20260913-v3',
        RUNTIME/'reference-arm-failure-bonus-20260913-v2',
        RUNTIME/'reference-arm-failure-additive-20260914-v3'}
    if Path(manifest['source']).resolve() not in allowed_sources:
        raise ValueError('Resume must preserve the checkpoint source')
    from resume_baseline import validate_source
    validate_source(Path(manifest['source']))
    os.environ.setdefault('FLASHINFER_WORKSPACE_BASE',str(RUNTIME))
    from inspect_training_checkpoint import inspect_checkpoint
    report=inspect_checkpoint(root/'runtime',expected_updates=state['completed_optimizer_updates'])
    checkpoint=Path(report['checkpoint'])
    paths=[checkpoint/'common.pt',checkpoint/'.metadata',Path(report['dataset_cursor']),
           root/'runtime/latest_checkpointed_iteration.txt']
    return dict(root=str(root),manifest=manifest,checkpoint_report=report,
        identity_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


def plan(job,resume_from,minutes=240,gpus=4):
    if gpus not in (4,8):
        raise ValueError("ARM continuation supports four or eight GPUs")
    if not 1 <= minutes <= (1440 if gpus == 4 else 480):
        raise ValueError('ARM continuation supports at most twenty-four hours on four GPUs or eight hours on eight GPUs')
    initial=origin(resume_from)
    p=training.plan(job,'arm',min(minutes,1440))
    p['requested_resources'].update(gpus=gpus,hours=minutes/60,gpu_hours=gpus*minutes/60,
        cpus=8*gpus,memory_gib=120*gpus)
    p['requested_iterations']=max(4,(minutes+59)//60)
    report=initial['checkpoint_report']; next_id=report['iteration']+1
    run_id=initial['manifest']['wandb_run_id']
    p.update(resume_from=initial['root'],resume_origin=initial,fresh_optimizer=False,
        initial_optimizer_updates=report['completed_optimizer_updates'],start_rollout_id=next_id,
        checkpoint=report['checkpoint'],wandb_run_id=run_id)
    # A new allocation gets separate logs/artifacts; optimizer, task cursor and
    # scientific W&B lineage continue from the verified native checkpoint.
    browsers=8*gpus
    p['requested_resources']['browsers']=browsers
    p['environment'].update(NUM_GPUS=str(gpus),TP_SIZE=str(gpus),SLIME_LOAD_CHECKPOINT=str(Path(initial['root'])/'runtime'),
        NUM_ROLLOUT=str(next_id+p['requested_iterations']),WANDB_RUN_ID=run_id,WANDB_RESUME='must',
        BROWSER_CONCURRENCY=str(browsers),SGLANG_CONCURRENCY=str(browsers))
    p['browser_config']['browser_rollout_concurrency']=browsers
    # Extending num_rollout changes the constructor's WD horizon. Restore all
    # scheduler settings and its counter from the checkpoint, without resetting.
    p['command'].append('--use-checkpoint-opt-param-scheduler')
    p['arm_config'].update(run_id=run_id,policy_id=f'{run_id}:uninitialized',
        checkpoint=report['checkpoint'],minimum_cycle_seconds=4500)
    p['gpu_restore_output']=str(Path(initial['root'])/f'gpu-restore-check-{job}-{gpus}gpu')
    p['resume_restore_receipt']=str(Path(p['gpu_restore_output'])/'result.json')
    p['budget_note']='Reserve 75 minutes per new TP4/32-browser cycle after the measured ~64-minute job 294976 cycle; requested iterations are a cap, not a promise. Preserve save/shutdown time.'
    return p


def continuation_identity(p):
    fields={k:p[k] for k in ('command','environment','source','checkpoint','start_rollout_id')}
    return hashlib.sha256(json.dumps(fields,sort_keys=True).encode()).hexdigest()


def validate_resume_plan(p,require_gpu_restore=False):
    current=origin(p['resume_from'])
    if current!=p['resume_origin']:
        raise ValueError('Resume checkpoint, counters, cursor or lineage changed after planning')
    if (p['initial_optimizer_updates']!=current['checkpoint_report']['completed_optimizer_updates']
            or p['start_rollout_id']!=current['checkpoint_report']['iteration']+1):
        raise ValueError('Incorrect resume optimizer offset or next rollout ID')
    if require_gpu_restore:
        receipt=json.loads(Path(p['resume_restore_receipt']).read_text())
        expected=dict(passed=True,job_id=p['job_id'],source=p['source'],
            source_checkpoint_root=str(Path(p['resume_from'])/'runtime'),
            checkpoint_completed_optimizer_updates=p['initial_optimizer_updates'],
            loaded_iteration=p['start_rollout_id']-1,next_rollout_id=p['start_rollout_id'],
            gpus=p['requested_resources']['gpus'],optimizer_updates_executed=0,browser_collections_executed=0,
            continuation_identity_sha256=continuation_identity(p))
        if any(receipt.get(k)!=v for k,v in expected.items()):
            raise ValueError('Missing matching native GPU restore receipt for this allocation/topology')


def stage_checkpoint_reference(p):
    """Link only the immutable preceding checkpoint; native new saves are local."""
    root=Path(p['output'])/'runtime'; root.mkdir(exist_ok=False)
    report=p['resume_origin']['checkpoint_report']; checkpoint=Path(report['checkpoint'])
    (root/checkpoint.name).symlink_to(checkpoint,target_is_directory=True)
    (root/'rollout').mkdir()
    cursor=Path(report['dataset_cursor'])
    shutil.copyfile(cursor,root/'rollout'/cursor.name)
    (root/'latest_checkpointed_iteration.txt').write_text(str(report['iteration'])+'\n')
    write_json(Path(p['output'])/'resume-provenance.json',dict(
        checkpoint=str(checkpoint),dataset_cursor=str(cursor),
        initial_optimizer_updates=p['initial_optimizer_updates'],
        identity_sha256=p['resume_origin']['identity_sha256'],source_checkpoint_modified=False))


def execute(p):
    from arm_launch_preflight import require_receipt
    require_receipt()
    validate_resume_plan(p)
    if os.getenv('SLURM_JOB_ID')!=p['job_id']:
        raise ValueError('Execute only inside the user-authorized allocation')
    launch_plan=RUNTIME/'logs'/f'arm-resume-{p["job_id"]}-launch.json'
    write_json(launch_plan,p)
    # The TP2 checkpoint must actually load as TP4 before browsers or ARM start.
    command=[str(RUNTIME/'venv/bin/python'),str(REPO/'scripts/check_arm_gpu_checkpoint_restore.py'),
        '--training-root',p['resume_from'],'--job-id',p['job_id'],'--gpus',str(p['requested_resources']['gpus']),
        '--output',p['gpu_restore_output'],'--continuation-plan',str(launch_plan),'--execute']
    subprocess.run(source_command(training.SOURCE,command),check=True)
    validate_resume_plan(p,require_gpu_restore=True)
    log_path=RUNTIME/'logs'/f'arm-resume-{p["job_id"]}-monitor.log'
    with log_path.open('x') as log:
        monitor=subprocess.Popen([str(RUNTIME/'venv/bin/python'),str(REPO/'scripts/monitor_arm_turn_bonus.py'),
            '--job-id',p['job_id'],'--watch','--interval','900','--hours',str(p['requested_resources']['hours'])],stdout=log,stderr=subprocess.STDOUT)
        try:
            training.execute(p)
        finally:
            # Own and await the monitor too; never let a batch shell exit while
            # its planned worker still depends on the allocation.
            try: monitor.wait(timeout=45)
            except subprocess.TimeoutExpired:
                monitor.terminate()
                try: monitor.wait(timeout=15)
                except subprocess.TimeoutExpired: monitor.kill(); monitor.wait()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-id',default='APPROVED_JOB')
    parser.add_argument('--resume-from',type=Path,required=True)
    parser.add_argument('--minutes',type=int,default=240)
    parser.add_argument('--gpus',type=int,choices=(4,8),default=4)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args(); p=plan(args.job_id,args.resume_from,args.minutes,args.gpus)
    if args.execute: execute(p)
    else: print(json.dumps(p,indent=2))


if __name__=='__main__': main()
