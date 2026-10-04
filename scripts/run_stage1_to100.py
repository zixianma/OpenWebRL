#!/usr/bin/env python3
"""Own a baseline/additive continuation and its iteration-100 evaluation."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from prepare_stage1_to100 import (BASELINE_SOURCE,BASELINE_PENDING,ADDITIVE_SOURCE,
                                 CONTROL,prepare_sources)
from resume_baseline import REPO,RUNTIME,write_json,allocation
from runtime_ports import PORT_ENV,lease_ports

ADDITIVE_ROOT=RUNTIME/'evaluations/arm-failure-additive-311962'


def additive_environment():
    os.environ.update(ARM_VARIANT_TARGET_ITERATION='100', ARM_VARIANT_MULTIMODAL_STORAGE='shared',
        ARM_VARIANT_PORT_SOURCE=str(ADDITIVE_SOURCE), ARM_VARIANT_EVAL_RESERVE_SECONDS='3600')
    os.environ.pop('ARM_CONTINUATION_LABEL_GUARD',None)


def prepare():
    prepare_sources()
    from resume_arm_failure_variants import plan,validate_plan
    additive_environment()
    p=plan('PREPARE',ADDITIVE_ROOT,12)
    validate_plan(p)
    if p['start_rollout_id']!=90 or p['initial_optimizer_updates']!=1150:
        raise ValueError('Additive resume origin changed')
    write_json(CONTROL/'additive-plan.json',p)
    pointer=json.loads((RUNTIME/'current_baseline.json').read_text())
    if pointer['completed_training_iterations']!=90 or pointer['durable_optimizer_updates']!=1016:
        raise ValueError('Baseline resume origin changed')
    write_json(CONTROL/'comparison-plan.json',dict(target=100,stage=1,
        baseline_checkpoint=pointer['last_valid_checkpoint'],baseline_adam=1016,
        additive_checkpoint=p['checkpoint'],additive_adam=1150,
        allocations={variant:dict(gpus=4,gpu_type='H200',hours=hours,cpus=32,memory_gib=480,browsers=32)
                     for variant,hours in [('baseline',24),('additive',12)]},
        total_max_gpu_hours=144,evaluation=dict(tasks=300,temperature=0,judge='gpt-4.1',
            prompt_variant='action_history',browser='local',max_turns=30,response_tokens=4096,
            save_every_task=True),training_objectives_unchanged=True,
        caveat='Iteration-aligned comparison; Adam updates differ and rollout/eval noise remains.'))


def eval_plan(job):
    from run_arm_iteration80_eval import plan
    p=plan('additive',job,100,RUNTIME/f'evaluations/arm-failure-additive-{job}')
    p['gpus']=4
    p['environment'].update(NUM_GPUS='4',TP_SIZE='4',BROWSER_CONCURRENCY='32',SGLANG_CONCURRENCY='32')
    return p


def audit_rollouts(directory,source):
    tasks=[json.loads(x)['metadata']['task_id'] for x in (source/'online_mind2web_monitor.jsonl').read_text().splitlines()]
    records=[json.loads(p.read_text()) for p in directory.glob('*.json')]
    if len(records)!=300 or len({r['task_id'] for r in records})!=300 or {r['task_id'] for r in records}!=set(tasks):
        raise ValueError('Iteration-100 evaluation did not save the full declared cohort')
    if not all((directory/r['rollout_file']).is_file() and (directory/r['rollout_file']).stat().st_size for r in records):
        raise ValueError('Missing trajectory archives')
    if not all(r['judge_model']=='gpt-4.1' and r['judge_prompt_variant']=='action_history' for r in records):
        raise ValueError('Evaluation judge changed')
    s=sum(r['metrics']['successes'] for r in records);v=sum(r['metrics']['valid_trajectories'] for r in records)
    return dict(tasks=300,successes=s,valid=v,invalid=300-v,overall=s/300,valid_only=s/v if v else None,
                saved_rollouts=300,saved_verdicts=300,rollouts=str(directory))


def worker(args):
    if os.getenv('SLURM_JOB_ID')!=args.job_id:
        raise ValueError('Authorized allocation required')
    if args.worker=='additive-train':
        from resume_arm_failure_variants import plan,execute
        additive_environment();execute(plan(args.job_id,ADDITIVE_ROOT,12))
    elif args.worker=='additive-eval':
        import evaluate_baseline_checkpoint as evaluation
        os.environ.update(OPENWEBRL_EXPECTED_SCHEDULER_OFFSET='0',OPENWEBRL_RECORD_EVAL_EXPECTED_TASK_COUNT='300')
        evaluation.run(eval_plan(args.job_id),REPO/'.env')


def execute(args):
    job=args.job_id
    os.environ.setdefault('FLASHINFER_WORKSPACE_BASE',str(RUNTIME))
    if os.getenv('SLURM_JOB_ID')!=job:raise ValueError('Authorized allocation required')
    hours=24 if args.variant=='baseline' else 12
    allocation(subprocess.check_output(['scontrol','show','job',job,'-o'],text=True),job,
               requested_gpus=4,maximum_hours=hours)
    from check_storage_quota import check_storage_quota
    check_storage_quota()
    root=RUNTIME/f'evaluations/stage1-{args.variant}-to100-{job}'
    root.mkdir(exist_ok=False)
    write_json(root/'controller-plan.json',dict(job_id=job,variant=args.variant,target=100,
        resources=dict(gpus=4,hours=hours,cpus=32,memory_gib=480),evaluation_in_allocation=True))
    os.environ['WANDB_PROJECT']='openwebrl'
    try:
        if args.variant=='baseline':
            scratch=RUNTIME/'multimodal-scratch'/f'baseline-to100-{job}'
            scratch.mkdir(exist_ok=True)
            if shutil.disk_usage(scratch).free<2*1024**4:raise OSError('Require 2 TiB shared scratch headroom')
            lease,base=lease_ports(job);os.environ[PORT_ENV]=str(base)
            os.environ['OPENWEBRL_BATCH_DRIVER_JOB']=job
            command=[sys.executable,str(REPO/'scripts/resume_baseline.py'),'--job-id',job,
                     '--gpus','4','--source',str(BASELINE_SOURCE),'--maximum-hours',str(hours),
                     '--wandb-run-id','qcq7i4ug']
            try:
                subprocess.run([*command,'--dry-run'],check=True)
                subprocess.run([*command,'--verify-resume-only','--launch'],check=True)
                pointer_path=RUNTIME/'current_baseline.json'
                pointer=json.loads(pointer_path.read_text())
                pointer.update(pending_evaluation_resume_source=str(BASELINE_PENDING),
                               next_prepared_source_directory=str(BASELINE_SOURCE))
                write_json(pointer_path,pointer)
                subprocess.run([*command,'--launch'],check=True)
            finally:lease.close()
            pointer=json.loads((RUNTIME/'current_baseline.json').read_text())
            run=Path(pointer['run_directory'])
            if int((run/'latest_checkpointed_iteration.txt').read_text())!=99:
                raise ValueError('Baseline stopped before iteration 100; checkpoint retained')
            result=audit_rollouts(run/'evaluation/after100/rollouts',BASELINE_SOURCE)
            write_json(root/'evaluation-audit.json',result)
            # Refresh only after independently verifying both checkpoint and eval.
            from resume_baseline import source_command
            inspection=json.loads(subprocess.check_output(source_command(BASELINE_SOURCE,[sys.executable,
                str(REPO/'scripts/inspect_training_checkpoint.py'),str(run),
                '--expected-scheduler-offset-updates','1']),text=True))
            write_json(run/'checkpoint_after100_audit.json',inspection)
            pointer.update(last_valid_checkpoint=inspection['checkpoint'],completed_training_iterations=100,
                durable_optimizer_updates=inspection['completed_optimizer_updates'],
                pending_evaluation_iteration_one_based=None,last_completed_evaluation_iteration_one_based=100,
                last_completed_evaluation_report=str(root/'evaluation-audit.json'),
                last_valid_checkpoint_report=str(run/'checkpoint_after100_audit.json'),
                status_note='Stage-1 iteration 100 and all 300 task-addressable scheduled evaluation records verified.')
            write_json(RUNTIME/'current_baseline.json',pointer)
        else:
            step=['srun',f'--jobid={job}','--nodes=1','--ntasks=1','--cpus-per-task=32',
                  '--gres=gpu:h200:4','--exact','--cpu-bind=none',sys.executable,str(Path(__file__).resolve()),
                  '--job-id',job,'--variant','additive']
            subprocess.run([*step,'--worker','additive-train'],check=True)
            from run_arm_iteration80_eval import checkpoint_ready
            training_root=RUNTIME/f'evaluations/arm-failure-additive-{job}'
            if checkpoint_ready('additive',100,training_root) is None:
                raise ValueError('Additive stopped before iteration 100; no earlier checkpoint substituted')
            subprocess.run([*step,'--worker','additive-eval'],check=True)
            directory=RUNTIME/f'evaluations/arm-additive-iter100-{job}/rollouts'
            from run_arm_iteration80_eval import source_for
            result=audit_rollouts(directory,source_for('additive'))
            write_json(root/'evaluation-audit.json',result)
        write_json(root/'status.json',dict(complete=True,iteration=100,evaluation=result))
    except BaseException as exc:
        write_json(root/'status.json',dict(complete=False,error_type=type(exc).__name__,error=str(exc)[:300]))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job-id',default='PREPARE');p.add_argument('--variant',choices=['baseline','additive'])
    p.add_argument('--execute',action='store_true');p.add_argument('--worker',choices=['additive-train','additive-eval'])
    args=p.parse_args()
    if args.worker:worker(args)
    elif args.execute:
        if not args.variant:p.error('--variant required for execution')
        execute(args)
    else:
        prepare();print(CONTROL/'comparison-plan.json')
