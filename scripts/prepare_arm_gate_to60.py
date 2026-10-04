#!/usr/bin/env python3
"""Own B/C20->40->60 training and evaluations in one eight-GPU allocation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from resume_baseline import REPO,RUNTIME,validate_source,write_json,allocation
from runtime_ports import patch_rollout_ports

CONTROL=RUNTIME/'arm-turn-bonus-preparation/bc-to60-20260921'
ORIGINS={v:RUNTIME/f'evaluations/arm-failure-additive-{job}'
         for v,job in [('B','313208'),('C','313210')]}
PARENTS={'B':RUNTIME/'reference-arm-gate-recovery-20260920-v3',
         'C':RUNTIME/'reference-arm-gate-c-restart-20260920-v1'}
SOURCES={v:RUNTIME/f'reference-arm-gate-{v.lower()}-to60-20260921-v1' for v in PARENTS}


def prepare_sources():
    from prepare_arm_turn_bonus import copy_plain
    for variant,parent in PARENTS.items():
        validate_source(parent);source=SOURCES[variant]
        name='slime/ray/rollout.py'
        data=patch_rollout_ports((parent/name).read_text()).encode()
        old=json.loads((parent/'reference_manifest.json').read_text())
        if not source.exists():
            shutil.copytree(parent,source,symlinks=True,copy_function=copy_plain,
                ignore=shutil.ignore_patterns('__pycache__','*.pyc','.git','.browser_use_sessions'))
            (source/name).write_bytes(data)
            manifest=json.loads(json.dumps(old))
            manifest['recipe_files_sha256'][name]=hashlib.sha256(data).hexdigest()
            manifest['runtime_port_leases']=True
            manifest['gate_to60']=dict(parent=str(parent),variant=variant,
                changed_files=[name],training_objective_unchanged=True)
            write_json(source/'reference_manifest.json',manifest)
        validate_source(source)
        new=json.loads((source/'reference_manifest.json').read_text())
        if ((source/name).read_bytes()!=data or any(new['recipe_files_sha256'].get(k)!=v
            for k,v in old['recipe_files_sha256'].items() if k!=name)):
            raise ValueError('Unexpected B/C scientific source change')


def configure(variant,target):
    if variant not in SOURCES or target not in (40,60):raise ValueError('Require B/C target40/60')
    os.environ.update(ARM_VARIANT_TARGET_ITERATION=str(target),
        ARM_VARIANT_MULTIMODAL_STORAGE='shared',ARM_VARIANT_PORT_SOURCE=str(SOURCES[variant]),
        ARM_VARIANT_OUTPUT_STAGE=f'iter{target}',
        ARM_VARIANT_EVAL_RESERVE_SECONDS='3600',WANDB_PROJECT='openwebrl')
    os.environ.pop('ARM_CONTINUATION_LABEL_GUARD',None)


def training_plan(variant,root,target,job='PREPARE'):
    from run_arm_gate_checkpoint_eval import checkpoint_ready,RUN_IDS,CREDIT
    from resume_arm_failure_variants import plan,validate_plan
    configure(variant,target)
    if checkpoint_ready(variant,root,target-20) is None:
        raise ValueError('Missing verified preceding B/C20 or B/C40 checkpoint')
    p=plan(job,root,24,gpus=8);validate_plan(p)
    if (p['start_rollout_id']!=target-20 or p['wandb_run_id']!=RUN_IDS[variant]
        or p['arm_config']['credit_assignment']!=CREDIT[variant]
        or p['arm_config']['candidate_gate']!='min2' or p['fresh_optimizer']):
        raise ValueError('B/C continuation identity or resume boundary changed')
    return p


def evaluation_plan(variant,root,target,job):
    from run_arm_gate_checkpoint_eval import plan
    p=plan(variant,root,job,target)
    p['gpus']=8
    p['environment'].update(NUM_GPUS='8',TP_SIZE='8',BROWSER_CONCURRENCY='32',SGLANG_CONCURRENCY='32')
    return p


def prepare():
    prepare_sources();CONTROL.mkdir(parents=True,exist_ok=True)
    stages=[]
    for variant,origin in ORIGINS.items():
        p=training_plan(variant,origin,40)
        write_json(CONTROL/f'{variant}-to40-plan.json',p)
        # Real iteration20 validates the unchanged evaluation path before GPUs.
        e=evaluation_plan(variant,origin,20,'PREPARE')
        write_json(CONTROL/f'{variant}-evaluation-preview.json',e)
        stages.append(dict(variant=variant,origin=str(origin),start=20,targets=[40,60],
            adam_updates=284,source=p['source'],run_id=p['wandb_run_id'],
            allocations=1,each=dict(gpus=8,hours=24,cpus=64,memory_gib=960,browsers=64),
            evaluation='full300, GPT-4.1/action_history, T0, saved rollouts at40 and60',
            dependency='same controller awaits train40, eval40, train60, eval60'))
    subprocess.run(['bash','-n',str(REPO/'scripts/resume_arm_gate_to60_8gpu.sbatch')],check=True)
    write_json(CONTROL/'proposal.json',dict(approved=(CONTROL/'approval-8gpu.json').is_file(),submitted=False,stages=stages,
        max_gpu_hours=384,expected_hours_per_variant='eight-GPU throughput must be measured; twenty-four-hour cap',
        measured_prior=dict(B=dict(collections=14,seconds=50413),C=dict(collections=10,seconds=35515)),
        same_optimizer_scheduler_cursor=True,eval_reserve_seconds_per_stage=3600,
        limit='normal QoS maximum24h; preserve partial progress if60 does not fit; no automatic budget extension'))
    return CONTROL/'proposal.json'


def execute(args):
    job=args.job_id
    if os.getenv('SLURM_JOB_ID')!=job:raise ValueError('Run only inside approved allocation')
    from prepare_arm_gate_dp import request, execute as execute_dp
    if request(job) and int(os.getenv('SLURM_RESTART_COUNT','0')) > 0:
        return execute_dp(args)
    allocation(subprocess.check_output(['scontrol','show','job',job,'-o'],text=True),job,
        requested_gpus=8,maximum_hours=24)
    prepare_sources()
    if args.target!=60:raise ValueError('Eight-GPU controller owns both40 and60')
    root=RUNTIME/f'evaluations/arm-gate-{args.variant.lower()}-to60-{job}'
    root.mkdir(exist_ok=False)
    origin=args.resume_from;completed=[]
    try:
        # User-authorized isolated topology replay inside queued C318935 only.
        # The controller awaits all diagnostic workers before starting production.
        from prepare_arm_tpdp_test import run_if_armed
        run_if_armed(job,args.variant,root)
        for target in (40,60):
            training_root=RUNTIME/f'evaluations/arm-failure-additive-{job}-iter{target}'
            write_json(root/'controller-plan.json',dict(variant=args.variant,target=60,current_target=target,
                training_root=str(training_root),resume_from=str(origin),job_id=job,evaluation_in_allocation=True))
            write_json(root/'status.json',dict(complete=False,stage='training',current_target=target,completed_stages=completed))
            step=['srun',f'--jobid={job}','--nodes=1','--ntasks=1','--cpus-per-task=64',
                '--gres=gpu:h200:8','--exact','--cpu-bind=none',sys.executable,str(Path(__file__).resolve()),
                '--variant',args.variant,'--target',str(target),'--job-id',job,'--resume-from',str(origin)]
            subprocess.run([*step,'--worker','train'],check=True)
            from run_arm_gate_checkpoint_eval import checkpoint_ready
            if checkpoint_ready(args.variant,training_root,target) is None:
                raise ValueError('Training stopped short of target; checkpoint retained; no extra allocation requested')
            write_json(root/'status.json',dict(complete=False,stage='evaluation',current_target=target,completed_stages=completed))
            subprocess.run([*step,'--worker','eval'],check=True)
            from run_stage1_to100 import audit_rollouts
            p=evaluation_plan(args.variant,training_root,target,job)
            audit=audit_rollouts(Path(p['output'])/'rollouts',Path(p['source']))
            write_json(root/f'evaluation-after{target}.json',audit)
            completed.append(target);origin=training_root
        write_json(root/'status.json',dict(complete=True,target=60,completed_stages=completed,evaluation=audit))
    except BaseException as exc:
        write_json(root/'status.json',dict(complete=False,error_type=type(exc).__name__,error=str(exc)[:300]));raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant',choices=SOURCES);p.add_argument('--target',type=int,choices=[40,60])
    p.add_argument('--resume-from',type=Path);p.add_argument('--job-id',default='PREPARE')
    p.add_argument('--execute',action='store_true');p.add_argument('--worker',choices=['train','eval'])
    args=p.parse_args()
    if args.execute or args.worker:
        if args.variant is None or args.target is None or args.resume_from is None:
            p.error('variant, target and resume-from required')
        if os.getenv('SLURM_JOB_ID')!=args.job_id:raise ValueError('Authorized allocation required')
    if args.worker=='train':
        from resume_arm_failure_variants import execute as train
        train(training_plan(args.variant,args.resume_from,args.target,args.job_id))
    elif args.worker=='eval':
        from prepare_arm_tpdp_test import finish_before_evaluation
        finish_before_evaluation(args.job_id,args.variant,args.target)
        import evaluate_baseline_checkpoint as evaluator
        os.environ.update(OPENWEBRL_EXPECTED_SCHEDULER_OFFSET='0',OPENWEBRL_RECORD_EVAL_EXPECTED_TASK_COUNT='300')
        evaluator.run(evaluation_plan(args.variant,RUNTIME/f'evaluations/arm-failure-additive-{args.job_id}-iter{args.target}',args.target,args.job_id),REPO/'.env')
    elif args.execute:execute(args)
    else:print(prepare())


if __name__=='__main__':main()
