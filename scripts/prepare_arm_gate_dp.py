#!/usr/bin/env python3
"""Preserve B/C lineages while moving existing eight-GPU jobs to TP2/DP4."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from resume_baseline import REPO, RUNTIME, validate_source, write_json, allocation

CONTROL = RUNTIME/'arm-turn-bonus-preparation/bc-tp2-migration-20260922'
JOBS = {'318934': 'B', '318935': 'C'}
SOURCES = {v: RUNTIME/f'reference-arm-gate-{v.lower()}-tp2-20260922-v1' for v in ('B', 'C')}


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Production DP patch boundary changed: '+old[:80])
    return text.replace(old, new)


def expected_files(variant):
    from prepare_arm_gate_to60 import SOURCES as parents
    return dp_files(parents[variant])


def dp_files(parent):
    """Apply only auxiliary DP transport changes to the chosen frozen recipe."""
    s = (parent/'openwebrl/arm_failure_aux.py').read_text()
    s = replace_once(s, "if (dp, cp, pp) != (1, 1, 1) or args.micro_batch_size != 1:",
                    "if dp not in (1, 2, 4) or (cp, pp) != (1, 1) or args.micro_batch_size != 1:")
    s = replace_once(s, 'ARM pilot requires DP1/CP1/PP1 and microbatch 1',
                    'ARM requires DP1/2/4, CP1/PP1 and microbatch 1')
    old = "    additions=window_schedule(manifest['records'],manifest['total_failure_rows'],counts,gbs,manifest['coefficient'])"
    capacity = ",manifest.get('max_auxiliary_per_window',32)" if old not in s else ''
    if capacity:old=old[:-1]+capacity+')'
    new = """    from megatron.core import mpu
    from openwebrl.arm_failure_dp import shard_windows
    dp = mpu.get_data_parallel_world_size(False)
    rank = mpu.get_data_parallel_rank(False)
    additions=window_schedule(manifest['records'],manifest['total_failure_rows'],
        [count*dp for count in counts],gbs,manifest['coefficient'])
    additions=shard_windows(additions,dp,rank)"""
    if capacity:new=new.replace("gbs,manifest['coefficient'])", "gbs,manifest['coefficient']"+capacity+')')
    s = replace_once(s, old, new)
    s = replace_once(s, '        for i, scale in addition:', '        for i, scale, padding in addition:')
    s = replace_once(s, "            mixed['arm_source'][-1] = 'arm_failure'",
                    "            mixed['arm_source'][-1] = 'arm_padding' if padding else 'arm_failure'")
    s = replace_once(s, "    if source != 'arm_failure':raise ValueError('Unknown additive source')",
                    "    if source not in ('arm_failure','arm_padding'):raise ValueError('Unknown additive source')")
    s = replace_once(s, "    advantage=batch['arm_advantage'][0];ratio=(logp-old).exp()", """    if source == 'arm_padding':
        value = logp.sum()*0
        zero = value.detach()
        return value, dict(arm_mixed_loss=zero,arm_failure_loss=zero,
            arm_mixed_count=zero,arm_failure_count=zero,arm_mixed_kl_sum=zero,
            arm_mixed_clip_sum=zero,arm_failure_kl_sum=zero,arm_failure_clip_sum=zero)
    advantage=batch['arm_advantage'][0];ratio=(logp-old).exp()""")
    return {'openwebrl/arm_failure_aux.py': s.encode(),
            'openwebrl/arm_failure_dp.py': (REPO/'openwebrl/arm_failure_dp.py').read_bytes()}


def prepare_sources():
    from prepare_arm_gate_to60 import SOURCES as parents
    from prepare_arm_turn_bonus import copy_plain
    for variant, source in SOURCES.items():
        parent = parents[variant]
        validate_source(parent)
        changed = expected_files(variant)
        for name, data in changed.items():
            compile(data, name, 'exec')
        if not source.exists():
            shutil.copytree(parent, source, symlinks=True, copy_function=copy_plain,
                ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git', '.browser_use_sessions'))
            for name, data in changed.items():
                (source/name).write_bytes(data)
            manifest = json.loads((parent/'reference_manifest.json').read_text())
            manifest['recipe_files_sha256'].update({k: hashlib.sha256(v).hexdigest() for k,v in changed.items()})
            manifest['arm_dp_transport'] = dict(parent=str(parent),changed_files=list(changed),
                global_batch=256,microbatch=1,objective_unchanged=True)
            write_json(source/'reference_manifest.json',manifest)
        validate_source(source)
        manifest = json.loads((source/'reference_manifest.json').read_text())['recipe_files_sha256']
        original = json.loads((parent/'reference_manifest.json').read_text())['recipe_files_sha256']
        if set(manifest) != set(original)|set(changed):
            raise ValueError('Unexpected production source inventory')
        if any(manifest[k] != v for k,v in original.items() if k not in changed):
            raise ValueError('Unexpected scientific source change')
        if any((source/k).read_bytes() != v for k,v in changed.items()):
            raise ValueError('Frozen DP implementation changed')


def configure(variant, target):
    from prepare_arm_gate_to60 import configure as original
    original(variant,target)
    os.environ['ARM_VARIANT_TP_SIZE'] = '2'


def plan(variant, root, target, job, hours):
    from resume_arm_failure_variants import plan as continuation, validate_plan
    from run_arm_gate_checkpoint_eval import RUN_IDS, CREDIT
    configure(variant,target)
    p = continuation(job,root,hours,gpus=8)
    validate_plan(p)
    if (p['gate_ablation'] != variant or p['wandb_run_id'] != RUN_IDS[variant]
            or p['arm_config']['credit_assignment'] != CREDIT[variant]
            or p['arm_config']['candidate_gate'] != 'min2' or p['fresh_optimizer']
            or p['environment']['TP_SIZE'] != '2'):
        raise ValueError('TP2 continuation changed scientific lineage')
    return p


def request(job):
    path = CONTROL/f'{job}.json'
    if job not in JOBS or not path.exists():
        return None
    data = json.loads(path.read_text())
    if data['variant'] != JOBS[job] or data['job_id'] != job:
        raise ValueError('Migration request identity mismatch')
    return data


def execute(args):
    """The requeued batch controller awaits validation, training and both evals."""
    job=args.job_id; req=request(job)
    if not req or req['stage'] != 'released' or int(os.getenv('SLURM_RESTART_COUNT','0')) < 1:
        raise ValueError('TP2 controller requires its explicitly requeued job')
    if os.getenv('SLURM_JOB_ID') != job:
        raise ValueError('Wrong allocation')
    prepare_sources()
    hours=req['remaining_seconds']/3600
    allocation(subprocess.check_output(['scontrol','show','job',job,'-o'],text=True),job,
        requested_gpus=8,maximum_hours=hours)
    root=RUNTIME/f'evaluations/arm-gate-{args.variant.lower()}-to60-{job}-tp2'
    root.mkdir(exist_ok=False)
    old_controller=RUNTIME/f'evaluations/arm-gate-{args.variant.lower()}-to60-{job}'
    origin=Path(req['resume_from']); completed=[]
    step=['srun',f'--jobid={job}','--nodes=1','--ntasks=1','--cpus-per-task=64',
          '--gres=gpu:h200:8','--exact','--cpu-bind=none',sys.executable,str(Path(__file__).resolve()),
          '--variant',args.variant,'--job-id',job]
    try:
        write_json(root/'status.json',dict(stage='full-batch-validation',complete=False))
        subprocess.run([*step,'--worker','validate'],check=True)
        for target in (40,60):
            training_root=RUNTIME/f'evaluations/arm-failure-additive-{job}-tp2-iter{target}'
            controller_plan=dict(variant=args.variant,target=60,current_target=target,training_root=str(training_root),
                resume_from=str(origin),job_id=job,evaluation_in_allocation=True,topology='TP2/DP4/microbatch1')
            write_json(root/'controller-plan.json',controller_plan)
            write_json(old_controller/'controller-plan.json',controller_plan)
            write_json(root/'status.json',dict(stage='training',complete=False,current_target=target,completed_stages=completed))
            subprocess.run([*step,'--worker','train','--target',str(target),'--resume-from',str(origin)],check=True)
            from run_arm_gate_checkpoint_eval import checkpoint_ready
            if checkpoint_ready(args.variant,training_root,target) is None:
                write_json(root/'status.json',dict(stage='budget-stop',complete=False,completed_stages=completed,
                    reason='Saved partial progress; no compute extension'))
                return
            write_json(root/'status.json',dict(stage='evaluation',complete=False,current_target=target))
            subprocess.run([*step,'--worker','eval','--target',str(target)],check=True)
            from run_stage1_to100 import audit_rollouts
            from prepare_arm_gate_to60 import evaluation_plan
            e=evaluation_plan(args.variant,training_root,target,job)
            write_json(root/f'evaluation-after{target}.json',audit_rollouts(Path(e['output'])/'rollouts',Path(e['source'])))
            completed.append(target); origin=training_root
        write_json(root/'status.json',dict(stage='complete',complete=True,completed_stages=completed))
    except BaseException as exc:
        write_json(root/'status.json',dict(stage='failed',complete=False,error_type=type(exc).__name__,error=str(exc)[:300]))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant',choices=('B','C'))
    parser.add_argument('--job-id')
    parser.add_argument('--target',type=int,choices=(40,60))
    parser.add_argument('--resume-from',type=Path)
    parser.add_argument('--worker',choices=('validate','train','eval'))
    args=parser.parse_args()
    if not args.worker:
        prepare_sources();print(json.dumps({k:str(v) for k,v in SOURCES.items()}));return
    req=request(args.job_id)
    if os.getenv('SLURM_JOB_ID') != args.job_id or not req or req['stage'] != 'released':
        raise ValueError('Authorized requeued allocation required')
    if args.worker == 'validate':
        from validate_arm_gate_dp import execute as validate
        validate(args.job_id,args.variant)
    elif args.worker == 'train':
        from resume_arm_failure_variants import execute as train
        train(plan(args.variant,args.resume_from,args.target,args.job_id,req['remaining_seconds']/3600))
        # This worker remains owned and awaited by the batch controller. Evaluate
        # saved intermediate milestones even when training stopped below target.
        from run_arm_milestone_queue import run_due_evaluations
        training_root=RUNTIME/f'evaluations/arm-failure-additive-{args.job_id}-tp2-iter{args.target}'
        run_due_evaluations(args.variant,training_root,args.target,args.job_id,skip_unavailable=True)
    else:
        from prepare_arm_gate_to60 import evaluation_plan
        from run_arm_milestone_queue import run_due_evaluations
        import evaluate_baseline_checkpoint as evaluator
        os.environ.update(OPENWEBRL_EXPECTED_SCHEDULER_OFFSET='0',OPENWEBRL_RECORD_EVAL_EXPECTED_TASK_COUNT='300')
        training_root=RUNTIME/f'evaluations/arm-failure-additive-{args.job_id}-tp2-iter{args.target}'
        run_due_evaluations(args.variant,training_root,args.target,args.job_id)
        evaluator.run(evaluation_plan(args.variant,training_root,args.target,args.job_id),REPO/'.env')


if __name__ == '__main__':main()
