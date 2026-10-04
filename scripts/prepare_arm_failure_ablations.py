#!/usr/bin/env python3
"""Prepare independent weight/coverage experiments from iteration zero; no sbatch."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from resume_baseline import REPO,RUNTIME,allocation,source_command,validate_source,write_json
import run_arm_turn_bonus_cycles as training
import prepare_arm_failure_coverage as pilot
import run_arm_failure_additive as additive

SOURCE=RUNTIME/'reference-arm-failure-ablations-20260922-v3'
CONTROL=RUNTIME/'arm-turn-bonus-preparation/failure-ablations-fromzero-20260922'
TARGET=20
VARIANTS=('weight','coverage')
RESOURCES=dict(gpus=8,hours=24,gpu_hours=192,cpus=64,memory_gib=960,browsers=64)


def prepare_source():
    from prepare_arm_turn_bonus import copy_plain
    from run_arm_gate_ablation import replace_definitions
    parent=pilot.SOURCE;validate_source(parent)
    name='openwebrl/arm_turn_bonus_runtime.py'
    text=replace_definitions((parent/name).read_text(),(REPO/name).read_text(),['record_completed_group'])
    anchor="    decision = calibration_decision(report)\n"
    if text.count(anchor)!=1:raise ValueError('Optimizer gate boundary changed')
    # Empty eligible pools are legitimate zero-auxiliary batches. Every nonempty
    # pool must finish deferred labeling before native optimizer updates.
    hook=("    if config.get('failure_ablation') == 'coverage':\n"
          "        coverage = report.get('failure_coverage') or {}\n"
          "        exercised = coverage.get('failure_coverage', {})\n"
          "        decision['checks']['complete_failure_coverage'] = bool(coverage.get('complete')) and (exercised.get('groups') == 0 or exercised.get('labels', 0) > 0)\n"
          "        decision['passed'] = all(decision['checks'].values())\n")
    changes={name:(text.replace(anchor,anchor+hook,1)).encode()}
    for module in pilot.MODULES:
        changes[module]=(REPO/module).read_bytes()
    name='openwebrl/arm_turn_bonus.py'
    changes[name]=replace_definitions((parent/name).read_text(),(REPO/name).read_text(),['ShadowSelector']).encode()
    # Preserve the exception type for future audits; this does not change statuses/rewards.
    name='openwebrl/generate_browser.py';text=(parent/name).read_text()
    old='env.step() failed, marking as ABORTED: {step_err}'
    if old not in text:raise ValueError('Browser diagnostic boundary changed')
    changes[name]=text.replace(old,'env.step() failed, marking as ABORTED: {type(step_err).__name__}: {step_err}').encode()
    if not SOURCE.exists():
        shutil.copytree(parent,SOURCE,symlinks=True,copy_function=copy_plain,
            ignore=shutil.ignore_patterns('__pycache__','*.pyc','.git','.browser_use_sessions'))
        for name,data in changes.items():(SOURCE/name).write_bytes(data)
        manifest=json.loads((parent/'reference_manifest.json').read_text())
        manifest['recipe_files_sha256'].update({n:hashlib.sha256(data).hexdigest() for n,data in changes.items()})
        manifest['failure_ablations']=dict(parent=str(parent),changed_files=list(changes),
            each_nonempty_failure_pool_requires_completed_usable_labels=True,failure_sampling='Bernoulli0.4 within historically admitted groups; mixed q0.2',
            other_changes='diagnostic logging and bounded serial auxiliary capacity')
        write_json(SOURCE/'reference_manifest.json',manifest)
    validate_source(SOURCE)
    old=json.loads((parent/'reference_manifest.json').read_text())['recipe_files_sha256']
    new=json.loads((SOURCE/'reference_manifest.json').read_text())['recipe_files_sha256']
    if (any((SOURCE/n).read_bytes()!=v for n,v in changes.items()) or
        any(new.get(n)!=v for n,v in old.items() if n not in changes)):
        raise ValueError('Unplanned frozen source change')


def plan(variant,job='PREPARE'):
    if variant not in VARIANTS:raise ValueError('Unknown failure ablation')
    prepare_source()
    p=additive.plan(job,minutes=1440)
    old=p['output'];out=RUNTIME/f'evaluations/arm-failure-additive-{job}'
    run=f'arm-failure-{variant}-fromzero-{job}'
    old_source=p['source']
    rewrite=lambda s:s.replace(old,str(out)).replace(old_source,str(SOURCE))
    p['command']=[rewrite(s) for s in p['command']]
    p['environment']={k:rewrite(v) for k,v in p['environment'].items()}
    p['arm_config']={k:rewrite(v) if isinstance(v,str) else v for k,v in p['arm_config'].items()}
    p.update(source=str(SOURCE),output=str(out),wandb_run_id=run,requested_iterations=TARGET,
        start_rollout_id=0,initial_optimizer_updates=0,fresh_optimizer=True,
        checkpoint=str(training.INITIAL),
        target_completed_iterations=TARGET,failure_coverage_pilot=False,diagnostic_only=False,
        diagnostic_live_shadow=False,variant_continuation=False,failure_ablation_training=variant,
        evaluation_in_allocation=True,evaluation_reserve_seconds=3600,compute_approved=True,
        requested_resources=dict(RESOURCES),comparison_wandb_run='arm-failure-additive-295786')
    p['environment'].update(NUM_GPUS='8',TP_SIZE='8',NUM_ROLLOUT=str(TARGET),
        HF_CHECKPOINT=str(training.INITIAL),SLIME_LOAD_CHECKPOINT=str(training.INITIAL),
        WANDB_PROJECT='openwebrl',WANDB_RESUME='never',WANDB_RUN_ID=run,
        BROWSER_CONCURRENCY='64',SGLANG_CONCURRENCY='64')
    p['browser_config']['browser_rollout_concurrency']=64
    for name,value in [('--wandb-project','openwebrl'),('--wandb-group',f'ARM additive from iteration0 | failure {variant}')]:
        p['command'][p['command'].index(name)+1]=value
    p['arm_config'].update(run_id=run,run_output=str(out),output=str(out),policy_id=run+':uninitialized',
        shadow_only=False,train_after_calibration=True,checkpoint=str(training.INITIAL),
        failure_ablation=variant,failure_beta=1.0 if variant=='weight' else .5,
        failure_turn_budget=0,failure_scored_fraction=.4 if variant=='coverage' else .2,
        failure_ablation_start_rollout_id=0)
    scratch=RUNTIME/'multimodal-scratch'/f'arm-variant-{job}'
    p['environment']['OPENWEBRL_MULTIMODAL_STORAGE_DIR']=str(scratch)
    p['multimodal_storage']=dict(mode='shared',directory=str(scratch),
        minimum_free_bytes=2*1024**4,retention='preserve; no automatic deletion')
    p['comparison']=dict(initialization=str(training.INITIAL),fresh_optimizer_scheduler_cursor=True,
        variant=variant,mixed_beta=.5,mixed_q=.2,failure_beta=p['arm_config']['failure_beta'],
        failure_turn_budget=0,failure_q=p['arm_config']['failure_scored_fraction'],
        admission='unchanged historical q0.2 usable-label admission; extra labels only in admitted groups',
        source_gate='five valid failed trajectories; distinct5 valid actions; response-index credit',
        endpoint=TARGET,fresh_wandb_lineage=True,
        limit='No automatic extension; require exact20 checkpoint before endpoint evaluation',
        control='Historical additive run from the same SFT initialization; not a simultaneous multi-seed control')
    validate_initialization(p)
    return p


def validate_initialization(p):
    forbidden=('resume_from','resume_origin','resume_restore_receipt','gpu_restore_output')
    if (any(k in p for k in forbidden) or p.get('fresh_optimizer') is not True
            or p.get('start_rollout_id')!=0 or p.get('initial_optimizer_updates')!=0
            or p['checkpoint']!=str(training.INITIAL)
            or p['environment'].get('SLIME_LOAD_CHECKPOINT')!=str(training.INITIAL)
            or p['environment'].get('HF_CHECKPOINT')!=str(training.INITIAL)
            or p['arm_config'].get('checkpoint')!=str(training.INITIAL)
            or p['arm_config'].get('failure_ablation_start_rollout_id')!=0
            or p['environment'].get('WANDB_RESUME')!='never'
            or any(k.startswith(('OPENWEBRL_REPLAY','SLIME_CKPT')) for k in p['environment'])
            or '--use-checkpoint-opt-param-scheduler' in p['command']
            or (training.INITIAL/'latest_checkpointed_iteration.txt').exists()):
        raise ValueError('New interventions must start from original SFT at iteration0 with fresh optimizer and cursor')


def fingerprint():
    files=[SOURCE/'reference_manifest.json',Path(__file__).resolve(),
        REPO/'scripts/run_arm_turn_bonus_cycles.py',REPO/'scripts/check_arm_native_launch.py',
        REPO/'scripts/run_arm_failure_additive.py',
        REPO/'scripts/check_storage_quota.py',REPO/'scripts/resume_arm_failure_variants.py',
        REPO/'scripts/run_arm_failure_ablations_8gpu.sbatch',REPO/'tests/test_arm_failure_ablation_launch.py',
        training.INITIAL/'config.json',training.INITIAL/'tokenizer_config.json']
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def validate_plan(p):
    if p.get('ablation_dp'):
        from prepare_arm_ablation_dp import validate_plan as validate_dp
        return validate_dp(p)
    validate_initialization(p)
    if p!=plan(p['failure_ablation_training'],p['job_id']):raise ValueError('Ablation launch plan changed')
    receipt=json.loads((CONTROL/'readiness.json').read_text())
    if (not receipt.get('cpu_passed') or receipt.get('fingerprint')!=fingerprint()
            or receipt.get('native_parse_passed')!=list(VARIANTS)):
        raise ValueError('Missing or stale CPU readiness')
    return dict(passed=True,variant=p['failure_ablation_training'],cpu_passed=True,
        starts_at_iteration_zero=True,fresh_optimizer_scheduler_cursor=True,gpu_initialization_pending=True)


def evaluation_plan(variant,training_root,job,iteration=TARGET):
    import run_arm_iteration80_eval as evaluator
    p=evaluator.plan('additive',job,iteration,training_root)
    p['gpus']=8;p['environment'].update(NUM_GPUS='8',TP_SIZE='8',BROWSER_CONCURRENCY='32',SGLANG_CONCURRENCY='32')
    run=f'arm-failure-{variant}-iter{iteration}-{job}'
    p.update(wandb_run_id=run,arm_variant=f'failure-{variant}',parent_training_run=f'arm-failure-{variant}-fromzero-{job}')
    p['environment']['WANDB_RUN_ID']=run
    p['command'][p['command'].index('--wandb-group')+1]=f'ARM failure {variant} | iter{iteration} full300'
    return evaluator.evaluation.configure_evaluation_tracking(p,'openwebrl-evals')


def train_worker(variant,job):
    from resume_arm_failure_variants import check_multimodal_storage
    p=plan(variant,job);validate_plan(p);check_multimodal_storage(p)
    launch=RUNTIME/f'logs/arm-failure-ablation-{job}.json';write_json(launch,p)
    with (RUNTIME/f'logs/arm-failure-ablation-monitor-{job}.log').open('w') as log:
        monitor=subprocess.Popen([sys.executable,str(REPO/'scripts/monitor_arm_turn_bonus.py'),
            '--job-id',job,'--watch','--interval','900','--hours','24'],stdout=log,stderr=subprocess.STDOUT)
        try:training.execute(p)
        finally:
            monitor.terminate()
            try:monitor.wait(timeout=30)
            except subprocess.TimeoutExpired:monitor.kill();monitor.wait()


def execute(variant,job):
    from prepare_arm_ablation_dp import request as dp_request, execute as execute_dp
    migration=dp_request(job)
    if migration and migration['stage'] in ('released','armed-fresh'):
        return execute_dp(variant,job)
    if os.getenv('SLURM_JOB_ID')!=job:raise ValueError('Authorized allocation required')
    allocation(subprocess.check_output(['scontrol','show','job',job,'-o'],text=True),job,requested_gpus=8,maximum_hours=24)
    validate_plan(plan(variant,job))
    root=RUNTIME/f'evaluations/arm-failure-ablation-{variant}-{job}';root.mkdir(exist_ok=False)
    train_root=RUNTIME/f'evaluations/arm-failure-additive-{job}'
    write_json(root/'controller-plan.json',dict(job_id=job,variant=variant,target=TARGET,start=0,
        training_root=str(train_root),resources=RESOURCES,evaluation_in_allocation=True))
    step=['srun',f'--jobid={job}','--nodes=1','--ntasks=1','--cpus-per-task=64',
        '--gres=gpu:h200:8','--exact','--cpu-bind=none',sys.executable,str(Path(__file__).resolve()),
        '--variant',variant,'--job-id',job]
    try:
        write_json(root/'status.json',dict(complete=False,stage='training'))
        subprocess.run([*step,'--worker','train'],check=True)
        from run_arm_iteration80_eval import checkpoint_ready
        if checkpoint_ready('additive',TARGET,train_root) is None:
            raise ValueError('Training stopped before20; preserve progress; do not substitute an earlier checkpoint')
        write_json(root/'status.json',dict(complete=False,stage='evaluation'))
        subprocess.run([*step,'--worker','eval'],check=True)
        from run_stage1_to100 import audit_rollouts
        p=evaluation_plan(variant,train_root,job)
        result=audit_rollouts(Path(p['output'])/'rollouts',Path(p['source']))
        write_json(root/'evaluation-audit.json',result)
        write_json(root/'status.json',dict(complete=True,target=TARGET,evaluation=result))
    except BaseException as exc:
        write_json(root/'status.json',dict(complete=False,error_type=type(exc).__name__,error=str(exc)[:300]));raise


def prepare():
    prepare_source();CONTROL.mkdir(parents=True,exist_ok=True)
    stages=[]
    for variant in VARIANTS:
        p=plan(variant);write_json(CONTROL/f'{variant}-plan.json',p)
        # Verify the evaluator using a real historical iteration20 checkpoint;
        # this is an evaluation-only preview, never a training initialization.
        preview=evaluation_plan(variant,RUNTIME/'evaluations/arm-failure-additive-295834','PREPARE',20)
        write_json(CONTROL/f'{variant}-eval-preview.json',preview)
        stages.append(dict(variant=variant,resources=RESOURCES,from_iteration=0,target_iteration=TARGET,
            evaluation='full300 GPT-4.1 T0 with saved rollouts/verdicts in same allocation'))
    subprocess.run(['bash','-n',str(REPO/'scripts/run_arm_failure_ablations_8gpu.sbatch')],check=True)
    write_json(CONTROL/'proposal.json',dict(approved=True,submitted=False,stages=stages,max_gpu_hours=384,
        estimated_hours_per_branch='Twenty fresh collections; throughput and coverage overhead must be measured within the24h cap',
        coverage_startup_gate='each nonempty eligible pool must finish deferred labeling before optimizer updates; empty pools remain ordinary updates'))
    return CONTROL


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant',choices=VARIANTS);parser.add_argument('--job-id',default='PREPARE')
    parser.add_argument('--execute',action='store_true');parser.add_argument('--worker',choices=('train','eval'))
    args=parser.parse_args()
    if args.execute or args.worker:
        if not args.variant or os.getenv('SLURM_JOB_ID')!=args.job_id:raise ValueError('Authorized variant/allocation required')
        if args.worker=='train':train_worker(args.variant,args.job_id)
        elif args.worker=='eval':
            import evaluate_baseline_checkpoint as evaluator
            os.environ.update(OPENWEBRL_EXPECTED_SCHEDULER_OFFSET='0',OPENWEBRL_RECORD_EVAL_EXPECTED_TASK_COUNT='300')
            evaluator.run(evaluation_plan(args.variant,RUNTIME/f'evaluations/arm-failure-additive-{args.job_id}',args.job_id),REPO/'.env')
        else:execute(args.variant,args.job_id)
    else:print(prepare())
