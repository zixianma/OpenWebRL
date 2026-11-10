#!/usr/bin/env python3
"""Approved two-H200 hour: isolated nonempty auxiliary update and GPU restore."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

from resume_baseline import REPO, RUNTIME, clean_environment, source_command, validate_source, write_json
import prepare_arm_failure_ablations as ablations

SOURCE = RUNTIME/'reference-arm-failure-gpu-diagnostic-20260922-v1'
CONTROL = RUNTIME/'arm-turn-bonus-preparation/failure-gpu-diagnostic-20260922'
PYTHON = RUNTIME/'venv/bin/python'

GPU_PROBE = '''    if os.getenv('ARM_FAILURE_GPU_DIAGNOSTIC') == '1' and not globals().get('_diagnostic_beta_checked'):
        doubled = -reducer(torch.minimum(ratio*(2*advantage),
            ratio.clamp(1-args.eps_clip,1+args.eps_clip_high)*(2*advantage)))*batch['arm_scale'][0]
        first_grad = torch.autograd.grad(value,logp,retain_graph=True)[0]
        second_grad = torch.autograd.grad(doubled,logp,retain_graph=True)[0]
        if (not bool(torch.isfinite(first_grad).all()) or not bool(torch.isfinite(second_grad).all())
                or not bool(first_grad.norm()>0) or not torch.allclose(second_grad,2*first_grad,rtol=.02,atol=1e-6)
                or not torch.allclose(doubled,2*value,rtol=.02,atol=1e-6)):
            raise ValueError('GPU beta=1 loss/gradient scaling failed')
        from openwebrl.arm_turn_bonus import write_json
        config=json.loads(Path(os.environ['OPENWEBRL_ARM_TURN_BONUS_CONFIG']).read_text())
        rank=torch.distributed.get_rank()
        write_json(Path(config['output'])/f'beta-scale-gpu-rank{rank}.json',dict(passed=True,
            diagnostic_only=True,gradient_norm_beta_half=float(first_grad.norm()),
            gradient_norm_beta_one=float(second_grad.norm()),ratio=float(second_grad.norm()/first_grad.norm()),
            scope='Native auxiliary loss derivative wrt actual GPU actor log probabilities; no second training experiment'))
        globals()['_diagnostic_beta_checked']=True
'''


def prepare_source():
    from prepare_arm_turn_bonus import copy_plain
    validate_source(ablations.SOURCE)
    module='openwebrl/arm_failure_aux.py'
    original=(ablations.SOURCE/module).read_text()
    anchor="    return value,dict(arm_mixed_loss=zero,arm_failure_loss=value.detach(),\n"
    if original.count(anchor)!=1:raise ValueError('Native auxiliary diagnostic boundary moved')
    changed=original.replace(anchor,GPU_PROBE+anchor)
    if not SOURCE.exists():
        shutil.copytree(ablations.SOURCE,SOURCE,symlinks=True,copy_function=copy_plain,
            ignore=shutil.ignore_patterns('__pycache__','*.pyc','.git','.browser_use_sessions'))
        (SOURCE/module).write_text(changed)
        manifest=json.loads((ablations.SOURCE/'reference_manifest.json').read_text())
        manifest['recipe_files_sha256'][module]=hashlib.sha256(changed.encode()).hexdigest()
        manifest['gpu_diagnostic']=dict(parent=str(ablations.SOURCE),diagnostic_only=True,
            change='Assert beta scaling on actual GPU log probabilities; returns original native loss')
        write_json(SOURCE/'reference_manifest.json',manifest)
    validate_source(SOURCE)
    if (SOURCE/module).read_text()!=changed:raise ValueError('Diagnostic instrumentation changed')


def plan(job='PREPARE'):
    p=ablations.plan('coverage',job)
    old=p['output'];out=RUNTIME/f'evaluations/arm-failure-gpu-diagnostic-{job}'
    p=json.loads(json.dumps(p).replace(old,str(out)).replace(str(ablations.SOURCE),str(SOURCE)))
    run=out.name
    p.update(source=str(SOURCE),output=str(out),experiment='isolated-failure-gpu-diagnostic',
        failure_ablation_training=False,diagnostic_only=True,requested_iterations=1,
        target_completed_iterations=1,evaluation_in_allocation=False,evaluation_reserve_seconds=0,
        wandb_run_id=run,requested_resources=dict(gpus=2,gpu_type='H200',hours=.7,gpu_hours=1.4,
            cpus=16,memory_gib=480,browsers=2))
    p['environment'].update(NUM_GPUS='2',TP_SIZE='2',NUM_ROLLOUT='1',
        BROWSER_CONCURRENCY='2',SGLANG_CONCURRENCY='2',WANDB_PROJECT='openwebrl-evals',
        WANDB_RUN_ID=run,WANDB_RESUME='never',ARM_FAILURE_GPU_DIAGNOSTIC='1',
        PYTHONPATH=str(REPO/'scripts'),OPENWEBRL_STREAMING_CHECKPOINT='1',
        SLIME_BROWSER_ENV_MODE='local_process',SLIME_BROWSER_LOCAL_PROCESS_PYTHON=str(PYTHON),
        SLIME_BROWSER_LOCAL_PROCESS_MAX_PROCESSES='2',SLIME_BROWSER_ROLLOUT_CONCURRENCY='2')
    p['browser_config']['browser_rollout_concurrency']=2
    p['arm_config'].update(run_id=run,run_output=str(out),output=str(out),policy_id=run+':uninitialized',
        diagnostic_only=True,minimum_cycle_seconds=1200,seconds_per_optimizer_update=90)
    p['command'][p['command'].index('--wandb-project')+1]='openwebrl-evals'
    p['command'][p['command'].index('--wandb-group')+1]='ARM nonempty failure GPU diagnostic'
    argv=shlex.split(subprocess.check_output(p['command'],env=dict(clean_environment(),
        **p['environment'],DRY_RUN='1'),text=True))
    p['command']=[str(PYTHON),str(SOURCE/'train.py'),*argv[2:],
        '--rollout-function-path','arm_failure_gpu_fixture.generate_rollout']
    p['comparison']=dict(diagnostic_only=True,allocation_gpu_hours=2,
        synthetic_outcomes=True,controlled_auxiliary_candidates=True,actual_actor_and_arm=True,
        expected_native_optimizer_updates=2,expected_nonempty_auxiliary_groups=1,
        gpu_restore_after_training=True,
        limits='TP2 plumbing only; no natural admission yield, performance gain, or TP8 throughput claim')
    return p


def fingerprint():
    files=[Path(__file__).resolve(),REPO/'scripts/arm_failure_gpu_fixture.py',
        REPO/'scripts/arm_failure_gpu_controller.py',REPO/'scripts/run_arm_failure_gpu_diagnostic.sbatch',
        SOURCE/'reference_manifest.json']
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def validate_plan(p):
    if p!=plan(p['job_id']):raise ValueError('Diagnostic plan changed')
    receipt=json.loads((CONTROL/'readiness.json').read_text())
    if not receipt.get('native_parse_passed') or receipt['fingerprint']!=fingerprint():
        raise ValueError('Diagnostic readiness missing/stale')
    return receipt


def check():
    prepare_source();CONTROL.mkdir(parents=True,exist_ok=True)
    p=plan();write_json(CONTROL/'plan.json',p)
    write_json(CONTROL/'browser.json',p['browser_config'])
    command=list(p['command']);command[command.index('--custom-config-path')+1]=str(CONTROL/'browser.json')
    probe=CONTROL/'native-check.py'
    probe.write_text('''import json
from pathlib import Path
from unittest.mock import patch
from slime.utils.arguments import parse_args
with patch('megatron.training.arguments.get_device_arch_version',return_value=9):args=parse_args()
expected=dict(actor_num_gpus_per_node=2,tensor_model_parallel_size=2,start_rollout_id=0,
    num_rollout=1,global_batch_size=256,ppo_epochs=2,rollout_batch_size=48,
    wandb_project='openwebrl-evals',rollout_function_path='arm_failure_gpu_fixture.generate_rollout')
assert all(getattr(args,k)==v for k,v in expected.items()),{k:getattr(args,k) for k in expected}
from openwebrl.arm_failure_aux import check_topology
check_topology(args)
from arm_failure_gpu_fixture import diagnostic_task
from playwright.async_api import async_playwright
print('NATIVE_DIAGNOSTIC_PARSE_OK',json.dumps(expected))
''')
    with (CONTROL/'native-check.log').open('w') as log:
        subprocess.run(source_command(SOURCE,[str(PYTHON),str(probe),*command[2:]]),cwd=SOURCE,
            env=dict(clean_environment(),**p['environment'],CUDA_VISIBLE_DEVICES=''),
            stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
    subprocess.run(['bash','-n',str(REPO/'scripts/run_arm_failure_gpu_diagnostic.sbatch')],check=True)
    for filename in ['run_arm_failure_gpu_diagnostic.py','arm_failure_gpu_fixture.py','arm_failure_gpu_controller.py']:
        ast.parse((REPO/'scripts'/filename).read_text())
    write_json(CONTROL/'readiness.json',dict(native_parse_passed=True,fingerprint=fingerprint(),
        compute_approved=True,approved_resources=dict(gpus=2,hours=1,cpus=16,memory_gib=480),
        gpu_validation_pending=True))
    print('DIAGNOSTIC_CPU_READY')


def execute(job):
    from check_storage_quota import check_storage_quota
    from arm_failure_gpu_controller import execute as execute_training
    p=plan(job);validate_plan(p)
    write_json(CONTROL/f'quota-{job}.json',check_storage_quota())
    root=Path(p['output'])
    # The controller owns and awaits training, then restoration in this srun.
    execute_training(p)
    status=json.loads((root/'status.json').read_text())
    iteration=root/'iterations/0000'
    auxiliary=json.loads((iteration/'failure_auxiliary.json').read_text())
    beta_reports=[json.loads(path.read_text()) for path in iteration.glob('beta-scale-gpu-rank*.json')]
    if (not status.get('training_complete') or status.get('completed_optimizer_updates')!=2
            or auxiliary['failure_groups']!=1 or not auxiliary['records']
            or len(beta_reports)!=2 or not all(r['passed'] for r in beta_reports)):
        raise ValueError('Nonempty native GPU update/gradient checks did not complete')
    write_json(root/'diagnostic-status.json',dict(stage='gpu-restore',complete=False))
    with (root/'restore-controller.log').open('w') as log:
        subprocess.run([str(PYTHON),str(REPO/'scripts/check_arm_gpu_checkpoint_restore.py'),
            '--training-root',str(root),'--job-id',job,'--gpus','2',
            '--output',str(root/'gpu-restore-check'),'--execute'],
            stdout=log,stderr=subprocess.STDOUT,check=True,timeout=950)
    restored=json.loads((root/'gpu-restore-check/result.json').read_text())
    if not restored.get('passed'):raise ValueError('GPU restoration failed')
    write_json(root/'diagnostic-result.json',dict(passed=True,diagnostic_only=True,
        completed_optimizer_updates=2,auxiliary_groups=1,auxiliary_labels=len(auxiliary['records']),
        beta_gradient_checks=beta_reports,gpu_restore=restored,
        browser_artifact=str(iteration/'browser-replay.json'),limitations=p['comparison']['limits']))
    write_json(root/'diagnostic-status.json',dict(stage='complete',complete=True))
    print(json.dumps(dict(job=job,passed=True,report=str(root/'diagnostic-result.json'))))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true');parser.add_argument('--execute',action='store_true')
    parser.add_argument('--job-id',default='PREPARE');args=parser.parse_args()
    if args.check:check()
    elif args.execute:
        try:execute(args.job_id)
        except BaseException as exc:
            root=RUNTIME/f'evaluations/arm-failure-gpu-diagnostic-{args.job_id}'
            if root.exists():
                write_json(root/'diagnostic-status.json',dict(stage='failed',complete=False,
                    failed=True,error_type=type(exc).__name__,error=str(exc)[:500]))
            raise
    else:print(json.dumps(plan(args.job_id),indent=2))
