#!/usr/bin/env python3
"""Budget-preserving TP2 migration for the approved beta/q40 allocations."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

from resume_baseline import REPO,RUNTIME,allocation,clean_environment,source_command,validate_source,write_json

CONTROL=RUNTIME/'arm-turn-bonus-preparation/ablation-tp2-migration-20260922'
JOBS={'318949':'weight','318950':'coverage'}
SOURCE=RUNTIME/'reference-arm-failure-ablations-tp2-20260922-v1'
VALIDATION_SOURCE=RUNTIME/'reference-arm-failure-ablations-dp-validation-20260922-v1'
ORIGIN=RUNTIME/'evaluations/arm-failure-additive-318949'
PYTHON=RUNTIME/'venv/bin/python'


def prepare_source():
    from prepare_arm_failure_ablations import SOURCE as parent
    from prepare_arm_gate_dp import dp_files
    from prepare_arm_turn_bonus import copy_plain
    validate_source(parent);changed=dp_files(parent)
    if not SOURCE.exists():
        shutil.copytree(parent,SOURCE,symlinks=True,copy_function=copy_plain,
            ignore=shutil.ignore_patterns('__pycache__','*.pyc','.git','.browser_use_sessions'))
        manifest=json.loads((parent/'reference_manifest.json').read_text())
        for name,data in changed.items():
            compile(data,name,'exec');(SOURCE/name).write_bytes(data)
            manifest['recipe_files_sha256'][name]=hashlib.sha256(data).hexdigest()
        manifest['arm_dp_transport']=dict(parent=str(parent),changed_files=list(changed),
            global_batch=256,microbatch=1,objective_unchanged=True)
        write_json(SOURCE/'reference_manifest.json',manifest)
    validate_source(SOURCE)
    old=json.loads((parent/'reference_manifest.json').read_text())['recipe_files_sha256']
    new=json.loads((SOURCE/'reference_manifest.json').read_text())['recipe_files_sha256']
    if (set(new)!=set(old)|set(changed) or any(new[n]!=h for n,h in old.items() if n not in changed)
            or any((SOURCE/n).read_bytes()!=v for n,v in changed.items())):
        raise ValueError('Unexpected scientific/source change in DP migration')


def request(job):
    p=CONTROL/f'{job}.json'
    if job not in JOBS or not p.exists():return None
    data=json.loads(p.read_text())
    if data['job_id']!=job or data['variant']!=JOBS[job]:raise ValueError('Migration identity mismatch')
    return data


def configure():
    for k in ('ARM_VARIANT_PORT_SOURCE','ARM_CONTINUATION_LABEL_GUARD'):
        os.environ.pop(k,None)
    os.environ.update(ARM_VARIANT_TP_SIZE='2',ARM_VARIANT_OUTPUT_STAGE='iter20',
        ARM_VARIANT_TARGET_ITERATION='20',ARM_VARIANT_EVAL_RESERVE_SECONDS='3600',
        ARM_VARIANT_MULTIMODAL_STORAGE='shared')


def plan(variant,job,hours,resume_from=None):
    if variant not in ('weight','coverage') or not 0<hours<=24:raise ValueError('Unknown recipe/budget')
    prepare_source()
    if resume_from:
        from resume_arm_failure_variants import plan as resume_plan
        configure();p=resume_plan(job,Path(resume_from),hours,gpus=8)
    else:
        from prepare_arm_failure_ablations import plan as fresh_plan
        p=fresh_plan(variant,job)
        old_source=p['source'];old_output=p['output']
        output=RUNTIME/f'evaluations/arm-failure-additive-{job}-tp2-iter20'
        rewrite=lambda v:v.replace(old_source,str(SOURCE)).replace(old_output,str(output))
        p['command']=[rewrite(v) for v in p['command']]
        p['environment']={k:rewrite(v) for k,v in p['environment'].items()}
        p['arm_config']={k:rewrite(v) if isinstance(v,str) else v for k,v in p['arm_config'].items()}
        p.update(source=str(SOURCE),output=str(output),ablation_dp=True,
            topology_continuation='TP2/DP4/microbatch1')
        p['environment']['TP_SIZE']='2'
        p['requested_resources'].update(hours=hours,gpu_hours=8*hours)
    c=p['arm_config']
    expected=(1.,.2) if variant=='weight' else (.5,.4)
    if (p['failure_ablation_training']!=variant or (c['failure_beta'],c['failure_scored_fraction'])!=expected
            or (c['beta'],c['scored_fraction'],c['failure_group_cap'])!=(.5,.2,8)
            or c.get('candidate_gate','distinct5')!='distinct5'
            or c.get('credit_assignment','response_index')!='response_index'
            or p['wandb_run_id']!=f'arm-failure-{variant}-fromzero-{job}'
            or p['environment']['TP_SIZE']!='2'):
        raise ValueError('DP migration changed scientific recipe or run identity')
    return p


def fingerprints():
    names=('prepare_arm_ablation_dp.py','prepare_arm_failure_ablations.py','prepare_arm_gate_dp.py',
        'validate_arm_gate_dp.py','resume_arm_failure_variants.py','resume_arm_turn_bonus.py',
        'run_arm_turn_bonus_cycles.py','check_arm_gpu_checkpoint_restore.py','requeue_arm_ablation_dp.py')
    paths=[REPO/'scripts'/n for n in names]+[SOURCE/'reference_manifest.json',VALIDATION_SOURCE/'reference_manifest.json',
        REPO/'openwebrl/arm_failure_dp.py',REPO/'openwebrl/arm_dp_validation.py']
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def require_ready():
    receipt=json.loads((CONTROL/'readiness.json').read_text())
    if not receipt.get('cpu_passed') or receipt['fingerprints']!=fingerprints():
        raise ValueError('Missing/stale DP migration readiness')
    return receipt


def native_parse(p,label,diagnostic=False):
    """Parse the real launch argv and scheduler on CPU; no models or browsers."""
    source=Path(p['source']);out=CONTROL/f'parse-{label}';out.mkdir(exist_ok=True)
    browser=p.get('browser_config') or json.loads((CONTROL/'browser.json').read_text())
    write_json(out/'browser.json',browser)
    env=dict(clean_environment(),**p['environment'])
    env.update(BROWSER_TRAIN_CONFIG=str(out/'browser.json'),
        CUDA_VISIBLE_DEVICES='',JUDGE_API_MODE='served',JUDGE_API_BASE='https://api.openai.com/v1')
    if diagnostic:argv=p['command']
    else:argv=shlex.split(subprocess.check_output(p['command'],env=dict(env,DRY_RUN='1'),text=True))
    probe=out/'parse.py'
    probe.write_text('''import sys
from pathlib import Path
from unittest.mock import patch
from slime.utils.arguments import parse_args
with patch('megatron.training.arguments.get_device_arch_version',return_value=9):a=parse_args()
assert a.actor_num_gpus_per_node==8 and a.tensor_model_parallel_size==2
assert a.global_batch_size==256 and a.micro_batch_size==1 and a.ppo_epochs==2
from openwebrl.arm_failure_aux import check_topology
check_topology(a,dp=4)
sys.path.insert(0,"'''+str(REPO/'scripts')+'''")
import check_arm_native_launch as check
check.SOURCE=Path("'''+str(source)+'''")
check.check_native_actor_entry(a)
if a.use_checkpoint_opt_param_scheduler:check.check_native_scheduler_resume(a)
print('ABLATION_DP_NATIVE_PARSE_PASSED')
''')
    with (out/'parse.log').open('w') as log:
        subprocess.run(source_command(source,[str(PYTHON),str(probe),*argv[2:]]),cwd=source,
            env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
    write_json(out/'passed.json',dict(passed=True,source=str(source),tp=2,dp=4,microbatch=1,diagnostic=diagnostic))


def validate_plan(p):
    require_ready()
    if p.get('resume_from'):
        from resume_arm_failure_variants import validate_plan as validate_resume
        configure();validate_resume(p)
    else:
        from prepare_arm_failure_ablations import validate_initialization
        validate_initialization(p)
    expected=plan(p['failure_ablation_training'],p['job_id'],p['requested_resources']['hours'],p.get('resume_from'))
    if p!=expected:raise ValueError('Prepared DP plan changed')
    return dict(passed=True,science_unchanged=True,topology='TP2/DP4/microbatch1',
        fresh_initialization=not bool(p.get('resume_from')),gpu_validation_required=True)


def prepare_validation():
    from validate_arm_gate_dp import prepare_source as prepare_validation_source
    prepare_source();prepare_validation_source(SOURCE,VALIDATION_SOURCE)
    CONTROL.mkdir(parents=True,exist_ok=True)
    checkpoint=ORIGIN/'runtime/iter_0000000'
    current=ORIGIN/'iterations/0001'
    # Immutable iteration1 checkpoint + complete iteration2 batch (before updates).
    for path in (checkpoint/'.metadata',current/'failure_auxiliary.json',ORIGIN/'runtime/rollout_recovery/1.pt'):
        if not path.exists():raise ValueError('Missing pinned beta validation fixture: '+str(path))
    view=CONTROL/'checkpoint-view';view.mkdir(exist_ok=True)
    for name,target in [('iter_0000000',checkpoint),('rollout',ORIGIN/'runtime/rollout')]:
        p=view/name
        if not p.exists():p.symlink_to(target,target_is_directory=True)
        elif p.resolve()!=target.resolve():raise ValueError('Pinned validation target changed')
    (view/'latest_checkpointed_iteration.txt').write_text('0\n')
    prior=json.loads((ORIGIN/'launch_manifest.json').read_text())
    write_json(CONTROL/'browser.json',prior['browser_config'])
    write_json(CONTROL/'pinned-dependencies.json',dict(checkpoint=str(checkpoint),
        recovery=str(ORIGIN/'runtime/rollout_recovery/1.pt'),auxiliary=str(current),
        reason='Full saved beta batch for both topology startup checks; diagnostic weights discarded'))


def validation_plan(job,out):
    prior=json.loads((ORIGIN/'launch_manifest.json').read_text());out=Path(out)
    current=ORIGIN/'iterations/0001'
    manifest=json.loads((current/'failure_auxiliary.json').read_text())
    env={k:v for k,v in prior['environment'].items() if not k.startswith(('OPENWEBRL_ARM_','OPENWEBRL_REPLAY_'))}
    env={k:v.replace(prior['source'],str(VALIDATION_SOURCE)) for k,v in env.items()}
    env.update(NUM_GPUS='8',TP_SIZE='2',NUM_ROLLOUT='2',SAVE_DIR=str(out/'runtime'),
        SLIME_LOAD_CHECKPOINT=str(CONTROL/'checkpoint-view'),WANDB_PROJECT='openwebrl-evals',
        WANDB_RUN_ID=f'arm-ablation-tp2-full-{job}',WANDB_RESUME='never',WANDB_MODE='online',
        WANDB_CACHE_DIR=str(out/'wandb-cache'),BROWSER_TRAIN_CONFIG=str(CONTROL/'browser.json'),
        OPENWEBRL_ARM_TURN_BONUS_CONFIG=str(out/'arm-config.json'),
        OPENWEBRL_ARM_FAILURE_AUX_MANIFEST=str(out/'fixture/failure_auxiliary.json'),
        OPENWEBRL_ARM_TPDP_DIAGNOSTIC='1',OPENWEBRL_TPDP_OUTPUT=str(out),
        OPENWEBRL_MULTIMODAL_STORAGE_DIR=str(out/'multimodal'),
        PYTHONPATH=str(REPO/'scripts'),OMP_NUM_THREADS='1',RAY_ADDRESS='local',OPENWEBRL_STREAMING_CHECKPOINT='1')
    command=[v.replace(prior['source'],str(VALIDATION_SOURCE)).replace(str(ORIGIN),str(out)) for v in prior['command']]
    command[command.index('--wandb-project')+1]='openwebrl-evals'
    command[command.index('--wandb-group')+1]='ARM beta/q40 full-batch TP2 validation with zero-loss32K probe'
    i=command.index('--save-debug-rollout-data');del command[i:i+2]
    if '--use-checkpoint-opt-param-scheduler' not in command:command.append('--use-checkpoint-opt-param-scheduler')
    argv=shlex.split(subprocess.check_output(command,env=dict(clean_environment(),**env,DRY_RUN='1'),text=True))
    argv.remove('--colocate')
    argv=[str(PYTHON),str(VALIDATION_SOURCE/'train.py'),*argv[2:],
        '--debug-train-only','--load-debug-rollout-data',str(ORIGIN/'runtime/rollout_recovery/1.pt')]
    updates=json.loads((current/'training_gate.json').read_text())['expected_optimizer_updates']
    return dict(output=str(out),source=str(VALIDATION_SOURCE),command=argv,environment=env,
        production=False,tp=2,dp=4,rollout_id=1,optimizer_updates=updates,auxiliary_rows=len(manifest['records']),
        expected_final_adam_updates=16+updates,validation_stress_context=32768,global_batch=256,ppo_epochs=2)


def validate_gpu(job):
    from inspect_training_checkpoint import inspect_checkpoint
    from prepare_arm_tpdp_test import restore_command
    require_ready();prepare_validation()
    out=RUNTIME/f'benchmarks/arm-ablation-tp2-full-{job}';out.mkdir(parents=True,exist_ok=False)
    p=validation_plan(job,out);write_json(out/'plan.json',p)
    current=ORIGIN/'iterations/0001';fixture=out/'fixture';fixture.mkdir()
    manifest=json.loads((current/'failure_auxiliary.json').read_text())
    (fixture/manifest['tensor_file']).symlink_to(current/manifest['tensor_file'])
    write_json(fixture/'failure_auxiliary.json',manifest)
    config=json.loads((current/'arm-config.json').read_text())
    config.update(output=str(out),run_output=str(out),diagnostic_only=True,
        diagnostic_saved_applied_beta=json.loads((current/'calibration.json').read_text())['applied_beta'])
    write_json(out/'arm-config.json',config)
    env=dict(clean_environment(),**p['environment'])
    from dotenv import dotenv_values
    for k,v in dotenv_values(REPO/'.env').items():
        if v and k.startswith('WANDB_'):env.setdefault(k,v)
    start=time.monotonic()
    with (out/'train.log').open('w') as log:
        subprocess.run(source_command(VALIDATION_SOURCE,['timeout','--signal=TERM','--kill-after=30','1800',*p['command']]),
            cwd=VALIDATION_SOURCE,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    report=inspect_checkpoint(out/'runtime',expected_updates=p['expected_final_adam_updates'])
    write_json(out/'checkpoint-validation.json',report)
    restore=out/'restore';restore.mkdir()
    argv,restore_env=restore_command(p,env,restore)
    argv[argv.index('--num-rollout')+1]='3'
    with (out/'restore.log').open('w') as log:
        subprocess.run(source_command(VALIDATION_SOURCE,['timeout','--signal=TERM','--kill-after=20','300',*argv]),
            cwd=VALIDATION_SOURCE,env=restore_env,stdout=log,stderr=subprocess.STDOUT,check=True)
    receipt=json.loads((restore/'resume_verification.json').read_text())
    if receipt['full_model_and_optimizer_load']!='passed' or receipt['loaded_iteration']!=1 or receipt['optimizer_updates_executed']:
        raise ValueError('Full batch diagnostic reload failed')
    ranks=[]
    for rank in range(8):
        rows=[json.loads(line) for line in (out/f'updates-rank{rank}.jsonl').read_text().splitlines()]
        if len(rows)!=p['optimizer_updates']:raise ValueError('Unexpected validation update count')
        ranks.extend(rows)
    write_json(out/'result.json',dict(passed=True,production_source=str(SOURCE),optimizer_updates=p['optimizer_updates'],
        batch=json.loads((out/'batch.json').read_text()),restore=receipt,wall_seconds=time.monotonic()-start,
        peak_allocated_gib=max(v['peak_allocated_bytes'] for v in ranks)/1024**3,
        seconds_per_update_rank0=[json.loads(line)['seconds'] for line in (out/'updates-rank0.jsonl').read_text().splitlines()],
        training_lineage_modified=False))


def execute(variant,job):
    req=request(job);require_ready()
    if (not req or req['variant']!=variant or os.getenv('SLURM_JOB_ID')!=job
            or req['stage'] not in ('released','armed-fresh')):raise ValueError('Requires approved migration allocation')
    if req['stage']=='released' and int(os.getenv('SLURM_RESTART_COUNT','0'))<1:
        raise ValueError('Beta must be explicitly requeued')
    hours=req['remaining_seconds']/3600
    allocation(subprocess.check_output(['scontrol','show','job',job,'-o'],text=True),job,requested_gpus=8,maximum_hours=hours)
    root=RUNTIME/f'evaluations/arm-failure-ablation-{variant}-{job}-tp2';root.mkdir(exist_ok=False)
    p=plan(variant,job,hours,req.get('resume_from'));validate_plan(p)
    write_json(root/'controller-plan.json',dict(variant=variant,job_id=job,target=20,
        training_root=p['output'],topology='TP2/DP4/microbatch1',evaluation_in_allocation=True))
    step=['srun',f'--jobid={job}','--nodes=1','--ntasks=1','--cpus-per-task=64',
        '--gres=gpu:h200:8','--exact','--cpu-bind=none',sys.executable,str(Path(__file__).resolve()),'--job-id',job]
    try:
        write_json(root/'status.json',dict(stage='full-batch-validation',complete=False))
        subprocess.run([*step,'--worker','validate'],check=True)
        write_json(root/'status.json',dict(stage='training',complete=False))
        subprocess.run([*step,'--worker','train'],check=True)
        from run_arm_iteration80_eval import checkpoint_ready
        if checkpoint_ready('additive',20,Path(p['output'])) is None:
            write_json(root/'status.json',dict(stage='budget-stop',complete=False,reason='Saved progress; no extension'))
            return
        write_json(root/'status.json',dict(stage='evaluation',complete=False))
        subprocess.run([*step,'--worker','eval'],check=True)
        from prepare_arm_failure_ablations import evaluation_plan
        from run_stage1_to100 import audit_rollouts
        e=evaluation_plan(variant,Path(p['output']),job)
        write_json(root/'evaluation-audit.json',audit_rollouts(Path(e['output'])/'rollouts',Path(e['source'])))
        write_json(root/'status.json',dict(stage='complete',complete=True,target=20))
    except BaseException as exc:
        write_json(root/'status.json',dict(stage='failed',complete=False,error_type=type(exc).__name__,error=str(exc)[:300]))
        raise


def worker(job,kind):
    req=request(job)
    if not req or os.getenv('SLURM_JOB_ID')!=job:raise ValueError('Wrong worker allocation')
    if kind=='validate':return validate_gpu(job)
    p=plan(req['variant'],job,req['remaining_seconds']/3600,req.get('resume_from'));validate_plan(p)
    validation=RUNTIME/f'benchmarks/arm-ablation-tp2-full-{job}/result.json'
    if not json.loads(validation.read_text()).get('passed'):raise ValueError('GPU validation has not passed')
    if kind=='train':
        if p.get('resume_from'):
            from resume_arm_failure_variants import execute as resume_execute
            return resume_execute(p)
        from resume_arm_failure_variants import check_multimodal_storage
        from run_arm_turn_bonus_cycles import execute as fresh_execute
        check_multimodal_storage(p)
        with (RUNTIME/f'logs/arm-variant-monitor-{job}.log').open('w') as log:
            child=subprocess.Popen([str(PYTHON),str(REPO/'scripts/monitor_arm_turn_bonus.py'),
                '--job-id',job,'--watch','--interval','900','--hours',str(p['requested_resources']['hours'])],stdout=log,stderr=subprocess.STDOUT)
            try:return fresh_execute(p)
            finally:
                child.terminate()
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:child.kill();child.wait()
    from prepare_arm_failure_ablations import evaluation_plan
    from run_arm_milestone_queue import run_due_evaluations
    import evaluate_baseline_checkpoint as evaluator
    os.environ.update(OPENWEBRL_EXPECTED_SCHEDULER_OFFSET='0',OPENWEBRL_RECORD_EVAL_EXPECTED_TASK_COUNT='300')
    run_due_evaluations(req['variant'],Path(p['output']),20,job)
    evaluator.run(evaluation_plan(req['variant'],Path(p['output']),job),REPO/'.env')


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--job-id',required=True,choices=JOBS);a.add_argument('--worker',required=True,choices=('validate','train','eval'))
    args=a.parse_args();worker(args.job_id,args.worker)
