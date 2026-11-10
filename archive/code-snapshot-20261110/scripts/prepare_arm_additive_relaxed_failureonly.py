#!/usr/bin/env python3
"""Prepare additive relaxed/failure-only ARM from iteration0; never submit jobs."""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from resume_baseline import REPO,RUNTIME,allocation,clean_environment,source_command,validate_source,write_json
from run_arm_turn_bonus_cycles import INITIAL

PARENT=RUNTIME/'reference-arm-gate-b-resume49-20260924-v1'
SOURCE=RUNTIME/'reference-arm-additive-relaxed-failureonly-20260926-v1'
CONTROL=RUNTIME/'arm-turn-bonus-preparation/additive-relaxed-failureonly-20260926'
PYTHON=RUNTIME/'venv/bin/python'
RESOURCES=dict(gpus=8,gpu_type='H200',hours=16,gpu_hours=128,cpus=64,memory_gib=960,browsers=64)
TARGET=20


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())


def prepare_source():
    from prepare_arm_turn_bonus import copy_plain
    validate_source(PARENT)
    name='openwebrl/arm_gate_recovery.py';old=(PARENT/name).read_text()
    before="    if (config['gate_scale_guard'] != 'pretraining_coverage_audit_v1'"
    after=("    mixed_outcome_only = config.get('mixed_outcome_only', False)\n"
           "    expected_beta = 0. if mixed_outcome_only else .5\n"
           "    if mixed_outcome_only and (mixed_outcome_only is not True\n"
           "            or not config.get('additive_failure_groups')\n"
           "            or config.get('failure_group_cap') != 8\n"
           "            or config.get('failure_loss_coefficient') != 1/6\n"
           "            or config.get('credit_assignment') != 'response_index'):\n"
           "        raise ValueError('Unsupported failure-only additive recipe')\n"
           "    if (config['gate_scale_guard'] != 'pretraining_coverage_audit_v1'")
    if old.count(before)!=1 or old.count("config.get('beta') != .5")!=1:
        raise ValueError('Guard patch boundary changed')
    new=old.replace(before,after,1).replace("config.get('beta') != .5","config.get('beta') != expected_beta",1)
    anchor="    result = deepcopy(decision)\n"
    new=new.replace(anchor,anchor+"    result['beta'] = expected_beta\n    result['mixed_outcome_only'] = mixed_outcome_only\n    result['scale_is_counterfactual_mixed_beta05'] = mixed_outcome_only\n",1)
    anchor="    return result\n"
    new=new.replace(anchor,("    if mixed_outcome_only:\n"
        "        result['rule'] = ('Mixed beta=0; failure beta=0.5/q=0.2. Preserve the B labeling guard, '\n"
        "            'using hypothetical mixed beta=0.5 only as a diagnostic; apply no mixed bonus.')\n")+anchor,1)
    compile(new,name,'exec')
    if not SOURCE.exists():
        shutil.copytree(PARENT,SOURCE,symlinks=True,copy_function=copy_plain,
            ignore=shutil.ignore_patterns('__pycache__','*.pyc','.git','.browser_use_sessions'))
        (SOURCE/name).write_text(new)
        m=read(PARENT/'reference_manifest.json')
        m['recipe_files_sha256'][name]=sha(SOURCE/name)
        m['additive_relaxed_failure_only']=dict(parent=str(PARENT),changed_files=[name],
            change='Permit mixed beta0 under explicit flag; unchanged B failure reward/loss and TP2/DP4 transport')
        write_json(SOURCE/'reference_manifest.json',m)
    validate_source(SOURCE)
    original=read(PARENT/'reference_manifest.json')['recipe_files_sha256']
    actual=read(SOURCE/'reference_manifest.json')['recipe_files_sha256']
    if (SOURCE/name).read_text()!=new or any(actual.get(k)!=v for k,v in original.items() if k!=name):
        raise ValueError('Unexpected frozen-source changes')


def plan(job='PREPARE'):
    from run_arm_gate_ablation import plan as base_plan
    p=base_plan(job,'B');previous=p['source'];old=p['output']
    out=RUNTIME/f'evaluations/arm-additive-relaxed-failureonly-{job}'
    run=f'arm-additive-relaxed-failureonly-{job}'
    rewrite=lambda s:s.replace(previous,str(SOURCE)).replace(old,str(out))
    p['command']=[rewrite(s) for s in p['command']]
    for key in ('environment','arm_config'):
        p[key]={k:rewrite(v) if isinstance(v,str) else v for k,v in p[key].items()}
    p.update(source=str(SOURCE),output=str(out),wandb_run_id=run,
        additive_relaxed_failure_only=True,compute_approved=False,
        requested_resources=copy.deepcopy(RESOURCES),requested_iterations=TARGET,
        target_completed_iterations=TARGET,start_rollout_id=0,initial_optimizer_updates=0,
        fresh_optimizer=True,variant_continuation=False,checkpoint=str(INITIAL),
        evaluation_in_allocation=True,evaluation_reserve_seconds=7200)
    scratch=RUNTIME/f'multimodal-scratch/arm-variant-{job}'
    p['environment'].update(NUM_GPUS='8',TP_SIZE='2',NUM_ROLLOUT=str(TARGET),
        WANDB_PROJECT='openwebrl',WANDB_RUN_ID=run,WANDB_RESUME='never',
        WANDB_NAME=f'Additive relaxed | failure-only ARM | q20 | {job}',
        HF_CHECKPOINT=str(INITIAL),SLIME_LOAD_CHECKPOINT=str(INITIAL),
        BROWSER_CONCURRENCY='64',SGLANG_CONCURRENCY='64',
        OPENWEBRL_MULTIMODAL_STORAGE_DIR=str(scratch),OPENWEBRL_CUDA_CACHE_LIMIT_GIB='48')
    p['browser_config']['browser_rollout_concurrency']=64
    p['multimodal_storage']=dict(mode='shared',directory=str(scratch),minimum_free_bytes=2*1024**4,
        retention='preserve; no automatic deletion')
    p['command'][p['command'].index('--wandb-group')+1]='Additive relaxed | failure-only ARM | q20 | from iteration0'
    p['arm_config'].update(run_id=run,output=str(out),run_output=str(out),policy_id=run+':uninitialized',
        beta=0.,mixed_outcome_only=True,scored_fraction=.2,
        checkpoint=str(INITIAL),initial_checkpoint=str(INITIAL),
        gate_scale_guard='pretraining_coverage_audit_v1',
        minimum_cycle_seconds=2700,seconds_per_optimizer_update=90)
    p['comparison']=dict(control='Gate B: same additive min2/index recipe, mixed beta0.5',
        changed_knob='mixed beta0.5->0',mixed_groups=48,failure_group_cap=8,
        failure_beta=.5,failure_q=.2,candidate_gate='min2',credit_assignment='response_index',
        target=TARGET,initialization='Original OpenWebRL-4B-SFT; fresh optimizer/scheduler/cursor',
        mixed_labels='Retained for identical admission/calibration and diagnostics; zero mixed bonus',
        no_automatic_extension=True)
    validate_recipe(p)
    return p


def validate_recipe(p):
    c=p['arm_config'];e=p['environment']
    if (p['checkpoint']!=str(INITIAL) or not p['fresh_optimizer'] or p['start_rollout_id']!=0
        or p['initial_optimizer_updates']!=0 or e['SLIME_LOAD_CHECKPOINT']!=str(INITIAL)
        or e['HF_CHECKPOINT']!=str(INITIAL) or e['WANDB_RESUME']!='never'
        or any(k in p for k in ('resume_from','resume_origin','replay_provenance'))
        or '--use-checkpoint-opt-param-scheduler' in p['command']
        or any(k.startswith('OPENWEBRL_REPLAY') for k in e)):
        raise ValueError('New intervention must begin from iteration0')
    if ((c['beta'],c['scored_fraction'],c['candidate_gate'],c['credit_assignment'],c['failure_group_cap'])
        !=(0.,.2,'min2','response_index',8) or c.get('mixed_outcome_only') is not True
        or not c['additive_failure_groups'] or c['failure_loss_coefficient']!=1/6
        or e['NUM_GPUS']!='8' or e['TP_SIZE']!='2'):
        raise ValueError('Failure-only B objective or topology changed')


def fingerprints():
    return {str(p):sha(p) for p in (Path(__file__).resolve(),SOURCE/'reference_manifest.json',
        CONTROL/'training-controller.py',REPO/'scripts/run_arm_turn_bonus_cycles.py')}


def validate_plan(p):
    validate_source(SOURCE);validate_recipe(p)
    if p!=plan(p['job_id']):raise ValueError('Prepared plan changed')
    r=read(CONTROL/'readiness.json')
    if not r.get('cpu_passed') or r['fingerprints']!=fingerprints():raise ValueError('Readiness missing or stale')
    return dict(passed=True,cpu_passed=True,gpu_initialization_pending=True,fresh_iteration_zero=True)


def native_check(p):
    cfg=CONTROL/'native-browser.json';write_json(cfg,p['browser_config'])
    env=dict(clean_environment(),**p['environment'])
    env.update(DRY_RUN='1',CUDA_VISIBLE_DEVICES='',BROWSER_TRAIN_CONFIG=str(cfg),
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    argv=shlex.split(subprocess.check_output(p['command'],env=env,text=True));env.pop('DRY_RUN')
    code="""from unittest.mock import patch
from slime.utils.arguments import parse_args
with patch('megatron.training.arguments.get_device_arch_version',return_value=9): a=parse_args()
expected={'actor_num_gpus_per_node':8,'tensor_model_parallel_size':2,'global_batch_size':256,'micro_batch_size':1,'ppo_epochs':2,'rollout_batch_size':48,'browser_rollout_concurrency':64,'judge_api_model':'gpt-4.1','judge_prompt_variant':'action_history','dynamic_sampling_filter_path':'openwebrl.arm_failure_additive.filter_groups','use_rollout_logprobs':True,'lr':1e-6,'num_rollout':20,'start_rollout_id':0,'use_checkpoint_opt_param_scheduler':False}
assert all(getattr(a,k)==v for k,v in expected.items()),{k:getattr(a,k) for k in expected}
from openwebrl.arm_failure_aux import check_topology
check_topology(a,dp=4)
print('ADDITIVE_RELAXED_FAILURE_ONLY_NATIVE_PARSE_OK')
"""
    with (CONTROL/'native-parse.log').open('w') as f:
        subprocess.run(source_command(SOURCE,[str(PYTHON),'-c',code,*argv[2:]]),cwd=SOURCE,
            env=env,check=True,stdout=f,stderr=subprocess.STDOUT,timeout=120)


def eval_plan(job,iteration):
    import run_arm_iteration80_eval as ev
    root=Path(plan(job)['output'])
    manifest=read(root/'launch_manifest.json')
    if not manifest.get('additive_relaxed_failure_only') or manifest['wandb_run_id']!=plan(job)['wandb_run_id']:
        raise ValueError('Wrong checkpoint lineage')
    p=ev.plan('additive',job,iteration,root)
    old=p['output'];out=str(RUNTIME/f'evaluations/arm-additive-relaxed-failureonly-iter{iteration}-{job}')
    p['command']=[s.replace(old,out) for s in p['command']]
    p['environment']={k:v.replace(old,out) for k,v in p['environment'].items()}
    run=f'arm-additive-relaxed-failureonly-iter{iteration}-{job}'
    p.update(output=out,gpus=8,wandb_run_id=run,parent_training_run=manifest['wandb_run_id'],
        arm_variant='additive-relaxed-failureonly')
    p['environment'].update(NUM_GPUS='8',TP_SIZE='8',BROWSER_CONCURRENCY='32',SGLANG_CONCURRENCY='32',WANDB_RUN_ID=run)
    p['command'][p['command'].index('--wandb-group')+1]=f'Additive relaxed failure-only | iter{iteration} full300'
    return ev.evaluation.configure_evaluation_tracking(p,'openwebrl-evals')


def prepare():
    prepare_source();CONTROL.mkdir(parents=True,exist_ok=True)
    text=(REPO/'scripts/run_arm_turn_bonus_cycles.py').read_text()
    before='from run_arm_gate_ablation import validate_plan'
    if text.count(before)!=1:raise ValueError('Training controller boundary changed')
    text=text.replace(before,'from prepare_arm_additive_relaxed_failureonly import validate_plan',1)
    text=text.replace('if reserve not in (0,3600):','if reserve not in (0,3600,7200):',1)
    controller=CONTROL/'training-controller.py'
    if controller.exists() and controller.read_text()!=text:raise ValueError('Frozen controller changed')
    controller.write_text(text);compile(text,str(controller),'exec')
    p=plan();write_json(CONTROL/'plan.json',p)
    write_json(CONTROL/'proposal.json',dict(method='Additive relaxed, failure-only ARM',submitted=False,
        method_approved=True,resource_approval_pending=True,resources=RESOURCES,start=0,target=20,
        full300_evaluations=[10,20],expected_hours='About11-14 including evaluation;16h cap',
        no_automatic_extension=True,storage_reserve_bytes=2*1024**4))
    native_check(p)
    # Run semantic tests against the actual frozen production source on CPU.
    code=f"import sys,unittest;sys.path.insert(0,{str(REPO/'scripts')!r});sys.path.insert(0,{str(REPO/'tests')!r});s=unittest.defaultTestLoader.loadTestsFromName('test_arm_additive_relaxed_failureonly');r=unittest.TextTestRunner(verbosity=2).run(s);raise SystemExit(not r.wasSuccessful())"
    with (CONTROL/'semantic-tests.log').open('w') as f:
        subprocess.run(source_command(SOURCE,[str(PYTHON),'-c',code]),cwd=SOURCE,
            env=dict(clean_environment(),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),
            stdout=f,stderr=subprocess.STDOUT,check=True,timeout=120)
    write_json(CONTROL/'readiness.json',dict(cpu_passed=True,native_parse_passed=True,
        gpu_validated=False,resource_approval_pending=True,fingerprints=fingerprints()))
    validate_plan(p)
    subprocess.run(['bash','-n',str(REPO/'scripts/run_arm_additive_relaxed_failureonly_8gpu.sbatch')],check=True)
    return CONTROL


def worker(job,stage):
    if stage=='train':
        from resume_arm_failure_variants import check_multimodal_storage
        p=plan(job);validate_plan(p);check_multimodal_storage(p)
        spec=importlib.util.spec_from_file_location('failure_only_controller',CONTROL/'training-controller.py')
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        with (CONTROL/f'{job}-monitor.log').open('a') as log:
            monitor=subprocess.Popen([str(PYTHON),str(REPO/'scripts/monitor_arm_turn_bonus.py'),
                '--job-id',job,'--watch','--interval','900','--hours','16'],stdout=log,stderr=subprocess.STDOUT)
            try:mod.execute(p)
            finally:
                monitor.terminate()
                try:monitor.wait(timeout=30)
                except subprocess.TimeoutExpired:monitor.kill();monitor.wait()
    else:
        import evaluate_baseline_checkpoint as evaluator
        from run_stage1_to100 import audit_rollouts
        p=eval_plan(job,int(stage))
        os.environ.update(OPENWEBRL_EXPECTED_SCHEDULER_OFFSET='0',OPENWEBRL_RECORD_EVAL_EXPECTED_TASK_COUNT='300')
        evaluator.run(p,REPO/'.env')
        write_json(CONTROL/f'{job}-eval{stage}-audit.json',audit_rollouts(Path(p['output'])/'rollouts',Path(p['source'])))


def execute(job):
    if os.getenv('SLURM_JOB_ID')!=job or os.getenv('ARM_ADDITIVE_FAILUREONLY_APPROVED')!='8gpu_16h':
        raise ValueError('Exact8 H200 x16h resource approval required')
    allocation(subprocess.check_output(['scontrol','show','job',job,'-o'],text=True),job,requested_gpus=8,maximum_hours=16)
    p=plan(job);validate_plan(p)
    root=RUNTIME/f'evaluations/arm-additive-relaxed-failureonly-controller-{job}';root.mkdir(exist_ok=False)
    write_json(root/'controller-plan.json',dict(job_id=job,training_root=p['output'],target=20,evaluations=[10,20]))
    step=['srun',f'--jobid={job}','--nodes=1','--ntasks=1','--cpus-per-task=64','--gres=gpu:h200:8',
        '--exact','--cpu-bind=none',str(PYTHON),str(Path(__file__).resolve()),'--job-id',job,'--worker']
    try:
        write_json(root/'status.json',dict(stage='training',complete=False))
        subprocess.run([*step,'train'],check=True)
        from run_arm_iteration80_eval import checkpoint_ready
        completed=[]
        for target in (10,20):
            if checkpoint_ready('additive',target,Path(p['output'])) is None:continue
            write_json(root/'status.json',dict(stage=f'evaluation{target}',complete=False))
            subprocess.run([*step,str(target)],check=True);completed.append(target)
        write_json(root/'status.json',dict(stage='complete' if completed==[10,20] else 'budget-stop',
            complete=completed==[10,20],evaluations_completed=completed,target=20))
    except BaseException as exc:
        write_json(root/'status.json',dict(stage='failed',complete=False,error_type=type(exc).__name__,error=str(exc)[:400]));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--job-id',default='PREPARE')
    parser.add_argument('--worker',choices=['train','10','20']);parser.add_argument('--execute',action='store_true')
    a=parser.parse_args()
    if a.prepare:print(prepare())
    elif a.worker:
        if os.getenv('SLURM_JOB_ID')!=a.job_id:raise ValueError('Wrong worker allocation')
        worker(a.job_id,a.worker)
    elif a.execute:execute(a.job_id)
    else:print(json.dumps(plan(a.job_id),indent=2))
