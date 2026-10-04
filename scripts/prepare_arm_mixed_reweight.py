#!/usr/bin/env python3
"""Prepare the matched relaxed-gate, mixed-only pair; never submit allocations."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess

from resume_baseline import REPO, RUNTIME, allocation, clean_environment, source_command, validate_source, write_json

PARENT = RUNTIME/'reference-arm-gate-b-tp2-20260922-v1'
SOURCE = RUNTIME/'reference-arm-mixed-reweight-20260927-v1'
CONTROL = RUNTIME/'arm-turn-bonus-preparation/mixed-reweight-20260927'
PYTHON = RUNTIME/'venv/bin/python'
INITIAL = Path('/gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT')
MODES = {'bonus': 'original_bonus', 'reweight': 'outcome_reweight'}
RESOURCES = dict(gpus=8, gpu_type='H200', hours=8, gpu_hours=64, cpus=64, memory_gib=960, browsers=64)
FILTER = 'slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonempty_nonzero_std'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_source():
    from prepare_arm_turn_bonus import copy_plain
    from prepare_arm_gate_dp import replace_once
    validate_source(PARENT)
    name = 'slime/ray/rollout.py'
    text = replace_once((PARENT/name).read_text(),
        'from openwebrl.arm_turn_bonus import post_process_rewards',
        'from openwebrl.arm_outcome_reweight import post_process_rewards')
    changes = {name: text.encode(), 'openwebrl/arm_outcome_reweight.py': (REPO/'openwebrl/arm_outcome_reweight.py').read_bytes()}
    # Carry the independently tested cancellation-safe browser cleanup into
    # both arms; do not reintroduce the old leaked-slot implementation.
    fixed_browser = RUNTIME/'reference-arm-gate-b-browser-cleanup-20260927-v1'
    validate_source(fixed_browser)
    name = 'openwebrl/env/local_process_env.py'
    changes[name] = (fixed_browser/name).read_bytes()
    for name, data in changes.items():
        compile(data, name, 'exec')
    if not SOURCE.exists():
        shutil.copytree(PARENT, SOURCE, symlinks=True, copy_function=copy_plain,
            ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git', '.browser_use_sessions'))
        manifest = read(PARENT/'reference_manifest.json')
        for name, data in changes.items():
            (SOURCE/name).write_bytes(data)
            manifest['recipe_files_sha256'][name] = hashlib.sha256(data).hexdigest()
        manifest['mixed_only_reweight'] = dict(parent=str(PARENT), changed_files=list(changes),
            modes=MODES, scalar_transform_before_dp=True, auxiliary_disabled_in_both=True,
            native_mixed_filter=FILTER, initialization='Original SFT iteration0', gpu_validated=False)
        write_json(SOURCE/'reference_manifest.json', manifest)
    validate_source(SOURCE)
    old, new = read(PARENT/'reference_manifest.json')['recipe_files_sha256'], read(SOURCE/'reference_manifest.json')['recipe_files_sha256']
    if any(new.get(k) != v for k,v in old.items() if k not in changes) or any((SOURCE/k).read_bytes() != v for k,v in changes.items()):
        raise ValueError('Unexpected frozen source change')


def plan(job, variant):
    if variant not in MODES:
        raise ValueError('Expected bonus or reweight')
    import run_arm_turn_bonus_cycles as training
    p = training.plan(job, 'arm', 480)
    old_source, old_output = p['source'], p['output']
    output = RUNTIME/f'evaluations/arm-mixed-{variant}-{job}'
    run = f'arm-mixed-{variant}-{job}'
    rewrite = lambda v: v.replace(old_source, str(SOURCE)).replace(old_output, str(output))
    p['command'] = [rewrite(v) for v in p['command']]
    for key in ('environment', 'arm_config'):
        p[key] = {k:rewrite(v) if isinstance(v,str) else v for k,v in p[key].items()}
    p.update(source=str(SOURCE), output=str(output), wandb_run_id=run, mixed_reweight_variant=variant,
        gate_ablation=variant, compute_approved=False, requested_resources=dict(RESOURCES),
        requested_iterations=10, target_completed_iterations=10, fresh_optimizer=True,
        initial_optimizer_updates=0, start_rollout_id=0, evaluation_in_allocation=True,
        evaluation_reserve_seconds=3600)
    # The shared-storage validator binds scratch to this canonical job path.
    # Job IDs already distinguish the two independently allocated variants.
    scratch = RUNTIME/f'multimodal-scratch/arm-variant-{job}'
    p['environment'].update(NUM_GPUS='8', TP_SIZE='2', NUM_ROLLOUT='10', WANDB_PROJECT='openwebrl',
        WANDB_RUN_ID=run, WANDB_RESUME='never', WANDB_NAME=f'ARM mixed only | {variant} | relaxed B | {job}',
        BROWSER_CONCURRENCY='64', SGLANG_CONCURRENCY='64', OPENWEBRL_MULTIMODAL_STORAGE_DIR=str(scratch),
        OPENWEBRL_CUDA_CACHE_LIMIT_GIB='48', PYTHONFAULTHANDLER='1', RAY_DEDUP_LOGS='0',
        RAY_TMPDIR=f'/tmp/am-{job}')
    p['multimodal_storage'] = dict(mode='shared', directory=str(scratch), minimum_free_bytes=2*1024**4,
                                  retention='preserve; no automatic deletion')
    p['browser_config']['browser_rollout_concurrency'] = 64
    # This backend derives the displayed name from --wandb-group and ignores
    # WANDB_NAME when it explicitly supplies wandb.init(name=...).
    p['command'][p['command'].index('--wandb-group')+1] = f'ARM mixed only | {variant} | relaxed B | from0'
    p['command'] += ['--dynamic-sampling-filter-path', FILTER]
    p['arm_config'].update(run_id=run, policy_id=run+':uninitialized', run_output=str(output), output=str(output),
        candidate_gate='min2', credit_assignment='response_index', gate_scale_guard='pretraining_coverage_audit_v1',
        advantage_mode=MODES[variant], reweight_lambda=.5, admit_all_failure_groups=False,
        additive_failure_groups=False, failure_group_cap=0, failure_loss_coefficient=0.,
        minimum_cycle_seconds=2700, seconds_per_optimizer_update=90)
    p['comparison'] = dict(control='New mixed-only original bonus with relaxed gate B',
        changed_knob='A+0.5u versus A*mean_one(exp(0.5*sign(A)*u))',
        ordinary_groups=48, failure_auxiliary_groups=0, k=5, q=.2, seed=42,
        initialization='Original SFT iteration0, fresh optimizer/scheduler/cursor',
        historical_B='Additional contextual reference; includes failure auxiliary and is not the matched control',
        first_allocation_endpoint=10, research_endpoint=60, further_allocations_not_approved=True)
    approval = CONTROL/'resource-approval.json'
    if approval.is_file():
        approved = read(approval)
        p['compute_approved'] = (approved.get('approved') is True
            and approved.get('resources_each') == RESOURCES and variant in approved.get('variants',[]))
    validate_recipe(p)
    return p


def validate_recipe(p):
    from openwebrl.arm_outcome_reweight import validate_config
    validate_config(p['arm_config'])
    e = p['environment']
    if (not p['fresh_optimizer'] or p['start_rollout_id'] != 0 or p['initial_optimizer_updates'] != 0
            or p['checkpoint'] != str(INITIAL) or e['SLIME_LOAD_CHECKPOINT'] != str(INITIAL)
            or e['HF_CHECKPOINT'] != str(INITIAL) or e['WANDB_RESUME'] != 'never'
            or p.get('resume_from') or '--use-checkpoint-opt-param-scheduler' in p['command']):
        raise ValueError('A new intervention must start at iteration0')
    if ('OPENWEBRL_ARM_FAILURE_AUX_MANIFEST' in e or any(k.startswith('OPENWEBRL_REPLAY') for k in e)
            or p['command'][p['command'].index('--dynamic-sampling-filter-path')+1] != FILTER
            or (e['NUM_GPUS'],e['TP_SIZE'],e['GLOBAL_BATCH_SIZE']) != ('8','2','256')):
        raise ValueError('Auxiliary signal, replay, filter or topology mismatch')


def fingerprints():
    return {str(p):sha(p) for p in (Path(__file__).resolve(), SOURCE/'reference_manifest.json',
        CONTROL/'training-controller.py', REPO/'scripts/run_arm_mixed_reweight_8gpu.sbatch',
        REPO/'tests/test_arm_outcome_reweight.py',REPO/'tests/test_arm_min2_gate.py',
        REPO/'tests/test_local_browser_cleanup.py')}


def validate_plan(p):
    validate_source(SOURCE); validate_recipe(p)
    if p != plan(p['job_id'],p['mixed_reweight_variant']):
        raise ValueError('Prepared matched-pair plan changed')
    receipt = read(CONTROL/'readiness.json')
    if not receipt['cpu_passed'] or not receipt['native_parse_passed'] or receipt['fingerprints'] != fingerprints():
        raise ValueError('Prepared readiness missing or stale')
    return dict(passed=True, cpu_passed=True, gpu_initialization_pending=True, auxiliary_disabled=True)


def native_check(p, destination):
    from resume_arm_failure_variants import check_multimodal_storage
    write_json(destination/'storage-preflight.json',check_multimodal_storage(p))
    write_json(destination/'browser.json',p['browser_config'])
    env = dict(clean_environment(), **p['environment'])
    env.update(DRY_RUN='1', CUDA_VISIBLE_DEVICES='', BROWSER_TRAIN_CONFIG=str(destination/'browser.json'),
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    argv = shlex.split(subprocess.check_output(p['command'],env=env,text=True)); env.pop('DRY_RUN')
    expected = dict(actor_num_gpus_per_node=8,tensor_model_parallel_size=2,global_batch_size=256,
        micro_batch_size=1,ppo_epochs=2,rollout_batch_size=48,browser_rollout_concurrency=64,
        judge_api_model='gpt-4.1',judge_prompt_variant='action_history',dynamic_sampling_filter_path=FILTER,
        use_rollout_logprobs=True,lr=1e-6,num_rollout=10,start_rollout_id=0,use_checkpoint_opt_param_scheduler=False)
    code = ("from unittest.mock import patch\nfrom slime.utils.arguments import parse_args\n"
        "from ray._private.utils import validate_socket_filepath\n"
        "from pathlib import Path\nimport os\n"
        "socket_path=str(Path(os.environ['RAY_TMPDIR'])/'ray'/'session_2099-12-31_23-59-59_999999_9999999'/'sockets'/'plasma_store')\n"
        "validate_socket_filepath(socket_path)\n"
        "print('MIXED_PAIR_RAY_SOCKET_PREFLIGHT_OK',len(socket_path.encode()))\n"
        "with patch('megatron.training.arguments.get_device_arch_version',return_value=9): a=parse_args()\n"
        f"expected={expected!r}\n"
        "assert all(getattr(a,k)==v for k,v in expected.items()),{k:getattr(a,k) for k in expected}\n"
        "from openwebrl.arm_failure_aux import enabled\nassert not enabled()\nprint('MIXED_PAIR_NATIVE_PARSE_OK')\n")
    with (destination/'native-parse.log').open('w') as stream:
        subprocess.run(source_command(SOURCE,[str(PYTHON),'-c',code,*argv[2:]]),cwd=SOURCE,env=env,
            stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=120)


def prepare():
    prepare_source(); CONTROL.mkdir(parents=True,exist_ok=True)
    from arm_browser_startup_guard import instrument_controller
    text = (REPO/'scripts/run_arm_turn_bonus_cycles.py').read_text()
    before = 'from run_arm_gate_ablation import validate_plan'
    if text.count(before) != 1:
        raise ValueError('Controller validation boundary changed')
    text = instrument_controller(text.replace(before,'from prepare_arm_mixed_reweight import validate_plan',1))
    path = CONTROL/'training-controller.py'
    if path.exists() and path.read_text() != text:
        raise ValueError('Frozen controller changed')
    path.write_text(text); compile(text,str(path),'exec')
    for variant in MODES:
        out = CONTROL/variant; out.mkdir(exist_ok=True)
        p = plan('PREPARE-'+variant,variant)
        write_json(out/'plan.json',p); native_check(p,out)
    if (SOURCE/'openwebrl/env/local_process_env.py').read_bytes() != (REPO/'openwebrl/env/local_process_env.py').read_bytes():
        raise ValueError('Browser cleanup fixture must test identical implementation bytes')
    code = (f"import sys,unittest;sys.path.insert(0,{str(REPO/'tests')!r});"
            "s=unittest.defaultTestLoader.loadTestsFromNames(['test_arm_outcome_reweight','test_arm_min2_gate','test_local_browser_cleanup']);"
            "r=unittest.TextTestRunner(verbosity=2).run(s);raise SystemExit(not r.wasSuccessful())")
    with (CONTROL/'frozen-tests.log').open('w') as stream:
        subprocess.run(source_command(SOURCE,[str(PYTHON),'-c',code]),cwd=SOURCE,
            env=dict(clean_environment(),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',WANDB_MODE='disabled'),
            stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=120)
    subprocess.run(['bash','-n',str(REPO/'scripts/run_arm_mixed_reweight_8gpu.sbatch')],check=True)
    approved=all(plan('PREPARE-'+v,v)['compute_approved'] for v in MODES)
    write_json(CONTROL/'readiness.json',dict(cpu_passed=True,native_parse_passed=True,
        storage_preflight_passed=True,gpu_validated=False,
        exact_resource_approval_pending=not approved,fingerprints=fingerprints()))
    for variant in MODES: validate_plan(plan('PREPARE-'+variant,variant))
    write_json(CONTROL/'proposal.json',dict(variants=list(MODES),per_variant=RESOURCES,total_gpu_hours=128,
        first_allocation_target=10,full300_evaluations=[10],research_endpoint=60,
        iteration10_not_guaranteed_within_budget=True,no_automatic_extension=True,submitted=False))
    return CONTROL


def evaluation_plan(job, variant):
    import run_arm_iteration80_eval as ev
    root = Path(plan(job,variant)['output'])
    manifest = read(root/'launch_manifest.json')
    if manifest.get('mixed_reweight_variant') != variant or manifest['source'] != str(SOURCE):
        raise ValueError('Wrong evaluation checkpoint lineage')
    p = ev.plan('original',job,10,root)
    old = p['output']; output = str(RUNTIME/f'evaluations/arm-mixed-{variant}-iter10-{job}')
    p['command'] = [v.replace(old,output) for v in p['command']]
    p['environment'] = {k:v.replace(old,output) for k,v in p['environment'].items()}
    p.update(output=output,gpus=8,wandb_run_id=f'arm-mixed-{variant}-iter10-{job}',
        parent_training_run=manifest['wandb_run_id'],arm_variant=f'mixed-{variant}')
    p['environment'].update(NUM_GPUS='8',TP_SIZE='2',WANDB_RUN_ID=p['wandb_run_id'])
    p['command'][p['command'].index('--wandb-group')+1] = f'ARM mixed {variant} | iter10 | full300'
    # Embedded evaluation belongs to the training experiment's project.
    return ev.evaluation.configure_evaluation_tracking(p,'openwebrl')


def worker(job, variant, stage):
    p = plan(job,variant); validate_plan(p)
    if stage == 'train':
        from resume_arm_failure_variants import check_multimodal_storage
        check_multimodal_storage(p)
        spec = importlib.util.spec_from_file_location('mixed_reweight_controller',CONTROL/'training-controller.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        module.execute(p)
    else:
        import evaluate_baseline_checkpoint as ev
        from run_stage1_to100 import audit_rollouts
        ep = evaluation_plan(job,variant)
        os.environ.update(OPENWEBRL_EXPECTED_SCHEDULER_OFFSET='0',OPENWEBRL_RECORD_EVAL_EXPECTED_TASK_COUNT='300')
        ev.run(ep,REPO/'.env')
        write_json(CONTROL/f'{job}-{variant}-eval10-audit.json',audit_rollouts(Path(ep['output'])/'rollouts',Path(ep['source'])))


def execute(job, variant):
    if os.getenv('SLURM_JOB_ID') != job or os.getenv('ARM_MIXED_PAIR_APPROVED') != '8gpu_8h_each':
        raise ValueError('Exact8 H200 x8h per variant resource approval required')
    allocation(subprocess.check_output(['scontrol','show','job',job,'-o'],text=True),job,requested_gpus=8,maximum_hours=8)
    p = plan(job,variant); validate_plan(p)
    root = RUNTIME/f'evaluations/arm-mixed-{variant}-controller-{job}'; root.mkdir(exist_ok=False)
    write_json(root/'controller-plan.json',dict(job_id=job,training_root=p['output'],target=10,evaluations=[10]))
    step = ['srun',f'--jobid={job}','--nodes=1','--ntasks=1','--cpus-per-task=64','--gres=gpu:h200:8',
        '--exact','--cpu-bind=none',str(PYTHON),str(Path(__file__).resolve()),'--job-id',job,'--variant',variant,'--worker']
    try:
        write_json(root/'status.json',dict(stage='training',complete=False))
        result = subprocess.run([*step,'train'])
        from run_arm_iteration80_eval import checkpoint_ready
        complete = checkpoint_ready('original',10,Path(p['output'])) is not None
        if complete:
            write_json(root/'status.json',dict(stage='evaluation10',complete=False))
            subprocess.run([*step,'eval10'],check=True)
        write_json(root/'status.json',dict(stage='complete' if complete else 'budget-stop',complete=complete,
            training_exit_code=result.returncode,evaluations_completed=[10] if complete else []))
        if result.returncode:
            raise RuntimeError(f'Training exited{result.returncode}; preserve and inspect artifacts')
    except BaseException as exc:
        write_json(root/'status.json',dict(stage='failed',complete=False,error_type=type(exc).__name__,error=str(exc)[:400])); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true'); parser.add_argument('--job-id',default='PREPARE')
    parser.add_argument('--variant',choices=MODES); parser.add_argument('--execute',action='store_true')
    parser.add_argument('--worker',choices=['train','eval10'])
    args = parser.parse_args()
    if args.prepare: print(prepare())
    else:
        if not args.variant: parser.error('--variant is required')
        if args.worker:
            if os.getenv('SLURM_JOB_ID') != args.job_id: raise ValueError('Wrong worker allocation')
            worker(args.job_id,args.variant,args.worker)
        elif args.execute:
            recovery = CONTROL/'continuations'/f'{args.job_id}-budget.json'
            if recovery.exists():
                from resume_arm_mixed_reweight import execute as resume
                resume(args.job_id)
            else:
                execute(args.job_id,args.variant)
        else: print(json.dumps(plan(args.job_id,args.variant),indent=2))
