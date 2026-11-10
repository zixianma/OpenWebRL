#!/usr/bin/env python3
"""Prepare a fresh frozen-backbone RL ablation; submission requires its own approval."""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import struct
import subprocess

import prepare_arm_mixed_reweight as matched
from resume_baseline import REPO, RUNTIME, allocation, clean_environment, source_command, validate_source, write_json
from openwebrl.vision_freeze_audit import FREEZE_PATTERN

PARENT = RUNTIME/'reference-arm-mixed-reweight-replay-20260928-v1'
SOURCE = RUNTIME/'reference-arm-frozen-vision-20260928-v1'
CONTROL = RUNTIME/'arm-turn-bonus-preparation/frozen-vision-20260928'
PYTHON = matched.PYTHON
RESOURCES = dict(gpus=8, gpu_type='H200', hours=16, gpu_hours=128, cpus=64, memory_gib=960, browsers=64)


def prepare_source():
    from prepare_arm_turn_bonus import copy_plain
    from prepare_arm_gate_dp import replace_once
    validate_source(PARENT)
    provider = 'slime/backends/megatron_utils/model_provider.py'
    text = replace_once((PARENT/provider).read_text(),
        '        freeze_model_params(model, args)\n',
        '        freeze_model_params(model, args)\n'
        '        if os.getenv("OPENWEBRL_VISION_FREEZE_AUDIT_DIR"):\n'
        '            from openwebrl.vision_freeze_audit import audit\n'
        '            audit(model, args)\n')
    changes = {provider: text.encode(), 'openwebrl/vision_freeze_audit.py':
               (REPO/'openwebrl/vision_freeze_audit.py').read_bytes()}
    for name, data in changes.items():
        compile(data, name, 'exec')
    if not SOURCE.exists():
        shutil.copytree(PARENT, SOURCE, symlinks=True, copy_function=copy_plain,
            ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git', '.browser_use_sessions'))
        manifest = matched.read(PARENT/'reference_manifest.json')
        for name, data in changes.items():
            (SOURCE/name).write_bytes(data)
            manifest['recipe_files_sha256'][name] = matched.sha(SOURCE/name)
        manifest['frozen_vision_ablation'] = dict(parent=str(PARENT), changed_files=list(changes),
            freeze_pattern=FREEZE_PATTERN, projection_trainable=True, language_trainable=True,
            initialization='Original SFT iteration0', gpu_validated=False)
        write_json(SOURCE/'reference_manifest.json', manifest)
    validate_source(SOURCE)
    old = matched.read(PARENT/'reference_manifest.json')['recipe_files_sha256']
    new = matched.read(SOURCE/'reference_manifest.json')['recipe_files_sha256']
    if (any(new.get(k) != v for k,v in old.items() if k not in changes)
            or any((SOURCE/k).read_bytes() != v for k,v in changes.items())):
        raise ValueError('Unexpected source change beyond the trainability audit')


def plan(job):
    p = matched.plan(job, 'bonus')
    old_source, old_output, old_run = p['source'], p['output'], p['wandb_run_id']
    output = str(RUNTIME/f'evaluations/arm-frozen-vision-{job}')
    run = f'arm-frozen-vision-{job}'
    def rewrite(value):
        if isinstance(value, str):
            return value.replace(old_source,str(SOURCE)).replace(old_output,output).replace(old_run,run)
        if isinstance(value, list): return [rewrite(v) for v in value]
        if isinstance(value, dict): return {k:rewrite(v) for k,v in value.items()}
        return value
    p = rewrite(p)
    p.update(source=str(SOURCE), output=output, wandb_run_id=run,
             requested_resources=dict(RESOURCES), compute_approved=False, frozen_vision=True)
    p['environment'].update(WANDB_NAME='ARM mixed bonus | relaxed B | frozen backbone | from0',
        OPENWEBRL_VISION_FREEZE_AUDIT_DIR=output+'/parameter-audit',
        RAY_TMPDIR=f'/tmp/fv-{job}')
    p['command'][p['command'].index('--wandb-group')+1] = p['environment']['WANDB_NAME']
    p['command'] += ['--freeze-params-name-list', FREEZE_PATTERN]
    p['arm_config'].update(minimum_cycle_seconds=3000, seconds_per_optimizer_update=120)
    p['comparison'] = dict(control='mixed-only bonus + relaxed B, full actor trainable',
        changed_knob='freeze vision backbone; keep main/DeepStack mergers and language trainable',
        matched_control_wandb='arm-mixed-bonus-334493', historical_B_is_not_matched_control=True,
        first_allocation_endpoint=10, full300_evaluations=[10],
        initialization='OpenWebRL-4B-SFT iteration0; fresh optimizer/scheduler/cursor')
    approval = CONTROL/'resource-approval.json'
    if approval.is_file():
        a = matched.read(approval)
        p['compute_approved'] = a.get('approved') is True and a.get('resources') == RESOURCES
    validate_recipe(p)
    return p


def validate_recipe(p):
    matched.validate_recipe(p)
    if (not p.get('frozen_vision') or p['source'] != str(SOURCE)
            or p['arm_config']['advantage_mode'] != 'original_bonus'
            or p['requested_resources'] != RESOURCES
            or p['command'][-2:] != ['--freeze-params-name-list', FREEZE_PATTERN]
            or p['command'].count('--freeze-params-name-list') != 1):
        raise ValueError('Frozen-backbone ablation changed an unexpected setting')


def fingerprints():
    return {str(p):matched.sha(p) for p in (Path(__file__).resolve(),
        REPO/'openwebrl/vision_freeze_audit.py', REPO/'tests/test_arm_frozen_vision.py',
        REPO/'scripts/run_arm_frozen_vision_8gpu.sbatch',
        Path(matched.__file__).resolve(), SOURCE/'reference_manifest.json', CONTROL/'training-controller.py')}


def validate_plan(p):
    validate_source(SOURCE); validate_recipe(p)
    if p != plan(p['job_id']): raise ValueError('Frozen-vision plan changed')
    ready = matched.read(CONTROL/'readiness.json')
    if not ready['cpu_passed'] or not ready['native_parse_passed'] or ready['fingerprints'] != fingerprints():
        raise ValueError('Frozen-vision readiness missing or stale')
    if not p['compute_approved']: raise ValueError('Separate exact resource approval pending')
    return dict(passed=True, cpu_passed=True, gpu_initialization_pending=True)


def header_audit():
    totals = {k:dict(parameters=0,tensors=0) for k in ('backbone','projection','language')}
    names = {}
    for path in matched.INITIAL.glob('*.safetensors'):
        with path.open('rb') as handle:
            length = struct.unpack('<Q',handle.read(8))[0]
            header = json.loads(handle.read(length))
        for name, value in header.items():
            if name == '__metadata__': continue
            if name in names: raise ValueError('Duplicate checkpoint tensor')
            if name.startswith(('model.visual.merger.', 'model.visual.deepstack_merger_list.')):
                group = 'projection'
            elif name.startswith('model.visual.'): group = 'backbone'
            else: group = 'language'
            names[name] = group
            totals[group]['tensors'] += 1
            totals[group]['parameters'] += math.prod(value['shape'])
            if name.startswith('model.visual.'):
                mcore = name.replace('model.visual.', 'vision_model.', 1)
                if bool(re.search(FREEZE_PATTERN,mcore)) != (group == 'backbone'):
                    raise ValueError('HF-to-Megatron vision freeze mask mismatch')
    if not all(x['parameters'] for x in totals.values()): raise ValueError('Empty model component')
    return dict(counts=totals, source=str(matched.INITIAL),
        audit='Safetensors headers only; unique stored tensors, not TP-replicated counts',
        frozen_parameters=totals['backbone']['parameters'],
        trainable_parameters=totals['projection']['parameters']+totals['language']['parameters'])


def native_check(p):
    from resume_arm_failure_variants import check_multimodal_storage
    write_json(CONTROL/'storage-preflight.json', check_multimodal_storage(p))
    write_json(CONTROL/'browser.json',p['browser_config'])
    env = dict(clean_environment(), **p['environment'])
    env.update(DRY_RUN='1',CUDA_VISIBLE_DEVICES='',BROWSER_TRAIN_CONFIG=str(CONTROL/'browser.json'),
               OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    argv = shlex.split(subprocess.check_output(p['command'],env=env,text=True)); env.pop('DRY_RUN')
    expected = dict(actor_num_gpus_per_node=8,tensor_model_parallel_size=2,global_batch_size=256,
        micro_batch_size=1,ppo_epochs=2,rollout_batch_size=48,browser_rollout_concurrency=64,
        judge_api_model='gpt-4.1',judge_prompt_variant='action_history',dynamic_sampling_filter_path=matched.FILTER,
        use_rollout_logprobs=True,lr=1e-6,num_rollout=10,start_rollout_id=0,
        freeze_params_name_list=[FREEZE_PATTERN],only_train_params_name_list=None,
        use_checkpoint_opt_param_scheduler=False)
    code = ('from unittest.mock import patch\nfrom slime.utils.arguments import parse_args\n'
        "with patch('megatron.training.arguments.get_device_arch_version',return_value=9): a=parse_args()\n"
        f'expected={expected!r}\n'
        'assert all(getattr(a,k)==v for k,v in expected.items()),{k:getattr(a,k) for k in expected}\n'
        "from openwebrl.arm_failure_aux import enabled\nassert not enabled()\nprint('FROZEN_VISION_NATIVE_PARSE_OK')\n")
    with (CONTROL/'native-parse.log').open('w') as log:
        subprocess.run(source_command(SOURCE,[str(PYTHON),'-c',code,*argv[2:]]),cwd=SOURCE,env=env,
            stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)


def prepare():
    from arm_browser_startup_guard import instrument_controller
    prepare_source(); CONTROL.mkdir(parents=True,exist_ok=True)
    text = (REPO/'scripts/run_arm_turn_bonus_cycles.py').read_text()
    before = 'from run_arm_gate_ablation import validate_plan'
    if text.count(before)!=1: raise ValueError('Unexpected controller validation boundary')
    text = instrument_controller(text.replace(before,'from prepare_arm_frozen_vision import validate_plan',1))
    (CONTROL/'training-controller.py').write_text(text)
    p = plan('PREPARE'); write_json(CONTROL/'plan.json',p)
    write_json(CONTROL/'parameter-counts.json',header_audit())
    native_check(p)
    with (CONTROL/'cpu-tests.log').open('w') as log:
        subprocess.run([str(PYTHON),'-m','unittest','discover','-s','tests','-p','test_arm_frozen_vision.py','-v'],
            cwd=REPO,env=dict(clean_environment(),PYTHONPATH=f'{REPO}:{REPO}/scripts',
                OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',WANDB_MODE='disabled'),
            stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
    subprocess.run(['bash','-n',str(REPO/'scripts/run_arm_frozen_vision_8gpu.sbatch')],check=True)
    write_json(CONTROL/'readiness.json',dict(cpu_passed=True,native_parse_passed=True,
        gpu_validated=False,compute_approved=p['compute_approved'],fingerprints=fingerprints()))
    write_json(CONTROL/'proposal.json',dict(resources=RESOURCES,target_iteration=10,full300_evaluations=[10],
        changed_knob=p['comparison']['changed_knob'],budget_approved=p['compute_approved'],submitted=False,
        initialization=p['comparison']['initialization'],no_automatic_extension=True))
    return CONTROL


def evaluation_plan(job):
    import run_arm_iteration80_eval as ev
    p = plan(job); root = Path(p['output'])
    manifest = matched.read(root/'launch_manifest.json')
    if manifest.get('frozen_vision') is not True or manifest['source'] != str(SOURCE):
        raise ValueError('Wrong frozen-vision evaluation lineage')
    ep = ev.plan('original',job,10,root)
    old = ep['output']; output = str(RUNTIME/f'evaluations/arm-frozen-vision-iter10-{job}')
    ep['command'] = [v.replace(old,output) for v in ep['command']]
    ep['environment'] = {k:v.replace(old,output) for k,v in ep['environment'].items()}
    ep.update(output=output,gpus=8,wandb_run_id=f'arm-frozen-vision-iter10-{job}',
        parent_training_run=manifest['wandb_run_id'],arm_variant='mixed-bonus-frozen-vision')
    ep['environment'].update(NUM_GPUS='8',TP_SIZE='2',WANDB_RUN_ID=ep['wandb_run_id'])
    ep['command'][ep['command'].index('--wandb-group')+1]='ARM frozen backbone | iter10 | full300'
    return ev.evaluation.configure_evaluation_tracking(ep,'openwebrl-evals')


def worker(job,stage):
    p = plan(job); validate_plan(p)
    if stage == 'train':
        spec = importlib.util.spec_from_file_location('frozen_vision_controller',CONTROL/'training-controller.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        module.execute(p)
    else:
        import evaluate_baseline_checkpoint as ev
        from resume_arm_mixed_to60 import audit_evaluation
        ep = evaluation_plan(job)
        os.environ.update(OPENWEBRL_EXPECTED_SCHEDULER_OFFSET='0',OPENWEBRL_RECORD_EVAL_EXPECTED_TASK_COUNT='300')
        ev.run(ep,REPO/'.env')
        write_json(CONTROL/f'{job}-eval10-audit.json',audit_evaluation(ep))


def execute(job):
    if os.getenv('SLURM_JOB_ID')!=job: raise ValueError('Execute in the authorized allocation')
    p = plan(job); validate_plan(p)
    allocation(subprocess.check_output(['scontrol','show','job',job,'-o'],text=True),job,
               requested_gpus=8,maximum_hours=16)
    root = RUNTIME/f'evaluations/arm-frozen-vision-controller-{job}'; root.mkdir(exist_ok=False)
    write_json(root/'controller-plan.json',dict(job_id=job,training_root=p['output'],target=10,evaluations=[10]))
    step = ['srun',f'--jobid={job}','--nodes=1','--ntasks=1','--cpus-per-task=64',
        '--gres=gpu:h200:8','--exact','--cpu-bind=none',str(PYTHON),str(Path(__file__).resolve()),
        '--job-id',job,'--worker']
    try:
        write_json(root/'status.json',dict(stage='training',complete=False))
        subprocess.run([*step,'train'],check=True)
        from run_arm_iteration80_eval import checkpoint_ready
        complete = checkpoint_ready('original',10,Path(p['output'])) is not None
        if complete:
            audits = list((Path(p['output'])/'parameter-audit').glob('rank-*-pid-*.json'))
            ranks = {int(f.name.split('-')[1]) for f in audits if matched.read(f).get('passed')}
            if ranks != set(range(8)):
                raise ValueError('Missing live parameter audit on one or more ranks')
            write_json(root/'status.json',dict(stage='evaluation10',complete=False))
            subprocess.run([*step,'eval10'],check=True)
        write_json(root/'status.json',dict(stage='complete' if complete else 'budget-stop',complete=complete,
            verified_complete=complete,evaluations_completed=[10] if complete else []))
    except BaseException as exc:
        write_json(root/'status.json',dict(stage='failed',complete=False,error_type=type(exc).__name__,
            error=str(exc)[:400])); raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--job-id',default='PREPARE')
    parser.add_argument('--execute',action='store_true');parser.add_argument('--worker',choices=['train','eval10'])
    args=parser.parse_args()
    if args.prepare: print(prepare())
    elif args.worker:
        if os.getenv('SLURM_JOB_ID')!=args.job_id: raise ValueError('Wrong worker allocation')
        worker(args.job_id,args.worker)
    elif args.execute: execute(args.job_id)
    else: print(json.dumps(plan(args.job_id),indent=2))
