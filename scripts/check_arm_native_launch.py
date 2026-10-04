#!/usr/bin/env python3
"""Parse/validate actual native training argv on CPU, without Ray, models or API."""
import argparse
import ast
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

from run_arm_turn_bonus_cycles import plan,SOURCE,REPO,RUNTIME
from resume_baseline import source_command,write_json,clean_environment


def check_native_actor_entry(args):
    """Execute real train_actor preconditions, stop before distributed/GPU work."""
    from types import SimpleNamespace
    from unittest.mock import patch
    if args.ppo_epochs<=1: raise ValueError('This readiness check expects native multi-epoch PPO')
    tree=ast.parse((SOURCE/'slime/backends/megatron_utils/actor.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MegatronTrainRayActor')
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='train_actor')
    class ReachedDistributedBoundary(Exception): pass
    def boundary(*a,**kw): raise ReachedDistributedBoundary()
    namespace=dict(mpu=SimpleNamespace(get_data_parallel_rank=boundary))
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),fn],type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),'native_train_actor_cpu_entry','exec'),namespace)
    try:
        # Additive actors enter distributed auxiliary preparation before the
        # first mpu call. Stop at that boundary, never initialize ranks on login.
        with patch('openwebrl.arm_failure_aux.prepare_auxiliary',side_effect=boundary):
            namespace['train_actor'](SimpleNamespace(args=args),0,{})
    except ReachedDistributedBoundary: return True
    raise ValueError('Native train_actor did not reach the expected distributed boundary')


def check_native_scheduler_resume(args):
    """Execute the native scheduler constructor/load using real checkpoint state."""
    from copy import deepcopy
    from types import SimpleNamespace
    import torch
    from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
    root=Path(args.load); rid=int((root/'latest_checkpointed_iteration.txt').read_text())
    common=torch.load(root/f'iter_{rid:07d}'/'common.pt',map_location='cpu',weights_only=False)
    saved=common['opt_param_scheduler']
    tree=ast.parse((SOURCE/'slime/backends/megatron_utils/model.py').read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='get_optimizer_param_scheduler')
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),fn],type_ignores=[])
    ns=dict(OptimizerParamScheduler=OptimizerParamScheduler)
    exec(compile(ast.fix_missing_locations(module),'native_optimizer_scheduler_cpu_entry','exec'),ns)
    build=ns['get_optimizer_param_scheduler']
    old=deepcopy(args);old.use_checkpoint_opt_param_scheduler=False
    previous=build(old,SimpleNamespace(param_groups=[{}]))
    constructor_wd_steps=previous.wd_incr_steps; reproduced=False
    try: previous.load_state_dict(deepcopy(saved))
    except AssertionError as exc:
        if 'weight decay iterations' not in str(exc): raise
        reproduced=True
    if not args.use_checkpoint_opt_param_scheduler or args.override_opt_param_scheduler:
        raise ValueError('Continuation must restore checkpoint scheduler settings without override')
    optimizer=SimpleNamespace(param_groups=[{}])
    scheduler=build(deepcopy(args),optimizer)
    scheduler.load_state_dict(deepcopy(saved))
    if scheduler.state_dict()!=saved:
        raise ValueError('Native scheduler state changed during resume')
    scheduler.step(args.global_batch_size)
    group=optimizer.param_groups[0]
    if (scheduler.num_steps!=saved['num_steps']+args.global_batch_size
            or group['lr']!=args.lr or group['weight_decay']!=args.weight_decay):
        raise ValueError('Resumed scheduler changed LR/WD or reset its sample counter')
    return dict(passed=True,checkpoint_iteration=rid,constructor_wd_steps=constructor_wd_steps,
        checkpoint_wd_steps=saved['wd_incr_steps'],old_failure_reproduced=reproduced,
        restored_sample_counter=saved['num_steps'],next_sample_counter=scheduler.num_steps,
        learning_rate=group['lr'],weight_decay=group['weight_decay'],gpu_or_model_loaded=False)


def main():
    global SOURCE
    if os.environ.get('ARM_NATIVE_PARSE_CHILD')=='1':
        if os.environ.get('ARM_NATIVE_SOURCE'):
            SOURCE = Path(os.environ['ARM_NATIVE_SOURCE'])
        sys.path.insert(0,str(SOURCE))
        import slime.utils.arguments as native_arguments
        if not Path(native_arguments.__file__).is_relative_to(SOURCE):
            raise ValueError('CPU preflight imported a different native source')
        # This is the sole CUDA architecture query in Megatron argument
        # validation. H200 is Hopper (major 9); do not initialize CUDA on login.
        from unittest.mock import patch
        with patch('megatron.training.arguments.get_device_arch_version',return_value=9):
            args=native_arguments.parse_args()
        expected=dict(start_rollout_id=0,eval_interval=None,num_rollout=4,actor_num_gpus_per_node=4,
            tensor_model_parallel_size=4,browser_rollout_concurrency=48,global_batch_size=256,
            ppo_epochs=2,judge_api_model='gpt-4.1',judge_prompt_variant='action_history',lr=1e-6,
            max_steps=15,rollout_temperature=.8,rollout_max_response_len=1024)
        expected.update(rollout_batch_size=48,use_rollout_logprobs=True,
            custom_generate_function_path='openwebrl.arm_turn_bonus.generate')
        if os.environ.get('ARM_NATIVE_INTERACTIVE_2GPU')=='1':
            expected.update(actor_num_gpus_per_node=2,tensor_model_parallel_size=2,
                browser_rollout_concurrency=8)
        if os.environ.get('ARM_NATIVE_RESUME_ROLLOUTS'):
            expected.update(num_rollout=int(os.environ['ARM_NATIVE_RESUME_ROLLOUTS']),
                browser_rollout_concurrency=8*int(os.environ.get('ARM_NATIVE_RESUME_GPUS','4')),start_rollout_id=None,
                actor_num_gpus_per_node=int(os.environ.get('ARM_NATIVE_RESUME_GPUS','4')),
                tensor_model_parallel_size=int(os.environ.get('ARM_NATIVE_RESUME_GPUS','4')),
                use_checkpoint_opt_param_scheduler=True,override_opt_param_scheduler=False)
        if os.environ.get('ARM_NATIVE_FRESH_ABLATION')=='1':
            expected.update(start_rollout_id=0,num_rollout=20,browser_rollout_concurrency=64,
                actor_num_gpus_per_node=8,tensor_model_parallel_size=8,
                use_checkpoint_opt_param_scheduler=False,override_opt_param_scheduler=False)
        if os.environ.get('ARM_NATIVE_ALL_FAILURE') == '1':
            expected.update(browser_rollout_concurrency=32,
                dynamic_sampling_filter_path='openwebrl.arm_failure_bonus.filter_groups')
        if os.environ.get('ARM_NATIVE_ADDITIVE')=='1':
            expected.update(dynamic_sampling_filter_path='openwebrl.arm_failure_additive.filter_groups')
        actual={k:getattr(args,k) for k in expected}
        if actual!=expected: raise ValueError(f'Native resolved configuration differs: {actual}')
        if os.environ.get('ARM_NATIVE_FRESH_ABLATION')=='1':
            from run_arm_turn_bonus_cycles import INITIAL
            if Path(args.load)!=INITIAL or Path(args.hf_checkpoint)!=INITIAL:
                raise ValueError('Fresh ablation resolved to a trained checkpoint')
        check_native_actor_entry(args)
        scheduler=check_native_scheduler_resume(args) if os.environ.get('ARM_NATIVE_RESUME_ROLLOUTS') else None
        write_json(Path(os.environ['ARM_NATIVE_PARSE_REPORT']),dict(passed=True,actual=actual,
            load=args.load,hf_checkpoint=args.hf_checkpoint,source=str(SOURCE),no_gpu_or_browser_started=True,
            native_actor_entry_passed=True,native_scheduler_resume=scheduler,
            hardware_probe_mock='H200 CUDA architecture major 9; GPU startup still requires allocation'))
        return
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--provider',choices=['arm','sol'],default='arm')
    parser.add_argument('--output',type=Path,default=RUNTIME/'arm-turn-bonus-preparation/native-cli-v3')
    parser.add_argument('--interactive-2gpu',action='store_true')
    parser.add_argument('--resume-from',type=Path)
    parser.add_argument('--resume-minutes',type=int,default=240)
    parser.add_argument('--resume-gpus',type=int,choices=(4,8),default=4)
    parser.add_argument('--plan-file',type=Path,help='Prepared all-failure experiment plan; CPU validation only')
    a=parser.parse_args()
    if a.plan_file:
        if a.resume_from or a.interactive_2gpu or a.provider != 'arm':
            raise ValueError('A prepared all-failure plan must be checked independently')
        p=json.loads(a.plan_file.read_text())
        if p.get('experiment') not in ('arm-all-failure-bonus','additive-all-failure'):
            raise ValueError('Unsupported external native plan')
        SOURCE=Path(p['source'])
    elif a.resume_from:
        if a.interactive_2gpu or a.provider!='arm': raise ValueError('Resume preflight uses ARM TP4')
        from resume_arm_turn_bonus import plan as resume_plan
        p=resume_plan('NATIVE_CPU_CHECK',a.resume_from,a.resume_minutes,a.resume_gpus)
    else:
        p=plan('NATIVE_CPU_CHECK',a.provider,interactive_2gpu=a.interactive_2gpu)
    root=a.output; root.mkdir(parents=True,exist_ok=True)
    config=root/f'{a.provider}-browser.json'; write_json(config,p['browser_config'])
    env=dict(clean_environment(),**p['environment']); env.update(DRY_RUN='1',BROWSER_TRAIN_CONFIG=str(config),
        SAVE_DIR=str(root/f'{a.provider}-unused-save'),CUDA_VISIBLE_DEVICES='',
        JUDGE_API_MODE='served',JUDGE_API_BASE='https://api.openai.com/v1')
    argv=shlex.split(subprocess.check_output(p['command'],env=env,text=True))
    env.pop('DRY_RUN'); env.update(ARM_NATIVE_PARSE_CHILD='1',
        ARM_NATIVE_INTERACTIVE_2GPU=str(int(a.interactive_2gpu)),
        ARM_NATIVE_PARSE_REPORT=str(root/f'{a.provider}-report.json'))
    if a.resume_from or p.get('resume_from'):
        env['ARM_NATIVE_RESUME_ROLLOUTS']=p['environment']['NUM_ROLLOUT']
        env['ARM_NATIVE_RESUME_GPUS']=str(p['requested_resources']['gpus'])
    if a.plan_file:
        env.update(ARM_NATIVE_SOURCE=str(SOURCE))
        env['ARM_NATIVE_ADDITIVE' if p['experiment']=='additive-all-failure' else 'ARM_NATIVE_ALL_FAILURE']='1'
        if p.get('failure_ablation_training'):
            from prepare_arm_failure_ablations import validate_initialization
            validate_initialization(p)
            env['ARM_NATIVE_FRESH_ABLATION']='1'
    with (root/f'{a.provider}-parse.log').open('w') as log:
        subprocess.run(source_command(SOURCE,[str(RUNTIME/'venv/bin/python'),str(Path(__file__).resolve()),*argv[2:]]),
            cwd=SOURCE,env=env,check=True,stdout=log,stderr=subprocess.STDOUT,timeout=120)
    print((root/f'{a.provider}-report.json').read_text())


if __name__=='__main__': main()
