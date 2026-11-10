#!/usr/bin/env python3
"""CPU-only native CLI check for all three audit actors; no model/API calls."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


def child():
    source=Path(os.environ['ARM_QUALITY_PARSE_SOURCE']);sys.path.insert(0,str(source))
    from unittest.mock import patch
    import slime.utils.arguments as arguments
    if not Path(arguments.__file__).is_relative_to(source):raise ValueError('Wrong frozen source imported')
    with patch('megatron.training.arguments.get_device_arch_version',return_value=9):
        args=arguments.parse_args()
    actual={k:getattr(args,k) for k in ['num_rollout','actor_num_gpus_per_node','tensor_model_parallel_size',
        'eval_function_path','rollout_max_response_len','rollout_max_context_len','wandb_project',
        'use_checkpoint_opt_param_scheduler','hf_checkpoint','load']}
    hook='arm_task_success' if os.getenv('OPENWEBRL_CPU_AUDIT_KIND')=='task_success' else 'arm_quality_audit'
    expected=dict(num_rollout=0,actor_num_gpus_per_node=2,tensor_model_parallel_size=2,
        eval_function_path=f'openwebrl.{hook}.generate_rollout',rollout_max_response_len=1024,
        rollout_max_context_len=32768,wandb_project='openwebrl-evals',
        use_checkpoint_opt_param_scheduler=os.environ['ARM_QUALITY_PARSE_ACTOR']!='sft')
    if any(actual[k]!=v for k,v in expected.items()):raise ValueError(f'Unexpected native args: {actual}')
    if hook=='arm_task_success':
        dataset=args.eval_datasets[0]
        if dataset.max_response_len!=4096 or dataset.temperature!=.8 or dataset.top_k!=-1:
            raise ValueError('Wrong task-success decoding protocol')
    Path(os.environ['ARM_QUALITY_PARSE_REPORT']).write_text(json.dumps(dict(passed=True,actual=actual,
        gpu_restoration_tested=False,hardware_probe_mock='H200 major 9; no CUDA initialization'),indent=2)+'\n')


def main():
    if os.getenv('OPENWEBRL_CPU_AUDIT_KIND')=='task_success':
        from run_arm_task_success_audit import plan,CONTROL,SOURCE
    else:
        from run_arm_quality_audit import plan,CONTROL,SOURCE
    from resume_baseline import clean_environment,source_command,write_json
    p=plan('PREPARE');reports=[]
    for stage in p['stages']:
        dest=CONTROL/f'cpu-native-{stage["actor"]}.json'
        env=clean_environment();env.update(stage['environment'])
        env.update(DRY_RUN='1',JUDGE_API_MODE='served',
            JUDGE_API_BASE='https://api.openai.com/v1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
            FLASHINFER_WORKSPACE_BASE=str(CONTROL))
        command=shlex.split(subprocess.check_output(stage['command'],env=env,text=True))
        env.update(ARM_QUALITY_PARSE_CHILD='1',ARM_QUALITY_PARSE_SOURCE=str(SOURCE),
            ARM_QUALITY_PARSE_ACTOR=stage['actor'],ARM_QUALITY_PARSE_REPORT=str(dest))
        subprocess.run(source_command(SOURCE,[sys.executable,str(Path(__file__).resolve()),*command[2:]]),
            env=env,cwd=SOURCE,check=True,stdout=subprocess.DEVNULL,timeout=90)
        reports.append(str(dest))
    write_json(CONTROL/'cpu-native-readiness.json',dict(passed=True,reports=reports,
        api_calls_made=0,gpu_restoration_tested=False))
    print(CONTROL/'cpu-native-readiness.json')


if __name__=='__main__':
    if os.getenv('ARM_QUALITY_PARSE_CHILD')=='1':child()
    else:main()
