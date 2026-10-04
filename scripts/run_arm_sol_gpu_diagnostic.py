#!/usr/bin/env python3
"""One approved two-H200 hour: live Sol pilot, then two synthetic native GPU cycles."""
import argparse
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time

from arm_launch_preflight import require_receipt
from evaluate_baseline_checkpoint import REPO,RUNTIME
from resume_baseline import allocation,clean_environment,source_command,write_json
import run_arm_turn_bonus_cycles as training
import run_sol_inference as inference

PYTHON=RUNTIME/'venv/bin/python'


def training_plan(job,minutes=38,interactive=False,attempt=0):
    if attempt<0: raise ValueError('Attempt must be nonnegative')
    p=training.plan(job,minutes=minutes)
    old=f'arm-turn-bonus-fresh-{job}'; new=f'arm-sol-gpu-diagnostic-{job}-training'
    if attempt: new+=f'-r{attempt}'
    p=json.loads(json.dumps(p).replace(old,new))
    p.update(diagnostic_only=True,requested_iterations=2)
    p['requested_resources'].update(gpus=2,cpus=16,memory_gib=480,gpu_hours=2*minutes/60,browsers=16)
    p['browser_config']['browser_rollout_concurrency']=16
    p['environment'].update(NUM_GPUS='2',TP_SIZE='2',NUM_ROLLOUT='2',BROWSER_CONCURRENCY='16',
        SGLANG_CONCURRENCY='16',ROLLOUT_BATCH_SIZE='24',ARM_GPU_DIAGNOSTIC='1',PYTHONPATH=str(REPO/'scripts'))
    if interactive:
        p['requested_resources'].update(cpus=8,memory_gib=240,browsers=8)
        p['browser_config']['browser_rollout_concurrency']=8
        p['environment'].update(BROWSER_CONCURRENCY='8',SGLANG_CONCURRENCY='8',OMP_NUM_THREADS='1')
    # The diagnostic invokes train.py directly after expanding the shell argv.
    # Preserve the shell launcher's browser exports as well as its arguments.
    p['environment'].update(SLIME_BROWSER_ENV_MODE='local_process',
        SLIME_BROWSER_LOCAL_PROCESS_PYTHON=str(PYTHON),
        SLIME_BROWSER_LOCAL_PROCESS_MAX_PROCESSES=p['environment']['BROWSER_CONCURRENCY'],
        SLIME_BROWSER_ROLLOUT_CONCURRENCY=p['environment']['BROWSER_CONCURRENCY'],
        OPENWEBRL_CUDA_CACHE_LIMIT_GIB='100')
    p.update(interactive_allocation=interactive,attempt=attempt)
    # TP2 / microbatch-one measurements are ~70 seconds per optimizer update.
    # Reserve eight updates plus the native ten-minute save/shutdown allowance.
    p['arm_config'].update(minimum_cycle_seconds=1320,seconds_per_optimizer_update=90,
        diagnostic_only=True,synthetic_rewards_and_labels=True)
    env=dict(clean_environment(),**p['environment'],DRY_RUN='1')
    argv=shlex.split(subprocess.check_output(p['command'],env=env,text=True))
    argv=argv[2:]
    # Multi-epoch native PPO requires log probabilities from the rollout actor.
    # The fixture obtains fresh responses/log probabilities each cycle.
    argv+=['--rollout-function-path','arm_gpu_smoke_fixture.generate_rollout']
    argv[argv.index('--wandb-group')+1]='arm-sol-gpu-diagnostic'
    p['command']=[str(PYTHON),str(training.SOURCE/'train.py'),*argv]
    p['limitation']='Synthetic text rewards and labels; GPU execution test only. TP2, not production TP4/48-browser validation.'
    return p


def inference_plan(job,minutes=18):
    p=inference.plan(job,minutes)
    old=p['output']; new=str(RUNTIME/f'evaluations/arm-sol-gpu-diagnostic-{job}-inference')
    p=json.loads(json.dumps(p).replace(old,new))
    p['segments']=p['segments'][:2]
    p['diagnostic_only']=True
    # Bound this startup pilot; preserve the historical decoding/judge recipe.
    for s in p['segments']:
        s['command'][s['command'].index('--task-timeout')+1]='600'
    return p


def training_result_passed(root):
    """A clean partial-budget stop is not a completed two-cycle diagnostic."""
    path=Path(root)/'status.json'
    if not path.exists(): return False
    state=json.loads(path.read_text())
    return (state.get('training_complete') is True and not state.get('failed')
        and state.get('completed_iterations')==2 and state.get('completed_optimizer_updates')==16)


def native_check(output):
    p=training_plan('GPU_NATIVE_CHECK')
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    write_json(output/'browser.json',p['browser_config'])
    command=list(p['command'])
    command[command.index('--custom-config-path')+1]=str(output/'browser.json')
    probe=output/'parse.py'
    probe.write_text('''import json,sys
from pathlib import Path
from unittest.mock import patch
import slime.utils.arguments as native
with patch('megatron.training.arguments.get_device_arch_version',return_value=9):
    args=native.parse_args()
expected=dict(start_rollout_id=0,eval_interval=None,num_rollout=2,actor_num_gpus_per_node=2,
    tensor_model_parallel_size=2,global_batch_size=256,ppo_epochs=2,use_rollout_logprobs=True,
    rollout_function_path='arm_gpu_smoke_fixture.generate_rollout')
actual={k:getattr(args,k) for k in expected}
assert actual==expected,actual
from check_arm_native_launch import check_native_actor_entry
check_native_actor_entry(args)
Path(__file__).with_suffix('.json').write_text(json.dumps(dict(passed=True,actual=actual,native_module=native.__file__,native_actor_entry_passed=True)))
''')
    env=dict(clean_environment(),**p['environment'],CUDA_VISIBLE_DEVICES='')
    with (output/'parse.log').open('w') as log:
        subprocess.run(source_command(training.SOURCE,[str(PYTHON),str(probe),*command[2:]]),
            cwd=training.SOURCE,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=150)
    write_json(output/'training-plan.json',p)
    print((output/'parse.json').read_text())


def execute(job,training_only=False):
    require_receipt()
    if os.environ.get('SLURM_JOB_ID')!=job or f'/job_{job}/' not in Path('/proc/self/cgroup').read_text():
        raise ValueError('Run inside the dedicated approved allocation')
    resources=allocation(subprocess.check_output(['scontrol','show','job',job,'-o'],text=True),job,requested_gpus=2)
    if resources['cpus']<16 or resources['allocated_memory_gib']<480:
        raise ValueError('Expected 16 CPU/480 GiB')
    root=RUNTIME/f'evaluations/arm-sol-gpu-diagnostic-{job}'
    root.mkdir(parents=True,exist_ok=False)
    deadline=time.time()+min(resources['maximum_seconds'],3540)
    records=[]; stages=('training',) if training_only else ('inference','training')
    write_json(root/'manifest.json',dict(job_id=job,diagnostic_only=True,gpus=2,max_hours=1,
        stages=list(stages),deadline=deadline))
    def interrupted(*_): raise InterruptedError('Diagnostic allocation shutdown')
    previous=[signal.signal(s,interrupted) for s in (signal.SIGTERM,signal.SIGINT)]
    child=None
    try:
        for stage in stages:
            remaining=int((deadline-time.time())/60)
            if remaining<8:
                records.append(dict(stage=stage,skipped=True,reason='Insufficient allocation time'))
                break
            minutes=min(18,remaining) if stage=='inference' else remaining
            write_json(root/'status.json',dict(stage=stage,records=records,minutes_available=remaining))
            with (root/f'{stage}.log').open('w') as log:
                child=subprocess.Popen([str(PYTHON),str(Path(__file__).resolve()),'--job-id',job,
                    '--stage',stage,'--minutes',str(minutes)],cwd=REPO,stdout=log,stderr=subprocess.STDOUT,
                    start_new_session=True)
                try: code=child.wait(timeout=max(1,deadline-time.time()))
                except BaseException:
                    if child.poll() is None: os.killpg(child.pid,signal.SIGTERM)
                    try: child.wait(timeout=45)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid,signal.SIGKILL); child.wait()
                    raise
            passed=code==0
            if stage=='training':
                passed=passed and training_result_passed(training_plan(job,minutes)['output'])
            records.append(dict(stage=stage,exit_code=code,passed=passed,minutes_budget=minutes))
            print(json.dumps(records[-1]),flush=True)
            child=None
        passed=len(records)==len(stages) and all(x.get('passed') for x in records)
        write_json(root/'status.json',dict(stage='complete',passed=passed,records=records,diagnostic_only=True))
        if not passed: raise RuntimeError('One or more diagnostic stages failed; inspect preserved reports')
    finally:
        for s,h in zip((signal.SIGTERM,signal.SIGINT),previous): signal.signal(s,h)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job-id',default='APPROVED_JOB'); p.add_argument('--execute',action='store_true')
    p.add_argument('--training-only',action='store_true')
    p.add_argument('--interactive',action='store_true',help='Use the 8-CPU/240-GiB existing-allocation training profile')
    p.add_argument('--attempt',type=int,default=0)
    p.add_argument('--stage',choices=['inference','training']); p.add_argument('--minutes',type=int,default=38)
    p.add_argument('--native-check',type=Path)
    a=p.parse_args()
    if a.native_check: native_check(a.native_check)
    elif a.stage=='inference': inference.execute(inference_plan(a.job_id,a.minutes))
    elif a.stage=='training': training.execute(training_plan(a.job_id,a.minutes,a.interactive,a.attempt))
    elif a.execute: execute(a.job_id,a.training_only)
    else: print(json.dumps(dict(training=training_plan(a.job_id),inference=inference_plan(a.job_id)),indent=2))


if __name__=='__main__': main()
