#!/usr/bin/env python3
"""Restore a completed ARM checkpoint on allocated GPUs without training."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import shlex

from evaluate_baseline_checkpoint import REPO,RUNTIME
from resume_baseline import allocation,clean_environment,source_command,validate_source,write_json
from inspect_training_checkpoint import inspect_checkpoint


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-root',type=Path,required=True)
    parser.add_argument('--job-id',required=True); parser.add_argument('--execute',action='store_true')
    parser.add_argument('--gpus',type=int,choices=(2,4,8),default=2)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--continuation-plan',type=Path)
    a=parser.parse_args(); root=a.training_root.resolve()
    if not root.is_relative_to(RUNTIME/'evaluations'):
        raise ValueError('Requires an isolated ARM run under runtime/evaluations')
    manifest=json.loads((root/'launch_manifest.json').read_text())
    state=json.loads((root/'status.json').read_text())
    diagnostic=manifest.get('diagnostic_only') and 'gpu-diagnostic' in root.name
    experiment=root.name.startswith(('arm-turn-bonus-fresh-','sol-turn-bonus-fresh-','arm-failure-additive-')) and not manifest.get('diagnostic_only')
    durable_checkpoint = state.get('last_valid_checkpoint')
    has_durable_checkpoint = bool(state.get('has_saved_checkpoints') or
                                  (durable_checkpoint and Path(durable_checkpoint).exists()))
    controller_failed_with_checkpoint = bool(state.get('failed') and has_durable_checkpoint)
    if (not (diagnostic or experiment) or
            (state.get('stage') != 'complete' and not controller_failed_with_checkpoint) or
            (state.get('failed') and not controller_failed_with_checkpoint)):
        raise ValueError('Requires a completed inactive ARM run or a failed run with a durable checkpoint')
    if not state.get('completed_optimizer_updates',0) or not has_durable_checkpoint:
        raise ValueError('Requires a durable trained checkpoint')
    source=Path(manifest['source']); validate_source(source)
    continuation_hash=None
    if a.continuation_plan:
        from resume_arm_turn_bonus import validate_resume_plan,continuation_identity
        continuation=json.loads(a.continuation_plan.read_text())
        validate_resume_plan(continuation)
        if (Path(continuation['resume_from']).resolve()!=root or continuation['job_id']!=a.job_id
                or continuation['source']!=str(source) or continuation['requested_resources']['gpus']!=a.gpus):
            raise ValueError('Continuation plan does not match the requested restore')
        # Exercise the actual continuation scheduler, run length and topology.
        # Using the origin launch args masked the WD-horizon mismatch in 294604.
        manifest=continuation
        continuation_hash=continuation_identity(continuation)
    os.environ.setdefault('FLASHINFER_WORKSPACE_BASE',str(RUNTIME))
    before=inspect_checkpoint(root/'runtime',expected_updates=state['completed_optimizer_updates'])
    output=(a.output or root/'gpu-restore-check').resolve()
    if not output.is_relative_to(root) or output.is_relative_to(root/'runtime'):
        raise ValueError('Restore output must be a new run artifact outside checkpoint storage')
    command=list(manifest['command'])
    env=dict(clean_environment(),**manifest.get('environment',{}))
    if command[0]=='bash':
        dry_env=dict(env,DRY_RUN='1',JUDGE_API_MODE='served',JUDGE_API_BASE='https://api.openai.com/v1')
        command=shlex.split(subprocess.check_output(command,env=dry_env,text=True))
    command.remove('--colocate'); command.remove('--use-wandb')
    for key,value in [('--load',str(root/'runtime')),('--save',str(output)),('--wandb-mode','disabled'),
            ('--num-gpus-per-node',str(a.gpus)),('--actor-num-gpus-per-node',str(a.gpus)),
            ('--tensor-model-parallel-size',str(a.gpus))]:
        command[command.index(key)+1]=value
    command+=['--debug-train-only']
    if a.continuation_plan:
        # The continuation directory is created only after restore succeeds.
        # Stage the identical browser settings inside this verification output.
        command[command.index('--custom-config-path')+1]=str(output/'browser-training-config.json')
    plan=dict(job_id=a.job_id,source=str(source),command=command,checkpoint=before,
        output=str(output),gpus=a.gpus,optimizer_updates_requested=0,browser_collections_requested=0)
    if not a.execute:
        print(json.dumps(plan,indent=2)); return
    if os.environ.get('SLURM_JOB_ID')!=a.job_id or f'/job_{a.job_id}/' not in Path('/proc/self/cgroup').read_text():
        raise ValueError('Execute inside the authorized interactive allocation')
    capacity=allocation(subprocess.check_output(['scontrol','show','job',a.job_id,'-o'],text=True),a.job_id,requested_gpus=a.gpus,
        maximum_hours=manifest['requested_resources']['hours'] if a.continuation_plan else 8)
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'plan.json',plan)
    if a.continuation_plan:
        write_json(output/'browser-training-config.json',manifest['browser_config'])
    for key in list(env):
        if key.startswith(('OPENWEBRL_ARM_','ARM_GPU_')): env.pop(key)
    env.update(PYTHONPATH=str(REPO/'scripts'),OPENWEBRL_VERIFY_RESUME_ONLY='1',OMP_NUM_THREADS='1',
        RAY_ADDRESS='local',RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES=str(8*1024**3),WANDB_MODE='disabled')
    with (output/'restore.log').open('w') as log:
        subprocess.run(source_command(source,['timeout','--signal=INT','--kill-after=45',
            str(min(capacity['maximum_seconds'],900)),*command]),cwd=source,env=env,
            stdout=log,stderr=subprocess.STDOUT,check=True)
    result=json.loads((output/'resume_verification.json').read_text())
    expected=int((root/'runtime/latest_checkpointed_iteration.txt').read_text())
    if (result['full_model_and_optimizer_load']!='passed' or result['loaded_iteration']!=expected
            or result['gpus']!=a.gpus or result['next_rollout_id']!=expected+1
            or result['optimizer_updates_executed'] or result['browser_collections_executed']):
        raise ValueError('Unexpected native GPU restore receipt')
    result.update(passed=True,job_id=a.job_id,source=str(source),checkpoint_completed_optimizer_updates=state['completed_optimizer_updates'],
        limitation='GPU model and optimizer restoration only; no additional training or browser collection')
    if continuation_hash: result['continuation_identity_sha256']=continuation_hash
    write_json(output/'result.json',result)
    print(json.dumps(result))


if __name__=='__main__': main()
