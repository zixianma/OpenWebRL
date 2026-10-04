#!/usr/bin/env python3
"""Bounded TP2/DP4, TP4/DP2 and TP8/DP1 replay in C318935; no new allocation."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

from resume_baseline import REPO, RUNTIME, allocation, clean_environment, source_command, validate_source, write_json

JOB='318935'
CONTROL=RUNTIME/'arm-turn-bonus-preparation/tpdp-replay-20260922'
SOURCE=RUNTIME/'reference-arm-tpdp-diagnostic-20260922-v2'
PARENT=RUNTIME/'reference-arm-gate-b-to60-20260921-v1'
ORIGIN=RUNTIME/'evaluations/arm-failure-additive-318934-iter40'
CHECKPOINT=ORIGIN/'runtime/iter_0000022'
PYTHON=RUNTIME/'venv/bin/python'
CASE_SECONDS=1250
CAP_SECONDS=2700


def replace_once(text,old,new):
    if text.count(old)!=1:raise ValueError('Diagnostic source boundary changed: '+old[:80])
    return text.replace(old,new)


def prepare_source():
    from prepare_arm_turn_bonus import copy_plain
    validate_source(PARENT)
    changed={}
    name='openwebrl/arm_failure_aux.py';s=(PARENT/name).read_text()
    s=replace_once(s,"    if (dp, cp, pp) != (1, 1, 1) or args.micro_batch_size != 1:\n        raise ValueError('ARM pilot requires DP1/CP1/PP1 and microbatch 1')",
        "    from openwebrl.arm_tpdp_diagnostic import require_diagnostic\n    require_diagnostic()\n    if dp not in (1,2,4) or (cp,pp)!=(1,1) or args.micro_batch_size!=1:\n        raise ValueError('Diagnostic requires DP1/2/4, CP1/PP1, microbatch1')")
    s=replace_once(s,"    additions=window_schedule(manifest['records'],manifest['total_failure_rows'],counts,gbs,manifest['coefficient'])",
        "    from megatron.core import mpu\n    from openwebrl.arm_tpdp_diagnostic import shard_windows, record_assignment\n    dp=mpu.get_data_parallel_world_size(False); rank=mpu.get_data_parallel_rank(False)\n    global_counts=[c*dp for c in counts]\n    additions=window_schedule(manifest['records'],manifest['total_failure_rows'],global_counts,gbs,manifest['coefficient'])\n    additions=shard_windows(additions,dp,rank)\n    record_assignment(additions,counts,dp,rank,manifest)")
    s=replace_once(s,'        for i, scale in addition:','        for i, scale, padding in addition:')
    s=replace_once(s,"            mixed['arm_source'][-1] = 'arm_failure'","            mixed['arm_source'][-1] = 'arm_padding' if padding else 'arm_failure'")
    s=replace_once(s,"    if source != 'arm_failure':raise ValueError('Unknown additive source')","    if source not in ('arm_failure','arm_padding'):raise ValueError('Unknown additive source')")
    s=replace_once(s,"    return value,dict(arm_mixed_loss=zero,arm_failure_loss=value.detach(),\n", "    if source=='arm_padding':\n        return value,dict(arm_mixed_loss=zero,arm_failure_loss=zero,arm_mixed_count=zero,arm_failure_count=zero,\n            arm_mixed_kl_sum=zero,arm_mixed_clip_sum=zero,arm_failure_kl_sum=zero,arm_failure_clip_sum=zero)\n    return value,dict(arm_mixed_loss=zero,arm_failure_loss=value.detach(),\n")
    changed[name]=s
    name='train.py';s=(PARENT/name).read_text()
    s=replace_once(s,'from openwebrl.arm_turn_bonus_cycles import before_collection, before_training, after_checkpoint',
        'from openwebrl.arm_tpdp_diagnostic import before_collection, before_training, after_checkpoint')
    s=replace_once(s,'from openwebrl.arm_turn_bonus_runtime import before_optimizer',
        'from openwebrl.arm_tpdp_diagnostic import before_optimizer')
    changed[name]=s
    name='slime/ray/rollout.py';s=(PARENT/name).read_text()
    s=replace_once(s,'from openwebrl.arm_turn_bonus_cycles import reset_collection',
        'from openwebrl.arm_tpdp_diagnostic import reset_collection')
    s=replace_once(s,'from openwebrl.arm_turn_bonus import post_process_rewards',
        'from openwebrl.arm_tpdp_diagnostic import post_process_rewards')
    s=replace_once(s,'        return train_data\n','        from openwebrl.arm_tpdp_diagnostic import trim_train_data\n        return trim_train_data(train_data)\n')
    changed[name]=s
    name='slime/backends/megatron_utils/model.py'
    changed[name]=(PARENT/name).read_text()+"\nfrom openwebrl.arm_tpdp_diagnostic import measure_step\ntrain_one_step = measure_step(train_one_step)\n"
    name='openwebrl/arm_tpdp_diagnostic.py';changed[name]=(REPO/name).read_text()
    for name,s in changed.items():compile(s,name,'exec')
    if not SOURCE.exists():
        shutil.copytree(PARENT,SOURCE,symlinks=True,copy_function=copy_plain,
            ignore=shutil.ignore_patterns('__pycache__','*.pyc','.git','.browser_use_sessions'))
        for name,s in changed.items():(SOURCE/name).write_text(s)
        manifest=json.loads((PARENT/'reference_manifest.json').read_text())
        manifest['recipe_files_sha256'].update({k:hashlib.sha256(v.encode()).hexdigest() for k,v in changed.items()})
        manifest['tpdp_diagnostic']=dict(parent=str(PARENT),production=False,changes=list(changed))
        write_json(SOURCE/'reference_manifest.json',manifest)
    validate_source(SOURCE)
    if any((SOURCE/k).read_text()!=v for k,v in changed.items()):
        raise ValueError('Diagnostic source differs from reviewed patches')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint():
    return {str(p):digest(p) for p in (Path(__file__).resolve(),
        REPO/'openwebrl/arm_tpdp_diagnostic.py',REPO/'tests/test_arm_tpdp_diagnostic.py',
        SOURCE/'reference_manifest.json',ORIGIN/'iterations/0023/failure_auxiliary.json',
        ORIGIN/'iterations/0023/arm-config.json',CHECKPOINT/'.metadata')}


def case_plan(tp,out):
    if tp not in (2,4,8):raise ValueError('Require TP2/TP4/TP8')
    out=Path(out)
    prior=json.loads((ORIGIN/'launch_manifest.json').read_text())
    env={k:v for k,v in prior['environment'].items() if not k.startswith(('OPENWEBRL_ARM_','OPENWEBRL_REPLAY_'))}
    env.update(NUM_GPUS='8',TP_SIZE=str(tp),NUM_ROLLOUT='24',SAVE_DIR=str(out/'runtime'),
        SLIME_LOAD_CHECKPOINT=str(CONTROL/'checkpoint-view'),WANDB_PROJECT='openwebrl-evals',
        WANDB_RUN_ID=f'arm-tpdp-{JOB}-tp{tp}',WANDB_RESUME='never',WANDB_MODE='online',
        WANDB_CACHE_DIR=str(out/'wandb-cache'),BROWSER_TRAIN_CONFIG=str(CONTROL/'browser.json'),
        OPENWEBRL_ARM_TURN_BONUS_CONFIG=str(out/'arm-config.json'),
        OPENWEBRL_ARM_FAILURE_AUX_MANIFEST=str(out/'fixture/failure_auxiliary.json'),
        OPENWEBRL_ARM_TPDP_DIAGNOSTIC='1',OPENWEBRL_TPDP_OUTPUT=str(out),
        OPENWEBRL_MULTIMODAL_STORAGE_DIR=str(out/'multimodal'),
        PYTHONPATH=str(REPO/'scripts'),OMP_NUM_THREADS='1',RAY_ADDRESS='local',
        OPENWEBRL_STREAMING_CHECKPOINT='1')
    env={k:v.replace(str(PARENT),str(SOURCE)) for k,v in env.items()}
    command=[x.replace(str(PARENT),str(SOURCE)).replace(str(ORIGIN),str(out)) for x in prior['command']]
    command[command.index('--wandb-project')+1]='openwebrl-evals'
    command[command.index('--wandb-group')+1]='ARM isolated TP/DP saved-batch diagnostic'
    i=command.index('--save-debug-rollout-data');del command[i:i+2]
    argv=shlex.split(subprocess.check_output(command,env=dict(clean_environment(),**env,DRY_RUN='1'),text=True))
    # Debug training uses no browser, SGLang engine, selector, or terminal judge.
    argv.remove('--colocate')
    argv=[str(PYTHON),str(SOURCE/'train.py'),*argv[2:],
        '--debug-train-only','--load-debug-rollout-data',str(ORIGIN/'runtime/rollout_recovery/23.pt')]
    return dict(tp=tp,dp=8//tp,output=str(out),command=argv,environment=env,
        production=False,checkpoint=str(CHECKPOINT),rollout_id=23,global_batch=256,
        ppo_epochs=2,optimizer_updates=2,auxiliary_rows=7,expected_final_adam_updates=324)


def prepare():
    prepare_source();CONTROL.mkdir(parents=True,exist_ok=True)
    view=CONTROL/'checkpoint-view';view.mkdir(exist_ok=True)
    if not (view/CHECKPOINT.name).exists():(view/CHECKPOINT.name).symlink_to(CHECKPOINT,target_is_directory=True)
    if not (view/'rollout').exists():(view/'rollout').symlink_to(ORIGIN/'runtime/rollout',target_is_directory=True)
    (view/'latest_checkpointed_iteration.txt').write_text('22\n')
    prior=json.loads((ORIGIN/'launch_manifest.json').read_text())
    write_json(CONTROL/'browser.json',prior['browser_config'])
    # These dependencies must remain available until the queued replay completes.
    write_json(CONTROL/'pinned-dependencies.json',dict(checkpoint=str(CHECKPOINT),
        recovery=str(ORIGIN/'runtime/rollout_recovery/23.pt'),auxiliary=str(ORIGIN/'iterations/0023'),
        reason='User-authorized isolated systems test before C318935; not a new intervention lineage'))
    for tp in (2,4,8):write_json(CONTROL/f'plan-tp{tp}.json',case_plan(tp,CONTROL/f'preview-tp{tp}'))
    probe=CONTROL/'parse-check.py'
    probe.write_text('''from unittest.mock import patch
from slime.utils.arguments import parse_args
with patch('megatron.training.arguments.get_device_arch_version',return_value=9):args=parse_args()
assert args.actor_num_gpus_per_node==8 and args.tensor_model_parallel_size in (2,4,8)
assert args.global_batch_size==256 and args.micro_batch_size==1 and args.ppo_epochs==2
assert args.debug_train_only and not args.offload_train and not args.offload_rollout
assert args.wandb_project=='openwebrl-evals' and args.num_rollout==24
from openwebrl.arm_failure_aux import check_topology
check_topology(args,dp=8//args.tensor_model_parallel_size)
print('TPDP_NATIVE_PARSE_PASSED')
''')
    for tp in (2,4,8):
        p=case_plan(tp,CONTROL/f'preview-tp{tp}')
        with (CONTROL/f'parse-tp{tp}.log').open('w') as log:
            subprocess.run(source_command(SOURCE,[str(PYTHON),str(probe),*p['command'][2:]]),
                cwd=SOURCE,env=dict(clean_environment(),**p['environment']),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
    subprocess.run([str(PYTHON),'-m','unittest','discover','-s',str(REPO/'tests'),
        '-p','test_arm_tpdp_diagnostic.py','-v'],cwd=REPO,check=True,timeout=60)
    write_json(CONTROL/'readiness.json',dict(cpu_passed=True,native_parse=[2,4,8],fingerprint=fingerprint(),
        gpu_validation='pending',job_id=JOB,hard_cap_seconds=CAP_SECONDS,production_promotion=False))


def prepare_fixture(out):
    import torch
    torch.set_num_threads(1)
    current=ORIGIN/'iterations/0023'
    manifest=json.loads((current/'failure_auxiliary.json').read_text())
    payload=torch.load(current/manifest['tensor_file'],map_location='cpu',weights_only=True,mmap=True)
    out=Path(out);fixture=out/'fixture';fixture.mkdir()
    torch.save(payload[:7],fixture/'failure_auxiliary.pt')
    manifest.update(records=manifest['records'][:7],expected_windows=1,
        tensor_sha256=digest(fixture/'failure_auxiliary.pt'),tensor_file='failure_auxiliary.pt')
    write_json(fixture/'failure_auxiliary.json',manifest)
    config=json.loads((current/'arm-config.json').read_text())
    config.update(output=str(out),run_output=str(out),diagnostic_only=True,
        diagnostic_saved_applied_beta=json.loads((current/'calibration.json').read_text())['applied_beta'])
    write_json(out/'arm-config.json',config)


def restore_command(p,environment,restore):
    """Disable tracking at its actual gate, not just the W&B mode setting.

    This frozen W&B adapter forces shared mode when --use-wandb is present,
    including when --wandb-mode=disabled. Verification must remove that flag.
    """
    out=Path(p['output']);argv=list(p['command'])
    argv.remove('--use-wandb')
    argv[argv.index('--load')+1]=str(out/'runtime')
    argv[argv.index('--save')+1]=str(restore)
    argv[argv.index('--num-rollout')+1]='25'
    argv[argv.index('--wandb-mode')+1]='disabled'
    env=dict(environment,OPENWEBRL_VERIFY_RESUME_ONLY='1',WANDB_MODE='disabled')
    env.pop('WANDB_RUN_ID',None);env.pop('WANDB_RESUME',None)
    return argv,env


def verify_saved_case(tp,out,restore_name='restore'):
    out=Path(out);p=json.loads((out/'plan.json').read_text())
    if p['tp']!=tp:raise ValueError('Saved case topology mismatch')
    from inspect_training_checkpoint import inspect_checkpoint
    report=inspect_checkpoint(out/'runtime',expected_updates=324)
    write_json(out/'checkpoint-validation.json',report)
    restore=out/restore_name;restore.mkdir()
    argv,env=restore_command(p,dict(clean_environment(),**p['environment']),restore)
    with (out/f'{restore_name}.log').open('w') as log:
        subprocess.run(source_command(SOURCE,['timeout','--signal=TERM','--kill-after=20','240',*argv]),
            cwd=SOURCE,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    receipt=json.loads((restore/'resume_verification.json').read_text())
    if receipt['loaded_iteration']!=23 or receipt['optimizer_updates_executed']!=0:
        raise ValueError('Diagnostic checkpoint restore failed')
    updates=[json.loads(line) for line in (out/'updates-rank0.jsonl').read_text().splitlines()]
    if len(updates)!=2:raise ValueError('Expected exactly two diagnostic optimizer updates')
    ranks=[]
    for rank in range(8):
        rows=[json.loads(x) for x in (out/f'updates-rank{rank}.jsonl').read_text().splitlines()]
        if len(rows)!=2:raise ValueError('Missing rank updates')
        ranks.extend(rows)
    result=dict(complete=True,tp=tp,dp=8//tp,updates=updates,restore=receipt,
        batch=json.loads((out/'batch.json').read_text()),peak_memory_bytes=max(x['peak_allocated_bytes'] for x in ranks),
        production_promotion=False)
    write_json(out/'result.json',result)
    return result


def worker(tp,out,case_seconds=CASE_SECONDS):
    if os.getenv('SLURM_JOB_ID')!=JOB:raise ValueError('Existing authorized C allocation required')
    allocation(subprocess.check_output(['scontrol','show','job',JOB,'-o'],text=True),JOB,
        requested_gpus=8,maximum_hours=24)
    if not 600<=case_seconds<=CASE_SECONDS:raise ValueError('Unsupported diagnostic time cap')
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    p=case_plan(tp,out);write_json(out/'plan.json',p)
    prepare_fixture(out)
    env=dict(clean_environment(),**p['environment'])
    from dotenv import dotenv_values
    for k,v in dotenv_values(REPO/'.env').items():
        if v and k.startswith('WANDB_'):env.setdefault(k,v)
    begin=time.monotonic()
    with (out/'train.log').open('w') as log:
        subprocess.run(source_command(SOURCE,['timeout','--signal=TERM','--kill-after=30',str(case_seconds-300),*p['command']]),
            cwd=SOURCE,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    result=verify_saved_case(tp,out)
    result['wall_seconds']=time.monotonic()-begin
    write_json(out/'result.json',result)


def run_if_armed(job,variant,controller_root):
    """Called synchronously by the existing batch controller before any training."""
    if str(job)!=JOB or variant!='C' or not (CONTROL/'armed.json').exists():return
    report=Path(controller_root)/'tpdp-diagnostic.json'
    results=[]
    try:
        receipt=json.loads((CONTROL/'readiness.json').read_text())
        if not receipt.get('cpu_passed') or receipt['fingerprint']!=fingerprint():
            raise ValueError('Diagnostic readiness missing/stale; skip without delaying training')
        if os.getenv('SLURM_JOB_ID')!=JOB:raise ValueError('Requires existing C allocation')
        capacity=allocation(subprocess.check_output(['scontrol','show','job',JOB,'-o'],text=True),JOB,
            requested_gpus=8,maximum_hours=24)
        if capacity['maximum_seconds']<CAP_SECONDS+7200:
            raise ValueError('Insufficient remaining allocation for test plus production')
        started=time.monotonic()
        for tp in (2,8):
            # A distinct bounded Slurm step owns Ray and every GPU worker. On
            # step exit Slurm tears down remaining step processes before return.
            if time.monotonic()-started+CASE_SECONDS>CAP_SECONDS:
                raise TimeoutError('Diagnostic budget exhausted')
            out=RUNTIME/f'benchmarks/arm-tpdp-{JOB}/tp{tp}'
            write_json(report,dict(stage=f'testing-tp{tp}',results=results,budget_seconds=CAP_SECONDS))
            command=['srun',f'--jobid={JOB}','--nodes=1','--ntasks=1','--cpus-per-task=64',
                '--gres=gpu:h200:8','--exact','--cpu-bind=none','--kill-on-bad-exit=1','--time=00:21:00',
                *source_command(SOURCE,['timeout','--signal=TERM','--kill-after=30',str(CASE_SECONDS),
                    str(PYTHON),str(Path(__file__).resolve()),'--worker','--tp',str(tp),'--output',str(out)])]
            code=subprocess.run(command).returncode
            if code or not (out/'result.json').exists():raise RuntimeError(f'TP{tp} replay failed ({code}); inspect {out}')
            results.append(json.loads((out/'result.json').read_text()))
        if results[0]['batch']!=results[1]['batch']:raise ValueError('Replay data mismatch')
        ratios=[a['grad_norm']/b['grad_norm'] for a,b in zip(results[0]['updates'],results[1]['updates'],strict=True)]
        write_json(report,dict(stage='complete',results=results,gradient_norm_ratios_dp4_over_dp1=ratios,
            numerical_agreement_within_5_percent=all(math.isfinite(x) and abs(x-1)<.05 for x in ratios),
            second_update_speedup=results[1]['updates'][1]['seconds']/results[0]['updates'][1]['seconds'],
            production_promotion=False,limitation='One fixed global window, seven auxiliary rows, two epochs; full collection scaling unmeasured'))
    except Exception as exc:
        write_json(report,dict(stage='failed-or-skipped',results=results,error=str(exc)[:700],
            continue_production=True,production_promotion=False))
        print('TP/DP diagnostic skipped/failed; retaining normal C training: '+str(exc),flush=True)


def comparison_result(results):
    """Require a restored, identical-batch control before ranking layouts."""
    by_tp={r['tp']:r for r in results}
    if set(by_tp)!={2,4,8}:raise ValueError('All three matched topology cases required')
    base=by_tp[8]
    for tp,row in by_tp.items():
        if (not row.get('complete') or row['batch']!=base['batch'] or len(row['updates'])!=2
                or row['restore'].get('full_model_and_optimizer_load')!='passed'):
            raise ValueError(f'TP{tp} missing matched data/update/restore evidence')
    if any(not math.isfinite(x['grad_norm']) or x['grad_norm']<=0 for x in base['updates']):
        raise ValueError('Control gradients must be finite and nonzero')
    ratios={tp:[a['grad_norm']/b['grad_norm'] for a,b in zip(row['updates'],base['updates'])]
            for tp,row in by_tp.items()}
    acceptable={tp:all(math.isfinite(x) and abs(x-1)<.05 for x in ratio) for tp,ratio in ratios.items()}
    candidates=[row for tp,row in by_tp.items() if acceptable[tp] and all(
        math.isfinite(x['seconds']) and x['seconds']>0 for x in row['updates'])]
    if not candidates:raise ValueError('No finite comparable case')
    winner=min(candidates,key=lambda row:row['updates'][1]['seconds'])
    return dict(stage='complete',results=results,gradient_norm_ratios_vs_tp8=ratios,
        numerical_agreement_within_5_percent=acceptable,fastest_validated_replay_tp=winner['tp'],
        fastest_validated_replay_dp=8//winner['tp'],second_update_speedup=base['updates'][1]['seconds']/winner['updates'][1]['seconds'],
        user_authorizes_next_launch_switch=True,production_promotion=False,
        next_launch_requirements=['Use the winning validated layout after full-batch/long-context checks',
            'Preserve the current checkpoint, optimizer, scheduler, task cursor, global batch and ARM objective',
            'GPU-restore the actual next checkpoint at its selected TP/DP; do not reset training',
            'No new paid allocation is authorized by the topology preference'])


def followup_steps():
    # Includes a separate cleanup allowance in each Slurm step. With the first
    # attempt charged300s, worst scheduled step caps sum to2640s, under45min.
    return [('reload-tp2',2,330,360,'--verify-saved'),
            ('candidate-tp4',4,850,900,'--worker'),
            ('control-tp8',8,1050,1080,'--worker')]


def finish_before_evaluation(job,variant,target):
    """Finish the saved diagnostic at C40's natural handoff, if time allows.

    The evaluation worker has acquired GPUs but has not launched any GPU process.
    It awaits separate bounded steps so Slurm cleans up each diagnostic's Ray
    workers before the next step or evaluation. No training is interrupted.
    """
    if str(job)!=JOB or variant!='C' or target!=40:return
    marker=CONTROL/'finish-at-c40.json'
    if not marker.exists():return
    report=RUNTIME/f'evaluations/arm-gate-c-to60-{JOB}/tpdp-followup.json'
    if report.exists():return
    try:
        if os.getenv('SLURM_JOB_ID')!=JOB:raise ValueError('Authorized allocation required')
        receipt=json.loads((CONTROL/'readiness.json').read_text())
        if receipt['fingerprint']!=fingerprint():raise ValueError('Followup readiness stale')
        capacity=allocation(subprocess.check_output(['scontrol','show','job',JOB,'-o'],text=True),JOB,
            requested_gpus=8,maximum_hours=24)
        used=json.loads(marker.read_text())['initial_attempt_seconds']
        steps=followup_steps();scheduled_seconds=sum(step[3] for step in steps)
        if used+scheduled_seconds>CAP_SECONDS:raise ValueError('Original45min test cap would be exceeded')
        if capacity['maximum_seconds']<scheduled_seconds+3600:
            raise ValueError('Preserve reserved evaluation time; insufficient room for diagnostic')
        root=RUNTIME/f'benchmarks/arm-tpdp-{JOB}'
        for name,tp,seconds,step_limit,mode in steps:
            write_json(report,dict(stage=name,original_attempt_seconds=used,production_promotion=False))
            step=['srun',f'--jobid={JOB}','--overlap','--immediate=10','--nodes=1','--ntasks=1',
                '--cpus-per-task=64','--gres=gpu:h200:8','--exact','--cpu-bind=none',
                '--kill-on-bad-exit=1',f'--time=00:{step_limit//60:02d}:00',
                *source_command(SOURCE,['timeout','--signal=TERM','--kill-after=20',str(seconds),
                    str(PYTHON),str(Path(__file__).resolve()),mode,'--tp',str(tp),'--output',str(root/f'tp{tp}'),
                    *(['--case-seconds',str(seconds)] if mode=='--worker' else [])])]
            subprocess.run(step,check=True)
        results=[json.loads((root/f'tp{tp}/result.json').read_text()) for tp in (2,4,8)]
        comparison=comparison_result(results)
        write_json(report,comparison);write_json(CONTROL/'next-launch-topology-result.json',comparison)
    except Exception as exc:
        write_json(report,dict(stage='failed-or-skipped',error=str(exc)[:600],continue_evaluation=True))
        print('Topology followup incomplete; continue scheduled C evaluation: '+str(exc),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--arm',action='store_true')
    parser.add_argument('--worker',action='store_true');parser.add_argument('--tp',type=int,choices=(2,4,8))
    parser.add_argument('--case-seconds',type=int,default=CASE_SECONDS)
    parser.add_argument('--verify-saved',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.verify_saved:
        if os.getenv('SLURM_JOB_ID')!=JOB:raise ValueError('Existing authorized C allocation required')
        allocation(subprocess.check_output(['scontrol','show','job',JOB,'-o'],text=True),JOB,
            requested_gpus=8,maximum_hours=24)
        verify_saved_case(args.tp,args.output,restore_name='restore-fixed')
    elif args.worker:worker(args.tp,args.output,args.case_seconds)
    elif args.prepare:prepare()
    elif args.arm:
        receipt=json.loads((CONTROL/'readiness.json').read_text())
        if receipt['fingerprint']!=fingerprint():raise ValueError('Stale diagnostic preparation')
        state=subprocess.check_output(['squeue','--jobs',JOB,'--noheader','--format=%T'],text=True).strip()
        if state!='PENDING':raise ValueError('Attach only before queued C job begins')
        write_json(CONTROL/'armed.json',dict(job_id=JOB,variant='C',within_existing_budget=True,
            max_seconds=CAP_SECONDS,order=[2,8],no_new_submission=True))
    else:parser.error('Choose --prepare, --arm, or --worker')


if __name__=='__main__':main()
