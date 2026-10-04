#!/usr/bin/env python3
"""Own the separately approved Gate B continuation through100 and its evals."""
import argparse
import copy
import fcntl
import importlib.util
import os
from pathlib import Path
import shlex
import subprocess
import time

import resume_arm_gate_b90 as previous
from resume_arm_mixed_to60 import atomic, audit_evaluation
from resume_baseline import allocation, clean_environment, source_command, validate_source

REPO, RUNTIME, SOURCE, PYTHON = previous.REPO, previous.RUNTIME, previous.SOURCE, previous.PYTHON
CONTROL = previous.CONTROL/'to100'
SUPERVISOR = RUNTIME/'arm-turn-bonus-preparation/mixed-reweight-20260927/supervisor'
ORIGINAL = RUNTIME/'evaluations/arm-failure-additive-334894-b90-iter90'
PREDECESSOR = RUNTIME/'evaluations/arm-failure-additive-335681-b90-r2-iter90'
RESOURCES = dict(gpus=8, gpu_type='H200', hours=16, gpu_hours=128, cpus=64, memory_gib=960, browsers=64)
read, origin, sha = previous.read, previous.origin, previous.sha
validate_resume_plan, continuation_identity = previous.validate_resume_plan, previous.continuation_identity


def budget(job):
    approved = read(CONTROL/'approval.json')
    if approved.get('approved') is not True or approved.get('resources') != RESOURCES:
        raise ValueError('Exact additional8 H200 x16h approval required')
    b = read(CONTROL/f'{job}-budget.json')
    cap = 57600
    if b['approved_total_seconds'] != cap:
        extra = read(CONTROL/'finish100-approval.json')
        if (extra.get('approved') is not True
                or extra.get('resources') != dict(gpus=8, gpu_type='H200', cpus=64,
                    memory_gib=960, allocation_seconds=10800, gpu_hours=24)
                or extra.get('additional_seconds') != 6495
                or extra.get('consumed_seconds_at_approval') != 53295
                or extra.get('remaining_seconds_at_approval') != 4305
                or b['prior_used_seconds'] < extra['consumed_seconds_at_approval']
                or b['attempt_seconds'] > extra['resources']['allocation_seconds']):
            raise ValueError('Exact completion approval and all prior consumed time required')
        cap += extra['additional_seconds']
    if (b['job_id'] != job or b['approved_total_seconds'] != cap
            or b['prior_used_seconds'] < 0 or not 600 <= b['attempt_seconds'] <= 57600
            or b['prior_used_seconds'] + b['attempt_seconds'] > cap):
        raise ValueError('Attempt exceeds the separate continuation approval')
    return b


def next_stage(completed, evaluated):
    if not 82 <= completed <= 100:
        raise ValueError('Unexpected Gate B checkpoint boundary')
    for milestone in (90, 100):
        if completed >= milestone and milestone not in evaluated:
            return 'eval', milestone
        if completed < milestone:
            return 'train', milestone
    return 'complete', 100


def evaluation_reserve(seconds):
    # A short retry cannot fit the existing 6000-second collection/training
    # admission floor plus an hour withheld for a checkpoint not yet reached.
    # Keep the actual training guards and allocation cap; eval100 remains a
    # required stage and runs only if its checkpoint and time budget permit it.
    return 0 if seconds < 10800 else 3600


def minimum_stage_seconds(stage, seconds):
    return 2100 if stage == 'eval' else 6000 + evaluation_reserve(seconds)


def plan(job, root, target, seconds=57600):
    o = origin(root); prior, report = o['manifest'], o['checkpoint_report']
    start = report['iteration']+1
    if next_stage(start, {i for i in (90, 100) if i <= start}) != ('train', target):
        raise ValueError('Do not skip the90 checkpoint boundary')
    if not 600 <= seconds <= 57600:
        raise ValueError('Invalid time request')
    # The new allocation starts only after the current recovery has produced a
    # durable TP4 checkpoint; CPU preparation uses the older82 receipt.
    if job.isdecimal() and (start < 83 or prior['environment']['TP_SIZE'] != '4'):
        raise ValueError('Require a verified durable TP4 recovery checkpoint')
    p = copy.deepcopy(prior)
    output = RUNTIME/f'evaluations/arm-failure-additive-{job}-b100-iter{target}'
    def rewrite(v):
        return v.replace(prior['output'], str(output)).replace(prior['source'], str(SOURCE)) if isinstance(v, str) else v
    p['command'] = [rewrite(v) for v in p['command']]
    for key in ('environment', 'arm_config'):
        p[key] = {k: rewrite(v) for k, v in p[key].items()}
    if '--use-checkpoint-opt-param-scheduler' not in p['command']:
        p['command'].append('--use-checkpoint-opt-param-scheduler')
    for key in ('recovery_budget', 'replay_provenance'):
        p.pop(key, None)
    for key in ('deadline_epoch_seconds', 'gate_replay_origin', 'replay_origin'):
        p['arm_config'].pop(key, None)
    for key in ('OPENWEBRL_REPLAY_FIRST_BATCH', 'OPENWEBRL_REPLAY_ROLLOUT_ID', 'OPENWEBRL_ARM_REPLAY_CURSOR'):
        p['environment'].pop(key, None)
    p.update(job_id=job, source=str(SOURCE), output=str(output), resume_from=str(root), resume_origin=o,
        start_rollout_id=start, requested_iterations=target-start, target_completed_iterations=target,
        initial_optimizer_updates=report['completed_optimizer_updates'], checkpoint=report['checkpoint'],
        fresh_optimizer=False, variant_continuation=True, gate_b100_continuation=True,
        requested_resources=dict(RESOURCES, hours=seconds/3600, gpu_hours=8*seconds/3600),
        evaluation_in_allocation=True, evaluation_reserve_seconds=evaluation_reserve(seconds),
        continuation_attempt_seconds=seconds,
        gpu_restore_output=str(Path(root)/f'gpu-restore-b100-{job}-to{target}'))
    p['resume_restore_receipt'] = str(Path(p['gpu_restore_output'])/'result.json')
    scratch = RUNTIME/f'multimodal-scratch/arm-variant-{job}'
    p['multimodal_storage'] = dict(mode='shared', directory=str(scratch), minimum_free_bytes=2*1024**4,
                                  retention='preserve; no automatic deletion')
    p['environment'].update(NUM_GPUS='8', TP_SIZE='4', NUM_ROLLOUT=str(target),
        SLIME_LOAD_CHECKPOINT=str(Path(root)/'runtime'), WANDB_RUN_ID=previous.RUN_ID,
        WANDB_PROJECT='openwebrl', WANDB_RESUME='must', PYTHONDONTWRITEBYTECODE='1',
        OPENWEBRL_MULTIMODAL_STORAGE_DIR=str(scratch), RAY_TMPDIR=f'/tmp/b100-{job}-{target}',
        OPENWEBRL_CUDA_CACHE_LIMIT_GIB='24', PYTHONFAULTHANDLER='1', RAY_DEDUP_LOGS='0')
    p['arm_config'].update(checkpoint=report['checkpoint'], policy_id=previous.RUN_ID+':uninitialized',
        minimum_cycle_seconds=6000, seconds_per_optimizer_update=330)
    pending = Path(root)/f'iterations/{start:04d}/collection_complete.json'
    if pending.exists():
        replay = previous.replay_provenance(root)
        p['replay_provenance'] = replay
        p['arm_config'].update(gate_replay_origin=replay, replay_origin=replay)
        p['environment'].update(OPENWEBRL_REPLAY_FIRST_BATCH=replay['batch'],
            OPENWEBRL_REPLAY_ROLLOUT_ID=str(replay['rollout_id']), OPENWEBRL_ARM_REPLAY_CURSOR=replay['dataset_cursor'])
    c, e = p['arm_config'], p['environment']
    if (p.get('gate_ablation'), p['wandb_run_id'], c['candidate_gate'], c['credit_assignment'],
        c['beta'], c['scored_fraction'], c['failure_group_cap'], c['failure_loss_coefficient'],
        c['additive_failure_groups'], e['GLOBAL_BATCH_SIZE'], e['BROWSER_CONCURRENCY']) != (
        'B', previous.RUN_ID, 'min2', 'response_index', .5, .2, 8, 1/6, True, '256', '64'):
        raise ValueError('Historical Gate B scientific recipe changed')
    return p


def fingerprints():
    files = [Path(__file__).resolve(), SOURCE/'reference_manifest.json',
        CONTROL/'training-controller.py', CONTROL/'restore-controller.py',
        REPO/'scripts/resume_arm_gate_b100_8gpu.sbatch', REPO/'tests/test_arm_gate_b100.py',
        Path(previous.__file__), REPO/'scripts/resume_arm_mixed_to60.py']
    return {str(p): sha(p) for p in files}


def validate_plan(p):
    validate_source(SOURCE)
    if p != plan(p['job_id'], Path(p['resume_from']), p['target_completed_iterations'], p['continuation_attempt_seconds']):
        raise ValueError('Prepared continuation changed')
    if read(CONTROL/'readiness.json')['fingerprints'] != fingerprints():
        raise ValueError('CPU readiness is stale')
    if p['job_id'].isdecimal() and budget(p['job_id'])['attempt_seconds'] != p['continuation_attempt_seconds']:
        raise ValueError('Plan and approved attempt disagree')
    return dict(passed=True, objective_unchanged=True, native_gpu_restore_required=True, target=100)


def roots(state):
    return list(dict.fromkeys([str(ORIGINAL), str(PREDECESSOR), *state.get('training_roots', [])]))


def choose_origin(state):
    found = []
    for name in roots(state):
        root = Path(name); marker = root/'runtime/latest_checkpointed_iteration.txt'
        if not marker.exists():
            continue
        idx = int(marker.read_text())
        if not (root/f'iterations/{idx:04d}/checkpoint-validation.json').exists():
            continue
        o = origin(root)
        found.append((o['checkpoint_report']['iteration'], len(found), root))
    if not found:
        raise ValueError('No durable inactive Gate B checkpoint')
    return max(found)[2]


def evaluation_plan(job, root, target):
    return previous.evaluation_plan(job, root, target)


def register(job, controller, training, state, **extra):
    b = budget(job)
    with (SUPERVISOR/'registry-update.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        registry = read(SUPERVISOR/'registry.json')
        j = next(v for v in registry['jobs'] if v['key'] == 'gate-b')
        if j['job_id'] != job:
            j.setdefault('supervised_attempts', []).append(dict(job_id=j['job_id'], controller_root=j['controller_root'],
                training_root=j['training_root'], prior_budget_seconds=j['approved_total_seconds']))
            j.setdefault('evaluation_history', []).extend(j.get('evaluations', []))
            j.pop('last_budget_check', None)
        j.update(job_id=job, controller_root=str(controller), training_root=str(training), requested_endpoint=100,
            target_iteration=100, requires_user=False, verified_complete=False,
            approved_total_seconds=b['approved_total_seconds'], prior_used_seconds=b['prior_used_seconds'], attempt_seconds=b['attempt_seconds'],
            budget_scope=b.get('budget_scope', 'separate16h extension to100'), budget_receipt=str(CONTROL/f'{job}-budget.json'),
            evaluations=[dict(iteration=int(k), root=v['root'], verified=True) for k,v in state['evaluations'].items()])
        j.update(extra)
        atomic(SUPERVISOR/'registry.json', registry)


def worker(job, stage, manifest):
    if os.getenv('SLURM_JOB_ID') != job:
        raise ValueError('Wrong allocation')
    p = read(manifest)
    if stage == 'eval':
        import evaluate_baseline_checkpoint as ev
        os.environ.update(OPENWEBRL_EXPECTED_SCHEDULER_OFFSET='0', OPENWEBRL_RECORD_EVAL_EXPECTED_TASK_COUNT='300')
        ev.run(p, REPO/'.env')
        return
    validate_plan(p)
    from resume_arm_failure_variants import check_multimodal_storage
    check_multimodal_storage(p)
    probe = f"import sys,runpy;sys.path.insert(0,{str(REPO/'scripts')!r});runpy.run_path({str(CONTROL/'restore-controller.py')!r},run_name='__main__')"
    subprocess.run(source_command(SOURCE, [str(PYTHON), '-B', '-c', probe, '--training-root', p['resume_from'],
        '--job-id', job, '--gpus', '8', '--output', p['gpu_restore_output'], '--continuation-plan', str(manifest), '--execute']), check=True)
    validate_resume_plan(p, require_gpu_restore=True)
    spec = importlib.util.spec_from_file_location('b100_training', CONTROL/'training-controller.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.execute(p)


def execute(job):
    lock = (CONTROL/'execution.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    b = budget(job)
    if os.getenv('SLURM_JOB_ID') != job or int(os.getenv('SLURM_RESTART_COUNT', '0')) != b.get('restart_count', 0):
        raise ValueError('Unregistered allocation/restart')
    def capacity():
        return allocation(subprocess.check_output(['scontrol','show','job',job,'-o'], text=True), job,
            requested_gpus=8, maximum_hours=b['attempt_seconds']/3600)['maximum_seconds']
    capacity()
    controller = CONTROL/f'controller-{job}'; controller.mkdir(exist_ok=False)
    statefile = CONTROL/'progress.json'
    state = read(statefile) if statefile.exists() else dict(training_roots=[], evaluations={})
    resume = choose_origin(state)
    # Reuse90 only after checking the checkpoint and all saved task artifacts.
    old_eval = RUNTIME/'evaluations/arm-gate-b-iter90-335681'
    if '90' not in state['evaluations'] and (old_eval/'status.json').exists():
        status = read(old_eval/'status.json')
        if status.get('complete') and status.get('returncode') == 0:
            ep = evaluation_plan('335681', PREDECESSOR, 90)
            audit = audit_evaluation(ep); audit.update(iteration=90)
            atomic(controller/'reused-evaluation90-plan.json', ep)
            audit['plan'] = str(controller/'reused-evaluation90-plan.json')
            state['evaluations']['90'] = audit; atomic(statefile, state)
    step = ['srun', f'--jobid={job}', '--nodes=1', '--ntasks=1', '--cpus-per-task=64',
        '--gres=gpu:h200:8', '--exact', '--cpu-bind=none', str(PYTHON), '-B', str(Path(__file__).resolve()), '--job-id', job]
    register(job, controller, resume, state)
    try:
        while True:
            completed = origin(resume)['checkpoint_report']['iteration']+1
            stage, target = next_stage(completed, {int(i) for i in state['evaluations']})
            if stage == 'complete':
                for item in state['evaluations'].values():
                    audit_evaluation(read(item['plan']))
                atomic(controller/'status.json', dict(complete=True, stage='complete', target=100, completed_stages=[90,100]))
                register(job, controller, resume, state, verified_complete=True,
                    artifact_checks='durable100 and full300 evaluation90/100 cohorts verified')
                return
            remaining = capacity()
            if remaining < minimum_stage_seconds(stage, b['attempt_seconds']):
                atomic(controller/'status.json', dict(complete=False, stage='budget-stop', durable_iteration=completed,
                    next_stage=stage, next_target=target, remaining_seconds=remaining)); return
            if stage == 'train':
                p = plan(job, resume, target, b['attempt_seconds']); validate_plan(p)
            else:
                training = next((Path(v) for v in reversed(roots(state))
                    if (Path(v)/f'iterations/{target-1:04d}/checkpoint-validation.json').exists()), resume)
                p = evaluation_plan(job, training, target)
            manifest = controller/f'{stage}-{target}-plan.json'; atomic(manifest, p)
            training_root = p['output'] if stage == 'train' else str(resume)
            atomic(controller/'controller-plan.json', dict(job_id=job, target=100, current_target=target, training_root=training_root))
            atomic(controller/'status.json', dict(complete=False, stage='training' if stage=='train' else 'evaluation',
                iteration=target, durable_iteration=completed, completed_stages=sorted(map(int,state['evaluations']))))
            evaluations = [dict(iteration=int(k),root=v['root'],verified=True) for k,v in state['evaluations'].items()]
            if stage == 'eval': evaluations.append(dict(iteration=target,root=p['output'],verified=False))
            register(job, controller, training_root, state, evaluations=evaluations)
            subprocess.run([*step, '--worker', stage, '--manifest', str(manifest)], check=True)
            if stage == 'train':
                state['training_roots'].append(p['output']); atomic(statefile, state)
                resume = choose_origin(state)
                actual = origin(resume)['checkpoint_report']['iteration']+1
                if actual < target:
                    for f in (Path(p['output'])/'iterations').glob('*/training_gate.json'):
                        if not read(f).get('calibration',{}).get('passed'):
                            raise ValueError('Calibration failure requires repair')
                    atomic(controller/'status.json', dict(complete=False,stage='budget-stop',durable_iteration=actual,target=target)); return
            else:
                audit = audit_evaluation(p); audit.update(iteration=target,plan=str(manifest))
                atomic(controller/f'evaluation{target}-audit.json', audit)
                state['evaluations'][str(target)] = audit; atomic(statefile,state)
    except BaseException as exc:
        atomic(controller/'status.json', dict(complete=False,stage='failed',error_type=type(exc).__name__,error=str(exc)[:600]))
        raise


def prepare():
    validate_source(SOURCE)
    for name in ('training-controller.py','restore-controller.py'):
        text = (previous.CONTROL/name).read_text().replace('resume_arm_gate_b90','resume_arm_gate_b100')
        compile(text,name,'exec'); (CONTROL/name).write_text(text)
    p = plan('PREPARE', ORIGINAL, 90)
    from resume_arm_failure_variants import check_multimodal_storage
    atomic(CONTROL/'storage-preflight.json', check_multimodal_storage(p))
    atomic(CONTROL/'browser-preflight.json', p['browser_config'])
    env = dict(clean_environment(), **p['environment'])
    env.update(DRY_RUN='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        BROWSER_TRAIN_CONFIG=str(CONTROL/'browser-preflight.json'))
    argv = shlex.split(subprocess.check_output(p['command'],env=env,text=True)); env.pop('DRY_RUN')
    expected = dict(actor_num_gpus_per_node=8,tensor_model_parallel_size=4,global_batch_size=256,
        micro_batch_size=1,ppo_epochs=2,rollout_batch_size=48,browser_rollout_concurrency=64,
        judge_api_model='gpt-4.1',judge_prompt_variant='action_history',use_rollout_logprobs=True,
        dynamic_sampling_filter_path='openwebrl.arm_failure_additive.filter_groups',lr=1e-6,
        num_rollout=90,use_checkpoint_opt_param_scheduler=True)
    code = ("from unittest.mock import patch\nfrom slime.utils.arguments import parse_args\n"
        "from ray._private.utils import validate_socket_filepath\nimport os\n"
        "validate_socket_filepath(os.environ['RAY_TMPDIR']+'/ray/session_2099-12-31_23-59-59_999999_9999999/sockets/plasma_store')\n"
        "with patch('megatron.training.arguments.get_device_arch_version',return_value=9): a=parse_args()\n"
        f"expected={expected!r}\n"
        "assert all(getattr(a,k)==v for k,v in expected.items()),{k:getattr(a,k) for k in expected}\n"
        "print('GATE_B100_NATIVE_PARSE_OK')\n")
    with (CONTROL/'native-parse.log').open('w') as log:
        subprocess.run(source_command(SOURCE,[str(PYTHON),'-B','-c',code,*argv[2:]]),env=env,cwd=SOURCE,
            stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
    atomic(CONTROL/'B82-to90-preview.json',p)
    ep = evaluation_plan('PREPARE',ORIGINAL,82)
    if ep['expected_task_count'] != 300 or not ep['require_task_rollouts']:
        raise ValueError('Expected full300 persistence')
    atomic(CONTROL/'evaluation-protocol-preview.json',ep)
    subprocess.run(['bash','-n',str(REPO/'scripts/resume_arm_gate_b100_8gpu.sbatch')],check=True)
    atomic(CONTROL/'readiness.json',dict(cpu_passed=True,native_parse_passed=True,fingerprints=fingerprints(),
        native_gpu_restore_required=True,predecessor_durable_TP4_required=True))
    validate_plan(p)
    print(CONTROL/'readiness.json')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true'); parser.add_argument('--job-id')
    parser.add_argument('--execute',action='store_true'); parser.add_argument('--worker',choices=['train','eval'])
    parser.add_argument('--manifest',type=Path); args=parser.parse_args()
    if args.prepare: prepare()
    elif args.execute: execute(args.job_id)
    elif args.worker: worker(args.job_id,args.worker,args.manifest)
    else: parser.error('Choose --prepare, --execute or --worker')
