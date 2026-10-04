#!/usr/bin/env python3
"""Prepare/own Gate B77 ->80/eval ->90/eval; never submit an allocation."""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time

from recover_arm_browser_slots import (
    CONTROL as PREVIOUS, ORIGIN, SOURCE, RUN_ID, PYTHON, origin,
    prepare_source, sha, validate_resume_plan, continuation_identity,
)
from resume_baseline import REPO, RUNTIME, allocation, clean_environment, source_command, write_json

CONTROL = RUNTIME/'arm-turn-bonus-preparation/gate-b90-20260927'
RESOURCES = dict(gpus=8, gpu_type='H200', hours=16, gpu_hours=128,
                 cpus=64, memory_gib=960, browsers=64)


def read(path):
    return json.loads(Path(path).read_text())


def recovery(job):
    path = CONTROL/f'{job}-recovery.json'
    if not path.exists():
        return None
    r = read(path)
    if (r['job_id'] != job or r['approved_total_seconds'] != 57600
            or r['prior_used_seconds'] + r['attempt_seconds'] > 57600
            or r['attempt_seconds'] <= 0 or r['cache_limit_gib'] != 24
            or r.get('tensor_parallel', 2) not in (2, 4)
            or r['target_iteration'] != 90):
        raise ValueError('Invalid bounded Gate B recovery')
    return r


def replay_provenance(root):
    """Verify the saved next batch without loading its tensors on the login node."""
    import zipfile
    root = Path(root)
    o = origin(root); report = o['checkpoint_report']; rid = report['iteration'] + 1
    folder = root/'iterations'/f'{rid:04d}'
    complete = read(folder/'collection_complete.json'); gate = read(folder/'training_gate.json')
    auxiliary = read(folder/'failure_auxiliary.json')
    batch = root/f'runtime/rollout_recovery/{rid}.pt'
    cursor = root/f'runtime/rollout/global_dataset_state_dict_{rid}.pt'
    if (complete['rollout_id'] != rid or complete['checkpoint'] != report['checkpoint']
            or complete['recovery_file'] != str(batch) or complete['dataset_cursor'] != str(cursor)
            or not gate['calibration']['passed'] or (root/f'runtime/iter_{rid:07d}').exists()):
        raise ValueError('Replay batch/checkpoint identity changed')
    if (auxiliary['candidate_gate'], auxiliary['credit_assignment'], auxiliary['beta'], auxiliary['q']) != ('min2', 'response_index', .5, .2):
        raise ValueError('Replay scientific recipe changed')
    archives = {}
    for path in (batch, cursor, folder/'failure_auxiliary.pt'):
        stat = path.stat()
        with zipfile.ZipFile(path) as z:
            if not any(n.endswith('/data.pkl') for n in z.namelist()):
                raise ValueError('Incomplete replay archive')
        archives[str(path)] = dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    return dict(source=str(root), rollout_id=rid, checkpoint=report['checkpoint'],
        batch=str(batch), batch_bytes=batch.stat().st_size, dataset_cursor=str(cursor),
        cursor_sha256=sha(cursor), auxiliary_manifest_sha256=sha(folder/'failure_auxiliary.json'),
        auxiliary_bytes=(folder/'failure_auxiliary.pt').stat().st_size,
        calibration_sha256=sha(folder/'calibration.json'), archives=archives)


def plan(job, root=ORIGIN, target=80):
    if target not in (80, 90):
        raise ValueError('Only the requested 80/90 milestones')
    checkpoint = origin(root)
    prior = checkpoint['manifest']; report = checkpoint['checkpoint_report']
    start = report['iteration'] + 1
    if not (77 <= start < target) or (target == 90 and start < 80):
        raise ValueError('Resume must preserve B77 or a later durable checkpoint')
    p = copy.deepcopy(prior)
    r = recovery(job)
    if r and (str(Path(root)) != r['resume_from'] or target != 90):
        raise ValueError('Recovery must resume the declared durable checkpoint toward90')
    suffix = f'-r{r["attempt"]}' if r else ''
    output = RUNTIME/f'evaluations/arm-failure-additive-{job}-b90{suffix}-iter{target}'
    def rewrite(value):
        return value.replace(prior['output'], str(output)).replace(prior['source'], str(SOURCE)) if isinstance(value, str) else value
    p['command'] = [rewrite(value) for value in p['command']]
    for key in ('environment', 'arm_config'):
        p[key] = {k: rewrite(v) for k, v in p[key].items()}
    p['arm_config'].pop('deadline_epoch_seconds', None)
    p.update(job_id=job, source=str(SOURCE), output=str(output), resume_from=str(root),
             resume_origin=checkpoint, start_rollout_id=start, requested_iterations=target-start,
             target_completed_iterations=target, initial_optimizer_updates=report['completed_optimizer_updates'],
             checkpoint=report['checkpoint'], fresh_optimizer=False, variant_continuation=True,
             requested_resources=dict(RESOURCES), evaluation_in_allocation=True,
             evaluation_reserve_seconds=7200 if target == 80 else 3600,
             gpu_restore_output=str(Path(root)/f'gpu-restore-b90-{job}-to{target}'),
             gate_b90_continuation=True)
    p['resume_restore_receipt'] = str(Path(p['gpu_restore_output'])/'result.json')
    scratch = RUNTIME/f'multimodal-scratch/arm-variant-{job}'
    p['multimodal_storage'] = dict(mode='shared', directory=str(scratch),
        minimum_free_bytes=2*1024**4, retention='preserve; no automatic deletion')
    p['environment'].update(NUM_GPUS='8', TP_SIZE='2', NUM_ROLLOUT=str(target),
        SLIME_LOAD_CHECKPOINT=str(Path(root)/'runtime'), WANDB_RUN_ID=RUN_ID,
        WANDB_PROJECT='openwebrl', WANDB_RESUME='must', PYTHONDONTWRITEBYTECODE='1',
        OPENWEBRL_MULTIMODAL_STORAGE_DIR=str(scratch), RAY_TMPDIR=f'/tmp/gb-{job}')
    p['arm_config'].update(checkpoint=report['checkpoint'], policy_id=RUN_ID+':uninitialized',
                          minimum_cycle_seconds=3600, seconds_per_optimizer_update=150)
    if r:
        replay = replay_provenance(root)
        if replay != r['replay_provenance']:
            raise ValueError('Saved replay artifacts changed after preparation')
        p['requested_resources'].update(hours=r['attempt_seconds']/3600,
            gpu_hours=8*r['attempt_seconds']/3600)
        p['gpu_restore_output'] = str(Path(root)/f'gpu-restore-b90-{job}-r{r["attempt"]}-to90')
        p['resume_restore_receipt'] = str(Path(p['gpu_restore_output'])/'result.json')
        p['replay_provenance'] = replay
        p['environment'].update(OPENWEBRL_CUDA_CACHE_LIMIT_GIB='24',
            TP_SIZE=str(r.get('tensor_parallel', 2)),
            PYTHONFAULTHANDLER='1', RAY_DEDUP_LOGS='0',
            OPENWEBRL_REPLAY_FIRST_BATCH=replay['batch'],
            OPENWEBRL_REPLAY_ROLLOUT_ID=str(replay['rollout_id']),
            OPENWEBRL_ARM_REPLAY_CURSOR=replay['dataset_cursor'],
            RAY_TMPDIR=f'/tmp/gb-{job}-r{r["attempt"]}')
        p['arm_config'].update(gate_replay_origin=replay, replay_origin=replay,
            seconds_per_optimizer_update=180)
        p['recovery_budget'] = r
    c, e = p['arm_config'], p['environment']
    if (p.get('gate_ablation'), p['wandb_run_id'], c['candidate_gate'], c['credit_assignment'],
        c['beta'], c['scored_fraction'], c['failure_group_cap'], c['failure_loss_coefficient'],
        c['additive_failure_groups'], e['GLOBAL_BATCH_SIZE'], e['BROWSER_CONCURRENCY']) != (
        'B', RUN_ID, 'min2', 'response_index', .5, .2, 8, 1/6, True, '256', '64'):
        raise ValueError('Gate B scientific recipe changed')
    return p


def fingerprints():
    return {str(p): sha(p) for p in (Path(__file__).resolve(), SOURCE/'reference_manifest.json',
        CONTROL/'training-controller.py', CONTROL/'restore-controller.py',
        REPO/'scripts/resume_arm_gate_b90_8gpu.sbatch', REPO/'scripts/arm_browser_startup_guard.py',
        REPO/'scripts/record_arm_gpu_memory.py')}


def validate_plan(p):
    if p != plan(p['job_id'], Path(p['resume_from']), p['target_completed_iterations']):
        raise ValueError('Continuation differs from the prepared plan')
    receipt = read(CONTROL/'readiness.json')
    if not receipt['cpu_passed'] or receipt['fingerprints'] != fingerprints():
        raise ValueError('Missing or stale readiness')
    return dict(passed=True, objective_unchanged=True, native_gpu_restore_required=True,
                browser_startup_guard=True, browser_root_cause_unconfirmed=True)


def evaluation_plan(job, root, target):
    from prepare_arm_gate_to60 import evaluation_plan as existing
    import evaluate_baseline_checkpoint as evaluator
    p = existing('B', root, target, job)
    p['environment']['PYTHONDONTWRITEBYTECODE']='1'
    return evaluator.configure_evaluation_tracking(p, 'openwebrl')


def native_check(p):
    write_json(CONTROL/'browser-preflight.json', p['browser_config'])
    env = dict(clean_environment(), **p['environment'])
    env.update(DRY_RUN='1', CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               BROWSER_TRAIN_CONFIG=str(CONTROL/'browser-preflight.json'))
    argv = shlex.split(subprocess.check_output(p['command'], env=env, text=True))
    env.pop('DRY_RUN')
    expected = dict(actor_num_gpus_per_node=8, tensor_model_parallel_size=int(p['environment']['TP_SIZE']), global_batch_size=256,
        micro_batch_size=1, ppo_epochs=2, rollout_batch_size=48, browser_rollout_concurrency=64,
        judge_api_model='gpt-4.1', judge_prompt_variant='action_history', use_rollout_logprobs=True,
        dynamic_sampling_filter_path='openwebrl.arm_failure_additive.filter_groups', lr=1e-6,
        num_rollout=int(p['environment']['NUM_ROLLOUT']), use_checkpoint_opt_param_scheduler=True)
    code = ("from unittest.mock import patch\nfrom slime.utils.arguments import parse_args\n"
            "from ray._private.utils import validate_socket_filepath\nimport os\n"
            "validate_socket_filepath(os.environ['RAY_TMPDIR']+'/ray/session_2099-12-31_23-59-59_999999_9999999/sockets/plasma_store')\n"
            "with patch('megatron.training.arguments.get_device_arch_version',return_value=9): a=parse_args()\n"
            f"expected={expected!r}\n"
            "assert all(getattr(a,k)==v for k,v in expected.items()),{k:getattr(a,k) for k in expected}\n"
            "print('GATE_B90_NATIVE_PARSE_OK')\n")
    with (CONTROL/'native-parse.log').open('w') as handle:
        subprocess.run(source_command(SOURCE, [str(PYTHON), '-c', code, *argv[2:]]),
            env=env, cwd=SOURCE, stdout=handle, stderr=subprocess.STDOUT, check=True, timeout=120)


def prepare():
    prepare_source(); CONTROL.mkdir(parents=True, exist_ok=True)
    sources = {'training-controller.py': PREVIOUS/'prepared-next-training-controller.py',
               'restore-controller.py': PREVIOUS/'restore-controller.py'}
    for name, source in sources.items():
        data = source.read_text().replace('from recover_arm_browser_slots import',
                                         'from resume_arm_gate_b90 import')
        compile(data, name, 'exec')
        (CONTROL/name).write_text(data)
    p = plan('PREPARE'); write_json(CONTROL/'B77-to80-plan.json', p)
    native_check(p)
    # Use durable77 to check the real evaluation protocol without evaluating it.
    ev = evaluation_plan('PREPARE', ORIGIN, 77)
    if ev['expected_task_count'] != 300 or not ev['require_task_rollouts']:
        raise ValueError('Full-cohort rollout persistence missing')
    write_json(CONTROL/'evaluation-protocol-preview-at77.json', ev)
    code = (f"import sys,unittest;sys.path.insert(0,{str(REPO/'tests')!r});"
            "s=unittest.defaultTestLoader.loadTestsFromNames(['test_arm_browser_startup_guard','test_local_browser_cleanup']);"
            "r=unittest.TextTestRunner(verbosity=2).run(s);raise SystemExit(not r.wasSuccessful())")
    with (CONTROL/'cpu-tests.log').open('w') as handle:
        subprocess.run(source_command(SOURCE, [str(PYTHON), '-c', code]), cwd=SOURCE,
            env=dict(clean_environment(), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'),
            stdout=handle, stderr=subprocess.STDOUT, check=True, timeout=120)
    subprocess.run(['bash', '-n', str(REPO/'scripts/resume_arm_gate_b90_8gpu.sbatch')], check=True)
    write_json(CONTROL/'readiness.json', dict(cpu_passed=True, native_parse_passed=True,
        fingerprints=fingerprints(), gpu_restore_pending=True, representative_collection_pending=True))
    validate_plan(p)
    write_json(CONTROL/'proposal.json', dict(prepared=True, submitted=False, approval_pending=True,
        resources=RESOURCES, resume_iteration=77, resume_optimizer_updates=988,
        stages=['restore77','train80','full300-eval80','restore80','train90','full300-eval90'],
        original_approved_seconds=86400, original_consumed_seconds=82633,
        original_remaining_seconds_upper_bound=3767, old_budget_not_added_to_new_request=True,
        first_collection_is_guarded_live_validation=True, no_automatic_budget_extension=True))
    print(json.dumps(dict(prepared=str(CONTROL), resources=RESOURCES, approval_pending=True)))


def require_approval():
    receipt = read(CONTROL/'resource-approval.json')
    if receipt.get('approved') is not True or receipt.get('resources') != RESOURCES:
        raise ValueError('Exact 8 H200 x16h approval is required before execution')


def worker(job, root, target, stage):
    require_approval()
    if os.getenv('SLURM_JOB_ID') != job:
        raise ValueError('Wrong worker allocation')
    if stage == 'eval':
        import evaluate_baseline_checkpoint as evaluator
        from run_stage1_to100 import audit_rollouts
        p = evaluation_plan(job, root, target)
        os.environ.update(OPENWEBRL_EXPECTED_SCHEDULER_OFFSET='0', OPENWEBRL_RECORD_EVAL_EXPECTED_TASK_COUNT='300')
        evaluator.run(p, REPO/'.env')
        write_json(CONTROL/f'{job}-eval{target}-audit.json',
                   audit_rollouts(Path(p['output'])/'rollouts', Path(p['source'])))
        return
    p = plan(job, root, target); validate_plan(p)
    from resume_arm_failure_variants import check_multimodal_storage
    check_multimodal_storage(p)
    manifest = CONTROL/f'{job}-to{target}-launch.json'; write_json(manifest, p)
    probe = f"import sys,runpy;sys.path.insert(0,{str(REPO/'scripts')!r});runpy.run_path({str(CONTROL/'restore-controller.py')!r},run_name='__main__')"
    monitor = subprocess.Popen([str(PYTHON),str(REPO/'scripts/record_arm_gpu_memory.py'),
        '--root',p['output'],'--output',str(CONTROL/f'{job}-r{p.get("recovery_budget",{}).get("attempt",0)}-gpu-memory.jsonl'),
        '--first-iteration',str(p['start_rollout_id']+1),'--minutes','60'],
        stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:
        subprocess.run(source_command(SOURCE, [str(PYTHON), '-B', '-c', probe, '--training-root', str(root),
            '--job-id', job, '--gpus', '8', '--output', p['gpu_restore_output'],
            '--continuation-plan', str(manifest), '--execute']), check=True)
        validate_resume_plan(p, require_gpu_restore=True)
        spec = importlib.util.spec_from_file_location('b90_training', CONTROL/'training-controller.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); module.execute(p)
    finally:
        monitor.terminate()
        try: monitor.wait(timeout=10)
        except subprocess.TimeoutExpired: monitor.kill(); monitor.wait()
        for logs in (Path(p['environment']['RAY_TMPDIR'])/'ray/session_latest/logs',
                     Path(p['environment']['RAY_TMPDIR'])/'session_latest/logs'):
            if logs.is_dir():
                destination = Path(p['output'])/'preserved-ray-logs'; destination.mkdir(parents=True,exist_ok=True)
                for pattern in ('worker-*.err','raylet.err','raylet.out'):
                    for path in logs.glob(pattern): shutil.copy2(path,destination/path.name)


def execute(job):
    require_approval()
    if os.getenv('SLURM_JOB_ID') != job:
        raise ValueError('Existing authorized allocation required')
    r = recovery(job)
    if r and int(os.getenv('SLURM_RESTART_COUNT', '0')) != r.get('slurm_restart_count', r['attempt']):
        raise ValueError('Recovery restart counter disagrees with charged attempt')
    allocation(subprocess.check_output(['scontrol','show','job',job,'-o'], text=True), job,
               requested_gpus=8, maximum_hours=r['attempt_seconds']/3600 if r else 16)
    suffix = f'-r{r["attempt"]}' if r else ''
    out = RUNTIME/f'evaluations/arm-gate-b90-controller-{job}{suffix}'; out.mkdir(exist_ok=False)
    root = Path(r['resume_from']) if r else ORIGIN
    completed = [80] if r else []
    targets = (90,) if r else (80,90)
    write_json(out/'evaluation-queue.json', dict(targets=list(targets), tasks_each=300, awaited=True))
    if r: write_json(out/'budget.json', r)
    step = ['srun', f'--jobid={job}', '--nodes=1', '--ntasks=1', '--cpus-per-task=64',
        '--gres=gpu:h200:8', '--exact', '--cpu-bind=none', str(PYTHON), '-B',
        str(Path(__file__).resolve()), '--job-id', job]
    try:
        for target in targets:
            p = plan(job, root, target); validate_plan(p)
            write_json(out/'controller-plan.json', dict(job_id=job, variant='B', target=90,
                current_target=target, training_root=p['output'], resume_from=str(root)))
            write_json(out/'status.json', dict(stage='training', complete=False, target=target, completed_stages=completed))
            subprocess.run([*step, '--worker','train','--root',str(root),'--target',str(target)], check=True)
            training = Path(p['output'])
            from run_arm_gate_checkpoint_eval import checkpoint_ready
            if checkpoint_ready('B', training, target) is None:
                write_json(out/'status.json', dict(stage='budget-stop', complete=False, missing_target=target))
                return
            write_json(out/'status.json', dict(stage='evaluation', complete=False, iteration=target))
            subprocess.run([*step, '--worker','eval','--root',str(training),'--target',str(target)], check=True)
            completed.append(target); root = training
        write_json(out/'status.json', dict(stage='complete', complete=True, completed_stages=completed))
    except BaseException as exc:
        write_json(out/'status.json', dict(stage='failed', complete=False, error_type=type(exc).__name__, error=str(exc)[:500]))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true'); parser.add_argument('--execute', action='store_true')
    parser.add_argument('--job-id', default='PREPARE'); parser.add_argument('--root', type=Path, default=ORIGIN)
    parser.add_argument('--target', type=int, choices=(80,90), default=80)
    parser.add_argument('--worker', choices=('train','eval'))
    args = parser.parse_args()
    if args.prepare: prepare()
    elif args.worker: worker(args.job_id, args.root, args.target, args.worker)
    elif args.execute: execute(args.job_id)
    else: print(json.dumps(plan(args.job_id, args.root, args.target), indent=2))
