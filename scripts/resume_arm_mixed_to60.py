#!/usr/bin/env python3
"""Continue the matched mixed-only pair, owning every tenth-iteration evaluation.

The extension has its own explicit approval and per-variant attempt ledger.
No allocation is submitted by this controller. Existing eight-hour attempts
and their frozen recovery implementation are left unchanged.
"""
import argparse
import copy
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
import zipfile

import resume_arm_mixed_reweight as previous
from resume_baseline import allocation, clean_environment, source_command, validate_source, write_json

REPO, RUNTIME, PYTHON, SOURCE = previous.REPO, previous.RUNTIME, previous.PYTHON, previous.SOURCE
CONTROL = previous.original.CONTROL/'to60'
SUPERVISOR = previous.original.CONTROL/'supervisor'
TARGET = 60
MILESTONES = list(range(10, TARGET+1, 10))
RUN_IDS = {'bonus': 'arm-mixed-bonus-334493', 'reweight': 'arm-mixed-reweight-334494'}
origin = previous.origin
validate_resume_plan = previous.validate_resume_plan
validate_replayed_rewards = previous.validate_replayed_rewards
continuation_identity = previous.continuation_identity
read = previous.read


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f'.{os.getpid()}.partial')
    write_json(temporary, value)
    temporary.replace(path)


def approval():
    a = read(CONTROL/'approval.json')
    if (a.get('approved') is not True or a.get('target') != TARGET
            or a.get('variants') != ['bonus', 'reweight']
            or a.get('additional_seconds_per_variant') != 172800
            or a.get('resources') != dict(gpus=8, gpu_type='H200', cpus=64, memory_gib=960)
            or a.get('max_attempt_seconds') != 86400):
        raise ValueError('Exact mixed-pair continuation resource approval missing')
    return a


def attempt(job, variant):
    a = approval()
    ledger = read(CONTROL/f'{variant}-attempts.json')
    if ledger['variant'] != variant:
        raise ValueError('Wrong budget lineage')
    rows = ledger['attempts']
    if len({r['job_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate allocation in budget ledger')
    if any(not 0 <= r.get('charged_seconds', r['attempt_seconds']) <= r['attempt_seconds']
           or not 600 <= r['attempt_seconds'] <= a['max_attempt_seconds'] for r in rows):
        raise ValueError('Invalid consumed time or attempt reservation')
    if sum(r.get('charged_seconds', r['attempt_seconds']) for r in rows) > a['additional_seconds_per_variant']:
        raise ValueError('Continuation reservations exceed total approved budget')
    row = next(r for r in rows if r['job_id'] == job)
    if (not 600 <= row['attempt_seconds'] <= a['max_attempt_seconds']
            or row.get('restart_count', 0) != 0):
        raise ValueError('Unrecorded retry or invalid time request')
    return row


def next_stage(completed, evaluated):
    """Never train past a due but unevaluated milestone."""
    if not 0 < completed <= TARGET:
        raise ValueError('Invalid completed iteration')
    due = [i for i in MILESTONES if i <= completed and i not in evaluated]
    if due:
        return 'eval', due[0]
    if completed == TARGET:
        return 'complete', TARGET
    return 'train', min(TARGET, (completed//10+1)*10)


def refresh_accounting(variant):
    path = CONTROL/f'{variant}-attempts.json'
    with (CONTROL/f'{variant}-budget.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = read(path)
        ids = ','.join(r['job_id'] for r in ledger['attempts'])
        raw = subprocess.check_output(['sacct', '-X', '-n', '-P', '-j', ids,
            '--format=JobID,State,ElapsedRaw'], text=True, timeout=30)
        records = {v[0]: v[1:] for line in raw.splitlines() if len(v := line.split('|')) >= 3}
        for row in ledger['attempts']:
            fields = records.get(row['job_id'])
            if fields and fields[1].isdigit():
                row.update(scheduler_state=fields[0], elapsed_seconds=int(fields[1]))
                if fields[0].split()[0] in ('COMPLETED', 'FAILED', 'TIMEOUT', 'OUT_OF_MEMORY', 'CANCELLED', 'NODE_FAIL'):
                    row['charged_seconds'] = int(fields[1])
        atomic(path, ledger)
        return ledger


def training_plan(job, variant, root, target, seconds=86400):
    o = origin(root)
    prior, report = o['manifest'], o['checkpoint_report']
    start = report['iteration']+1
    if (prior['mixed_reweight_variant'] != variant or prior['wandb_run_id'] != RUN_IDS[variant]
            or next_stage(start, [i for i in MILESTONES if i <= start]) != ('train', target)):
        raise ValueError('Wrong lineage or nonadjacent milestone')
    if not 600 <= seconds <= 86400:
        raise ValueError('Attempt must fit the approved per-job cap')
    p = copy.deepcopy(prior)
    output = RUNTIME/f'evaluations/arm-mixed-{variant}-{job}-iter{target}'
    def rewrite(value):
        return value.replace(prior['output'], str(output)).replace(prior['source'], str(SOURCE)) if isinstance(value, str) else value
    p['command'] = [rewrite(x) for x in p['command']]
    for key in ('environment', 'arm_config'):
        p[key] = {k: rewrite(v) for k, v in p[key].items()}
    if '--use-checkpoint-opt-param-scheduler' not in p['command']:
        p['command'].append('--use-checkpoint-opt-param-scheduler')
    p.pop('recovery_budget', None)
    p.update(job_id=job, source=str(SOURCE), output=str(output), resume_from=str(Path(root)),
        resume_origin=o, fresh_optimizer=False, variant_continuation=True, mixed_to60=True,
        start_rollout_id=start, initial_optimizer_updates=report['completed_optimizer_updates'],
        checkpoint=report['checkpoint'], requested_iterations=target-start,
        target_completed_iterations=target, evaluation_reserve_seconds=3600,
        to60_attempt_seconds=seconds,
        gpu_restore_output=str(Path(root)/f'gpu-restore-{job}-to{target}'))
    p['resume_restore_receipt'] = str(Path(p['gpu_restore_output'])/'result.json')
    p['requested_resources'].update(hours=seconds/3600, gpu_hours=8*seconds/3600)
    p['environment'].update(SLIME_LOAD_CHECKPOINT=str(Path(root)/'runtime'), WANDB_RESUME='must',
        NUM_ROLLOUT=str(target), OPENWEBRL_CUDA_CACHE_LIMIT_GIB='48',
        PYTHONDONTWRITEBYTECODE='1', RAY_TMPDIR=f'/tmp/m60-{job}-{target}')
    scratch = str(RUNTIME/f'multimodal-scratch/arm-variant-{job}')
    p['environment']['OPENWEBRL_MULTIMODAL_STORAGE_DIR'] = scratch
    p['multimodal_storage']['directory'] = scratch
    for key in ('deadline_epoch_seconds', 'gate_replay_origin', 'replay_origin'):
        p['arm_config'].pop(key, None)
    for key in ('OPENWEBRL_REPLAY_FIRST_BATCH', 'OPENWEBRL_REPLAY_ROLLOUT_ID', 'OPENWEBRL_ARM_REPLAY_CURSOR'):
        p['environment'].pop(key, None)
    p['arm_config'].update(checkpoint=report['checkpoint'], policy_id=p['wandb_run_id']+':uninitialized',
        minimum_cycle_seconds=3000, seconds_per_optimizer_update=120)
    replay = previous.replay_provenance(root)
    if replay:
        p['arm_config'].update(replay_origin=replay, gate_replay_origin=replay)
        p['environment'].update(OPENWEBRL_REPLAY_FIRST_BATCH=replay['batch'],
            OPENWEBRL_REPLAY_ROLLOUT_ID=str(replay['rollout_id']), OPENWEBRL_ARM_REPLAY_CURSOR=replay['dataset_cursor'])
    c, e = p['arm_config'], p['environment']
    if ((c['advantage_mode'], c['beta'], c['scored_fraction'], c['reweight_lambda'],
         c['candidate_gate'], c['credit_assignment'], c['failure_group_cap'],
         c['failure_loss_coefficient'], c['additive_failure_groups'], c['admit_all_failure_groups']) !=
            (previous.original.MODES[variant], .5, .2, .5, 'min2', 'response_index', 0, 0., False, False)
            or (e['NUM_GPUS'], e['TP_SIZE'], e['GLOBAL_BATCH_SIZE'], e['BROWSER_CONCURRENCY']) != ('8','2','256','64')
            or 'OPENWEBRL_ARM_FAILURE_AUX_MANIFEST' in e
            or p['command'][p['command'].index('--dynamic-sampling-filter-path')+1] != previous.original.FILTER):
        raise ValueError('Scientific recipe changed')
    return p


def fingerprints():
    files = [Path(__file__).resolve(), SOURCE/'reference_manifest.json',
        CONTROL/'training-controller.py', CONTROL/'restore-controller.py',
        REPO/'scripts/resume_arm_mixed_to60_8gpu.sbatch', Path(previous.__file__),
        REPO/'tests/test_arm_mixed_to60.py']
    return {str(p): previous.original.sha(p) for p in files}


def validate_plan(p):
    validate_source(SOURCE)
    expected = training_plan(p['job_id'], p['mixed_reweight_variant'], p['resume_from'],
                             p['target_completed_iterations'], p['to60_attempt_seconds'])
    if p != expected or read(CONTROL/'readiness.json')['fingerprints'] != fingerprints():
        raise ValueError('Continuation plan or verified implementation changed')
    if p['job_id'].isdecimal():
        if attempt(p['job_id'], p['mixed_reweight_variant'])['attempt_seconds'] != p['to60_attempt_seconds']:
            raise ValueError('Plan and allocation budget differ')
    return dict(passed=True, objective_unchanged=True, native_gpu_restore_required=True,
                target=p['target_completed_iterations'], save_rollouts=True)


def evaluation_plan(job, variant, root, target):
    import run_arm_iteration80_eval as ev
    p = ev.plan('original', job, target, root)
    old = p['output']
    out = str(RUNTIME/f'evaluations/arm-mixed-{variant}-iter{target}-{job}')
    p['command'] = [x.replace(old, out) for x in p['command']]
    p['environment'] = {k: v.replace(old, out) for k, v in p['environment'].items()}
    p.update(output=out, gpus=8, wandb_run_id=Path(out).name,
             arm_variant=f'mixed-{variant}', parent_training_run=RUN_IDS[variant])
    p['environment'].update(NUM_GPUS='8', TP_SIZE='2', WANDB_RUN_ID=p['wandb_run_id'])
    p['command'][p['command'].index('--wandb-group')+1] = f'ARM mixed {variant} | iter{target} | full300'
    return ev.evaluation.configure_evaluation_tracking(p, 'openwebrl')


def audit_evaluation(ep):
    from run_stage1_to100 import audit_rollouts
    root = Path(ep['output'])
    status = read(root/'status.json')
    if not status.get('complete') or status.get('returncode') != 0:
        raise ValueError('Evaluation worker has not completed successfully')
    result = audit_rollouts(root/'rollouts', Path(ep['source']))
    for f in (root/'rollouts').glob('*.json'):
        row = read(f)
        archive = (root/'rollouts'/row['rollout_file']).resolve()
        if not archive.is_relative_to((root/'rollouts').resolve()):
            raise ValueError('Rollout archive escaped its evaluation directory')
        with zipfile.ZipFile(archive) as z:
            if not z.namelist():
                raise ValueError('Empty rollout ZIP archive')
    return dict(result, iteration=ep['completed_iteration'] if 'completed_iteration' in ep else None,
                checkpoint=ep['checkpoint'], verified_complete=True, root=str(root))


def choose_origin(variant, extra=()):
    proposal = read(CONTROL/'proposal.json')['variants'][variant]
    candidates = [proposal['resume_from'], *extra]
    if variant == 'bonus':
        candidates.append(str(RUNTIME/'evaluations/arm-mixed-bonus-335682-r0'))
    statefile = CONTROL/f'{variant}-progress.json'
    if statefile.exists():
        candidates.extend(read(statefile).get('training_roots', []))
    found = []
    for name in dict.fromkeys(candidates):
        root = Path(name)
        marker = root/'runtime/latest_checkpointed_iteration.txt'
        if not marker.exists():
            continue
        idx = int(marker.read_text())
        # A new output may contain only a reference to its parent checkpoint.
        if not (root/f'iterations/{idx:04d}/checkpoint-validation.json').exists():
            continue
        o = origin(root)
        if o['manifest']['wandb_run_id'] != RUN_IDS[variant]:
            raise ValueError('Resume candidate belongs to another run')
        found.append((idx, len(found), root))
    if not found:
        raise ValueError('No verified inactive resume checkpoint')
    return max(found)[2]


def update_supervisor(job, variant, controller, training_root, **extra):
    # Both variants can transition concurrently. Registry updates must not lose
    # the other variant's replacement ID or pending evaluation.
    ledger = read(CONTROL/f'{variant}-attempts.json')['attempts']
    index = next(i for i, row in enumerate(ledger) if row['job_id'] == job)
    prior_used = sum(row.get('elapsed_seconds', 0) for row in ledger[:index])
    with (SUPERVISOR/'registry-update.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        registry = read(SUPERVISOR/'registry.json')
        item = next(j for j in registry['jobs'] if j['key'] == 'mixed-'+variant)
        if item['job_id'] != job:
            item.setdefault('supervised_attempts', []).append(dict(job_id=item['job_id'],
                controller_root=item['controller_root'], training_root=item['training_root']))
        item.update(job_id=job, controller_root=str(controller), training_root=str(training_root),
            requires_user=False, verified_complete=False, target_iteration=TARGET,
            budget_scope='separate48h extension', budget_receipt=str(CONTROL/f'{variant}-attempts.json'),
            approved_total_seconds=172800, attempt_seconds=attempt(job, variant)['attempt_seconds'],
            prior_used_seconds=prior_used,
            requested_endpoint=TARGET, additional_budget_approved=True)
        item.pop('approval_blocker', None)
        item.update(extra)
        atomic(SUPERVISOR/'registry.json', registry)


def worker(job, variant, stage, manifest):
    if os.getenv('SLURM_JOB_ID') != job:
        raise ValueError('Wrong worker allocation')
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
    subprocess.run(source_command(SOURCE, [str(PYTHON), '-B', '-c', probe,
        '--training-root', p['resume_from'], '--job-id', job, '--gpus', '8',
        '--output', p['gpu_restore_output'], '--continuation-plan', str(manifest), '--execute']), check=True)
    validate_resume_plan(p, require_gpu_restore=True)
    spec = importlib.util.spec_from_file_location('mixed_to60_training', CONTROL/'training-controller.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.execute(p)


def execute(job, variant):
    lineage_lock = (CONTROL/f'{variant}-execution.lock').open('a')
    fcntl.flock(lineage_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    refresh_accounting(variant)
    a = attempt(job, variant)
    if os.getenv('SLURM_JOB_ID') != job or int(os.getenv('SLURM_RESTART_COUNT', '0')) != 0:
        raise ValueError('Unregistered allocation/restart')
    def capacity():
        return allocation(subprocess.check_output(['scontrol', 'show', 'job', job, '-o'], text=True),
                          job, requested_gpus=8, maximum_hours=a['attempt_seconds']/3600)
    capacity()
    root = CONTROL/f'controller-{variant}-{job}'
    root.mkdir(exist_ok=False)
    statefile = CONTROL/f'{variant}-progress.json'
    state = read(statefile) if statefile.exists() else dict(training_roots=[], evaluations={})
    resume = choose_origin(variant)
    step = ['srun', f'--jobid={job}', '--nodes=1', '--ntasks=1', '--cpus-per-task=64',
        '--gres=gpu:h200:8', '--exact', '--cpu-bind=none', str(PYTHON), '-B', str(Path(__file__).resolve()),
        '--job-id', job, '--variant', variant]
    update_supervisor(job, variant, root, resume)
    try:
        while True:
            o = origin(resume)
            completed = o['checkpoint_report']['iteration']+1
            stage, target = next_stage(completed, {int(i) for i in state['evaluations']})
            if stage == 'complete':
                # Revalidate all six complete cohorts before closing this run.
                for item in state['evaluations'].values():
                    audit_evaluation(read(item['plan']))
                atomic(root/'status.json', dict(complete=True, stage='complete', target=TARGET,
                    completed_iteration=completed, evaluations_completed=MILESTONES))
                update_supervisor(job, variant, root, resume, verified_complete=True,
                    artifact_checks='durable60 and six full300 rollout/verdict cohorts verified')
                return
            remaining = capacity()['maximum_seconds']
            if remaining < (2100 if stage == 'eval' else 6600):
                atomic(root/'status.json', dict(complete=False, stage='budget-stop',
                    durable_iteration=completed, next_stage=stage, next_target=target, remaining_seconds=remaining))
                return
            if stage == 'train':
                p = training_plan(job, variant, resume, target, a['attempt_seconds'])
                validate_plan(p)
            else:
                training = next((Path(x) for x in reversed(state['training_roots'])
                    if (Path(x)/f'iterations/{target-1:04d}/checkpoint-validation.json').exists()), resume)
                p = evaluation_plan(job, variant, training, target)
            planfile = root/f'{stage}-{target}-plan.json'
            atomic(planfile, p)
            atomic(root/'controller-plan.json', dict(job_id=job, target=TARGET, iteration=target,
                training_root=p['output'] if stage == 'train' else str(resume), current_stage=stage))
            atomic(root/'status.json', dict(complete=False, stage='training' if stage == 'train' else 'evaluation',
                iteration=target, durable_iteration=completed, completed_stages=sorted(map(int, state['evaluations']))))
            evaluations = [dict(iteration=int(k), root=v['root'], verified=True)
                           for k, v in state['evaluations'].items()]
            if stage == 'eval':
                evaluations.append(dict(iteration=target, root=p['output'], verified=False))
            update_supervisor(job, variant, root, p['output'] if stage == 'train' else resume,
                              evaluations=evaluations)
            subprocess.run([*step, '--worker', stage, '--manifest', str(planfile)], check=True)
            if stage == 'train':
                state['training_roots'].append(p['output'])
                atomic(statefile, state)
                resume = choose_origin(variant)
                actual = origin(resume)['checkpoint_report']['iteration']+1
                if actual < target:
                    for gatefile in (Path(p['output'])/'iterations').glob('*/training_gate.json'):
                        if not read(gatefile).get('calibration', {}).get('passed'):
                            raise ValueError('Calibration failed; repair required before continuation')
                    atomic(root/'status.json', dict(complete=False, stage='budget-stop',
                        durable_iteration=actual, target=target, training_status=read(Path(p['output'])/'status.json')))
                    return
            else:
                audit = audit_evaluation(p)
                audit.update(iteration=target, plan=str(planfile))
                atomic(root/f'evaluation{target}-audit.json', audit)
                state['evaluations'][str(target)] = audit
                atomic(statefile, state)
    except BaseException as exc:
        atomic(root/'status.json', dict(complete=False, stage='failed', error_type=type(exc).__name__, error=str(exc)[:600]))
        raise


def native_check(p, destination):
    from resume_arm_failure_variants import check_multimodal_storage
    destination.mkdir(parents=True, exist_ok=True)
    atomic(destination/'storage-preflight.json', check_multimodal_storage(p))
    atomic(destination/'browser.json', p['browser_config'])
    env = dict(clean_environment(), **p['environment'])
    env.update(DRY_RUN='1', CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               BROWSER_TRAIN_CONFIG=str(destination/'browser.json'))
    argv = shlex.split(subprocess.check_output(p['command'], env=env, text=True))
    env.pop('DRY_RUN')
    expected = dict(actor_num_gpus_per_node=8, tensor_model_parallel_size=2, global_batch_size=256,
        micro_batch_size=1, ppo_epochs=2, rollout_batch_size=48, browser_rollout_concurrency=64,
        judge_api_model='gpt-4.1', judge_prompt_variant='action_history', use_rollout_logprobs=True,
        dynamic_sampling_filter_path=previous.original.FILTER, lr=1e-6,
        num_rollout=p['target_completed_iterations'], use_checkpoint_opt_param_scheduler=True)
    code = ("from unittest.mock import patch\nfrom slime.utils.arguments import parse_args\n"
        "from ray._private.utils import validate_socket_filepath\nimport os\n"
        "validate_socket_filepath(os.environ['RAY_TMPDIR']+'/ray/session_2099-12-31_23-59-59_999999_9999999/sockets/plasma_store')\n"
        "with patch('megatron.training.arguments.get_device_arch_version',return_value=9): a=parse_args()\n"
        f"expected={expected!r}\n"
        "assert all(getattr(a,k)==v for k,v in expected.items()),{k:getattr(a,k) for k in expected}\n"
        "print('MIXED_TO60_NATIVE_PARSE_OK')\n")
    with (destination/'native-parse.log').open('w') as log:
        subprocess.run(source_command(SOURCE, [str(PYTHON), '-B', '-c', code, *argv[2:]]),
            cwd=SOURCE, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)


def prepare():
    approval()
    validate_source(SOURCE)
    for name in ('training-controller.py', 'restore-controller.py'):
        text = (previous.CONTROL/name).read_text().replace('resume_arm_mixed_reweight', 'resume_arm_mixed_to60')
        compile(text, name, 'exec')
        (CONTROL/name).write_text(text)
    for variant in RUN_IDS:
        o = choose_origin(variant)
        p = training_plan('PREPARE-'+variant, variant, o, 10)
        native_check(p, CONTROL/('preflight-'+variant))
        atomic(CONTROL/f'{variant}-initial-plan.json', p)
        # Check real saved checkpoint with the exact future evaluation protocol.
        ep = evaluation_plan('PREPARE-'+variant, variant, o, origin(o)['checkpoint_report']['iteration']+1)
        if ep['expected_task_count'] != 300 or not ep['require_task_rollouts']:
            raise ValueError('Full300 rollout/verdict persistence missing')
        atomic(CONTROL/f'{variant}-evaluation-preview.json', ep)
    subprocess.run(['bash', '-n', str(REPO/'scripts/resume_arm_mixed_to60_8gpu.sbatch')], check=True)
    atomic(CONTROL/'readiness.json', dict(cpu_passed=True, native_parse_passed=True,
        fingerprints=fingerprints(), native_gpu_restore_required_per_stage=True))
    for variant in RUN_IDS:
        validate_plan(read(CONTROL/f'{variant}-initial-plan.json'))
    print(CONTROL/'readiness.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--job-id')
    parser.add_argument('--variant', choices=RUN_IDS)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--worker', choices=['train', 'eval'])
    parser.add_argument('--manifest', type=Path)
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.worker:
        worker(args.job_id, args.variant, args.worker, args.manifest)
    elif args.execute:
        execute(args.job_id, args.variant)
    else:
        parser.error('Choose --prepare, --execute or --worker')
