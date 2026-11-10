#!/usr/bin/env python3
"""Own approved first-pass local WebVoyager/DeepShop workers; never submit jobs.

Each method has one 43,200-second budget across every registered attempt.
Later repeats are deliberately unsupported. Completed artifacts require a
separate independent review before they can be skipped on a replacement job.
"""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

import prepare_arm_localbench90 as prepare
import run_arm_stealth90_o4 as shared

CONTROL, REPO, RUNTIME = prepare.CONTROL, prepare.REPO, prepare.RUNTIME
SUPERVISOR = shared.base.SUPERVISOR
METHODS = ('baseline', 'additive', 'gate-b', 'mixed-bonus', 'mixed-reweight', 'original')
RESOURCES = dict(methods=list(METHODS), jobs=6, gpus_per_job=1, gpu_type='H200',
                 hours_per_method=12, cpus_per_job=8, memory_gib_per_job=240,
                 total_gpu_hours=72)
CAP_SECONDS = 43200
TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY',
            'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE'}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp-{os.getpid()}')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temporary, path)


@contextmanager
def locked(path, nonblocking=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0))
        yield


def first_repeat(method, repeat):
    if method not in METHODS or repeat != 1:
        raise ValueError('Only the six methods and repeat1 are executable; repeat2/3 remain disabled')


def checkpoint_binding(method, spec):
    cp = spec.get('selected_checkpoint90')
    if cp is None:
        return dict(training_wandb_run_id=spec['training_wandb_run_id'],
                    requires_independent_checkpoint90=True)
    return dict(checkpoint=cp['path'], native_zero_based_iteration=89,
                training_wandb_run_id=spec['training_wandb_run_id'],
                checkpoint_validation_sha256=cp['validation_report']['sha256'],
                adam_updates=cp['adam_updates'], scheduler_num_steps=cp['scheduler_num_steps'],
                scheduler_minus_optimizer_updates=cp['scheduler_minus_optimizer_updates'])


def approval_template():
    matrix = read(CONTROL / 'evaluation-matrix.json')
    return dict(approved=False, repeat=1, resources=RESOURCES,
                source_manifest_sha256=sha(prepare.SOURCE / 'reference_manifest.json'),
                checkpoint_bindings={m: checkpoint_binding(m, matrix['methods'][m]) for m in METHODS},
                approval_scope='Only first-pass local595 then local150 per method; excludes training and stealth budgets')


def approval(method, repeat=1):
    first_repeat(method, repeat)
    data = read(CONTROL / 'local-suite-approval.json')
    if (data.get('approved') is not True or data.get('repeat') != 1
            or data.get('resources') != RESOURCES
            or data.get('source_manifest_sha256') != sha(prepare.SOURCE / 'reference_manifest.json')):
        raise PermissionError('Exact first-pass local-suite resources and frozen source approval required')
    if set(data.get('checkpoint_bindings', {})) != set(METHODS):
        raise PermissionError('Approval must bind all six method lineages')
    if method in data.get('disabled_methods', []):
        raise PermissionError('Method cancelled by user scope correction: '+method)
    if method in data.get('launch_paused_pending_scope_confirmation', []):
        raise PermissionError('Method awaits user scope clarification: '+method)
    return data


def selected_checkpoint(method, approved):
    spec = read(CONTROL / 'evaluation-matrix.json')['methods'][method]
    cp = spec.get('selected_checkpoint90')
    if not spec.get('checkpoint90_ready') or not cp or cp.get('native_zero_based_iteration') != 89:
        raise ValueError('Verified checkpoint90 is unavailable; never substitute iteration80 or stale85')
    if Path(cp['path']).name != 'iter_0000089':
        raise ValueError('Native checkpoint must be iter_0000089')
    report_path = Path(cp['validation_report']['path'])
    if sha(report_path) != cp['validation_report']['sha256']:
        raise ValueError('Checkpoint validation receipt changed')
    report = read(report_path)
    if (report['iteration'] != 89 or Path(report['checkpoint']).resolve() != Path(cp['path']).resolve()
            or report['completed_optimizer_updates'] != cp['adam_updates']
            or report['scheduler_num_steps'] != cp['scheduler_num_steps']
            or report['scheduler_minus_optimizer_updates'] != cp['scheduler_minus_optimizer_updates']):
        raise ValueError('Checkpoint identity/counters differ from matrix')
    for name, size in report['shard_files'].items():
        if (Path(cp['path']) / name).stat().st_size != size:
            raise ValueError('Checkpoint shard changed')
    for component in cp['small_component_evidence']:
        if sha(component['path']) != component['sha256']:
            raise ValueError('Checkpoint metadata/common/cursor changed')
    binding = approved['checkpoint_bindings'][method]
    expected = checkpoint_binding(method, spec)
    if method == 'original':
        if binding.get('training_wandb_run_id') != spec['training_wandb_run_id']:
            raise PermissionError('Original checkpoint changed training lineage')
        pointer = spec.get('checkpoint90_independent_review', {})
        if not pointer.get('path') or sha(pointer['path']) != pointer.get('sha256'):
            raise ValueError('Original90 requires a bound independent checkpoint review')
        review = read(pointer['path'])
        wanted = dict(verified_complete=True, completed_training_iteration=90,
                      checkpoint=cp['path'], training_wandb_run_id=spec['training_wandb_run_id'],
                      checkpoint_validation_sha256=cp['validation_report']['sha256'])
        if any(review.get(k) != v for k, v in wanted.items()):
            raise ValueError('Original90 independent review does not match checkpoint and lineage')
        if not binding.get('requires_independent_checkpoint90') and binding != expected:
            raise PermissionError('Original checkpoint differs from approval')
    elif binding != expected:
        raise PermissionError('Checkpoint differs from approved method binding')
    return spec, cp


def plans(method, job, approved):
    first_repeat(method, 1)
    if not re.fullmatch(r'[0-9]+', str(job)):
        raise ValueError('Registered numeric Slurm job ID required')
    spec, cp = selected_checkpoint(method, approved)
    previews = prepare.preview(method, cp['path'], cp['adam_updates'],
                               cp['scheduler_minus_optimizer_updates'], spec['training_wandb_run_id'])
    result = json.loads(json.dumps(previews).replace('PREPARE', str(job)))
    for p in result:
        # Ray adds a long session/socket suffix; Linux AF_UNIX allows107 bytes.
        p['environment']['RAY_TMPDIR'] = f'/tmp/lb-{job}-{p["benchmark"][0]}'
        p['source_manifest_sha256'] = approved['source_manifest_sha256']
        p['checkpoint_validation_sha256'] = cp['validation_report']['sha256']
    if [p['benchmark'] for p in result] != ['webvoyager', 'deepshop']:
        raise ValueError('Expected awaited WebVoyager then DeepShop stages')
    return result


def scientific_identity(p):
    fields = ('method', 'benchmark', 'repeat', 'checkpoint', 'source', 'source_manifest_sha256',
              'checkpoint_validation_sha256', 'parent_training_run', 'completed_optimizer_updates',
              'expected_task_count', 'expected_rollout_task_ids', 'judge_model', 'judge_prompt_variant',
              'protocol', 'browser_env')
    return hashlib.sha256(json.dumps({k: p[k] for k in fields}, sort_keys=True).encode()).hexdigest()


def ledger():
    path = CONTROL / 'local-suite-attempts.json'
    data = read(path) if path.exists() else dict(approved_seconds_per_method=CAP_SECONDS, methods={})
    if data.get('approved_seconds_per_method') != CAP_SECONDS:
        raise ValueError('Method budget cap changed')
    ids = [str(a['job_id']) for v in data['methods'].values() for a in v['attempts']]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate allocation ID in local-suite ledger')
    return data


def refresh_attempts(attempts, current_job=None):
    for attempt in attempts:
        job = str(attempt['job_id'])
        raw = subprocess.check_output(['sacct', '-X', '-n', '-P', '-j', job,
                                       '--format=JobIDRaw,State,ElapsedRaw,ExitCode'], text=True)
        matches = [line.split('|') for line in raw.splitlines() if line.split('|')[0] == job]
        if len(matches) != 1 or len(matches[0]) < 4:
            raise ValueError('Accounting unavailable; budget cannot be reset')
        _, state, elapsed, exit_code, *_ = matches[0]
        if int(elapsed) < 0:
            raise ValueError('Negative scheduler time')
        attempt.update(charged_seconds=max(int(elapsed), attempt.get('charged_seconds', 0)),
                       scheduler_state=state.split()[0], exit_code=exit_code)
        if job != current_job and state.split()[0] not in TERMINAL:
            raise ValueError('Another live or queued attempt exists for this method')


def exact_resources(info, job, maximum_seconds, running=True):
    fields = dict(re.findall(r'(\w+)=(\S+)', info))
    if (fields.get('JobId') != job or not fields.get('UserId', '').endswith(f'({os.getuid()})')
            or fields.get('JobState') not in ({'RUNNING'} if running else {'PENDING', 'RUNNING'})
            or fields.get('NumNodes') not in ({'1'} if running else {'1', '1-1'})
            or fields.get('NumCPUs') != '8'
            or fields.get('Restarts') != '0' or fields.get('Requeue') != '0'):
        raise ValueError('Wrong job ownership/state, resources or automatic requeue')
    for key in (('ReqTRES', 'AllocTRES') if running else ('ReqTRES',)):
        tres = dict(part.split('=', 1) for part in fields.get(key, '').split(',') if '=' in part)
        match = re.fullmatch(r'([\d.]+)([KMGT])', tres.get('mem', ''))
        gib = float(match[1]) * {'K': 2**-20, 'M': 2**-10, 'G': 1, 'T': 1024}[match[2]] if match else 0
        if (tres.get('cpu') != '8' or tres.get('node') != '1' or tres.get('gres/gpu') != '1'
                or tres.get('gres/gpu:h200') != '1' or gib != 240):
            raise ValueError('Exact 1H200/8CPU/240GiB required in ' + key)
    limit = re.fullmatch(r'(?:(\d+)-)?(\d+):(\d{2}):(\d{2})', fields.get('TimeLimit', ''))
    if not limit or int(limit[3]) >= 60 or int(limit[4]) >= 60:
        raise ValueError('Unverifiable scheduler TimeLimit')
    seconds = int(limit[1] or 0)*86400 + int(limit[2])*3600 + int(limit[3])*60 + int(limit[4])
    if not 0 < seconds <= maximum_seconds <= CAP_SECONDS:
        raise ValueError('Allocation time exceeds registered attempt or method budget')
    return seconds


def register_attempt(method, job, maximum_seconds, repeat=1):
    approval(method, repeat)
    if not re.fullmatch(r'[0-9]+', str(job)):
        raise ValueError('Numeric Slurm ID required')
    with locked(CONTROL / 'local-suite-budget.lock'):
        data = ledger()
        if any(str(a['job_id']) == job for v in data['methods'].values() for a in v['attempts']):
            raise ValueError('Attempt already registered; do not reset its charge')
        attempts = data['methods'].setdefault(method, dict(attempts=[]))['attempts']
        refresh_attempts(attempts)
        if maximum_seconds <= 0 or sum(a['charged_seconds'] for a in attempts) + maximum_seconds > CAP_SECONDS:
            raise ValueError('Retry allocation exceeds remaining 43,200-second method cap')
        info = subprocess.check_output(['scontrol', 'show', 'job', job, '-o'], text=True)
        exact_resources(info, job, maximum_seconds, running=False)
        attempts.append(dict(job_id=job, maximum_seconds=maximum_seconds, charged_seconds=0,
                             registered_epoch=time.time()))
        atomic(CONTROL / 'local-suite-attempts.json', data)
    return dict(method=method, job_id=job, maximum_seconds=maximum_seconds)


def accounting(method, job, repeat=1):
    approved = approval(method, repeat)
    if os.environ.get('SLURM_JOB_ID') != job:
        raise PermissionError('Execute inside the registered allocation')
    with locked(CONTROL / 'local-suite-budget.lock'):
        data = ledger()
        attempts = data['methods'].get(method, {}).get('attempts', [])
        current = next((a for a in attempts if str(a['job_id']) == job), None)
        if current is None:
            raise PermissionError('Allocation ID not registered to this method')
        refresh_attempts(attempts, current_job=job)
        atomic(CONTROL / 'local-suite-attempts.json', data)
        prior = sum(a['charged_seconds'] for a in attempts if a is not current)
        maximum = int(current['maximum_seconds'])
        if prior + maximum > CAP_SECONDS or current.get('scheduler_state') != 'RUNNING':
            raise ValueError('Method budget exceeded or running accounting unavailable')
        info = subprocess.check_output(['scontrol', 'show', 'job', job, '-o'], text=True)
        limit = exact_resources(info, job, maximum)
        resources = shared.allocation(info, job, requested_gpus=1, maximum_hours=12)
        consumed = prior + current['charged_seconds']
        remaining = min(resources['maximum_seconds'], maximum-current['charged_seconds']-180,
                        CAP_SECONDS-consumed-180)
        result = dict(approved_seconds=CAP_SECONDS, consumed_seconds=consumed,
                      remaining_seconds=max(0, int(remaining)), attempts=attempts,
                      scheduler_time_limit_seconds=limit, checked_epoch=time.time())
        atomic(CONTROL / f'local-{method}-budget.json', result)
    if remaining < 1200:
        raise RuntimeError('Fewer than20 usable minutes remain within approved method budget')
    return approved, result


def registered_jobs(method):
    return [str(a['job_id']) for a in ledger()['methods'].get(method, {}).get('attempts', [])]


def verified_stage(p):
    """Only a hash-bound independent review permits reuse across attempts."""
    path = CONTROL / 'stage-reviews' / f'{p["method"]}-{p["benchmark"]}-r1.json'
    if not path.exists():
        return None
    review = read(path)
    if (review.get('verified_complete') is not True or review.get('independent_review') is not True
            or review.get('scientific_plan_sha256') != scientific_identity(p)
            or review.get('job_id') not in registered_jobs(p['method'])):
        raise ValueError('Stage review is not an independent matching registered result')
    root = Path(review['root'])
    expected = RUNTIME / f'evaluations/arm90-local-{p["method"]}-{p["benchmark"]}-r1-{review["job_id"]}'
    if root.resolve() != expected.resolve() or not review.get('artifact_sha256'):
        raise ValueError('Verified stage root differs from the isolated registered output')
    required = {'evaluation_manifest.json', 'metrics.json', 'audit.json',
                'checkpoint_restore_evidence.json', 'status.json'}
    if not required <= set(review['artifact_sha256']):
        raise ValueError('Review does not bind all required completion evidence')
    for name in required:
        if sha(root/name) != review['artifact_sha256'][name]:
            raise ValueError('Verified stage evidence changed')
    manifest, status = read(root/'evaluation_manifest.json'), read(root/'status.json')
    if (scientific_identity(manifest) != scientific_identity(p)
            or status.get('complete') is not True or status.get('returncode') != 0):
        raise ValueError('Verified stage manifest/status mismatch')
    return dict(root=str(root), review_path=str(path), review_sha256=sha(path), verified=True)


def reject_unreviewed_outputs(p):
    for job in registered_jobs(p['method']):
        root = RUNTIME / f'evaluations/arm90-local-{p["method"]}-{p["benchmark"]}-r1-{job}'
        if root.exists():
            proof_path = CONTROL/'precollection-reviews'/f'{p["method"]}-{p["benchmark"]}-{job}.json'
            if proof_path.exists():
                proof = read(proof_path)
                required = {'evaluation_manifest.json', 'evaluation.log', 'checkpoint_validation.json'}
                if (proof.get('verified_precollection_abort') is not True or proof.get('job_id') != job
                        or proof.get('root') != str(root) or proof.get('scheduler_state') not in TERMINAL
                        or not required <= set(proof.get('artifact_sha256', {}))):
                    raise ValueError('Precollection abort review does not match preserved attempt')
                for name in required:
                    if sha(root/name) != proof['artifact_sha256'][name]:
                        raise ValueError('Precollection evidence changed')
                if list((root/'rollouts').glob('*')) or proof.get('rollout_files') != 0:
                    raise ValueError('Precollection attempt has task evidence')
                restore = root/'checkpoint_restore_evidence.json'
                if restore.exists():
                    if (proof.get('native_restore_completed') is not True
                            or proof.get('evaluation_started') is not False
                            or proof.get('no_task_collection_verified') is not True
                            or proof['artifact_sha256'].get(restore.name) != sha(restore)):
                        raise ValueError('Model-restoration evidence requires explicit precollection review')
                continue
            raise ValueError('Existing unreviewed or partial output requires targeted recovery; preserve it')


def register_supervisor(method, job, root, stage_plans, budget):
    with locked(SUPERVISOR / 'registry-update.lock'):
        path = SUPERVISOR / 'registry.json'
        registry = read(path)
        key = f'arm90-local-suite-{method}-r1'
        item = next((x for x in registry['jobs'] if x['key'] == key), None)
        if item is None:
            item = dict(key=key, agent_reviewed_evaluation_roots=[])
            registry['jobs'].append(item)
        elif item.get('job_id') != job:
            item.setdefault('supervised_attempts', []).append(dict(job_id=item.get('job_id'),
                                                        controller_root=item.get('controller_root')))
        item.update(job_id=job, controller_root=str(root), training_root=str(root),
                    evaluations=[dict(iteration=90, label=p['benchmark'], root=p['output'], verified=False)
                                 for p in stage_plans], approved_total_seconds=CAP_SECONDS,
                    budget_receipt=str(CONTROL/'local-suite-approval.json'), last_budget_check=budget,
                    verified_complete=False, requires_user=False, require_agent_completion_review=True,
                    agent_completion_reviewed=False,
                    requested_endpoint='First local595 WebVoyager and local150 DeepShop; independent reviews required')
        atomic(path, registry)


def audit_stage(p):
    root = Path(p['output'])
    restore_path = root/'checkpoint_restore_evidence.json'
    if not restore_path.exists():
        # The native actor can forward its exact restoration line to stdout
        # without leaving a discoverable worker-log FD for the polling helper.
        log = root/'evaluation.log'
        pattern = re.compile(prepare.evaluator.restore_pattern(p))
        line = None
        if log.is_file():
            with log.open(errors='replace') as handle:
                line = next((s.rstrip('\n') for s in handle if pattern.search(s)), None)
        if line is None:
            raise ValueError('Fresh native89 GPU restoration evidence is missing')
        atomic(restore_path, dict(checkpoint=p['checkpoint'], job_id=p['job_id'],
               native_zero_based_iteration=89, restore_line=line, source='evaluation.log',
               evaluation_log_sha256=sha(log), forwarded_stdout_fallback=True))
    restore = read(restore_path)
    if (restore.get('checkpoint') != p['checkpoint'] or restore.get('job_id') != p['job_id']
            or not re.search(r'at iteration 89\b', restore.get('restore_line', ''))):
        raise ValueError('Fresh native89 GPU restoration evidence is missing or mismatched')
    result = shared.audit(p, expected_judge=('gpt-4o', p['benchmark']))
    result.update(artifacts_ready=True, verified_complete=False, independent_review_pending=True,
                  scientific_plan_sha256=scientific_identity(p),
                  archive_check_scope='Existing shared metadata/presence audit; independent full-archive review remains required')
    atomic(root/'audit.json', result)
    return result


def execute(method, job, repeat=1):
    first_repeat(method, repeat)
    with locked(CONTROL/f'local-{method}-active.lock', nonblocking=True):
        approved, budget = accounting(method, job, repeat)
        stage_plans = plans(method, job, approved)
        root = CONTROL/f'controller-local-{method}-{job}'
        root.mkdir(exist_ok=False)
        state = dict(complete=False, verified_complete=False, independent_review_pending=True,
                     method=method, job_id=job, repeat=1, stage='starting', stages=[])
        register_supervisor(method, job, root, stage_plans, budget)
        atomic(root/'status.json', state)
        try:
            for p in stage_plans:
                verified = verified_stage(p)
                if verified:
                    state['stages'].append(dict(benchmark=p['benchmark'], reused=verified))
                    continue
                reject_unreviewed_outputs(p)
                _, budget = accounting(method, job)
                state.update(stage=p['benchmark'], current_output=p['output'])
                manifest = root/f'{p["benchmark"]}-plan.json'
                atomic(manifest, p)
                atomic(root/'status.json', state)
                subprocess.run(['timeout', '--signal=INT', '--kill-after=120',
                    str(budget['remaining_seconds']-120), 'srun', f'--jobid={job}', '--nodes=1',
                    '--ntasks=1', '--cpus-per-task=8', '--gres=gpu:h200:1', '--exact', '--cpu-bind=none',
                    str(RUNTIME/'venv/bin/python'), '-B', str(Path(__file__).resolve()),
                    '--worker', '--method', method, '--job-id', job, '--repeat', '1',
                    '--manifest', str(manifest)], check=True)
                result = audit_stage(p)
                state['stages'].append(dict(benchmark=p['benchmark'], output=p['output'], audit=result))
                atomic(root/'status.json', state)
            state.update(stage='artifacts_ready', artifacts_ready=True, complete=False)
        except BaseException as exc:
            state.update(stage='failed', error=type(exc).__name__, detail=str(exc)[:600])
            raise
        finally:
            # Bookkeeping must not hide an earlier worker failure or discard artifacts.
            try:
                _, latest = accounting(method, job)
                state['last_budget_check'] = latest
            except Exception as exc:
                state['accounting_error'] = dict(type=type(exc).__name__, detail=str(exc)[:400])
            atomic(root/'status.json', state)


def worker(method, job, manifest, repeat=1):
    first_repeat(method, repeat)
    approved, _ = accounting(method, job, repeat)
    p = read(manifest)
    expected = next((x for x in plans(method, job, approved) if x['benchmark'] == p.get('benchmark')), None)
    if p != expected:
        raise ValueError('Worker manifest differs from approved frozen stage preview')
    if Path(p['output']).exists():
        raise ValueError('Never overwrite an existing worker output')
    os.environ['OPENWEBRL_EXPECTED_SCHEDULER_OFFSET'] = p['environment']['OPENWEBRL_EXPECTED_SCHEDULER_OFFSET']
    os.environ['OPENWEBRL_RECORD_EVAL_EXPECTED_TASK_COUNT'] = str(p['expected_task_count'])
    prepare.evaluator.run(p, REPO/'.env')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--approval-template', action='store_true')
    mode.add_argument('--register', action='store_true')
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--worker', action='store_true')
    parser.add_argument('--method', choices=METHODS)
    parser.add_argument('--job-id')
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--maximum-seconds', type=int)
    parser.add_argument('--manifest', type=Path)
    args = parser.parse_args()
    if args.approval_template:
        print(json.dumps(approval_template(), indent=2))
    elif args.register:
        if not args.maximum_seconds:
            parser.error('--register requires --maximum-seconds')
        print(json.dumps(register_attempt(args.method, args.job_id, args.maximum_seconds, args.repeat)))
    elif args.execute:
        execute(args.method, args.job_id, args.repeat)
    elif args.worker:
        if not args.manifest:
            parser.error('--worker requires --manifest')
        worker(args.method, args.job_id, args.manifest, args.repeat)


if __name__ == '__main__':
    main()
