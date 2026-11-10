#!/usr/bin/env python3
"""Own the user-requested official SFT fifth arm inside the existing allocation."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
from openwebrl.luna_qwen_metrics import MODES, digest, episode_usage, summarize, write_json
from openwebrl.luna_qwen_eval import SFT_PROTOCOL
from resume_baseline import allocation, validate_source
from run_arm_quality_audit import stop

RUNTIME = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime')
ROOT = RUNTIME / 'luna-qwen-inference-20261004'
CONTROL = ROOT / 'sft-extension'
ACTOR = Path('/gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_extension(plan, approved, base):
    if not approved.get('approved') or approved['plan_sha256'] != digest(plan):
        raise ValueError('The requested fifth arm must have a frozen authorized plan')
    if plan['protocol'] != SFT_PROTOCOL or plan['actor_path'] != str(ACTOR):
        raise ValueError('Official SFT checkpoint/protocol changed')
    if plan['schedule'] != [dict(task_id=i['task_id'], modes=['sft_luna5']) for i in base['schedule']]:
        raise ValueError('Fifth arm must use exactly the original ten tasks')
    for key in ('luna_usd', 'luna_calls', 'judge_usd', 'judge_calls'):
        if plan['limits'][key] != base['limits'][key]:
            raise ValueError('Extension cannot increase the original shared API caps')
    if plan['resources'] != base['resources']:
        raise ValueError('Extension cannot increase the original allocation budget')
    for role in ('luna', 'judge'):
        if (CONTROL / f'pilot/{role}-budget').resolve() != (ROOT / f'pilot/{role}-budget').resolve():
            raise ValueError('Fifth arm must share the original API ledgers')


def worker(config):
    import prepare_luna_qwen_inference as common
    if config.get('stage') == 'context_repair':
        base = read(ROOT / 'plan.json')
        common.CONTROL, common.SOURCE, common.ACTOR = ROOT, Path(base['source']), Path(base['actor_path'])
    else:
        common.CONTROL, common.SOURCE, common.ACTOR = CONTROL, SOURCE, ACTOR
    common.worker(config)


def boundary_failure(record):
    """Select only failed near-context HTTP requests, never reward-based retries."""
    if record['valid'] or record['mode'] == 'luna':
        return False
    requests = [read(p) for p in Path(record['artifact']).parent.glob('requests/*.json')]
    return any(r.get('provider') == 'qwen' and r.get('error_type') == 'HTTPStatusError'
        and 0 < r.get('sampling', {}).get('max_new_tokens', 4097) <= 4096
        for r in requests)


def repair_context(job):
    """Repair the proven engine boundary bug at the next owned stage handoff."""
    if os.environ.get('SLURM_JOB_ID') != job or f'/job_{job}/' not in Path('/proc/self/cgroup').read_text():
        raise ValueError('Repair must stay inside the original allocation')
    directory = ROOT / 'repairs/context-boundary-20261004'
    request = read(directory / 'repair.json')
    if not request['engine_rejects_equality_verified']:
        raise ValueError('Boundary failure must be independently diagnosed')
    base = read(ROOT / 'plan.json')
    validate_source(Path(base['source']))
    if sha(Path(base['source']) / 'openwebrl/luna_qwen_policy.py') != request['fixed_policy_sha256']:
        raise ValueError('Validated boundary fix changed')
    approval = read(ROOT / 'approval.json')
    attempt = next(a for a in approval['attempts'] if a['job_id'] == job)
    prior = sum(a['elapsed_seconds'] for a in approval['attempts'] if a['job_id'] != job)
    if prior + attempt['time_limit_seconds'] > approval['resources']['total_seconds']:
        raise ValueError('Repair cannot reset the original allocation budget')
    candidates = [p for p in (ROOT / 'pilot/records').glob('*.json') if boundary_failure(read(p))]
    archived = directory / 'superseded-records'
    archived.mkdir(exist_ok=True)
    pending = [p for p in archived.glob('*.json') if not (ROOT / 'pilot/records' / p.name).exists()]
    if not candidates and not pending:
        if list(archived.glob('*.json')):
            write_json(directory / 'current.json', dict(job_id=job, stage='repaired', complete=True,
                replaced_records=len(list(archived.glob('*.json'))), updated_unix=time.time(),
                recovery='Repaired during normal resumed collection'))
        return
    # Do not archive any committed result unless collection time remains.
    res = allocation(subprocess.check_output(['scontrol', 'show', 'job', job, '-o'], text=True), job,
        requested_gpus=4, maximum_hours=attempt['time_limit_seconds']/3600)
    if res['cpus'] != 32 or res['allocated_memory_gib'] != 480:
        raise ValueError('Repair CPU/RAM allocation changed')
    with (directory / 'owner.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for path in candidates:
            target = archived / path.name
            if target.exists():
                raise ValueError('Refuse to overwrite an archived attempt')
            path.rename(target)
        children = []
        deadline = time.monotonic() + res['maximum_seconds'] - 120
        try:
            # IDs4–7 preserve every original worker0–3 log and W&B identity.
            for index in range(4, 8):
                config = directory / f'worker-{job}-{index}.json'
                write_json(config, dict(job_id=job, worker_id=index, stage='context_repair'))
                children.append(subprocess.Popen(['srun', '--jobid=' + job, '--nodes=1', '--ntasks=1',
                    '--cpus-per-task=8', '--gpus=h200:1', '--gres=gpu:h200:1', '--mem=120G',
                    '--exact', '--cpu-bind=cores', sys.executable, str(Path(__file__).resolve()),
                    '--worker', str(config)], start_new_session=True))
            while any(child.poll() is None for child in children):
                if any(child.poll() not in (None, 0) for child in children):
                    raise RuntimeError('Context repair worker failed')
                if time.monotonic() >= deadline:
                    raise TimeoutError('Original allocation budget ending during repair')
                state = dict(job_id=job, stage='repairing_context_boundary', updated_unix=time.time(),
                    affected_records=len(list(archived.glob('*.json'))), remaining_seconds=deadline-time.monotonic())
                write_json(directory / 'current.json', state)
                write_json(ROOT / 'current.json', state)
                time.sleep(15)
            if any(child.returncode for child in children):
                raise RuntimeError('Context repair failed')
            for old in archived.glob('*.json'):
                replacement = ROOT / 'pilot/records' / old.name
                if not replacement.exists() or boundary_failure(read(replacement)):
                    raise ValueError('Missing verified replacement after context repair')
            write_json(directory / 'current.json', dict(job_id=job, stage='repaired', complete=True,
                replaced_records=len(list(archived.glob('*.json'))), updated_unix=time.time()))
        finally:
            for child in children:
                stop(child)


def combine():
    from report_luna_qwen_inference import plot, require_comparison_integrity
    require_comparison_integrity(ROOT)
    base, plan = read(ROOT / 'plan.json'), read(CONTROL / 'plan.json')
    records = []
    for directory, protocol in ((ROOT, base['protocol']), (CONTROL, plan['protocol'])):
        for path in (directory / 'pilot/records').glob('*.json'):
            row = read(path)
            if row['protocol_sha256'] != digest(protocol) or sha(row['artifact']) != row['artifact_sha256']:
                raise ValueError('Unverified record in combined comparison')
            usage = episode_usage(Path(row['artifact']).parent, row['mode'], row['episode_seconds'], base['pricing'])
            for key in ('model_input_tokens', 'model_output_tokens', 'model_usage_missing_requests'):
                if row[key] != usage[key]:
                    raise ValueError('Saved token totals disagree with receipts')
            records.append(dict(row, **usage))
    result = summarize(records, [i['task_id'] for i in base['schedule']], base['pricing'], modes=(*MODES, 'sft_luna5'))
    result.update(protocols=dict(original_four=base['protocol'], sft_luna5=plan['protocol']),
        pilot_not_full300=True,
        timing_caveat=plan.get('collection_order', 'Original four arms interleaved; requested SFT fifth arm collected afterward.'),
        api_ledgers={role: read(ROOT / f'pilot/{role}-budget/ledger.json') for role in ('luna', 'judge')})
    # Keep research attempts separate from the 50 committed benchmark episodes.
    physical = []
    for directory in (ROOT, CONTROL):
        for identity in (directory / 'pilot/attempts').glob('*/*/identity.json'):
            usage_path = identity.parent / 'physical-usage.json'
            if usage_path.exists():
                physical.append(read(usage_path))
            else:
                info = read(identity)
                physical.append(dict(episode_usage(identity.parent, info['mode'], 0, base['pricing']),
                    interrupted=True, missing_final_timing=True))
    result['physical_collection'] = dict(attempts=len(physical), committed=len(records),
        interrupted=sum(r['interrupted'] for r in physical),
        attempts_missing_final_timing=sum(r.get('missing_final_timing', False) for r in physical),
        model_input_tokens=sum(r['model_input_tokens'] for r in physical),
        model_output_tokens=sum(r['model_output_tokens'] for r in physical),
        unknown_usage_requests=sum(r['model_usage_missing_requests'] for r in physical))
    destination = ROOT / 'report-five-arm'
    write_json(destination / 'summary.json', result)
    import csv
    with (destination / 'metrics.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['mode', 'successes', 'valid', 'tasks', 'overall', 'cost_lower_usd', 'cost_upper_usd',
                         'median_seconds', 'p95_seconds', 'mean_input_tokens', 'mean_output_tokens',
                         'mean_browser_steps', 'mean_local_gpu_reserved_seconds'])
        for mode, row in result['by_mode'].items():
            metrics = row['metrics']
            writer.writerow([mode, row['successes'], row['valid'], row['episodes'], row['overall'],
                row['cost_lower_mean_usd'], row['cost_upper_mean_usd'], metrics['episode_seconds']['p50'],
                metrics['episode_seconds']['p95'], metrics['model_input_tokens']['mean'],
                metrics['model_output_tokens']['mean'], metrics['browser_steps']['mean'],
                metrics['local_gpu_reserved_seconds']['mean']])
    plot(result, destination)
    return result


def execute(job):
    if os.environ.get('SLURM_JOB_ID') != job or f'/job_{job}/' not in Path('/proc/self/cgroup').read_text():
        raise ValueError('Additional collection must stay inside the approved active allocation')
    base, plan = read(ROOT / 'plan.json'), read(CONTROL / 'plan.json')
    validate_extension(plan, read(CONTROL / 'approval.json'), base)
    approval = read(ROOT / 'approval.json')
    attempt = next(a for a in approval['attempts'] if a['job_id'] == job)
    used = 0
    for previous in approval['attempts']:
        if previous['job_id'] == job:
            continue
        rows = subprocess.check_output(['sacct', '-X', '-n', '-P', '-j', previous['job_id'],
            '--format=JobIDRaw,State,ElapsedRaw'], text=True).splitlines()
        row = next(r.split('|') for r in rows if r.split('|')[0] == previous['job_id'])
        if row[1].split()[0] not in ('COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT'):
            raise ValueError('A previous allocation remains active')
        used += int(row[2])
    if used + attempt['time_limit_seconds'] > approval['resources']['total_seconds']:
        raise ValueError('All attempts must remain within the original 90 minutes')
    validate_source(SOURCE)
    if plan['source_manifest_sha256'] != sha(SOURCE / 'reference_manifest.json'):
        raise ValueError('Frozen SFT source changed')
    for name, expected in plan['actor_files'].items():
        stat = (ACTOR / name).stat()
        if dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns) != expected:
            raise ValueError('Official SFT weights changed after verification')
    try:
        res = allocation(subprocess.check_output(['scontrol', 'show', 'job', job, '-o'], text=True), job,
            requested_gpus=4, maximum_hours=attempt['time_limit_seconds']/3600)
    except ValueError as exc:
        write_json(CONTROL / 'current.json', dict(job_id=job, stage='needs_remaining_budget',
            complete=False, reason=str(exc), updated_unix=time.time()))
        return
    if res['cpus'] != 32 or res['allocated_memory_gib'] != 480:
        raise ValueError('CPU/RAM allocation changed')
    if len(list((ROOT / 'pilot/records').glob('*.json'))) != 40:
        raise ValueError('Finish the original four arms before this collection stage')
    if len(list((CONTROL / 'pilot/records').glob('*.json'))) == 10:
        combine()
        return
    with (CONTROL / 'owner.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        children = []
        deadline = time.monotonic() + res['maximum_seconds'] - 120
        def terminate(signum, frame):
            raise SystemExit(128 + signum)
        signal.signal(signal.SIGTERM, terminate)
        try:
            for index in range(4):
                path = CONTROL / f'worker-{job}-{index}.json'
                write_json(path, dict(job_id=job, worker_id=index))
                command = ['srun', '--jobid=' + job, '--nodes=1', '--ntasks=1', '--cpus-per-task=8',
                    '--gpus=h200:1', '--gres=gpu:h200:1', '--mem=120G', '--exact', '--cpu-bind=cores',
                    sys.executable, str(Path(__file__).resolve()), '--worker', str(path)]
                children.append(subprocess.Popen(command, start_new_session=True))
            while any(child.poll() is None for child in children):
                if any(child.poll() not in (None, 0) for child in children):
                    raise RuntimeError('An SFT worker failed; preserve attempts for recovery')
                if time.monotonic() >= deadline:
                    raise TimeoutError('Original shared allocation budget ending')
                assignments = [CONTROL / 'workers' / job / str(w) / 'device-assignment.json' for w in range(4)]
                if all(p.exists() for p in assignments) and len({read(p)['gpu_uuid'] for p in assignments}) != 4:
                    raise ValueError('SFT workers must have four distinct GPUs')
                write_json(CONTROL / 'current.json', dict(job_id=job, stage='running',
                    updated_unix=time.time(), remaining_seconds=deadline-time.monotonic()))
                time.sleep(15)
            if any(child.returncode for child in children):
                raise RuntimeError('SFT collection failed')
            result = combine()
            write_json(CONTROL / 'current.json', dict(job_id=job, stage='completed',
                complete=result['complete'], episodes=10, total_episodes=50, updated_unix=time.time()))
        except BaseException as exc:
            write_json(CONTROL / 'current.json', dict(job_id=job, stage='failed_or_interrupted',
                complete=False, error_type=type(exc).__name__, updated_unix=time.time()))
            raise
        finally:
            for child in children:
                stop(child)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute')
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--combine', action='store_true')
    parser.add_argument('--repair-context')
    args = parser.parse_args()
    if args.worker:
        worker(read(args.worker))
    elif args.repair_context:
        repair_context(args.repair_context)
    elif args.execute:
        execute(args.execute)
    elif args.combine:
        combine()
    else:
        parser.error('Choose --worker, --execute or --combine')
