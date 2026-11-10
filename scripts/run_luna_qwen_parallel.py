#!/usr/bin/env python3
"""Own concurrent official-SFT and Qwen recovery within one approved job."""
import argparse
from contextlib import ExitStack
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
from openwebrl.luna_qwen_metrics import digest, write_json
from openwebrl.luna_qwen_eval import verified_record
from prepare_luna_qwen_inference import check_approval
from resume_baseline import allocation, validate_source
from run_arm_quality_audit import stop
from run_luna_qwen_sft_extension import ROOT, CONTROL, ACTOR, read, sha, validate_extension, combine, boundary_failure


def pending_blocks(root, schedule):
    return sum(any(not (root / 'pilot/records' / (digest([item['task_id'], mode])[:24] + '.json')).exists()
                   for mode in item['modes']) for item in schedule)


def choose_family(pending, active, initial=None):
    """Start both families; give newly available slots to SFT whenever useful."""
    order = (initial, 'qwen' if initial == 'sft' else 'sft') if initial else ('sft', 'qwen')
    return next((family for family in order if pending[family] > active[family]), None)


def execute(job):
    if os.environ.get('SLURM_JOB_ID') != job or f'/job_{job}/' not in Path('/proc/self/cgroup').read_text():
        raise ValueError('Parallel workers must stay inside the approved allocation')
    plans = {'qwen': read(ROOT / 'plan.json'), 'sft': read(CONTROL / 'plan.json')}
    base, sft = plans['qwen'], plans['sft']
    approval = read(ROOT / 'approval.json')
    used = 0
    for previous in approval['attempts']:
        if previous['job_id'] == job:
            continue
        output = subprocess.check_output(['sacct', '-X', '-n', '-P', '-j', previous['job_id'],
            '--format=JobIDRaw,State,ElapsedRaw'], text=True)
        row = next(r.split('|') for r in output.splitlines() if r.split('|')[0] == previous['job_id'])
        if row[1].split()[0] not in ('COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT'):
            raise ValueError('A previous allocation remains active')
        used += int(row[2])
    attempt = check_approval(approval, base, job, used)
    validate_extension(sft, read(CONTROL / 'approval.json'), base)
    if base.get('execution_strategy') != 'parallel_sft_priority':
        raise ValueError('Parallel execution must be recorded in the authorized plan')
    roots = {'qwen': ROOT, 'sft': CONTROL}
    for family, plan in plans.items():
        source = Path(plan['source'])
        validate_source(source)
        if sha(source / 'reference_manifest.json') != plan['source_manifest_sha256']:
            raise ValueError('Frozen source manifest changed')
        actor = Path(plan['actor_path'])
        for name, expected in plan['actor_files'].items():
            stat = (actor / name).stat()
            if dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns) != expected:
                raise ValueError('Actor changed after independent verification')
        for path in (roots[family] / 'pilot/records').glob('*.json'):
            verified_record(path, digest(plan['protocol']))
    res = allocation(subprocess.check_output(['scontrol', 'show', 'job', job, '-o'], text=True), job,
        requested_gpus=4, maximum_hours=attempt['time_limit_seconds']/3600)
    if res['cpus'] != 32 or res['allocated_memory_gib'] != 480:
        raise ValueError('CPU/RAM allocation changed')
    deadline = time.monotonic() + res['maximum_seconds'] - 120
    journal = ROOT / 'parallel-priority' / job
    journal.mkdir(parents=True, exist_ok=True)
    children, history = {}, []
    generations = [0] * 4
    def terminate(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, terminate)
    with ExitStack() as stack:
        for root in roots.values():
            lock = stack.enter_context((root / 'owner.lock').open('a+'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            while True:
                for slot, entry in list(children.items()):
                    code = entry['process'].poll()
                    if code is not None:
                        entry['receipt'].update(returncode=code, ended_epoch=time.time())
                        del children[slot]
                        if code:
                            raise RuntimeError(f"{entry['receipt']['family']} worker failed; preserve all attempts")
                if time.monotonic() >= deadline:
                    raise TimeoutError('Original shared allocation budget ending')
                pending = {family: pending_blocks(root, plans[family]['schedule']) for family, root in roots.items()}
                active = {family: sum(e['receipt']['family'] == family for e in children.values()) for family in roots}
                for slot in range(4):
                    if slot in children:
                        continue
                    initial = ('sft' if slot < 2 else 'qwen') if generations[slot] == 0 else None
                    family = choose_family(pending, active, initial)
                    if family is None:
                        continue
                    worker_id = slot + 4 * generations[slot]
                    generations[slot] += 1
                    config = journal / f'{family}-worker-{worker_id}.json'
                    write_json(config, dict(job_id=job, worker_id=worker_id,
                        stage='context_repair' if family == 'qwen' else 'sft'))
                    command = ['srun', '--jobid=' + job, '--nodes=1', '--ntasks=1', '--cpus-per-task=8',
                        '--gpus=h200:1', '--gres=gpu:h200:1', '--mem=120G', '--exact', '--cpu-bind=cores',
                        sys.executable, str(SOURCE / 'scripts/run_luna_qwen_sft_extension.py'), '--worker', str(config)]
                    receipt = dict(slot=slot, family=family, worker_id=worker_id, started_epoch=time.time(), command=command)
                    history.append(receipt)
                    children[slot] = dict(process=subprocess.Popen(command, start_new_session=True), receipt=receipt)
                    active[family] += 1
                write_json(journal / 'workers.json', history)
                if not children:
                    if any(pending.values()):
                        raise RuntimeError('Unfinished tasks but no worker owns them')
                    break
                assigned = []
                for entry in children.values():
                    receipt = entry['receipt']
                    path = roots[receipt['family']] / 'workers' / job / str(receipt['worker_id']) / 'device-assignment.json'
                    if path.exists():
                        assigned.append(read(path)['gpu_uuid'])
                if len(set(assigned)) != len(assigned):
                    raise ValueError('Concurrent workers must own distinct physical GPUs')
                counts = {family: len(list((root / 'pilot/records').glob('*.json'))) for family, root in roots.items()}
                write_json(ROOT / 'current.json', dict(job_id=job, stage='running_parallel',
                    updated_unix=time.time(), remaining_seconds=deadline-time.monotonic(),
                    active_workers=active, recorded_episodes=counts, total_target=50))
                write_json(CONTROL / 'current.json', dict(job_id=job,
                    stage='completed' if counts['sft'] == 10 else 'running_parallel',
                    complete=counts['sft'] == 10, episodes=counts['sft'], target_episodes=10,
                    active_workers=active['sft'], shared_budget=True, updated_unix=time.time()))
                time.sleep(10)
            # Collection is complete only after every worker has exited cleanly.
            original_report = Path(base['source']) / 'scripts/report_luna_qwen_inference_original_four.py'
            subprocess.run([sys.executable, str(original_report), '--root', str(ROOT),
                '--output', str(ROOT / 'report')], check=True)
            combined = combine()
            repair = ROOT / 'repairs/context-boundary-20261004'
            archived = list((repair / 'superseded-records').glob('*.json'))
            for old in archived:
                replacement = ROOT / 'pilot/records' / old.name
                if not replacement.exists() or boundary_failure(read(replacement)):
                    raise ValueError('Context recovery has no verified replacement')
            write_json(repair / 'current.json', dict(job_id=job, stage='repaired', complete=True,
                replaced_records=len(archived), updated_unix=time.time()))
            write_json(ROOT / 'pilot/status.json', dict(complete=True, episodes=40, tasks=10))
            write_json(CONTROL / 'pilot/status.json', dict(complete=True, episodes=10, tasks=10))
            write_json(CONTROL / 'current.json', dict(job_id=job, stage='completed', complete=True,
                episodes=10, updated_unix=time.time()))
            write_json(ROOT / 'current.json', dict(job_id=job, stage='completed', complete=combined['complete'],
                episodes=50, updated_unix=time.time(), final_scheduler_audit_required=True))
        except BaseException as exc:
            write_json(ROOT / 'current.json', dict(job_id=job, stage='failed_or_interrupted', complete=False,
                error_type=type(exc).__name__, updated_unix=time.time(), execution_strategy='parallel_sft_priority'))
            raise
        finally:
            for entry in children.values():
                stop(entry['process'])
            result = subprocess.run(['sacct', '-X', '-n', '-P', '-j', job,
                '--format=JobIDRaw,State,ElapsedRaw,AllocTRES'], text=True, capture_output=True)
            write_json(ROOT / 'accounting' / (job + '.json'), dict(job_id=job,
                final_scheduler_audit_required=True, captured_unix=time.time(), sacct=result.stdout))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', required=True)
    execute(parser.parse_args().execute)
