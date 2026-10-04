#!/usr/bin/env python3
"""One TP1 stealth evaluation per ARM method in two approved allocations."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import run_arm_stealth90_repeats as base
from resume_baseline import allocation, validate_source, write_json

CONTROL = base.CONTROL / 'first-pass'
SOURCE = base.RUNTIME / 'reference-arm-stealth90-tp1-first-20260928'
RESOURCES = dict(jobs=2, gpus_per_job=1, gpu_type='H200', hours_per_job=7,
                 cpus_per_job=8, memory_gib_per_job=240, max_concurrent_jobs=2,
                 total_gpu_hours=14, browsers_per_job=4)
METHODS = ('additive', 'gate-b')
LABELS = [(method, 1) for method in METHODS]


def atomic(path, value):
    temp = path.with_name(path.name + f'.{os.getpid()}.partial')
    write_json(temp, value)
    temp.replace(path)


def prepare_source():
    base.prepare_source()
    if SOURCE.exists():
        validate_source(SOURCE)
        assert base.read(SOURCE/'reference_manifest.json')['single_gpu_evaluation']['parent'] == str(base.SOURCE)
        return
    shutil.copytree(base.SOURCE, SOURCE, symlinks=True,
        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git', '.env*', '.browser_use_sessions'))
    patches = {
        'scripts/run_h200_browser.sh': (' --sequence-parallel', ''),
        'openwebrl/browser_training_config.yaml': ('browser_rollout_concurrency: 8', 'browser_rollout_concurrency: 4'),
    }
    for name, (old, new) in patches.items():
        path = SOURCE/name
        text = path.read_text()
        if text.count(old) != 1:
            raise ValueError(f'Unexpected TP1 source: {name}')
        path.write_text(text.replace(old, new, 1))
    manifest = base.read(SOURCE/'reference_manifest.json')
    manifest['recipe_files_sha256'].update({n: hashlib.sha256((SOURCE/n).read_bytes()).hexdigest() for n in patches})
    manifest['single_gpu_evaluation'] = dict(parent=str(base.SOURCE), tensor_parallel_size=1,
        sequence_parallel=False, browser_concurrency=4, zero_optimizer_updates=True)
    manifest['browser_use_evaluation'].update(task_concurrency=4)
    write_json(SOURCE/'reference_manifest.json', manifest)
    validate_source(SOURCE)


def plan(method, repeat, job='PREPARE'):
    if method not in METHODS:
        raise ValueError('New baseline evaluations were removed at user request')
    if repeat != 1:
        raise ValueError('Only one initial evaluation per method is approved')
    p = base.build_plan(method, repeat, job, gpus=1, source=SOURCE)
    p['environment'].update(BROWSER_CONCURRENCY='4', SGLANG_CONCURRENCY='4',
        RAY_TMPDIR=f'/tmp/stealth90-ray-{job}-{method}-r{repeat}',
        WANDB_CACHE_DIR=str(Path(p['output'])/'wandb-cache'))
    p['browser_concurrency'] = 4
    return p


def prepare():
    CONTROL.mkdir(parents=True, exist_ok=True)
    prepare_source()
    for method, repeat in LABELS:
        write_json(CONTROL/f'{method}-r{repeat}-preview.json', plan(method, repeat))
    request = dict(status='prepared_for_approved_reduced_first_pass', resources=RESOURCES,
        full300_evaluations=2, task_attempts=600, waves=[LABELS],
        scheduling='Two separate jobs, one per method, no repeat or baseline jobs.',
        estimate='About4–5h per cohort, concurrently, plus queue time; seven-hour cap per cohort includes startup/recovery margin.',
        source=str(SOURCE), browser_capacity_receipt=str(base.CONTROL/'browser-capacity.json'),
        protocol='Same full300 GPT-4.1/action_history,T0,4096 tokens,30 turns; actor-only Browser Use stealth.',
        approval_scope='User accepted the proposed seven-hour single-GPU profile with only one initial run per method; reduced subset is two jobs.',
        baseline='Historical references only; no contemporaneous repeated baseline comparison.',
        implementation='scripts/run_arm_stealth90_parallel.py',
        submission_template='scripts/evaluate_arm_stealth90_1gpu.sbatch')
    write_json(CONTROL/'request.json', request)
    previous = base.read(base.CONTROL/'request.json')
    previous.update(status='superseded_by_user_parallel_preference', replacement=str(CONTROL/'request.json'))
    write_json(base.CONTROL/'request.json', previous)
    return request


def register(job, label, root, p):
    with (base.SUPERVISOR/'registry-update.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        registry = base.read(base.SUPERVISOR/'registry.json')
        key = 'stealth90-'+label
        item = next((j for j in registry['jobs'] if j['key'] == key), None)
        if item is None:
            item = dict(key=key, agent_reviewed_evaluation_roots=[])
            registry['jobs'].append(item)
        elif item['job_id'] != job:
            item.setdefault('supervised_attempts', []).append(dict(job_id=item['job_id'],
                controller_root=item['controller_root'], training_root=item['training_root']))
        item.update(job_id=job, controller_root=str(root), training_root=str(root),
            evaluations=[dict(iteration=90, label=label, root=p['output'], verified=False)],
            approved_total_seconds=7*3600, budget_receipt=str(CONTROL/'approval.json'),
            verified_complete=False, requires_user=False, requested_endpoint='full300 with archived verdicts')
        atomic(base.SUPERVISOR/'registry.json', registry)


def execute(job, method, repeat):
    label = f'{method}-r{repeat}'
    approval = base.read(CONTROL/'approval.json')
    ids = approval.get('job_ids_by_cohort', {}).get(label, [])
    if (not approval.get('approved') or approval.get('resources') != RESOURCES
            or job not in ids or os.getenv('SLURM_JOB_ID') != job):
        raise ValueError('Exact resource and time approval with registered attempt IDs required')
    resources = allocation(subprocess.check_output(['scontrol','show','job',job,'-o'],text=True),
                           job, requested_gpus=1, maximum_hours=7)
    raw = subprocess.check_output(['sacct','-X','-n','-P','-j',','.join(ids),
                                  '--format=JobID,ElapsedRaw'],text=True)
    used = sum(int(f[1]) for row in raw.splitlines() if len(f := row.split('|'))>=2
               and f[0] in ids and f[1].isdigit())
    remaining = min(resources['maximum_seconds'], 7*3600-used-180)
    if remaining < 1200:
        raise ValueError('Less than20 minutes remain in this cohort approval')
    root = CONTROL/f'controller-{label}-{job}'; root.mkdir(exist_ok=False)
    p = plan(method, repeat, job)
    write_json(root/'evaluation-plan.json', p)
    state = dict(complete=False, stage='evaluation', iteration=90, cohort=label,
                 job_id=job, consumed_seconds_at_start=used, remaining_seconds=remaining)
    atomic(root/'status.json', state)
    register(job, label, root, p)
    # Guard against duplicate/replacement overlap within the two-cohort study.
    slot = None
    try:
        for index in range(2):
            candidate = (CONTROL/f'parallel-slot-{index}.lock').open('a')
            try:
                fcntl.flock(candidate, fcntl.LOCK_EX|fcntl.LOCK_NB)
                slot = candidate
                break
            except BlockingIOError:
                candidate.close()
        if slot is None:
            raise RuntimeError('Two evaluation workers already active; inspect job dependencies')
        command = ['timeout','--signal=INT','--kill-after=120',str(int(remaining)-120),
            'srun',f'--jobid={job}','--nodes=1','--ntasks=1','--cpus-per-task=8',
            '--gres=gpu:h200:1','--exact','--cpu-bind=none',
            str(base.RUNTIME/'venv/bin/python'),'-B',str(base.REPO/'scripts/run_arm_stealth90_repeats.py'),
            '--worker','--manifest',str(root/'evaluation-plan.json')]
        subprocess.run(command, check=True)
        result = base.audit(p)
        state.update(complete=True, stage='artifacts_ready', evaluation=result)
        atomic(root/'status.json', state)
    except BaseException as exc:
        state.update(stage='failed', error=type(exc).__name__, detail=str(exc)[:600])
        atomic(root/'status.json', state)
        raise
    finally:
        if slot is not None:
            slot.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--job-id')
    parser.add_argument('--method', choices=list(METHODS))
    parser.add_argument('--repeat', type=int, choices=[1])
    args = parser.parse_args()
    if args.prepare:
        print(json.dumps(prepare(), indent=2))
    elif args.execute:
        execute(args.job_id, args.method, args.repeat)
    else:
        parser.error('Specify --prepare or --execute')
