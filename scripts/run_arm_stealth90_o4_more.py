#!/usr/bin/env python3
"""Two additional full300 repeats per method; no allocation submission here."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import run_arm_stealth90_o4 as first

base = first.base
CONTROL = base.RUNTIME / 'arm-turn-bonus-preparation/stealth90-o4-more-20260929'
SOURCE = CONTROL / 'source'
PARENT = first.SOURCE
METHODS = ('baseline', 'additive', 'gate-b')
RESOURCES = dict(jobs=6, gpus_per_job=1, gpu_type='H200', hours_per_job=7,
                 cpus_per_job=8, memory_gib_per_job=240, max_concurrent_jobs=3,
                 total_gpu_hours=42, browsers_per_job=3)


def prepare_source():
    first.validate_protocol_source(PARENT)
    CONTROL.mkdir(parents=True, exist_ok=True)
    if not SOURCE.exists():
        shutil.copytree(PARENT, SOURCE, symlinks=True,
            ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git', '.env*'))
        name = 'openwebrl/eval_benchmark.py'
        shutil.copy2(base.REPO/'scripts/arm_stealth_o4_generator.py', SOURCE/name)
        manifest = base.read(SOURCE/'reference_manifest.json')
        manifest['recipe_files_sha256'][name] = first.sha(SOURCE/name)
        manifest['stealth_more_repeats'] = dict(parent=str(PARENT),
            change='Fresh full300 repeats; unchanged protocol, credit-exhaustion fail-fast fix')
        first.write_json(SOURCE/'reference_manifest.json', manifest)
    first.validate_protocol_source(SOURCE)
    if first.sha(SOURCE/'openwebrl/eval_benchmark.py') != first.sha(base.REPO/'scripts/arm_stealth_o4_generator.py'):
        raise ValueError('Frozen generator differs from reviewed fail-fast implementation')


def validate(p):
    if p['method'] not in METHODS or p['repeat'] not in (2, 3):
        raise ValueError('Expected comparison method and repeat2/3')
    previous = first.SOURCE
    try:
        first.SOURCE = SOURCE
        first.validate_plan(p)
    finally:
        first.SOURCE = previous
    if p['expected_task_count'] != 300 or len(set(p['expected_rollout_task_ids'])) != 300:
        raise ValueError('Each repeat must contain all300 unique tasks')
    if p.get('provider_recovery'):
        raise ValueError('Fresh repeats cannot inherit recovery subset')
    cmd = p['command']
    if cmd.count('--seed') != 1 or cmd[cmd.index('--seed')+1] != str(1234+p['repeat']-1):
        raise ValueError('Repeat-specific SGLang server RNG seed missing')
    expected_name = f'stealth90-o4-t06-{p["method"]}-r{p["repeat"]}-{p["job_id"]}'
    if Path(p['output']).name != expected_name or p['wandb_run_id'] != expected_name:
        raise ValueError('Repeat artifacts and tracking identities must be unique')


def plan(method, repeat, job='PREPARE'):
    if method not in METHODS or repeat not in (2, 3):
        raise ValueError('Expected comparison method and repeat2/3')
    previous = first.SOURCE
    try:
        first.SOURCE = SOURCE
        p = first.plan(method, job)
    finally:
        first.SOURCE = previous
    old = Path(p['output']).name
    new = f'stealth90-o4-t06-{method}-r{repeat}-{job}'
    p = json.loads(json.dumps(p).replace(old, new))
    p.update(repeat=repeat, cohort_label=f'o4-t06-{method}-r{repeat}',
             server_rng_seed=1234+repeat-1)
    p['command'].extend(['--seed', str(p['server_rng_seed'])])
    base.evaluator.configure_evaluation_tracking(p)
    validate(p)
    return p


def prepare():
    prepare_source()
    plans = [plan(m, r) for r in (2, 3) for m in METHODS]
    assert all(p['expected_rollout_task_ids'] == plans[0]['expected_rollout_task_ids'] for p in plans)
    for p in plans:
        first.write_json(CONTROL/f'{p["method"]}-r{p["repeat"]}-preview.json', p)
    request = dict(status='prepared_awaiting_exact_resource_approval', resources=RESOURCES,
        protocol=first.PROTOCOL, task_attempts=1800, checkpoints={m:str(base.checkpoint(m)[0]) for m in METHODS},
        scheduling='Round2: baseline, Additive and GateB together; all round3 jobs depend afterok on all round2 attempts. Update dependencies after recovery. Nine browsers maximum.',
        estimates='Observed first cohorts about5h including interrupted collection/recovery;7h cap per new cohort, release on completion.',
        reporting='Per-repeat overall and valid-only, valid denominators, fixed100/full300; arithmetic mean and sample SD across3 evaluations; no pass@3 or independent-training-seed claim.',
        comparison_limit='Three repeated evaluations per method measure evaluation variability, not independent training-seed uncertainty; record collection dates and compare methods within each round.',
        seeds={'r2':1235,'r3':1236}, seed_scope='SGLang server RNG; live websites and asynchronous generation are not fully deterministic.',
        budget_scope='Each new cohort has its own25200-second cap including every retry; previous cohorts retain their own accounting.',
        approval_file=str(CONTROL/'approval.json'), prepared_epoch=time.time())
    first.write_json(CONTROL/'request.json', request)
    return request


def accounting(method, repeat, job):
    approval = base.read(CONTROL/'approval.json')
    key = f'{method}-r{repeat}'
    ids = approval.get('job_ids_by_cohort', {}).get(key, [])
    if (approval.get('approved') is not True or approval.get('resources') != RESOURCES
            or job not in ids or os.environ.get('SLURM_JOB_ID') != job):
        raise ValueError('Exact resource approval and registered attempt required')
    raw = subprocess.check_output(['sacct','-X','-n','-P','-j',','.join(ids),
        '--format=JobID,State,ElapsedRaw,ExitCode'], text=True)
    attempts = []
    for line in raw.splitlines():
        f = line.split('|')
        if len(f) >= 4 and f[0] in ids:
            attempts.append(dict(job_id=f[0], state=f[1], elapsed_seconds=int(f[2]), exit_code=f[3]))
    if {a['job_id'] for a in attempts} != set(ids):
        raise ValueError('Missing scheduler accounting for an approved attempt')
    used = sum(a['elapsed_seconds'] for a in attempts)
    receipt = dict(cohort=key, approved_seconds=25200, consumed_seconds=used,
                   remaining_seconds=max(0,25200-used), attempts=attempts, checked_epoch=time.time())
    first.write_json(CONTROL/f'{key}-attempts.json', receipt)
    return receipt


def register(job, method, repeat, root, p, budget):
    with (base.SUPERVISOR/'registry-update.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        registry = base.read(base.SUPERVISOR/'registry.json')
        key = f'stealth90-o4-t06-{method}-r{repeat}'
        item = next((j for j in registry['jobs'] if j['key'] == key), None)
        if item is None:
            item = dict(key=key, agent_reviewed_evaluation_roots=[])
            registry['jobs'].append(item)
        elif item.get('job_id') != job:
            item.setdefault('supervised_attempts', []).append(dict(job_id=item['job_id'], controller_root=item['controller_root']))
        item.update(job_id=job, controller_root=str(root), training_root=str(root),
            evaluations=[dict(iteration=90,label=p['cohort_label'],root=p['output'],verified=False)],
            approved_total_seconds=25200, budget_receipt=str(CONTROL/'approval.json'),
            last_budget_check=budget, verified_complete=False, requires_user=False,
            require_agent_completion_review=True, agent_completion_reviewed=False,
            requested_endpoint='Fresh full300 o4-mini/T0.6 archives and verdicts')
        first.previous.atomic(base.SUPERVISOR/'registry.json', registry)


def execute(job, method, repeat):
    budget = accounting(method, repeat, job)
    resources = first.allocation(subprocess.check_output(['scontrol','show','job',job,'-o'],text=True),
                                 job, requested_gpus=1, maximum_hours=7)
    remaining = min(resources['maximum_seconds'], budget['remaining_seconds'])-180
    if remaining < 1200:
        raise ValueError('Less than20min remain in this cohort budget')
    root = CONTROL/f'controller-{method}-r{repeat}-{job}'
    root.mkdir(exist_ok=False)
    first.write_json(root/'provider-balance.json', first.check_provider_balance())
    p = plan(method, repeat, job)
    first.write_json(root/'evaluation-plan.json', p)
    state = dict(complete=False,stage='evaluation',iteration=90,repeat=repeat,job_id=job)
    first.previous.atomic(root/'status.json', state)
    register(job,method,repeat,root,p,budget)
    with (CONTROL/f'{method}-r{repeat}-active.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            subprocess.run(['timeout','--signal=INT','--kill-after=120',str(int(remaining)-120),
                'srun',f'--jobid={job}','--nodes=1','--ntasks=1','--cpus-per-task=8',
                '--gres=gpu:h200:1','--exact','--cpu-bind=none',str(base.RUNTIME/'venv/bin/python'),
                '-B',str(Path(__file__).resolve()),'--worker','--manifest',str(root/'evaluation-plan.json')],check=True)
            state.update(complete=True,stage='artifacts_ready',evaluation=first.audit(p))
        except BaseException as exc:
            state.update(stage='failed',error=type(exc).__name__,detail=str(exc)[:600])
            raise
        finally:
            first.previous.atomic(root/'status.json',state)
            accounting(method,repeat,job)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--execute',action='store_true')
    parser.add_argument('--worker',action='store_true');parser.add_argument('--method',choices=METHODS)
    parser.add_argument('--repeat',type=int,choices=(2,3));parser.add_argument('--job-id')
    parser.add_argument('--manifest',type=Path);args=parser.parse_args()
    if args.prepare:print(json.dumps(prepare(),indent=2))
    elif args.worker:
        p=base.read(args.manifest);validate(p)
        os.environ['OPENWEBRL_EXPECTED_SCHEDULER_OFFSET']=p['environment']['OPENWEBRL_EXPECTED_SCHEDULER_OFFSET']
        os.environ['OPENWEBRL_RECORD_EVAL_EXPECTED_TASK_COUNT']='300'
        base.evaluator.run(p,base.REPO/'.env')
    elif args.execute:execute(args.job_id,args.method,args.repeat)
    else:parser.error('Choose --prepare, --execute or --worker')
