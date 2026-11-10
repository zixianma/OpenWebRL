#!/usr/bin/env python3
"""Uniform G8 outcome-only continuation 60 -> 90 with full-300 evaluations at 70/80/90.

Adapted from the to60 extension controller. Differences: the external per-stage storage
housekeeping receipt is replaced by a fresh quota check (this controller never deletes);
training stages run up to the next evaluation milestone when time allows instead of two
iterations; and the batch process holds the allocation after a failure so the supervising
agent can repair and retry inside it (idle hold capped at 1 h per failure, 2 h total).

Usage: --execute (batch entry), --register-job/--seconds[/--afterok] (ledger, before release),
--eval-worker N (srun step), no flags prints a preview.
"""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

C = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/training-controls/uniform8-original2102-20261006/to90-20261024')
sys.path[:0] = [str(C), str(Path(__file__).resolve().parent)]
import campaign_recipe as recipe  # noqa: E402
import legacy_support as support  # noqa: E402
from resume_baseline import allocation, validate_source, write_json  # noqa: E402

TOTAL = 42*3600
MAX = 86400
START, TARGET, EVALS = 60, 90, (70, 80, 90)
MIN_FREE_BYTES = 2*1024**4
HOLD_IDLE_SECONDS, HOLD_TOTAL_IDLE_SECONDS = 3600, 7200
TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE'}
RESOURCES = dict(gpus=8, gpu_type='H200', cpus=64, memory_gib=960, total_hours=42, total_gpu_hours=336,
                 allocations=[24, 18], max_hours_per_allocation=24)
CAPS = dict(judge_cap_usd=300, judge_call_cap=30000, browser_dispatch_cap=40000, browser_concurrency=64)


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def require(ok, why):
    if not ok:
        raise ValueError(why)


def approved():
    a = read(C/'approval-request.json')
    require(a.get('approved') is True and a.get('approval_id') and a.get('resources') == RESOURCES, 'Exact approval required')
    require(all(type(a.get(k)) is type(v) and a[k] == v for k, v in CAPS.items()), 'Approval caps changed')
    require(a['target_iteration'] == TARGET and a['resume_completed_iteration'] == START and a['full300_evaluations'] == list(EVALS)
            and a['unused_prior_budgets_added'] is False, 'Approval scope changed')
    return a


def verify_prepared():
    p = read(C/'readiness.json')
    require(p['cpu_checks_passed'], 'CPU readiness missing')
    for name, h in p['execution_file_sha256'].items():
        require(sha(name) == h, 'Prepared execution file changed: '+name)
    validate_source(recipe.SOURCE)
    anchor = read(C/'resume-anchor.json')
    for name, h in anchor['proof_sha256'].items():
        require(sha(name) == h, 'Iteration-60 resume proof changed: '+name)
    for n, size in anchor['shard_bytes'].items():
        require((Path(anchor['checkpoint'])/n).stat().st_size == size, 'Iteration-60 checkpoint file changed: '+n)
    return anchor


def sacct_row(job):
    raw = subprocess.check_output(['sacct', '-X', '-n', '-P', '-j', str(job), '--format=JobIDRaw,State,ElapsedRaw,ExitCode'], text=True)
    rows = [line.split('|') for line in raw.splitlines() if line.split('|')[0] == str(job)]
    require(len(rows) == 1, f'Missing Slurm accounting row for {job}')
    return dict(state=rows[0][1].split()[0], elapsed=int(rows[0][2]), exit_code=rows[0][3])


def accounting(job):
    """Charge every attempt against the 42 h cap; return this attempt's usable deadline."""
    approved()
    with (C/'budget.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = read(C/'attempts.json')
        require(ledger['approved_total_seconds'] == TOTAL and ledger.get('prior_budget_seconds_added') == 0, 'Budget pooling forbidden')
        attempts = copy.deepcopy(ledger['attempts'])
        current = next((a for a in attempts if a['job_id'] == job), None)
        require(current is not None, 'Unregistered job')
        prior = reserved_future = 0
        for a in attempts:
            row = sacct_row(a['job_id'])
            require(row['elapsed'] >= int(a.get('charged_seconds', 0)), 'Accounting moved backwards')
            if a['job_id'] != job:
                if row['state'] == 'PENDING':
                    info = dict(re.findall(r'(\w+)=(\S+)', subprocess.check_output(['scontrol', 'show', 'job', a['job_id'], '-o'], text=True)))
                    require(a.get('predecessor_job_id') and info.get('Dependency', '').startswith('afterok:'+a['predecessor_job_id']),
                            'Pending attempt is not an afterok successor')
                    reserved_future += int(a['maximum_seconds'])
                else:
                    require(row['state'] in TERMINAL, 'Another live writer exists: '+a['job_id'])
                    prior += row['elapsed']
            a.update(charged_seconds=row['elapsed'], scheduler_state=row['state'], exit_code=row['exit_code'])
        require(prior+int(current['maximum_seconds'])+reserved_future <= TOTAL, 'Attempts exceed the 42 h cap')
        require(sacct_row(job)['state'] == 'RUNNING', 'Current job must be running')
        info = subprocess.check_output(['scontrol', 'show', 'job', job, '-o'], text=True)
        fields = dict(re.findall(r'(\w+)=(\S+)', info))
        for name in ('ReqTRES', 'AllocTRES'):
            tres = dict(x.split('=', 1) for x in fields[name].split(','))
            require(all(tres.get(k) == v for k, v in {'cpu': '64', 'mem': '960G', 'node': '1', 'gres/gpu': '8'}.items()),
                    'Wrong requested/allocated resources')
        resources = allocation(info, job, requested_gpus=8, maximum_hours=24)
        ledger.update(attempts=attempts, total_charged_seconds=sum(a['charged_seconds'] for a in attempts), last_checked_epoch=time.time())
        write_json(C/'attempts.json', ledger)
        mine = next(a for a in attempts if a['job_id'] == job)
        usable = min(resources['maximum_seconds'], int(mine['maximum_seconds'])-mine['charged_seconds'], TOTAL-prior-mine['charged_seconds'])-180
        require(usable >= 600, 'Approved time exhausted')
        return time.time()+usable


def register(job, seconds, afterok=None):
    require(str(job).isdigit() and 0 < seconds <= MAX, 'Invalid job or attempt cap')
    approved()
    with (C/'budget.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = read(C/'attempts.json')
        rows = ledger['attempts']
        require(all(a['job_id'] != str(job) for a in rows), 'Already registered')
        reserved = sum(int(a['charged_seconds']) if a.get('scheduler_state') in TERMINAL else int(a['maximum_seconds']) for a in rows)
        require(reserved+seconds <= TOTAL, 'Registration would exceed the 42 h cap')
        rows.append(dict(job_id=str(job), maximum_seconds=int(seconds), charged_seconds=0, ordinal=len(rows)+1,
                         predecessor_job_id=str(afterok) if afterok else None, dependency_type='afterok' if afterok else None,
                         registered_epoch=time.time()))
        write_json(C/'attempts.json', ledger)


def plan_evaluation(job, iteration):
    require(iteration in EVALS, 'Unexpected evaluation')
    p = support.eval_plan(job, iteration)
    p['environment']['SLIME_BROWSER_ENV_MODE'] = 'local_process'
    p['environment']['OPENWEBRL_EXPECTED_SCHEDULER_OFFSET'] = '0'
    p['command'][p['command'].index('--eval-config')+1] = str(recipe.SOURCE/'openwebrl/online_mind2web_monitor.yaml')
    require(p['wandb_project'] == 'openwebrl-evals' and p['environment']['OPENWEBRL_SAMPLING_CONTROL'] == str(C), 'Evaluation budget/project leak')
    return p


def audit_existing_evaluation(root, iteration):
    p = read(root/'evaluation_manifest.json')
    require(p['checkpoint'] == str(recipe.RUN/f'iter_{iteration-1:07d}') and p['source'] == str(recipe.SOURCE)
            and p['parent_training_run'] == recipe.RUN_ID and p['browser_env'] == 'local_process', 'Existing milestone identity mismatch')
    require(p['wandb_project'] == 'openwebrl-evals' and p['checkpoint_index'] == iteration-1 and p['optimizer_updates_requested'] == 0,
            'Wrong milestone/no-update identity')
    result = support.audit_eval(root)
    require(read(root/'launcher_exit.json')['returncode'] == 0, 'Evaluation launcher incomplete')
    return result


def stage_target(completed, remaining):
    next_eval = next(x for x in EVALS if x > completed)
    # Observed G8 maximum 78 min per iteration; budget 90 min each plus 40 min for an evaluation.
    slots = int((remaining-2400-300)//5400)
    return min(next_eval, completed+slots) if slots >= 1 else None


def next_action(completed, remaining, eval_ready):
    require(START <= completed <= TARGET, 'Unexpected durable state')
    for milestone in EVALS:
        if milestone <= completed and not eval_ready(milestone):
            return ('evaluate', milestone) if remaining >= 2400 else ('handoff', None)
    if completed == TARGET:
        return ('done', None)
    target = stage_target(completed, remaining)
    return ('train', target) if target else ('handoff', None)


def quota_check():
    # Headroom is measured to the hard limit (user decision 2026-10-25): the soft limit is
    # within 1 TiB, and crossing it only starts the grace period.
    from check_storage_quota import check_storage_quota
    report = check_storage_quota(minimum_free_bytes=0)
    require(report['hard_headroom_bytes'] >= MIN_FREE_BYTES, 'Scrubbed hard-limit headroom below 2 TiB')
    report.update(launch_quota_basis='hard_limit', launch_headroom_bytes=report['hard_headroom_bytes'],
                  minimum_free_bytes=MIN_FREE_BYTES)
    return report


def status(**values):
    write_json(C/'status.json', dict(updated_at_epoch_seconds=time.time(), **values))


def execute_inner(job):
    approved()
    verify_prepared()
    require(os.getenv('SLURM_JOB_ID') == job and f'/job_{job}/' in Path('/proc/self/cgroup').read_text(), 'Use the registered Slurm allocation')
    with (C.parent/'training-owner.lock').open('a') as lineage_lock, (C/'training-owner.lock').open('a') as lock:
        fcntl.flock(lineage_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        deadline = accounting(job)
        write_json(C/'storage-startup.json', quota_check())
        marker = recipe.RUN/'latest_checkpointed_iteration.txt'
        completed = int(marker.read_text())+1
        require(START <= completed <= TARGET, 'Missing/wrong continuation checkpoint')
        support.checkpoint_review()

        def ready(i):
            root = recipe.RUNTIME/f'evaluations/{recipe.RUN_ID}-iter{i}'
            if not root.exists():
                return False
            audit_existing_evaluation(root, i)
            return True
        try:
            while True:
                action, target = next_action(completed, deadline-time.time(), ready)
                status(complete=False, stage={'train': 'actor-startup', 'evaluate': 'evaluation'}.get(action, action),
                       job_id=job, target=target, durable_iteration=completed, stage_started_epoch_seconds=time.time())
                if action == 'handoff':
                    write_json(C/f'handoff-{job}.json', dict(durable_iteration=completed, workers_stopped=True,
                                                            paid_successor_required=completed < TARGET, epoch=time.time()))
                    return
                if action == 'done':
                    status(complete=True, verified_complete=False, stage='awaiting_agent_artifact_review', job_id=job, durable_iteration=TARGET,
                           evaluations={i: audit_existing_evaluation(recipe.RUNTIME/f'evaluations/{recipe.RUN_ID}-iter{i}', i) for i in EVALS})
                    return
                write_json(C/'storage-stage-check.json', quota_check())
                if action == 'train':
                    report = support.train_stage(job, target, completed, deadline-2400)
                    completed = report['iteration']+1
                else:
                    root = recipe.RUNTIME/f'evaluations/{recipe.RUN_ID}-iter{target}'
                    require(not root.exists(), 'Partial evaluation exists; review it before any retry')
                    step = ['srun', f'--jobid={job}', '--nodes=1', '--ntasks=1', '--cpus-per-task=64', '--gres=gpu:h200:8', '--exact',
                            '--cpu-bind=none', str(recipe.PYTHON), '-B', str(Path(__file__).resolve()), '--job-id', job, '--eval-worker', str(target)]
                    support.run_child(step, support.credentials(), C/f'evaluation-{job}-iter{target}.log', deadline, browser_guard=False)
                    audit_existing_evaluation(root, target)
        except BaseException as e:
            status(complete=False, failed=True, stage='stopped', job_id=job, error_type=type(e).__name__, error=str(e)[:600],
                   durable_iteration=int(marker.read_text())+1)
            raise


def hold(job, reason):
    """Keep the allocation after a failure; run agent requests from C/<job>-hold-request.json.

    {"action": "retry"} reruns the controller (fresh code); {"action": "run", "argv": [...]}
    runs a command; {"action": "release"} ends. Returns True once a retry succeeds."""
    request = C/f'{job}-hold-request.json'
    idle_used = 0.
    while True:
        limit = min(HOLD_IDLE_SECONDS, HOLD_TOTAL_IDLE_SECONDS-idle_used)
        write_json(C/f'hold-{job}.json', dict(holding=True, reason=str(reason)[:600], idle_limit_seconds=int(limit),
                                              idle_used_seconds=int(idle_used), request_path=str(request), epoch=time.time()))
        start, r = time.time(), None
        while time.time()-start < limit:
            if request.is_file():
                r = read(request)
                request.rename(request.with_name(f'{request.name}.{int(time.time())}.consumed'))
                break
            time.sleep(15)
        idle_used += time.time()-start
        with (C/f'hold-{job}.jsonl').open('a') as log:
            log.write(json.dumps(dict(epoch=time.time(), request=r, idle_used_seconds=int(idle_used)))+'\n')
        if r is None or r.get('action') == 'release':
            write_json(C/f'hold-{job}.json', dict(holding=False, released=True, idle_cap=r is None, epoch=time.time()))
            return False
        argv = inner_argv(job) if r.get('action') == 'retry' else list(map(str, r.get('argv', [])))
        code = subprocess.run(argv).returncode if argv else 2
        with (C/f'hold-{job}.jsonl').open('a') as log:
            log.write(json.dumps(dict(epoch=time.time(), argv=argv, exit_code=code))+'\n')
        if code == 0 and r.get('action') == 'retry':
            write_json(C/f'hold-{job}.json', dict(holding=False, retry_succeeded=True, epoch=time.time()))
            return True
        reason = f'{r.get("action")} exited {code}'


def inner_argv(job):
    return [str(recipe.PYTHON), '-B', str(Path(__file__).resolve()), '--job-id', job, '--execute-inner']


def execute(job):
    require(os.getenv('SLURM_JOB_ID') == job, 'Use the registered Slurm allocation')
    if subprocess.run(inner_argv(job)).returncode == 0:
        return 0
    failure = read(C/'status.json') if (C/'status.json').exists() else {}
    return 0 if hold(job, f"{failure.get('error_type')}: {failure.get('error')}") else 1


def preview():
    marker = int((recipe.RUN/'latest_checkpointed_iteration.txt').read_text())+1
    return dict(approved=read(C/'approval-request.json')['approved'], resources=RESOURCES, caps=CAPS,
                current_completed_iteration=marker, first_stage_target=stage_target(marker, 86400-180),
                training=recipe.training_command(marker+1, resume=True), evaluations=list(EVALS), run_id=recipe.RUN_ID)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job-id')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--execute-inner', action='store_true')
    p.add_argument('--eval-worker', type=int, choices=EVALS)
    p.add_argument('--register-job')
    p.add_argument('--seconds', type=int)
    p.add_argument('--afterok')
    a = p.parse_args()
    if a.register_job:
        register(a.register_job, a.seconds, a.afterok)
    elif a.execute:
        sys.exit(execute(a.job_id))
    elif a.execute_inner:
        execute_inner(a.job_id)
    elif a.eval_worker:
        approved(); verify_prepared(); os.environ['OPENWEBRL_EXPECTED_SCHEDULER_OFFSET'] = '0'
        from evaluate_baseline_checkpoint import run
        run(plan_evaluation(a.job_id, a.eval_worker), recipe.REPO/'.env')
    else:
        print(json.dumps(preview(), indent=2))
