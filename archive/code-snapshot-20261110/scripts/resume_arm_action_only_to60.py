#!/usr/bin/env python3
"""Continue the mixed-only action-reward-only run (383265) from iteration 10 to 60.

Approved 2026-10-24: 8 H200 / 64 CPU / 960 GiB for 56 h total across all attempts
(24 h + 24 h + 8 h afterok chain), GPT-4.1 judge cap $350, full-300 local
GPT-4.1/T0 evaluations at 20/30/40/50/60. The first job's unused 12 h balance is not added.

This reuses resume_arm_mixed_to60 (the controller that took the matched bonus/reweight
pair from 10 to 60) unchanged, rebinding its lineage constants to the action-only run:
source, run ID, recipe check (advantage_mode=action_only), approval, ledger and
controllers. Objective, optimizer/scheduler/cursor and W&B run are preserved. The batch
process holds the allocation after a failure (1 h per failure, 2 h total) so the agent can
repair and retry inside it.

--prepare (CPU, after 383265 has finished), --register-job/--seconds[/--afterok],
--execute (batch entry), plus the base controller's --worker entry.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import check_storage_quota as quota
import prepare_arm_mixed_action_only as ao
import resume_arm_mixed_reweight as rw
import resume_arm_mixed_to60 as base
from resume_baseline import validate_source, write_json

VARIANT = 'action-only'
RUN_ID = 'arm-mixed-action-only-383265'
FIRST_ROOT = ao.RUNTIME/'evaluations/arm-mixed-action-only-383265'
EVAL10 = ao.RUNTIME/'evaluations/arm-mixed-action-only-iter10-383265'
CONTROL = ao.CONTROL/'to60-20261024'
TOTAL_SECONDS, MAX_ATTEMPT_SECONDS = 56*3600, 86400
RESOURCES = dict(gpus=8, gpu_type='H200', cpus=64, memory_gib=960)
JUDGE_CAP_USD = 350
HOLD_IDLE_SECONDS, HOLD_TOTAL_IDLE_SECONDS = 3600, 7200

# Rebind the shared continuation machinery to this lineage (this process only).
# __spec__.origin, not __file__: the worker re-imports this module after __file__ is patched below.
BASE_FILE = Path(base.__spec__.origin).resolve()
base.__file__ = str(Path(__file__).resolve())  # worker srun steps re-enter this module, not the base
rw.original = ao
rw.SOURCE = ao.SOURCE
base.SOURCE = ao.SOURCE
base.CONTROL = CONTROL
base.RUN_IDS = {VARIANT: RUN_ID}


def approval():
    a = base.read(CONTROL/'approval.json')
    if (a.get('approved') is not True or a.get('target') != 60 or a.get('variants') != [VARIANT]
            or a.get('additional_seconds_per_variant') != TOTAL_SECONDS or a.get('resources') != RESOURCES
            or a.get('max_attempt_seconds') != MAX_ATTEMPT_SECONDS or a.get('judge_cap_usd') != JUDGE_CAP_USD):
        raise ValueError('Exact action-only to60 approval missing')
    return a


def fingerprints():
    files = [Path(__file__).resolve(), BASE_FILE, Path(rw.__file__), Path(ao.__file__),
             ao.SOURCE/'reference_manifest.json', CONTROL/'training-controller.py', CONTROL/'restore-controller.py',
             base.REPO/'scripts/resume_arm_action_only_to60_8gpu.sbatch']
    return {str(p): ao.sha(p) for p in files}


def update_supervisor(*args, **kwargs):
    """The mixed-pair supervisor registry is not used for this lineage."""


_soft_quota_check = quota.check_storage_quota


def hard_quota_check(minimum_free_bytes=2*1024**4, **kwargs):
    """Headroom measured to the scrubbed hard limit (user decision 2026-10-25)."""
    report = _soft_quota_check(0, **kwargs)
    if report['hard_headroom_bytes'] < minimum_free_bytes:
        raise OSError('Scrubbed hard-limit headroom %.3f TiB is below %.3f TiB'
                      % (report['hard_headroom_bytes']/1024**4, minimum_free_bytes/1024**4))
    report.update(launch_quota_basis='hard_limit', launch_headroom_bytes=report['hard_headroom_bytes'],
                  minimum_free_bytes=minimum_free_bytes)
    return report


quota.check_storage_quota = hard_quota_check  # callers import the name at call time
base.approval = approval
base.fingerprints = fingerprints
base.update_supervisor = update_supervisor

# Names the copied controllers import from this module.
validate_plan = base.validate_plan
validate_resume_plan = rw.validate_resume_plan
validate_replayed_rewards = rw.validate_replayed_rewards
continuation_identity = rw.continuation_identity


def register(job, seconds, afterok=None):
    approval()
    if not str(job).isdigit() or not 600 <= seconds <= MAX_ATTEMPT_SECONDS:
        raise ValueError('Invalid job or attempt cap')
    path = CONTROL/f'{VARIANT}-attempts.json'
    with (CONTROL/f'{VARIANT}-budget.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = base.read(path) if path.exists() else dict(variant=VARIANT, attempts=[], approved_total_seconds=TOTAL_SECONDS)
        rows = ledger['attempts']
        if any(r['job_id'] == str(job) for r in rows):
            raise ValueError('Already registered')
        if sum(r.get('charged_seconds', r['attempt_seconds']) for r in rows)+seconds > TOTAL_SECONDS:
            raise ValueError('Registration would exceed the 56 h cap')
        rows.append(dict(job_id=str(job), attempt_seconds=int(seconds), afterok=str(afterok) if afterok else None,
                         restart_count=0, registered_epoch=time.time()))
        base.atomic(path, ledger)


def prepare():
    """CPU readiness; requires the first job to be inactive with a verified checkpoint."""
    approval()
    validate_source(ao.SOURCE)
    for name in ('training-controller.py', 'restore-controller.py'):
        text = (ao.CONTROL.parent/'mixed-reweight-20260927/to60'/name).read_text()
        text = text.replace('resume_arm_mixed_to60', Path(__file__).stem)
        compile(text, name, 'exec')
        (CONTROL/name).write_text(text)
    base.atomic(CONTROL/'proposal.json', dict(variants={VARIANT: dict(resume_from=str(FIRST_ROOT))}))
    o = base.choose_origin(VARIANT)
    completed = base.origin(o)['checkpoint_report']['iteration']+1
    statefile = CONTROL/f'{VARIANT}-progress.json'
    state = base.read(statefile) if statefile.exists() else dict(training_roots=[], evaluations={})
    if completed >= 10 and '10' not in state['evaluations'] and (EVAL10/'status.json').exists():
        ep = base.read(EVAL10/'evaluation_manifest.json')
        audit = base.audit_evaluation(ep)
        audit.update(iteration=10, plan=str(EVAL10/'evaluation_manifest.json'))
        state['evaluations']['10'] = audit
        base.atomic(statefile, state)
    stage, target = base.next_stage(completed, {int(i) for i in state['evaluations']})
    if stage == 'train':
        p = base.training_plan('PREPARE-'+VARIANT, VARIANT, o, target)
        base.native_check(p, CONTROL/'preflight')
        base.atomic(CONTROL/'initial-plan.json', p)
    ep = base.evaluation_plan('PREPARE-'+VARIANT, VARIANT, o, max(10, completed//10*10))
    if ep['expected_task_count'] != 300 or not ep['require_task_rollouts']:
        raise ValueError('Full300 rollout/verdict persistence missing')
    base.atomic(CONTROL/'evaluation-preview.json', ep)
    subprocess.run(['bash', '-n', str(base.REPO/'scripts/resume_arm_action_only_to60_8gpu.sbatch')], check=True)
    base.atomic(CONTROL/'readiness.json', dict(cpu_passed=True, native_parse_passed=stage == 'train',
        fingerprints=fingerprints(), completed_iteration=completed, next_stage=[stage, target],
        evaluations_seeded=sorted(state['evaluations'])))
    if stage == 'train':
        validate_plan(base.read(CONTROL/'initial-plan.json'))
    print(json.dumps(base.read(CONTROL/'readiness.json'), indent=1)[:1500])


def hold(job, reason):
    request = CONTROL/f'{job}-hold-request.json'
    idle_used = 0.
    while True:
        limit = min(HOLD_IDLE_SECONDS, HOLD_TOTAL_IDLE_SECONDS-idle_used)
        write_json(CONTROL/f'hold-{job}.json', dict(holding=True, reason=str(reason)[:600], request_path=str(request),
                                                    idle_limit_seconds=int(limit), epoch=time.time()))
        start, r = time.time(), None
        while time.time()-start < limit:
            if request.is_file():
                r = base.read(request)
                request.rename(request.with_name(f'{request.name}.{int(time.time())}.consumed'))
                break
            time.sleep(15)
        idle_used += time.time()-start
        if r is None or r.get('action') == 'release':
            write_json(CONTROL/f'hold-{job}.json', dict(holding=False, released=True, idle_cap=r is None, epoch=time.time()))
            return 1
        argv = (inner(job) if r.get('action') == 'retry' else list(map(str, r.get('argv', []))))
        code = subprocess.run(argv).returncode if argv else 2
        with (CONTROL/f'hold-{job}.jsonl').open('a') as log:
            log.write(json.dumps(dict(epoch=time.time(), argv=argv, exit_code=code))+'\n')
        if code == 0 and r.get('action') == 'retry':
            return 0
        reason = f'{r.get("action")} exited {code}'


def inner(job):
    return [str(ao.PYTHON), '-B', str(Path(__file__).resolve()), '--job-id', job, '--variant', VARIANT, '--execute-inner']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--job-id')
    parser.add_argument('--variant', choices=[VARIANT], default=VARIANT)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--execute-inner', action='store_true')
    parser.add_argument('--worker', choices=['train', 'eval'])
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--register-job')
    parser.add_argument('--seconds', type=int)
    parser.add_argument('--afterok')
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.register_job:
        register(args.register_job, args.seconds, args.afterok)
    elif args.worker:
        base.worker(args.job_id, VARIANT, args.worker, args.manifest)
    elif args.execute_inner:
        base.execute(args.job_id, VARIANT)
    elif args.execute:
        code = subprocess.run(inner(args.job_id)).returncode
        sys.exit(0 if code == 0 else hold(args.job_id, f'controller exited {code}; see {CONTROL}'))
    else:
        parser.error('Choose --prepare, --register-job, --execute or --worker')
