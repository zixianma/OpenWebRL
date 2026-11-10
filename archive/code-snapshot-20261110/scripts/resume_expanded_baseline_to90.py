#!/usr/bin/env python3
"""Own the approved 96h continuation and full300 milestones through iteration90."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import prepare_expanded_baseline_to90 as prep
import run_expanded_baseline as base
from resume_baseline import allocation, validate_source, write_json

CONTROL = prep.CONTROL
SOURCE = base.RUNTIME/'reference-expanded-outcome-4102-to90-20261005-v1'
SUPERVISOR = base.RUNTIME/'arm-turn-bonus-preparation/mixed-reweight-20260927/supervisor'
TOTAL_SECONDS = 96 * 3600
MILESTONES = tuple(range(10, 91, 10))
TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED'}


def endpoint_policy():
    path = CONTROL/'endpoint.json'
    if not path.exists():
        return dict(target_iteration=90, legacy_controller_jobs=[])
    policy = base.read(path)
    target = policy.get('target_iteration')
    if (policy.get('user_requested') is not True or type(target) is not int
            or target not in range(20, 91, 10)):
        raise ValueError('Invalid reduced endpoint; refusing to restore the old target')
    return policy


def endpoint():
    return endpoint_policy()['target_iteration']


def milestones():
    return tuple(i for i in MILESTONES if i <= endpoint())


def approval():
    a = base.read(CONTROL/'approval.json')
    expected = dict(gpus=8, gpu_type='H200', cpus=64, memory_gib=960,
                    additional_hours=96, additional_gpu_hours=768,
                    max_hours_per_allocation=24, blocks_hours=[24]*4)
    if (a.get('approved') is not True or a['resources'] != expected
            or a['target_iteration'] != 90 or a['approved_additional_seconds'] != TOTAL_SECONDS
            or a['additional_judge_cap_usd'] != 400 or a['judge_lifetime_cap_usd'] != 600):
        raise PermissionError('Exact continuation approval missing')
    return a


def accounting(job):
    approval()
    with (CONTROL/'budget.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = base.read(CONTROL/'attempts.json')
        rows = ledger['attempts']
        if len({r['job_id'] for r in rows}) != len(rows):
            raise ValueError('Duplicate budget job')
        raw = subprocess.check_output(['sacct','-X','-n','-P','-j',','.join(r['job_id'] for r in rows),
                                       '--format=JobIDRaw,State,ElapsedRaw'], text=True)
        records = {v[0]: v[1:] for line in raw.splitlines() if len(v := line.split('|')) >= 3}
        reserved = 0
        current = None
        for row in rows:
            if row['job_id'] not in records:
                raise ValueError('Missing accounting: preserve budget, do not reset')
            state, elapsed, *_ = records[row['job_id']]
            state = state.split()[0]; elapsed = int(elapsed)
            charged = max(elapsed, row.get('charged_seconds', 0))
            row.update(scheduler_state=state, elapsed_seconds=elapsed)
            if state in TERMINAL:
                row['charged_seconds'] = charged
            if not 0 < row['maximum_seconds'] <= 86400:
                raise ValueError('Invalid attempt cap')
            reserved += charged if state in TERMINAL else max(charged, row['maximum_seconds'])
            if row['job_id'] == job:
                current = row
            elif state == 'RUNNING':
                raise ValueError('Another allocation is writing this lineage')
        if reserved > TOTAL_SECONDS:
            raise ValueError('Retries and queued reservations exceed approved96h')
        if current is None or current['scheduler_state'] != 'RUNNING':
            raise ValueError('Current allocation is not registered/running')
        write_json(CONTROL/'attempts.json', ledger)
    info = subprocess.check_output(['scontrol','show','job',job,'-o'], text=True)
    resource = allocation(info, job, requested_gpus=8, maximum_hours=24)
    seconds = min(resource['maximum_seconds'], current['maximum_seconds']-current['elapsed_seconds'])-180
    if seconds < 600:
        raise ValueError('Insufficient approved runtime remaining')
    write_json(CONTROL/f'budget-{job}.json', dict(reserved_seconds=reserved,
        approved_seconds=TOTAL_SECONDS, remaining_current_seconds=seconds,
        original_budget_excluded=True, epoch=time.time()))
    return time.time()+seconds


def status(**fields):
    fields.setdefault('complete', False)
    fields.setdefault('failed', False)
    fields['updated_at_epoch_seconds'] = time.time()
    write_json(base.CONTROL/'status.json', fields)
    write_json(CONTROL/'status.json', fields)


def next_stage(completed, evaluated):
    limit = endpoint()
    if not 20 <= completed <= limit:
        raise ValueError('Unexpected continuation checkpoint')
    due = [i for i in milestones() if i <= completed and i not in evaluated]
    if due:
        return 'eval', due[0]
    if completed == limit:
        return 'complete', limit
    return 'train', (completed//10+1)*10


def training_plan(target, deadline):
    if target > endpoint():
        raise ValueError('Training stage exceeds the user stopping iteration')
    p = prep.training_plan(target)
    old = str(prep.original.SOURCE)
    p['command'] = [s.replace(old, str(SOURCE)) for s in p['command']]
    p['environment'] = {k: v.replace(old, str(SOURCE)) for k,v in p['environment'].items()}
    p['environment'].update(OPENWEBRL_TRAINING_DEADLINE_EPOCH=str(deadline),
                            OPENWEBRL_TRAINING_MIN_CYCLE_SECONDS='4500')
    return p


def replay(completed, job, deadline):
    if completed == 20:
        # Reconstruct only through the original audited metadata/image pipeline.
        # Do not inspect the damaged20.pt as a torch ZIP archive.
        output=base.RUN/'rollout_recovery_rebuilt/20.pt'
        scan=base.read(base.CONTROL/'replay21-metadata-salvage-scan.json')
        parity=base.read(base.CONTROL/'replay21-first-sample-parity.json')
        broken=base.RUN/'rollout_recovery/20.pt'
        if (not scan['complete_pickle'] or not parity['cached_pixel_parity']['full_match']
                or broken.stat().st_size != scan['file_bytes']):
            raise ValueError('Preserved reconstruction input failed its audit')
        if not output.exists():
            command=base.source_command(SOURCE,[str(base.PYTHON),'-B',str(CONTROL/'rebuild_replay21.py')])
            env=base.clean_env();env.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
            status(stage='rebuilding_saved_batch',job_id=job,durable_iteration=completed)
            base.run_child(command,env,CONTROL/f'rebuild-replay21-{job}.log',
                           min(deadline,time.time()+600),browser_guard=False)
        receipt=base.read(base.CONTROL/'replay21-reconstruction.json')
        if not receipt['complete'] or receipt['output']!=str(output):
            raise ValueError('Rebuilt replay archive is not verified')
        return base.replay_batch({'pending_replay_batch':str(output)},[base.RUN],base.RUN,completed-1)
    return base.replay_batch({}, [base.RUN], base.RUN, completed-1)


def train(job, target, completed, deadline):
    p = training_plan(target, deadline)
    env = base.credentials(); env.update(p['environment'])
    env['SLIME_BROWSER_LOCAL_PROCESS_LOG_DIR'] = str(base.RUN/'browser-server-logs')
    saved = replay(completed, job, deadline)
    if saved:
        env.update(OPENWEBRL_REPLAY_FIRST_BATCH=saved['batch'],
                   OPENWEBRL_REPLAY_ROLLOUT_ID=str(saved['rollout_id']),
                   OPENWEBRL_REPLAY_CONSUMED_GROUPS=str(saved['consumed_groups']))
    lease, port = base.lease_ports(job); env[base.PORT_ENV] = str(port)
    write_json(CONTROL/f'training-{job}-to{target}.json', dict(p, replay=saved))
    try:
        base.run_child(p['command'], env, CONTROL/f'training-{job}-to{target}.log', deadline)
    finally:
        lease.close()
    report = base.checkpoint_review()
    actual = report['iteration']+1
    if not completed <= actual <= target:
        raise ValueError('Unexpected completed checkpoint after training')
    return actual, report


def evaluated():
    out = {}
    for i in milestones():
        root = base.RUNTIME/f'evaluations/{base.RUN_ID}-iter{i}'
        if root.exists():
            # Partial artifacts require agent repair; never silently overwrite.
            out[i] = base.audit_eval(root)
    return out


def claim_supervision(job):
    p = SUPERVISOR/'registry.json'
    with (SUPERVISOR/'registry-update.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        r = base.read(p); j = next(x for x in r['jobs'] if x['key']=='outcome-expanded4102')
        rows = base.read(CONTROL/'attempts.json')['attempts']
        index = next(i for i,x in enumerate(rows) if x['job_id']==job)
        j.update(job_id=job, target_iteration=endpoint(), verified_complete=False,
            requires_user=False, approved_total_seconds=TOTAL_SECONDS,
            attempt_seconds=rows[index]['maximum_seconds'],
            budget_ledger=str(CONTROL/'attempts.json'), budget_receipt=str(CONTROL/'approval.json'),
            budget_scope='Separate approved96h extension; original24h not added',
            controller=str(Path(__file__).resolve()), judge_budget_usd=600,
            scheduled_continuations=[x['job_id'] for x in rows[index+1:]],
            startup_validated=False, gpu_checkpoint_restore_verified=False,
            first_optimizer_update_pending=True, latest_agent_check=time.time())
        old = {x['iteration']:x for x in j['evaluations']}
        j['evaluations'] = [old.get(i, dict(iteration=i,root=str(base.RUNTIME/f'evaluations/{base.RUN_ID}-iter{i}'),verified=False)) for i in milestones()]
        write_json(p,r)


def preflight():
    a = approval(); ready = base.read(CONTROL/'readiness.json')
    for name, expected in ready['file_hashes'].items():
        if base.digest(Path(name)) != expected:
            raise ValueError('Prepared continuation changed: '+name)
    validate_source(SOURCE)
    if base.digest(base.CONTROL/'combined-tasks.parquet') != a['data_sha256']:
        raise ValueError('Training pool changed')
    if not ready['tests_passed']:
        raise ValueError('Readiness checks incomplete')
    # Sharing the same filesystem lock protects the old recovery and extension.
    original = subprocess.check_output(['sacct','-X','-n','-P','-j','344274',
                                       '--format=JobIDRaw,State'],text=True)
    if not any(v[0]=='344274' and v[1].split()[0] in TERMINAL
               for line in original.splitlines() if len(v:=line.split('|'))>=2):
        raise ValueError('Original recovery is not terminal')


def worker(job, iteration):
    approval()
    if iteration > endpoint():
        raise ValueError('Evaluation exceeds the user stopping iteration')
    from evaluate_baseline_checkpoint import run
    os.environ['OPENWEBRL_EXPECTED_SCHEDULER_OFFSET'] = '0'
    base.SOURCE = SOURCE
    run(base.eval_plan(job, iteration), base.REPO/'.env')
    # An already-running Python controller retains its old loop in memory.
    # It awaits this freshly launched worker. At the reduced final milestone,
    # finish its artifacts and stop that exact allocation before returning.
    if (job in endpoint_policy().get('legacy_controller_jobs', [])
            and iteration == endpoint()):
        finish_legacy_controller(job)


def finish(job, report, done):
    completed = report['iteration'] + 1
    if next_stage(completed, done) != ('complete', endpoint()):
        raise ValueError('Endpoint checkpoint and all evaluations required before release')
    status(complete=True, stage='awaiting_agent_artifact_review', job_id=job,
           durable_iteration=completed, target_iteration=endpoint(),
           checkpoint=report, evaluations=done)
    # Completed attempts may have aged out of Slurm's live queue. Query the
    # user's live jobs once instead of asking squeue about each historical ID;
    # squeue rejects an aged-out ID even though the attempt finished correctly.
    raw = subprocess.check_output(
        ['squeue', '-h', '-u', str(os.getuid()), '-o', '%i|%T'], text=True)
    live_states = dict(line.strip().split('|', 1) for line in raw.splitlines()
                       if line.strip())
    released = []
    for row in base.read(CONTROL/'attempts.json')['attempts']:
        other = row['job_id']
        if other == job:
            continue
        if live_states.get(other) == 'PENDING':
            subprocess.run(['scancel',other], check=True)
            released.append(other)
    write_json(CONTROL/'completion-release.json', dict(
        released_jobs=released, target_iteration=endpoint(), epoch=time.time()))


def finish_legacy_controller(job):
    if os.getenv('SLURM_JOB_ID') != job or f'/job_{job}/' not in Path('/proc/self/cgroup').read_text():
        raise ValueError('Legacy endpoint stop requires its exact allocation')
    finish(job, base.checkpoint_review(), evaluated())
    write_json(CONTROL/'legacy-endpoint-stop.json', dict(
        job_id=job, target_iteration=endpoint(), epoch=time.time(),
        reason='User reduced endpoint; terminate old in-memory controller after final eval',
        independent_agent_review_pending=True))
    subprocess.run(['scancel',job], check=True)
    # Never return to the old controller, even while cancellation propagates.
    while True:
        time.sleep(30)


def execute(job):
    if os.getenv('SLURM_JOB_ID') != job or f'/job_{job}/' not in Path('/proc/self/cgroup').read_text():
        raise ValueError('Use the exact registered allocation')
    base.SOURCE = SOURCE
    with (base.CONTROL/'training-owner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            preflight(); deadline=accounting(job); claim_supervision(job)
            from check_storage_quota import check_storage_quota
            write_json(CONTROL/f'storage-{job}.json', check_storage_quota())
            report = base.checkpoint_review(); completed = report['iteration']+1
            done = evaluated()
            while True:
                action, target = next_stage(completed, done)
                if action == 'complete':
                    finish(job, report, done)
                    return
                required = 4500 if action=='train' else 3600
                if deadline-time.time() < required:
                    status(stage='partial_budget_stop',job_id=job,durable_iteration=completed,
                           target_iteration=endpoint(),next_stage=action,next_target=target)
                    return
                if action=='train':
                    status(stage='actor-startup',job_id=job,target=target,durable_iteration=completed)
                    previous=completed
                    completed, report=train(job,target,completed,deadline)
                    if completed<target:
                        # A safe boundary exit is valid only near the deadline.
                        if deadline-time.time()>4700:
                            raise ValueError('Training stopped early without budget cause')
                        status(stage='partial_budget_stop',job_id=job,durable_iteration=completed,
                               target_iteration=endpoint(),last_stage_start=previous)
                        return
                else:
                    status(stage='evaluation',job_id=job,target=target,iteration=target,durable_iteration=completed)
                    step=['srun',f'--jobid={job}','--nodes=1','--ntasks=1','--cpus-per-task=64',
                          '--gres=gpu:h200:8','--exact','--cpu-bind=none',str(base.PYTHON),
                          str(Path(__file__).resolve()),'--job-id',job,'--eval-worker',str(target)]
                    base.run_child(step,base.credentials(),CONTROL/f'evaluation-{job}-iter{target}.log',deadline,browser_guard=False)
                    done[target]=base.audit_eval(base.RUNTIME/f'evaluations/{base.RUN_ID}-iter{target}')
        except BaseException as exc:
            marker=base.RUN/'latest_checkpointed_iteration.txt'
            status(failed=True,stage='stopped',job_id=job,error_type=type(exc).__name__,
                   error=str(exc)[:800],durable_iteration=int(marker.read_text())+1,target_iteration=endpoint())
            raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job-id',required=True)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--eval-worker',type=int,choices=MILESTONES)
    a=p.parse_args()
    if a.eval_worker: worker(a.job_id,a.eval_worker)
    elif a.execute: execute(a.job_id)
    else: print(json.dumps(training_plan(30,time.time()+86400),indent=2))
