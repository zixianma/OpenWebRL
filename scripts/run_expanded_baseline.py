#!/usr/bin/env python3
"""Own fresh4102-task training to60 and full300 evaluations every10 iterations.

No sbatch submission is performed. An approved, registered8-H200 allocation is
required. All attempts share the original24h GPU and$200 judge ledgers.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import zipfile

from prepare_expanded_baseline import (CONTROL, REPO, RUNTIME, RUN, RUN_ID, SOURCE,
    PYTHON, TARGET_ITERATION, EVAL_ITERATIONS, TRAINING_STAGES,
    clean_env, digest, training_command)
from resume_baseline import allocation, source_command, write_json, replay_batch
from runtime_ports import lease_ports, PORT_ENV
from arm_browser_startup_guard import BrowserStartupGuard


def read(path):
    return json.loads(Path(path).read_text())


def verify_approval(job):
    approval = read(CONTROL/'approval-request.json')
    expected = dict(jobs=1,gpus=8,gpu_type='H200',hours=24,gpu_hours=192,cpus=64,memory_gib=960)
    if approval.get('approved') is not True or approval['resources'] != expected or approval['judge_cap_usd'] != 200:
        raise PermissionError('Exact8H200x24h/64CPU/960GiB/$200 approval is required')
    if approval.get('target_iteration') != TARGET_ITERATION or tuple(approval.get('full300_evaluations',[])) != EVAL_ITERATIONS:
        raise PermissionError('Training endpoint and evaluation milestones do not match approval')
    attempts = read(CONTROL/'attempts.json')['attempts']
    if len({a['job_id'] for a in attempts}) != len(attempts):
        raise ValueError('Duplicate budget-ledger job ID')
    current = next((a for a in attempts if a['job_id'] == job), None)
    if current is None:
        raise PermissionError('Job not registered in this experiment budget')
    return attempts, current


def accounting(job):
    attempts, current = verify_approval(job)
    # ElapsedRaw includes all allocated time in failed and successful attempts.
    total = 0
    for a in attempts:
        raw = subprocess.check_output(['sacct','-X','-n','-P','-j',a['job_id'],
            '--format=JobIDRaw,State,ElapsedRaw,ExitCode'],text=True)
        matches = [line.split('|') for line in raw.splitlines() if line.split('|')[0]==a['job_id']]
        if len(matches)!=1:
            raise ValueError('Accounting unavailable; do not reset the budget')
        _, state, elapsed, exit_code, *_ = matches[0]
        if a['job_id']!=job and state.split()[0] not in {'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED'}:
            raise ValueError('Another writer or queued attempt exists for this lineage')
        charged=max(int(elapsed),a.get('charged_seconds',0))
        a.update(charged_seconds=charged,scheduler_state=state,exit_code=exit_code)
        if a['job_id']!=job:
            total+=charged
    write_json(CONTROL/'attempts.json',dict(attempts=attempts,approved_total_seconds=86400))
    info=subprocess.check_output(['scontrol','show','job',job,'-o'],text=True)
    resources=allocation(info,job,requested_gpus=8,maximum_hours=24)
    # Bound the allocation as well as workers: retries cannot allocate a fresh24h.
    requested=int(current['maximum_seconds'])
    if total+requested>86400 or requested<=0:
        raise ValueError('Current allocation exceeds the remaining original budget')
    remaining=min(resources['maximum_seconds'],requested-current.get('charged_seconds',0)-180)
    if current.get('scheduler_state') != 'RUNNING':
        raise RuntimeError('Accounting has not caught up to the running allocation')
    if remaining<600:
        raise RuntimeError('Approved time exhausted')
    return resources,time.time()+remaining


def credentials():
    from dotenv import dotenv_values
    env=clean_env()
    for k,v in dotenv_values(REPO/'.env').items():
        if v and k.startswith(('WANDB_','JUDGE_','OPENAI_')):
            env.setdefault(k,v)
    if not env.get('WANDB_API_KEY') or not (env.get('OPENAI_API_KEY') or env.get('JUDGE_API_KEY')):
        raise ValueError('Training/judge credentials not configured')
    return env


def checkpoint_review():
    out=CONTROL/'checkpoint-review.json'
    env=clean_env();env.update(FLASHINFER_WORKSPACE_BASE=str(RUNTIME),OMP_NUM_THREADS='1')
    subprocess.run(source_command(SOURCE,[str(PYTHON),str(REPO/'scripts/inspect_training_checkpoint.py'),
        str(RUN),'--sample-payloads','--report',str(out)]),env=env,check=True,stdout=subprocess.DEVNULL)
    report=read(out)
    if report['scheduler_minus_optimizer_updates']!=0 or not report['sampled_cpu_payloads']['finite']:
        raise ValueError('Checkpoint integrity/optimizer state failed')
    return report


def run_child(command, env, log, deadline, *, browser_guard=True):
    log.parent.mkdir(parents=True,exist_ok=True)
    guard=BrowserStartupGuard()
    with log.open('w') as handle:
        started=time.time()
        proc=subprocess.Popen(command,cwd=SOURCE,env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while proc.poll() is None:
                if (CONTROL/'judge-budget/halt.json').exists():
                    raise RuntimeError('Judge ledger halted the run; preserve checkpoint and completed records')
                if time.time()>deadline:
                    raise TimeoutError('Original approved allocation boundary reached')
                if browser_guard:
                    health=guard.poll(log)
                    write_json(CONTROL/'browser-startup-health.json',health)
                    if health['stop']:
                        raise RuntimeError(health['issue'])
                    progress=RUN/'progress.log'
                    if progress.exists() and progress.stat().st_mtime>=started:
                        with progress.open('rb') as f:
                            f.seek(max(0,progress.stat().st_size-16384))
                            text=f.read().decode(errors='replace')
                        phases=re.findall(r'\[TrainProgress\] rollout=(\d+)/\d+ phase=(\S+)',text)
                        if phases:
                            iteration,phase=phases[-1]
                            current=read(CONTROL/'status.json')
                            current.update(iteration=int(iteration),stage='collection' if phase in {'generate','generate_rollout'} else 'training',
                                updated_at_epoch_seconds=time.time())
                            marker=RUN/'latest_checkpointed_iteration.txt'
                            if marker.exists():current['completed_iterations']=int(marker.read_text())+1
                            write_json(CONTROL/'status.json',current)
                time.sleep(10)
            if proc.returncode:
                raise RuntimeError(f'Worker failed with exit{proc.returncode}; inspect {log}')
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGINT)
                try:
                    proc.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid,signal.SIGKILL)
                    proc.wait()


def eval_plan(job, iteration):
    from evaluate_baseline_checkpoint import build_plan,configure_evaluation_tracking
    root=RUNTIME/f'evaluations/{RUN_ID}-iter{iteration}'
    p=build_plan(SOURCE,RUN/f'iter_{iteration-1:07d}',root,job,gpus=8)
    p.update(wandb_run_id=f'{RUN_ID}-iter{iteration}',parent_training_run=RUN_ID,
        allocation_hours_cap=24,exclusive_step_wait_seconds=120)
    p['environment'].update(TP_SIZE='2',BROWSER_CONCURRENCY='64',SGLANG_CONCURRENCY='64',
        TRAIN_DATA=str(CONTROL/'combined-tasks.parquet'),
        WANDB_RUN_ID=p['wandb_run_id'],OPENWEBRL_CUDA_CACHE_LIMIT_GIB='48',
        OPENWEBRL_EXPANDED_BASELINE_CONTROL=str(CONTROL),
        JUDGE_API_MODE='served',JUDGE_API_BASE='https://api.openai.com/v1',
        OPENWEBRL_MULTIMODAL_STORAGE_DIR=str(root/'multimodal'))
    p['command'][p['command'].index('--wandb-group')+1]=RUN_ID+'-evaluation'
    p['expected_rollout_task_ids']=[json.loads(line)['metadata']['task_id'] for line in (SOURCE/'online_mind2web_monitor.jsonl').read_text().splitlines()]
    return configure_evaluation_tracking(p)


def audit_eval(root):
    root=Path(root);status=read(root/'status.json');plan=read(root/'evaluation_manifest.json')
    if not status['complete'] or status['returncode']!=0:
        raise ValueError('Incomplete evaluation')
    records={}
    for f in (root/'rollouts').glob('*.json'):
        r=read(f);ident=r['task_id']
        assert ident not in records and (r['judge_model'],r['judge_prompt_variant'])==('gpt-4.1','action_history')
        archive=(root/'rollouts'/r['rollout_file']).resolve()
        assert archive.is_relative_to((root/'rollouts').resolve()) and archive.stat().st_size
        with zipfile.ZipFile(archive) as z:
            assert sum(n.endswith('/data.pkl') for n in z.namelist())==1
        records[ident]=r
    assert len(records)==300 and set(records)==set(plan['expected_rollout_task_ids'])
    restore=read(root/'checkpoint_restore_evidence.json');assert restore['checkpoint']==plan['checkpoint']
    wins=sum(r['metrics']['successes'] for r in records.values())
    valid=sum(r['metrics']['valid_trajectories'] for r in records.values())
    result=dict(tasks=300,successes=wins,valid=valid,invalid=300-valid,overall=wins/300,
        valid_only=wins/valid if valid else None,saved_rollouts=300,saved_verdicts=300,
        checkpoint=plan['checkpoint'],rollouts=str(root/'rollouts'))
    write_json(root/'controller-artifact-audit.json',result)
    return result


def train_stage(job, target, completed, deadline):
    plan=training_command(target,resume=completed>0)
    env=credentials();env.update(plan['environment'])
    env['SLIME_BROWSER_LOCAL_PROCESS_LOG_DIR']=str(RUN/'browser-server-logs')
    if completed:
        checkpoint_review()
        replay=replay_batch({},[RUN],RUN,completed-1)
        if replay:
            env.update(OPENWEBRL_REPLAY_FIRST_BATCH=replay['batch'],
                OPENWEBRL_REPLAY_ROLLOUT_ID=str(replay['rollout_id']),
                OPENWEBRL_REPLAY_CONSUMED_GROUPS=str(replay['consumed_groups']))
    lease,base=lease_ports(job);env[PORT_ENV]=str(base)
    write_json(CONTROL/f'training-{job}-to{target}.json',plan)
    try:
        run_child(plan['command'],env,CONTROL/f'training-{job}-to{target}.log',deadline)
    finally:
        lease.close()
    report=checkpoint_review()
    if report['iteration']!=target-1:
        raise ValueError('Training stopped before requested checkpoint')
    return report


def worker(job,iteration):
    from evaluate_baseline_checkpoint import run
    verify_approval(job)
    os.environ['OPENWEBRL_EXPECTED_SCHEDULER_OFFSET']='0'
    p=eval_plan(job,iteration)
    run(p,REPO/'.env')


def execute(job):
    if os.getenv('SLURM_JOB_ID')!=job or f'/job_{job}/' not in Path('/proc/self/cgroup').read_text():
        raise ValueError('Use the exact approved allocation')
    with (CONTROL/'training-owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        resources,deadline=accounting(job)
        from check_storage_quota import check_storage_quota
        write_json(CONTROL/'storage-startup.json',check_storage_quota())
        prepared=read(CONTROL/'launcher-preparation.json')
        assert prepared['source_compilation_passed'] and prepared['native_launcher_dry_runs_passed']
        preflight=read(CONTROL/'cpu-preflight.json')
        assert preflight['passed']
        for relative,expected in preflight['launcher_file_hashes'].items():
            if digest(REPO/relative)!=expected:
                raise ValueError('Prepared launcher changed: '+relative)
        for relative,expected in prepared['frozen_file_hashes'].items():
            if digest(SOURCE/relative)!=expected:
                raise ValueError('Frozen source changed: '+relative)
        assert digest(CONTROL/'combined-tasks.parquet')==prepared['data_sha256']
        RUN.mkdir(parents=True,exist_ok=True)
        link=CONTROL/'runtime'
        if not link.is_symlink():
            link.symlink_to(RUN,target_is_directory=True)
        if link.resolve()!=RUN.resolve():
            raise ValueError('Wrong monitor progress directory')
        write_json(CONTROL/'launch_manifest.json',dict(job_id=job,source=str(SOURCE),
            output=str(RUN),wandb_run_id=RUN_ID,started_epoch=time.time()))
        manifest=RUN/'launch_manifest.json'
        if not manifest.exists():
            write_json(manifest,dict(kind='fresh outcome-only expanded4102 baseline',
                source=str(SOURCE),data_sha256=prepared['data_sha256'],wandb_run_id=RUN_ID,
                initialization='OpenWebRL/OpenWebRL-4B-SFT iteration0',
                optimizer='fresh',scheduler='fresh',cursor=0,training=training_command(1)))
        elif read(manifest)['data_sha256']!=prepared['data_sha256']:
            raise ValueError('Existing lineage has different training data')
        marker=RUN/'latest_checkpointed_iteration.txt'
        completed=int(marker.read_text())+1 if marker.exists() else 0
        if not completed and any(RUN.glob('iter_*')):
            raise ValueError('Incomplete checkpoint exists; inspect it before fresh restart')
        try:
            # The first actual batch validates TP2/DP4 and produces a checkpoint.
            # Continuing to10 reloads that full model/optimizer state on GPU.
            for target in TRAINING_STAGES:
                if completed<target:
                    if deadline-time.time()<3600:
                        raise TimeoutError('Insufficient approved time for another training stage')
                    write_json(CONTROL/'status.json',dict(complete=False,stage='actor-startup',job_id=job,target=target,durable_iteration=completed))
                    report=train_stage(job,target,completed,deadline-2400)
                    completed=report['iteration']+1
                if target in EVAL_ITERATIONS:
                    root=RUNTIME/f'evaluations/{RUN_ID}-iter{target}'
                    if root.exists():
                        audit_eval(root)  # Partial evals stop for targeted recovery; never overwrite.
                        continue
                    if deadline-time.time()<2400:
                        raise TimeoutError('Evaluation due; retain checkpoint for recovery')
                    write_json(CONTROL/'status.json',dict(complete=False,stage='evaluation',job_id=job,target=target,iteration=target,durable_iteration=completed))
                    step=['srun',f'--jobid={job}','--nodes=1','--ntasks=1','--cpus-per-task=64',
                        '--gres=gpu:h200:8','--exact','--cpu-bind=none',str(PYTHON),str(Path(__file__).resolve()),
                        '--job-id',job,'--eval-worker',str(target)]
                    run_child(step,credentials(),CONTROL/f'evaluation-{job}-iter{target}.log',deadline,browser_guard=False)
                    audit_eval(root)
            write_json(CONTROL/'status.json',dict(complete=True,stage='awaiting_agent_artifact_review',job_id=job,
                durable_iteration=completed,target_iteration=TARGET_ITERATION,checkpoint=checkpoint_review(),
                evaluations={i:audit_eval(RUNTIME/f'evaluations/{RUN_ID}-iter{i}') for i in EVAL_ITERATIONS}))
        except BaseException as e:
            write_json(CONTROL/'status.json',dict(complete=False,failed=True,stage='stopped',job_id=job,
                error_type=type(e).__name__,error=str(e)[:800],durable_iteration=int(marker.read_text())+1 if marker.exists() else 0))
            raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job-id',required=True)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--eval-worker',type=int,choices=EVAL_ITERATIONS)
    a=p.parse_args()
    if a.eval_worker:
        worker(a.job_id,a.eval_worker)
    elif a.execute:
        execute(a.job_id)
    else:
        print(json.dumps(training_command(1),indent=2))
