#!/usr/bin/env python3
"""Prepare/own eight single-GPU screen shards; never submits an allocation."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

import run_arm_task_screen as base
from openwebrl.arm_task_screen_sharded import partition, validate_partition
from openwebrl.arm_task_screen import summarize
from resume_baseline import allocation, clean_environment, source_command, validate_source, write_json
from run_arm_quality_audit import environment, stop
from runtime_ports import PORT_ENV, lease_ports
from check_storage_quota import check_storage_quota

REPO, RUNTIME = base.REPO, base.RUNTIME
CONTROL = base.CONTROL/'sharded-20261002'
SOURCE = RUNTIME/'reference-arm-task-screen-sharded-20261002-v2'
RESOURCES = dict(jobs=8, gpus_per_job=1, gpu_type='H200', hours_per_job=8,
    cpus_per_job=8, memory_gib_per_job=120, total_gpu_hours=64, max_concurrent_jobs=8)


def prepare():
    # The old, unlaunched plan remains an immutable provenance record.
    old = base.read(base.CONTROL/'launch-plan.json')
    validate_source(base.SOURCE)
    if base.sha(base.COHORT/'weighted-tasks.jsonl') != base.TASK_SHA:
        raise ValueError('Selected task cohort changed')
    if base.read(base.POOL/'cohorts-20260930/approved-selection.json')['selection'] != base.COHORT.name:
        raise ValueError('User selection changed')
    ids = old['screen_config']['task_order']
    shards = partition(ids)
    rows = [json.loads(line) for line in (base.SOURCE/'screen-tasks.jsonl').read_text().splitlines()]
    if [str(r['metadata']['task_id']) for r in rows] != ids:
        raise ValueError('Old screen task order mismatch')
    if not SOURCE.exists():
        shutil.copytree(base.SOURCE, SOURCE, symlinks=True, copy_function=base.copy_plain,
            ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git', '.browser_use_sessions'))
        changed = ['openwebrl/arm_task_screen.py', 'openwebrl/arm_task_screen_sharded.py',
                   'openwebrl/env/local_process_env.py']
        for name in changed:
            base.copy_plain(REPO/name, SOURCE/name)
        launcher = 'scripts/run_h200_browser.sh'
        text = (SOURCE/launcher).read_text()
        if text.count(' --sequence-parallel') != 1:
            raise ValueError('Unexpected TP1 launcher anchor')
        (SOURCE/launcher).write_text(text.replace(' --sequence-parallel', ''))
        changed.append(launcher)
        for index in range(8):
            taskname = f'screen-shard-{index}.jsonl'
            (SOURCE/taskname).write_text(''.join(json.dumps(r)+'\n' for r in rows[index::8]))
            cfg = copy.deepcopy(base.read(base.SOURCE/'screen-eval.json'))
            cfg['eval']['datasets'][0].update(path=str(SOURCE/taskname),
                custom_generate_function_path='openwebrl.arm_task_screen_sharded.generate')
            evalname = f'screen-shard-{index}-eval.json'
            write_json(SOURCE/evalname, cfg)
            changed.extend([taskname, evalname])
        browser = base.read(base.SOURCE/'screen-browser.json')
        browser['browser_rollout_concurrency'] = 8
        write_json(SOURCE/'screen-sharded-browser.json', browser)
        write_json(SOURCE/'screen-partition.json', dict(task_order=ids, shards=shards,
            selected_native_sha256=base.TASK_SHA, primary_attempts=10000,
            seed_policy='task/attempt seed independent of shard and job; unchanged'))
        changed.extend(['screen-sharded-browser.json', 'screen-partition.json'])
        manifest = base.read(SOURCE/'reference_manifest.json')
        manifest['recipe_files_sha256'].update({name:base.sha(SOURCE/name) for name in changed})
        manifest['single_gpu_evaluation'] = dict(sequence_parallel=False, tensor_parallel_size=1)
        manifest['sharded_task_screen'] = dict(parent=str(base.SOURCE), changed_files=changed,
            optimizer_updates=0, unchanged_scientific_protocol=True)
        write_json(SOURCE/'reference_manifest.json', manifest)
    validate_source(SOURCE)
    for name in ['openwebrl/arm_task_screen.py', 'openwebrl/arm_task_screen_sharded.py',
                 'openwebrl/env/local_process_env.py']:
        if base.sha(SOURCE/name) != base.sha(REPO/name):
            raise ValueError('Freeze a new source version after source edits')
    frozen = base.read(SOURCE/'screen-partition.json')
    validate_partition(frozen['shards'], ids)
    if frozen['task_order'] != ids:
        raise ValueError('Global task order changed')
    CONTROL.mkdir(parents=True, exist_ok=True)
    return old


def plan(shard, job='PREPARE'):
    if shard not in range(8):
        raise ValueError('Shard must be0..7')
    old = prepare()
    root = CONTROL/f'shard-{shard}'
    attempt = root/'workers'/str(job)
    def rewrite(v):
        if isinstance(v, str):
            return v.replace(old['stage']['output'],str(attempt)).replace(str(base.SOURCE),str(SOURCE))
        if isinstance(v, list): return [rewrite(x) for x in v]
        if isinstance(v, dict): return {k:rewrite(x) for k,x in v.items()}
        return v
    stage = rewrite(copy.deepcopy(old['stage']))
    stage.update(job_id=str(job), output=str(attempt), source=str(SOURCE), gpus=1,
        wandb_run_id=f'arm-task-screen-shard{shard}-{job}')
    for flag,value in {'--eval-config':str(SOURCE/f'screen-shard-{shard}-eval.json'),
        '--eval-function-path':'openwebrl.arm_task_screen_sharded.generate_rollout'}.items():
        stage['command'][stage['command'].index(flag)+1] = value
    stage['environment'].update(NUM_GPUS='1', TP_SIZE='1', BROWSER_CONCURRENCY='8', SGLANG_CONCURRENCY='8',
        BROWSER_TRAIN_CONFIG=str(SOURCE/'screen-sharded-browser.json'),
        OPENWEBRL_MULTIMODAL_STORAGE_DIR=str(root/'multimodal'/str(job)),
        OPENWEBRL_TASK_SCREEN_CONFIG=str(attempt/'screen-config.json'),
        WANDB_RUN_ID=stage['wandb_run_id'], RAY_TMPDIR=f'/tmp/ts-{job}-{shard}',
        SLIME_BROWSER_LOCAL_PROCESS_LOG_DIR=str(root/'browser-logs'/str(job)))
    stage = base.configure_evaluation_tracking(stage, 'openwebrl-evals')
    cfg = copy.deepcopy(old['screen_config'])
    cfg.update(output=str(root), job_id=str(job), shard=shard,
        tasks=str(SOURCE/f'screen-shard-{shard}.jsonl'), task_order=partition(cfg['task_order'])[shard],
        partition_manifest=str(SOURCE/'screen-partition.json'),
        attempt_output=str(root/'attempts'/str(job)/str(time.time_ns())))
    return dict(shard=shard, job_id=str(job), source=str(SOURCE), output=str(root), stage=stage,
        screen_config=cfg, resources=RESOURCES, actor_manifest=old['actor_manifest'],
        primary_tasks=250, primary_attempts=1250, selected_native_sha256=base.TASK_SHA,
        judge_max_usd=18.75, judge_max_calls=3750, browser_concurrency=8,
        storage_reserve_bytes=5*1024**4, optimizer_updates=0)


def fingerprint():
    return {str(p):base.sha(p) for p in [SOURCE/'reference_manifest.json',
        REPO/'scripts/run_arm_task_screen_shards.py', REPO/'scripts/run_arm_task_screen_1gpu.sbatch']}


def native_check(p):
    env = dict(clean_environment(), **p['stage']['environment'])
    env.update(DRY_RUN='1', CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    argv = shlex.split(subprocess.check_output(p['stage']['command'],env=env,text=True))
    env.pop('DRY_RUN')
    code = """from unittest.mock import patch
from slime.utils.arguments import parse_args
with patch('megatron.training.arguments.get_device_arch_version',return_value=9): a=parse_args()
expected={'num_rollout':0,'tensor_model_parallel_size':1,'actor_num_gpus_per_node':1,
'rollout_num_gpus_per_engine':1,'sequence_parallel':False,
'browser_rollout_concurrency':8,'judge_api_model':'gpt-4.1','judge_prompt_variant':'action_history',
'eval_function_path':'openwebrl.arm_task_screen_sharded.generate_rollout',
'use_checkpoint_opt_param_scheduler':False,'max_steps':15,
'context_num_screenshots':1,'turn_history_reasoning_mode':'full'}
assert all(getattr(a,k)==v for k,v in expected.items()),{k:getattr(a,k) for k in expected}
assert a.load.endswith('/OpenWebRL-4B-SFT')
d=a.eval_datasets[0]
assert (d.n_samples_per_eval_prompt,d.temperature,d.top_p,d.top_k,d.max_response_len)==(5,.8,1.,-1,1024)
print('TASK_SCREEN_SHARD_NATIVE_PARSE_OK')
"""
    with (CONTROL/'native-parse.log').open('w') as out:
        subprocess.run(source_command(SOURCE,[str(RUNTIME/'venv/bin/python'),'-c',code,*argv[2:]]),
            env=env,cwd=SOURCE,stdout=out,stderr=subprocess.STDOUT,check=True,timeout=120)
    write_json(CONTROL/'readiness.json',dict(native_parse_passed=True,gpu_validation_pending=True,
        fingerprint=fingerprint(), selected_native_sha256=base.TASK_SHA))


def verify_approval(approval, job, shard, previous_seconds):
    if (not approval.get('approved') or approval.get('resources') != RESOURCES or
            approval.get('judge_max_usd') != 150. or approval.get('judge_max_calls') != 30000 or
            approval.get('judge_id') != 'gpt-4.1/action_history'):
        raise ValueError('Exact compute/judge budget and protocol approval required')
    attempts = approval.get('attempts',[])
    if len({r['job_id'] for r in attempts}) != len(attempts):
        raise ValueError('Duplicate job accounting')
    current = [r for r in attempts if r['job_id'] == job and r['shard'] == shard]
    if len(current) != 1:
        raise ValueError('Register the actual job/shard before execution')
    if previous_seconds + current[0]['time_limit_seconds'] > 8*3600:
        raise ValueError('Retry would exceed this shard\'s original8h approval')
    return current[0]


def accounted_seconds(approval, shard, job):
    ids = [r['job_id'] for r in approval['attempts'] if r['shard']==shard and r['job_id']!=job]
    if not ids: return 0
    if any(not i.isdecimal() for i in ids): raise ValueError('Invalid job ID')
    raw = subprocess.check_output(['sacct','-X','-n','-P','-j',','.join(ids),
        '--format=JobID,State,ElapsedRaw'],text=True)
    rows = {v[0]:v for line in raw.splitlines() if len(v:=line.split('|'))>=3}
    if any(i not in rows or rows[i][1] in ('RUNNING','PENDING','CONFIGURING','COMPLETING') for i in ids):
        raise ValueError('Every previous attempt must have final accounting')
    return sum(int(rows[i][2]) for i in ids)


def worker(p):
    s = p['stage']; root = Path(s['output']); root.mkdir(parents=True,exist_ok=True)
    env = environment(); env.update(s['environment'])
    devices = env.get('CUDA_VISIBLE_DEVICES','').split(',')
    if len(devices)!=1 or not devices[0]: raise ValueError('Requires one assigned GPU')
    write_json(root/'screen-config.json',p['screen_config'])
    write_json(root/'evaluation_manifest.json',s)
    Path(env['OPENWEBRL_MULTIMODAL_STORAGE_DIR']).mkdir(parents=True,exist_ok=True)
    lease, port = lease_ports(p['job_id']); env[PORT_ENV] = str(port)
    child = None
    try:
        with (root/'evaluation.log').open('a') as log:
            child = subprocess.Popen(source_command(SOURCE,s['command']),env=env,cwd=SOURCE,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            if child.wait(): raise RuntimeError('Screen worker failed; inspect before retry')
    finally:
        stop(child); lease.close()


def merge(require_complete=False):
    manifest = base.read(SOURCE/'screen-partition.json')
    validate_partition(manifest['shards'], manifest['task_order'])
    records = [base.read(p) for i in range(8) for p in (CONTROL/f'shard-{i}'/'records').glob('*.json')]
    result = summarize(records,manifest['task_order'],require_complete=require_complete)
    write_json(CONTROL/'aggregate-results.json',result)
    return result


def execute(p):
    job, shard = p['job_id'], p['shard']
    if os.environ.get('SLURM_JOB_ID') != job:
        raise ValueError('Run inside the approved allocation')
    approval = base.read(CONTROL/'approval.json')
    used = accounted_seconds(approval,shard,job)
    current = verify_approval(approval,job,shard,used)
    res = allocation(subprocess.check_output(['scontrol','show','job',job,'-o'],text=True),
        job,requested_gpus=1,maximum_hours=current['time_limit_seconds']/3600)
    if res['cpus']<8 or res['allocated_memory_gib']<120:
        raise ValueError('Wrong CPU/memory profile')
    ready = base.read(CONTROL/'readiness.json')
    if not ready['native_parse_passed'] or ready['fingerprint'] != fingerprint():
        raise ValueError('Run native preflight after source edits')
    check_storage_quota(p['storage_reserve_bytes'])
    root = Path(p['output']); root.mkdir(parents=True,exist_ok=True)
    with (root/'owner.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write_json(root/f'plan-{job}.json',p)
        write_json(root/'launch_manifest.json',p['stage'])
        progress = root/'runtime/progress.log'
        progress.parent.mkdir(parents=True,exist_ok=True)
        progress.write_text(f'[TaskScreenProgress] job={job} stage=startup\n')
        last_count = -1
        child = None
        deadline = time.monotonic()+res['maximum_seconds']-120
        try:
            child = subprocess.Popen(['srun',f'--jobid={job}','--nodes=1','--ntasks=1',
                '--cpus-per-task=8','--gres=gpu:h200:1','--mem=120G','--exact','--cpu-bind=none',
                sys.executable,str(Path(__file__).resolve()),'--job-id',job,'--shard',str(shard),'--worker'],
                start_new_session=True)
            while child.poll() is None:
                if time.monotonic()>deadline: raise TimeoutError('Allocation ending; preserve durable records')
                paths = list((root/'records').glob('*.json'))
                if len(paths) != last_count:
                    with progress.open('a') as log:
                        log.write(f'[TaskScreenProgress] epoch={time.time()} attempts={len(paths)}/1250\n')
                    last_count = len(paths)
                write_json(root/'status.json',dict(job_id=job,shard=shard,complete=False,stage='collection',
                    epoch=time.time(),primary_attempts=len(paths),
                    last_record_epoch=max((p.stat().st_mtime for p in paths),default=None)))
                time.sleep(30)
            if child.returncode: raise RuntimeError('Screen worker failed; needs diagnosis')
            result = summarize([base.read(p) for p in (root/'records').glob('*.json')],
                p['screen_config']['task_order'],require_complete=True)
            write_json(root/'verification.json',dict(complete=True,archives_and_verdicts_checked=True,
                primary_attempts=result['primary_attempts'],selected_native_sha256=base.TASK_SHA))
            write_json(root/'status.json',dict(job_id=job,shard=shard,complete=True,
                stage='verified_complete',epoch=time.time()))
            # A cross-shard lock avoids concurrent aggregate writers.
            with (CONTROL/'aggregate.lock').open('a+') as aggregate_lock:
                fcntl.flock(aggregate_lock,fcntl.LOCK_EX)
                merge()
        except BaseException as e:
            write_json(root/'status.json',dict(job_id=job,shard=shard,complete=False,failed=True,epoch=time.time(),
                error_type=type(e).__name__,error=str(e)[:500]))
            raise
        finally:
            stop(child)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-id',default='PREPARE')
    parser.add_argument('--shard',type=int,choices=range(8),default=0)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--merge',action='store_true')
    args = parser.parse_args()
    if args.worker:
        validate_source(SOURCE)
        worker(base.read(CONTROL/f'shard-{args.shard}'/f'plan-{args.job_id}.json')); return
    if args.merge:
        result = merge(); print(json.dumps({k:v for k,v in result.items() if k!='dispositions'},indent=2)); return
    if args.execute:
        execute(plan(args.shard,args.job_id)); return
    plans = [plan(i) for i in range(8)]
    native_check(plans[0])
    for p in plans:
        write_json(CONTROL/f'launch-shard-{p["shard"]}.json',p)
    write_json(CONTROL/'proposal.json',dict(resources=RESOURCES,judge_max_usd=150.,judge_max_calls=30000,
        judge_id='gpt-4.1/action_history',primary_tasks=2000,primary_attempts=10000,
        status='prepared_pending_exact_budget_and_judge_confirmation',approved=False))
    print(json.dumps(base.read(CONTROL/'proposal.json'),indent=2))


if __name__ == '__main__':
    main()
