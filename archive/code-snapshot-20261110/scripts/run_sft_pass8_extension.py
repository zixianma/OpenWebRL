#!/usr/bin/env python3
"""Prepare three more SFT attempts on all 707 historical pass@5 misses.

Preparation and native parsing are CPU-only. This script never submits Slurm jobs.
The original five attempts are immutable inputs; approval is bound to the prepared
plan, and each shard charges every scheduler attempt against its original cap.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
RUNTIME = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime')
ROOT = RUNTIME/'evaluations/sft-training-pass8-20261006/extension3-misses-v1'
PRIOR_PREPARATION = RUNTIME/'evaluations/sft-training-pass8-20261006/extension3-v1'
PARENT = RUNTIME/'reference-arm-task-screen-sharded-20261002-v2'
SOURCE = RUNTIME/'reference-sft-pass8-misses-20261006-v1'
OLD = RUNTIME/'arm-turn-bonus-preparation/task-pool-expansion-20260922/curation-v3-20260929/actor-screen-min5-2000'
ACTOR = Path('/gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT')
REVISION = '15e777db2ddba2e0e82080ebccd3ad8d215b7f0a'
POLICY = 'OpenWebRL-4B-SFT:iteration0'
RESOURCES = dict(jobs=4, gpus_per_job=1, gpu_type='H200', cpus_per_job=8,
                 memory_gib_per_job=120, total_seconds_per_shard=36000,
                 total_gpu_hours=40, max_concurrent_jobs=4)
LIMITS = dict(browsers_per_shard=1600, concurrent_browsers_per_shard=8,
              judge_http_calls_per_shard=4800, judge_usd_per_shard=12.5,
              browsers_total=6400, judge_http_calls_total=19200, judge_usd_total=50.,
              judge_id='gpt-4.1/action_history')
FINAL = {'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL',
         'BOOT_FAIL','DEADLINE','REVOKED','PREEMPTED'}
ADDITIONS = ('openwebrl/arm_task_pass8.py', 'scripts/run_sft_pass8_extension.py',
             'scripts/run_sft_pass8_extension_1gpu.sbatch')


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    with temp.open('w') as stream:
        json.dump(value, stream, indent=2); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    temp.replace(path)


def freeze(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError('Refuse to overwrite frozen input: '+str(path))
    else:
        write(path, value)


def partition(ids):
    if len(ids) != 707 or len(set(ids)) != 707:
        raise ValueError('Exactly 707 unique historical zero-success task IDs required')
    return [ids[i::4] for i in range(4)]


def select_misses(historical_ids, records):
    """Select every zero-success task, retaining misses with invalid attempts."""
    if len(historical_ids)!=2000 or len(set(historical_ids))!=2000:
        raise ValueError('Historical coverage denominator must remain exactly 2,000 unique tasks')
    allowed={(task,index) for task in historical_ids for index in range(5)}
    indexed={}
    for record in records:
        key=(record['task_id'],record['attempt'])
        if key not in allowed or key in indexed or type(record.get('valid')) is not bool:
            raise ValueError('Missing, repeated or malformed historical attempt identity')
        if record['valid']:
            if record['reward'] not in (-1.,0.,1.):raise ValueError('Unsupported historical reward')
        elif record['reward'] is not None:
            raise ValueError('Invalid historical attempt must not supply a task outcome')
        indexed[key]=record
    if set(indexed)!=allowed:raise ValueError('All 10,000 preserved historical attempts are required')
    misses=[];valid_misses=[];invalid_misses=[];any_invalid=[]
    for task in historical_ids:
        group=[indexed[task,index] for index in range(5)]
        has_invalid=any(not r['valid'] for r in group)
        if has_invalid:any_invalid.append(task)
        if not any(r['valid'] and r['reward']==1. for r in group):
            misses.append(task)
            (invalid_misses if has_invalid else valid_misses).append(task)
    counts=dict(historical_tasks=2000,historical_primary_attempts=10000,
        historical_successful_tasks=2000-len(misses),extension_tasks=len(misses),
        five_valid_failure_tasks=len(valid_misses),zero_success_with_invalid_tasks=len(invalid_misses),
        historical_tasks_with_any_invalid=len(any_invalid))
    if counts!=dict(historical_tasks=2000,historical_primary_attempts=10000,
            historical_successful_tasks=1293,extension_tasks=707,five_valid_failure_tasks=682,
            zero_success_with_invalid_tasks=25,historical_tasks_with_any_invalid=44):
        raise ValueError('Historical coverage or invalid strata differ from the immutable audit')
    return misses,dict(counts,selection='All historical tasks with no valid positive outcome in attempts 0..4',
        extension_task_order_sha256=digest(misses),new_attempt_indices=[5,6,7],
        new_primary_episodes=2121,successful_tasks_replayed=0)


def validate_launch_shells(source):
    """Reject destructive or broad process cleanup in the actual wrapper chain."""
    for name in ('scripts/h200_env.sh', 'scripts/run_h200_browser.sh',
                 'scripts/run_sft_pass8_extension_1gpu.sbatch'):
        text = (Path(source)/name).read_text()
        if re.search(r'(^|[;\n]|&&|\|\|)\s*(?:sudo\s+)?(?:rm|pkill|killall)\b', text):
            raise ValueError('Destructive or unscoped cleanup requires exact approval: '+name)
        if 'ray stop' in text or 'git reset --hard' in text or 'git clean ' in text:
            raise ValueError('Unsafe launch wrapper: '+name)


def prepare():
    from resume_baseline import validate_source
    if (ROOT/'plan.json').exists():
        return validate_prepared()
    validate_source(PARENT)
    if not all((REPO/name).is_file() for name in ADDITIONS):
        raise ValueError('Collector and controller must exist before freezing preparation')
    original = read(OLD/'launch-plan.json')
    historical_ids = original['screen_config']['task_order']
    rows = [json.loads(line) for line in (PARENT/'screen-tasks.jsonl').read_text().splitlines() if line.strip()]
    if [str(row['metadata']['task_id']) for row in rows] != historical_ids:
        raise ValueError('Historical task order disagrees with frozen task rows')
    old_aggregate = read(OLD/'sharded-20261002/aggregate-results.json')
    if old_aggregate.get('complete') is not True or old_aggregate.get('primary_attempts') != 10000:
        raise ValueError('Reuse requires a complete original five-attempt collection')
    historical_audit=read(ROOT/'historical-audit.json')
    if historical_audit.get('verified_historical_records') != 10000 or historical_audit.get('old_records_mutated') is not False:
        raise ValueError('Independent immutable-history audit is required before preparation')
    prior_approval=PRIOR_PREPARATION/'approval.json'
    if prior_approval.exists() and read(prior_approval).get('attempts'):
        raise ValueError('Prior scope has scheduler attempts; reconcile common campaign charges before narrowing')
    historical_records=[]
    for proof in read(ROOT/'historical-inputs.json'):
        path=Path(proof['record'])
        if sha(path)!=proof['record_sha256']:
            raise ValueError('Historical record changed before selecting zero-success tasks')
        record=read(path)
        if (record['task_id'],record['attempt'])!=(proof['task_id'],proof['attempt']):
            raise ValueError('Historical manifest identity mismatch')
        historical_records.append(record)
    ids,selection=select_misses(historical_ids,historical_records);shards=partition(ids)
    by_id={str(row['metadata']['task_id']):row for row in rows}
    extension_rows=[by_id[task] for task in ids]
    actor = read(ACTOR/'download_manifest.json')
    if actor.get('repo_id') != 'OpenWebRL/OpenWebRL-4B-SFT' or actor.get('revision') != REVISION:
        raise ValueError('Official SFT checkpoint identity mismatch')
    actor_proof=read(ROOT/'parent-model-hash-check.json')
    if (actor_proof.get('model_bytes_match_download_manifest') is not True
            or actor_proof.get('model_sha256')!=actor.get('model_sha256')
            or actor_proof.get('revision')!=REVISION):
        raise ValueError('Original official checkpoint byte-hash audit is required')
    scope=read(ROOT/'scope-authorization.json')
    if (scope.get('experiment_scope_authorized') is not True
            or scope.get('scope_choice_pending') is not False
            or scope.get('exact_resource_approval_pending') is not False):
        raise ValueError('The narrower task scope and inherited resource ceilings must be authorized')
    if not SOURCE.exists():
        # The preserved parent is 29 MB of code/configs, not a live checkout or
        # trajectory tree. Copy bytes rather than hard-linking mutable files.
        shutil.copytree(PARENT, SOURCE, symlinks=True,
                        ignore=shutil.ignore_patterns('__pycache__','*.pyc','.git','.browser_use_sessions'))
    if (SOURCE/'pass8-partition.json').exists():
        raise ValueError('Partial frozen preparation exists; inspect and preserve it before recovery')
    changed = list(ADDITIONS)
    for name in ADDITIONS:
        shutil.copyfile(REPO/name, SOURCE/name)
    # Freeze controller support code with the same source as the controller.
    for name in ('scripts/resume_baseline.py','scripts/runtime_ports.py','scripts/check_storage_quota.py'):
        shutil.copyfile(REPO/name, SOURCE/name); changed.append(name)
    partition_manifest = dict(task_order=ids,extension_task_order=ids,historical_task_order=historical_ids,
        shards=shards,num_shards=4,selection=selection,
        historical_attempts=[0,1,2,3,4], new_attempts=[5,6,7],
        seed_policy='Unchanged task/attempt seed function; new global indices 5,6,7',
        historical_task_sha256=sha(PARENT/'screen-tasks.jsonl'))
    write(SOURCE/'pass8-partition.json', partition_manifest); changed.append('pass8-partition.json')
    for i in range(4):
        task = f'pass8-shard-{i}.jsonl'; evaluation = f'pass8-shard-{i}-eval.json'
        (SOURCE/task).write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in extension_rows[i::4]))
        config = copy.deepcopy(read(PARENT/'screen-eval.json'))
        config['eval']['defaults'].update(n_samples_per_eval_prompt=3)
        config['eval']['datasets'][0].update(name=f'sft-pass8-misses-shard{i}',
            path=str(SOURCE/task), custom_generate_function_path='openwebrl.arm_task_pass8.generate')
        write(SOURCE/evaluation, config); changed.extend((task,evaluation))
    manifest = read(SOURCE/'reference_manifest.json')
    manifest['recipe_files_sha256'].update({name:sha(SOURCE/name) for name in changed})
    manifest['pass8_extension'] = dict(parent=str(PARENT),parent_manifest_sha256=sha(PARENT/'reference_manifest.json'),
        changed_files=changed, optimizer_updates=0, actor_revision=REVISION,
        scientific_protocol_unchanged=True,historical_records_read_only=True,
        extension_scope='All 707 historical zero-success tasks, including 25 with invalid attempts',
        historical_tasks=2000,new_primary_episodes=2121)
    write(SOURCE/'reference_manifest.json', manifest)
    validate_source(SOURCE); validate_launch_shells(SOURCE)
    actor_files = {}
    for path in ACTOR.iterdir():
        if path.is_file():
            st=path.stat(); actor_files[path.name] = dict(size=st.st_size,mtime_ns=st.st_mtime_ns)
    plan = dict(root=str(ROOT),source=str(SOURCE),parent_source=str(PARENT),
        source_manifest_sha256=sha(SOURCE/'reference_manifest.json'),
        controller_sha256=sha(SOURCE/'scripts/run_sft_pass8_extension.py'),
        actor_manifest=actor,actor_files=actor_files,checkpoint=str(ACTOR),
        original_plan=str(OLD/'launch-plan.json'),original_plan_sha256=sha(OLD/'launch-plan.json'),
        original_aggregate=str(OLD/'sharded-20261002/aggregate-results.json'),
        original_aggregate_sha256=sha(OLD/'sharded-20261002/aggregate-results.json'),
        historical_inputs=str(ROOT/'historical-inputs.json'),
        historical_inputs_sha256=sha(ROOT/'historical-inputs.json'),
        historical_audit=str(ROOT/'historical-audit.json'),
        historical_audit_sha256=sha(ROOT/'historical-audit.json'),
        actor_hash_proof=str(ROOT/'parent-model-hash-check.json'),
        actor_hash_proof_sha256=sha(ROOT/'parent-model-hash-check.json'),
        scope_authorization=str(ROOT/'scope-authorization.json'),
        scope_authorization_sha256=sha(ROOT/'scope-authorization.json'),
        old_record_roots=[str(OLD/f'sharded-20261002/shard-{i}/records') for i in range(8)],
        historical_task_order=historical_ids,extension_task_order=ids,task_order=ids,
        shards=shards,resources=RESOURCES,limits=LIMITS,extension_selection=selection,
        new_primary_episodes=2121,new_episodes_per_shard=[len(group)*3 for group in shards],optimizer_updates=0,
        prior_preparation_root=str(PRIOR_PREPARATION),budget_campaign_id='sft-training-pass8-20261006',
        new_budget_added=False,unused_prior_approvals_added=False,
        protocol=dict(temperature=.8,top_p=1.,top_k=-1,max_response_tokens=1024,max_steps=15,
            seed=20260930,policy_id=POLICY,judge_id='gpt-4.1/action_history',browser='local_process',
            images=True,selector=None,checkpoint_revision=REVISION),
        status='prepared_pending_exact_compute_and_api_budget_approval',approved=False)
    freeze(ROOT/'plan.json',plan)
    freeze(ROOT/'approval-proposal.json',dict(approved=False,plan_sha256=digest(plan),
        resources=RESOURCES,limits=LIMITS,attempts=[]))
    for i in range(4):
        freeze(ROOT/f'launch-shard-{i}.json',make_launch(plan,i,'PREPARE'))
    return plan


def validate_prepared():
    from resume_baseline import validate_source
    plan = read(ROOT/'plan.json')
    if (plan['root'] != str(ROOT) or plan['source'] != str(SOURCE)
            or plan['resources'] != RESOURCES or plan['limits'] != LIMITS
            or partition(plan['extension_task_order']) != plan['shards']
            or plan['task_order']!=plan['extension_task_order']
            or len(plan['historical_task_order'])!=2000 or len(set(plan['historical_task_order']))!=2000
            or not set(plan['extension_task_order']).issubset(plan['historical_task_order'])
            or [task for task in plan['historical_task_order'] if task in set(plan['extension_task_order'])]!=plan['extension_task_order']
            or plan['new_primary_episodes']!=2121 or plan['new_episodes_per_shard']!=[531,531,531,528]):
        raise ValueError('Prepared plan identity or budget drift')
    for path, expected in ((SOURCE/'reference_manifest.json',plan['source_manifest_sha256']),
        (SOURCE/'scripts/run_sft_pass8_extension.py',plan['controller_sha256']),
        (Path(plan['original_plan']),plan['original_plan_sha256']),
        (Path(plan['original_aggregate']),plan['original_aggregate_sha256']),
        (Path(plan['historical_inputs']),plan['historical_inputs_sha256']),
        (Path(plan['historical_audit']),plan['historical_audit_sha256']),
        (Path(plan['actor_hash_proof']),plan['actor_hash_proof_sha256']),
        (Path(plan['scope_authorization']),plan['scope_authorization_sha256'])):
        if sha(path) != expected: raise ValueError('Frozen input changed: '+str(path))
    validate_source(SOURCE); validate_launch_shells(SOURCE)
    for name,expected in plan['actor_files'].items():
        st=(ACTOR/name).stat()
        if dict(size=st.st_size,mtime_ns=st.st_mtime_ns) != expected:
            raise ValueError('Official checkpoint file identity changed: '+name)
    if read(ACTOR/'download_manifest.json') != plan['actor_manifest']:
        raise ValueError('Official checkpoint revision proof changed')
    return plan


def make_launch(plan, shard, job):
    if shard not in range(4) or (job != 'PREPARE' and not str(job).isdecimal()):
        raise ValueError('Invalid shard/job identity')
    original = read(plan['original_plan']); root = ROOT/f'shard-{shard}'
    attempt = root/'workers'/str(job)
    def replace(value):
        if isinstance(value,str):
            return value.replace(original['stage']['output'],str(attempt)).replace(original['source'],str(SOURCE))
        if isinstance(value,list):return [replace(v) for v in value]
        if isinstance(value,dict):return {k:replace(v) for k,v in value.items()}
        return value
    stage=replace(copy.deepcopy(original['stage']))
    stage.update(job_id=str(job),output=str(attempt),source=str(SOURCE),gpus=1,
        parent_training_run=None,wandb_run_id=f'sft-pass8-misses-shard{shard}',
        metric_prefix='eval/training-task-pass8-misses',
        protocol='Historical SFT protocol; new global attempts 5,6,7 only; no training')
    for flag,value in {'--eval-config':str(SOURCE/f'pass8-shard-{shard}-eval.json'),
        '--eval-function-path':'openwebrl.arm_task_pass8.generate_rollout',
        '--wandb-group':'SFT-training-pool-pass8-misses-20261006',
        '--wandb-project':'openwebrl-evals'}.items():
        stage['command'][stage['command'].index(flag)+1]=value
    env=stage['environment']; env.pop('OPENWEBRL_TASK_SCREEN_CONFIG',None)
    env.update(NUM_GPUS='1',TP_SIZE='1',NUM_ROLLOUT='0',N_SAMPLES='3',
        BROWSER_CONCURRENCY='8',SGLANG_CONCURRENCY='8',WANDB_PROJECT='openwebrl-evals',
        WANDB_RUN_ID=stage['wandb_run_id'],WANDB_RESUME='allow',
        BROWSER_TRAIN_CONFIG=str(SOURCE/'screen-sharded-browser.json'),
        OPENWEBRL_TASK_PASS8_CONFIG=str(attempt/'collector-config.json'),
        OPENWEBRL_MULTIMODAL_STORAGE_DIR=str(root/'multimodal'/str(job)),
        SLIME_BROWSER_LOCAL_PROCESS_LOG_DIR=str(root/'browser-logs'/str(job)),
        RAY_TMPDIR=f'/tmp/pass8-{job}-{shard}')
    stage['wandb_url']=f'https://wandb.ai/zixianma/openwebrl-evals/runs/{stage["wandb_run_id"]}'
    cfg=dict(output=str(root),attempt_output=str(root/'attempts'/str(job)),
        job_id=str(job),tasks=str(SOURCE/f'pass8-shard-{shard}.jsonl'),task_order=plan['shards'][shard],
        historical_task_order=plan['historical_task_order'],extension_task_order=plan['extension_task_order'],
        partition_manifest=str(SOURCE/'pass8-partition.json'),shard=shard,seed=20260930,
        policy_id=POLICY,checkpoint=str(ACTOR),max_steps=15,inference_timeout=180,
        browser_cap=1600,judge_max_calls=4800,judge_max_usd=12.5,
        old_record_roots=plan['old_record_roots'])
    if plan.get('historical_inputs'):
        cfg.update(historical_inputs=plan['historical_inputs'],
                   historical_inputs_sha256=plan['historical_inputs_sha256'])
    return dict(plan_sha256=digest(plan),job_id=str(job),shard=shard,output=str(root),
                stage=stage,collector_config=cfg)


def native_check(plan):
    from resume_baseline import clean_environment,source_command
    launch=make_launch(plan,0,'PREPARE');stage=launch['stage']
    env=dict(clean_environment(),**stage['environment'])
    env.update(DRY_RUN='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    argv=shlex.split(subprocess.check_output(stage['command'],env=env,text=True,cwd=SOURCE))
    env.pop('DRY_RUN')
    code="""from unittest.mock import patch
from slime.utils.arguments import parse_args
with patch('megatron.training.arguments.get_device_arch_version',return_value=9): a=parse_args()
expected={'num_rollout':0,'tensor_model_parallel_size':1,'actor_num_gpus_per_node':1,
'rollout_num_gpus_per_engine':1,'sequence_parallel':False,'browser_rollout_concurrency':8,
'judge_api_model':'gpt-4.1','judge_prompt_variant':'action_history','max_steps':15,
'context_num_screenshots':1,'turn_history_reasoning_mode':'full',
'eval_function_path':'openwebrl.arm_task_pass8.generate_rollout','use_checkpoint_opt_param_scheduler':False,
'wandb_project':'openwebrl-evals'}
assert all(getattr(a,k)==v for k,v in expected.items()),{k:getattr(a,k) for k in expected}
assert a.load=='/gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT'
d=a.eval_datasets[0]
assert (d.n_samples_per_eval_prompt,d.temperature,d.top_p,d.top_k,d.max_response_len)==(3,.8,1.,-1,1024)
with open(d.path) as f: assert sum(bool(line.strip()) for line in f)==177
print('SFT_PASS8_EXTENSION_NATIVE_PARSE_OK')
"""
    log=ROOT/'native-parse.log'
    with log.open('w') as stream:
        subprocess.run(source_command(SOURCE,[str(RUNTIME/'venv/bin/python'),'-c',code,*argv[2:]]),
            cwd=SOURCE,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=180)
    write(ROOT/'readiness.json',dict(plan_sha256=digest(plan),native_parse_passed=True,
        native_parse_log_sha256=sha(log),gpu_validation_pending=True,
        no_gpu_no_browser_no_api_calls=True,no_job_submitted=True))


def verify_approval(plan, approval, job, shard, previous_seconds):
    if (approval.get('approved') is not True or approval.get('plan_sha256') != digest(plan)
            or approval.get('resources') != RESOURCES or approval.get('limits') != LIMITS):
        raise ValueError('Exact compute, API and browser approval required for this prepared plan')
    attempts=approval.get('attempts',[])
    if len({str(row['job_id']) for row in attempts}) != len(attempts):
        raise ValueError('Duplicate scheduler attempt')
    for row in attempts:
        if (not str(row['job_id']).isdecimal() or type(row.get('shard')) is not int
                or row['shard'] not in range(4) or type(row.get('time_limit_seconds')) is not int
                or not 0 < row['time_limit_seconds'] <= RESOURCES['total_seconds_per_shard']):
            raise ValueError('Malformed attempt registration')
    current=[row for row in attempts if str(row['job_id'])==str(job) and row['shard']==shard]
    if len(current)!=1:raise ValueError('Actual job/shard must be registered before execution')
    seconds=current[0]['time_limit_seconds']
    if type(previous_seconds) is not int or previous_seconds<0 or previous_seconds+seconds>36000:
        raise ValueError('Retry exceeds the original all-attempt shard budget')
    return seconds


def accounted_seconds(approval, shard, job, query=None):
    ids=[str(row['job_id']) for row in approval.get('attempts',[]) if row['shard']==shard and str(row['job_id'])!=str(job)]
    if not ids:return 0,[]
    if len(set(ids))!=len(ids) or any(not value.isdecimal() for value in ids):
        raise ValueError('Malformed prior scheduler identities')
    query=query or subprocess.check_output
    raw=query(['sacct','-X','-n','-P','-j',','.join(ids),
               '--format=JobIDRaw,State,ElapsedRaw,AllocTRES'],text=True)
    rows={}
    for line in raw.splitlines():
        cols=line.split('|')
        if cols[0] in ids:
            if cols[0] in rows:raise ValueError('Duplicate scheduler accounting row')
            rows[cols[0]]=cols
    proof=[]
    for identity in ids:
        row=rows.get(identity)
        if not row or len(row)<4 or row[1].split()[0].rstrip('+') not in FINAL:
            raise ValueError('Every previous shard attempt needs final scheduler accounting')
        seconds=int(row[2])
        if seconds<0:raise ValueError('Negative scheduler charge')
        proof.append(dict(job_id=identity,state=row[1],elapsed_seconds=seconds,alloc_tres=row[3]))
    return sum(row['elapsed_seconds'] for row in proof),proof


def validate_allocation(info, job, seconds, now=None, uid=None):
    fields=dict(re.findall(r'(\w+)=(\S+)',info));uid=os.getuid() if uid is None else uid
    tres=dict(value.split('=',1) for value in fields.get('AllocTRES','').split(',') if '=' in value)
    if (fields.get('JobId')!=str(job) or fields.get('JobState')!='RUNNING'
            or not fields.get('UserId','').endswith(f'({uid})') or fields.get('NumNodes')!='1'
            or fields.get('NumCPUs')!='8' or tres.get('gres/gpu')!='1' or tres.get('gres/gpu:h200')!='1'
            or tres.get('mem') not in ('120G','122880M')):
        raise ValueError('Exact owned 1 H200 / 8 CPU / 120 GiB allocation required')
    value=fields['TimeLimit'];days,clock=value.split('-',1) if '-' in value else ('0',value)
    parts=[int(x) for x in clock.split(':')]
    if len(parts)==2:parts.insert(0,0)
    duration=int(days)*86400+parts[0]*3600+parts[1]*60+parts[2]
    if duration>seconds:raise ValueError('Actual scheduler limit exceeds registered remaining budget')
    end=datetime.strptime(fields['EndTime'],'%Y-%m-%dT%H:%M:%S').timestamp()
    left=min(duration,int(end-(time.time() if now is None else now)))-180
    if left<300:raise ValueError('Insufficient remaining time including shutdown margin')
    return left


def process_identity(pid):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return (int(fields[2]),int(fields[3]),int(fields[19]))  # pgrp, session, birth ticks
    except FileNotFoundError:return None


def stop_owned(child, identity):
    if child is None or child.poll() is not None:return
    if identity is None or process_identity(child.pid)!=identity or identity[0:2]!=(child.pid,child.pid):
        raise RuntimeError('Child PID identity changed; refuse to signal an unrelated process')
    try:os.killpg(child.pid,signal.SIGTERM)
    except ProcessLookupError:
        child.wait(timeout=15);return
    try:child.wait(timeout=45)
    except subprocess.TimeoutExpired:
        if process_identity(child.pid)!=identity:raise RuntimeError('Child identity changed during shutdown')
        os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=15)


def worker(plan, launch):
    from resume_baseline import clean_environment,source_command
    from runtime_ports import PORT_ENV,lease_ports
    from dotenv import dotenv_values
    job=launch['job_id']; root=Path(launch['stage']['output'])
    if Path(__file__).resolve()!=SOURCE/'scripts/run_sft_pass8_extension.py':
        raise ValueError('Worker must use the exact frozen controller')
    if os.getenv('SLURM_JOB_ID')!=job:raise ValueError('Worker must run inside its registered allocation')
    devices=os.getenv('CUDA_VISIBLE_DEVICES','').split(',')
    if len(devices)!=1 or not devices[0]:raise ValueError('Worker needs exactly one assigned GPU')
    env=clean_environment()
    for key in list(env):
        if key.startswith(('OPENWEBRL_ARM_','OPENWEBRL_TASK_SCREEN_')):env.pop(key)
    for key,value in dotenv_values('/gpfs/projects/krishna/zixianma/OpenWebRL/.env').items():
        if value and key.startswith(('OPENAI_','JUDGE_','WANDB_','AZURE_')):env.setdefault(key,value)
    if env.get('OPENAI_API_KEY') and not env.get('JUDGE_API_BASE'):
        env.update(JUDGE_API_BASE='https://api.openai.com/v1',JUDGE_API_MODE='served')
    env.update(launch['stage']['environment'],WANDB_PROJECT='openwebrl-evals')
    root.mkdir(parents=True,exist_ok=True)
    write(root/'collector-config.json',launch['collector_config'])
    write(root/'evaluation_manifest.json',launch['stage'])
    Path(env['OPENWEBRL_MULTIMODAL_STORAGE_DIR']).mkdir(parents=True,exist_ok=True)
    child=None;identity=None;lease,port=lease_ports(job);env[PORT_ENV]=str(port)
    try:
        with (root/'evaluation.log').open('a') as log:
            child=subprocess.Popen(source_command(SOURCE,launch['stage']['command']),env=env,cwd=SOURCE,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            identity=process_identity(child.pid)
            if child.wait():raise RuntimeError('Collector worker failed; diagnose before retry')
    finally:
        stop_owned(child,identity);lease.close()


def execute(plan, shard, job):
    if Path(__file__).resolve()!=SOURCE/'scripts/run_sft_pass8_extension.py':
        raise ValueError('Execute the exact frozen controller')
    if os.getenv('SLURM_JOB_ID')!=job:raise ValueError('Run within the approved allocation')
    approval=read(ROOT/'approval.json');used,proof=accounted_seconds(approval,shard,job)
    seconds=verify_approval(plan,approval,job,shard,used)
    remaining=validate_allocation(subprocess.check_output(['scontrol','show','job',job,'-o'],text=True),job,seconds)
    devices=os.getenv('CUDA_VISIBLE_DEVICES','').split(',')
    if len(devices)!=1 or not devices[0]:raise ValueError('Controller requires its one assigned GPU identity')
    ready=read(ROOT/'readiness.json')
    if ready.get('plan_sha256')!=digest(plan) or ready.get('native_parse_passed') is not True:
        raise ValueError('CPU native preflight is required for this exact plan')
    from check_storage_quota import check_storage_quota
    check_storage_quota(1024**4)
    launch=make_launch(plan,shard,job);root=Path(launch['output']);root.mkdir(parents=True,exist_ok=True)
    with (root/'owner.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        attempt=root/'scheduler-attempts'/job
        if (attempt/'started.json').exists():raise ValueError('Preserve existing scheduler attempt; do not replay its identity')
        write(attempt/'started.json',dict(job_id=job,shard=shard,epoch=time.time(),
            plan_sha256=digest(plan),prior_charged_seconds=used,prior_attempts=proof,time_limit_seconds=seconds))
        write(root/f'launch-{job}.json',launch)
        write(ROOT/f'current-shard-{shard}.json',dict(job_id=job,shard=shard,
            launch=str(root/f'launch-{job}.json'),status=str(root/'status.json'),
            plan_sha256=digest(plan),epoch=time.time(),verified_complete=False))
        child=None;identity=None;deadline=time.monotonic()+remaining
        try:
            with (attempt/'worker.log').open('a') as log:
                child=subprocess.Popen(['srun',f'--jobid={job}','--nodes=1','--ntasks=1','--cpus-per-task=8',
                    '--gres=gpu:h200:1','--mem=120G','--exact','--cpu-bind=none',
                    str(RUNTIME/'venv/bin/python'),'-B',str(SOURCE/'scripts/run_sft_pass8_extension.py'),
                    '--shard',str(shard),'--job-id',job,'--worker'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                identity=process_identity(child.pid)
                while child.poll() is None:
                    if time.monotonic()>deadline:raise TimeoutError('Shutdown margin reached; preserve completed attempts')
                    write(root/'status.json',dict(job_id=job,shard=shard,stage='running',complete=False,
                        epoch=time.time(),canonical_records=len(list((root/'records').glob('*.json'))),
                        prior_charged_seconds=used))
                    with (attempt/'gpu.log').open('a') as gpu:
                        subprocess.run(['nvidia-smi','--query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw',
                            '--format=csv,noheader,nounits',f'--id={devices[0]}'],stdout=gpu,stderr=subprocess.STDOUT,timeout=20,check=False)
                    time.sleep(30)
            if child.returncode:raise RuntimeError('Worker failed; do not retry unchanged')
            receipt=read(root/'collection-complete.json')
            if receipt.get('complete') is not True:
                raise ValueError('Collector exited without independently auditable completion evidence')
            write(root/'status.json',dict(job_id=job,shard=shard,stage='collection_complete',complete=True,
                verified_complete=False,epoch=time.time(),collection_receipt_sha256=sha(root/'collection-complete.json')))
        except BaseException as exc:
            write(root/'status.json',dict(job_id=job,shard=shard,complete=False,stage='failed',epoch=time.time(),
                error_type=type(exc).__name__,error=str(exc)[:800]));raise
        finally:
            stop_owned(child,identity)
            write(attempt/'controller-exited.json',dict(epoch=time.time(),scheduler_final_accounting_pending=True))


def status_snapshot():
    """Read-only durable evidence for an external active-agent supervisor.

    This is not a scheduler watcher or a promise of automatic recovery. Status
    timestamps let the supervisor distinguish fresh collection from stale data.
    """
    plan=validate_prepared();approval_path=ROOT/'approval.json'
    approval=read(approval_path) if approval_path.exists() else {}
    shards=[]
    for shard in range(4):
        root=ROOT/f'shard-{shard}';pointer=ROOT/f'current-shard-{shard}.json'
        status=root/'status.json';progress=root/'progress.json'
        shards.append(dict(shard=shard,current=read(pointer) if pointer.exists() else None,
            status=read(status) if status.exists() else None,
            progress=read(progress) if progress.exists() else None,
            primary_records=len(list((root/'records').glob('*.json'))),
            collection_receipt=str(root/'collection-complete.json') if (root/'collection-complete.json').exists() else None))
    return dict(root=str(ROOT),plan_sha256=digest(plan),epoch=time.time(),
        approved=approval.get('approved') is True and approval.get('plan_sha256')==digest(plan),
        registered_attempts=approval.get('attempts',[]),resources=RESOURCES,limits=LIMITS,
        new_primary_episodes=sum(row['primary_records'] for row in shards),expected_new_primary_episodes=2121,
        verified_complete=False,verification_receipt=str(ROOT/'verified-complete.json')
            if (ROOT/'verified-complete.json').exists() else None,
        scheduler_queried=False,active_repair_supervision_established=False,shards=shards)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-id',default='PREPARE')
    parser.add_argument('--shard',type=int,choices=range(4),default=0)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--native-check',action='store_true')
    parser.add_argument('--status',action='store_true')
    args=parser.parse_args()
    if args.status:
        print(json.dumps(status_snapshot(),indent=2));return
    if args.worker:
        plan=validate_prepared();launch=read(ROOT/f'shard-{args.shard}/launch-{args.job_id}.json')
        if launch!=make_launch(plan,args.shard,args.job_id):raise ValueError('Frozen worker launch changed')
        # Worker is reachable only after the owning controller's approval check.
        approval=read(ROOT/'approval.json');used,_=accounted_seconds(approval,args.shard,args.job_id)
        verify_approval(plan,approval,args.job_id,args.shard,used)
        worker(plan,launch);return
    if args.execute:execute(validate_prepared(),args.shard,args.job_id);return
    plan=prepare() if args.prepare else validate_prepared()
    if args.native_check:native_check(plan)
    print(json.dumps({key:plan[key] for key in ('resources','limits','new_primary_episodes','status')},indent=2))


if __name__=='__main__':main()
