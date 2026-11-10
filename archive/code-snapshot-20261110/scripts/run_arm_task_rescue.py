#!/usr/bin/env python3
"""Prepare/own four actor+ARM collection shards; never submit allocations."""
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
import urllib.request

from run_arm_task_screen import REPO, RUNTIME, POOL, INITIAL, read, sha, freeze
from prepare_arm_turn_bonus import copy_plain
from resume_baseline import allocation, clean_environment, source_command, validate_source, write_json
from evaluate_baseline_checkpoint import configure_evaluation_tracking
from run_arm_quality_audit import environment, stop
from run_arm_turn_bonus_calibration import SELECTOR, selector_preflight
from runtime_ports import PORT_ENV, lease_ports
from check_storage_quota import check_storage_quota
from openwebrl.arm_task_rescue import eligible_ids, summarize, ARM_REVISION, SEED
from openwebrl.arm_task_screen import summarize as summarize_screen

SCREEN = POOL/'actor-screen-min5-2000/sharded-20261002'
CONTROL = POOL/'actor-arm-allfailure-20261002'
PARENT = RUNTIME/'reference-arm-task-screen-sharded-20261002-v2'
SOURCE = RUNTIME/'reference-arm-task-rescue-20261002-v1'
MODEL = RUNTIME/'arm-turn-bonus-preparation/arm-refresh-20260924/baseline40/model-compat'
RESOURCES = dict(jobs=4, gpus_per_job=2, gpu_type='H200', hours_per_job=3,
                 cpus_per_job=16, memory_gib_per_job=240, total_gpu_hours=24)
ATTEMPTS = 1


def prepare():
    final = read(SCREEN/'agent-completion-review.json')
    assert final['verified_complete'] and final['primary_attempts'] == 10000
    manifest = read(PARENT/'screen-partition.json')
    records = [read(p) for i in range(8) for p in (SCREEN/f'shard-{i}/records').glob('*.json')]
    screened = summarize_screen(records, manifest['task_order'], require_complete=True)
    ids = eligible_ids(screened, manifest['task_order'])
    if len(ids) != 682:
        raise ValueError('Verified all-failure cohort changed')
    rows = [json.loads(line) for line in (PARENT/'screen-tasks.jsonl').read_text().splitlines()]
    selected = [r for r in rows if str(r['metadata']['task_id']) in set(ids)]
    assert [str(r['metadata']['task_id']) for r in selected] == ids
    compat = read(MODEL.parent/'model-compat-manifest.json')
    assert Path(compat['weights']).resolve() == SELECTOR.resolve() and SELECTOR.name == ARM_REVISION
    for f in MODEL.glob('*.safetensors'):
        assert f.resolve().parent == SELECTOR.resolve(), 'Must use released ARM, not refreshed adapters'
    CONTROL.mkdir(parents=True, exist_ok=True)
    validate_source(PARENT)
    if not SOURCE.exists():
        shutil.copytree(PARENT, SOURCE, symlinks=True, copy_function=copy_plain,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git', '.browser_use_sessions'))
    name = 'openwebrl/arm_task_rescue.py'
    freeze(SOURCE/name, (REPO/name).read_text())
    partitions = [ids[i::4] for i in range(4)]
    frozen = dict(task_order=ids, shards=partitions, actor_screen_receipt_sha256=sha(SCREEN/'agent-completion-review.json'),
                  actor_screen_cohort_sha256=final['selected_native_sha256'], attempts_per_task=ATTEMPTS,
                  selector_revision=ARM_REVISION, seed=SEED)
    freeze(SOURCE/'rescue-partition.json', json.dumps(frozen, indent=2)+'\n')
    changed = [name, 'rescue-partition.json']
    for i in range(4):
        names = [f'rescue-shard-{i}.jsonl', f'rescue-shard-{i}-eval.json']
        freeze(SOURCE/names[0], ''.join(json.dumps(r)+'\n' for r in selected[i::4]))
        cfg = {'eval': {'defaults': dict(n_samples_per_eval_prompt=ATTEMPTS, temperature=.8, top_p=1., top_k=-1, max_response_len=1024),
                       'datasets': [dict(name='actor-arm-allfailure',path=str(SOURCE/names[0]),input_key='prompt',metadata_key='metadata',
                                         custom_generate_function_path='openwebrl.arm_task_rescue.generate')]}}
        freeze(SOURCE/names[1], json.dumps(cfg,indent=2)+'\n')
        changed += names
    source_manifest = read(SOURCE/'reference_manifest.json')
    source_manifest['recipe_files_sha256'].update({n:sha(SOURCE/n) for n in changed})
    source_manifest['actor_arm_allfailure'] = dict(parent=str(PARENT),changed_files=changed,optimizer_updates=0)
    write_json(SOURCE/'reference_manifest.json',source_manifest)
    validate_source(SOURCE)
    write_json(CONTROL/'cohort.json', frozen)
    return partitions


def plan(shard, job='PREPARE', partitions=None):
    partitions = prepare() if partitions is None else partitions
    old = read(SCREEN/'shard-0/workers/340180/evaluation_manifest.json')
    root = CONTROL/f'shard-{shard}'
    worker_root = root/'workers'/job
    def rewrite(value):
        if isinstance(value,str): return value.replace(old['source'],str(SOURCE)).replace(old['output'],str(worker_root)).replace('340180',job)
        if isinstance(value,list): return [rewrite(v) for v in value]
        if isinstance(value,dict): return {k:rewrite(v) for k,v in value.items()}
        return value
    stage = rewrite(copy.deepcopy(old))
    stage.update(job_id=job, source=str(SOURCE), output=str(worker_root), gpus=1,
                 wandb_run_id=f'arm-task-rescue-shard{shard}-{job}', optimizer_updates_requested=0)
    command = stage['command']
    for flag,value in {'--eval-config':str(SOURCE/f'rescue-shard-{shard}-eval.json'),
                       '--eval-function-path':'openwebrl.arm_task_rescue.generate_rollout',
                       '--wandb-group':'ARM-task-selection-allfailure-682'}.items():
        command[command.index(flag)+1] = value
    for k in list(stage['environment']):
        if k.startswith('OPENWEBRL_ARM_') or k in ['OPENWEBRL_TASK_SCREEN_CONFIG','OPENWEBRL_RESCUE_YIELD_CONFIG']:
            del stage['environment'][k]
    stage['environment'].update(NUM_GPUS='1',TP_SIZE='1',NUM_ROLLOUT='0',BROWSER_CONCURRENCY='8',SGLANG_CONCURRENCY='40',
        BROWSER_TRAIN_CONFIG=str(SOURCE/'screen-sharded-browser.json'),WANDB_RUN_ID=stage['wandb_run_id'],WANDB_RESUME='allow',
        OPENWEBRL_TASK_RESCUE_CONFIG=str(worker_root/'rescue-config.json'),
        OPENWEBRL_MULTIMODAL_STORAGE_DIR=str(root/'multimodal'/job),RAY_TMPDIR=f'/tmp/armrs-{job}-{shard}',
        SLIME_BROWSER_LOCAL_PROCESS_LOG_DIR=str(root/'browser-logs'/job))
    stage = configure_evaluation_tracking(stage,'openwebrl-evals')
    cfg = dict(output=str(root),attempt_output=str(root/'attempts'/job/str(time.time_ns())),
        tasks=str(SOURCE/f'rescue-shard-{shard}.jsonl'),task_order=partitions[shard],shard=shard,
        partition_manifest=str(SOURCE/'rescue-partition.json'),policy_id='OpenWebRL-4B-SFT:iteration0',
        seed=SEED,attempts_per_task=ATTEMPTS,max_steps=15,inference_timeout=180,
        judge_max_usd=5.,judge_max_calls=750,selector_endpoint='set_at_worker_start')
    return dict(job_id=job,shard=shard,source=str(SOURCE),output=str(root),stage=stage,rescue_config=cfg,
        resources=RESOURCES,storage_reserve_bytes=2*1024**4,primary_tasks=len(partitions[shard]),
        primary_attempts=len(partitions[shard])*ATTEMPTS,selector_model=str(MODEL),selector_revision=ARM_REVISION,
        approval_status='Exact allocation/judge budget and storage clearance pending')


def fingerprint():
    return {str(p):sha(p) for p in [SOURCE/'reference_manifest.json',REPO/'scripts/run_arm_task_rescue.py',REPO/'scripts/run_arm_task_rescue_2gpu.sbatch']}


def verify_approval(approval, job, shard, previous_seconds):
    if (not approval.get('approved') or approval.get('resources') != RESOURCES or
            approval.get('judge_max_usd') != 20. or approval.get('judge_max_calls') != 3000 or
            approval.get('attempts_per_task') != ATTEMPTS):
        raise ValueError('Exact compute/judge/attempt approval required')
    rows = approval.get('attempts',[])
    if len({r['job_id'] for r in rows}) != len(rows): raise ValueError('Duplicate accounting')
    current = [r for r in rows if r['job_id']==job and r['shard']==shard]
    if len(current)!=1 or previous_seconds+current[0]['time_limit_seconds']>10800:
        raise ValueError('Retry would exceed original per-shard3h cap')
    return current[0]


def native_check(p):
    env = dict(clean_environment(),**p['stage']['environment'])
    env.update(DRY_RUN='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    argv=shlex.split(subprocess.check_output(p['stage']['command'],env=env,text=True));env.pop('DRY_RUN')
    code="""from unittest.mock import patch
from slime.utils.arguments import parse_args
with patch('megatron.training.arguments.get_device_arch_version',return_value=9): a=parse_args()
expected={'num_rollout':0,'tensor_model_parallel_size':1,'actor_num_gpus_per_node':1,'sequence_parallel':False,
'browser_rollout_concurrency':8,'judge_api_model':'gpt-4.1','judge_prompt_variant':'action_history','max_steps':15,
'context_num_screenshots':1,'turn_history_reasoning_mode':'full','eval_function_path':'openwebrl.arm_task_rescue.generate_rollout'}
assert all(getattr(a,k)==v for k,v in expected.items()),{k:getattr(a,k) for k in expected}
assert a.load.endswith('/OpenWebRL-4B-SFT') and not a.use_checkpoint_opt_param_scheduler
d=a.eval_datasets[0];assert (d.n_samples_per_eval_prompt,d.temperature,d.top_p,d.top_k,d.max_response_len)==(1,.8,1.,-1,1024)
print('ARM_TASK_RESCUE_NATIVE_PARSE_OK')
"""
    with (CONTROL/'native-parse.log').open('w') as out:
        subprocess.run(source_command(SOURCE,[str(RUNTIME/'venv/bin/python'),'-c',code,*argv[2:]]),cwd=SOURCE,env=env,stdout=out,stderr=subprocess.STDOUT,check=True,timeout=120)
    write_json(CONTROL/'readiness.json',dict(native_parse_passed=True,gpu_startup_pending=True,fingerprint=fingerprint()))


def worker(p):
    job=p['job_id'];stage=p['stage'];out=Path(stage['output']);out.mkdir(parents=True,exist_ok=True)
    if os.environ.get('SLURM_JOB_ID') != job: raise ValueError('Dedicated approved job required')
    env=environment();env.update(stage['environment']);devices=env.get('CUDA_VISIBLE_DEVICES','').split(',')
    if len(devices)!=2 or not all(devices): raise ValueError('Exactly two assigned GPUs required')
    lease,port=lease_ports(job);env[PORT_ENV]=str(port);endpoint=f'http://127.0.0.1:{port+250}'
    p['rescue_config']['selector_endpoint']=endpoint
    write_json(out/'rescue-config.json',p['rescue_config']);write_json(out/'evaluation_manifest.json',stage)
    Path(env['OPENWEBRL_MULTIMODAL_STORAGE_DIR']).mkdir(parents=True,exist_ok=True)
    children=[];logs=[]
    try:
        selector_env=dict(env,CUDA_VISIBLE_DEVICES=devices[1],PYTHONPATH=str(SOURCE),OMP_NUM_THREADS='1',
            TRITON_CACHE_DIR=str(out/'triton-cache'),LD_LIBRARY_PATH=str(RUNTIME/'arm-reproduction/refresh-torch28/torch/lib'),
            CPATH=str(RUNTIME/'src/python-headers/Include')+':'+str(RUNTIME/'src/python-headers'))
        logs.append((out/'selector.log').open('a'))
        selector=subprocess.Popen([str(RUNTIME/'arm-reproduction/venv-refresh/bin/python'),str(SOURCE/'scripts/serve_arm.py'),
            '--mode','selection','--model',str(MODEL),'--base',str(MODEL),'--port',str(port+250)],env=selector_env,cwd=SOURCE,
            stdout=logs[-1],stderr=subprocess.STDOUT,start_new_session=True);children.append(selector)
        deadline=time.monotonic()+600
        while True:
            if selector.poll() is not None or time.monotonic()>deadline: raise RuntimeError('Selector startup failed')
            try:
                with urllib.request.urlopen(endpoint+'/health',timeout=2) as response: health=json.load(response)
                if health['model']!=str(MODEL) or health.get('adapter'): raise ValueError('Wrong selector identity')
                write_json(out/'selector-health.json',health);break
            except OSError:time.sleep(5)
        selector_preflight(endpoint,out,expected_api_model=str(MODEL))
        logs.append((out/'evaluation.log').open('a'))
        actor=subprocess.Popen(source_command(SOURCE,stage['command']),env=dict(env,CUDA_VISIBLE_DEVICES=devices[0]),cwd=SOURCE,
            stdout=logs[-1],stderr=subprocess.STDOUT,start_new_session=True);children.append(actor)
        while actor.poll() is None:
            if selector.poll() is not None: raise RuntimeError('Selector exited during actor collection')
            time.sleep(10)
        if actor.returncode: raise RuntimeError('Actor+ARM worker failed; inspect saved artifacts')
    finally:
        for child in reversed(children):stop(child)
        for handle in logs:handle.close()
        lease.close()


def execute(p):
    job=p['job_id'];shard=p['shard'];approval=read(CONTROL/'approval.json')
    if os.environ.get('SLURM_JOB_ID')!=job: raise ValueError('Run in approved allocation')
    previous=[r['job_id'] for r in approval['attempts'] if r['shard']==shard and r['job_id']!=job]
    used=0
    if previous:
        if not all(i.isdecimal() for i in previous):raise ValueError('Invalid previous job ID')
        raw=subprocess.check_output(['sacct','-X','-n','-P','-j',','.join(previous),'--format=JobIDRaw,State,ElapsedRaw'],text=True)
        rows={v[0]:v for line in raw.splitlines() if len(v:=line.split('|'))>=3}
        if any(i not in rows or rows[i][1].split()[0] not in ['COMPLETED','CANCELLED','FAILED','TIMEOUT'] for i in previous):raise ValueError('Prior attempts must have final accounting')
        used=sum(int(rows[i][2]) for i in previous)
    current=verify_approval(approval,job,shard,used)
    res=allocation(subprocess.check_output(['scontrol','show','job',job,'-o'],text=True),job,requested_gpus=2,maximum_hours=current['time_limit_seconds']/3600)
    if res['cpus']<16 or res['allocated_memory_gib']<240:raise ValueError('Wrong CPU/memory allocation')
    validate_source(SOURCE)
    ready=read(CONTROL/'readiness.json')
    if not ready['native_parse_passed'] or ready['fingerprint']!=fingerprint():raise ValueError('Stale native preflight')
    check_storage_quota(p['storage_reserve_bytes']);root=Path(p['output']);root.mkdir(parents=True,exist_ok=True)
    with (root/'owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);write_json(root/f'plan-{job}.json',p);write_json(root/'launch_manifest.json',p['stage'])
        progress=root/'runtime/progress.log';progress.parent.mkdir(exist_ok=True);progress.write_text('[TaskRescueProgress] startup\n')
        child=None;deadline=time.monotonic()+res['maximum_seconds']-120;last=-1
        try:
            child=subprocess.Popen(['srun',f'--jobid={job}','--nodes=1','--ntasks=1','--cpus-per-task=16','--gres=gpu:h200:2',
                '--mem=240G','--exact','--cpu-bind=none',sys.executable,str(Path(__file__).resolve()),'--job-id',job,'--shard',str(shard),'--worker'],start_new_session=True)
            while child.poll() is None:
                if time.monotonic()>deadline:raise TimeoutError('Allocation ending; preserve attempts')
                paths=list((root/'records').glob('*.json'));count=len(paths)
                if count!=last:
                    with progress.open('a') as log:log.write(f'[TaskRescueProgress] epoch={time.time()} attempts={count}/{p["primary_attempts"]}\n')
                    last=count
                write_json(root/'status.json',dict(job_id=job,shard=shard,complete=False,stage='collection',epoch=time.time(),primary_attempts=count))
                time.sleep(30)
            if child.returncode:raise RuntimeError('Guided worker failed; diagnosis required')
            summary=summarize([read(q) for q in (root/'records').glob('*.json')],p['rescue_config']['task_order'],ATTEMPTS,require_complete=True)
            write_json(root/'verification.json',summary);write_json(root/'status.json',dict(job_id=job,complete=True,stage='verified_complete',epoch=time.time()))
        except BaseException as exc:
            write_json(root/'status.json',dict(job_id=job,failed=True,complete=False,error_type=type(exc).__name__,error=str(exc)[:500],epoch=time.time()));raise
        finally:stop(child)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-id',default='PREPARE');parser.add_argument('--shard',type=int,choices=range(4),default=0)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--execute',action='store_true');parser.add_argument('--worker',action='store_true')
    args=parser.parse_args()
    if args.prepare:
        partitions=prepare()
        for i in range(4):write_json(CONTROL/f'plan-shard-{i}.json',plan(i,partitions=partitions))
        native_check(plan(0,partitions=partitions))
        write_json(CONTROL/'proposal.json',dict(resources=RESOURCES,judge_max_usd=20.,judge_max_calls=3000,attempts_per_task=ATTEMPTS,
            tasks=682,primary_attempts=682*ATTEMPTS,storage_reserve_bytes=2*1024**4,ready_cpu=True,approval_pending=True))
        print(json.dumps(dict(prepared=True,shards=[len(s) for s in partitions],resources=RESOURCES,approval_pending=True)));return
    if args.worker:
        p=read(CONTROL/f'shard-{args.shard}'/f'plan-{args.job_id}.json')
        if p['job_id']!=args.job_id or p['shard']!=args.shard:raise ValueError('Controller/worker identity changed')
        worker(p)
        return
    p=plan(args.shard,args.job_id,partitions=read(CONTROL/'cohort.json')['shards'])
    if args.execute:execute(p)
    else:print(json.dumps(p,indent=2))


if __name__=='__main__':main()
