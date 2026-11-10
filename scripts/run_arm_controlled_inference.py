#!/usr/bin/env python3
"""Prepare/own the controlled inference experiment. Never submits allocations."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request

REPO=Path('/gpfs/projects/krishna/zixianma/OpenWebRL')
RUNTIME=Path('/gpfs/scrubbed/zixianma/openwebrl-runtime')
sys.path.insert(0,str(REPO))
from openwebrl.arm_controlled_inference import PROTOCOL, schedule, summarize, verify_record
from openwebrl.arm_rescue_yield import write_json
from prepare_arm_turn_bonus import copy_plain
from resume_baseline import validate_source, allocation, source_command
from run_arm_quality_audit import environment, stop
from runtime_ports import lease_ports

CONTROL=RUNTIME/'arm-turn-bonus-preparation/controlled-inference-20261004'
PARENT=RUNTIME/'reference-arm-task-rescue-20261002-v1'
SOURCE=RUNTIME/'reference-arm-controlled-inference-20261004-v1'
ACTOR=Path('/gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT')
MODEL=RUNTIME/'arm-turn-bonus-preparation/arm-refresh-20260924/baseline40/model-compat'
RESOURCES=dict(jobs=2,gpus_per_job=2,gpu_type='H200',hours_per_job=8,
               cpus_per_job=16,memory_gib_per_job=240,total_gpu_hours=32)
CHANGED=['openwebrl/arm_controlled_inference.py','openwebrl/arm_inference.py','openwebrl/arm_terminal_budget.py',
         'openwebrl/run_evaluate.py','openwebrl/eval/reward_online_mind2web.py','scripts/serve_arm.py',
         'openwebrl/env/browser_runtime.py','openwebrl/env/local_process_env.py','openwebrl/env/web_env.py']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())


def prepare():
    validate_source(PARENT);CONTROL.mkdir(parents=True,exist_ok=True)
    if not SOURCE.exists():
        shutil.copytree(PARENT,SOURCE,symlinks=True,copy_function=copy_plain,
            ignore=shutil.ignore_patterns('__pycache__','*.pyc','.git','.browser_use_sessions'))
    for name in CHANGED:copy_plain(REPO/name,SOURCE/name)
    # Instrument only this isolated source. This is measurement, not a policy change.
    name='openwebrl/generate_browser.py';text=(PARENT/name).read_text()
    if text.count('await env.step(actions)')!=2:raise ValueError('Unexpected browser step call sites')
    text=text.replace('await env.step(actions)','await _controlled_timed_step(args, env, actions)')
    text+='''\n\nasync def _controlled_timed_step(args, env, actions):
    import time
    started = time.monotonic()
    try:
        return await env.step(actions)
    finally:
        callback = getattr(args, 'browser_step_timing_callback', None)
        if callback is not None:
            callback(time.monotonic() - started)
'''
    (SOURCE/name).write_text(text)
    manifest=read(SOURCE/'reference_manifest.json')
    manifest['recipe_files_sha256'].update({n:sha(SOURCE/n) for n in [*CHANGED,name]})
    manifest['controlled_inference']=dict(parent=str(PARENT),changed_files=[*CHANGED,name],optimizer_updates=0)
    write_json(SOURCE/'reference_manifest.json',manifest);validate_source(SOURCE)
    taskfile=REPO/'openwebrl/data/eval/online-mind2web.jsonl';target=CONTROL/'tasks.jsonl'
    if target.exists() and target.read_bytes()!=taskfile.read_bytes():raise ValueError('Benchmark changed')
    shutil.copyfile(taskfile,target)
    rows=[json.loads(line) for line in target.read_text().splitlines()]
    ids=[str(r.get('task_id') or r['metadata']['task_id']) for r in rows]
    if len(ids)!=300:raise ValueError('Need the complete benchmark')
    pairs=schedule(ids);sched=CONTROL/'schedule.json'
    if sched.exists() and read(sched)!=pairs:raise ValueError('Schedule changed')
    write_json(sched,pairs)
    compat=read(MODEL.parent/'model-compat-manifest.json')
    if Path(compat['weights']).name!=PROTOCOL['selector_revision']:raise ValueError('Wrong ARM weights')
    for p in MODEL.glob('*.safetensors'):
        if p.resolve().parent!=Path(compat['weights']).resolve():raise ValueError('ARM adapter/weight mismatch')
    experiment=dict(protocol=PROTOCOL,task_file_sha256=sha(target),schedule_sha256=sha(sched),
        source_manifest_sha256=sha(SOURCE/'reference_manifest.json'),
        resources=RESOURCES,judge_max_usd=50.,judge_max_calls=7200,attempts=1800,tasks=300,
        status='Prepared; exact allocation and API cap approval pending',compute_approved=False,
        primary='Paired overall SelectionARM success minus ordinary episode pass@5',
        secondary=['pass@1..5 curve','ARM versus actor0','paired discordant counts','common-valid sensitivity',
                   'actor/selector tokens, browser steps/time, request/episode latency, device samples, total allocation GPU-hours'],
        retry_policy='Resume only missing committed slots; preserve incomplete physical attempts and charge every attempt to the original caps',
        cost_limitations='Shared actor batching prevents exact per-policy GPU kernel attribution; device logs and allocated GPU-hours measure whole-job cost',
        pricing_source='https://developers.openai.com/api/docs/models/o4-mini',
        price_usd_per_million=dict(input=1.1,output=4.4),judge_sdk_retries=0)
    write_json(CONTROL/'experiment.json',experiment)
    for shard in range(2):write_json(CONTROL/f'plan-shard-{shard}.json',dict(shard=shard,tasks=150,attempts=900,resources=RESOURCES))
    subprocess.run([sys.executable,'-m','py_compile',*[str(SOURCE/n) for n in CHANGED],str(SOURCE/name)],check=True)
    # Import the actual deployed modules in their matching environment, without
    # touching CUDA, browsers, external APIs, or actor/selector weights.
    env=environment();env.update(PYTHONPATH=str(SOURCE),CUDA_VISIBLE_DEVICES='',WANDB_MODE='disabled',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    code="from openwebrl import arm_controlled_inference, run_evaluate; from openwebrl.env.web_env import WebEnv; print('CONTROLLED_INFERENCE_IMPORT_OK')"
    with (CONTROL/'import-check.log').open('w') as log:
        subprocess.run([str(RUNTIME/'venv/bin/python'),'-c',code],cwd=SOURCE,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
    ready=dict(cpu_import_passed=True,gpu_browser_startup_pending=True,source_manifest_sha256=sha(SOURCE/'reference_manifest.json'),
               controller_sha256=sha(__file__),batch_sha256=sha(REPO/'scripts/run_arm_controlled_inference_2gpu.sbatch'))
    write_json(CONTROL/'readiness.json',ready)
    return experiment


def verify_approval(approval,job,shard,used):
    if not approval.get('approved') or approval['resources']!=RESOURCES or approval.get('judge_max_usd')!=50. or approval.get('judge_max_calls')!=7200:
        raise ValueError('Exact allocation and judge-cap approval required')
    attempts=approval['attempts']
    if len({a['job_id'] for a in attempts})!=len(attempts):raise ValueError('Duplicate allocation accounting')
    found=[a for a in attempts if a['job_id']==job and a['shard']==shard]
    if len(found)!=1 or used+found[0]['time_limit_seconds']>8*3600:raise ValueError('Retry exceeds original eight-hour shard cap')
    return found[0]


def worker(config):
    root=Path(config['worker_root']);root.mkdir(parents=True,exist_ok=True)
    env=environment();env.update(PYTHONPATH=str(SOURCE),WANDB_PROJECT='openwebrl-evals',OMP_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1',SLIME_BROWSER_ENV_MODE='local_process',
        OPENWEBRL_MULTIMODAL_STORAGE_DIR=str(root/'multimodal'),RAY_TMPDIR=str(root/'ray'))
    devices=env.get('CUDA_VISIBLE_DEVICES','').split(',')
    if len(devices)!=2 or not all(devices):raise ValueError('Exactly two assigned GPUs required')
    lease,port=lease_ports(config['job_id']);children=[];handles=[]
    cfg=dict(output=config['output'],manifest=str(CONTROL/'experiment.json'),schedule=str(CONTROL/'schedule.json'),
        tasks=str(CONTROL/'tasks.jsonl'),shard=config['shard'],actor=str(ACTOR),selector_model=str(MODEL),
        actor_port=port,selector_endpoint=f'http://127.0.0.1:{port+1}',browser_port_start=port+10)
    write_json(root/'worker-config.json',cfg)
    def spawn(command,label,child_env):
        handle=(root/f'{label}.log').open('a');handles.append(handle)
        if label in ('actor','collection'):
            command=source_command(SOURCE,command)
        child=subprocess.Popen(command,cwd=SOURCE,env=child_env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        children.append(child);return child
    try:
        actor=spawn([str(RUNTIME/'venv/bin/python'),'-m','sglang.launch_server','--model-path',str(ACTOR),
            '--host','127.0.0.1','--port',str(port),'--dtype','bfloat16','--tp','1','--mem-fraction-static','.45',
            '--context-length','32768','--max-running-requests','40','--chunked-prefill-size','4096','--disable-cuda-graph'],
            'actor',dict(env,CUDA_VISIBLE_DEVICES=devices[0]))
        selector_env=dict(env,CUDA_VISIBLE_DEVICES=devices[1],
            LD_LIBRARY_PATH=str(RUNTIME/'arm-reproduction/refresh-torch28/torch/lib'),
            CPATH=str(RUNTIME/'src/python-headers/Include')+':'+str(RUNTIME/'src/python-headers'),TRITON_CACHE_DIR=str(root/'triton'))
        selector=spawn([str(RUNTIME/'arm-reproduction/venv-refresh/bin/python'),str(SOURCE/'scripts/serve_arm.py'),
            '--mode','selection','--model',str(MODEL),'--base',str(MODEL),'--port',str(port+1)],'selector',selector_env)
        deadline=time.monotonic()+900
        for url in [f'http://127.0.0.1:{port}/get_model_info',cfg['selector_endpoint']+'/health']:
            while True:
                if actor.poll() is not None or selector.poll() is not None:raise RuntimeError('Model server exited')
                try:
                    with urllib.request.urlopen(url,timeout=5) as response:json.load(response)
                    break
                except OSError:
                    if time.monotonic()>deadline:raise TimeoutError('Model server startup timeout')
                    time.sleep(5)
        meter=spawn(['nvidia-smi','--query-gpu=timestamp,index,uuid,utilization.gpu,memory.used,power.draw,temperature.gpu',
                     '--format=csv','-l','10'],'devices',env)
        collection=spawn([str(RUNTIME/'venv/bin/python'),'-m','openwebrl.arm_controlled_inference',
                          '--config',str(root/'worker-config.json')],'collection',dict(env,CUDA_VISIBLE_DEVICES=devices[0]))
        while collection.poll() is None:
            if actor.poll() is not None or selector.poll() is not None:raise RuntimeError('Serving failure during collection')
            time.sleep(10)
        if collection.returncode:raise RuntimeError('Collection failed; inspect durable attempts')
    finally:
        for child in reversed(children):stop(child)
        for handle in handles:handle.close()
        lease.close()


def execute(job,shard):
    if os.getenv('SLURM_JOB_ID')!=job:raise ValueError('Dedicated approved allocation required')
    approval=read(CONTROL/'approval.json');old=[a['job_id'] for a in approval['attempts'] if a['shard']==shard and a['job_id']!=job]
    used=0
    if old:
        if not all(i.isdecimal() for i in old):raise ValueError('Invalid previous job IDs')
        accounting=subprocess.check_output(['sacct','-X','-n','-P','-j',','.join(old),'--format=JobIDRaw,State,ElapsedRaw'],text=True)
        records={v[0]:v for line in accounting.splitlines() if len(v:=line.split('|'))>=3}
        if any(i not in records or records[i][1].split()[0] not in ('COMPLETED','FAILED','TIMEOUT','CANCELLED') for i in old):
            raise ValueError('Prior attempts need final accounting')
        used=sum(int(records[i][2]) for i in old)
    current=verify_approval(approval,job,shard,used)
    res=allocation(subprocess.check_output(['scontrol','show','job',job,'-o'],text=True),job,requested_gpus=2,maximum_hours=current['time_limit_seconds']/3600)
    if res['cpus']<16 or res['allocated_memory_gib']<240:raise ValueError('Wrong CPU/RAM allocation')
    validate_source(SOURCE);ready=read(CONTROL/'readiness.json')
    if not ready['cpu_import_passed'] or ready['source_manifest_sha256']!=sha(SOURCE/'reference_manifest.json') or ready['controller_sha256']!=sha(__file__):
        raise ValueError('Stale readiness')
    root=CONTROL/f'shard-{shard}';root.mkdir(parents=True,exist_ok=True)
    config=dict(job_id=job,shard=shard,output=str(root),worker_root=str(root/'workers'/job))
    write_json(root/f'worker-{job}.json',config)
    with (root/'owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        child=None;deadline=time.monotonic()+res['maximum_seconds']-120
        try:
            child=subprocess.Popen(['srun',f'--jobid={job}','--nodes=1','--ntasks=1','--cpus-per-task=16','--gres=gpu:h200:2',
                '--mem=240G','--exact','--cpu-bind=none',sys.executable,str(Path(__file__).resolve()),'--worker',str(root/f'worker-{job}.json')],start_new_session=True)
            while child.poll() is None:
                if time.monotonic()>deadline:raise TimeoutError('Budget ending; preserve partial collection')
                write_json(root/'status.json',dict(job_id=job,shard=shard,epoch=time.time(),complete=False,
                    attempts=len(list((root/'records').glob('*.json'))),expected=900))
                time.sleep(30)
            if child.returncode:raise RuntimeError('Controlled inference worker failed')
            cohort=[t for t in read(CONTROL/'schedule.json') if t['shard']==shard]
            protocol_sha=hashlib.sha256(json.dumps(PROTOCOL,sort_keys=True).encode()).hexdigest();records=[]
            for item in cohort:
                for attempt in item['attempts']:
                    records.append(verify_record(read(root/'records'/(attempt['key']+'.json')),item,attempt,root,protocol_sha))
            write_json(root/'verification.json',summarize(records,[t['task_id'] for t in cohort],True))
            write_json(root/'status.json',dict(job_id=job,complete=True,epoch=time.time(),attempts=len(records)))
        except BaseException as exc:
            write_json(root/'status.json',dict(job_id=job,complete=False,failed=True,error_type=type(exc).__name__,error=str(exc),epoch=time.time()));raise
        finally:stop(child)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare',action='store_true');p.add_argument('--execute',action='store_true')
    p.add_argument('--job-id');p.add_argument('--shard',type=int,choices=[0,1]);p.add_argument('--worker',type=Path);a=p.parse_args()
    if a.prepare:print(json.dumps(prepare(),indent=2))
    elif a.worker:worker(read(a.worker))
    elif a.execute:execute(a.job_id,a.shard)
    else:p.error('Choose --prepare, --execute, or --worker')
