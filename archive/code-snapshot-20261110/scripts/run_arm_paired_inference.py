#!/usr/bin/env python3
"""Own one approved two-GPU/four-hour shard of the three-arm inference run."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

SOURCE=Path(__file__).resolve().parents[1]
RUNTIME=Path('/gpfs/scrubbed/zixianma/openwebrl-runtime')
sys.path.insert(0,str(SOURCE))
from openwebrl.arm_paired_inference import PROTOCOL,schedule,summarize
from openwebrl.arm_controlled_inference import verify_record
from openwebrl.controlled_sft_state import digest
from resume_baseline import source_command,validate_source
from runtime_ports import lease_ports
from browser_prompt_assets import validate_browser_prompt_assets
from arm_inference_runtime import environment,allocation_deadline,check_approval,scheduler_accounting,stop_owned,write

RESOURCES=dict(gpus=2,gpu_type='H200',cpus=16,memory_gib=240,total_seconds=14400)
LIMITS=dict(judge_usd=10.,judge_calls=1980)


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def validate(root):
    root=Path(root).resolve();p=read(root/'plan.json')
    if (p['source']!=str(SOURCE) or p['root']!=str(root) or p['resources']!=RESOURCES
            or p['limits']!=LIMITS or p['protocol']!=PROTOCOL):raise ValueError('Frozen protocol/resources changed')
    validate_source(SOURCE)
    validate_browser_prompt_assets(SOURCE)
    policy=SOURCE/'openwebrl/env/prompts/system_prompt_browser_env.md'
    if sha(policy)!=PROTOCOL['actor_system_policy_sha256']:
        raise ValueError('Actor system policy differs from saved historical trajectories')
    for name,expected in p['artifacts'].items():
        if sha(name)!=expected:raise ValueError('Frozen artifact changed: '+name)
    for name,expected in p['model_files'].items():
        st=Path(name).stat()
        if dict(size=st.st_size,mtime_ns=st.st_mtime_ns)!=expected:raise ValueError('Verified model file changed')
    tasks=[json.loads(x)['metadata']['task_id'] for x in Path(p['task_file']).read_text().splitlines()]
    if (len(tasks)!=300 or read(root/'schedule.json')!=schedule(tasks)
            or type(p['shard']) is not int or p['shard'] not in (0,1)):raise ValueError('Full300 paired schedule changed')
    return p


def execute(root,job):
    root=Path(root).resolve();p=validate(root);a=read(root/'approval.json')
    collection_root=root/p.get('collection_directory','collection')
    if collection_root.parent != root:
        raise ValueError('Collection must be an immediate shard subdirectory')
    if os.environ.get('SLURM_JOB_ID')!=str(job) or f'/job_{job}/' not in Path('/proc/self/cgroup').read_text():
        raise ValueError('Must own the registered Slurm allocation')
    used,_=scheduler_accounting(a,str(job));seconds=check_approval(p,a,str(job),used)
    info=subprocess.check_output(['scontrol','show','job','-o',str(job)],text=True)
    deadline=allocation_deadline(info,str(job),seconds,RESOURCES)
    env=environment();devices=env.get('CUDA_VISIBLE_DEVICES','').split(',')
    if len(devices)!=2 or len(set(devices))!=2 or not all(devices):raise ValueError('Two isolated GPUs required')
    env.update(PYTHONPATH=str(SOURCE),WANDB_PROJECT='openwebrl-evals',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',SLIME_BROWSER_ENV_MODE='local_process')
    journal=root/'scheduler-attempts'/str(job);journal.mkdir(parents=True,exist_ok=False)
    children=[];handles=[];lease=None;completed=False
    def spawn(command,name,child_env):
        h=(journal/(name+'.log')).open('a');handles.append(h)
        child=subprocess.Popen(command,cwd=SOURCE,env=child_env,stdout=h,stderr=subprocess.STDOUT,start_new_session=True)
        children.append(child);return child
    def heartbeat(stage):
        write(root/'current.json',dict(job_id=str(job),stage=stage,epoch=time.time(),
            records=len(list((collection_root/'records').glob('*.json'))),expected=450,
            prior_actual_seconds=used,current_attempt_limit_seconds=seconds,verified_complete=False))
        if time.time()>deadline:raise TimeoutError('Approved allocation ending; retain every attempt')
    def interrupted(sig,frame):raise SystemExit(128+sig)
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    with (root/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            lease,port=lease_ports(job)
            write(journal/'allocation.json',dict(job_id=str(job),prior_actual_seconds=used,time_limit_seconds=seconds,
                resources=RESOURCES,limits=LIMITS,plan_sha256=digest(p)))
            actor=spawn(source_command(SOURCE,[str(RUNTIME/'venv/bin/python'),'-m','sglang.launch_server',
                '--model-path',p['actor'],'--host','127.0.0.1','--port',str(port),'--dtype','bfloat16','--tp','1',
                '--mem-fraction-static','.45','--context-length','32768','--max-running-requests','40',
                '--chunked-prefill-size','4096','--disable-cuda-graph']), 'actor',dict(env,CUDA_VISIBLE_DEVICES=devices[0]))
            selector=spawn([str(RUNTIME/'arm-reproduction/venv/bin/python'),str(SOURCE/'scripts/serve_arm.py'),
                '--mode','selection','--model',p['selector'],'--base',p['actor'],'--port',str(port+1)],
                'selector',dict(env,CUDA_VISIBLE_DEVICES=devices[1],
                    CPATH=str(RUNTIME/'src/python-headers/Include')+':'+str(RUNTIME/'src/python-headers')))
            piotr=spawn([str(RUNTIME/'arm-reproduction/venv/bin/python'),str(SOURCE/'scripts/serve_arm.py'),
                '--mode','selection','--model',p['piotr_selector'],'--base',p['actor'],'--port',str(port+2)],
                'piotr',dict(env,CUDA_VISIBLE_DEVICES=devices[1],
                    CPATH=str(RUNTIME/'src/python-headers/Include')+':'+str(RUNTIME/'src/python-headers')))
            startup=time.time()+900
            pending={'actor':f'http://127.0.0.1:{port}/get_model_info','selector':f'http://127.0.0.1:{port+1}/health',
                'piotr':f'http://127.0.0.1:{port+2}/health'}
            while pending:
                heartbeat('model_startup')
                if any(c.poll() is not None for c in (actor,selector,piotr)):raise RuntimeError('Model server failed')
                if time.time()>startup:raise TimeoutError('Model startup timeout')
                for name,url in list(pending.items()):
                    try:
                        with urllib.request.urlopen(url,timeout=3) as response:card=json.load(response)
                        if name=='actor' and card['model_path']!=p['actor']:raise ValueError('Wrong actor')
                        if name=='selector' and (card['model']!=p['selector'] or card.get('adapter')):raise ValueError('Wrong ARM')
                        if name=='piotr' and (card['model']!=p['piotr_selector'] or card.get('adapter')):raise ValueError('Wrong Piotr ARM')
                        write(journal/(name+'-identity.json'),card);pending.pop(name)
                    except OSError:pass
                if pending:time.sleep(2)
            spawn(['nvidia-smi','--query-gpu=timestamp,index,uuid,utilization.gpu,memory.used,power.draw','--format=csv','-l','15'],'gpu',env)
            config=dict(output=str(collection_root),manifest=str(root/'experiment.json'),schedule=str(root/'schedule.json'),
                tasks=p['task_file'],shard=p['shard'],actor=p['actor'],selector_model=p['selector'],actor_port=port,
                selector_endpoint=f'http://127.0.0.1:{port+1}',browser_port_start=port+10,
                additional_selectors=dict(piotr=dict(endpoint=f'http://127.0.0.1:{port+2}',model=p['piotr_selector'])),
                judge_caps=dict(max_calls=LIMITS['judge_calls'],max_usd=LIMITS['judge_usd']),
                judge_budget_root=str(root/'collection/judge-budget'),
                wandb_group='ARM-original-protocol-new-selector',wandb_id=p['experiment_id'],
                execution_source=str(SOURCE),scheduler_job_id=str(job))
            write(journal/'worker-config.json',config)
            collection=spawn(source_command(SOURCE,[str(RUNTIME/'venv/bin/python'),'-m','openwebrl.arm_paired_inference',
                '--config',str(journal/'worker-config.json')]),'collection',dict(env,CUDA_VISIBLE_DEVICES=devices[0]))
            while collection.poll() is None:
                heartbeat('collecting')
                if any(c.poll() is not None for c in (actor,selector,piotr)):raise RuntimeError('Model service failed during collection')
                time.sleep(10)
            if collection.returncode:raise RuntimeError('Collection failed; preserve and diagnose')
            cohort=[i for i in read(root/'schedule.json') if i['shard']==p['shard']];rows=[];ph=hashlib.sha256(json.dumps(PROTOCOL,sort_keys=True).encode()).hexdigest()
            for item in cohort:
                for attempt in item['attempts']:
                    rows.append(verify_record(read(collection_root/'records'/(attempt['key']+'.json')),item,attempt,collection_root,ph))
            write(root/'collection-audit.json',summarize(rows,[i['task_id'] for i in cohort],True))
            completed=True;heartbeat('collected_pending_final_accounting')
        finally:
            clean=stop_owned(children,str(job))
            write(journal/'teardown.json',dict(verified=all(x['process_gone'] for x in clean),processes=clean,
                collection_complete=completed,verified_complete=False))
            if lease is not None:lease.close()
            for h in handles:h.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path)
    g=p.add_mutually_exclusive_group(required=True);g.add_argument('--dry-run',action='store_true');g.add_argument('--execute')
    args=p.parse_args()
    if args.dry_run:
        plan=validate(args.root);print(json.dumps(dict(prepared=True,episodes=450,resources=RESOURCES,limits=LIMITS,
            approved=read(args.root/'approval.json')['approved'],protocol=plan['protocol']),indent=2))
    else:execute(args.root,args.execute)


if __name__=='__main__':main()
