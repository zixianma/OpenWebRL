#!/usr/bin/env python3
"""Prepare a separate-slice MIG probe; --execute requires an approved allocation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import urllib.request

from resume_baseline import source_command,validate_source,write_json

REPO=Path(__file__).resolve().parents[1]
RUNTIME=Path('/gpfs/scrubbed/zixianma/openwebrl-runtime')
PYTHON=RUNTIME/'venv/bin/python'
ARM_PYTHON=RUNTIME/'arm-reproduction/venv/bin/python'
PARENT=RUNTIME/'reference-arm-task-success-20260921-v2'
SOURCE=RUNTIME/'reference-arm-mig-probe-20260921-v6'
CONTROL=RUNTIME/'arm-turn-bonus-preparation/mig-probe-20260921'
ACTOR=Path('/gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT')
SELECTOR=Path('/gpfs/scrubbed/zixianma/checkpoints/web/arm/selection/81b452d800d9f859687074f82680dd5257e02d89')


def prepare():
    # These orchestration helpers live in the working repository, not the
    # frozen inference source. Service/client workers must not import them.
    from prepare_arm_turn_bonus import copy_plain
    validate_source(PARENT);CONTROL.mkdir(parents=True,exist_ok=True)
    files=['openwebrl/arm_mig_probe.py','scripts/run_arm_mig_probe.py']
    if not SOURCE.exists():
        shutil.copytree(PARENT,SOURCE,symlinks=True,copy_function=copy_plain,
            ignore=shutil.ignore_patterns('__pycache__','*.pyc','.git','.browser_use_sessions'))
        for f in files:copy_plain(REPO/f,SOURCE/f)
        m=json.loads((SOURCE/'reference_manifest.json').read_text())
        m['recipe_files_sha256'].update({f:hashlib.sha256((SOURCE/f).read_bytes()).hexdigest() for f in files})
        m['mig_probe']=dict(parent=str(PARENT),diagnostic_only=True)
        write_json(SOURCE/'reference_manifest.json',m)
    validate_source(SOURCE)
    if any((SOURCE/f).read_bytes()!=(REPO/f).read_bytes() for f in files):raise ValueError('Freeze a new source after probe changes')
    panel=json.loads((RUNTIME/'arm-turn-bonus-preparation/quality-audit-20260921-v2/comparison-plan.json').read_text())
    rows=[json.loads(x) for x in Path(panel['states']).read_text().splitlines()]
    rows.sort(key=lambda x:(len(x['actor_prompt']),x['id']))
    from openwebrl.arm_mig_probe import archived_history_responses
    for row in (rows[0],rows[-1]):
        archived_history_responses(row['history'])
        if not Path(row['screenshot']).is_file() or not all(Path(i['path']).is_file() for i in row['actor_images']):
            raise ValueError('Missing archived probe images')
    tasks=[json.loads(x) for x in (SOURCE/'task_success_tasks.jsonl').read_text().splitlines()]
    p=dict(resources=dict(partition='gpu-h200-mig',profile='h200_1g.18gb',slices=2,
        hours=1,cpus=2,memory_gib=60,estimated_compute_usd=.9*.143*2),
        judge_max_usd=1,judge_max_requests=6,actor=str(ACTOR),states=[rows[0],rows[-1]],tasks=tasks[:2],
        diagnostic_only=True,bc_gate='Require durable iteration20, verified standalone export and parity before a B/C MIG evaluation',
        new_allocation_approved=(CONTROL/'approval-20260921-relaunch.json').is_file())
    write_json(CONTROL/'plan.json',p)
    bc={}
    for variant,job in [('B','313208'),('C','313210')]:
        root=RUNTIME/f'evaluations/arm-failure-additive-{job}'
        checkpoint=root/'runtime/iter_0000019'
        bc[variant]=dict(training_root=str(root),checkpoint=str(checkpoint),checkpoint_files_present=(checkpoint/'common.pt').is_file(),
            migration_authorized=False,required_gates=['durable iteration20 and correct B/C lineage','standalone HF export and weight/parity verification','single-slice memory/load test','native GPT4.1 actor-only rollout and verdict persistence'])
    write_json(CONTROL/'bc-followup-readiness.json',bc)
    return p


def service(a):
    root=Path(a.output)
    # This process lives in its own one-slice srun/cgroup. Probe then exec server.
    code="""import ctypes,json,os,torch,uuid
cuda=ctypes.CDLL('libcuda.so.1')
cuda.cuInit.argtypes=[ctypes.c_uint];cuda.cuInit.restype=ctypes.c_int
cuda.cuDeviceGet.argtypes=[ctypes.POINTER(ctypes.c_int),ctypes.c_int];cuda.cuDeviceGet.restype=ctypes.c_int
cuda.cuDeviceGetUuid_v2.argtypes=[ctypes.c_void_p,ctypes.c_int];cuda.cuDeviceGetUuid_v2.restype=ctypes.c_int
assert cuda.cuInit(0)==0
device=ctypes.c_int();assert cuda.cuDeviceGet(ctypes.byref(device),0)==0
ident=(ctypes.c_ubyte*16)();assert cuda.cuDeviceGetUuid_v2(ctypes.byref(ident),device)==0
p=torch.cuda.get_device_properties(0)
print(json.dumps(dict(count=torch.cuda.device_count(),name=p.name,memory_bytes=p.total_memory,
    uuid=str(uuid.UUID(bytes=bytes(ident))),uuid_api='cuDeviceGetUuid_v2',
    torch_uuid=str(p.uuid),cuda_visible_devices=os.getenv('CUDA_VISIBLE_DEVICES'),
    slurm_step_id=os.getenv('SLURM_STEP_ID'))))"""
    report=json.loads(subprocess.check_output([str(PYTHON),'-c',code],text=True))
    if report['count']!=1 or not 15*1024**3<=report['memory_bytes']<=21*1024**3:
        raise ValueError('Expected exactly one 18-GB MIG device in this step')
    write_json(root/f'{a.service}-device.json',report)
    # Verify isolation before either service reserves model or KV-cache memory.
    until=time.monotonic()+90
    while not (root/'devices-validated.json').exists():
        if time.monotonic()>until:raise TimeoutError('Controller did not verify MIG device isolation')
        time.sleep(1)
    config=json.loads((root/'probe-config.json').read_text());env=os.environ.copy()
    env.update(PYTHONPATH=str(SOURCE),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        CPATH=str(RUNTIME/'src/python-headers/Include')+':'+str(RUNTIME/'src/python-headers'))
    if a.service=='actor':
        command=[str(PYTHON),'-m','sglang.launch_server','--model-path',str(ACTOR),
            '--host','127.0.0.1','--port',str(config['actor_port']),'--dtype','bfloat16','--tp','1',
            '--context-length','32768','--mem-fraction-static','.86','--max-total-tokens','32768',
            '--max-running-requests','1','--chunked-prefill-size','512','--disable-cuda-graph']
    else:
        command=[str(ARM_PYTHON),str(SOURCE/'scripts/serve_arm.py'),'--mode','selection',
            '--model',str(SELECTOR),'--base',str(ACTOR),'--port',str(config['selector_port'])]
    os.execvpe(command[0],command,env)


def execute(job):
    from runtime_ports import lease_ports
    from run_arm_quality_audit import environment,stop
    if os.getenv('SLURM_JOB_ID')!=job:raise ValueError('Must run inside the approved MIG job')
    raw=subprocess.check_output(['scontrol','show','job',job,'-o'],text=True)
    fields=dict(x.split('=',1) for x in raw.split() if '=' in x)
    limit=fields['TimeLimit'].split(':')
    if len(limit)!=3 or any('-' in x for x in limit):raise ValueError('Unexpected time limit')
    seconds=sum(int(x)*m for x,m in zip(limit,[3600,60,1]))
    if fields['Partition']!='gpu-h200-mig' or 'gres/gpu:h200_1g.18gb=2' not in fields.get('AllocTRES','') or not 0<seconds<=3600:
        raise ValueError('Requires exactly two MIG slices and at most one hour')
    plan=prepare();root=RUNTIME/f'evaluations/arm-mig-probe-{job}';root.mkdir(exist_ok=False)
    port_lease,base=lease_ports(job);deadline=time.monotonic()+seconds-90
    config=dict(plan,output=str(root),actor_port=base+10,selector_port=base+11,
        selector_endpoint=f'http://127.0.0.1:{base+11}',run_id=f'arm-mig-probe-{job}')
    receipt=CONTROL/f'recovery-{job}.json'
    if receipt.is_file():
        from openwebrl.arm_mig_probe import recovered_records
        config['recovery']=json.loads(receipt.read_text())
        recovered_records(config)
        config['judge_budget_root']=config['recovery']['judge_budget_root']
    write_json(root/'probe-config.json',config)
    env=environment();env.update(PYTHONPATH=str(SOURCE),WANDB_PROJECT='openwebrl-evals',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        SLIME_BROWSER_ENV_MODE='local_process',SLIME_BROWSER_LOCAL_PROCESS_PYTHON=str(PYTHON),
        SLIME_BROWSER_LOCAL_PROCESS_MAX_PROCESSES='1',SLIME_BROWSER_ROLLOUT_CONCURRENCY='1',
        SLIME_BROWSER_LOCAL_PROCESS_PORT_START=str(base+20),SLIME_BROWSER_LOCAL_PROCESS_PORT_END=str(base+39),
        SLIME_BROWSER_LOCAL_PROCESS_LOG_DIR=str(root/'browsers'),
        SLIME_BROWSER_QUERY_BLACKLIST_PATH=str(SOURCE/'reference_empty_blacklist.txt'),
        SLIME_ADAPTIVE_QUERY_BLACKLIST_PATH=str(SOURCE/'reference_empty_blacklist.txt'),
        OPENWEBRL_MULTIMODAL_STORAGE_DIR=str(root/'image-scratch'))
    Path(env['OPENWEBRL_MULTIMODAL_STORAGE_DIR']).mkdir()
    children=[];handles=[];client=None
    def launch(cmd,name):
        f=(root/name).open('w');handles.append(f)
        proc=subprocess.Popen(cmd,cwd=SOURCE,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        children.append(proc);return proc
    def check():
        if time.monotonic()>=deadline:raise TimeoutError('MIG pilot allocation deadline')
        if any(c.poll() is not None for c in children if c is not client):raise RuntimeError('MIG service stopped; inspect logs')
    def interrupted(*_):raise InterruptedError('Allocation signaled')
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,interrupted)
    try:
        for name in ('actor','selector'):
            launch(['srun',f'--jobid={job}','--exact','--exclusive','--nodes=1','--ntasks=1',
                '--cpus-per-task=1','--mem=30G','--gres=gpu:h200_1g.18gb:1',str(PYTHON),
                str(SOURCE/'scripts/run_arm_mig_probe.py'),'--service',name,'--output',str(root)],name+'.log')
        ready=set();until=min(deadline,time.monotonic()+900)
        while len(ready)!=2:
            check()
            device_files=[root/f'{name}-device.json' for name in ('actor','selector')]
            if not (root/'devices-validated.json').exists() and all(p.is_file() for p in device_files):
                from openwebrl.arm_mig_probe import validate_devices
                reports=[json.loads(p.read_text()) for p in device_files]
                validate_devices(*reports)
                write_json(root/'devices-validated.json',dict(passed=True,devices=reports))
            for name,port,path in [('actor',base+10,'/health_generate'),('selector',base+11,'/health')]:
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}{path}',timeout=2) as response:
                        if response.status==200:ready.add(name)
                except OSError:pass
            if time.monotonic()>until:raise TimeoutError('MIG service startup timeout')
            time.sleep(3)
        from openwebrl.arm_mig_probe import validate_devices
        validate_devices(*[json.loads((root/f'{name}-device.json').read_text()) for name in ('actor','selector')])
        client=launch(source_command(SOURCE,[str(PYTHON),str(SOURCE/'scripts/run_arm_mig_probe.py'),
            '--client','--output',str(root)]),'probe.log')
        while client.poll() is None:
            check();write_json(root/'status.json',dict(stage='probe',complete=False));time.sleep(10)
        if client.returncode:raise RuntimeError('MIG functional probe failed')
        result=json.loads((root/'probe-result.json').read_text())
        if not result.get('passed'):raise ValueError('MIG probe incomplete')
        write_json(root/'status.json',dict(stage='complete',complete=True,result=result))
    except BaseException as exc:
        write_json(root/'status.json',dict(stage='failed',complete=False,error_type=type(exc).__name__,error=str(exc)[:300]));raise
    finally:
        for child in reversed(children):stop(child)
        for f in handles:f.close()
        port_lease.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execute',action='store_true')
    parser.add_argument('--job-id');parser.add_argument('--service',choices=['actor','selector'])
    parser.add_argument('--client',action='store_true');parser.add_argument('--output');a=parser.parse_args()
    if a.service:service(a)
    elif a.client:
        # Import SGLang before making the asyncio loop (same as standalone eval).
        from openwebrl import run_evaluate
        from openwebrl.arm_mig_probe import run
        import asyncio
        asyncio.run(run(json.loads((Path(a.output)/'probe-config.json').read_text())))
    elif a.execute:execute(a.job_id)
    else:
        p=prepare();print(json.dumps({k:p[k] for k in ['resources','judge_max_usd','diagnostic_only','bc_gate']},indent=2))
