#!/usr/bin/env python3
"""Own all processes of an exactly approved four-condition SFT allocation.

Dry-run is read-only. This controller cannot submit or extend Slurm allocations.
It charges prior attempts, uses a shared queue and never silently replays claims.
Collection completion is distinct from final independently verified completion.
"""
from __future__ import annotations
import argparse
import asyncio
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import time
import urllib.request

SOURCE=Path(__file__).resolve().parents[1]
RUNTIME=Path('/gpfs/scrubbed/zixianma/openwebrl-runtime')
sys.path.insert(0,str(SOURCE))
from openwebrl.controlled_sft_state import SuiteState,digest,validate_shard_cohort
from openwebrl.controlled_sft_serving_recovery import (
    recovery_config, server_max_running_requests, reuse_jev_diagnostic, run_saved_context_acceptance,
)
from runtime_ports import lease_ports
from resume_baseline import validate_source,source_command

FINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','BOOT_FAIL','DEADLINE','REVOKED','PREEMPTED'}
def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with tmp.open('w') as f: json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    tmp.replace(path)


def validate_topology(resources, *, sharded=None):
    profiles = {
        2: dict(gpus=2, gpu_type='H200', cpus=16, memory_gib=240, total_seconds=28800,
                sft_replicas=1, kev_replicas=1, collectors=8),
        8: dict(gpus=8, gpu_type='H200', cpus=64, memory_gib=960, total_seconds=57600,
                sft_replicas=7, kev_replicas=1, collectors=7),
    }
    expected = profiles.get(resources.get('gpus'))
    if not expected or any(resources.get(key) != value for key,value in expected.items()):
        raise ValueError('Unsupported or drifted approved serving topology')
    if sharded is not None and (resources['gpus'] == 2) != sharded:
        raise ValueError('The two-GPU topology requires one isolated task shard')
    return resources


def worker_actor_urls(resources, actor_urls):
    validate_topology(resources)
    if len(actor_urls) != resources['sft_replicas'] or len(set(actor_urls)) != len(actor_urls):
        raise ValueError('Actor endpoint count differs from the verified serving topology')
    return [actor_urls[index % len(actor_urls)] for index in range(resources['collectors'])]


def smoke_performance_receipt(plan, state, scheduler_directory):
    """Summarize actual saved smoke requests without making diagnostic calls."""
    observations = []
    with state.connect() as db:
        rows = db.execute("SELECT a.artifact_directory,q.condition FROM queue q JOIN attempts a ON a.item_id=q.item_id WHERE q.category='smoke' AND a.status='finished'").fetchall()
    if len(rows) != plan['smoke_episodes']:
        raise ValueError('Performance handoff requires every planned smoke episode')
    for row in rows:
        for path in (Path(row['artifact_directory'])/'events').glob('[0-9]*.json'):
            event = read(path)
            record = event.get('record', {})
            if event.get('event') == 'http_received' and record.get('provider') == 'sft':
                meta = record.get('response', {}).get('meta_info', {})
                latency = record.get('elapsed_seconds')
                if not isinstance(latency, (int, float)) or latency < 0 or not isinstance(meta.get('prompt_tokens'), int):
                    raise ValueError('Smoke actor timing/token evidence is malformed')
                observations.append(dict(condition=row['condition'], elapsed_seconds=latency,
                    prompt_tokens=meta['prompt_tokens'], output_tokens=meta.get('completion_tokens'),
                    evidence_path=str(path), evidence_sha256=sha(path)))
    if {row['condition'] for row in observations} != set(plan['conditions']):
        raise ValueError('Smoke performance receipt lacks actor HTTP evidence from every arm')
    latencies = sorted(row['elapsed_seconds'] for row in observations)
    identities = [scheduler_directory/f'sft-{i}-identity.json' for i in range(plan['resources']['sft_replicas'])]
    for path in identities:
        if read(path).get('model_path') != plan['worker_config']['actor_checkpoint']:
            raise ValueError('Smoke actor server identity differs from the frozen checkpoint')
    max_prompt = max(row['prompt_tokens'] for row in observations)
    return dict(plan_sha256=digest(plan), campaign_id=plan.get('campaign_id'), shard=plan.get('shard'),
        smoke_episodes=len(rows), actor_http_responses=len(observations), collectors=plan['resources']['collectors'],
        sft_replicas=plan['resources']['sft_replicas'], server_max_running_requests=server_max_running_requests(plan),
        scientific_inference_timeout_seconds=180, median_actor_http_seconds=statistics.median(latencies),
        p95_actor_http_seconds=latencies[min(len(latencies)-1, int(.95*len(latencies)))],
        max_actor_http_seconds=max(latencies), max_observed_prompt_tokens=max_prompt,
        long_context_observed=max_prompt >= 28000,
        maximum_concurrent_long_context_acceptance_verified=False,
        limitation='Ordinary smoke latency does not establish worst-case concurrency or full-cohort completion time.',
        actor_identities=[dict(path=str(path), sha256=sha(path)) for path in identities],
        gpu_log=dict(path=str(scheduler_directory/'gpu.log'), snapshot_sha256=sha(scheduler_directory/'gpu.log')),
        requests=observations)


def validate_device_assignments(resources, devices):
    validate_topology(resources)
    if len(devices) != resources['gpus'] or len(set(devices)) != len(devices) or not all(devices):
        raise ValueError('GPU assignments must exactly match the isolated approved topology')
    return devices[:resources['sft_replicas']], devices[resources['sft_replicas']]


def validate_frozen(root):
    root=Path(root).resolve();plan=read(root/'plan.json');source=Path(plan['source']).resolve()
    if Path(plan['root']).resolve()!=root or source!=root/'source':
        raise ValueError('Prepared plan and counters are bound to their original durable root')
    if source!=SOURCE.resolve() or plan['controller_sha256']!=sha(__file__):
        raise ValueError('Execute the exact frozen controller from this prepared source')
    validate_source(source)
    for path,expected in [(source/'reference_manifest.json',plan['source_manifest_sha256']),
        (root/'scientific-config.json',plan['scientific_config_sha256']),
        (Path(plan['task_file']),plan['task_file_sha256'])]:
        if sha(path)!=expected: raise ValueError('Frozen artifact changed: '+str(path))
    sharded = plan.get('shard') is not None
    validate_topology(plan['resources'], sharded=sharded)
    recovery_config(plan)
    primary_count = 600 if sharded else 1200
    if set(plan['conditions'])!={'L01','L08','L10','L11'} or plan['primary_episodes']!=primary_count:
        raise ValueError('Unexpected scientific cohort')
    if (plan['smoke_episodes'] != 12 or len({x['item_id'] for x in plan['schedule']}) != primary_count + 12
            or len(plan['schedule']) != primary_count + 12):
        raise ValueError('Cohort needs all unique primary items plus12 separate smokes')
    task_ids = [json.loads(line)['metadata']['task_id'] for line in Path(plan['task_file']).read_text().splitlines() if line.strip()]
    validate_shard_cohort(plan, task_ids)
    actor=Path(plan['worker_config']['actor_checkpoint'])
    for name,expected in plan['actor_files'].items():
        st=(actor/name).stat()
        if {'size':st.st_size,'mtime_ns':st.st_mtime_ns}!=expected:
            raise ValueError('Actor file identity changed')
    proof=read(root/'kev-file-proof.json')
    if sha(root/'kev-file-proof.json')!=plan['worker_config']['kev_proof']['file_proof_sha256']:
        raise ValueError('Kev file-proof receipt changed')
    for name,expected in proof['files'].items():
        st=Path(name).stat()
        if st.st_size!=expected['size'] or st.st_mtime_ns!=expected['mtime_ns']:
            raise ValueError('Kev file identity changed')
    source_proof=read(root/'kev-source-proof.json');kev_source=Path(source_proof['root'])
    if sha(root/'kev-source-proof.json')!=plan['kev_source_proof_sha256']:
        raise ValueError('Kev serving source proof changed')
    if (subprocess.check_output(['git','-C',str(kev_source),'rev-parse','HEAD'],text=True).strip()!=source_proof['commit']
            or subprocess.check_output(['git','-C',str(kev_source),'status','--porcelain','--untracked-files=no'],text=True).strip()):
        raise ValueError('Kev serving source no longer matches pinned clean commit')
    for name,expected in source_proof['files'].items():
        if sha(kev_source/name)!=expected:raise ValueError('Kev serving source file changed: '+name)
    for name,expected in plan['offline_receipts'].items():
        if sha(root/name)!=expected:raise ValueError('Frozen offline receipt changed')
    return plan


def check_approval(plan,approval,job,prior_seconds):
    if (approval.get('approved') is not True or approval.get('plan_sha256')!=digest(plan)
        or approval.get('resources')!=plan['resources'] or approval.get('limits')!=plan['limits']):
        raise ValueError('Exact new resources and API/browser caps have not been approved')
    attempts=approval.get('attempts',[])
    if len({str(a['job_id']) for a in attempts})!=len(attempts): raise ValueError('Duplicate scheduler attempt')
    current=[a for a in attempts if str(a['job_id'])==str(job)]
    if len(current)!=1: raise ValueError('Current allocation is not registered')
    seconds=current[0]['time_limit_seconds']
    if type(seconds) is not int or seconds<=0 or prior_seconds<0 or prior_seconds+seconds>plan['resources']['total_seconds']:
        raise ValueError('All-attempt scheduler cap exceeded; retries never reset it')
    return seconds


def scheduler_accounting(approval,job):
    total=0;rows=[]
    for attempt in approval.get('attempts',[]):
        old=str(attempt['job_id'])
        if old==str(job): continue
        raw=subprocess.check_output(['sacct','-X','-n','-P','-j',old,'--format=JobIDRaw,State,ElapsedRaw,AllocTRES'],text=True)
        matches=[r.split('|') for r in raw.splitlines() if r.split('|')[0]==old]
        if len(matches)!=1 or matches[0][1].split()[0] not in FINAL:
            raise ValueError('Every earlier allocation needs final scheduler accounting')
        elapsed=int(matches[0][2])
        if elapsed<0: raise ValueError('Negative scheduler time')
        total+=elapsed;rows.append(dict(job_id=old,elapsed_seconds=elapsed,sacct=raw))
    return total,rows


def validate_allocation(info,job,seconds,now=None,uid=None,resources=None):
    resources = resources or dict(gpus=8,gpu_type='H200',cpus=64,memory_gib=960,total_seconds=57600,
                                  sft_replicas=7,kev_replicas=1,collectors=7)
    validate_topology(resources)
    fields=dict(re.findall(r'(\w+)=(\S+)',info));uid=os.getuid() if uid is None else uid
    tres=dict(x.split('=',1) for x in fields.get('AllocTRES','').split(',') if '=' in x)
    if (fields.get('JobId')!=str(job) or fields.get('JobState')!='RUNNING'
        or not fields.get('UserId','').endswith(f'({uid})') or fields.get('NumNodes')!='1'
        or fields.get('NumCPUs')!=str(resources['cpus']) or tres.get('gres/gpu')!=str(resources['gpus'])
        or tres.get('gres/gpu:h200')!=str(resources['gpus'])
        or tres.get('mem') not in (f"{resources['memory_gib']}G",f"{resources['memory_gib']*1024}M")):
        raise ValueError('Allocation must exactly match the owned single-node approved GPU/CPU/memory profile')
    value=fields['TimeLimit'];days,clock=value.split('-',1) if '-' in value else ('0',value)
    parts=[int(x) for x in clock.split(':')]
    if len(parts)==2:parts.insert(0,0)
    duration=int(days)*86400+parts[0]*3600+parts[1]*60+parts[2]
    if duration>seconds: raise ValueError('Current scheduler limit exceeds remaining approved time')
    end=datetime.strptime(fields['EndTime'],'%Y-%m-%dT%H:%M:%S').timestamp()
    left=min(duration,int(end-(time.time() if now is None else now)))-180
    if left<300: raise ValueError('Insufficient allocation time after shutdown margin')
    return left


def environment():
    from dotenv import dotenv_values
    env={k:v for k,v in os.environ.items() if not k.startswith('KEV_')}
    for k,v in dotenv_values('/gpfs/projects/krishna/zixianma/OpenWebRL/.env').items():
        if v and k.startswith(('OPENAI_','JUDGE_','WANDB_','TYPESAFE_','JEV_')):env.setdefault(k,v)
    if not (env.get('OPENAI_API_KEY') or env.get('JUDGE_API_KEY')) or not (env.get('TYPESAFE_API_KEY') or env.get('JEV_API_KEY')):
        raise ValueError('Required actor-selector/judge credentials are missing')
    env.update(PYTHONPATH=str(SOURCE)+':'+str(RUNTIME/'src/Megatron-LM'),WANDB_PROJECT='openwebrl-evals',
               HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1')
    return env


def kev_environment(device):
    """Pin all HF cache aliases inherited from the allocation activation script."""
    cache=RUNTIME/'hf-cache'
    hub=str(cache/'hub')
    return dict(CUDA_VISIBLE_DEVICES=device,HF_HOME=str(cache),HF_HUB_CACHE=hub,
                HUGGINGFACE_HUB_CACHE=hub,TRANSFORMERS_CACHE=hub,
                HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                KEV_BACKEND='torch',KEV_DTYPE='bf16',KEV_FUSED='1',KEV_CUDA_GRAPHS='1',
                KEV_MERGE='1',KEV_LORA_SCALE='1',KEV_TRUNCATE_STATES='0',KEV_DATE_FACTS='0')


def initialize_wandb(plan, journal, env):
    # .env credentials live in the explicit child environment, not os.environ.
    import wandb
    return wandb.init(project='openwebrl-evals',id=plan['experiment_id'],resume='allow',config=dict(
        protocol=plan['protocol_name'],plan_sha256=digest(plan),resources=plan['resources'],limits=plan['limits']),
        dir=str(journal),settings=wandb.Settings(init_timeout=90,api_key=env.get('WANDB_API_KEY')))


def stop_owned(children,job,grace=60):
    records=[]
    for p in children:
        members=process_group_members(p.pid,job)
        records.append(dict(pgid=p.pid,original_live_group_members=members))
        if members:
            try:os.killpg(p.pid,signal.SIGTERM)
            except ProcessLookupError:pass
    end=time.monotonic()+grace
    while time.monotonic()<end and any(process_group_members(p.pid,job) for p in children):
        time.sleep(.1)
    for p in children:
        if process_group_members(p.pid,job):
            try:os.killpg(p.pid,signal.SIGKILL)
            except ProcessLookupError:pass
        p.wait(timeout=10)
    end=time.monotonic()+10
    while time.monotonic()<end and any(process_group_members(p.pid,job) for p in children):time.sleep(.1)
    for record in records:
        record['remaining_live_group_members']=process_group_members(record['pgid'],job)
        record['process_gone']=not record['remaining_live_group_members']
    return records


def process_group_members(pgid,job,proc_root=Path('/proc')):
    members=[]
    for directory in proc_root.iterdir():
        if not directory.name.isdigit():continue
        try:
            parts=(directory/'stat').read_text().rsplit(')',1)[1].split()
            if int(parts[2])!=pgid:continue
            if directory.stat().st_uid!=os.getuid() or f'/job_{job}/' not in (directory/'cgroup').read_text():
                raise ValueError('Process group contains an unowned or different-allocation process')
            if parts[0]!='Z':members.append(int(directory.name))
        except (FileNotFoundError,ProcessLookupError):continue
    return members


def cleanup_browser_groups(root,job,proc_root=Path('/proc')):
    receipts=[]
    for path in Path(root).glob('attempts/*/browser/server-process.json'):
        info=read(path)
        if str(info.get('job_id'))!=str(job):continue
        pid=info['pid'];proc=proc_root/str(pid);identity_changed=False;observed_identity=None
        if info['pgid']!=pid:raise ValueError('Recorded child is not a process-group leader')
        try:
            parts=(proc/'stat').read_text().rsplit(')',1)[1].split()
            observed_identity=dict(uid=proc.stat().st_uid,start_ticks=parts[19],pgid=int(parts[2]))
            identity_changed=observed_identity['uid']!=os.getuid() or str(info['proc_start_ticks'])!=parts[19]
        except (FileNotFoundError,ProcessLookupError):pass
        before=process_group_members(info['pgid'],job,proc_root)
        if identity_changed:
            # PID numbers can be recycled during a long collection. An
            # unrelated process is never signaled; a proven empty original
            # group is already clean. A nonempty/foreign group still fails
            # closed, including a recycled process-group ID.
            if before:
                raise ValueError(f'Owned browser process identity changed for pid={pid}; refusing signal to nonempty group')
            receipts.append(dict(registry=str(path),pid=pid,original_live_group_members=[],
                remaining_live_group_members=[],process_gone=True,identity_changed=True,
                observed_identity=observed_identity,method='original group independently scanned empty; recycled PID not signaled'))
            continue
        if before:
            try:os.killpg(info['pgid'],signal.SIGTERM)
            except ProcessLookupError:pass
            end=time.monotonic()+10
            while process_group_members(info['pgid'],job,proc_root) and time.monotonic()<end:time.sleep(.1)
            if process_group_members(info['pgid'],job,proc_root):
                try:os.killpg(info['pgid'],signal.SIGKILL)
                except ProcessLookupError:pass
                end=time.monotonic()+5
                while process_group_members(info['pgid'],job,proc_root) and time.monotonic()<end:time.sleep(.1)
        remaining=process_group_members(info['pgid'],job,proc_root)
        receipts.append(dict(registry=str(path),pid=pid,original_live_group_members=before,
                             remaining_live_group_members=remaining,process_gone=not remaining))
    return receipts


def execute(root,job):
    root=Path(root).resolve();plan=validate_frozen(root);approval=read(root/'approval.json')
    if os.getenv('SLURM_JOB_ID')!=str(job) or f'/job_{job}/' not in Path('/proc/self/cgroup').read_text():
        raise ValueError('Controller must run inside its registered allocation')
    used,previous=scheduler_accounting(approval,job)
    seconds=check_approval(plan,approval,job,used)
    resources=plan['resources']
    left=validate_allocation(subprocess.check_output(['scontrol','show','job',str(job),'-o'],text=True),job,seconds,resources=resources)
    deadline=time.monotonic()+left
    env=environment();env['OPENWEBRL_KEV_FORWARD_LOCK']=str(root/'kev-forward.lock')
    devices=env.get('CUDA_VISIBLE_DEVICES','').split(',')
    actor_devices,kev_device=validate_device_assignments(resources,devices)
    journal=root/'scheduler-attempts'/str(job);journal.mkdir(parents=True,exist_ok=True)
    write(journal/'prior-accounting.json',dict(prior_seconds=used,attempts=previous))
    children=[];handles=[];lease=None;state=None;wandb_run=None;completed=False
    def interrupted(signum,frame):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    def spawn(argv,name,extra=None,cwd=SOURCE,activate=True):
        handle=(journal/(name+'.log')).open('a');handles.append(handle)
        e=dict(env);e.update(extra or {})
        p=subprocess.Popen(source_command(SOURCE,argv) if activate else argv,cwd=cwd,env=e,
            stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        children.append(p);return p
    with (root/'controller.lock').open('a') as owner:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            write(journal/'allocation-validated.json',dict(job_id=str(job),plan_sha256=digest(plan),
                resources=plan['resources'],limits=plan['limits'],prior_actual_seconds=used,
                current_attempt_limit=seconds,work_deadline_unix=time.time()+max(0,deadline-time.monotonic())))
            state=SuiteState(root)
            initial=state.snapshot()
            if initial['halt'] or initial['queue'].get('active') or initial['queue'].get('interrupted'):
                raise RuntimeError('Unresolved prior work requires diagnosed recovery; never replay unchanged')
            lease,base=lease_ports(job)
            actor_urls=[f'http://127.0.0.1:{base+i*16}' for i in range(resources['sft_replicas'])]
            servers=[]
            for i,url in enumerate(actor_urls):
                servers.append(spawn([str(RUNTIME/'venv/bin/python'),'-m','sglang.launch_server',
                    '--model-path',plan['worker_config']['actor_checkpoint'],'--host','127.0.0.1','--port',str(base+i*16),
                    '--dtype','bfloat16','--tp','1','--mem-fraction-static','.60','--context-length','32768',
                    '--max-running-requests',str(server_max_running_requests(plan)),'--chunked-prefill-size','4096','--disable-cuda-graph'],
                    f'sft-server-{i}',{'CUDA_VISIBLE_DEVICES':actor_devices[i]}))
            kev_url=f'http://127.0.0.1:{base+128}'
            kev_env=kev_environment(kev_device)
            servers.append(spawn([str(RUNTIME/'kev-eval-venv/bin/python'),'-m','kev.serve','--run',plan['kev_spec']['server_run'],
                '--host','127.0.0.1','--port',str(base+128)],'kev-server',kev_env,cwd=RUNTIME/'kev-upstream',activate=False))
            gpu=spawn(['nvidia-smi','--query-gpu=timestamp,index,uuid,utilization.gpu,memory.used,power.draw',
                '--format=csv','-l','15'],'gpu',activate=False)
            recovery=recovery_config(plan)
            diagnostic=None
            if recovery is not None:
                reuse_jev_diagnostic(plan,journal,job)
            else:
                diagnostic=spawn([str(RUNTIME/'venv/bin/python'),'-m','openwebrl.controlled_sft_eval',
                    '--root',str(root),'--jev-context-preflight'],'jev-context-diagnostic',{'CUDA_VISIBLE_DEVICES':''})
            startup=min(deadline,time.monotonic()+1200)
            card=None;pending=set(range(resources['sft_replicas']))
            from openwebrl.kev_eval import check_server
            while pending or card is None or (diagnostic is not None and diagnostic.poll() is None):
                if any(p.poll() is not None for p in servers):raise RuntimeError('Model server failed during startup')
                if diagnostic is not None and diagnostic.poll() not in (None,0):raise RuntimeError('Jev maximum-state diagnostic failed; preserve receipt')
                if time.monotonic()>startup:raise TimeoutError('Model startup exceeded bounded allocation time')
                for i in list(pending):
                    try:
                        with urllib.request.urlopen(actor_urls[i]+'/get_model_info',timeout=3) as response:info=json.load(response)
                        if info.get('model_path')!=plan['worker_config']['actor_checkpoint']:raise ValueError('SFT server identity mismatch')
                        write(journal/f'sft-{i}-identity.json',info);pending.remove(i)
                    except OSError:pass
                if card is None:
                    try:card=check_server(kev_url+'/v1/systemone',plan['kev_spec'])
                    except (OSError,Exception) as exc:
                        if isinstance(exc,ValueError):raise
                write(root/'current.json',dict(stage='loading_models',job_id=job,pending_sft=sorted(pending),kev_ready=card is not None))
                if pending or card is None or (diagnostic is not None and diagnostic.poll() is None):time.sleep(3)
            if diagnostic is not None and diagnostic.returncode!=0:raise RuntimeError('Jev maximum-state diagnostic exited unsuccessfully')
            if recovery is not None:
                if len(actor_urls)!=1:raise ValueError('Serving recovery is scoped to one actor per shard')
                async def accept():
                    import httpx
                    async with httpx.AsyncClient(trust_env=False,timeout=180,follow_redirects=False) as client:
                        return await run_saved_context_acceptance(plan,state,actor_urls[0],journal,client)
                asyncio.run(accept())
            runtime=dict(plan_sha256=digest(plan),job_id=str(job),actor_base_urls=actor_urls,kev_base_url=kev_url,
                         kev_proof=dict(plan['worker_config']['kev_proof'],server_card=card))
            runtime_path=journal/'worker-runtime.json';write(runtime_path,runtime)
            wandb_run=initialize_wandb(plan,journal,env)
            for phase in ('smoke','primary'):
                if phase=='smoke' and state.snapshot()['phase']=='primary':continue
                if phase=='primary' and state.snapshot()['phase']=='smoke':
                    write(journal/'smoke-performance.json',smoke_performance_receipt(plan,state,journal))
                    state.enable_primary()
                workers=[spawn([str(RUNTIME/'venv/bin/python'),'-m','openwebrl.controlled_sft_eval',
                    '--root',str(root),'--worker-id',f'{job}-{phase}-{i}','--actor-base-url',actor_url,
                    '--kev-base-url',kev_url,'--runtime-config',str(runtime_path)],f'{phase}-worker-{i}',
                    {'CUDA_VISIBLE_DEVICES':'' ,'OPENWEBRL_ROLLOUT_PORT_BASE':str(base+144+i*12)})
                    for i,actor_url in enumerate(worker_actor_urls(resources,actor_urls))]
                while any(p.poll() is None for p in workers):
                    if any(p.poll() is not None for p in servers):raise RuntimeError('Model server exited during collection')
                    if any(p.poll() not in (None,0) for p in workers):raise RuntimeError('Collector failed; preserve interrupted attempts')
                    if time.monotonic()>deadline:raise TimeoutError('Approved all-attempt allocation boundary reached')
                    snapshot=state.snapshot();write(root/'current.json',dict(stage=phase,job_id=job,updated_unix=time.time(),**snapshot))
                    wandb_run.log({f'finished/{k}':v for k,v in snapshot['counts_by_condition'].items()})
                    if snapshot['halt']:raise RuntimeError('Shared budget/transport halt')
                    time.sleep(10)
                if any(p.returncode for p in workers):raise RuntimeError('Collector exited unsuccessfully')
            snapshot=state.snapshot()
            if snapshot['queue']!={'finished':len(plan['schedule'])} or snapshot['counters'].get('active_browsers') or snapshot['unsettled_reservations']:
                raise ValueError('Incomplete episodes or unresolved accounting remain')
            # Verify every durable result reference before collection closeout.
            with state.connect() as db:
                for row in db.execute("SELECT result_json FROM attempts WHERE status='finished'"):
                    item=json.loads(row['result_json'])
                    if sha(item['result_path'])!=item['result_sha256']:raise ValueError('Committed result artifact changed')
            # The final worker can finish after the last polling snapshot.
            wandb_run.log({f'finished/{k}':v for k,v in snapshot['counts_by_condition'].items()})
            completed=True
            write(root/'collection-complete.json',dict(collection_complete=True,verified_complete=False,
                primary_episodes=plan['primary_episodes'],smoke_episodes=plan['smoke_episodes'],
                campaign_id=plan.get('campaign_id'),shard=plan.get('shard'),job_id=job,final_independent_audit_required=True))
        finally:
            try:
                direct_groups=stop_owned(children,job)
                cleanup=cleanup_browser_groups(root,job)
                clean=all(x['process_gone'] for x in cleanup+direct_groups)
                write(journal/'owned-process-teardown.json',dict(verified=clean,browsers=cleanup,direct_process_groups=direct_groups,
                    all_direct_children_awaited=all(p.poll() is not None for p in children)))
                if state is not None:state.mark_interrupted_after_teardown(teardown_verified=clean,reason='controller closeout')
            finally:
                try:
                    if wandb_run is not None:wandb_run.finish(exit_code=0 if completed else 1)
                finally:
                    if lease is not None:lease.close()
                    for h in handles:h.close()
                    raw=subprocess.run(['sacct','-X','-n','-P','-j',str(job),'--format=JobIDRaw,State,ElapsedRaw,AllocTRES'],capture_output=True,text=True)
                    write(journal/'scheduler-accounting.json',dict(sacct=raw.stdout,prior_actual_seconds=used,
                        current_attempt_limit=seconds,final_scheduler_accounting_required=True,returncode=raw.returncode))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    g=p.add_mutually_exclusive_group(required=True);g.add_argument('--dry-run',action='store_true');g.add_argument('--execute')
    a=p.parse_args()
    if a.dry_run:
        plan=validate_frozen(a.root)
        print(json.dumps(dict(frozen_preflight_passed=True,launch_authorized=read(a.root/'approval.json')['approved'],
            resources=plan['resources'],limits=plan['limits'],primary_episodes=plan['primary_episodes'],
            smoke_episodes=plan['smoke_episodes'],campaign_id=plan.get('campaign_id'),shard=plan.get('shard')),indent=2))
    else:execute(a.root,a.execute)


if __name__=='__main__':main()
