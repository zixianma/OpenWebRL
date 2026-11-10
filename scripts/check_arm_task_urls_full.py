#!/usr/bin/env python3
"""Resumable CPU-only URL coverage, reusing the independently audited pilot."""
import argparse
import asyncio
from collections import Counter, defaultdict
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = Path('/gpfs/projects/krishna/zixianma/OpenWebRL')
RUNTIME = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime')
POOL = RUNTIME/'arm-turn-bonus-preparation/task-pool-expansion-20260922/curation-v3-20260929'
PILOT = POOL/'live-browser-20260929'
ROOT = POOL/'live-browser-full-20260929'
SUPERVISOR = RUNTIME/'arm-turn-bonus-preparation/mixed-reweight-20260927/supervisor'
RESOURCES = dict(gpus=0, cpus=8, memory_gib=32, hours=13, browser_workers=8)
SHARDS = 4
BUDGET_SECONDS = 13*3600


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def split_probes(probes, reused, count=SHARDS):
    remaining=sorted((p for p in probes if p['probe_id'] not in reused),key=lambda p:p['probe_id'])
    return [remaining[i::count] for i in range(count)]


def normal_observations(history):
    return [x for x in history if x.get('reason')!='browser_infrastructure']


def final_decision(observations):
    if any(x['classification']=='page_available' for x in observations):return 'page_available'
    if len(observations)<2:return None
    if all(x['classification']=='unavailable' for x in observations[:2]):return 'unavailable_after_retry'
    return 'inconclusive'


def accounting_rows(raw, ids):
    rows=[]
    for line in raw.splitlines():
        f=line.split('|')
        if f[0] in ids:
            rows.append(dict(job_id=f[0],state=f[1],elapsed_seconds=int(f[2]),exit_code=f[3]))
    if len(rows)!=len(ids) or {x['job_id'] for x in rows}!=set(ids):
        raise ValueError('Every approved attempt must have exactly one accounting record')
    return rows


def prepare():
    sys.path.insert(0,str(REPO/'scripts'))
    from rebuild_arm_task_pool import load_upstream
    upstream,_=load_upstream(POOL/'upstream')
    pilot=read(PILOT/'manifest.json');audit=read(PILOT/'run/agent-completion-review.json')
    assert audit['verified'] and all(audit['checks'].values())
    assert sha(PILOT/'manifest.json')==audit['manifest_sha256']
    assert sha(Path(pilot['input']))==pilot['input_sha256']
    by_url={}
    with Path(pilot['input']).open() as f:
        for line in f:
            r=json.loads(line);url=upstream.normalize_start_url(r['website'])
            p=by_url.setdefault(url,dict(probe_id=hashlib.sha256(url.encode()).hexdigest()[:24],
                url=url,task_ids=[],sources=Counter()))
            p['task_ids'].append(str(r['task_id']));p['sources'][r['benchmark_name']]+=1
    probes=list(by_url.values());reused={p['probe_id'] for p in pilot['probes']}
    assert len(probes)==125761 and len(reused)==200
    assert len({p['probe_id'] for p in probes})==len(probes)
    assert sum(len(p['task_ids']) for p in probes)==184546
    frozen=ROOT/'frozen';frozen.mkdir(parents=True,exist_ok=True)
    for src,name in [(REPO/'scripts/check_arm_task_urls.py','check_arm_task_urls.py'),
                     (Path(__file__),'controller.py')]:
        target=frozen/name
        if target.exists() and sha(target)!=sha(src):raise ValueError('Frozen controller changed')
        if not target.exists():shutil.copy2(src,target)
    assert sha(frozen/'check_arm_task_urls.py')==read(PILOT/'approval.json')['checker_sha256']
    common={k:pilot[k] for k in ['source','source_hashes','input','input_sha256','policy','interpretation']}
    manifests=[]
    for index,part in enumerate(split_probes(probes,reused)):
        root=ROOT/f'shard{index}';root.mkdir(exist_ok=True)
        assignments=[]
        for offset in range(0,len(part),64):
            chunk=root/'chunks'/f'{offset//64:04d}.json'
            content=dict(common,probes=part[offset:offset+64])
            if chunk.exists() and read(chunk)!=content:raise ValueError('Frozen URL chunk changed')
            write(chunk,content)
            assignments.extend(dict(p,chunk=str(chunk),chunk_sha256=sha(chunk)) for p in part[offset:offset+64])
        path=root/'manifest.json'
        manifest=dict(schema=1,shard=index,resources=RESOURCES,approved_seconds=BUDGET_SECONDS,
            probes=assignments,source_hashes=pilot['source_hashes'],source=pilot['source'],
            input_sha256=pilot['input_sha256'],pilot_audit_sha256=sha(PILOT/'run/agent-completion-review.json'),
            worker_sha256=sha(frozen/'check_arm_task_urls.py'),controller_sha256=sha(frozen/'controller.py'))
        if path.exists() and read(path)!=manifest:raise ValueError('Frozen shard changed')
        write(path,manifest);manifests.append(dict(shard=index,urls=len(part),path=str(path),sha256=sha(path)))
    request=dict(status='prepared',resources_per_job=RESOURCES,jobs=4,
        total_cpu_hours_cap=416,total_gpu_hours=0,maximum_seconds_per_shard=BUDGET_SECONDS,
        expected_hours_per_job='11–13 after scheduling, extrapolated from200-URL pilot; not guaranteed',
        total_urls=125761,reused_pilot_urls=200,new_urls=125561,total_tasks=184546,
        manifests=manifests,estimated_artifact_gib=106,model_api_calls=0,stealth_sessions=0,
        procedure='Same frozen Chromium/source/classifier as pilot; max2 non-infrastructure observations per URL; preserve infrastructure errors and retry only within approved cap.',
        submitted=False)
    write(ROOT/'request.json',request)
    return request


def remaining_seconds(shard):
    approval=read(ROOT/'approval.json');job=os.environ.get('SLURM_JOB_ID')
    ids=approval['job_ids_by_shard'][str(shard)]
    if not (approval.get('approved') and approval['resources_per_job']==RESOURCES and job in ids):
        raise ValueError('Exact approved CPU resources and registered attempt required')
    if not all(str(x).isdecimal() for x in ids):raise ValueError('Expected numeric Slurm IDs')
    root=ROOT/f'shard{shard}'
    with (root/'budget.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        raw=subprocess.check_output(['sacct','-X','-n','-P','-j',','.join(ids),
            '--format=JobID,State,ElapsedRaw,ExitCode'],text=True)
        rows=accounting_rows(raw,ids);used=sum(x['elapsed_seconds'] for x in rows)
        receipt=dict(approved_seconds=BUDGET_SECONDS,consumed_seconds=used,
            remaining_seconds=max(0,BUDGET_SECONDS-used),attempts=rows,checked_epoch=time.time())
        write(root/'attempts.json',receipt)
    return receipt['remaining_seconds']


def load_worker():
    spec=importlib.util.spec_from_file_location('frozen_browser_checker',ROOT/'frozen/check_arm_task_urls.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


async def check_probe(worker, probe, directory):
    path=directory/'decision.json'
    if path.exists():
        row=read(path)
        if row['url']!=probe['url'] or row['probe_id']!=probe['probe_id']:raise ValueError('Wrong saved URL decision')
        return row
    paths=sorted(directory.glob('attempt*/result.json'))
    history=[read(x) for x in paths]
    observations=normal_observations(history)
    for _ in range(2):
        if final_decision(observations) is not None:break
        # Preserve every prior attempt, including timed-out workers without results.
        prior=[int(p.name[7:]) for p in directory.glob('attempt*') if p.name[7:].isdecimal()]
        number=max(prior,default=0)+1
        if history:await asyncio.sleep(2)
        row=await worker.attempt(Path(probe['chunk']),probe,directory/f'attempt{number:04d}')
        history.append(row);observations=normal_observations(history)
    decision=final_decision(observations)
    if decision is None:
        raise RuntimeError('Browser infrastructure failure; preserve attempts and repair before resuming')
    selected=observations[:2]
    result=dict(probe_id=probe['probe_id'],url=probe['url'],decision=decision,
        attempts=len(history),normal_observations=len(selected),
        reasons=[x['reason'] for x in selected],covered_candidate_tasks=len(probe['task_ids']),
        sources=probe['sources'],finished_epoch=time.time())
    write(path,result);return result


async def execute(shard):
    root=ROOT/f'shard{shard}';manifest=read(root/'manifest.json');job=os.environ.get('SLURM_JOB_ID')
    approval=read(ROOT/'approval.json')
    if sha(root/'manifest.json')!=approval['manifest_sha256_by_shard'][str(shard)]:
        raise ValueError('Approved manifest mismatch')
    for name,key in [('controller.py','controller_sha256'),('check_arm_task_urls.py','worker_sha256')]:
        if sha(ROOT/'frozen'/name)!=manifest[key]:raise ValueError('Frozen executable mismatch')
    if sha(PILOT/'run/agent-completion-review.json')!=manifest['pilot_audit_sha256']:
        raise ValueError('Reused pilot audit changed')
    for name,expected in manifest['source_hashes'].items():
        if sha(Path(manifest['source'])/name)!=expected:raise ValueError('Browser source changed')
    chunks={p['chunk']:p['chunk_sha256'] for p in manifest['probes']}
    for name,expected in chunks.items():
        if sha(Path(name))!=expected:raise ValueError('URL chunk changed')
    remaining=remaining_seconds(shard)-120
    if remaining<180:raise ValueError('Insufficient approved time remaining')
    live=subprocess.check_output(['scontrol','show','job',job,'-o'],text=True)
    fields=dict(x.split('=',1) for x in live.split() if '=' in x)
    if (fields.get('NumCPUs')!='8' or 'gres/gpu' in fields.get('AllocTRES','')
            or 'mem=32G' not in fields.get('AllocTRES','')):
        raise ValueError('Live allocation differs from8CPU/32GiB/0GPU approval')
    output=root/'run';output.mkdir(exist_ok=True)
    with (output/'active.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write(output/'launch_manifest.json',dict(job_id=job,shard=shard,
            resources=fields['AllocTRES'],manifest_sha256=sha(root/'manifest.json'),started_epoch=time.time()))
        with (SUPERVISOR/'registry-update.lock').open('a') as registry_lock:
            fcntl.flock(registry_lock,fcntl.LOCK_EX);reg=read(SUPERVISOR/'registry.json')
            key=f'task-pool-live-full-shard{shard}'
            item=next((j for j in reg['jobs'] if j['key']==key),None)
            if item is None:item=dict(key=key);reg['jobs'].append(item)
            item.update(job_id=job,priority=3,controller_root=str(output),training_root=str(output),evaluations=[],
                approved_total_seconds=BUDGET_SECONDS,budget_receipt=str(ROOT/'approval.json'),
                attempts_ledger=str(root/'attempts.json'),verified_complete=False,requires_user=False,
                require_agent_completion_review=True,agent_completion_reviewed=False,
                requested_endpoint=f"{len(manifest['probes'])} URL decisions/evidence; merge with audited pilot and other shards for125761URL/184546task coverage")
            write(SUPERVISOR/'registry.json',reg)
        worker=load_worker();queue=asyncio.Queue()
        for probe in manifest['probes']:queue.put_nowait(probe)
        completed=0;counts=Counter();last_write=0
        progress=output/'runtime/progress.log';progress.parent.mkdir(exist_ok=True)
        def status(force=False):
            nonlocal last_write
            now=time.time()
            if force or now-last_write>=10:
                write(output/'status.json',dict(complete=False,stage='collection',completed_urls=completed,
                    expected_urls=len(manifest['probes']),counts=dict(counts),updated_epoch=now))
                with progress.open('a') as f:f.write(f'{now} completed_urls={completed}/{len(manifest["probes"])}\n')
                last_write=now
        status(True)
        async def run_worker():
            nonlocal completed
            while not queue.empty():
                probe=queue.get_nowait()
                row=await check_probe(worker,probe,output/probe['probe_id'])
                completed+=1;counts[row['decision']]+=1;queue.task_done();status()
        workers=[asyncio.create_task(run_worker()) for _ in range(8)]
        try:
            await asyncio.wait_for(asyncio.gather(*workers),timeout=remaining)
        except BaseException as exc:
            for task in workers:task.cancel()
            await asyncio.gather(*workers,return_exceptions=True)
            write(output/'status.json',dict(complete=False,stage='failed',error=type(exc).__name__,
                detail=str(exc)[:800],completed_urls=completed,expected_urls=len(manifest['probes']),
                counts=dict(counts),updated_epoch=time.time()))
            raise
        finally:remaining_seconds(shard)
        result=dict(complete=True,returncode=0,stage='artifacts_ready',completed_urls=completed,
            expected_urls=len(manifest['probes']),counts=dict(counts),updated_epoch=time.time())
        write(output/'status.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare',action='store_true');group.add_argument('--execute',action='store_true')
    parser.add_argument('--shard',type=int,choices=range(SHARDS))
    args=parser.parse_args()
    if args.prepare:print(json.dumps(prepare(),indent=2))
    elif args.shard is None:parser.error('--execute requires --shard')
    else:print(json.dumps(asyncio.run(execute(args.shard)),indent=2))
