#!/usr/bin/env python3
"""Acquire the three published WebArena Map frontend files after exact approval.

No extraction, Docker/Apptainer, browser, GPU, model, backend-data acquisition,
automatic retry, or SSH. The allocation owner awaits this synchronous worker.
"""
import argparse
import email.message
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import re
import secrets
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

RUNTIME = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011')
SCRIPT = Path(__file__).resolve()
METADATA = RUNTIME/'next-stage-preparation/map-zenodo-record-12636845.json'
METADATA_SHA = '40bba6f09df4c57f047b86710dc53fef2463130c6c758316c9e3466389cc350d'
FILES = (
    ('openstreetmap-website-db.tar.gz',104093527,'a93ef4f0560d55234099a73137d5266d'),
    ('openstreetmap-website-web.tar.gz',634664156,'d3564d63255077ac7dcc05e5d42b5c00'),
    ('openstreetmap-website.tar.gz',493171981,'2f4423c7d4982246825fec3bd03ce9de'))
RESOURCES = dict(account='zixianma',partition='gpu-h200',qos='normal',nodes=1,cpus=2,
    memory_gib=4,gpus=0,total_allocation_seconds_including_retries=1800,
    maximum_allocation_attempts=1,maximum_transfers_per_file=1,api_usd=0,model_calls=0)
LIMITS = dict(charged_download_bytes=1300000000,retained_bytes=2*2**30,
    temporary_bytes=64*2**20,response_header_reservation=65536,
    network_timeout_seconds=30,cleanup_reserve_seconds=60,maximum_responses=3)


class Closed(ValueError):pass


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def checked(value):
    path=Path(value)
    if not path.is_absolute() or path.resolve()!=path or RUNTIME not in path.parents or any(x.is_symlink() for x in (path,*path.parents)):
        raise Closed('Only canonical nonsymlink paths below the project runtime are writable')
    return path


def put(path,value):
    path=checked(path)
    raw=(json.dumps(value,sort_keys=True,indent=2)+'\n').encode()
    if len(raw)>2**20:raise Closed('Receipt exceeded one MiB')
    fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as out:out.write(raw);out.flush();os.fsync(out.fileno())


def read(path):return json.loads(checked(path).read_text())


def files():
    return [dict(filename=n,bytes=s,published_md5=m,url=f'https://zenodo.org/api/records/12636845/files/{n}/content') for n,s,m in FILES]


def prepare(root):
    root=checked(root)
    if sha(METADATA)!=METADATA_SHA:raise Closed('Reviewed publication metadata changed')
    root.mkdir(mode=0o700,parents=True,exist_ok=False)
    source=root/'acquisition.py'
    with source.open('xb') as out:out.write(SCRIPT.read_bytes())
    proposal=dict(schema_version=1,status='pending_user_approval',resources=RESOURCES,limits=LIMITS,
        root=str(root),frozen_source=str(source),source_sha256=sha(source),files=files(),
        publication=dict(doi='10.5281/zenodo.12636845',metadata_sha256=METADATA_SHA,
            metadata_source='https://zenodo.org/api/records/12636845',
            discovery='BrowserGym recommended gasse/webarena-setup at 8d0e0a20d50e4a8c02d1520195182e19fc3c080f'),
        scope='Download and published checksum verification only. No extraction, services or benchmark.',
        total_file_bytes=sum(x[1] for x in FILES),
        failures='Preserve every partial file and reserve. One allocation and one request per file; no retries or redirects.',
        accounting='Full declared body plus 64KiB header reservation per request, retained even on unknown completion. TLS/TCP framing excluded.',
        limitations='Published MD5 checks transport identity; compute SHA256 locally. This is not a hosting or reset proof.')
    put(root/'proposal.json',proposal)
    put(root/'approval.json',dict(status='pending',proposal_sha256=digest(proposal),exact_user_approval=None))
    return proposal


def approved(root):
    p,a=read(root/'proposal.json'),read(root/'approval.json')
    if (p.get('resources')!=RESOURCES or p.get('limits')!=LIMITS or p.get('files')!=files() or
        p.get('root')!=str(root) or p.get('frozen_source')!=str(root/'acquisition.py') or
        sha(root/'acquisition.py')!=p.get('source_sha256') or
        a.get('status')!='approved' or a.get('proposal_sha256')!=digest(p) or
        not isinstance(a.get('exact_user_approval'),str) or not a['exact_user_approval'].strip()):
        raise Closed('Exact reviewed Map acquisition approval required')
    if SCRIPT != root/'acquisition.py' or sha(SCRIPT)!=p['source_sha256']:
        raise Closed('Execute only the approved frozen acquisition source')
    return p


def safe_env(root):
    env={k:v for k,v in os.environ.items() if not k.startswith(('SLURM_','SALLOC_','SBATCH_','SRUN_','APPTAINER_','SINGULARITY_'))}
    for key in ('TMPDIR','XDG_CONFIG_HOME','XDG_CACHE_HOME','XDG_DATA_HOME','XDG_RUNTIME_DIR','PIP_CACHE_DIR'):
        path=checked(root/key.lower());path.mkdir(mode=0o700,exist_ok=False);env[key]=str(path)
    env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',CUDA_VISIBLE_DEVICES='',
               HISTFILE='/dev/null',BASH_ENV='/dev/null',ENV='/dev/null',OMP_NUM_THREADS='1')
    return env


def launch(root):
    p=approved(root)
    nonce=secrets.token_hex(16)
    argv=['salloc','--account=zixianma','--partition=gpu-h200','--qos=normal','--nodes=1',
        '--ntasks=1','--cpus-per-task=2','--mem=4G','--time=00:30:00','--job-name=wa-map-files',
        '--chdir='+str(root),'srun','--ntasks=1','--cpus-per-task=2',sys.executable,'-I','-B',
        str(SCRIPT),'--run','--root',str(root),'--nonce',nonce]
    put(root/'launch.json',dict(nonce=nonce,proposal_sha256=digest(p),argv=argv,epoch=time.time()))
    env=safe_env(root)
    os.chdir(root)
    os.execvpe(argv[0],argv,env)


def allocation(snapshot,job,now):
    try:
        if snapshot.get('errors')!=[] or len(snapshot['jobs'])!=1:raise ValueError('One scheduler job required')
        r=snapshot['jobs'][0]
        def integer(name):
            v=r[name]
            if v.get('set') is not True or v.get('infinite') is not False or type(v.get('number')) is not int:raise ValueError(name)
            return v['number']
        start,end,seconds=integer('start_time'),integer('end_time'),60*integer('time_limit')
        tres=dict(x.split('=',1) for x in r['tres_alloc_str'].split(','))
        if (str(r['job_id'])!=job or r['user_id']!=os.getuid() or r['job_state']!=['RUNNING'] or
            r['restart_cnt']!=0 or r['account']!='zixianma' or r['partition']!='gpu-h200' or r['qos']!='normal' or
            integer('cpus')!=2 or integer('node_count')!=1 or integer('memory_per_node')!=4096 or
            tres.get('cpu')!='2' or tres.get('node')!='1' or tres.get('mem') not in ('4G','4096M') or
            any(k.startswith('gres/gpu') and v!='0' for k,v in tres.items())):raise ValueError('Resources mismatch')
        deadline=min(end,start+seconds)
        if not math.isfinite(now) or not 0<seconds<=1800 or not start<=now<deadline-60:raise ValueError('Clock mismatch')
    except (KeyError,ValueError,TypeError,AttributeError) as error:raise Closed('Invalid scheduler allocation proof') from error
    return dict(job_id=job,start_epoch=start,deadline=deadline,work_deadline=deadline-60,
                scheduler_end_epoch=end,scheduler_time_limit_seconds=seconds)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):raise Closed('Redirect is outside the three fixed transfer requests')


def headers_ok(response,item):
    disposition=email.message.Message();disposition['content-disposition']=response.headers.get('Content-Disposition','')
    if (response.status!=200 or response.url!=item['url'] or
        response.headers.get('Content-Encoding','identity')!='identity' or
        response.headers.get('Content-Length')!=str(item['bytes']) or
        disposition.get_filename()!=item['filename']):raise Closed('Unexpected publication response identity or length')


def transfer(root,item,index,deadline,opener=None):
    if time.time()>=deadline:raise Closed('Transfer deadline exhausted')
    put(root/f'transfer-{index}-reservation.json',dict(item=item,charged_bytes=item['bytes']+65536,epoch=time.time()))
    target=checked(root/item['filename'])
    sha256,md5=hashlib.sha256(),hashlib.md5(usedforsecurity=False)
    total=0
    # Bound headers before http.client receives them: a status line plus at
    # most 63 header/terminating lines, each read at most 1024 bytes. Oversized
    # responses fail while retaining the complete 64KiB header reservation.
    http.client._MAXLINE=1023
    http.client._MAXHEADERS=62
    opener=opener or urllib.request.build_opener(NoRedirect())
    request=urllib.request.Request(item['url'],headers={'Accept-Encoding':'identity','User-Agent':'OpenWebRL-Map-identity'})
    with opener.open(request,timeout=30) as response:
        headers_ok(response,item)
        fd=os.open(target,os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600)
        with os.fdopen(fd,'wb') as out:
            while total<item['bytes']:
                if time.time()>=deadline:raise Closed('Transfer deadline exhausted')
                chunk=response.read(min(2**20,item['bytes']-total))
                if not chunk:raise Closed('Incomplete publication body')
                out.write(chunk);sha256.update(chunk);md5.update(chunk);total+=len(chunk)
            if response.read(1):raise Closed('Publication body exceeded declared length')
            out.flush();os.fsync(out.fileno())
    if md5.hexdigest()!=item['published_md5']:raise Closed('Published MD5 mismatch; partial evidence preserved')
    receipt=dict(filename=item['filename'],bytes=total,md5=md5.hexdigest(),sha256=sha256.hexdigest(),
                 charged_bytes=item['bytes']+65536,epoch=time.time())
    put(root/f'transfer-{index}-verified.json',receipt)
    return receipt


def run(root,nonce):
    p=approved(root);claim=read(root/'launch.json')
    if not nonce or claim.get('nonce')!=nonce or claim.get('proposal_sha256')!=digest(p):raise Closed('Unknown allocation claim')
    job=os.environ.get('SLURM_JOB_ID','')
    if not job.isdigit() or not re.search(r'job[_/\-]'+re.escape(job)+r'(?:/|$)',Path('/proc/self/cgroup').read_text()):raise Closed('Worker must belong to its owned Slurm job')
    put(root/'run-claim.json',dict(job_id=job,nonce=nonce,epoch=time.time()))
    try:
        raw=subprocess.check_output(['scontrol','--json','show','job',job],text=True,timeout=15)
        put(root/'scheduler.json',json.loads(raw))
        context=allocation(json.loads(raw),job,time.time())
        put(root/'allocation.json',context)
        def expired(signum,frame):raise Closed('Owned allocation signal or work deadline')
        signal.signal(signal.SIGALRM,expired);signal.signal(signal.SIGTERM,expired)
        signal.setitimer(signal.ITIMER_REAL,max(.001,context['work_deadline']-time.time()))
        stat=os.statvfs(root)
        if stat.f_bavail*stat.f_frsize<2*2**30:raise Closed('Two GiB retained headroom required')
        if sum(i['bytes']+65536 for i in files())>LIMITS['charged_download_bytes']:raise Closed('Download cap too small')
        receipts=[transfer(root,item,i,context['work_deadline']) for i,item in enumerate(files())]
        put(root/'completed.json',dict(status='acquired_published_checksums_verified',files=receipts,
            charged_bytes=sum(x['charged_bytes'] for x in receipts),model_calls=0,browser_calls=0,
            extracted=False,remaining_owned_children=[],epoch=time.time()))
    except BaseException as error:
        put(root/'failure.json',dict(error_type=type(error).__name__,error=str(error)[:2000],epoch=time.time()))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    for name in ('prepare','launch','run'):mode.add_argument('--'+name,action='store_true')
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--nonce',default='')
    args=parser.parse_args();root=checked(args.root)
    if args.prepare:print(json.dumps({'proposal_sha256':digest(prepare(root)),'root':str(root)}))
    elif args.launch:launch(root)
    else:run(root,args.nonce)


if __name__=='__main__':main()
