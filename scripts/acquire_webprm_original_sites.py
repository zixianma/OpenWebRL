#!/usr/bin/env python3
"""Bounded original Shopping/GitLab acquisition; no image execution.

Freezes and reuses the CMS range transport, byte ledger, archive reader and
process owner. Each new profile binds its official archive identity explicitly.
No CMS proposal, approval, launch, predecessor or budget is reused.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import resource
import secrets
import shutil
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[1]
SCRIPT=Path(__file__).resolve()
RUNTIME=Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011')
DEPENDENCIES=('scripts/acquire_webprm_cms_original.py','scripts/inspect_webprm_cms_archive.py',
              'openwebrl/__init__.py','openwebrl/webprm_lifecycle.py','openwebrl/jev_eval.py',
              'scripts/webprm_acquisition_recovery.py','scripts/webprm_cms_confirmation_transport.py')
PROFILES={
 'shopping':dict(filename='shopping_final_0712.tar',image_tag='shopping_final_0712:latest',
    archive_bytes=67575898112,drive_id='1gxXalk9O0p9eu1YkIJcmZta1nvvyAJpA',
    charged_download_bytes=70000000000,retained_bytes=72*2**30,expanded_regular_bytes=128*2**30),
 'gitlab':dict(filename='gitlab-populated-final-port8023.tar',image_tag='gitlab-populated-final-port8023:latest',
    archive_bytes=77755595776,drive_id='19W8qM0DPyRvWCLyQe0qtnCWAHGruolMR',
    charged_download_bytes=80000000000,retained_bytes=80*2**30,expanded_regular_bytes=192*2**30)}
RESOURCES=dict(account='zixianma',partition='gpu-h200',qos='normal',nodes=1,cpus=2,memory_gib=8,gpus=0,
    total_allocation_seconds_including_retries=7200,maximum_allocation_attempts=3,
    maximum_transfer_attempts=1,model_provider_calls=0,api_usd=0)


class Closed(ValueError):pass


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def checked(value):
    path=Path(value)
    if not path.is_absolute() or path.resolve()!=path or RUNTIME not in path.parents or any(p.is_symlink() for p in (path,*path.parents)):
        raise Closed('Canonical nonsymlink runtime path required')
    return path


def put(path,value):
    path=checked(path);raw=(json.dumps(value,sort_keys=True,indent=2)+'\n').encode()
    if len(raw)>4*2**20:raise Closed('Receipt exceeds four MiB')
    fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as out:out.write(raw);out.flush();os.fsync(out.fileno())


def read(path):return json.loads(checked(path).read_text())


def limits(profile):
    chunks=math.ceil(profile['archive_bytes']/(64*2**20))
    return dict(download_charged_bytes=profile['charged_download_bytes'],metadata_received_bytes=16*2**20,
        retained_bytes=profile['retained_bytes'],temporary_bytes=512*2**20,
        initial_persistent_free_bytes=profile['retained_bytes'],initial_temporary_free_bytes=2**30,
        closure_reserve_seconds=120,max_redirects_per_range=3,max_http_responses=chunks*4+8,
        receipt_reserve_bytes=16*2**20,network_connections=1,response_header_bytes_per_request=65536)


def prepare(root,site,*,predecessor=None,prior_job=None,repair_receipt=None):
    root=checked(root);profile=PROFILES[site]
    recipe=RUNTIME/'recipe-sources/webarena-deployment-README.md'
    if sha(recipe)!='50aac7e58d83f3e61b1e90ca8f5ffa40e1e725c45109601aa8a7fd4ce0f4a0ff':raise Closed('Official recipe changed')
    root.mkdir(mode=0o700,parents=True,exist_ok=False)
    sources={}
    for relative in DEPENDENCIES:
        target=checked(root/'dependencies'/relative);target.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
        with target.open('xb') as out:out.write((REPO/relative).read_bytes())
        sources[str(target.relative_to(root))]=sha(target)
    target=root/'acquisition.py'
    with target.open('xb') as out:out.write(SCRIPT.read_bytes())
    sources['acquisition.py']=sha(target)
    startup=dict(attempt_number=1,prior_allocation_seconds=0,requested_allocation_seconds=7200)
    inherited=None
    if predecessor is not None:
        previous=checked(predecessor);pp=read(previous/'proposal.json');pa=read(previous/'approval.json')
        if (pp.get('resources')!=RESOURCES or pp.get('profile')!=profile or pp.get('limits')!=limits(profile) or
                pp.get('site')!=site):raise Closed('Successor cannot change scientific/resource scope')
        for name,expected in pp['source_sha256'].items():
            if sha(previous/name)!=expected:raise Closed('Preserved predecessor source changed')
        if not str(prior_job).isdigit() or repair_receipt is None:raise Closed('Terminal job and tested repair receipt required')
        repair=read(checked(repair_receipt))
        if repair.get('replacement_source_sha256')!=sources:raise Closed('Source-fix test does not bind this replacement')
        snapshot=json.loads(subprocess.check_output(['scontrol','--json','show','job',str(prior_job)],text=True,timeout=15))
        spec=importlib.util.spec_from_file_location('startup_recovery',root/'dependencies/scripts/webprm_acquisition_recovery.py')
        recovery=importlib.util.module_from_spec(spec);spec.loader.exec_module(recovery)
        startup=recovery.recovery_proof(previous,pp,pa,read(previous/'launch.json'),snapshot,repair)
        if startup['predecessor_job_id']!=str(prior_job):raise Closed('Terminal scheduler job changed')
        inherited=pa['exact_user_approval']
    proposal=dict(schema_version=2,status='pending_user_approval',root=str(root),site=site,
        resources=RESOURCES,profile=profile,limits=limits(profile),source_sha256=sources,startup_recovery=startup,
        recipe_sha256=sha(recipe),scope='Official original archive download and bounded offline internal identity only.',
        no_extraction=True,no_services=True,no_browser_calls=True,no_network_retry=True,no_mirror_fallback=True,
        recovery_policy='At most three manually prepared startup attempts share 7200 seconds. Terminal scheduler accounting and a diagnosed tested source fix are required; creating any network claim or ledger permanently prohibits successors.',
        profile_binding='Transport filename/size/DriveID, range ledger caps, and inspector exact image tag/expanded-size bounds are set once before the fork; effective configuration is saved.',
        identity='Official HTTPS source plus exact recorded size/name; pin observed SHA256/config/layer identities. No independently published historical archive checksum.',
        file_semantics='One transfer containing contiguous 64MiB ranges or one exact-size full200 body reserved before reading; at most three reviewed Google redirect/confirmation steps per range, one fixed original-file confirmation form only. Strong ETag, exact filename and response framing required. Failed transfers/launch claims are retained; no resume or network retry.',
        confirmation=dict(maximum_forms=1,fixed_drive_id=profile['drive_id'],
            fixed_action='https://drive.usercontent.google.com/download',
            fields=['id','export','confirm','uuid'],metadata_body_cap_bytes=16*2**20,
            metadata_also_included_in_download_charge=True,
            implementation='reviewed CMS confirmation transport, isolated site-bound globals before fork; legacy receipt protocol label identifies the implementation, original_drive_id identifies the site'),
        storage='Archive streams directly into retained runtime; no archive copy, extraction or import. Expanded-byte bounds apply only to read-only tar metadata inspection, not storage allocation.')
    if inherited:
        put(previous/'successor-claim.json',dict(root=str(root),proposal_sha256=digest(proposal)))
    put(root/'proposal.json',proposal)
    put(root/'approval.json',dict(status='approved' if inherited else 'pending',proposal_sha256=digest(proposal),
        exact_user_approval=inherited,inherited_startup_only=bool(inherited)))
    return proposal


def approved(root):
    p,a=read(root/'proposal.json'),read(root/'approval.json')
    profile=PROFILES.get(p.get('site'))
    if (profile is None or p.get('root')!=str(root) or p.get('profile')!=profile or p.get('resources')!=RESOURCES or
        p.get('limits')!=limits(profile) or a.get('status')!='approved' or a.get('proposal_sha256')!=digest(p) or
        not isinstance(a.get('exact_user_approval'),str) or not a['exact_user_approval'].strip()):raise Closed('Exact new site acquisition approval required')
    expected={'acquisition.py',*('dependencies/'+x for x in DEPENDENCIES)}
    if set(p.get('source_sha256',{}))!=expected or SCRIPT!=root/'acquisition.py':raise Closed('Run only the frozen acquisition package')
    for name,expected_sha in p['source_sha256'].items():
        if sha(checked(root/name))!=expected_sha:raise Closed('Frozen dependency changed')
    validate_startup(root,p)
    return p


def validate_startup(root,p):
    startup=p.get('startup_recovery',{})
    if startup.get('attempt_number')==1:
        if startup!=dict(attempt_number=1,prior_allocation_seconds=0,requested_allocation_seconds=RESOURCES['total_allocation_seconds_including_retries']):raise Closed('Initial attempt accounting changed')
        return
    previous=checked(startup.get('predecessor_root',''))
    pp,pa=read(previous/'proposal.json'),read(previous/'approval.json')
    if (pp.get('resources')!=p['resources'] or pp.get('profile')!=p['profile'] or pp.get('limits')!=p['limits'] or
            read(previous/'successor-claim.json')!=dict(root=str(root),proposal_sha256=digest(p))):raise Closed('Successor ownership or scope changed')
    for name,expected in pp['source_sha256'].items():
        if sha(previous/name)!=expected:raise Closed('Predecessor source changed')
    spec=importlib.util.spec_from_file_location('startup_recovery',root/'dependencies/scripts/webprm_acquisition_recovery.py')
    recovery=importlib.util.module_from_spec(spec);spec.loader.exec_module(recovery)
    if (startup['source_fix']['replacement_source_sha256']!=p['source_sha256'] or
            recovery.recovery_proof(previous,pp,pa,read(previous/'launch.json'),startup['terminal_snapshot'],startup['source_fix'])!=startup):raise Closed('Invalid inherited startup recovery')


def load_library(root,profile):
    """Bind one isolated frozen library to one recorded site profile, before fork."""
    base=checked(root/'dependencies');sys.path.insert(0,str(base))
    path=base/'scripts/acquire_webprm_cms_original.py'
    spec=importlib.util.spec_from_file_location('frozen_original_acquisition',path)
    lib=importlib.util.module_from_spec(spec);spec.loader.exec_module(lib)
    from scripts import inspect_webprm_cms_archive as inspector
    import openwebrl.webprm_lifecycle as lifecycle
    import openwebrl.jev_eval as jev_eval
    # Inspection's function globals belong to this frozen inspector. No source
    # is rewritten and no approved CMS source/configuration is modified.
    if Path(inspector.__file__).resolve()!=base/'scripts/inspect_webprm_cms_archive.py':raise Closed('Inspector escaped frozen dependency root')
    if (Path(lifecycle.__file__).resolve()!=base/'openwebrl/webprm_lifecycle.py' or
        Path(jev_eval.__file__).resolve()!=base/'openwebrl/jev_eval.py'):
        raise Closed('Process owner escaped frozen dependency root')
    lib.ARCHIVE_BYTES=profile['archive_bytes'];lib.ARCHIVE_NAME=profile['filename']
    lib.DRIVE_ID=profile['drive_id'];lib.SOURCE_URL='https://drive.google.com/uc?export=download&id='+profile['drive_id']+'&confirm=t'
    lib.LIMITS=limits(profile)
    inspector.IMAGE_TAG=profile['image_tag']
    if profile in PROFILES.values():
        bind_confirmation(root,lib,profile)
    return lib,inspector


def bind_confirmation(root,lib,profile):
    """Bind the reviewed frozen handler once in this isolated site's process.

    The handler's superclass retains its imported CMS module, so bind that
    module too. Both identities are checked before any network operation.
    The running CMS and preserved acquisition packages are never modified.
    """
    base=checked(root/'dependencies')
    path=base/'scripts/webprm_cms_confirmation_transport.py'
    spec=importlib.util.spec_from_file_location('frozen_site_confirmation',path)
    transport=importlib.util.module_from_spec(spec);spec.loader.exec_module(transport)
    if Path(transport.cms.__file__).resolve()!=base/'scripts/acquire_webprm_cms_original.py':
        raise Closed('Confirmation superclass escaped frozen dependency root')
    for bound in (transport,transport.cms):
        bound.ARCHIVE_BYTES=profile['archive_bytes'];bound.ARCHIVE_NAME=profile['filename']
        bound.DRIVE_ID=profile['drive_id'];bound.SOURCE_URL=lib.SOURCE_URL
        bound.LIMITS=lib.LIMITS
    lib.TransferLedger=transport.CumulativeLedger
    lib.download=transport.download
    lib.confirmation_transport=transport


def previous_usage(root,p,lib):
    """Count retained startup artifacts and scratch across the bounded lineage."""
    retained=temporary=0;seen={str(root)};cursor=p
    while cursor['startup_recovery']['attempt_number']>1:
        prior=checked(cursor['startup_recovery']['predecessor_root'])
        if str(prior) in seen or len(seen)>=RESOURCES['maximum_allocation_attempts']:raise Closed('Cyclic or excessive startup lineage')
        seen.add(str(prior));retained+=lib.apparent_bytes(prior)
        for context_path in prior.glob('attempt-*/context.json'):
            scratch=lib.checked_path(Path(read(context_path)['scratch']))
            if scratch.parent!=Path('/tmp') or not scratch.name.startswith(('webprm-original-','webprm-map-backend-')):raise Closed('Unexpected preserved scratch ownership')
            if scratch.exists():temporary+=lib.apparent_bytes(scratch)
        cursor=read(prior/'proposal.json')
    return retained,temporary


def context(snapshot,job,now,*,maximum_seconds=7200):
    try:
        if snapshot.get('errors')!=[] or len(snapshot['jobs'])!=1:raise ValueError('One error-free scheduler job required')
        row=snapshot['jobs'][0]
        def number(key):
            val=row[key]
            if val.get('set') is not True or val.get('infinite') is not False or type(val.get('number')) is not int:raise ValueError(key)
            return val['number']
        start,end,seconds=number('start_time'),number('end_time'),number('time_limit')*60
        tres=dict(x.split('=',1) for x in row['tres_alloc_str'].split(','))
        if (str(row['job_id'])!=job or row['user_id']!=os.getuid() or row['job_state']!=['RUNNING'] or row['restart_cnt']!=0 or
            row['account']!='zixianma' or row['partition']!='gpu-h200' or row['qos']!='normal' or number('cpus')!=2 or
            number('node_count')!=1 or number('memory_per_node')!=8192 or tres.get('cpu')!='2' or tres.get('node')!='1' or
            tres.get('mem') not in ('8G','8192M') or any(k.startswith('gres/gpu') and v!='0' for k,v in tres.items())):raise ValueError('Resource mismatch')
        deadline=min(end,start+seconds)
        if not math.isfinite(now) or not 0<seconds<=maximum_seconds or not start<=now<deadline-120:raise ValueError('Bounded epoch clock required')
    except (KeyError,ValueError,TypeError,AttributeError) as error:raise Closed('Invalid original-site allocation proof') from error
    return dict(job_id=job,start_epoch=start,allocation_deadline=deadline,work_deadline=deadline-120,
                scheduler_end_epoch=end,scheduler_time_limit_seconds=seconds)


def launch(root):
    p=approved(root);lib,_=load_library(root,p['profile']);nonce=secrets.token_hex(16)
    seconds=p['startup_recovery']['requested_allocation_seconds']
    duration=f'{seconds//3600:02d}:{seconds//60%60:02d}:00'
    argv=['salloc','--account=zixianma','--partition=gpu-h200','--qos=normal','--nodes=1','--ntasks=1',
        '--cpus-per-task=2','--mem=8G','--time='+duration,'--comment=webprm-acquire:'+nonce,'--job-name=wa-'+p['site']+'-identity','--chdir='+str(root),
        'srun','--ntasks=1','--cpus-per-task=2',sys.executable,'-I','-B',str(SCRIPT),'--run','--root',str(root),'--nonce',nonce]
    put(root/'launch.json',dict(nonce=nonce,proposal_sha256=digest(p),requested_allocation_seconds=seconds,argv=argv,epoch=time.time()))
    env=lib.safe_environment(root/'launch-cache');env={k:v for k,v in env.items() if not k.startswith('SLURM_')}
    os.chdir(root);os.execvpe(argv[0],argv,env)


def run(root,nonce):
    p=approved(root);claim=read(root/'launch.json');job=os.environ.get('SLURM_JOB_ID','')
    if (not nonce or claim.get('nonce')!=nonce or claim.get('proposal_sha256')!=digest(p) or not job.isdigit() or
        not re.search(r'job[_/\-]'+re.escape(job)+r'(?:/|$)',Path('/proc/self/cgroup').read_text())):raise Closed('Owned Slurm allocation claim required')
    put(root/'run-claim.json',dict(job_id=job,nonce=nonce,epoch=time.time()))
    try:
        raw=subprocess.check_output(['scontrol','--json','show','job',job],text=True,timeout=15)
        put(root/'scheduler.json',json.loads(raw));clock=context(json.loads(raw),job,time.time(),maximum_seconds=p['startup_recovery']['requested_allocation_seconds'])
        lib,inspector=load_library(root,p['profile']);profile=p['profile']
        prior_retained,prior_temporary=previous_usage(root,p,lib)
        if prior_retained>=profile['retained_bytes'] or prior_temporary>=512*2**20:raise Closed('Preserved startups exhausted storage cap')
        lib.LIMITS=dict(lib.LIMITS,temporary_bytes=512*2**20-prior_temporary)
        if shutil.disk_usage(root).free<profile['retained_bytes'] or shutil.disk_usage('/tmp').free<2**30:raise Closed('Approved free space unavailable')
        scratch=Path('/tmp')/('webprm-original-'+p['site']+'-'+job+'-'+secrets.token_hex(4));scratch.mkdir(mode=0o700)
        attempt=checked(root/('attempt-'+job));attempt.mkdir(mode=0o700)
        put(attempt/'context.json',dict(clock,scratch=str(scratch),effective_profile=profile,
            effective_limits=lib.LIMITS,inspector_tag=inspector.IMAGE_TAG,proposal_sha256=digest(p)))

        def worker(output,scratch,deadline):
            os.setsid();os.chdir(output);os.environ.update(lib.safe_environment(scratch))
            resource.setrlimit(resource.RLIMIT_AS,(6*2**30,6*2**30))
            resource.setrlimit(resource.RLIMIT_FSIZE,(profile['retained_bytes'],profile['retained_bytes']))
            try:
                put(root/'network-claim.json',dict(job_id=job,epoch=time.time(),maximum_transfer_attempts=1))
                ledger=lib.TransferLedger(output,cap=profile['charged_download_bytes'],metadata_cap=16*2**20)
                transfer=lib.download(output,ledger,deadline,size=profile['archive_bytes'])
                proof=inspector.inspection(transfer['archive_path'],deadline=deadline,
                    limits=inspector.Limits(archive_bytes=profile['archive_bytes'],layer_members=2000000,
                        cumulative_regular_bytes=profile['expanded_regular_bytes']))
                if proof['archive_sha256']!=transfer['archive_sha256'] or proof['image_tag']!=profile['image_tag']:raise Closed('Transferred/inspected archive identity differs')
                lib.put(output/'archive-identity.json',proof)
                lib.put(output/'stage-result.json',dict(status='acquired_internal_identity_verified',archive_sha256=proof['archive_sha256'],
                    ready_to_host=False,extracted=False,model_calls=0,browser_calls=0))
            except BaseException as error:
                lib.put(output/'failure.json',dict(error_type=type(error).__name__,error=str(error)[:2000],epoch=time.time()))
                raise SystemExit(1)

        closure=lib.own_worker(attempt,scratch,clock['work_deadline'],target=worker,prior_retained_bytes=prior_retained)
        result=read(attempt/'stage-result.json')
        if (result.get('status')!='acquired_internal_identity_verified' or closure['remaining_owned_processes'] or
            lib.apparent_bytes(root)+prior_retained>profile['retained_bytes'] or lib.apparent_bytes(scratch)+prior_temporary>512*2**20 or
            time.time()>=clock['allocation_deadline']):raise Closed('Final closure/storage/clock verification failed')
        put(root/'completed.json',dict(status='acquisition_complete',site=p['site'],job_id=job,result=result,
            retained_bytes=lib.apparent_bytes(root)+prior_retained,temporary_bytes=lib.apparent_bytes(scratch)+prior_temporary,epoch=time.time()))
    except BaseException as error:
        put(root/'failure.json',dict(error_type=type(error).__name__,error=str(error)[:2000],epoch=time.time()))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group(required=True)
    for name in ('prepare','launch','run'):mode.add_argument('--'+name,action='store_true')
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--site',choices=tuple(PROFILES));parser.add_argument('--nonce',default='')
    parser.add_argument('--predecessor',type=Path);parser.add_argument('--prior-job');parser.add_argument('--repair-receipt',type=Path)
    args=parser.parse_args();root=checked(args.root)
    if args.prepare:
        if args.site is None:parser.error('--prepare requires --site')
        print(json.dumps(dict(proposal_sha256=digest(prepare(root,args.site,predecessor=args.predecessor,prior_job=args.prior_job,repair_receipt=args.repair_receipt)),root=str(root))))
    elif args.launch:launch(root)
    else:run(root,args.nonce)


if __name__=='__main__':main()
