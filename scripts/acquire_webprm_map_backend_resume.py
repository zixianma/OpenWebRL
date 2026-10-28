#!/usr/bin/env python3
"""One explicitly approved append-only Map transfer exception; preparation is offline."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import resource
import secrets
import shutil
import subprocess
import sys
import time
SCRIPT=Path(__file__).resolve()
SOURCE_ROOT=SCRIPT.parent/'dependencies' if SCRIPT.name=='acquisition.py' else SCRIPT.parents[1]
sys.path.insert(0,str(SOURCE_ROOT))
from scripts import acquire_webprm_map_backend as backend
from scripts import webprm_map_resume as resume
from scripts import webprm_map_resume_proof as recovery
from scripts import webprm_cms_oci_transport as oci
Closed,checked,put,read,sha,digest=backend.Closed,backend.checked,backend.put,backend.read,backend.sha,backend.digest
DEPENDENCIES=tuple(dict.fromkeys((*backend.DEPS,'scripts/acquire_webprm_map_backend.py',
    'scripts/webprm_map_resume.py','scripts/webprm_map_resume_proof.py','scripts/webprm_cms_oci_transport.py')))
EVIDENCE={'prior-failure-audit.json':'audit.json','prior-terminal-scheduler.json':'terminal-scheduler.json',
    'prior-prefix-proof.json':'closed-prefix-sha256.json','prior-scratch-observation.json':'prior-scratch-observation.json'}


def validate_sources(root,p):
    expected={'acquisition.py',*('dependencies/'+x for x in DEPENDENCIES),
        *('metadata/'+x.name for x in backend.metadata_files()),*('metadata/'+x for x in EVIDENCE)}
    if set(p.get('source_sha256',{}))!=expected:raise Closed('Incomplete frozen Map exception source identity')
    for name,value in p['source_sha256'].items():
        if sha(checked(root/name))!=value:raise Closed('Frozen Map exception source changed')


def exception_approval(proof):
    return dict(protocol=resume.PROTOCOL,approved=True,proof_sha256=digest(proof),
        additional_transfer_campaigns=1,additional_allocation_attempts=1,
        requested_allocation_seconds=proof['requested_allocation_seconds'],
        append_original_partial_inode=True,maximum_tls_handshake_reconnects=3,
        further_successors_allowed=False)


def prepare(root,previous,evidence):
    root,previous,evidence=checked(root),checked(previous),checked(evidence)
    if (previous/'successor-claim.json').exists():raise Closed('Predecessor already has an exclusive successor')
    root.mkdir(mode=0o700,parents=True,exist_ok=False);sources={}
    for name in DEPENDENCIES:
        target=root/'dependencies'/name;target.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
        with target.open('xb') as output:output.write((SOURCE_ROOT/name).read_bytes())
        sources[str(target.relative_to(root))]=sha(target)
    with (root/'acquisition.py').open('xb') as output:output.write(SCRIPT.read_bytes())
    sources['acquisition.py']=sha(root/'acquisition.py');(root/'metadata').mkdir(mode=0o700)
    paths={x.name:x for x in backend.metadata_files()}
    paths.update({name:evidence/source for name,source in EVIDENCE.items()})
    for name,path in paths.items():
        target=root/'metadata'/name
        with target.open('xb') as output:output.write(checked(path).read_bytes())
        sources['metadata/'+name]=sha(target)
    proof=recovery.proof(backend,previous,root/'metadata',sources);items,images=backend.inputs(root)
    p=dict(schema_version=1,status='pending_user_approval',root=str(root),site='map_backend',
        protocol=resume.PROTOCOL,resources=backend.RESOURCES,limits=backend.LIMITS,profile=backend.PROFILE,
        source_sha256=sources,transfer_exception=proof,items=items,total_unique_body_bytes=sum(x['bytes'] for x in items),
        scope='One additional transfer campaign and one 13020-second allocation within original 14400 seconds; 2CPU/8GiB, zero GPUs/model/browser/API. No hosting/extraction.',
        storage='Append only to the original partial inode after fixed-prefix rehash; preserve its original prefix and all failure evidence. No overwrite, truncation, rename, delete, hardlink, or prefix copy. Reserve full future predecessor growth before admission plus4MiB prior scratch inside200GiB/512MiB totals.',
        retries='At most3 global additional pre-response TLS-handshake-timeout connections. Each gets a new durable byte/header/response reservation; failed reservations remain charged. No retry of accepted headers, status/entity/body failures or any other transport error. No further successor.',
        identity='Pinned original S3 ETag/length and exact206 If-Match ranges. New content multipart-MD5 checks use candidate part boundaries and accept only exact published ETag match; local SHA256 also recorded. No published SHA256. Pinned OCI compressed checksums, not historical binary equivalence.',
        multipart_candidate_part_bytes=resume.PART_BYTES,ready_to_host=False)
    put(previous/'successor-claim.json',dict(root=str(root),proposal_sha256=digest(p),protocol=resume.PROTOCOL))
    put(root/'proposal.json',p);put(root/'approval.json',dict(status='pending',proposal_sha256=digest(p),
        exact_user_approval=None,additional_transfer_exception=None,inherited_authority=False))
    return p


def approved(root):
    p,a=read(root/'proposal.json'),read(root/'approval.json')
    if (SCRIPT!=root/'acquisition.py' or p.get('root')!=str(root) or p.get('protocol')!=resume.PROTOCOL or
        p.get('resources')!=backend.RESOURCES or p.get('limits')!=backend.LIMITS or p.get('profile')!=backend.PROFILE or
        a.get('status')!='approved' or a.get('proposal_sha256')!=digest(p) or
        not isinstance(a.get('exact_user_approval'),str) or not a['exact_user_approval'].strip() or
        a.get('inherited_authority') is not False):raise Closed('Exact new Map append/transfer/TLS exception approval required')
    validate_sources(root,p)
    previous=checked(p['transfer_exception']['predecessor_root'])
    if read(previous/'successor-claim.json')!=dict(root=str(root),proposal_sha256=digest(p),protocol=resume.PROTOCOL):raise Closed('Exclusive predecessor-successor ownership differs')
    proof=recovery.proof(backend,previous,root/'metadata',p['source_sha256'])
    if (proof!=p['transfer_exception'] or a.get('additional_transfer_exception')!=exception_approval(proof) or
            p.get('items')!=backend.inputs(root)[0] or p.get('multipart_candidate_part_bytes')!=resume.PART_BYTES):
        raise Closed('Exact cumulative exception/prefix/source approval differs')
    return p


def library(root):
    lib=backend.library(root)
    for mod in (backend,resume,recovery,oci,oci.cms,oci.registry):
        if not Path(mod.__file__).resolve().is_relative_to(root/'dependencies'):raise Closed('Map dependency escaped frozen root')
    lib.LIMITS=dict(backend.LIMITS,temporary_bytes=backend.LIMITS['temporary_bytes']-resume.SCRATCH_RESERVE)
    oci.cms.LIMITS=lib.LIMITS
    return lib


def storage_reservation(lib,previous,proof):
    prior=lib.apparent_bytes(previous)
    reserved=prior+proof['preserved_partial_append_growth_bytes']
    if reserved>=backend.LIMITS['retained_bytes']-backend.LIMITS['receipt_reserve_bytes']:
        raise Closed('Preserved prefix/future growth exhausted storage cap')
    return prior,reserved


def launch(root):
    p=approved(root);lib=library(root);nonce=secrets.token_hex(16);seconds=p['transfer_exception']['requested_allocation_seconds']
    argv=['salloc','--account=zixianma','--partition=gpu-h200','--qos=normal','--nodes=1','--ntasks=1','--cpus-per-task=2','--mem=8G',
        f'--time={seconds//3600:02d}:{seconds//60%60:02d}:00','--comment=webprm-acquire:'+nonce,'--job-name=wa-map-resume',
        '--chdir='+str(root),'srun','--ntasks=1','--cpus-per-task=2',sys.executable,'-I','-B',str(SCRIPT),'--run','--root',str(root),'--nonce',nonce]
    put(root/'launch.json',dict(nonce=nonce,proposal_sha256=digest(p),requested_allocation_seconds=seconds,argv=argv,epoch=time.time()))
    env={k:v for k,v in lib.safe_environment(root/'launch-cache').items() if not k.startswith('SLURM_')}
    os.chdir(root);os.execvpe(argv[0],argv,env)


def run(root,nonce):
    p=approved(root);claim=read(root/'launch.json');job=os.environ.get('SLURM_JOB_ID','')
    if (not nonce or claim.get('nonce')!=nonce or claim.get('proposal_sha256')!=digest(p) or not job.isdigit() or
        not re.search(r'job[_/\-]'+re.escape(job)+r'(?:/|$)',Path('/proc/self/cgroup').read_text())):
        raise Closed('Owned Slurm allocation claim required')
    put(root/'run-claim.json',dict(job_id=job,nonce=nonce,epoch=time.time()))
    try:
        snapshot=json.loads(subprocess.check_output(['scontrol','--json','show','job',job],text=True,timeout=15));put(root/'scheduler.json',snapshot)
        if snapshot['jobs'][0].get('comment')!='webprm-acquire:'+nonce:raise Closed('Scheduler nonce differs')
        proof=p['transfer_exception'];clock=backend.base.context(snapshot,job,time.time(),maximum_seconds=proof['requested_allocation_seconds'])
        lib=library(root);items,images=backend.inputs(root);previous=checked(proof['predecessor_root'])
        prior,retained_reserved=storage_reservation(lib,previous,proof)
        if shutil.disk_usage(root).free<backend.LIMITS['initial_persistent_free_bytes'] or shutil.disk_usage('/tmp').free<2**30:
            raise Closed('Approved storage headroom unavailable')
        scratch=Path('/tmp')/('webprm-map-backend-'+job+'-'+secrets.token_hex(4));scratch.mkdir(mode=0o700)
        attempt=root/('attempt-'+job);attempt.mkdir(mode=0o700)
        put(attempt/'context.json',dict(clock,scratch=str(scratch),proposal_sha256=digest(p),prior_retained_bytes=prior,
            prior_retained_reserved_including_append=retained_reserved,prior_temporary_reserved_bytes=resume.SCRATCH_RESERVE))
        def worker(output,scratch,deadline):
            os.setsid();os.chdir(output);os.environ.update(lib.safe_environment(scratch))
            resource.setrlimit(resource.RLIMIT_AS,(6*2**30,)*2);resource.setrlimit(resource.RLIMIT_FSIZE,(backend.LIMITS['retained_bytes'],)*2)
            try:
                put(root/'network-claim.json',dict(job_id=job,epoch=time.time(),additional_transfer_campaigns=1,protocol=resume.PROTOCOL))
                cls=resume.ledger_class(lib,backend.LIMITS,max(x['bytes'] for x in items if x['kind']=='image_blob'))
                ledger=cls(output,**proof['prior_network_accounting']);tokens=oci.Registry(lib,ledger,deadline);receipts=[]
                for index,item in enumerate(items):
                    if item['kind']=='dataset':
                        value=resume.transfer_dataset(backend,output,item,lib,ledger,deadline,prefix=proof['prefix_proof'] if index==0 else None)
                    else:value=oci.transfer_blob(output,item,ledger,deadline,tokens)
                    put(output/f'item-{index:03d}-verified.json',value);receipts.append(value)
                backend.oci_metadata(output,images)
                put(output/'stage-result.json',dict(status='acquired_transport_and_dataset_content_identity_verified',items=receipts,
                    charged_bytes=ledger.data['charged_bytes']+proof['prior_network_accounting']['prior_charged_bytes'],
                    prior_charged_bytes=proof['prior_network_accounting']['prior_charged_bytes'],
                    cumulative_responses=len(ledger.data['reservations'])+proof['prior_network_accounting']['prior_responses'],
                    tls_handshake_reconnects=ledger.data['tls_handshake_reconnects'],raw_diff_ids_verified=False,extracted=False,
                    ready_to_host=False,model_calls=0,browser_calls=0))
            except BaseException as error:
                put(output/'failure.json',dict(error_type=type(error).__name__,error=str(error)[:2000],epoch=time.time()));raise SystemExit(1)
        closure=lib.own_worker(attempt,scratch,clock['work_deadline'],target=worker,prior_retained_bytes=retained_reserved)
        result=read(attempt/'stage-result.json')
        if (closure['remaining_owned_processes'] or lib.apparent_bytes(root)+lib.apparent_bytes(previous)>backend.LIMITS['retained_bytes'] or
            lib.apparent_bytes(scratch)+resume.SCRATCH_RESERVE>backend.LIMITS['temporary_bytes'] or time.time()>=clock['allocation_deadline']):
            raise Closed('Closure/storage/allocation boundary verification failed')
        put(root/'completed.json',dict(status='acquisition_complete',result=result,job_id=job,ready_to_host=False,epoch=time.time()))
    except BaseException as error:
        put(root/'failure.json',dict(error_type=type(error).__name__,error=str(error)[:2000],epoch=time.time()));raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);modes=parser.add_mutually_exclusive_group(required=True)
    for name in ('prepare','launch','run'):modes.add_argument('--'+name,action='store_true')
    parser.add_argument('--root',required=True,type=Path);parser.add_argument('--predecessor',type=Path)
    parser.add_argument('--evidence',type=Path);parser.add_argument('--nonce',default='');args=parser.parse_args();root=checked(args.root)
    if args.prepare:
        if args.predecessor is None or args.evidence is None:parser.error('Preparation requires predecessor and evidence')
        print(json.dumps(dict(root=str(root),proposal_sha256=digest(prepare(root,args.predecessor,args.evidence)))))
    elif args.launch:launch(root)
    else:run(root,args.nonce)
if __name__=='__main__':main()
