#!/usr/bin/env python3
"""Bounded Shopping/GitLab OCI acquisition with explicit unverified provenance.

No original-image equivalence, extraction, hosting, or benchmark claim. Each
site requires its own exact source/resource approval; preparation is offline.
"""
from __future__ import annotations
import argparse
import importlib.util
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
REPO=SCRIPT.parents[1]
DEPENDENCIES=('scripts/acquire_webprm_original_sites.py','scripts/acquire_webprm_cms_original.py',
 'scripts/inspect_webprm_cms_archive.py','scripts/webprm_acquisition_recovery.py',
 'scripts/webprm_cms_confirmation_transport.py','scripts/acquire_webprm_map_backend.py',
 'scripts/webprm_cms_oci_transport.py','scripts/webprm_transfer_recovery.py',
 'openwebrl/__init__.py','openwebrl/webprm_lifecycle.py','openwebrl/jev_eval.py')
SOURCE_ROOT=SCRIPT.parent/'dependencies' if SCRIPT.name=='acquisition.py' else REPO
sys.path.insert(0,str(SOURCE_ROOT))
from scripts import acquire_webprm_original_sites as base
Closed,checked,put,read,sha,digest=base.Closed,base.checked,base.put,base.read,base.sha,base.digest
RUNTIME=base.RUNTIME
METADATA=RUNTIME/'next-stage-preparation/next-acquisition-review-v1'
ASSESSMENT_SHA='79d62411159f934a03942b17e2c309f929e5a9bc73e5f23903b8a1154436d31d'
RESOURCES=dict(base.RESOURCES)
IDENTITY='third_party_reconstruction_original_state_unverified'
TRANSFER_EXCEPTION='explicit_second_transfer_within_original_caps_v1'
PRIOR_SCRATCH_RESERVE=4*2**20
PINNED={
 'shopping':dict(repository='webarenaimages/shopping_final_0712',
  manifest_sha256='8f9618d9194a46bc9f7bc96dc14deba0b968abce6c393d59cc7cd1dfec86ed6f',
  config_sha256='73e22b803149f8efa3f1171d2347fbdcc1063a2a20ddfbab4d5a20ea0fc7cc9c',layers=28,compressed_bytes=45436833004),
 'gitlab':dict(repository='webarenaimages/gitlab-populated-final',
  manifest_sha256='5e367035617c9419c2123a8319b77a26cdeb1525fbd82913ba8fcfc12d7e2d57',
  config_sha256='3e6c8590e9ac1696f933a3373a37e67ce0db9648419e711f7bde1f591aa673ed',layers=12,compressed_bytes=48384761020)}


def profile(site):
    return dict(PINNED[site],charged_download_bytes=base.PROFILES[site]['charged_download_bytes'],
                retained_bytes=base.PROFILES[site]['retained_bytes'],expanded_raw_bytes=base.PROFILES[site]['expanded_regular_bytes'])


def limits(site):
    return dict(base.limits(base.PROFILES[site]),expanded_raw_layer_bytes=profile(site)['expanded_raw_bytes'])


def inputs(root,site):
    p=profile(site);manifest_path=root/'metadata/manifest.json';config_path=root/'metadata/config.json'
    if sha(manifest_path)!=p['manifest_sha256'] or sha(config_path)!=p['config_sha256']:
        raise Closed('Candidate manifest/config digest changed')
    manifest,config=read(manifest_path),read(config_path)
    if (manifest.get('schemaVersion')!=2 or manifest.get('mediaType')!='application/vnd.docker.distribution.manifest.v2+json' or
        manifest.get('config',{}).get('digest')!='sha256:'+p['config_sha256'] or manifest['config'].get('size')!=config_path.stat().st_size or
        config.get('os')!='linux' or config.get('architecture')!='amd64' or config.get('rootfs',{}).get('type')!='layers' or
        len(manifest.get('layers',[]))!=p['layers'] or len(config['rootfs'].get('diff_ids',[]))!=p['layers']):
        raise Closed('Candidate platform/config/layer topology changed')
    unique={}
    for layer,diff in zip(manifest['layers'],config['rootfs']['diff_ids']):
        if (layer.get('mediaType')!='application/vnd.docker.image.rootfs.diff.tar.gzip' or type(layer.get('size')) is not int or layer['size']<=0 or
            not re.fullmatch('sha256:[0-9a-f]{64}',layer.get('digest','')) or not re.fullmatch('sha256:[0-9a-f]{64}',diff)):
            raise Closed('Invalid pinned layer descriptor')
        h=layer['digest'][7:]
        item=dict(kind='image_blob',filename='oci/blobs/sha256/'+h,bytes=layer['size'],sha256=h,raw_diff_id=diff,
                  registry='https://registry-1.docker.io',repository=p['repository'],
                  url='https://registry-1.docker.io/v2/'+p['repository']+'/blobs/'+layer['digest'])
        if h in unique and unique[h]!=item:raise Closed('Shared layer identity conflict')
        unique.setdefault(h,item)
    if sum(x['size'] for x in manifest['layers'])!=p['compressed_bytes']:raise Closed('Candidate compressed total changed')
    return list(unique.values()),manifest,config


def transfer_proof(root,previous,sources,repair_receipt):
    spec=importlib.util.spec_from_file_location('explicit_transfer_recovery',root/'dependencies/scripts/webprm_transfer_recovery.py')
    recovery=importlib.util.module_from_spec(spec);spec.loader.exec_module(recovery)
    evidence=read(root/'metadata/prior-failure-audit.json')
    proof=recovery.proof(previous,evidence['terminal']['scheduler_snapshot'],sources,repair_receipt,
                         prior_temporary_reserve_bytes=PRIOR_SCRATCH_RESERVE)
    if (evidence.get('site')!=read(previous/'proposal.json')['site'] or
            evidence.get('job_id')!=proof['predecessor_job_id'] or evidence.get('predecessor_root')!=str(previous) or
            evidence.get('status')!='failed_transfer_closed_independently_audited' or
            evidence.get('ledger_sha256')!=proof['prior_evidence_sha256']['attempt-'+proof['predecessor_job_id']+'/transfer-ledger.json']):
        raise Closed('Preserved failure audit identity differs from transfer proof')
    observation=read(root/'metadata/prior-scratch-observation.json')
    rows=[x for x in observation.get('rows',[]) if x.get('site')==evidence['site']]
    context=read(previous/('attempt-'+proof['predecessor_job_id'])/'context.json')
    if (len(rows)!=1 or rows[0].get('path')!=context.get('scratch') or rows[0].get('exists') is not True or
            type(rows[0].get('apparent_bytes')) is not int or not 0<=rows[0]['apparent_bytes']<PRIOR_SCRATCH_RESERVE or
            rows[0].get('reserved_bytes')!=PRIOR_SCRATCH_RESERVE or
            observation.get('node')!=evidence['terminal']['scheduler_snapshot']['jobs'][0]['nodes']):
        raise Closed('Prior-node scratch observation/reservation differs')
    return proof


def transfer_startup(proof):
    return {k:proof[k] for k in ('attempt_number','predecessor_root','predecessor_job_id',
                                 'prior_allocation_seconds','requested_allocation_seconds')}


def transfer_approval(proof):
    return dict(protocol=TRANSFER_EXCEPTION,approved=True,proof_sha256=digest(proof),
                additional_transfer_attempts=1,cumulative_transfer_attempts=2)


def prepare(root,site,*,predecessor=None,prior_job=None,repair_receipt=None,
            transfer_failure_audit=None,prior_scratch_receipt=None):
    root=checked(root)
    if transfer_failure_audit is not None and (predecessor is None or repair_receipt is None or prior_scratch_receipt is None):
        raise Closed('Extra transfer preparation needs predecessor, failure audit, repair and prior-node scratch evidence')
    if prior_scratch_receipt is not None and transfer_failure_audit is None:
        raise Closed('Scratch evidence alone cannot select an additional transfer')
    assessment=read(METADATA/'third-party-source-assessment.json')
    if sha(METADATA/'third-party-source-assessment.json')!=ASSESSMENT_SHA:raise Closed('Reviewed source assessment changed')
    pin=next(x for x in assessment['pins'] if x['site']==site)
    root.mkdir(mode=0o700,parents=True,exist_ok=False);sources={}
    for relative in DEPENDENCIES:
        target=root/'dependencies'/relative;target.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
        with target.open('xb') as out:out.write((REPO/relative).read_bytes())
        sources[str(target.relative_to(root))]=sha(target)
    with (root/'acquisition.py').open('xb') as out:out.write(SCRIPT.read_bytes())
    sources['acquisition.py']=sha(root/'acquisition.py');(root/'metadata').mkdir(mode=0o700)
    for name,source in (('manifest.json',Path(pin['manifest_path'])),('config.json',Path(pin['config_path'])),
                        ('source-assessment.json',METADATA/'third-party-source-assessment.json'),
                        ('drive-preflight.json',METADATA/'drive-preflight.json'),('oci-metadata-ledger.json',METADATA/'oci-metadata-ledger.json')):
        with (root/'metadata'/name).open('xb') as out:out.write(source.read_bytes())
        sources['metadata/'+name]=sha(root/'metadata'/name)
    if transfer_failure_audit is not None:
        for name,path in (('prior-failure-audit.json',transfer_failure_audit),('prior-scratch-observation.json',prior_scratch_receipt)):
            with (root/'metadata'/name).open('xb') as out:out.write(checked(path).read_bytes())
            sources['metadata/'+name]=sha(root/'metadata'/name)
    items,_,_=inputs(root,site)
    startup=dict(attempt_number=1,prior_allocation_seconds=0,requested_allocation_seconds=7200);inherited=None;extra=None
    if predecessor is not None:
        previous=checked(predecessor);pp,pa=read(previous/'proposal.json'),read(previous/'approval.json')
        if pp.get('transfer_recovery') is not None:
            raise Closed('The explicit second-transfer exception does not authorize another successor')
        if (pp.get('resources')!=RESOURCES or pp.get('profile')!=profile(site) or pp.get('limits')!=limits(site) or
            pp.get('identity_status')!=IDENTITY or pa.get('third_party_unverified_state_source_accepted') is not True):
            raise Closed('Successor cannot change disclosed source or resource scope')
        for name,h in pp['source_sha256'].items():
            if sha(previous/name)!=h:raise Closed('Preserved predecessor changed')
        if pp.get('items')!=items:raise Closed('Successor cannot change pinned transfer items')
        if transfer_failure_audit is not None:
            extra=transfer_proof(root,previous,sources,checked(repair_receipt));startup=transfer_startup(extra)
            if prior_job is not None and str(prior_job)!=extra['predecessor_job_id']:raise Closed('Terminal job identity changed')
        else:
            if not str(prior_job).isdigit() or repair_receipt is None:raise Closed('Terminal job and tested source fix required')
            repair=read(checked(repair_receipt))
            if repair.get('replacement_source_sha256')!=sources:raise Closed('Repair tests do not bind the replacement')
            snapshot=json.loads(subprocess.check_output(['scontrol','--json','show','job',str(prior_job)],text=True,timeout=15))
            spec=importlib.util.spec_from_file_location('recovery',root/'dependencies/scripts/webprm_acquisition_recovery.py')
            recovery=importlib.util.module_from_spec(spec);spec.loader.exec_module(recovery)
            startup=recovery.recovery_proof(previous,pp,pa,read(previous/'launch.json'),snapshot,repair)
            if startup['predecessor_job_id']!=str(prior_job):raise Closed('Terminal job identity changed')
            inherited=pa['exact_user_approval']
    p=dict(schema_version=1,status='pending_exact_source_and_resource_approval',root=str(root),site=site,
           identity_status=IDENTITY,resources=RESOURCES,profile=profile(site),limits=limits(site),
           source_sha256=sources,source_assessment_sha256=ASSESSMENT_SHA,items=items,
           prior_metadata_accounting=assessment['per_site'][site],
           total_unique_compressed_bytes=sum(x['bytes'] for x in items),
           startup_recovery=startup,
           author_disclaimer=assessment['author_disclaimer'],
           recovery_policy='At most three manually prepared startup allocations share7200seconds. A changed/tested successor deducts every prior terminal allocation and is forbidden after any network claim; never auto-resubmit.',
           scope='Digest-pinned third-party compressed layers and bounded streaming rawdiffID verification only.',
           original_equivalence_verified=False,extraction=False,hosting=False,browser_calls=0,model_calls=0,
           file_semantics='One transfer campaign, one connection, no network retry/resume/fallback. Exact ranges or full200 reserved before reading; metadata and unknown reservations remain cumulatively charged.',
           limitations=['Original released archive/config/diffIDs unavailable; names and candidate digests are not original-state proof',
                        'RawdiffID verification checks the candidate only and does not establish benchmark state equivalence',
                        'Two hours is a bounded attempt including download/decompression/closure, not a throughput guarantee'])
    if extra is not None:
        p.update(transfer_recovery=extra,transfer_exception_protocol=TRANSFER_EXCEPTION,
            recovery_policy='One explicitly approved additional transfer campaign after the preserved first failure. Both allocations share7200seconds and all original byte/storage/metadata/response/API caps. No automatic retry, partial resume or further successor.',
            file_semantics='Original partials and unknown reservations are preserved and charged. This package transfers into a new isolated directory from byte zero, once, only after exact additional-transfer exception approval.')
    if inherited or extra is not None:put(previous/'successor-claim.json',dict(root=str(root),proposal_sha256=digest(p)))
    put(root/'proposal.json',p)
    put(root/'approval.json',dict(status='approved' if inherited else 'pending',proposal_sha256=digest(p),exact_user_approval=inherited,
                                 third_party_unverified_state_source_accepted=bool(inherited)))
    return p


def approved(root):
    p,a=read(root/'proposal.json'),read(root/'approval.json');site=p.get('site')
    if (site not in PINNED or p.get('root')!=str(root) or p.get('profile')!=profile(site) or p.get('limits')!=limits(site) or
        p.get('resources')!=RESOURCES or p.get('identity_status')!=IDENTITY or p.get('original_equivalence_verified') is not False or
        a.get('status')!='approved' or a.get('proposal_sha256')!=digest(p) or a.get('third_party_unverified_state_source_accepted') is not True or
        not isinstance(a.get('exact_user_approval'),str) or not a['exact_user_approval'].strip()):
        raise Closed('Exact disclosed third-party source and resource approval required')
    expected={'acquisition.py',*('dependencies/'+x for x in DEPENDENCIES),*('metadata/'+x for x in
        ('manifest.json','config.json','source-assessment.json','drive-preflight.json','oci-metadata-ledger.json'))}
    if 'transfer_recovery' in p:expected.update('metadata/'+x for x in ('prior-failure-audit.json','prior-scratch-observation.json'))
    if SCRIPT!=root/'acquisition.py' or set(p.get('source_sha256',{}))!=expected:raise Closed('Use the complete frozen package')
    for name,h in p['source_sha256'].items():
        if sha(checked(root/name))!=h:raise Closed('Frozen source or metadata changed')
    assessment=read(root/'metadata/source-assessment.json')
    if (sha(root/'metadata/source-assessment.json')!=ASSESSMENT_SHA or p.get('prior_metadata_accounting')!=assessment['per_site'][site] or
        sha(root/'metadata/drive-preflight.json')!=assessment['accounting']['drive_receipt_sha256'] or
        sha(root/'metadata/oci-metadata-ledger.json')!=assessment['accounting']['oci_receipt_sha256'] or p.get('items')!=inputs(root,site)[0]):
        raise Closed('Metadata identity/accounting changed')
    if 'transfer_recovery' in p:
        proof=p['transfer_recovery'];previous=checked(proof['predecessor_root']);pp=read(previous/'proposal.json')
        if (p.get('transfer_exception_protocol')!=TRANSFER_EXCEPTION or p['startup_recovery']!=transfer_startup(proof) or
                a.get('additional_transfer_exception')!=transfer_approval(proof)):
            raise Closed('Exact additional-transfer exception approval required; previous source approval is insufficient')
        if any(pp.get(k)!=p.get(k) for k in ('site','resources','profile','limits','items','identity_status','prior_metadata_accounting')):
            raise Closed('Additional-transfer successor changed original scope')
        if read(previous/'successor-claim.json')!=dict(root=str(root),proposal_sha256=digest(p)):
            raise Closed('Exclusive additional-transfer successor ownership changed')
        if transfer_proof(root,previous,p['source_sha256'],Path(proof['repair_receipt']['path']))!=proof:
            raise Closed('Preserved transfer evidence or cumulative accounting changed')
    else:
        if p.get('transfer_exception_protocol') is not None or a.get('additional_transfer_exception') is not None:
            raise Closed('Additional-transfer approval cannot alter the startup-only default')
        base.validate_startup(root,p)
    return p


def network_accounting(p):
    return p['transfer_recovery']['prior_network_accounting'] if 'transfer_recovery' in p else p['prior_metadata_accounting']


def previous_usage(root,p,lib):
    retained,temporary=base.previous_usage(root,p,lib)
    if 'transfer_recovery' in p:
        temporary=max(temporary,p['transfer_recovery']['prior_temporary_reserve_bytes'])
    return retained,temporary


def libraries(root,site):
    lib,_=base.load_library(root,base.PROFILES[site])
    from scripts import webprm_cms_oci_transport as oci
    from scripts import acquire_webprm_map_backend as registry
    for module,name in ((oci,'webprm_cms_oci_transport'),(registry,'acquire_webprm_map_backend'),(oci.cms,'acquire_webprm_cms_original')):
        if Path(module.__file__).resolve()!=root/'dependencies/scripts'/(name+'.py'):raise Closed('OCI dependency escaped frozen package')
    lib.LIMITS=limits(site);oci.cms.LIMITS=lib.LIMITS;lib.confirmation_transport.LIMITS=lib.LIMITS
    return lib,oci


def launch(root):
    p=approved(root);lib,_=libraries(root,p['site']);nonce=secrets.token_hex(16)
    seconds=p['startup_recovery']['requested_allocation_seconds']
    argv=['salloc','--account=zixianma','--partition=gpu-h200','--qos=normal','--nodes=1','--ntasks=1','--cpus-per-task=2',
          '--mem=8G',f'--time={seconds//3600:02d}:{seconds//60%60:02d}:00','--comment=webprm-acquire:'+nonce,'--job-name=wa-'+p['site']+'-thirdparty',
          '--chdir='+str(root),'srun','--ntasks=1','--cpus-per-task=2',sys.executable,'-I','-B',str(SCRIPT),'--run','--root',str(root),'--nonce',nonce]
    put(root/'launch.json',dict(nonce=nonce,proposal_sha256=digest(p),requested_allocation_seconds=seconds,argv=argv,epoch=time.time()))
    env={k:v for k,v in lib.safe_environment(root/'launch-cache').items() if not k.startswith('SLURM_')}
    os.chdir(root);os.execvpe(argv[0],argv,env)


def run(root,nonce):
    p=approved(root);claim=read(root/'launch.json');job=os.environ.get('SLURM_JOB_ID','')
    if (not nonce or claim.get('nonce')!=nonce or claim.get('proposal_sha256')!=digest(p) or not job.isdigit() or
        not re.search(r'job[_/\-]'+re.escape(job)+r'(?:/|$)',Path('/proc/self/cgroup').read_text())):raise Closed('Owned Slurm claim required')
    put(root/'run-claim.json',dict(job_id=job,nonce=nonce,epoch=time.time()))
    try:
        raw=subprocess.check_output(['scontrol','--json','show','job',job],text=True,timeout=15)
        snapshot=json.loads(raw);put(root/'scheduler.json',snapshot);clock=base.context(snapshot,job,time.time(),maximum_seconds=p['startup_recovery']['requested_allocation_seconds'])
        lib,oci=libraries(root,p['site']);items,manifest,_=inputs(root,p['site']);caps=p['limits']
        prior_retained,prior_temporary=previous_usage(root,p,lib)
        if prior_retained>=caps['retained_bytes'] or prior_temporary>=caps['temporary_bytes']:raise Closed('Preserved startup storage cap exhausted')
        lib.LIMITS=dict(lib.LIMITS,temporary_bytes=caps['temporary_bytes']-prior_temporary)
        if shutil.disk_usage(root).free<caps['retained_bytes'] or shutil.disk_usage('/tmp').free<caps['initial_temporary_free_bytes']:
            raise Closed('Approved storage headroom unavailable')
        scratch=Path('/tmp')/('webprm-original-'+p['site']+'-thirdparty-'+job+'-'+secrets.token_hex(4));scratch.mkdir(mode=0o700)
        attempt=root/('attempt-'+job);attempt.mkdir(mode=0o700)
        put(attempt/'context.json',dict(clock,scratch=str(scratch),proposal_sha256=digest(p),identity_status=IDENTITY,
            prior_retained_bytes=prior_retained,prior_temporary_bytes=prior_temporary,
            prior_network_accounting=network_accounting(p)))
        def worker(output,scratch,deadline):
            os.setsid();os.chdir(output);os.environ.update(lib.safe_environment(scratch))
            resource.setrlimit(resource.RLIMIT_AS,(6*2**30,)*2);resource.setrlimit(resource.RLIMIT_FSIZE,(caps['retained_bytes'],)*2)
            try:
                put(root/'network-claim.json',dict(job_id=job,epoch=time.time(),maximum_transfer_attempts=1,
                    transfer_exception_protocol=p.get('transfer_exception_protocol'),
                    cumulative_transfer_attempt_number=2 if 'transfer_recovery' in p else 1))
                ledger=lib.TransferLedger(output,cap=caps['download_charged_bytes'],metadata_cap=caps['metadata_received_bytes'],**network_accounting(p))
                tokens=oci.Registry(lib,ledger,deadline);receipts=[]
                for i,item in enumerate(items):
                    proof=oci.transfer_blob(output,item,ledger,deadline,tokens);put(output/f'item-{i:03d}-verified.json',proof);receipts.append(proof)
                rawproof=oci.verify_raw_layers(output,items,deadline,maximum=caps['expanded_raw_layer_bytes']);put(output/'raw-layer-verification.json',rawproof)
                blobs=output/'oci/blobs/sha256'
                for name,key in (('manifest.json','manifest_sha256'),('config.json','config_sha256')):
                    target=blobs/p['profile'][key]
                    with target.open('xb') as out:out.write((root/'metadata'/name).read_bytes());out.flush();os.fsync(out.fileno())
                put(output/'oci/oci-layout',dict(imageLayoutVersion='1.0.0'))
                put(output/'oci/index.json',dict(schemaVersion=2,manifests=[dict(mediaType=manifest['mediaType'],
                    digest='sha256:'+p['profile']['manifest_sha256'],size=(root/'metadata/manifest.json').stat().st_size)]))
                put(output/'stage-result.json',dict(status='candidate_transport_and_raw_identity_verified',identity_status=IDENTITY,
                    original_equivalence_verified=False,items=receipts,raw_layer_proof=rawproof,
                    charged_bytes=ledger.data['charged_bytes']+ledger.data['prior_charged_bytes'],extracted=False,ready_to_host=False))
            except BaseException as error:
                lib.put(output/'failure.json',dict(error_type=type(error).__name__,error=str(error)[:2000],epoch=time.time()));raise SystemExit(1)
        closure=lib.own_worker(attempt,scratch,clock['work_deadline'],target=worker,prior_retained_bytes=prior_retained)
        result=read(attempt/'stage-result.json')
        if (closure['remaining_owned_processes'] or lib.apparent_bytes(root)+prior_retained>caps['retained_bytes'] or
            lib.apparent_bytes(scratch)+prior_temporary>caps['temporary_bytes'] or time.time()>=clock['allocation_deadline']):raise Closed('Closure/storage/deadline boundary failed')
        put(root/'completed.json',dict(status='candidate_acquisition_complete',job_id=job,result=result,identity_status=IDENTITY,epoch=time.time()))
    except BaseException as error:
        put(root/'failure.json',dict(error_type=type(error).__name__,error=str(error)[:2000],epoch=time.time()));raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group(required=True)
    for name in ('prepare','launch','run'):mode.add_argument('--'+name,action='store_true')
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--site',choices=PINNED);parser.add_argument('--nonce',default='')
    parser.add_argument('--predecessor',type=Path);parser.add_argument('--prior-job');parser.add_argument('--repair-receipt',type=Path)
    parser.add_argument('--transfer-failure-audit',type=Path);parser.add_argument('--prior-scratch-receipt',type=Path)
    args=parser.parse_args();root=checked(args.root)
    if args.prepare:
        if not args.site:parser.error('--prepare requires --site')
        print(json.dumps(dict(root=str(root),proposal_sha256=digest(prepare(root,args.site,predecessor=args.predecessor,
            prior_job=args.prior_job,repair_receipt=args.repair_receipt,transfer_failure_audit=args.transfer_failure_audit,
            prior_scratch_receipt=args.prior_scratch_receipt)))))
    elif args.launch:launch(root)
    else:run(root,args.nonce)


if __name__=='__main__':main()
