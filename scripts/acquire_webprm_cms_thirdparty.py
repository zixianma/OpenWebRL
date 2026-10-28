#!/usr/bin/env python3
"""Acquire a pinned third-party CMS reconstruction; original-state identity is unverified.

Preparation is offline. A new exact source/compute/transfer approval is mandatory.
No extraction, image execution, hosting, browser, model or provider call occurs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import secrets
import shutil
import sys
import time

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from scripts import acquire_webprm_cms_original as cms
from scripts import resume_webprm_cms_acquisition as owner
from scripts import resume_webprm_cms_confirmation as previous
from scripts import webprm_cms_confirmation_transport as transport
from scripts import acquire_webprm_map_backend as registry
from scripts.webprm_cms_oci_transport import Registry, transfer_blob, verify_raw_layers

OLD=cms.RUNTIME/'next-stage-preparation/cms-confirmation-successor-v1'
OLD_SHA='71d3b960147154e7f75976383e40557af53a153b2e0c0df4a3f38d4b1974b39c'
METADATA=cms.RUNTIME/'next-stage-preparation/cms-original-mirror-metadata-v1'
ASSESSMENT_SHA='351cebd639ffe9d81a01b708408fcdcd13b08d557dd07cc3b4c7570d2e0a0d9f'
MANIFEST_SHA='428cdedfc10996d7f5378299cff541c93c187d1243643e31d439a66c0156147d'
CONFIG_SHA='d0709482086f0e4931771795ae2b1e4b13fb9237c22328c8df0d074763465ad4'
REPOSITORY='webarenaimages/shopping_admin_final_0719'
PROFILE='cms_disclosed_third_party_pinned_oci_v1'
PRIOR_CHARGED=1326678
PRIOR_METADATA_KNOWN=81494
PRIOR_METADATA_UNKNOWN_RESERVED=65536
PRIOR_RESPONSES=18
PRIOR_TEMPORARY_RESERVED=4*2**20
RESOURCES=dict(owner.RESOURCES,prior_elapsed_seconds=22,remaining_total_seconds=3578,
    maximum_additional_startup_allocations=2,maximum_transfer_attempts=1)
LIMITS=dict(cms.LIMITS,temporary_bytes=512*2**20-PRIOR_TEMPORARY_RESERVED,
    expanded_raw_layer_bytes=12*2**30)
SOURCES=previous.SOURCES+('scripts/acquire_webprm_cms_thirdparty.py','scripts/webprm_cms_oci_transport.py',
    'tests/test_acquire_webprm_cms_thirdparty.py',
    'scripts/acquire_webprm_map_backend.py','scripts/acquire_webprm_original_sites.py',
    'scripts/webprm_acquisition_recovery.py','openwebrl/__init__.py')
_budget=owner.budget


def metadata_source():
    path=METADATA/'identity-assessment.json'
    if cms.sha(path)!=ASSESSMENT_SHA:raise cms.GateClosed('Third-party source assessment changed')
    assessment=cms.read(path);ledger=cms.read(METADATA/'ledger.json')
    if (cms.sha(METADATA/'ledger.json')!=assessment['metadata_accounting']['ledger_sha256'] or
            ledger['charged_bytes']!=992004 or ledger['known_body_bytes']!=74500 or len(ledger['requests'])!=13):
        raise cms.GateClosed('Metadata request/accounting evidence changed')
    values={}
    for name,key,expected in (('cms-original-registry-manifest','manifest',MANIFEST_SHA),
                               ('cms-original-registry-config-content','config',CONFIG_SHA)):
        row=next(x for x in ledger['requests'] if x['name']==name)
        path=cms.checked_path(row['body_path'])
        if path.parent!=METADATA or cms.sha(path)!=expected:raise cms.GateClosed('Pinned upstream metadata changed')
        values[key]=path
    return assessment,values


def inputs(root):
    manifest_path=root/'metadata/manifest.json';config_path=root/'metadata/config.json'
    if cms.sha(manifest_path)!=MANIFEST_SHA or cms.sha(config_path)!=CONFIG_SHA:
        raise cms.GateClosed('Pinned manifest/config content changed')
    manifest,config=cms.read(manifest_path),cms.read(config_path)
    if (manifest.get('schemaVersion')!=2 or manifest.get('mediaType')!='application/vnd.docker.distribution.manifest.v2+json' or
            manifest.get('config')!={'mediaType':'application/vnd.docker.container.image.v1+json','size':config_path.stat().st_size,'digest':'sha256:'+CONFIG_SHA} or
            config.get('os')!='linux' or config.get('architecture')!='amd64' or config.get('rootfs',{}).get('type')!='layers' or
            len(manifest.get('layers',[]))!=31 or len(config['rootfs'].get('diff_ids',[]))!=31):
        raise cms.GateClosed('Pinned platform/config/layer topology differs')
    rows=[];unique={}
    for i,(layer,diff) in enumerate(zip(manifest['layers'],config['rootfs']['diff_ids'])):
        if (set(layer)!={'mediaType','size','digest'} or layer['mediaType']!='application/vnd.docker.image.rootfs.diff.tar.gzip' or
                type(layer['size']) is not int or layer['size']<=0 or
                not re.fullmatch('sha256:[0-9a-f]{64}',layer['digest']) or not re.fullmatch('sha256:[0-9a-f]{64}',diff)):
            raise cms.GateClosed('Invalid layer descriptor or raw diff ID')
        item=dict(index=i,kind='image_blob',filename='oci/blobs/sha256/'+layer['digest'][7:],bytes=layer['size'],
            sha256=layer['digest'][7:],raw_diff_id=diff,registry='https://registry-1.docker.io',repository=REPOSITORY,
            url='https://registry-1.docker.io/v2/'+REPOSITORY+'/blobs/'+layer['digest'])
        if layer['digest'] in unique and any(unique[layer['digest']][k]!=item[k] for k in ('bytes','raw_diff_id')):
            raise cms.GateClosed('Shared layer descriptor conflicts')
        unique.setdefault(layer['digest'],item);rows.append(item)
    if sum(x['bytes'] for x in rows)!=2156742179:raise cms.GateClosed('Pinned aggregate layer size differs')
    return rows,list(unique.values()),manifest,config


def prior_evidence():
    rows=previous.prior_evidence();p=cms.read(OLD/'proposal.json');a=cms.read(OLD/'approval.json')
    terminal=cms.read(OLD/'startup-01/terminal.json');attempt=OLD/'startup-01/attempt-355928'
    ledger=cms.read(attempt/'transfer-ledger.json');closure=cms.read(attempt/'closure.json')
    if (cms.digest(p)!=OLD_SHA or a.get('status')!='approved' or a.get('proposal_sha256')!=OLD_SHA or
            owner.terminal_accounting(terminal['scheduler_snapshot'],'355928',3540)!=terminal or
            terminal['elapsed_seconds']!=7 or terminal['state']!='FAILED' or
            closure.get('remaining_owned_processes')!={} or closure.get('child_exitcode')!=1 or
            ledger.get('charged_bytes')!=201125 or ledger.get('prior_charged_bytes')!=133549 or
            ledger.get('known_received_bytes')!=4517 or ledger.get('metadata_received_bytes')!=4517 or
            len(ledger.get('reservations',[]))!=3 or any(x['state']!='complete' for x in ledger['reservations']) or
            (attempt/'archive-attempt-001.partial').stat().st_size!=0 or (OLD/'completed.json').exists() or
            cms.read(OLD/'transfer-claim.json').get('job_id')!='355928'):
        raise cms.GateClosed('Closed upstream-quota predecessor differs')
    prefix='/gpfs/projects/krishna/zixianma/OpenWebRL/'
    for name,expected in p['source_sha256'].items():
        if not name.startswith(prefix) or cms.sha(OLD/'frozen-source'/name[len(prefix):])!=expected:
            raise cms.GateClosed('Historical frozen source changed')
    names=('proposal.json','approval.json','startup-01/terminal.json','transfer-claim.json',
        'startup-01/attempt-355928/closure.json','startup-01/attempt-355928/transfer-ledger.json',
        'startup-01/attempt-355928/response-0002-metadata.bin','closed-attempt-independent-audit-v1.json')
    rows.append(dict(terminal,root=str(OLD),charged_download_bytes=201125,
        files_sha256={name:cms.sha(OLD/name) for name in names}))
    assessment,_=metadata_source()
    rows.append(dict(root=str(METADATA),elapsed_seconds=0,kind='bounded_read_only_metadata_no_allocation',
        charged_download_bytes=992004,files_sha256={'identity-assessment.json':ASSESSMENT_SHA,
            'ledger.json':assessment['metadata_accounting']['ledger_sha256']}))
    if sum(x['elapsed_seconds'] for x in rows)!=22:raise cms.GateClosed('Cumulative allocation accounting changed')
    return rows


def budget(root):
    result=_budget(root);result['prior_seconds']=22;return result


def approved(root):
    p,a=cms.read(root/'proposal.json'),cms.read(root/'approval.json')
    if (p.get('resource_profile')!=PROFILE or p.get('campaign_root')!=str(root) or
            p.get('resources')!=RESOURCES or p.get('limits')!=LIMITS or p.get('predecessors')!=prior_evidence() or
            p.get('identity_status')!='third_party_reconstruction_original_state_unverified' or
            a.get('status')!='approved' or a.get('proposal_sha256')!=cms.digest(p) or
            a.get('third_party_unverified_state_source_accepted') is not True or
            a.get('one_additional_network_campaign_approved') is not True or
            a.get('two_startup_allocations_within_remaining_time_approved') is not True or
            not isinstance(a.get('exact_user_approval'),str) or not a['exact_user_approval'].strip()):
        raise cms.GateClosed('Exact disclosed third-party source and bounded resource/transfer approval required')
    current=owner.source_hashes()
    mutable={str(REPO/'scripts/webprm_cms_oci_transport.py'),str(REPO/'tests/test_acquire_webprm_cms_thirdparty.py')}
    if set(current)!=set(p['source_sha256']) or any(current[k]!=v for k,v in p['source_sha256'].items() if k not in mutable):
        raise cms.GateClosed('Frozen acquisition ownership/accounting/metadata implementation changed')
    for name,expected in p['metadata_sha256'].items():
        if cms.sha(root/name)!=expected:raise cms.GateClosed('Frozen upstream metadata changed')
    if p['layers']!=inputs(root)[0]:raise cms.GateClosed('Pinned layer descriptors differ')
    return p


def install():
    owner.PROFILE=PROFILE;owner.RESOURCES=RESOURCES;owner.MAX_STARTUPS=2;owner.SOURCES=SOURCES
    owner.SCRIPT=Path(__file__).resolve();owner.prior_evidence=prior_evidence;owner.budget=budget;owner.approved=approved
    cms.LIMITS.clear();cms.LIMITS.update(LIMITS)


def write_oci(root,package,manifest):
    blobs=root/'oci/blobs/sha256'
    for name,expected in (('manifest.json',MANIFEST_SHA),('config.json',CONFIG_SHA)):
        target=blobs/expected;raw=(package/'metadata'/name).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=expected:raise cms.GateClosed('Pinned OCI metadata changed')
        with target.open('xb') as output:output.write(raw);output.flush();os.fsync(output.fileno())
    cms.put(root/'oci/oci-layout',dict(imageLayoutVersion='1.0.0'))
    cms.put(root/'oci/index.json',dict(schemaVersion=2,manifests=[dict(mediaType=manifest['mediaType'],
        digest='sha256:'+MANIFEST_SHA,size=(package/'metadata/manifest.json').stat().st_size,
        annotations={'org.opencontainers.image.ref.name':REPOSITORY+':pinned-third-party'})]))


def worker(attempt,scratch,deadline):
    os.setsid();os.chdir(attempt);os.environ.update(cms.safe_environment(scratch));os.umask(0o077)
    resource.setrlimit(resource.RLIMIT_AS,(6*2**30,6*2**30))
    resource.setrlimit(resource.RLIMIT_FSIZE,(LIMITS['retained_bytes'],LIMITS['retained_bytes']))
    try:
        package=attempt.parent.parent;rows,items,manifest,_=inputs(package)
        ledger=transport.CumulativeLedger(attempt,prior_charged_bytes=PRIOR_CHARGED,
            prior_metadata_bytes=PRIOR_METADATA_KNOWN+PRIOR_METADATA_UNKNOWN_RESERVED,prior_responses=PRIOR_RESPONSES)
        tokens=Registry(cms,ledger,deadline);transfers=[]
        for item in items:
            transfers.append(transfer_blob(attempt,item,ledger,deadline,tokens))
            cms.put(attempt/'download-progress.json',dict(completed_blobs=len(transfers),unique_blobs=len(items),
                completed_bytes=sum(x['bytes'] for x in transfers)),replace=(attempt/'download-progress.json').exists())
        proof=verify_raw_layers(attempt,items,deadline,maximum=LIMITS['expanded_raw_layer_bytes'])
        write_oci(attempt,package,manifest)
        cms.put(attempt/'identity-proof.json',dict(proof,manifest_sha256=MANIFEST_SHA,config_sha256=CONFIG_SHA,
            ordered_layer_count=len(rows),transfers=transfers,original_state_identity_verified=False,
            source='third_party_webarenaimages_not_uploaded_by_webarena_maintainers'))
        cms.put(attempt/'stage-result.json',dict(status='pinned_third_party_image_and_raw_diff_ids_verified',
            identity_proof_sha256=cms.sha(attempt/'identity-proof.json'),ready_to_host=False,
            original_state_identity_verified=False,model_provider_calls=0))
    except BaseException as error:
        cms.put(attempt/'failure.json',dict(status='failed_preserving_partial_evidence',error_type=type(error).__name__,
            reason=str(error)[:500] if isinstance(error,(cms.GateClosed,registry.Closed)) else 'Transport/IO/identity failure',automatic_retry=False))
        raise SystemExit(1)


def run(root,number,nonce):
    path=cms.checked_path(root/f'startup-{number:02d}');stage='startup_before_transfer'
    try:
        p=approved(root);claim=cms.read(path/'launch.json');job=os.environ.get('SLURM_JOB_ID','')
        if (not job.isdigit() or f'/job_{job}/' not in Path('/proc/self/cgroup').read_text() or claim.get('nonce')!=nonce or
                claim.get('proposal_sha256')!=cms.digest(p) or claim.get('source_sha256')!=owner.source_hashes()):
            raise cms.GateClosed('Owned exact approved acquisition allocation required')
        cms.put(path/'run-claim.json',dict(job_id=job,nonce=nonce,at_epoch=time.time()))
        if budget(root)['additional_startups']!=number or not 1<=number<=2:raise cms.GateClosed('Startup accounting differs')
        context=cms.saved_allocation_context(path,job,nonce,claim,
            dict(cms.RESOURCES,total_allocation_seconds_including_retries=claim['requested_seconds']))
        prior=sum(cms.apparent_bytes(Path(x['root'])) for x in p['predecessors'])
        required=sum(x['bytes'] for x in inputs(root)[1])+16*2**20
        if (shutil.disk_usage(root).free<required or shutil.disk_usage('/tmp').free<2**30 or
                prior+cms.apparent_bytes(root)+required>LIMITS['retained_bytes']):raise cms.GateClosed('Cumulative storage admission failed')
        cms.put(root/'transfer-claim.json',dict(startup=number,job_id=job,at_epoch=time.time(),resume_allowed=False))
        stage='sole_transfer_attempt'
        scratch=cms.checked_path(Path('/tmp')/f'webprm-cms-thirdparty-{job}-{secrets.token_hex(4)}');scratch.mkdir(mode=0o700)
        attempt=path/('attempt-'+job);attempt.mkdir(mode=0o700)
        cms.put(attempt/'context.json',dict(context,scratch=str(scratch),proposal_sha256=cms.digest(p)))
        outside=cms.apparent_bytes(root)+prior-cms.apparent_bytes(path)
        cms.own_worker(attempt,scratch,context['work_deadline'],target=worker,prior_retained_bytes=outside)
        result=cms.read(attempt/'stage-result.json')
        if (result.get('status')!='pinned_third_party_image_and_raw_diff_ids_verified' or
                result['identity_proof_sha256']!=cms.sha(attempt/'identity-proof.json') or
                time.time()>=context['allocation_deadline'] or prior+cms.apparent_bytes(root)>LIMITS['retained_bytes'] or
                cms.apparent_bytes(scratch)>LIMITS['temporary_bytes']):
            raise cms.GateClosed('Final third-party identity/time/storage gate failed')
        cms.put(root/'completed.json',dict(status=result['status'],job_id=job,startup=number,
            result_path=str(attempt/'stage-result.json'),original_state_identity_verified=False,ready_to_host=False))
    except Exception as error:
        cms.put(path/'failure.json',dict(status='failed',stage=stage,error_type=type(error).__name__,error=str(error)[:500],automatic_retry=False))
        raise


def prepare(root):
    root=cms.checked_path(root);assessment,metadata=metadata_source();rows=prior_evidence()
    root.mkdir(mode=0o700,parents=True,exist_ok=False);(root/'metadata').mkdir(mode=0o700)
    for key,path in metadata.items():
        with (root/'metadata'/(key+'.json')).open('xb') as out:out.write(path.read_bytes())
    cms.put(root/'metadata/source-assessment.json',assessment)
    layers,items,_,_=inputs(root)
    for name in SOURCES:
        target=root/'frozen-source'/name;target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as out:out.write((REPO/name).read_bytes())
    p=dict(schema_version=1,status='pending_user_approval',resource_profile=PROFILE,campaign_root=str(root),
        resources=RESOURCES,limits=LIMITS,predecessors=rows,source_sha256=owner.source_hashes(),
        metadata_sha256={str(x.relative_to(root)):cms.sha(x) for x in (root/'metadata').iterdir()},layers=layers,
        identity_status='third_party_reconstruction_original_state_unverified',
        original_state_identity_verified=False,maintainer_uploaded=False,
        primary_source_disclaimer='https://github.com/web-arena-x/webarena/issues/231#issuecomment-3583249074',
        pinned_manifest_sha256=MANIFEST_SHA,pinned_config_sha256=CONFIG_SHA,
        unique_blob_count=len(items),unique_blob_bytes=sum(x['bytes'] for x in items),
        cumulative_network=dict(prior_charged_bytes=PRIOR_CHARGED,prior_metadata_known_bytes=PRIOR_METADATA_KNOWN,
            prior_metadata_unknown_reserved_bytes=PRIOR_METADATA_UNKNOWN_RESERVED,prior_responses=PRIOR_RESPONSES,
            total_charged_bytes_cap=10000000000,remaining_charged_bytes=10000000000-PRIOR_CHARGED),
        cumulative_storage=dict(retained_bytes_cap=12*2**30,temporary_bytes_cap=512*2**20,
            prior_temporary_bytes_reserved_not_measured=PRIOR_TEMPORARY_RESERVED,new_temporary_bytes_cap=LIMITS['temporary_bytes']),
        startup_request='Two manual startup allocations share 3,578 remaining seconds; one slot beyond the single remaining earlier slot. First 59 minutes. No additional allocation time.',
        startup_repair='A manual successor requires terminal zero-network failure, changed transport/tests only, and a hash-bound passing offline test review. Ownership/accounting source remains frozen.',
        transfer_request='One additional network campaign for the pinned unique blob set; no retry/resume, no Drive requests. Anonymous registry token and bounded official-registry CDN redirects included.',
        scope='Download exact pinned OCI blobs; verify compressed SHA256 and streaming raw diffIDs within 12 GiB. No extraction, execution, hosting, browser, model or API calls.',
        limitation='Artifact identity is pinned; byte equality and original benchmark state equality to the unavailable released CMS archive are not verified. Use is an explicit disclosed third-party-source reconstruction choice.')
    cms.put(root/'proposal.json',p);cms.put(root/'approval.json',dict(status='pending',proposal_sha256=cms.digest(p),
        exact_user_approval=None,third_party_unverified_state_source_accepted=False,
        one_additional_network_campaign_approved=False,two_startup_allocations_within_remaining_time_approved=False))
    return p


def main():
    parser=argparse.ArgumentParser(description=__doc__);m=parser.add_mutually_exclusive_group(required=True)
    for mode in ('prepare','launch','run','reconcile','plan'):m.add_argument('--'+mode,action='store_true')
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--startup',type=int)
    parser.add_argument('--nonce');parser.add_argument('--repair-review',type=Path)
    args=parser.parse_args();root=cms.checked_path(args.root);install()
    if args.prepare:print(json.dumps(dict(status='pending_user_approval',proposal_sha256=cms.digest(prepare(root)))))
    elif args.launch:raise SystemExit(owner.launch(root,args.repair_review))
    elif args.run:run(root,args.startup,args.nonce)
    elif args.reconcile:print(json.dumps(owner.reconcile(root,args.startup)))
    else:print(json.dumps(owner.next_request(root,args.repair_review)))


if __name__=='__main__':main()
