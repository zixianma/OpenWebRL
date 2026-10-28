#!/usr/bin/env python3
"""Bounded acquisition of official Map datasets and pinned backend OCI blobs.

Download and identity only. No image execution, extraction, service, browser,
provider call or automatic retry. A manual startup successor requires a closed
zero-network predecessor and a tested source fix; all starts share four hours.
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
import urllib.error
import urllib.parse
import urllib.request

SCRIPT=Path(__file__).resolve()
REPO=SCRIPT.parents[1]
RUNTIME=Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011')
N=RUNTIME/'next-stage-preparation'
DEPS=('scripts/acquire_webprm_original_sites.py','scripts/acquire_webprm_cms_original.py',
      'scripts/inspect_webprm_cms_archive.py','scripts/webprm_acquisition_recovery.py',
      'openwebrl/__init__.py','openwebrl/webprm_lifecycle.py','openwebrl/jev_eval.py')
RESOURCES=dict(account='zixianma',partition='gpu-h200',qos='normal',nodes=1,cpus=2,memory_gib=8,gpus=0,
    total_allocation_seconds_including_retries=14400,maximum_allocation_attempts=3,
    maximum_transfer_attempts=1,model_provider_calls=0,browser_calls=0,api_usd=0)
LIMITS=dict(download_charged_bytes=195000000000,metadata_received_bytes=16*2**20,
    retained_bytes=200*2**30,temporary_bytes=512*2**20,initial_persistent_free_bytes=200*2**30,
    initial_temporary_free_bytes=2**30,closure_reserve_seconds=120,max_redirects_per_range=2,
    max_http_responses=12000,receipt_reserve_bytes=64*2**20,network_connections=1,response_header_bytes_per_request=65536)
PINSET_SHA='f6287fcbf2f0a44a8f0e7b8ee144ad2b622a94dee2dee0d47cb95bee23920677'
DATA_SHA='6328be6e312d9fb043e80a7bee99f57883d999cd89eb1a4af1634f9825e984c5'
PROFILE=dict(dataset_bytes=189212856320,image_compressed_layer_bytes=1042832734,
    source_recipe_sha256='77bd92a00288dbfd2f2f3a18b8d597578fff4f764498cf1baefba0acc8127f1a',
    pins_sha256=PINSET_SHA,dataset_metadata_sha256=DATA_SHA)
CHUNK=64*2**20
# Registries control only their own established delivery domains. Signed URL
# queries stay in memory; authorization is sent only to the registry origin.
CDN_HOSTS=frozenset(('production.cloudflare.docker.com','production.cloudfront.docker.com',
    'pkg-containers.githubusercontent.com',
    'docker-images-prod.6aa30f8b08e16409b46e0173d6de2f56.r2.cloudflarestorage.com'))


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def original(root=None):
    path=(root/'dependencies' if root else REPO)/'scripts/acquire_webprm_original_sites.py'
    m=module('map_original_helpers',path);m.RESOURCES=RESOURCES
    return m


base=original(SCRIPT.parent if SCRIPT.name=='acquisition.py' else None)
Closed,checked,put,read,sha,digest=base.Closed,base.checked,base.put,base.read,base.sha,base.digest


def metadata_files():
    if sha(N/'map-backend-image-pins-20261011.json')!=PINSET_SHA or sha(RUNTIME/'map-data-head-metadata.json')!=DATA_SHA:raise Closed('Reviewed backend metadata changed')
    pins=json.loads((N/'map-backend-image-pins-20261011.json').read_text())
    paths=[RUNTIME/'map-data-head-metadata.json',N/'map-backend-image-pins-20261011.json']
    for pin in pins:
        repository=pin['image'].split(':')[0];prefix='map-backend-'+repository.replace('/','__')
        for i,key in enumerate(('tag_manifest_digest','amd64_manifest_digest','config_digest')):
            path=N/(prefix+'-'+str(i)+'.json')
            if 'sha256:'+sha(path)!=pin[key]:raise Closed('Pinned image metadata digest changed')
            paths.append(path)
    return paths


def inputs(root):
    metadata=root/'metadata'
    if sha(metadata/'map-data-head-metadata.json')!=DATA_SHA or sha(metadata/'map-backend-image-pins-20261011.json')!=PINSET_SHA:raise Closed('Frozen backend metadata changed')
    datasets=[]
    for row in read(metadata/'map-data-head-metadata.json')['objects']:
        name=row['filename'];url='https://webarena-map-server-data.s3.amazonaws.com/'+name
        if row['url']!=url or not re.fullmatch('[a-z_]+[.]tar',name):raise Closed('Unexpected original dataset origin/name')
        datasets.append(dict(kind='dataset',filename='datasets/'+name,url=url,bytes=int(row['headers']['Content-Length']),etag=row['headers']['ETag']))
    images=[];blobs={}
    for pin in read(metadata/'map-backend-image-pins-20261011.json'):
        repo=pin['image'].split(':')[0];prefix='map-backend-'+repo.replace('/','__')
        values=[]
        for i,key in enumerate(('tag_manifest_digest','amd64_manifest_digest','config_digest')):
            path=metadata/(prefix+'-'+str(i)+'.json')
            if 'sha256:'+sha(path)!=pin[key]:raise Closed('Frozen OCI metadata identity changed')
            values.append(read(path))
        index,manifest,config=values
        if (not any(x['digest']==pin['amd64_manifest_digest'] and x.get('platform',{}).get('architecture')=='amd64' for x in index['manifests']) or
            manifest['config']['digest']!=pin['config_digest'] or config['os']!='linux' or config['architecture']!='amd64' or
            len(config['rootfs']['diff_ids'])!=len(manifest['layers'])):raise Closed('Image platform/config/layer topology changed')
        if sum(x['size'] for x in manifest['layers'])!=pin['compressed_layer_bytes']:raise Closed('Pinned layer size total changed')
        for layer in manifest['layers']:
            identity=layer['digest']
            if not re.fullmatch('sha256:[0-9a-f]{64}',identity) or type(layer['size']) is not int or layer['size']<=0:raise Closed('Invalid layer digest/size')
            item=dict(kind='image_blob',filename='oci/blobs/sha256/'+identity[7:],url=pin['registry']+'/v2/'+repo+'/blobs/'+identity,
                bytes=layer['size'],sha256=identity[7:],registry=pin['registry'],repository=repo)
            if identity in blobs and blobs[identity]['bytes']!=item['bytes']:raise Closed('Shared blob identity/size conflict')
            blobs.setdefault(identity,item)
        images.append(dict(pin,repository=repo,metadata_prefix=prefix,manifest=manifest,config=config))
    if sum(x['bytes'] for x in datasets)!=PROFILE['dataset_bytes']:raise Closed('Original dataset size total changed')
    if sum(x['compressed_layer_bytes'] for x in images)!=PROFILE['image_compressed_layer_bytes']:raise Closed('Original image size total changed')
    return datasets+list(blobs.values()),images


def prepare(root,*,predecessor=None,prior_job=None,repair_receipt=None):
    root=checked(root);paths=metadata_files()
    if sha(RUNTIME/'recipe-sources/webarena-map-backend.yaml')!=PROFILE['source_recipe_sha256']:raise Closed('Original backend recipe changed')
    root.mkdir(mode=0o700,parents=True,exist_ok=False);sources={}
    for name in DEPS:
        target=checked(root/'dependencies'/name);target.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
        with target.open('xb') as out:out.write((REPO/name).read_bytes())
        sources[str(target.relative_to(root))]=sha(target)
    with (root/'acquisition.py').open('xb') as out:out.write(SCRIPT.read_bytes())
    sources['acquisition.py']=sha(root/'acquisition.py')
    (root/'metadata').mkdir(mode=0o700)
    for source in paths:
        target=root/'metadata'/source.name
        with target.open('xb') as out:out.write(source.read_bytes())
        sources[str(target.relative_to(root))]=sha(target)
    items,images=inputs(root)
    startup=dict(attempt_number=1,prior_allocation_seconds=0,requested_allocation_seconds=14400);inherited=None
    if predecessor:
        previous=checked(predecessor);pp,pa=read(previous/'proposal.json'),read(previous/'approval.json')
        if pp.get('resources')!=RESOURCES or pp.get('profile')!=PROFILE or pp.get('limits')!=LIMITS:raise Closed('Successor changes approved scope')
        for name,wanted in pp['source_sha256'].items():
            if sha(previous/name)!=wanted:raise Closed('Predecessor sources changed')
        if not str(prior_job).isdigit() or repair_receipt is None:raise Closed('Closed predecessor and tested fix required')
        repair=read(checked(repair_receipt))
        if repair.get('replacement_source_sha256')!=sources:raise Closed('Source-fix tests do not bind replacement')
        snapshot=json.loads(subprocess.check_output(['scontrol','--json','show','job',str(prior_job)],text=True,timeout=15))
        recovery=module('map_recovery',root/'dependencies/scripts/webprm_acquisition_recovery.py')
        startup=recovery.recovery_proof(previous,pp,pa,read(previous/'launch.json'),snapshot,repair)
        if startup['predecessor_job_id']!=str(prior_job):raise Closed('Predecessor job identity changed')
        inherited=pa['exact_user_approval']
    proposal=dict(schema_version=1,status='pending_user_approval',root=str(root),site='map_backend',resources=RESOURCES,
        profile=PROFILE,limits=LIMITS,source_sha256=sources,startup_recovery=startup,items=items,
        total_unique_body_bytes=sum(x['bytes'] for x in items),image_references=[{k:v for k,v in x.items() if k not in ('manifest','config')} for x in images],
        scope='Four official original datasets plus three pinned backend image blob sets. Download/checksum only; no extraction, execution, hosting or browser/model/provider calls.',
        accounting='195GB cumulative HTTP body+bounded 64KiB headers, including anonymous auth and redirects. Unknown reservations remain fully charged; TLS/TCP framing excluded. 200GiB retained and512MiBtemporary across starts.',
        recovery_policy='At most three manual startup attempts share14400seconds; terminal elapsed including grace is deducted, next duration rounded down. Tested source fixes only. Any network claim prohibits successors; no range retries or resume.',
        identity='S3multipartETags pin entity stability, not content SHA256. Compute dataset SHA256 while streaming. Verify image compressed-layerSHA256 against pinned OCI manifests; rawdiffIDs/tar contents require later inspection.',
        limitations='Observed2026-10-11image tag identities, not verified historical binary equivalence. Four hours is a bounded attempt, not a throughput/completion guarantee.')
    if inherited:put(previous/'successor-claim.json',dict(root=str(root),proposal_sha256=digest(proposal)))
    put(root/'proposal.json',proposal);put(root/'approval.json',dict(status='approved' if inherited else 'pending',
        proposal_sha256=digest(proposal),exact_user_approval=inherited,inherited_startup_only=bool(inherited)))
    return proposal


def approved(root):
    p,a=read(root/'proposal.json'),read(root/'approval.json')
    if (p.get('root')!=str(root) or p.get('resources')!=RESOURCES or p.get('limits')!=LIMITS or p.get('profile')!=PROFILE or
        a.get('status')!='approved' or a.get('proposal_sha256')!=digest(p) or not a.get('exact_user_approval') or
        SCRIPT!=root/'acquisition.py'):raise Closed('Exact frozen Map backend acquisition approval required')
    expected={'acquisition.py',*('dependencies/'+x for x in DEPS),*('metadata/'+p.name for p in metadata_files())}
    if set(p.get('source_sha256',{}))!=expected:raise Closed('Incomplete frozen dependency/metadata identity')
    for name,wanted in p['source_sha256'].items():
        if sha(checked(root/name))!=wanted:raise Closed('Frozen Map dependency changed')
    if p.get('items')!=inputs(root)[0]:raise Closed('Dataset/blob identities changed')
    base.validate_startup(root,p)
    return p


def library(root):
    lib,_=base.load_library(root,dict(filename='unused-map.tar',archive_bytes=1,drive_id='unused',image_tag='unused',
        charged_download_bytes=LIMITS['download_charged_bytes'],retained_bytes=LIMITS['retained_bytes'],expanded_regular_bytes=1))
    lib.LIMITS=LIMITS
    return lib


def request_once(lib,ledger,request,maximum,deadline,*,open_response=None):
    while True:
        if time.time()>=deadline:raise Closed('Transfer deadline exhausted')
        reservation=ledger.reserve(maximum,request.full_url,None,None)
        try:response=(open_response or lib.opener())(request,timeout=min(30,max(.001,deadline-time.time())))
        except urllib.error.HTTPError as error:response=error
        except urllib.error.URLError as error:
            # Only an explicitly approved successor ledger can enable this
            # capability. Each connection already owns a durable reservation;
            # failed attempts remain unknown and charged before reconnecting.
            retry=getattr(ledger,'retry_tls_handshake',None)
            if retry is not None and retry(reservation,error):continue
            raise
        return reservation,response


def consume_metadata(ledger,reservation,response,cap):
    body=bytearray();complete=False
    try:
        while len(body)<cap:
            chunk=response.read(min(65536,cap-len(body)))
            if not chunk:complete=True;break
            body.extend(chunk)
        if len(body)==cap:raise Closed('Metadata response cap reached')
        return bytes(body)
    finally:response.close();ledger.settle(reservation,len(body),complete=complete,metadata=True)


class Registry:
    def __init__(self,lib,ledger,deadline,open_response=None):
        self.lib,self.ledger,self.deadline,self.open_response=lib,ledger,deadline,open_response;self.tokens={}

    def token(self,item):
        key=(item['registry'],item['repository']);current=self.tokens.get(key)
        if current and time.time()<current[1]:return current[0]
        host=urllib.parse.urlsplit(item['registry']).hostname
        if host=='registry-1.docker.io':origin,service='https://auth.docker.io','registry.docker.io'
        elif host=='ghcr.io':origin,service='https://ghcr.io','ghcr.io'
        else:raise Closed('Unreviewed registry origin')
        url=origin+'/token?'+urllib.parse.urlencode(dict(service=service,scope='repository:'+item['repository']+':pull'))
        i,response=request_once(self.lib,self.ledger,urllib.request.Request(url,headers={'Accept-Encoding':'identity'}),65536,self.deadline,open_response=self.open_response)
        status=response.status;raw=consume_metadata(self.ledger,i,response,65536)
        if status!=200:raise Closed('Anonymous registry authentication failed')
        data=json.loads(raw);token=data.get('token',data.get('access_token'))
        if not isinstance(token,str) or not token or len(token)>16384:raise Closed('Invalid bounded registry token')
        self.tokens[key]=(token,time.time()+min(180,max(1,int(data.get('expires_in',300))-60)))
        return token


def allowed_delivery(url,item):
    p=urllib.parse.urlsplit(url);origin=urllib.parse.urlsplit(item['url']).hostname
    if (p.scheme!='https' or p.username or p.password or p.port not in (None,443) or p.fragment or len(url)>16384 or
            (p.hostname!=origin and (item['kind']!='image_blob' or p.hostname not in CDN_HOSTS))):raise Closed('Unreviewed TLS delivery host')
    return url


def transfer(root,item,lib,ledger,deadline,registry,*,open_response=None,chunk=CHUNK):
    target=checked(root/item['filename']);target.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
    h=hashlib.sha256();written=0
    fd=os.open(target,os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as out:
        while written<item['bytes']:
            start=written;end=min(item['bytes']-1,start+chunk-1);maximum=end-start+1;url=item['url']
            for redirects in range(LIMITS['max_redirects_per_range']+1):
                allowed_delivery(url,item)
                headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity','User-Agent':'OpenWebRL-Map-original-identity'}
                if item['kind']=='dataset':headers['If-Match']=item['etag']
                elif urllib.parse.urlsplit(url).hostname==urllib.parse.urlsplit(item['registry']).hostname:headers['Authorization']='Bearer '+registry.token(item)
                i,response=request_once(lib,ledger,urllib.request.Request(url,headers=headers),maximum,deadline,open_response=open_response)
                if response.status in (301,302,303,307,308):
                    location=response.headers.get('Location','');response.close();ledger.settle(i,0,complete=False)
                    if redirects>=LIMITS['max_redirects_per_range']:raise Closed('Registry redirect count exceeded')
                    url=allowed_delivery(urllib.parse.urljoin(url,location),item);continue
                received=0;complete=False
                try:
                    wanted=f'bytes {start}-{end}/{item["bytes"]}'
                    valid=(response.status==206 and response.headers.get('Content-Range')==wanted)
                    valid|=(item['kind']=='image_blob' and response.status==200 and start==0 and maximum==item['bytes'])
                    if (not valid or response.headers.get('Content-Length')!=str(maximum) or
                        response.headers.get('Content-Encoding','identity')!='identity' or
                        item['kind']=='dataset' and response.headers.get('ETag')!=item['etag']):raise Closed('Range/length/entity identity changed')
                    while received<maximum:
                        if time.time()>=deadline:raise Closed('Transfer deadline exhausted')
                        data=response.read(min(2**20,maximum-received))
                        if not data:raise Closed('Truncated dataset/blob range')
                        out.write(data);h.update(data);received+=len(data);written+=len(data)
                    # Content-Length framing bounds reads: never read a byte
                    # beyond the reservation; later SHA/entity checks apply.
                    complete=True
                finally:response.close();ledger.settle(i,received,complete=complete)
                break
            else:raise Closed('No accepted response')
        out.flush();os.fsync(out.fileno())
    result=dict(filename=item['filename'],bytes=written,sha256=h.hexdigest())
    if item.get('sha256') and result['sha256']!=item['sha256']:raise Closed('Pinned OCI compressed blob checksum mismatch')
    return result


def oci_metadata(root,images):
    """Create per-image OCI layouts sharing checked already-downloaded blobs."""
    blobdir=checked(root/'oci/blobs/sha256');blobdir.mkdir(mode=0o700,parents=True,exist_ok=True)
    manifests=[]
    for item in images:
        prefix=item['metadata_prefix']
        for i,key in ((1,'amd64_manifest_digest'),(2,'config_digest')):
            source=checked(root.parent/'metadata'/(prefix+'-'+str(i)+'.json'));target=blobdir/item[key][7:]
            if not target.exists():
                with target.open('xb') as out:out.write(source.read_bytes());out.flush();os.fsync(out.fileno())
            if sha(target)!=item[key][7:]:raise Closed('OCI metadata content address conflict')
        manifests.append(dict(mediaType=item['manifest']['mediaType'],digest=item['amd64_manifest_digest'],
            size=(root.parent/'metadata'/(prefix+'-1.json')).stat().st_size,
            annotations={'org.opencontainers.image.ref.name':item['image']}))
    put(root/'oci/oci-layout',dict(imageLayoutVersion='1.0.0'))
    put(root/'oci/index.json',dict(schemaVersion=2,manifests=manifests))


def launch(root):
    p=approved(root);lib=library(root);nonce=secrets.token_hex(16);seconds=p['startup_recovery']['requested_allocation_seconds']
    argv=['salloc','--account=zixianma','--partition=gpu-h200','--qos=normal','--nodes=1','--ntasks=1','--cpus-per-task=2','--mem=8G',
        f'--time={seconds//3600:02d}:{seconds//60%60:02d}:00','--comment=webprm-acquire:'+nonce,'--job-name=wa-map-backend',
        '--chdir='+str(root),'srun','--ntasks=1','--cpus-per-task=2',sys.executable,'-I','-B',str(SCRIPT),'--run','--root',str(root),'--nonce',nonce]
    put(root/'launch.json',dict(nonce=nonce,proposal_sha256=digest(p),requested_allocation_seconds=seconds,argv=argv,epoch=time.time()))
    env=lib.safe_environment(root/'launch-cache');env={k:v for k,v in env.items() if not k.startswith('SLURM_')}
    os.chdir(root);os.execvpe(argv[0],argv,env)


def run(root,nonce):
    p=approved(root);claim=read(root/'launch.json');job=os.environ.get('SLURM_JOB_ID','')
    if (not nonce or claim.get('nonce')!=nonce or claim.get('proposal_sha256')!=digest(p) or not job.isdigit() or
        not re.search(r'job[_/\-]'+re.escape(job)+r'(?:/|$)',Path('/proc/self/cgroup').read_text())):raise Closed('Owned Slurm allocation claim required')
    put(root/'run-claim.json',dict(job_id=job,nonce=nonce,epoch=time.time()))
    try:
        raw=subprocess.check_output(['scontrol','--json','show','job',job],text=True,timeout=15);snapshot=json.loads(raw)
        put(root/'scheduler.json',snapshot);clock=base.context(snapshot,job,time.time(),maximum_seconds=p['startup_recovery']['requested_allocation_seconds'])
        lib=library(root);items,images=inputs(root)
        prior_retained,prior_temporary=base.previous_usage(root,p,lib)
        if prior_retained>=LIMITS['retained_bytes'] or prior_temporary>=LIMITS['temporary_bytes']:raise Closed('Preserved startups exhausted storage cap')
        lib.LIMITS=dict(LIMITS,temporary_bytes=LIMITS['temporary_bytes']-prior_temporary)
        if shutil.disk_usage(root).free<LIMITS['initial_persistent_free_bytes'] or shutil.disk_usage('/tmp').free<2**30:raise Closed('Approved storage headroom unavailable')
        scratch=Path('/tmp')/('webprm-map-backend-'+job+'-'+secrets.token_hex(4));scratch.mkdir(mode=0o700)
        attempt=root/('attempt-'+job);attempt.mkdir(mode=0o700)
        put(attempt/'context.json',dict(clock,scratch=str(scratch),proposal_sha256=digest(p)))
        def worker(output,scratch,deadline):
            os.setsid();os.chdir(output);os.environ.update(lib.safe_environment(scratch))
            resource.setrlimit(resource.RLIMIT_AS,(6*2**30,6*2**30));resource.setrlimit(resource.RLIMIT_FSIZE,(LIMITS['retained_bytes'],)*2)
            try:
                put(root/'network-claim.json',dict(job_id=job,epoch=time.time(),maximum_transfer_attempts=1))
                ledger=lib.TransferLedger(output,cap=LIMITS['download_charged_bytes'],metadata_cap=LIMITS['metadata_received_bytes'])
                registry=Registry(lib,ledger,deadline);receipts=[]
                for i,item in enumerate(items):
                    proof=transfer(output,item,lib,ledger,deadline,registry);put(output/f'item-{i:03d}-verified.json',proof);receipts.append(proof)
                oci_metadata(output,images)
                put(output/'stage-result.json',dict(status='acquired_transport_identity_verified',items=receipts,
                    charged_bytes=ledger.data['charged_bytes'],raw_diff_ids_verified=False,extracted=False,
                    ready_to_host=False,model_calls=0,browser_calls=0))
            except BaseException as error:
                lib.put(output/'failure.json',dict(error_type=type(error).__name__,error=str(error)[:2000],epoch=time.time()));raise SystemExit(1)
        closure=lib.own_worker(attempt,scratch,clock['work_deadline'],target=worker,prior_retained_bytes=prior_retained)
        result=read(attempt/'stage-result.json')
        if (closure['remaining_owned_processes'] or lib.apparent_bytes(root)+prior_retained>LIMITS['retained_bytes'] or
            lib.apparent_bytes(scratch)+prior_temporary>LIMITS['temporary_bytes'] or time.time()>=clock['allocation_deadline']):raise Closed('Closure/storage/allocation boundary verification failed')
        put(root/'completed.json',dict(status='acquisition_complete',result=result,job_id=job,ready_to_host=False,epoch=time.time()))
    except BaseException as error:
        put(root/'failure.json',dict(error_type=type(error).__name__,error=str(error)[:2000],epoch=time.time()));raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);modes=parser.add_mutually_exclusive_group(required=True)
    for name in ('prepare','launch','run'):modes.add_argument('--'+name,action='store_true')
    parser.add_argument('--root',required=True,type=Path);parser.add_argument('--nonce',default='')
    parser.add_argument('--predecessor',type=Path);parser.add_argument('--prior-job');parser.add_argument('--repair-receipt',type=Path)
    args=parser.parse_args();root=checked(args.root)
    if args.prepare:print(json.dumps(dict(root=str(root),proposal_sha256=digest(prepare(root,predecessor=args.predecessor,prior_job=args.prior_job,repair_receipt=args.repair_receipt)))))
    elif args.launch:launch(root)
    else:run(root,args.nonce)


if __name__=='__main__':main()
