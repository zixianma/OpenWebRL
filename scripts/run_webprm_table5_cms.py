#!/usr/bin/env python3
"""Exact-context Magento (CMS shopping_admin, Shopping) build/copy/serve stages for the Table 5 queue.

One script serves both third-party Magento sites; the campaign's single hosted domain
selects the site module (webprm_cms_site or webprm_shopping_site), whose constants
give the endpoint, validated hosting recipe, materializer limits and receipt kinds.

Mirrors scripts/run_webprm_table5_reddit.py with the hosting recipe validated by job
357831 (scripts/validate_webprm_cms_hosting.py), whose primitives are reused unchanged:
build materializes the pinned OCI base once; copy makes a fresh per-cell state as the
namespace-mapped root and installs the reviewed runtime controls; serve brings the
services up in a fresh user+net namespace, rewrites the origin to 127.0.0.1:7780,
publishes a generic original-site instance receipt and stays up until stop.json.
It cannot allocate resources and rejects missing/unapproved/stale parent context first.
"""
import argparse
import json
import math
import multiprocessing
import os
from pathlib import Path
import secrets
import signal
import socket
import sys
import time

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from scripts.launch_webprm_table5_campaign import BROWSER,read,sha,digest,checked,cell_key,DISPOSAL_PROTOCOL
from scripts.run_webprm_table5_reddit import context
from scripts import validate_webprm_cms_hosting as owner
from openwebrl import webprm_cms_hosting as h
from openwebrl import webprm_cms_materializer as materializer
from openwebrl import webprm_cms_site as site
from openwebrl import webprm_shopping_site as shopping_site
from openwebrl.webprm_lifecycle import OwnedEpisodeTree,_process,_processes,_subreaper
from openwebrl.webprm_webarena_sites import INSTANCE_SCHEMA

SITES={'shopping_admin':site,'shopping':shopping_site}


def origin_commands(s):
    return (['config:set','admin/security/password_is_forced','0'],
            ['config:set','admin/security/password_lifetime','0'],
            ['setup:store-config:set','--base-url='+s.ORIGIN])


def secure_sql(s):
    return b"UPDATE core_config_data SET value='"+s.ORIGIN.encode()+b"/' WHERE path='web/secure/base_url';\n"


ORIGIN_COMMANDS=origin_commands(site)
SECURE_SQL=secure_sql(site)


def site_of(owned):
    """The campaign hosts exactly one Magento domain."""
    domains=[d for d in owned['proposal']['domains'] if d in SITES]
    if len(domains)!=1 or len(owned['proposal']['domains'])!=1:
        raise ValueError('A Magento campaign hosts exactly one validated Magento domain')
    return SITES[domains[0]]


def ref(path):
    return dict(path=str(path),sha256=sha(path))


def put(path,value):
    """Atomic publication: readers polling for existence never see a partial file."""
    path=Path(path);tmp=path.with_name('.'+path.name+'.tmp')
    h.put(tmp,value);os.rename(tmp,path)


def hosting_plan(owned):
    s=site_of(owned);entry=owned['proposal']['domains'].get(s.DOMAIN,{})
    if (entry.get('endpoint')!=s.ENDPOINT or entry.get('hosting_proposal')!=s.HOSTING_PROPOSAL
            or entry.get('hosting_validation')!=s.HOSTING_VALIDATION
            or entry.get('base_content_sha256')!=s.BASE_CONTENT_SHA256):
        raise ValueError('Campaign '+s.LABEL+' domain entry differs from the validated hosting recipe')
    proposal,_=s.hosting()
    if proposal['service_plan']['sha256']!=s.hosting_profile().plan_sha:raise ValueError('Reviewed '+s.LABEL+' service plan differs')
    return proposal,h.read(h.reference(proposal['service_plan']))


def stage_paths(owned,mode,state,out):
    scratch=checked(owned['scratch']);output=checked(owned['output'])
    state,out=checked(state),checked(out)
    if scratch not in state.parents or output not in out.parents:
        raise ValueError('State/output must belong to this fresh owned attempt')
    if mode=='build':
        if state!=scratch/'baseline' or out!=output/'original-build':
            raise ValueError('Build must create the fixed private immutable baseline')
        return state,out
    if mode not in ('copy','serve'):raise ValueError('Unknown Magento stage')
    claim=read(checked(out.parent/'state-claim.json'))
    cells=read(owned['proposal']['cohort_path']);cells=cells.get('cohort',cells)['cells']
    cell=next((c for c in cells if c['cell_id']==claim.get('cell_id')),None)
    if (cell is None or cell['domain']!=site_of(owned).DOMAIN or claim.get('protocol')!=DISPOSAL_PROTOCOL
            or claim.get('policy_sha256')!=digest(owned['proposal']['ephemeral_policy'])
            or claim.get('campaign_root')!=owned['proposal']['campaign_root']
            or claim.get('scratch_root')!=str(scratch) or claim.get('base_path')!=str(scratch/'baseline')
            or claim.get('state_path')!=str(state) or claim.get('destination_absent_at_claim') is not True
            or claim.get('preserve_historical_states') is not True
            or state!=scratch/'states'/cell_key(cell)
            or out!=output/'cells'/cell_key(cell)/('copy' if mode=='copy' else 'site')
            or mode=='copy' and state.exists() or mode=='serve' and not state.is_dir()):
        raise ValueError('Stage requires this fixed cell\'s fresh private state claim')
    return state,out


def controls_path(owned,state):
    # Node-local and private: the DB credential file must never enter durable cell evidence.
    return checked(owned['scratch'])/'controls'/state.name


def build(owned,state,out):
    proposal,_=hosting_plan(owned);deadline=owned['deadline'];s=site_of(owned)
    if state.exists() or out.exists():raise ValueError('Build destination/receipt must be fresh')
    out.mkdir(mode=0o700)
    proof=h.read(h.reference(proposal['layer_proof']));manifest=h.read(h.reference(proposal['manifest']))
    raw={r['compressed_sha256']:r for r in proof['unique_layers']}
    layers=[dict(path=str(Path(proposal['blob_root'])/l['digest'][7:]),compressed_bytes=l['size'],**raw[l['digest'][7:]])
            for l in manifest['layers']]
    receipt=materializer.materialize(layers,state,out/'materialization',min(deadline,time.time()+s.MATERIALIZE_SECONDS),
        s.materializer_limits())
    path=materializer.private_path(Path(receipt['manifest']['path']))
    if sha(path)!=receipt['manifest']['sha256']:raise ValueError('Materialization manifest changed')
    content=s.content_sha256(json.loads(path.read_text()))
    if content!=s.BASE_CONTENT_SHA256:raise ValueError('Materialized '+s.LABEL+' base differs from the validated base')
    put(out/'receipt.json',dict(schema_version=1,kind=s.BUILD_KIND,status='completed',baseline_path=str(state),
        materialization=receipt['manifest'],materialized_content_sha256=content,final_regular_bytes=receipt['final_regular_bytes'],
        hosting_proposal=s.HOSTING_PROPOSAL,hosting_validation=s.HOSTING_VALIDATION,controller_sha256=sha(__file__),
        original_state_equivalence_verified=False,finished_epoch=time.time()))
    return dict(receipt_path=str(out/'receipt.json'),status='completed')


def copy(owned,state,out):
    _,plan=hosting_plan(owned);deadline=owned['deadline'];s=site_of(owned)
    build_path=checked(owned['output'])/'original-build/receipt.json';built=read(build_path)
    source=checked(owned['scratch'])/'baseline'
    if (built.get('kind')!=s.BUILD_KIND or built.get('status')!='completed' or built.get('baseline_path')!=str(source)
            or built.get('materialized_content_sha256')!=s.BASE_CONTENT_SHA256):
        raise ValueError('Copy requires this allocation\'s completed validated '+s.LABEL+' build')
    controls=controls_path(owned,state)
    if out.exists() or controls.exists():raise ValueError('Copy receipt/controls must be fresh')
    out.mkdir(mode=0o700);controls.parent.mkdir(mode=0o700,exist_ok=True)
    argv=['unshare','--user','--map-root-user','cp','-a','--reflink=auto','--',str(source),str(state)]
    owner.command(argv,out/'copy.log',deadline,timeout=1800)
    owner.prepare_controls(state,controls,plan)
    put(out/'receipt.json',dict(schema_version=1,kind=s.COPY_KIND,status='completed',returncode=0,argv=argv,
        state_path=str(state),source_path=str(source),build_receipt=ref(build_path),controls_path=str(controls),
        runtime_adaptations=ref(controls/'runtime-adaptations.json'),controller_sha256=sha(__file__),
        finished_epoch=time.time()))
    return dict(receipt_path=str(out/'receipt.json'),status='completed')


def instance_receipt(owned,state,out):
    s=site_of(owned)
    copy_ref=ref(out.parent/'copy/receipt.json');copied=read(copy_ref['path'])
    base=s.base_identity();instance_id=owned['job_id']+'-'+out.parent.name[:16]
    provenance=dict(domain=s.DOMAIN,base_identity_sha256=digest(base),instance_id=instance_id,state_path=str(state),
        reset_validation=s.HOSTING_VALIDATION,build_receipt=copied['build_receipt'],copy_receipt=copy_ref)
    me=_process(os.getpid())
    return dict(schema=INSTANCE_SCHEMA,domain=s.DOMAIN,scope='original',instance_id=instance_id,
        endpoint=s.ENDPOINT,state_path=str(state),base_identity=base,base_identity_sha256=digest(base),
        state_provenance=provenance,state_provenance_sha256=digest(provenance),job_id=os.environ['SLURM_JOB_ID'],
        hostname=socket.gethostname(),controller_pid=owned['owner_pid'],controller_start_ticks=owned['owner_start_ticks'],
        service_pid=os.getpid(),service_start_ticks=me['start_ticks'],network_namespace=os.readlink('/proc/self/ns/net'),
        service_validation=s.HOSTING_VALIDATION,reset_validation=s.HOSTING_VALIDATION,
        third_party_image=True,original_state_equivalence_verified=False,**s.instance_extras())


def validate_handoff(owned,state,out,token):
    launch=read(out/'launch.json')
    if (not token or launch.get('launch_token')!=token or launch.get('state')!=str(state) or launch.get('output')!=str(out)
            or launch.get('controls')!=str(controls_path(owned,state))):
        raise ValueError('Inside launch identity mismatch')
    if os.geteuid()!=0 or os.readlink('/proc/self/ns/net')==launch['outer_network_namespace']:
        raise ValueError('Inside requires a new owned rootmapped network namespace')
    if os.getppid()!=launch['parent_pid']:raise ValueError('Inside process is not the recorded supervisor child')
    parent=_process(launch['parent_pid'])
    if parent['start_ticks']!=launch['parent_start_ticks'] or parent['state']=='Z':
        raise ValueError('Outer supervisor is stale')
    if ref(out.parent/'copy/receipt.json')!=launch['copy_receipt']:raise ValueError('Copy receipt changed across launch')
    return launch


def inside(owned,state,out,token):
    """Runs as the namespace's mapped root; this process is the instance's service_pid."""
    launch=validate_handoff(owned,state,out,token);proposal,_=hosting_plan(owned);s=site_of(owned)
    deadline=owned['deadline'];controls=Path(launch['controls'])
    prefix=owner.container_prefix(dict(state=str(state),controls=str(controls)),proposal)
    owner.command(['/usr/sbin/ip','link','set','lo','up'],out/'loopback.log',deadline)
    # The native DB endpoint/credentials stay in node-local private controls only.
    raw=owner.command(prefix+['php','-r',"$a=require '/var/www/magento2/app/etc/env.php'; echo json_encode($a['db']['connection']['default']);"],
        controls/'private-db-discovery.json',deadline,separate_stderr=True)
    owner.bytes_new(controls/'private-db.cnf',owner.private_db_options(json.loads(raw)))
    mysql=prefix+['mysql','--defaults-extra-file=/opt/webprm-cms/private-db.cnf','--batch','--raw','--skip-column-names','magentodb']
    previous=_subreaper(True);baseline=set(_processes());process=tree=None;stop=[False];exitcode=None
    handlers={sig:signal.signal(sig,lambda *_:stop.__setitem__(0,True)) for sig in (signal.SIGTERM,signal.SIGINT)}
    try:
        process=multiprocessing.get_context('fork').Process(target=owner.service_child,
            args=(prefix+['/bin/sh','/opt/webprm-cms/bootstrap.sh'],str(out/'services.log')))
        try:process.start()
        finally:
            if process.pid is not None:tree=OwnedEpisodeTree(process,baseline)
        for attempt in range(120):
            tree.refresh();materializer.check(deadline)
            if not process.is_alive():raise RuntimeError(s.LABEL+' services exited during startup')
            try:
                owner.command(mysql,out/f'db-ready-{attempt:03d}.log',deadline,input_bytes=b'SELECT 1;\n',timeout=10);break
            except h.Closed:time.sleep(2)
        else:raise RuntimeError('Imported '+s.LABEL+' database did not become ready')
        magento=prefix+['php','/var/www/magento2/bin/magento']
        for i,argv in enumerate(origin_commands(s)):
            owner.command(magento+argv,out/f'origin-php-{i}.log',deadline,timeout=600)
        owner.command(mysql,out/'origin-secure.log',deadline,input_bytes=secure_sql(s))
        owner.command(magento+['cache:flush'],out/'cache-flush.log',deadline,timeout=600)
        owner.command(prefix+['supervisorctl','-c','/opt/webprm-cms/supervisord.conf','start','cron'],out/'start-cron.log',deadline)
        put(out/'instance.json',instance_receipt(owned,state,out))
        while process.is_alive() and time.time()<deadline and not stop[0] and not (out/'stop.json').exists():
            tree.refresh();time.sleep(.5)
        if not process.is_alive():
            put(out/'service-exit.json',dict(returncode=process.exitcode,at_epoch=time.time()))
            raise RuntimeError(s.LABEL+' services exited while the cell was active')
    finally:
        try:
            if process is not None and process.is_alive():
                try:owner.command(prefix+['supervisorctl','-c','/opt/webprm-cms/supervisord.conf','shutdown'],
                        out/'shutdown.log',max(deadline,time.time()+60),timeout=30)
                except Exception:pass
                process.join(timeout=30)
            closure=tree.stop(10) if tree else dict(remaining_owned_processes={},child_exitcode=None)
            exitcode=closure.get('child_exitcode')
            put(out/'service-closure.json',closure)
        finally:
            _subreaper(previous)
            for sig,handler in handlers.items():signal.signal(sig,handler)
    if closure['remaining_owned_processes']:raise RuntimeError(s.LABEL+' service processes remain')
    return exitcode


def child(argv):
    os.setsid();os.execvpe(argv[0],argv,dict(os.environ))


def supervise(owned,state,out,context_path):
    """Owns and reaps the whole namespace tree; the inside process shuts services down."""
    deadline=owned['deadline']
    if not math.isfinite(deadline) or deadline<=time.time():raise ValueError('Service deadline already expired')
    controls=controls_path(owned,state)
    out.mkdir(mode=0o700);token=secrets.token_hex(24)
    put(out/'launch.json',dict(state=str(state),output=str(out),controls=str(controls),launch_token=token,
        parent_pid=os.getpid(),parent_start_ticks=_process(os.getpid())['start_ticks'],
        outer_network_namespace=os.readlink('/proc/self/ns/net'),copy_receipt=ref(out.parent/'copy/receipt.json'),
        owner_context_path=str(context_path),owner_context_sha256=sha(context_path),service_adapter_sha256=sha(__file__)))
    argv=['unshare','--user','--map-root-user','--net',BROWSER,'-B',str(Path(__file__).resolve()),'--context',str(context_path),
        '--mode','serve','--state',str(state),'--output',str(out),'--inside','--launch-token',token]
    stop=[False];handlers={sig:signal.signal(sig,lambda *_:stop.__setitem__(0,True)) for sig in (signal.SIGTERM,signal.SIGINT)}
    previous=None;p=tree=None;result=None
    try:
        previous=_subreaper(True);baseline=set(_processes())
        p=multiprocessing.get_context('fork').Process(target=child,args=(argv,))
        try:p.start()
        finally:
            if p.pid is not None:tree=OwnedEpisodeTree(p,baseline)
        while p.is_alive() and time.time()<deadline and not stop[0]:
            tree.refresh();time.sleep(.5)
        if stop[0] and p.is_alive():
            # Forward the stop to the inside owner for a graceful service shutdown.
            try:os.kill(p.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            p.join(timeout=60)
    finally:
        try:
            result=tree.stop(10) if tree is not None else dict(owned_pids=[],remaining_owned_processes={},child_exitcode=None,child_started=False)
            result.update(finished_epoch=time.time(),scope='original',domain=site_of(owned).DOMAIN);put(out/'closure.json',result)
        finally:
            if previous is not None:_subreaper(previous)
            for sig,handler in handlers.items():signal.signal(sig,handler)
    if result['remaining_owned_processes']:raise RuntimeError('Owned processes remain')
    if result['child_exitcode'] not in (0,-signal.SIGTERM,-signal.SIGKILL):raise RuntimeError('Owned Magento service controller failed')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--context',required=True);parser.add_argument('--mode',choices=('build','copy','serve'),required=True)
    parser.add_argument('--state',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--inside',action='store_true');parser.add_argument('--launch-token')
    args=parser.parse_args();owned=context(args.context)
    state,out=stage_paths(owned,args.mode,args.state,args.output)
    if args.mode!='serve' and (args.inside or args.launch_token):raise ValueError('Preparation has no namespace handoff')
    if args.mode=='build':result=build(owned,state,out)
    elif args.mode=='copy':result=copy(owned,state,out)
    elif args.inside:return inside(owned,state,out,args.launch_token)
    else:result=supervise(owned,state,out,checked(args.context))
    print(json.dumps(dict(mode=args.mode,status=result.get('status','closed'))),flush=True)


if __name__=='__main__':main()
