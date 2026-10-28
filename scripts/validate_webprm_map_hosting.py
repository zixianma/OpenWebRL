#!/usr/bin/env python3
"""Owned Map reconstruction validation; prepare is offline, execution needs exact approval.

The pinned frontend image contains no populated PostgreSQL volume. This owner
tests the released fresh-schema initialization and records browse equivalence
as unsupported. It never invents OSM rows. Independent full copies A/B test
native geocoding/routes, frontend SQL mutation, fresh schema and browser session.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import re
import secrets
import shlex
import shutil
import subprocess
import sys
import tarfile
import time

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts import validate_webprm_cms_hosting as common
from openwebrl import webprm_cms_hosting as h
from openwebrl.webprm_map_controls import web_database_environment,private_hosts
from openwebrl.webprm_lifecycle import OwnedEpisodeTree,_process,_processes,_subreaper

N=common.N
PROTOCOL='map_fresh_schema_owned_validation_v1'
RESOURCES=dict(account='zixianma',partition='gpu-h200',qos='normal',nodes=1,gpus=0,cpus=8,memory_gib=128,total_seconds=14400,max_allocations=1)
LIMITS=dict(scratch_bytes=384*2**30,retained_bytes=4*2**30,runtime_reserve_bytes=16*2**30,closure_seconds=600,model_calls=0,api_usd=0,download_bytes=0)
FILES=tuple(dict.fromkeys((*common.SOURCES,'scripts/validate_webprm_map_hosting.py','openwebrl/webprm_map_controls.py','tests/test_webprm_map_controls.py','tests/test_validate_webprm_map_hosting.py')))


def profile(proposal):
    p=h.Profile('map','Map',PROTOCOL,RESOURCES,LIMITS,proposal['service_plan']['sha256'],{},False,
        apparent_entries=30000000,sample_seconds=60,cumulative_scratch=True)
    h.PROFILES[PROTOCOL]=p
    return p


def approved(proposal,approval):
    p=profile(proposal)
    h.validate(proposal,approval)
    basis=h.read(h.reference(proposal['source_basis']))
    for reference in basis['references']:h.reference(reference)
    if (proposal.get('frontend_database')!='released_fresh_schema_reconstruction' or
            proposal.get('browse_equivalence_supported') is not False or
            proposal.get('original_state_equivalence_verified') is not False):
        raise h.Closed('Map reconstruction disclosure changed')
    return p


def recipe():
    spec=importlib.util.spec_from_file_location('frozen_map_recipe',REPO/'scripts/webprm_map_recipe.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod


def controls_pass(result):
    """Fail closed on missing/invalid backend or released native controls."""
    if result.get('backend_pass')!={'tiles':True,'geocode':True,'routes':True}:
        raise h.Closed('All five-city native backends must pass')
    browser=result.get('browser',{})
    if browser.get('released_map_tasks')!=26 or browser.get('home',{}).get('status')!=200:
        raise h.Closed('Released Map frontend coverage changed')
    for key in ('task90_url_match','task87_program_html','task159_program_html'):
        control=browser.get('evaluator',{}).get(key,{})
        if any(control.get(label,{}).get('valid') is not True or control[label].get('score')!=score
               for label,score in (('positive',1),('negative',0))):
            raise h.Closed('Released native Map positive/negative control failed: '+key)
    if any(not isinstance(browser.get('ui_routes',{}).get(k),str) or
           browser['ui_routes'][k].startswith('error:') for k in ('Foot','Bicycle')):
        raise h.Closed('Native foot/bicycle UI controls failed')


def record_controls(r,result,out):
    """Keep raw native evidence even when a later assertion rejects a phase."""
    r.backends(result)
    h.put(out/'backend-controls-raw.json',result)
    try:
        r.browser(result,storage_output=str(out/'storage-state-private.json'),diagnostics_output=str(out/'browser-progress-raw.json'))
    except BaseException as error:
        h.put(out/'native-controls-raw.json',result)
        h.put(out/'native-controls-verdict.json',dict(status='browser_failed',error_type=type(error).__name__,error=str(error),raw=common.ref(out/'native-controls-raw.json')))
        raise
    h.put(out/'native-controls-raw.json',result)
    try:
        controls_pass(result)
    except BaseException as error:
        h.put(out/'native-controls-verdict.json',dict(status='failed',error_type=type(error).__name__,error=str(error),raw=common.ref(out/'native-controls-raw.json')))
        raise
    h.put(out/'native-controls-verdict.json',dict(status='passed',raw=common.ref(out/'native-controls-raw.json')))


def reset_pass(first,second):
    for result in (first,second):controls_pass(result)
    if (first['instance_id']==second['instance_id'] or first['base_identity']!=second['base_identity'] or
            first['database_before']!=second['database_before'] or any(row['database_marker_after']!=1 for row in (first,second)) or
            second['database_marker_before']!=0 or
            first['browser'].get('initial_cookie_count')!=0 or second['browser'].get('initial_cookie_count')!=0 or
            not first['browser'].get('storage_state_sha256') or
            first['browser']['storage_state_sha256']==second['browser'].get('storage_state_sha256') or
            first.get('closure',{}).get('remaining_owned_processes')!={} or
            second.get('closure',{}).get('remaining_owned_processes')!={}):
        raise h.Closed('Fresh Map DB/session/state reset failed')
    return dict(fresh_reset_verified=True,native_frontend_controls_verified=True,original_state_equivalence_verified=False,
                frontend_database='released_fresh_schema_reconstruction',browse_equivalence_supported=False,
                native_authentication_required=False,model_calls=0,api_usd=0,download_bytes=0)


def extract(path,destination,prefix,deadline,maximum):
    """Python 3.12 data-filter extraction; full streaming input, bounded selected bytes."""
    destination.mkdir(mode=0o700,parents=True,exist_ok=False)
    total=count=0
    if not hasattr(tarfile,'data_filter'):raise h.Closed('Production Python 3.12 safe tar filter required')
    with tarfile.open(path,'r|*') as archive:
        for item in archive:
            if time.time()>=deadline:raise h.Closed('Map extraction deadline')
            if not item.name.startswith(prefix):continue
            name=item.name[len(prefix):].lstrip('/')
            if not name:continue
            count+=1;total+=item.size if item.isfile() else 0
            if count>5000000 or total>maximum or item.sparse:raise h.Closed('Map extraction byte/member/sparse cap')
            item.name=name
            # Source links need their target stripped using the same archive prefix.
            if item.islnk():
                if not item.linkname.startswith(prefix):raise h.Closed('Map hardlink leaves selected original subtree')
                item.linkname=item.linkname[len(prefix):].lstrip('/')
            archive.extract(item,destination,filter='data',set_attrs=True)
            archive.members.clear()
    if not count:raise h.Closed('Expected Map archive prefix absent')
    return dict(selected_bytes=total,selected_members=count,source=str(path),prefix=prefix)


def checked_context(path,inside=False):
    context=h.read(path);proposal=h.read(context['proposal']);approval=h.read(context['approval']);approved(proposal,approval)
    if (h.digest(proposal)!=context['proposal_sha256'] or time.time()>=context['deadline'] or
            context['job_id']!=os.environ.get('SLURM_JOB_ID') or REPO!=Path(proposal['frozen_source'])):
        raise h.Closed('Frozen approved Map context changed')
    owner=_process(context['owner_pid'])
    if owner['start_ticks']!=context['owner_start_ticks'] or owner['state']=='Z':raise h.Closed('Map owner exited')
    pid=os.getppid();seen=set()
    while pid!=context['owner_pid'] and pid>1 and pid not in seen:
        seen.add(pid);pid=_process(pid)['ppid']
    if pid!=context['owner_pid']:raise h.Closed('Map worker escaped allocation owner')
    if inside and (os.geteuid()!=0 or os.readlink('/proc/self/ns/net')==context['outer_net']):
        raise h.Closed('Fresh rootmapped private namespace required')
    return context,proposal


def command(argv,output,deadline,**kwargs):
    return common.command(argv,output,deadline,**kwargs)


def input_sha(path,deadline):
    """Stream large acquired datasets without allocating their size in RAM."""
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        before=os.fstat(stream.fileno())
        while True:
            if time.time()>=deadline:raise h.Closed('Map source verification deadline')
            chunk=stream.read(2**20)
            if not chunk:break
            digest.update(chunk)
        after=os.fstat(stream.fileno())
    identity=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    if identity(before)!=identity(after) or identity(after)!=identity(Path(path).stat()):
        raise h.Closed('Map source identity changed during readback')
    return digest.hexdigest()


def inside(path):
    context,proposal=checked_context(path,inside=True);r=recipe();state=Path(context['state']);out=Path(context['output'])
    out.mkdir(mode=0o700);logs=state/'logs';logs.mkdir(mode=0o700)
    r.W=state
    command(['/usr/sbin/ip','link','set','lo','up'],out/'loopback.log',context['deadline'])
    rf=state/'rootfs';d=state/'data'
    jobs={'tile':r.prefix(rf/'tile',[(d/'tile','/data/database')])+['/bin/bash','/webprm/ctl.sh'],
          'nominatim':r.prefix(rf/'nominatim',[(d/'nominatim','/var/lib/postgresql/14/main')])+['/bin/bash','/webprm/ctl.sh'],
          'db':r.prefix(rf/'db',[])+['/bin/bash','/webprm/ctl.sh'],
          'web':r.prefix(rf/'web',[],fakeroot=False)+['/bin/bash','/webprm/ctl.sh']}
    for mode,port in (('car',5000),('bike',5001),('foot',5002)):
        jobs['osrm-'+mode]=r.prefix(rf/'osrm',[(d/'osrm'/mode,'/data')],fakeroot=False,writable=False)+[
            'osrm-routed','--algorithm','mld','--threads','2','-p',str(port),'/data/us-northeast-latest.osrm']
    master=state/'master.sh';master.write_text('#!/bin/bash\n'+''.join(
        shlex.join(argv)+' > '+shlex.quote(str(logs/name)+'.log')+' 2>&1 &\n' for name,argv in jobs.items())+'wait\n')
    previous=_subreaper(True);baseline=set(_processes());tree=None;result=None
    try:
        process=multiprocessing.get_context('fork').Process(target=common.service_child,args=(['/bin/bash',str(master)],str(out/'services.log')))
        try:process.start()
        finally:
            if process.pid is not None:tree=OwnedEpisodeTree(process,baseline)
        stop=min(context['deadline'],time.time()+1800)
        while time.time()<stop:
            tree.refresh()
            if not process.is_alive():raise h.Closed('Map service owner exited during startup')
            if r.get(r.ORIGIN+'/',30)[0]==200 and r.get('http://127.0.0.1:8085/status.php',10)[0]==200:break
            time.sleep(5)
        else:raise h.Closed('Map native services failed startup')
        sql=r.prefix(rf/'db',[])+['psql','-h','127.0.0.1','-U','openstreetmap','-d','openstreetmap','-At','-v','ON_ERROR_STOP=1']
        baseline_sql=b"SELECT tablename FROM pg_tables WHERE schemaname='public' ORDER BY tablename; SELECT json_build_object('current_nodes',(SELECT count(*) FROM current_nodes),'current_ways',(SELECT count(*) FROM current_ways),'current_relations',(SELECT count(*) FROM current_relations));\n"
        before=command(sql,out/'database-before.txt',context['deadline'],input_bytes=baseline_sql,separate_stderr=True)
        marker_sql=b"SELECT count(*) FROM pg_tables WHERE schemaname='public' AND tablename='webprm_reset_marker';\n"
        marker=int(command(sql,out/'marker-before.txt',context['deadline'],input_bytes=marker_sql,separate_stderr=True).strip())
        if marker:raise h.Closed('Private fresh frontend unexpectedly contains reset marker')
        result=dict(instance_id=context['instance_id'],base_identity=context['base_identity'],database_before=hashlib.sha256(before).hexdigest(),database_marker_before=marker)
        record_controls(r,result,out)
        command(sql,out/'database-mutate.txt',context['deadline'],input_bytes=b"CREATE TABLE webprm_reset_marker (marker text NOT NULL); INSERT INTO webprm_reset_marker VALUES ('owned-reset-control');\n",separate_stderr=True)
        result['database_marker_after']=int(command(sql,out/'marker-after.txt',context['deadline'],input_bytes=marker_sql,separate_stderr=True).strip())
    finally:
        try:
            closure=tree.stop(30) if tree else dict(remaining_owned_processes={},child_exitcode=None)
            h.put(out/'service-closure.json',closure)
        finally:_subreaper(previous)
    if result is None or closure['remaining_owned_processes']:raise h.Closed('Map phase lacks closed native evidence')
    result['closure']=closure;h.put(out/'result.json',result)


def worker(path):
    os.setsid();context,proposal=checked_context(path);scratch=Path(context['scratch']);attempt=Path(context['attempt']);deadline=context['deadline']
    os.environ.update(h.safe_env(scratch));os.umask(0o077)
    os.environ['PYTHONPATH']=proposal['nltk_pythonpath']+':'+proposal['frozen_source'];os.environ['NLTK_DATA']=proposal['nltk_data']
    r=recipe();base=scratch/'base';base.mkdir(mode=0o700)
    for row in proposal['inputs']:
        if input_sha(row['path'],deadline)!=row['sha256']:raise h.Closed('Acquired Map input identity changed')
    rf=base/'rootfs';rf.mkdir()
    for name,image in (('tile','tile'),('nominatim','nominatim'),('osrm','osrm'),('db','website-db'),('web','website-web')):
        command(['apptainer','build','--sandbox',str(rf/name),str(r.SIF/(image+'.sif'))],attempt/('build-'+name+'.log'),deadline,timeout=1800)
    data=base/'data';data.mkdir()
    extraction=[]
    for path,dest,prefix,cap in ((r.TILE_TAR,data/'tile','projects/ogma3/docker/volumes/osm-data/_data/',48*2**30),
        (r.DS/'nominatim_volumes.tar',data/'nominatim','projects/metis2/docker/docker/volumes/nominatim-data/_data/',48*2**30),
        (r.DS/'osrm_routing.tar',data/'osrm','',32*2**30),
        (r.FRONT,base/'source','openstreetmap-website/',4*2**30)):
        extraction.append(extract(path,dest,prefix,deadline,cap))
    hosts=private_hosts(os.uname().nodename)
    for name in ('tile','nominatim','osrm','db','web'):
        for rel in ('opt/native-fakeroot','webprm'):(rf/name/rel).mkdir(parents=True,exist_ok=True)
        hp=rf/name/'etc/hosts'
        if hp.is_symlink():raise h.Closed('Unexpected hosts symlink')
        hp.write_text(hosts)
    for name,script in (('tile',r.TILE_CTL),('nominatim',r.NOM_CTL),('db',r.DB_CTL),('web',r.WEB_CTL)):
        (rf/name/'webprm/ctl.sh').write_text(script)
    r.edit(rf/'tile/etc/apache2/ports.conf',lambda s:re.sub(r'(?m)^Listen 80$','Listen 8080',s),'tile/apache-port')
    r.edit(r.inroot(rf/'tile','etc/apache2/sites-enabled/000-default.conf'),lambda s:s.replace('<VirtualHost *:80>','<VirtualHost *:8080>'),'tile/apache-vhost')
    peer=lambda s:re.sub(r'(?m)^(local\s+\S+\s+\S+\s+)(peer|md5|scram-sha-256)[ \t]*$',r'\1trust',s)
    r.edit(rf/'tile/etc/postgresql/15/main/pg_hba.conf',peer,'tile/local-auth')
    r.edit(rf/'nominatim/etc/apache2/ports.conf',lambda s:re.sub(r'(?m)^Listen 80$','',s),'nominatim/apache-port')
    r.edit(r.inroot(rf/'nominatim','etc/apache2/sites-enabled/000-default.conf'),lambda s:s.replace('Listen 8080','Listen 8085').replace('*:8080','*:8085'),'nominatim/apache-vhost')
    r.edit(rf/'nominatim/etc/postgresql/14/main/pg_hba.conf',peer,'nominatim/local-auth')
    r.edit(rf/'nominatim/nominatim/.env',web_database_environment,'nominatim/explicit-native-web-role')
    source=base/'source'
    r.edit(source/'config/settings.yml',lambda s:s.replace('nominatim_url: "http://metis.lti.cs.cmu.edu:8085/"','nominatim_url: "http://127.0.0.1:8085/"').replace('fossgis_osrm_url: "http://metis.lti.cs.cmu.edu:"','fossgis_osrm_url: "http://127.0.0.1:"'),'frontend/owned-endpoints')
    r.edit(source/'vendor/assets/leaflet/leaflet.osm.js',lambda s:s.replace('http://ogma.lti.cs.cmu.edu:8080/tile/','http://127.0.0.1:8080/tile/'),'frontend/owned-tiles')
    r.edit(source/'config/database.yml',lambda s:s.replace('host: db','host: 127.0.0.1'),'frontend/owned-database')
    command(['cp','-a',str(source)+'/.',str(rf/'web/app')+'/'],attempt/'frontend-source-copy.log',deadline)
    h.put(attempt/'configuration-changes.json',r.CHANGES);h.put(attempt/'extraction.json',extraction)
    size=h.apparent(base,maximum_entries=30000000)
    if 3*size+LIMITS['runtime_reserve_bytes']>LIMITS['scratch_bytes']:raise h.Closed('Three independent Map copies exceed scratch cap')
    identity=h.digest(dict(inputs=proposal['inputs'],configurations=r.CHANGES,frontend_database=proposal['frontend_database']))
    results=[]
    for label in ('a','b'):
        state=scratch/('state-'+label)
        command(['unshare','--user','--map-root-user','cp','-a','--reflink=auto','--',str(base),str(state)],attempt/('copy-'+label+'.log'),deadline,timeout=1800)
        child=dict(context,state=str(state),output=str(attempt/('instance-'+label)),instance_id=context['job_id']+'-'+label,base_identity=identity,
            owner_pid=os.getpid(),owner_start_ticks=_process(os.getpid())['start_ticks'],outer_net=os.readlink('/proc/self/ns/net'))
        child_path=attempt/('context-'+label+'.json');h.put(child_path,child)
        command(['unshare','--user','--map-root-user','--net',common.PYTHON,'-B',__file__,'--inside',str(child_path)],attempt/('namespace-'+label+'.log'),deadline,timeout=4800)
        results.append(h.read(Path(child['output'])/'result.json'))
    gate=reset_pass(*results)
    h.put(attempt/'validation-result.json',dict(status='passed',gate=gate,first=common.ref(attempt/'instance-a/result.json'),second=common.ref(attempt/'instance-b/result.json'),
        base_apparent_bytes=size,original_state_equivalence_verified=False,browse_equivalence_supported=False,model_calls=0,api_usd=0,downloads=0))


def run(proposal_path,approval_path):
    proposal,approval=h.read(proposal_path),h.read(approval_path);p=approved(proposal,approval)
    root=Path(proposal['campaign_root']);rows=h.history(root,p);job=os.environ.get('SLURM_JOB_ID','')
    if (not job.isdigit() or not rows or rows[-1]['job_id']!=job or rows[-1]['proposal_sha256']!=h.digest(proposal) or
            '/job_'+job+'/' not in Path('/proc/self/cgroup').read_text() or REPO!=Path(proposal['frozen_source'])):
        raise h.Closed('Durably registered frozen current Map allocation required')
    budget=h.prior_budget(root,rows[:-1],common.terminal_rows(rows[:-1]),p)
    if rows[-1]['requested_seconds']!=budget['remaining_seconds']:raise h.Closed('Map cumulative clock changed')
    raw=subprocess.check_output(['scontrol','--json','show','job',job],text=True,timeout=15)
    clock=h.scheduler(raw,job,requested_seconds=budget['remaining_seconds'],profile=p)
    attempt=root/('attempt-'+job);attempt.mkdir(mode=0o700);scratch=Path('/tmp')/('mapv-'+job+'-'+secrets.token_hex(4));scratch.mkdir(mode=0o700)
    h.put(attempt/'scheduler.json',dict(raw=raw))
    context=dict(proposal=str(proposal_path),approval=str(approval_path),proposal_sha256=h.digest(proposal),job_id=job,
        owner_pid=os.getpid(),owner_start_ticks=_process(os.getpid())['start_ticks'],deadline=clock['work_deadline'],
        scratch=str(scratch),attempt=str(attempt))
    path=attempt/'context.json';h.put(path,context)
    if shutil.disk_usage(scratch).free<LIMITS['scratch_bytes']:
        h.put(attempt/'closure.json',dict(remaining_owned_processes={},child_exitcode=None))
        h.put(attempt/'storage-final.json',dict(scratch_bytes=h.apparent(scratch),retained_bytes=h.apparent(root)))
        raise h.Closed('Insufficient Map scratch capacity')
    h.own(worker,(path,),attempt=attempt,scratch=scratch,campaign_root=root,deadline=clock['work_deadline'],profile=p)
    result=attempt/'validation-result.json'
    if h.read(result).get('status')!='passed':raise h.Closed('Map native reset validation incomplete')
    h.put(root/'completed.json',dict(status='hosting_validation_passed_pending_independent_audit',job_id=job,
        validation=common.ref(result),closure=common.ref(attempt/'closure.json'),original_state_equivalence_verified=False,browse_equivalence_supported=False,live_adapter_enabled=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--proposal',type=Path);parser.add_argument('--approval',type=Path)
    parser.add_argument('--inside',type=Path);parser.add_argument('--run',action='store_true');parser.add_argument('--submit',action='store_true')
    args=parser.parse_args()
    if args.inside:return inside(args.inside)
    if args.run:return run(args.proposal,args.approval)
    p,a=h.read(args.proposal),h.read(args.approval);approved(p,a)
    from scripts import submit_webprm_cms_hosting as submitter
    print(json.dumps(submitter.submit(args.proposal,args.approval) if args.submit else submitter.preview(args.proposal,args.approval)[2],indent=2))


if __name__=='__main__':main()
