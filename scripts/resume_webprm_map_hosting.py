#!/usr/bin/env python3
"""Explicit resumed-A/fresh-B Map hosting recovery, never a two-fresh-state claim.

All entry points require a hash-bound approval for this distinct protocol. The
original failed attempt, immutable base and original two-fresh verifier survive.
This module does not implement source acquisition or model calls.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import re
import secrets
import shlex
import shutil
import stat
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts import validate_webprm_map_hosting as old
from openwebrl.webprm_map_controls import private_hosts
from openwebrl.webprm_lifecycle import OwnedEpisodeTree,_process,_processes,_subreaper
h=old.h
PROTOCOL='map_resumed_a_fresh_b_owned_validation_v1'
RESOURCES=dict(account='vlm',partition='gpu-h200',qos='normal',nodes=1,gpus=0,cpus=8,memory_gib=128,total_seconds=14400,max_allocations=2)
LIMITS=dict(old.LIMITS)
PRIOR_BYTES=212292624932
NODE='g007'
OLD_JOB='362807'
SCHEMA_SQL=b"""SELECT json_build_object('schema', (SELECT json_agg(x ORDER BY x.table_name,x.ordinal_position) FROM (SELECT table_name,column_name,ordinal_position,column_default,is_nullable,data_type,udt_name,character_maximum_length,numeric_precision,numeric_scale FROM information_schema.columns WHERE table_schema='public') x),'constraints',(SELECT json_agg(x ORDER BY x.table_name,x.constraint_name) FROM (SELECT c.relname AS table_name,n.conname AS constraint_name,pg_get_constraintdef(n.oid) AS definition FROM pg_constraint n JOIN pg_class c ON c.oid=n.conrelid JOIN pg_namespace s ON s.oid=c.relnamespace WHERE s.nspname='public') x),'indexes',(SELECT json_agg(x ORDER BY x.tablename,x.indexname) FROM (SELECT tablename,indexname,indexdef FROM pg_indexes WHERE schemaname='public') x),'sequences',(SELECT json_agg(x ORDER BY x.sequence_name) FROM (SELECT sequence_name,data_type,start_value,minimum_value,maximum_value,increment,cycle_option FROM information_schema.sequences WHERE sequence_schema='public') x),'migrations',(SELECT json_agg(version ORDER BY version) FROM schema_migrations),'counts',json_build_object('current_nodes',(SELECT count(*) FROM current_nodes),'current_ways',(SELECT count(*) FROM current_ways),'current_relations',(SELECT count(*) FROM current_relations)));\n"""
MARKER_SQL=b"SELECT count(*) FROM pg_tables WHERE schemaname='public' AND tablename='webprm_reset_marker';\n"


def storage_charge(prior_recorded,old_live,new_live):
    """Old shrinkage never refunds historical use; old growth is charged once."""
    if any(type(x) is not int or x<0 for x in (prior_recorded,old_live,new_live)):
        raise h.Closed('Nonnegative integer storage measurements required')
    return dict(scratch_bytes=new_live+max(0,old_live-prior_recorded),
                prior_scratch_bytes=prior_recorded,physical_scratch_bytes=new_live,
                prior_live_scratch_bytes=old_live,total_charged_bytes=max(prior_recorded,old_live)+new_live)


def tree_fingerprint(root,deadline):
    """Full readback of every regular byte; no symlink traversal or timestamp claim."""
    root=Path(root)
    if root.is_symlink() or not root.is_dir():raise h.Closed('Retained real directory required')
    digest=hashlib.sha256();count=size=0
    def visit(directory):
        nonlocal count,size
        for entry in sorted(directory.iterdir(),key=lambda p:p.name):
            if time.time()>=deadline:raise h.Closed('Fingerprint deadline')
            s=entry.lstat();count+=1;size+=s.st_size
            if count>30000000:raise h.Closed('Fingerprint member cap')
            kind=stat.S_IFMT(s.st_mode)
            item=dict(path=str(entry.relative_to(root)),mode=s.st_mode)
            if stat.S_ISREG(kind):item.update(size=s.st_size,sha256=old.input_sha(entry,deadline))
            elif stat.S_ISLNK(kind):item['target']=os.readlink(entry)
            elif not stat.S_ISDIR(kind):raise h.Closed('Unexpected special retained file')
            digest.update(json.dumps(item,sort_keys=True,separators=(',',':')).encode()+b'\n')
            if stat.S_ISDIR(kind):visit(entry)
    visit(root)
    return dict(sha256=digest.hexdigest(),entries=count,apparent_bytes=h.apparent(root,maximum_entries=30000000))


def profile(proposal):
    p=h.Profile('map-recovery','Map resumed-A/fresh-B',PROTOCOL,RESOURCES,LIMITS,proposal['service_plan']['sha256'],{},False,
                apparent_entries=30000000,sample_seconds=15,cumulative_scratch=True)
    h.PROFILES[PROTOCOL]=p
    return p


def approved(proposal,approval):
    p=profile(proposal);h.validate(proposal,approval)
    if (proposal.get('recovery_protocol')!=PROTOCOL or proposal.get('node')!=NODE or
        proposal.get('previous_job_id')!=OLD_JOB or proposal.get('previous_elapsed_seconds')!=839 or
        proposal.get('next_requested_seconds')!=13560 or proposal.get('prior_scratch_bytes')!=PRIOR_BYTES or
        proposal.get('allocation_exception')!={'additional_allocations':1,'previous_maximum_allocations':1} or
        approval.get('additional_allocation_slot_approved') is not True or approval.get('writable_A_resume_approved') is not True or
        approval.get('resumed_A_not_two_fresh_states_accepted') is not True):
        raise h.Closed('Exact extra-slot and writable-A recovery approval required')
    for ref in proposal['old_state'].values():
        if isinstance(ref,dict):h.reference(ref)
    basis=h.read(h.reference(proposal['source_basis']))
    for ref in basis['references']:h.reference(ref)
    if (proposal.get('frontend_database')!='released_fresh_schema_reconstruction' or
        proposal.get('original_state_equivalence_verified') is not False or proposal.get('browse_equivalence_supported') is not False):
        raise h.Closed('Recovery source-equivalence disclosure changed')
    return p


def checked_context(path,inside=False):
    c=h.read(path);p=h.read(c['proposal']);approved(p,h.read(c['approval']))
    if (h.digest(p)!=c['proposal_sha256'] or time.time()>=c['deadline'] or
        c['job_id']!=os.environ.get('SLURM_JOB_ID') or REPO!=Path(p['frozen_source']) or os.uname().nodename!=NODE):
        raise h.Closed('Frozen approved node/clock/context changed')
    owner=_process(c['owner_pid'])
    if owner['start_ticks']!=c['owner_start_ticks'] or owner['state']=='Z':raise h.Closed('Recovery owner exited')
    pid=os.getppid();seen=set()
    while pid!=c['owner_pid'] and pid>1 and pid not in seen:
        seen.add(pid);pid=_process(pid)['ppid']
    if pid!=c['owner_pid']:raise h.Closed('Recovery worker escaped owner')
    if inside and (os.geteuid()!=0 or os.readlink('/proc/self/ns/net')==c['outer_net']):
        raise h.Closed('Private root-mapped network namespace required')
    return c,p


def baseline(raw):
    value=json.loads(raw)
    if (not isinstance(value,dict) or not value.get('schema') or not value.get('constraints') or
        not value.get('indexes') or not value.get('migrations') or
        value.get('counts')!={'current_nodes':0,'current_ways':0,'current_relations':0}):
        raise h.Closed('Complete native fresh-schema fixture required; no invented OSM objects')
    return value


def recovery_pass(rows,base_before,base_after):
    if len(rows)!=4 or base_before!=base_after:raise h.Closed('Immutable base changed or missing recovery phases')
    if [r.get('phase') for r in rows]!=['a-baseline','b-baseline','a-controls','b-controls']:
        raise h.Closed('Exact recovery phase sequence required')
    if any(r.get('database_baseline')!=rows[0]['database_baseline'] or
           r.get('schema_dump_sha256')!=rows[0].get('schema_dump_sha256') or
           r.get('database_marker_before')!=0 or r.get('closure',{}).get('remaining_owned_processes')!={} for r in rows):
        raise h.Closed('Post-migration full schema/fixture/reset/closure mismatch')
    baseline(json.dumps(rows[0]['database_baseline']))
    if not re.fullmatch('[0-9a-f]{64}',rows[0].get('schema_dump_sha256','')):raise h.Closed('Full schema dump fingerprint required')
    for row in rows[:2]:
        if row.get('native_controls_run') is not False or 'browser' in row or 'database_marker_after' in row:
            raise h.Closed('Baseline-only phase mutated native controls')
    for row in rows[2:]:
        old.controls_pass(row)
        if (row.get('native_controls_run') is not True or row.get('database_marker_after')!=1 or
            row.get('database_marker_data')!=['owned-reset-control'] or
            row['browser'].get('initial_cookie_count')!=0 or row['browser'].get('initial_origin_count')!=0 or
            not row['browser'].get('storage_state_sha256')):raise h.Closed('Native/browser/marker recovery control failed')
    if rows[2]['browser']['storage_state_sha256']==rows[3]['browser']['storage_state_sha256']:
        raise h.Closed('Browser session reset evidence missing')
    if any(r.get('fresh_claim') is not (r['phase'].startswith('b-')) for r in rows):
        raise h.Closed('Resumed A cannot claim fresh initialization')
    return dict(recovery_protocol=PROTOCOL,resumed_A_verified=True,fresh_B_from_original_base_verified=True,
                two_fresh_states_verified=False,two_cold_starts_verified=False,throughput_comparison=False,
                native_frontend_controls_verified=True,original_state_equivalence_verified=False,
                browse_equivalence_supported=False,validation_states_are_production_baselines=False,PRM_labels_collected=0)


def overlay_scripts(r,phase):
    """Readonly controller mounts; unique stale-PID evidence, fail on collision."""
    scripts={n:getattr(r,n.upper()+'_CTL') for n in ('tile','nom','db','web')}
    for name,text in list(scripts.items()):
        text=text.replace('set -x\n','set -ex\n',1)
        # Old scripts deliberately preserve stale PIDs. On restart preserve again
        # under a distinct phase suffix, with an explicit fail-closed collision.
        pattern=r'if \[ -e ([^ ]+/postmaster.pid) \]; then test ! -e \1.webprm-preserved; mv -- \1 \1.webprm-preserved; fi'
        def replace(m):
            target=m[1];saved=target+'.webprm-preserved-'+phase
            return f'if [ -e {target} ]; then if [ -e {saved} ]; then echo stale-pid-preservation-collision >&2; exit 91; fi; cp -- {target} /webprm/pid-evidence/{name}-postmaster.pid; mv -- {target} {saved}; fi'
        text=re.sub(pattern,replace,text)
        # PG11's native entrypoint also requires safe stale-PID handling.
        if name=='db':
            target='/var/lib/postgresql/data/postmaster.pid';saved=target+'.webprm-preserved-'+phase
            text=text.replace('su -s /bin/bash postgres',f'if [ -e {target} ]; then if [ -e {saved} ]; then exit 91; fi; cp -- {target} /webprm/pid-evidence/db-postmaster.pid; mv -- {target} {saved}; fi\nsu -s /bin/bash postgres',1)
        # Record migrations outside the original retained image. No original
        # migrate/master/log file is overwritten.
        text=text.replace('/tmp/migrate.log','/webprm/migrate.log')
        text=text.replace('/tmp/warm.log','/webprm/warm.log')
        scripts[name]=text
    return scripts


def inside(path):
    c,p=checked_context(path,inside=True);r=old.recipe();state=Path(c['state']);out=Path(c['output']);out.mkdir(mode=0o700)
    logs=Path(c['phase_scratch']);logs.mkdir(mode=0o700)
    r.W=state
    old.command(['/usr/sbin/ip','link','set','lo','up'],out/'loopback.log',c['deadline'])
    hosts=logs/'hosts';hosts.write_text(private_hosts(NODE));hosts.chmod(0o400)
    scripts=overlay_scripts(r,c['job_id']+'-'+c['phase'])
    for name in ('tile','nominatim','db','web','osrm'):
        overlay=logs/name;overlay.mkdir(mode=0o700);(overlay/'pid-evidence').mkdir(mode=0o700)
        script=scripts.get({'nominatim':'nom'}.get(name,name))
        if script:(overlay/'ctl.sh').write_text(script)
    rf=state/'rootfs';d=state/'data'
    # DB log/cache/run files remain private to the resumed/fresh state; all new
    # orchestration logs are outside old state. Hosts/controllers are read only.
    def prefix(name,binds=(),**kw):
        ctl={'nominatim':'nom'}.get(name,name)
        # /webprm and /etc/hosts already exist in the original sandboxes.
        # A per-service overlay replaces only that operational mountpoint.
        if not (rf/name/'webprm').is_dir() or not (rf/name/'etc/hosts').is_file():
            raise h.Closed('Existing bind destination required; no old rootfs mkdir')
        extra=[(logs/name,'/webprm'),(hosts,'/etc/hosts:ro')]
        if ctl in scripts:extra.append((logs/name/'ctl.sh','/webprm/ctl.sh:ro'))
        return r.prefix(rf/name,[*binds,*extra],**kw)
    jobs={'tile':prefix('tile',[(d/'tile','/data/database')])+['/bin/bash','/webprm/ctl.sh'],
          'nominatim':prefix('nominatim',[(d/'nominatim','/var/lib/postgresql/14/main')])+['/bin/bash','/webprm/ctl.sh'],
          'db':prefix('db')+['/bin/bash','/webprm/ctl.sh'],
          'web':prefix('web',fakeroot=False)+['/bin/bash','/webprm/ctl.sh']}
    for mode,port in (('car',5000),('bike',5001),('foot',5002)):
        jobs['osrm-'+mode]=prefix('osrm',[(d/'osrm'/mode,'/data')],fakeroot=False,writable=False)+['osrm-routed','--algorithm','mld','--threads','2','-p',str(port),'/data/us-northeast-latest.osrm']
    master=logs/'master.sh';master.write_text('#!/bin/bash\n'+''.join(shlex.join(argv)+' > '+shlex.quote(str(logs/name)+'.log')+' 2>&1 &\n' for name,argv in jobs.items())+'wait\n')
    previous=_subreaper(True);process_baseline=set(_processes());proc=None;tree=None;result=None;start=time.time()
    try:
        proc=multiprocessing.get_context('fork').Process(target=old.common.service_child,args=(['/bin/bash',str(master)],str(out/'services.log')))
        try:proc.start()
        finally:
            if proc.pid is not None:tree=OwnedEpisodeTree(proc,process_baseline)
        stop=min(c['deadline'],time.time()+1200)
        while time.time()<stop:
            tree.refresh()
            if not proc.is_alive():raise h.Closed('Recovery service owner exited')
            if r.get(r.ORIGIN+'/',20)[0]==200 and r.get('http://127.0.0.1:8085/status.php',10)[0]==200:break
            time.sleep(5)
        else:raise h.Closed('Recovery native service startup failed')
        sql=prefix('db')+['psql','-h','127.0.0.1','-U','openstreetmap','-d','openstreetmap','-At','-v','ON_ERROR_STOP=1']
        before=baseline(old.command(sql,out/'database-baseline.json',c['deadline'],input_bytes=SCHEMA_SQL,separate_stderr=True))
        schema=old.command(prefix('db')+['pg_dump','-h','127.0.0.1','-U','openstreetmap','-d','openstreetmap','--schema-only','--no-owner','--no-privileges'],out/'full-schema.sql',c['deadline'],separate_stderr=True)
        schema_sha=hashlib.sha256(schema).hexdigest()
        marker=int(old.command(sql,out/'marker-before.txt',c['deadline'],input_bytes=MARKER_SQL,separate_stderr=True).strip())
        if (marker or (c.get('expected_baseline') is not None and before!=c['expected_baseline']) or
            (c.get('expected_schema_sha256') is not None and schema_sha!=c['expected_schema_sha256'])):raise h.Closed('Full baseline/schema/marker mismatch before native controls')
        result=dict(phase=c['phase'],fresh_claim=c['phase'].startswith('b-'),database_baseline=before,schema_dump_sha256=schema_sha,database_marker_before=marker,native_controls_run=c['controls'])
        if c['controls']:
            r.backends(result);r.browser(result,storage_output=str(out/'storage-state-private.json'));old.controls_pass(result)
            old.command(sql,out/'database-mutate.txt',c['deadline'],input_bytes=b"CREATE TABLE webprm_reset_marker (marker text NOT NULL); INSERT INTO webprm_reset_marker VALUES ('owned-reset-control');\n",separate_stderr=True)
            result['database_marker_after']=int(old.command(sql,out/'marker-after.txt',c['deadline'],input_bytes=MARKER_SQL,separate_stderr=True).strip())
            result['database_marker_data']=old.command(sql,out/'marker-data.txt',c['deadline'],input_bytes=b'SELECT marker FROM webprm_reset_marker ORDER BY marker;\n',separate_stderr=True).decode().splitlines()
    finally:
        try:
            closure=tree.stop(30) if tree else dict(remaining_owned_processes={},child_exitcode=None)
            h.put(out/'service-closure.json',closure)
        finally:_subreaper(previous)
        # Preserve service/migration evidence after descendants stop; no log
        # truncation hides corruption. The worker owns bounded output storage.
        diagnostic={}
        for log in logs.rglob('*.log'):
            data=log.read_bytes();diagnostic[log.name]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
            if re.search(rb'PANIC:|invalid checkpoint record|could not locate a valid checkpoint|database files are incompatible|database system was interrupted.*corrupt',data,re.I):
                diagnostic['corruption_detected']=True
        h.put(out/'service-log-identities.json',diagnostic)
    if result is None or closure['remaining_owned_processes'] or diagnostic.get('corruption_detected'):
        raise h.Closed('Recovery phase incomplete, unclosed or corrupt')
    result.update(closure=closure,elapsed_seconds=time.time()-start);h.put(out/'result.json',result)


def old_lineage(p):
    refs=p['old_state'];context=h.read(h.reference(refs['context']));closure=h.read(h.reference(refs['closure']));storage=h.read(h.reference(refs['storage']))
    if (context['job_id']!=OLD_JOB or context['state']!=refs['resumed_A'] or context['scratch']!=refs['root'] or
        closure.get('remaining_owned_processes')!={} or storage.get('scratch_bytes')!=PRIOR_BYTES):raise h.Closed('Closed old state identity changed')
    oldattempt=Path(context['attempt'])
    # Node-local scratch can disappear at allocation closure. Fail here before
    # source readback, copying, or opening A for writable service recovery.
    for name in ('root','base','resumed_A'):
        path=Path(refs[name])
        if path.is_symlink() or not path.is_dir():raise h.Closed('Retained Map '+name+' unavailable; no fresh-restart fallback')
    for state in (Path(refs['base']),Path(refs['resumed_A'])):
        if any(not (state/'rootfs'/name/'webprm').is_dir() for name in ('tile','nominatim','db','web','osrm')):
            raise h.Closed('Retained Map rootfs layout incomplete; no fallback')
    if any((oldattempt/'instance-a'/name).exists() for name in ('database-before.txt','database-mutate.txt','marker-before.txt','result.json')):
        raise h.Closed('Old attempt reached baseline/native mutation; resume protocol inapplicable')
    for pid,identity in _processes().items():
        # All old namespace/service descendants were closed. Detect surviving
        # same-user commands referencing retained state before any writable use.
        try:
            proc=Path(f'/proc/{pid}')
            if proc.stat().st_uid!=os.getuid():continue
            cmd=(proc/'cmdline').read_bytes()
            if refs['root'].encode() in cmd:raise h.Closed('Retained Map state still has a live writer/reference')
            links=[proc/'cwd',proc/'root',*list((proc/'fd').iterdir())]
            if any(os.readlink(link).startswith(refs['root']+'/') for link in links if link.is_symlink()):
                raise h.Closed('Retained Map state still has an open handle')
        except (FileNotFoundError,ProcessLookupError):continue
        except PermissionError as error:raise h.Closed('Cannot exclude retained-state writer') from error
    return context


def worker(path):
    os.setsid();c,p=checked_context(path);scratch=Path(c['scratch']);attempt=Path(c['attempt']);deadline=c['deadline']
    os.environ.update(h.safe_env(scratch));os.umask(0o077)
    os.environ['PYTHONPATH']=p['nltk_pythonpath']+':'+p['frozen_source'];os.environ['NLTK_DATA']=p['nltk_data']
    oldcontext=old_lineage(p);refs=p['old_state'];base=Path(refs['base']);state_a=Path(refs['resumed_A'])
    for row in p['inputs']:
        if old.input_sha(row['path'],deadline)!=row['sha256']:raise h.Closed('Canonical acquired Map input changed')
    before=tree_fingerprint(base,deadline);a_before=tree_fingerprint(state_a,deadline)
    old_live=h.apparent(refs['root'],maximum_entries=30000000)
    charged=storage_charge(PRIOR_BYTES,old_live,h.apparent(scratch,maximum_entries=30000000))
    if charged['total_charged_bytes']+before['apparent_bytes']+LIMITS['runtime_reserve_bytes']>LIMITS['scratch_bytes']:
        raise h.Closed('Measured resumed-A/fresh-B admission exceeds unchanged cumulative cap')
    if shutil.disk_usage(scratch).free<before['apparent_bytes']+LIMITS['runtime_reserve_bytes']:raise h.Closed('Insufficient measured free scratch')
    h.put(attempt/'pre-mutation-fingerprints.json',dict(base=before,A=a_before,storage=charged,base_identity=oldcontext['base_identity']))
    # Preserve original operational config bytes, and identify the complete A
    # data directory before normal WAL recovery; original failure logs untouched.
    configs={}
    for file in sorted((state_a/'rootfs').glob('*/webprm/ctl.sh'))+sorted((state_a/'rootfs').glob('*/etc/hosts')):
        if file.is_symlink():raise h.Closed('Unexpected original config symlink')
        configs[str(file)]=dict(sha256=h.sha(file),before=file.read_text())
    h.put(attempt/'original-config-before.json',configs)
    results=[];state_b=scratch/'state-b'
    for phase,state in (('a-baseline',state_a),('b-baseline',state_b),('a-controls',state_a),('b-controls',state_b)):
        if phase=='b-baseline':
            old.command(['unshare','--user','--map-root-user','cp','-a','--reflink=auto','--',str(base),str(state_b)],attempt/'copy-b.log',deadline,timeout=1800)
            copied=tree_fingerprint(state_b,deadline)
            if any(copied[key]!=before[key] for key in ('sha256','entries')):raise h.Closed('Independent B copy differs from immutable base')
        expected=results[0]['database_baseline'] if results else None
        child=dict(c,state=str(state),output=str(attempt/phase),phase_scratch=str(scratch/phase),phase=phase,controls=phase.endswith('controls'),
                   expected_baseline=expected,expected_schema_sha256=results[0]['schema_dump_sha256'] if results else None,
                   owner_pid=os.getpid(),owner_start_ticks=_process(os.getpid())['start_ticks'],outer_net=os.readlink('/proc/self/ns/net'))
        childpath=attempt/(phase+'-context.json');h.put(childpath,child)
        old.command(['unshare','--user','--map-root-user','--net',old.common.PYTHON,'-B',__file__,'--inside',str(childpath)],attempt/(phase+'-namespace.log'),deadline,timeout=3600)
        results.append(h.read(Path(child['output'])/'result.json'))
    after=tree_fingerprint(base,deadline);gate=recovery_pass(results,before,after)
    # Original host/config/controller bytes must survive all readonly overlays.
    if any(h.sha(Path(name))!=row['sha256'] for name,row in configs.items()):raise h.Closed('Original retained config overwritten')
    h.put(attempt/'post-mutation-fingerprints.json',dict(base=after,A=tree_fingerprint(state_a,deadline)))
    h.put(attempt/'validation-result.json',dict(status='passed',gate=gate,phases=results,base_before=before,base_after=after,model_calls=0,api_usd=0,downloads=0))


def run(proposal_path,approval_path):
    p,a=h.read(proposal_path),h.read(approval_path);profile_=approved(p,a);root=Path(p['campaign_root']);rows=h.history(root,profile_);job=os.environ.get('SLURM_JOB_ID','')
    if (not rows or rows[-1]['job_id']!=job or rows[-1]['proposal_sha256']!=h.digest(p) or REPO!=Path(p['frozen_source']) or
        '/job_'+job+'/' not in Path('/proc/self/cgroup').read_text() or os.uname().nodename!=NODE):raise h.Closed('Registered current native g007 recovery allocation required')
    budget=h.prior_budget(root,rows[:-1],old.common.terminal_rows(rows[:-1]),profile_)
    if budget!={'prior_seconds':839,'remaining_seconds':13560,'prior_scratch_bytes':PRIOR_BYTES} or rows[-1]['requested_seconds']!=13560:
        raise h.Closed('Recovery may not regain time or scratch budget')
    raw=subprocess.check_output(['scontrol','--json','show','job',job],text=True,timeout=15)
    clock=h.scheduler(raw,job,requested_seconds=13560,profile=profile_)
    if json.loads(raw)['jobs'][0]['nodes']!=NODE:raise h.Closed('Retained node must be g007')
    attempt=root/('attempt-'+job);attempt.mkdir(mode=0o700)
    scratch=Path('/tmp')/('maprecovery-'+job+'-'+secrets.token_hex(4));scratch.mkdir(mode=0o700)
    h.put(attempt/'scheduler.json',dict(raw=raw))
    c=dict(proposal=str(proposal_path),approval=str(approval_path),proposal_sha256=h.digest(p),job_id=job,owner_pid=os.getpid(),
           owner_start_ticks=_process(os.getpid())['start_ticks'],deadline=clock['work_deadline'],scratch=str(scratch),attempt=str(attempt))
    path=attempt/'context.json';h.put(path,c)
    with os.fdopen(os.open(root/'map-recovery-exclusive.lock',os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600),'r+b') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        # The NEW frozen h.own supports old-root growth charging. No old frozen
        # owner or accounting receipt is changed.
        h.own(worker,(path,),attempt=attempt,scratch=scratch,campaign_root=root,deadline=clock['work_deadline'],profile=profile_,
              prior_scratch_bytes=budget['prior_scratch_bytes'],prior_scratch_paths=(p['old_state']['root'],))
    result=h.read(attempt/'validation-result.json')
    if result.get('status')!='passed':raise h.Closed('Recovery proof incomplete')
    h.put(root/'completed-recovery.json',dict(status='hosting_recovery_passed_pending_independent_audit',job_id=job,validation=old.common.ref(attempt/'validation-result.json'),
            protocol=PROTOCOL,two_fresh_states_verified=False,original_state_equivalence_verified=False,live_adapter_enabled=False))


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
