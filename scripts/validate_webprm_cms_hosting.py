#!/usr/bin/env python3
"""Prospective CMS hosting owner. Prepare/preview is inert; run needs approval.

No scheduler submission or release is performed by this script. A held job must
be registered durably before the resource owner releases it. The actual image
is never materialized by preparation, tests, or the default preview command.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import resource
import secrets
import shutil
import shlex
import signal
import subprocess
import sys
import threading
import time

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from openwebrl import webprm_cms_hosting as h
from openwebrl import webprm_cms_materializer as materializer
from openwebrl.webprm_cms_native import browser_control, reset_gate, validate_product
from openwebrl.webprm_lifecycle import OwnedEpisodeTree,_process,_processes,_subreaper
from openwebrl.webprm_webarena import verify_source

R=Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011')
N=R/'next-stage-preparation'
PLAN=N/'cms-thirdparty-startup-includes-v1/owned-services-plan.json'
ACQUIRED=N/'cms-thirdparty-oci-acquisition-v1'
NATIVE=Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-baselines-20261009/sources/arbiter')
PYTHON='/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-baselines-20261009/browser-env/bin/python'
CHROMIUM='/gpfs/scrubbed/zixianma/openwebrl-runtime/browsers/chromium-1208/chrome-linux64/chrome'
SOURCES=('scripts/validate_webprm_cms_hosting.py','scripts/submit_webprm_cms_hosting.py','openwebrl/webprm_cms_hosting.py',
    'openwebrl/webprm_cms_materializer.py','openwebrl/webprm_cms_native.py','openwebrl/webprm_lifecycle.py',
    'openwebrl/jev_eval.py','openwebrl/webprm_source.py','openwebrl/webprm_webarena.py','openwebrl/__init__.py',
    'tests/test_webprm_cms_hosting.py','tests/test_webprm_cms_materializer.py','tests/test_webprm_cms_native.py',
    'tests/test_validate_webprm_cms_hosting.py','tests/test_submit_webprm_cms_hosting.py')
BOOTSTRAP='''#!/bin/sh
set -eu
test -f /run/mysqld/.init
test -f /run/postgresql/.init
test -d /var/lib/mysql/magentodb
test -f /var/lib/mysql/ibdata1
test "$(cat /var/lib/mysql/mysql_upgrade_info)" = "10.6.12-MariaDB"
mkdir -p /run/nginx /run/mysqld /run/webprm-cms /var/tmp/nginx/client_body /tmp/webprm-elasticsearch /usr/share/java/elasticsearch/logs
chown nginx:nginx /run/nginx /var/tmp/nginx /var/tmp/nginx/client_body
chown -R mysql:mysql /run/mysqld /var/lib/mysql
chown -R elastico:elastico /usr/share/java/elasticsearch/data /usr/share/java/elasticsearch/logs /tmp/webprm-elasticsearch
exec supervisord -c /opt/webprm-cms/supervisord.conf -n -j /run/webprm-cms/supervisord.pid
'''


def ref(path):
    return dict(path=str(path),sha256=h.sha(path))


LOG_BYTES=32*2**20


def bytes_new(path,raw,mode=0o600):
    path=Path(path);path.parent.mkdir(parents=True,mode=0o700,exist_ok=True)
    with os.fdopen(os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,mode),'wb') as out:
        out.write(raw);out.flush();os.fchmod(out.fileno(),mode);os.fsync(out.fileno())


def command(argv,output,deadline,*,input_bytes=None,timeout=300,separate_stderr=False):
    """All descendants also belong to the outer subreaper; bounded private log."""
    materializer.check(deadline)
    if not argv or any(x=='rm' for x in argv):
        raise h.Closed('Forbidden command')
    # Cap the private logs, not the child: an RLIMIT_FSIZE here is inherited by
    # cp/services and kills them (SIGXFSZ) on any file over the cap (job 356545).
    def drain(stream,path):
        with os.fdopen(os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),'wb') as log:
            kept=0
            for chunk in iter(lambda:stream.read(65536),b''):
                if kept<LOG_BYTES:log.write(chunk[:LOG_BYTES-kept]);kept+=len(chunk[:LOG_BYTES-kept])
    process=subprocess.Popen(argv,stdin=subprocess.PIPE if input_bytes is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,stderr=subprocess.PIPE if separate_stderr else subprocess.STDOUT)
    readers=[threading.Thread(target=drain,args=(process.stdout,output),daemon=True)]
    if separate_stderr:readers.append(threading.Thread(target=drain,args=(process.stderr,str(output)+'.stderr'),daemon=True))
    for reader in readers:reader.start()
    try:
        if input_bytes is not None:
            process.stdin.write(input_bytes);process.stdin.close()
        returncode=process.wait(timeout=min(timeout,max(.1,deadline-time.time())))
    except BaseException:
        process.kill();process.wait();raise
    finally:
        for reader in readers:reader.join(timeout=30)
    if returncode:
        raise h.Closed('Owned command failed; private output retained: '+Path(output).name)
    return Path(output).read_bytes()


def terminal_rows(rows):
    if not rows:return {}
    raw=subprocess.check_output(['sacct','-X','-j',','.join(r['job_id'] for r in rows),'--noheader','--parsable2',
        '--format=JobID,State,ElapsedRaw'],text=True,timeout=15)
    found={}
    for line in raw.strip().splitlines():
        bits=line.split('|')
        if len(bits)<3 or bits[0] in found or not bits[2].isdigit():
            raise h.Closed('Ambiguous terminal accounting')
        found[bits[0]]=dict(state=bits[1].split()[0],elapsed_seconds=int(bits[2]))
    return found


def validate_context(context_path,*,inside=False):
    context=h.read(context_path);proposal=h.read(context['proposal_path']);approval=h.read(context['approval_path'])
    plan=h.validate(proposal,approval)
    if (context['proposal_sha256']!=h.digest(proposal) or context['approval_sha256']!=h.sha(context['approval_path']) or
            context['job_id']!=os.environ.get('SLURM_JOB_ID') or time.time()>=context['deadline'] or
            context['source_script_sha256']!=h.sha(__file__) or
            Path(__file__).resolve().parents[1]!=Path(proposal['frozen_source'])):
        raise h.Closed('Frozen approved context mismatch')
    owner=_process(context['owner_pid'])
    if owner['start_ticks']!=context['owner_start_ticks'] or owner['state']=='Z':
        raise h.Closed('Owning process is stale')
    pid=os.getppid();seen=set()
    while pid!=context['owner_pid'] and pid>1 and pid not in seen:
        seen.add(pid);pid=_process(pid)['ppid']
    if pid!=context['owner_pid']:
        raise h.Closed('Worker is not an approved owner descendant')
    if inside and (os.geteuid()!=0 or os.readlink('/proc/self/ns/net')==context['outer_network_namespace']):
        raise h.Closed('CMS services require a fresh owned rootmapped network namespace')
    return context,proposal,plan


MOUNT_POINTS=('/opt/native-fakeroot','/opt/webprm-cms')


def prepare_controls(state,folder,plan):
    folder.mkdir(mode=0o700)
    changes=[]
    for target,reference in plan['container_file_install_map'].items():
        source=h.reference(reference)
        if target.startswith('/opt/webprm-cms/'):
            destination=folder/target.removeprefix('/opt/webprm-cms/')
        elif target=='/etc/nginx/conf.d/default.conf':
            destination=h.private_path(state/target.lstrip('/'))
            original=destination.read_bytes()
            saved=destination.with_name(destination.name+'.preserved-webprm')
            if saved.exists():raise h.Closed('Runtime state was already prepared')
            destination.rename(saved)
            changes.append(dict(path=target,original_sha256=hashlib.sha256(original).hexdigest(),preserved_path=str(saved),replacement_sha256=reference['sha256']))
        else:raise h.Closed('Unexpected prospective runtime adaptation')
        bytes_new(destination,source.read_bytes(),0o644)
    bytes_new(folder/'bootstrap.sh',BOOTSTRAP.encode(),0o644)
    # --writable sandboxes cannot create bind destinations (job 357128); add the two
    # empty mount points to this private state copy only, never to the base.
    for target in MOUNT_POINTS:
        parent=state
        for part in Path(target).relative_to('/').parts:
            parent=parent/part
            if parent.is_symlink() or (parent.exists() and not parent.is_dir()):
                raise h.Closed('Bind mount point is not a plain directory: '+target)
            if not parent.exists():
                parent.mkdir(mode=0o755);changes.append(dict(path=str(parent.relative_to(state)),created_empty_mount_point=True))
    h.put(folder/'runtime-adaptations.json',dict(changes=changes,entrypoint_invoked=False,database_initialized=False,
        prospective_service_adaptations=plan['runtime_adaptations']))


def image_env(proposal):
    """The released image's config Env (PATH incl. elasticsearch/bin, ES_HOME, LD_PRELOAD iconv),
    which Docker applies to every process; --cleanenv drops it (job 357728/probe 357750)."""
    manifest=h.read(h.reference(proposal['manifest']));digest=manifest['config']['digest']
    if not digest.startswith('sha256:'):raise h.Closed('Image config digest required')
    raw=(Path(proposal['blob_root'])/digest[7:]).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=digest[7:]:raise h.Closed('Image config blob changed')
    env=json.loads(raw)['config'].get('Env') or []
    if not all(isinstance(e,str) and '=' in e and not e.startswith('=') and '\0' not in e for e in env):
        raise h.Closed('Unbounded image Env entry')
    return env


def container_prefix(context,proposal):
    addon=Path(proposal['fakeroot_addon'])
    for name,expected in proposal['fakeroot_source_sha256'].items():
        if h.sha(addon/name)!=expected:raise h.Closed('Native fakeroot addon changed')
    return ['apptainer','exec','--cleanenv','--no-eval','--fakeroot','--ignore-fakeroot-command',
        '--writable','--containall','--no-home','--no-mount','bind-paths,home,cwd,hostfs,tmp',
        '--bind',str(addon)+':/opt/native-fakeroot:ro','--bind',context['controls']+':/opt/webprm-cms:ro',
        context['state'],'env',*image_env(proposal),'FAKEROOTDONTTRYCHOWN=1','/opt/native-fakeroot/usr/bin/fakeroot',
        '-l','/opt/native-fakeroot/usr/lib/libfakeroot.so','-f','/opt/native-fakeroot/usr/bin/faked','--']


def service_child(argv,log):
    os.setsid()
    with os.fdopen(os.open(log,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),'wb') as stream:
        os.dup2(stream.fileno(),1);os.dup2(stream.fileno(),2)
        os.execvpe(argv[0],argv,os.environ)


def private_db_options(values):
    if not {'username','password','dbname','host'}<=values.keys() or values['dbname']!='magentodb' or values['host']!='127.0.0.1':
        raise h.Closed('Imported private database endpoint differs')
    def escape(value):
        if not isinstance(value,str) or any(c in value for c in '\r\n\0') or len(value)>512:
            raise h.Closed('Unbounded database credential')
        return '"'+value.replace('\\','\\\\').replace('"','\\"')+'"'
    return ('[client]\nuser='+escape(values['username'])+'\npassword='+escape(values['password'])+
            '\nhost=127.0.0.1\nport=3306\nprotocol=tcp\n').encode()


def inside(context_path):
    context,proposal,plan=validate_context(context_path,inside=True)
    output=Path(context['instance_output']);output.mkdir(mode=0o700,exist_ok=False)
    deadline=context['deadline'];prefix=container_prefix(context,proposal)
    command(['/usr/sbin/ip','link','set','lo','up'],output/'loopback.log',deadline)
    command(prefix+['/bin/sh','-c',"test \"$(id -u)\" = 0; su elastico -c 'test \"$(id -u)\" != 0'"],output/'uid-probe.log',deadline)
    # Import only the hash-reviewed native config array. Output stays private.
    raw=command(prefix+['php','-r',"$a=require '/var/www/magento2/app/etc/env.php'; echo json_encode($a['db']['connection']['default']);"],output/'private-db-discovery.json',deadline,separate_stderr=True)
    db=json.loads(raw);bytes_new(Path(context['controls'])/'private-db.cnf',private_db_options(db))
    mysql=prefix+['mysql','--defaults-extra-file=/opt/webprm-cms/private-db.cnf','--batch','--raw','--skip-column-names','magentodb']
    previous=_subreaper(True);baseline=set(_processes());process=tree=None;phase_result=None
    try:
        process=multiprocessing.get_context('fork').Process(target=service_child,
            args=(prefix+['/bin/sh','/opt/webprm-cms/bootstrap.sh'],str(output/'services.log')))
        try:process.start()
        finally:
            if process.pid is not None:tree=OwnedEpisodeTree(process,baseline)
        for attempt in range(120):
            tree.refresh();materializer.check(deadline)
            if not process.is_alive():raise h.Closed('Foreground CMS services exited during startup')
            try:
                command(mysql,output/f'db-ready-{attempt:03d}.log',deadline,input_bytes=b'SELECT 1;\n',timeout=10)
                break
            except h.Closed:
                time.sleep(2)
        else:raise h.Closed('Imported database did not become ready')
        before_sql=b"SELECT JSON_OBJECT('path',path,'value',value) FROM core_config_data WHERE path IN ('web/unsecure/base_url','web/secure/base_url','admin/security/password_is_forced','admin/security/password_lifetime') ORDER BY path;\n"
        command(mysql,output/'origin-before.jsonl',deadline,input_bytes=before_sql)
        magento=prefix+['php','/var/www/magento2/bin/magento']
        for i,argv in enumerate((['config:set','admin/security/password_is_forced','0'],
                                ['config:set','admin/security/password_lifetime','0'],
                                ['setup:store-config:set','--base-url=http://127.0.0.1:7780'])):
            command(magento+argv,output/f'origin-php-{i}.log',deadline,timeout=300)
        command(mysql,output/'origin-secure.log',deadline,input_bytes=b"UPDATE core_config_data SET value='http://127.0.0.1:7780/' WHERE path='web/secure/base_url';\n")
        command(magento+['cache:flush'],output/'cache-flush.log',deadline,timeout=300)
        command(mysql,output/'origin-after.jsonl',deadline,input_bytes=before_sql)
        command(prefix+['supervisorctl','-c','/opt/webprm-cms/supervisord.conf','start','cron'],output/'start-cron.log',deadline)
        query=b"SELECT JSON_OBJECT('id',p.entity_id,'name',v.value) FROM catalog_product_entity p JOIN catalog_product_entity_varchar v ON v.entity_id=p.entity_id JOIN eav_attribute a ON a.attribute_id=v.attribute_id WHERE a.attribute_code='name' AND v.store_id=0 ORDER BY p.entity_id ASC LIMIT 1;\n"
        product=validate_product(json.loads(command(mysql,output/'fixed-product.json',deadline,input_bytes=query,separate_stderr=True)))
        common=dict(native_source=proposal['native_source'],chromium=proposal['chromium'],account_binding=proposal['account_binding'],
            instance_id=context['instance_id'],base_sha256=context['base_sha256'],mutation_name=context['mutation_name'],product=product)
        if context['phase']=='first':
            base=browser_control(dict(common,phase='baseline',output=str(output/'native-baseline')))
            mutation=browser_control(dict(common,phase='mutate',output=str(output/'native-mutate')))
            observed=validate_product(json.loads(command(mysql,output/'mutated-product.json',deadline,input_bytes=query,separate_stderr=True)))
            if observed!=mutation['product']:raise h.Closed('Native UI mutation disagrees with persisted database')
            media=Path(context['state'])/'var/www/magento2/pub/media'/context['media_filename']
            bytes_new(h.private_path(media),context['mutation_name'].encode(),0o644)
            phase_result=dict(status='passed',baseline=base,mutation=mutation,media_ref=ref(media),instance_id=context['instance_id'])
        elif context['phase']=='second':
            earlier=h.read(h.reference(context['first_result']))
            if product!=earlier['baseline']['product']:raise h.Closed('Fresh imported DB does not restore product sentinel')
            reset=browser_control(dict(common,phase='reset',output=str(output/'native-reset')))
            media=Path(context['state'])/'var/www/magento2/pub/media'/context['media_filename']
            gate=reset_gate(earlier['baseline'],earlier['mutation'],reset,media_absent=not os.path.lexists(media))
            phase_result=dict(status='passed',reset=reset,gate=gate,instance_id=context['instance_id'])
        else:raise h.Closed('Unknown CMS fresh-state phase')
        command(prefix+['supervisorctl','-c','/opt/webprm-cms/supervisord.conf','status'],output/'service-status.log',deadline)
        h.put(output/'outcome-before-service-close.json',phase_result)
        command(prefix+['supervisorctl','-c','/opt/webprm-cms/supervisord.conf','shutdown'],output/'shutdown.log',deadline,timeout=30)
        process.join(timeout=30)
        if process.is_alive():raise h.Closed('CMS supervisor did not exit after shutdown')
    finally:
        try:
            closure=tree.stop(10) if tree else dict(remaining_owned_processes={},child_exitcode=None)
            h.put(output/'service-closure.json',closure)
        finally:_subreaper(previous)
    if closure['remaining_owned_processes'] or closure['child_exitcode']!=0 or phase_result is None:
        raise h.Closed('CMS namespace did not close completely')
    h.put(output/'result.json',phase_result)


def worker(context_path):
    os.setsid()
    context,proposal,plan=validate_context(context_path)
    scratch=Path(context['scratch']);os.environ.update(h.safe_env(scratch));os.umask(0o077)
    os.environ['PYTHONPATH']=proposal['nltk_pythonpath']+':'+proposal['frozen_source']
    os.environ['NLTK_DATA']=proposal['nltk_data']
    attempt=Path(context['attempt']);deadline=context['deadline']
    proof=h.read(h.reference(proposal['layer_proof']));manifest=h.read(h.reference(proposal['manifest']))
    raw={r['compressed_sha256']:r for r in proof['unique_layers']}
    layers=[dict(path=str(Path(proposal['blob_root'])/l['digest'][7:]),compressed_bytes=l['size'],**raw[l['digest'][7:]]) for l in manifest['layers']]
    base=scratch/'base'
    receipt=materializer.materialize(layers,base,attempt/'materialization',min(deadline,time.time()+1800))
    base_bytes=h.apparent(base)
    if 3*base_bytes+context['prior_scratch_bytes']+h.LIMITS['runtime_reserve_bytes']>h.LIMITS['scratch_bytes']:
        raise h.Closed('Full base plus two private states fails storage admission')
    base_sha=receipt['manifest']['sha256'];first=None
    for phase,label in (('first','a'),('second','b')):
        state=scratch/('state-'+label);controls=attempt/('controls-'+label)
        # Copy as the namespace's mapped root: image files such as mode-0 run/crond.reboot
        # are unreadable to the plain host user (job 356561) but not to the services' root.
        command(['unshare','--user','--map-root-user','cp','-a','--reflink=auto','--',str(base),str(state)],
            attempt/('copy-'+label+'.log'),deadline,timeout=600)
        prepare_controls(state,controls,plan)
        child=dict(context,state=str(state),controls=str(controls),phase=phase,instance_id=context['job_id']+'-'+label,
            instance_output=str(attempt/('instance-'+label)),base_sha256=base_sha,
            owner_pid=os.getpid(),owner_start_ticks=_process(os.getpid())['start_ticks'],
            outer_network_namespace=os.readlink('/proc/self/ns/net'))
        if first:child['first_result']=first
        child_path=attempt/('context-'+label+'.json');h.put(child_path,child)
        command(['unshare','--user','--map-root-user','--net',PYTHON,'-B',str(Path(__file__).resolve()),
            '--inside',str(child_path)],attempt/('namespace-'+label+'.log'),deadline,timeout=1500)
        folder=Path(child['instance_output'])
        if h.read(folder/'service-closure.json')['remaining_owned_processes']!={}:
            raise h.Closed('Previous service state still has owned processes')
        if label=='a':first=ref(folder/'result.json')
    result=h.read(attempt/'instance-b/result.json')
    if result.get('gate',{}).get('fresh_reset_verified') is not True:
        raise h.Closed('Native CMS fresh-reset gate missing')
    h.put(attempt/'validation-result.json',dict(status='passed',first=first,second=ref(attempt/'instance-b/result.json'),
        materialization=receipt,original_state_equivalence_verified=False,source='third_party_digest_pinned_oci_candidate',
        live_adapter_enabled=False,model_calls=0,api_usd=0,downloads=0))


def run(proposal_path,approval_path):
    proposal,approval=h.read(proposal_path),h.read(approval_path);h.validate(proposal,approval)
    root=h.private_path(proposal['campaign_root']);rows=h.history(root);job=os.environ.get('SLURM_JOB_ID','')
    if (not job.isdigit() or not rows or rows[-1]['job_id']!=job or
            rows[-1]['proposal_sha256']!=h.digest(proposal) or '/job_'+job+'/' not in Path('/proc/self/cgroup').read_text() or
            Path(__file__).resolve().parents[1]!=Path(proposal['frozen_source'])):
        raise h.Closed('Durably registered current frozen allocation required')
    budget=h.prior_budget(root,rows[:-1],terminal_rows(rows[:-1]))
    if rows[-1]['requested_seconds']!=budget['remaining_seconds']:raise h.Closed('Prior allocation clock changed')
    raw=subprocess.check_output(['scontrol','--json','show','job',job],text=True,timeout=15)
    clock=h.scheduler(raw,job,requested_seconds=budget['remaining_seconds'])
    attempt=root/('attempt-'+job);attempt.mkdir(mode=0o700,exist_ok=False)
    h.put(attempt/'scheduler.json',dict(raw=raw,captured_epoch=time.time()))
    scratch=h.private_path(Path('/tmp')/('cmsv-'+job+'-'+secrets.token_hex(3)));scratch.mkdir(mode=0o700)
    worker_entered=False
    try:
        if shutil.disk_usage(scratch).free<h.LIMITS['scratch_bytes']-budget['prior_scratch_bytes']:
            raise h.Closed('Insufficient private scratch capacity')
        context=dict(proposal_path=str(proposal_path),approval_path=str(approval_path),proposal_sha256=h.digest(proposal),
            approval_sha256=h.sha(approval_path),job_id=job,owner_pid=os.getpid(),owner_start_ticks=_process(os.getpid())['start_ticks'],
            outer_network_namespace=os.readlink('/proc/self/ns/net'),source_script_sha256=h.sha(__file__),
            deadline=clock['work_deadline'],attempt=str(attempt),scratch=str(scratch),prior_scratch_bytes=budget['prior_scratch_bytes'],
            mutation_name='OpenWebRL CMS reset '+secrets.token_hex(8),media_filename='webprm-reset-'+secrets.token_hex(8)+'.txt')
        path=attempt/'context.json';h.put(path,context)
        worker_entered=True
        h.own(worker,(path,),attempt=attempt,scratch=scratch,campaign_root=root,deadline=clock['work_deadline'],
              prior_scratch_bytes=budget['prior_scratch_bytes'])
    except BaseException as error:
        h.put(attempt/'failure.json',dict(error_type=type(error).__name__,worker_owner_entered=worker_entered,
            outcome='failed_preserving_all_state_and_evidence'))
        raise
    finally:
        if not worker_entered:
            h.put(attempt/'closure.json',dict(remaining_owned_processes={},child_exitcode=None,owned_pids=[],
                child_started=False,stage='before_worker_creation'))
            h.put(attempt/'storage-final.json',dict(scratch_bytes=h.apparent(scratch),retained_bytes=h.apparent(root)))
    if h.read(attempt/'validation-result.json').get('status')!='passed':raise h.Closed('CMS validation incomplete')
    h.put(root/'completed.json',dict(status='hosting_validation_passed_pending_independent_audit',job_id=job,
        validation=ref(attempt/'validation-result.json'),closure=ref(attempt/'closure.json'),
        original_state_equivalence_verified=False,live_adapter_enabled=False))


def prepare(root):
    root=h.private_path(root);root.mkdir(mode=0o700,exist_ok=False)
    if h.sha(PLAN)!=h.PLAN_SHA:raise h.Closed('CMS service plan changed')
    if not Path(PYTHON).is_file() or not Path(CHROMIUM).is_file():
        raise h.Closed('Preserved browser runtime is missing')
    frozen=root/'frozen-source';frozen.mkdir(mode=0o700)
    for name in SOURCES:bytes_new(frozen/name,(REPO/name).read_bytes())
    batch='\n'.join(['#!/bin/bash','#SBATCH --account=zixianma','#SBATCH --partition=gpu-h200',
        '#SBATCH --qos=normal','#SBATCH --nodes=1','#SBATCH --ntasks=1','#SBATCH --cpus-per-task=4',
        '#SBATCH --mem=16G','#SBATCH --time=120','#SBATCH --job-name=webprm-cms-validate',
        '#SBATCH --output='+str(root/'slurm-%j.log'),'set -euo pipefail',
        'export TMPDIR=/tmp PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 BASH_ENV=/dev/null ENV=/dev/null',
        'exec '+shlex.quote(PYTHON)+' -B '+shlex.quote(str(frozen/'scripts/validate_webprm_cms_hosting.py'))+
        ' --proposal '+shlex.quote(str(root/'proposal.json'))+' --approval "$1" --run',''])
    bytes_new(frozen/'hosting.sbatch',batch.encode())
    native=h.put(root/'native-source.json',verify_source(NATIVE))
    addon=R/'native-fakeroot-addon'
    addons={str(p.relative_to(addon)):h.sha(p) for p in addon.rglob('*') if p.is_file() and not p.is_symlink()}
    proposal=dict(schema_version=1,status='prepared_not_approved',protocol=h.PROTOCOL,campaign_root=str(root/'campaign'),
        frozen_source=str(frozen),source_sha256={name:h.sha(frozen/name) for name in SOURCES},
        resources=h.RESOURCES,limits=h.LIMITS,source='third_party_digest_pinned_oci_candidate',
        original_state_equivalence_verified=False,service_plan=ref(PLAN),
        batch_script=ref(frozen/'hosting.sbatch'),
        materialization_preflight=ref(N/'cms-thirdparty-materialization-preflight-v1/receipt.json'),
        acquisition_audit=ref(ACQUIRED/'independent-final-audit-356009.json'),
        layer_proof=ref(ACQUIRED/'startup-01/attempt-356009/identity-proof.json'),manifest=ref(ACQUIRED/'metadata/manifest.json'),
        blob_root=str(ACQUIRED/'startup-01/attempt-356009/oci/blobs/sha256'),native_source=str(NATIVE),
        native_source_receipt=native,account_binding=ref(N/'cms-native-accounts-v1/account-binding.json'),
        fakeroot_addon=str(addon),fakeroot_source_sha256=addons,python=PYTHON,chromium=CHROMIUM,
        nltk_pythonpath=str(R/'nltk-runtime/venv/lib/python3.12/site-packages'),nltk_data=str(R/'nltk-runtime/nltk_data'))
    h.put(root/'proposal.json',proposal)
    return dict(proposal_path=str(root/'proposal.json'),proposal_sha256=h.digest(proposal),
                status='offline_prepared_requires_review_and_exact_hosting_approval',allocations=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',type=Path);parser.add_argument('--proposal',type=Path);parser.add_argument('--approval',type=Path)
    parser.add_argument('--register-held');parser.add_argument('--run',action='store_true');parser.add_argument('--inside',type=Path)
    args=parser.parse_args()
    if args.prepare:result=prepare(args.prepare)
    elif args.inside:return inside(args.inside)
    elif args.run:return run(args.proposal,args.approval)
    elif args.register_held:
        proposal,approval=h.read(args.proposal),h.read(args.approval);h.validate(proposal,approval)
        rows=h.history(proposal['campaign_root']);raw=subprocess.check_output(['scontrol','--json','show','job',args.register_held],text=True,timeout=15)
        result=h.register(proposal,approval,args.register_held,raw,terminal_rows(rows))
    else:
        proposal=h.read(args.proposal);result=dict(status='unapproved_resource_preview',resources=proposal['resources'],limits=proposal['limits'],
            proposal_sha256=h.digest(proposal),allocations=0,source=proposal['source'])
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
