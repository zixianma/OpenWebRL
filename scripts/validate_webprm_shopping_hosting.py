#!/usr/bin/env python3
"""Prospective Shopping hosting owner. Prepare/preview is inert; run needs approval.

Same contract as scripts/validate_webprm_cms_hosting.py, whose primitives (bounded
commands, namespace copy, reviewed controls, hash-bound image Env, fakeroot prefix,
private DB options) are reused unchanged under the SHOPPING hosting profile:
materialize the pinned Shopping OCI layers once; state A: services, origin
127.0.0.1:7770, released customer login + /wishlist/ check, released start URLs,
native evaluator controls, released order helper, one persisted account mutation
plus a media sentinel; state B (fresh copy of the same base): the mutation and
sentinel are gone and the native controls pass again. No scheduler submission or
release is performed by this script (scripts/submit_webprm_cms_hosting.py does).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
from pathlib import Path
import random
import secrets
import shlex
import shutil
import string
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from openwebrl import webprm_cms_hosting as h
from openwebrl import webprm_cms_materializer as materializer
from openwebrl.webprm_shopping_native import ORIGIN, browser_control, customer_sql, reset_gate, released_accounts, validate_sentinel
from openwebrl.webprm_lifecycle import OwnedEpisodeTree,_process,_processes,_subreaper
from openwebrl.webprm_webarena import digest, verify_source
from scripts import validate_webprm_cms_hosting as cms
from scripts.validate_webprm_cms_hosting import (PYTHON, CHROMIUM, NATIVE, R, N, bytes_new, command, container_prefix,
    prepare_controls, private_db_options, ref, service_child, terminal_rows)

PROFILE=h.SHOPPING
PLAN=N/'shopping-hosting-plan-v1/owned-services-plan.json'
ACQUIRED=N/'shopping-thirdparty-oci-transfer-exception-v1'
PROBE=N/'shopping-bringup-probe-v3/probe-result.json'
# 2.25 GB manifest / ~56 GB apparent base (probe 358020: 2894 s, 85.8 GB peak RSS).
LIMITS=materializer.Limits(raw_bytes=80*2**30,final_bytes=80*2**30,members=30000000)
MATERIALIZE_SECONDS=5400
SOURCES=('scripts/validate_webprm_shopping_hosting.py','scripts/validate_webprm_cms_hosting.py','scripts/submit_webprm_cms_hosting.py',
    'openwebrl/webprm_cms_hosting.py','openwebrl/webprm_cms_materializer.py','openwebrl/webprm_cms_native.py',
    'openwebrl/webprm_shopping_native.py','openwebrl/webprm_webarena_helpers.py','openwebrl/webprm_lifecycle.py',
    'openwebrl/jev_eval.py','openwebrl/webprm_source.py','openwebrl/webprm_webarena.py','openwebrl/__init__.py',
    'tests/test_webprm_cms_hosting.py','tests/test_validate_webprm_shopping_hosting.py','tests/test_webprm_shopping_native.py')
ORIGIN_COMMANDS=(['config:set','admin/security/password_is_forced','0'],
                 ['config:set','admin/security/password_lifetime','0'],
                 ['setup:store-config:set','--base-url='+ORIGIN])
SECURE_SQL=b"UPDATE core_config_data SET value='"+ORIGIN.encode()+b"/' WHERE path='web/secure/base_url';\n"


def content_sha256(manifest):
    """Destination-independent identity of a materialization manifest (as webprm_cms_site)."""
    return digest({key:value for key,value in manifest.items() if key!='destination'})


def validate_context(context_path,*,inside=False):
    context=h.read(context_path);proposal=h.read(context['proposal_path']);approval=h.read(context['approval_path'])
    plan=h.validate(proposal,approval)
    if (h.profile_of(proposal) is not PROFILE or context['proposal_sha256']!=h.digest(proposal) or
            context['approval_sha256']!=h.sha(context['approval_path']) or
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
        raise h.Closed('Shopping services require a fresh owned rootmapped network namespace')
    return context,proposal,plan


def inside(context_path):
    context,proposal,plan=validate_context(context_path,inside=True)
    output=Path(context['instance_output']);output.mkdir(mode=0o700,exist_ok=False)
    deadline=context['deadline'];prefix=container_prefix(context,proposal)
    command(['/usr/sbin/ip','link','set','lo','up'],output/'loopback.log',deadline)
    command(prefix+['/bin/sh','-c',"test \"$(id -u)\" = 0; su elastico -c 'test \"$(id -u)\" != 0'"],output/'uid-probe.log',deadline)
    raw=command(prefix+['php','-r',"$a=require '/var/www/magento2/app/etc/env.php'; echo json_encode($a['db']['connection']['default']);"],output/'private-db-discovery.json',deadline,separate_stderr=True)
    db=json.loads(raw);bytes_new(Path(context['controls'])/'private-db.cnf',private_db_options(db))
    mysql=prefix+['mysql','--defaults-extra-file=/opt/webprm-cms/private-db.cnf','--batch','--raw','--skip-column-names','magentodb']
    account,_=released_accounts(proposal['native_source'],proposal['account_binding'])
    sentinel=customer_sql(account['username'])
    previous=_subreaper(True);baseline=set(_processes());process=tree=None;phase_result=None
    try:
        process=multiprocessing.get_context('fork').Process(target=service_child,
            args=(prefix+['/bin/sh','/opt/webprm-cms/bootstrap.sh'],str(output/'services.log')))
        try:process.start()
        finally:
            if process.pid is not None:tree=OwnedEpisodeTree(process,baseline)
        for attempt in range(150):
            tree.refresh();materializer.check(deadline)
            if not process.is_alive():raise h.Closed('Foreground Shopping services exited during startup')
            try:
                command(mysql,output/f'db-ready-{attempt:03d}.log',deadline,input_bytes=b'SELECT 1;\n',timeout=10)
                break
            except h.Closed:
                time.sleep(2)
        else:raise h.Closed('Imported database did not become ready')
        before_sql=b"SELECT JSON_OBJECT('path',path,'value',value) FROM core_config_data WHERE path IN ('web/unsecure/base_url','web/secure/base_url','admin/security/password_is_forced','admin/security/password_lifetime') ORDER BY path;\n"
        command(mysql,output/'origin-before.jsonl',deadline,input_bytes=before_sql)
        magento=prefix+['php','/var/www/magento2/bin/magento']
        for i,argv in enumerate(ORIGIN_COMMANDS):
            command(magento+argv,output/f'origin-php-{i}.log',deadline,timeout=600)
        command(mysql,output/'origin-secure.log',deadline,input_bytes=SECURE_SQL)
        command(magento+['cache:flush'],output/'cache-flush.log',deadline,timeout=600)
        command(mysql,output/'origin-after.jsonl',deadline,input_bytes=before_sql)
        command(prefix+['supervisorctl','-c','/opt/webprm-cms/supervisord.conf','start','cron'],output/'start-cron.log',deadline)
        customer=validate_sentinel(json.loads(command(mysql,output/'fixed-customer.json',deadline,input_bytes=sentinel,separate_stderr=True)))
        common=dict(native_source=proposal['native_source'],chromium=proposal['chromium'],account_binding=proposal['account_binding'],
            instance_id=context['instance_id'],base_sha256=context['base_sha256'],mutation_name=context['mutation_name'],customer=customer)
        media=Path(context['state'])/'var/www/magento2/pub/media'/context['media_filename']
        if context['phase']=='first':
            base=browser_control(dict(common,phase='baseline',output=str(output/'native-baseline')))
            mutation=browser_control(dict(common,phase='mutate',output=str(output/'native-mutate')))
            observed=validate_sentinel(json.loads(command(mysql,output/'mutated-customer.json',deadline,input_bytes=sentinel,separate_stderr=True)))
            if observed!=mutation['customer']:raise h.Closed('Native UI mutation disagrees with persisted database')
            bytes_new(h.private_path(media),context['mutation_name'].encode(),0o644)
            phase_result=dict(status='passed',baseline=base,mutation=mutation,media_ref=ref(media),instance_id=context['instance_id'])
        elif context['phase']=='second':
            earlier=h.read(h.reference(context['first_result']))
            if customer!=earlier['baseline']['customer']:raise h.Closed('Fresh imported DB does not restore customer sentinel')
            reset=browser_control(dict(common,phase='reset',output=str(output/'native-reset')))
            gate=reset_gate(earlier['baseline'],earlier['mutation'],reset,media_absent=not os.path.lexists(media))
            phase_result=dict(status='passed',reset=reset,gate=gate,instance_id=context['instance_id'])
        else:raise h.Closed('Unknown Shopping fresh-state phase')
        command(prefix+['supervisorctl','-c','/opt/webprm-cms/supervisord.conf','status'],output/'service-status.log',deadline)
        h.put(output/'outcome-before-service-close.json',phase_result)
        command(prefix+['supervisorctl','-c','/opt/webprm-cms/supervisord.conf','shutdown'],output/'shutdown.log',deadline,timeout=60)
        process.join(timeout=60)
        if process.is_alive():raise h.Closed('Shopping supervisor did not exit after shutdown')
    finally:
        try:
            closure=tree.stop(10) if tree else dict(remaining_owned_processes={},child_exitcode=None)
            h.put(output/'service-closure.json',closure)
        finally:_subreaper(previous)
    if closure['remaining_owned_processes'] or closure['child_exitcode']!=0 or phase_result is None:
        raise h.Closed('Shopping namespace did not close completely')
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
    base=scratch/'base';started=time.time()
    receipt=materializer.materialize(layers,base,attempt/'materialization',min(deadline,time.time()+MATERIALIZE_SECONDS),LIMITS)
    timing=dict(materialize_seconds=round(time.time()-started))
    path=h.private_path(Path(receipt['manifest']['path']))
    if h.sha(path)!=receipt['manifest']['sha256']:raise h.Closed('Materialization manifest changed')
    content=content_sha256(json.loads(path.read_text()))
    base_bytes=h.apparent(base,maximum_entries=PROFILE.apparent_entries)
    if 3*base_bytes+context['prior_scratch_bytes']+PROFILE.limits['runtime_reserve_bytes']>PROFILE.limits['scratch_bytes']:
        raise h.Closed('Full base plus two private states fails storage admission')
    base_sha=receipt['manifest']['sha256'];first=None
    for phase,label in (('first','a'),('second','b')):
        state=scratch/('state-'+label);controls=attempt/('controls-'+label)
        started=time.time()
        command(['unshare','--user','--map-root-user','cp','-a','--reflink=auto','--',str(base),str(state)],
            attempt/('copy-'+label+'.log'),deadline,timeout=1800)
        timing['copy_'+label+'_seconds']=round(time.time()-started)
        prepare_controls(state,controls,plan)
        child=dict(context,state=str(state),controls=str(controls),phase=phase,instance_id=context['job_id']+'-'+label,
            instance_output=str(attempt/('instance-'+label)),base_sha256=base_sha,
            owner_pid=os.getpid(),owner_start_ticks=_process(os.getpid())['start_ticks'],
            outer_network_namespace=os.readlink('/proc/self/ns/net'))
        if first:child['first_result']=first
        child_path=attempt/('context-'+label+'.json');h.put(child_path,child)
        started=time.time()
        command(['unshare','--user','--map-root-user','--net',PYTHON,'-B',str(Path(__file__).resolve()),
            '--inside',str(child_path)],attempt/('namespace-'+label+'.log'),deadline,timeout=3000)
        timing['instance_'+label+'_seconds']=round(time.time()-started)
        folder=Path(child['instance_output'])
        if h.read(folder/'service-closure.json')['remaining_owned_processes']!={}:
            raise h.Closed('Previous service state still has owned processes')
        if label=='a':first=ref(folder/'result.json')
    result=h.read(attempt/'instance-b/result.json')
    if result.get('gate',{}).get('fresh_reset_verified') is not True:
        raise h.Closed('Native Shopping fresh-reset gate missing')
    h.put(attempt/'validation-result.json',dict(status='passed',first=first,second=ref(attempt/'instance-b/result.json'),
        materialization=dict(receipt,materialized_content_sha256=content,base_apparent_bytes=base_bytes),timing=timing,
        original_state_equivalence_verified=False,source='third_party_digest_pinned_oci_candidate',
        live_adapter_enabled=False,model_calls=0,api_usd=0,downloads=0))


def run(proposal_path,approval_path):
    proposal,approval=h.read(proposal_path),h.read(approval_path);h.validate(proposal,approval)
    if h.profile_of(proposal) is not PROFILE:raise h.Closed('Shopping hosting profile required')
    root=h.private_path(proposal['campaign_root']);rows=h.history(root,PROFILE);job=os.environ.get('SLURM_JOB_ID','')
    if (not job.isdigit() or not rows or rows[-1]['job_id']!=job or
            rows[-1]['proposal_sha256']!=h.digest(proposal) or '/job_'+job+'/' not in Path('/proc/self/cgroup').read_text() or
            Path(__file__).resolve().parents[1]!=Path(proposal['frozen_source'])):
        raise h.Closed('Durably registered current frozen allocation required')
    budget=h.prior_budget(root,rows[:-1],terminal_rows(rows[:-1]),PROFILE)
    if rows[-1]['requested_seconds']!=budget['remaining_seconds']:raise h.Closed('Prior allocation clock changed')
    raw=subprocess.check_output(['scontrol','--json','show','job',job],text=True,timeout=15)
    clock=h.scheduler(raw,job,requested_seconds=budget['remaining_seconds'],profile=PROFILE)
    attempt=root/('attempt-'+job);attempt.mkdir(mode=0o700,exist_ok=False)
    h.put(attempt/'scheduler.json',dict(raw=raw,captured_epoch=time.time()))
    scratch=h.private_path(Path('/tmp')/('shopv-'+job+'-'+secrets.token_hex(3)));scratch.mkdir(mode=0o700)
    worker_entered=False
    try:
        if shutil.disk_usage(scratch).free<PROFILE.limits['scratch_bytes']-budget['prior_scratch_bytes']:
            raise h.Closed('Insufficient private scratch capacity')
        letters=''.join(random.SystemRandom().choice(string.ascii_letters) for _ in range(16))
        context=dict(proposal_path=str(proposal_path),approval_path=str(approval_path),proposal_sha256=h.digest(proposal),
            approval_sha256=h.sha(approval_path),job_id=job,owner_pid=os.getpid(),owner_start_ticks=_process(os.getpid())['start_ticks'],
            outer_network_namespace=os.readlink('/proc/self/ns/net'),source_script_sha256=h.sha(__file__),
            deadline=clock['work_deadline'],attempt=str(attempt),scratch=str(scratch),prior_scratch_bytes=budget['prior_scratch_bytes'],
            mutation_name='OpenWebRL Reset '+letters,media_filename='webprm-reset-'+secrets.token_hex(8)+'.txt')
        path=attempt/'context.json';h.put(path,context)
        worker_entered=True
        h.own(worker,(path,),attempt=attempt,scratch=scratch,campaign_root=root,deadline=clock['work_deadline'],
              prior_scratch_bytes=budget['prior_scratch_bytes'],profile=PROFILE)
    except BaseException as error:
        h.put(attempt/'failure.json',dict(error_type=type(error).__name__,error=str(error)[:500],worker_owner_entered=worker_entered,
            outcome='failed_preserving_all_state_and_evidence'))
        raise
    finally:
        if not worker_entered:
            h.put(attempt/'closure.json',dict(remaining_owned_processes={},child_exitcode=None,owned_pids=[],
                child_started=False,stage='before_worker_creation'))
            h.put(attempt/'storage-final.json',dict(scratch_bytes=h.apparent(scratch),retained_bytes=h.apparent(root)))
    if h.read(attempt/'validation-result.json').get('status')!='passed':raise h.Closed('Shopping validation incomplete')
    h.put(root/'completed.json',dict(status='hosting_validation_passed_pending_independent_audit',job_id=job,
        validation=ref(attempt/'validation-result.json'),closure=ref(attempt/'closure.json'),
        original_state_equivalence_verified=False,live_adapter_enabled=False))


def prepare(root,campaign_root=None,predecessor=None):
    """A successor package passes the shared campaign root so prior attempts keep their charge."""
    root=h.private_path(root);root.mkdir(mode=0o700,exist_ok=False)
    if h.sha(PLAN)!=PROFILE.plan_sha:raise h.Closed('Shopping service plan changed')
    if not Path(PYTHON).is_file() or not Path(CHROMIUM).is_file():
        raise h.Closed('Preserved browser runtime is missing')
    frozen=root/'frozen-source';frozen.mkdir(mode=0o700)
    for name in SOURCES:bytes_new(frozen/name,(REPO/name).read_bytes())
    r=PROFILE.resources
    batch='\n'.join(['#!/bin/bash','#SBATCH --account=zixianma','#SBATCH --partition=gpu-h200',
        '#SBATCH --qos=normal','#SBATCH --nodes=1','#SBATCH --ntasks=1',f"#SBATCH --cpus-per-task={r['cpus']}",
        f"#SBATCH --mem={r['memory_gib']}G",f"#SBATCH --time={r['total_seconds']//60}",'#SBATCH --job-name=webprm-shop-validate',
        '#SBATCH --output='+str(root/'slurm-%j.log'),'set -euo pipefail',
        'export TMPDIR=/tmp PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 BASH_ENV=/dev/null ENV=/dev/null',
        'exec '+shlex.quote(PYTHON)+' -B '+shlex.quote(str(frozen/'scripts/validate_webprm_shopping_hosting.py'))+
        ' --proposal '+shlex.quote(str(root/'proposal.json'))+' --approval "$1" --run',''])
    bytes_new(frozen/'hosting.sbatch',batch.encode())
    native=h.put(root/'native-source.json',verify_source(NATIVE))
    addon=R/'native-fakeroot-addon'
    addons={str(p.relative_to(addon)):h.sha(p) for p in addon.rglob('*') if p.is_file() and not p.is_symlink()}
    proposal=dict(schema_version=1,status='prepared_not_approved',protocol=PROFILE.protocol,
        campaign_root=str(h.private_path(campaign_root) if campaign_root else root/'campaign'),
        frozen_source=str(frozen),source_sha256={name:h.sha(frozen/name) for name in SOURCES},
        resources=PROFILE.resources,limits=PROFILE.limits,source='third_party_digest_pinned_oci_candidate',
        original_state_equivalence_verified=False,service_plan=ref(PLAN),batch_script=ref(frozen/'hosting.sbatch'),
        acquisition_audit=ref(ACQUIRED/'completed.json'),layer_proof=ref(ACQUIRED/'attempt-356538/raw-layer-verification.json'),
        manifest=ref(ACQUIRED/'metadata/manifest.json'),blob_root=str(ACQUIRED/'attempt-356538/oci/blobs/sha256'),
        bringup_probe=ref(PROBE),native_source=str(NATIVE),native_source_receipt=native,
        account_binding=ref(N/'shopping-native-accounts-v1/account-binding.json'),
        fakeroot_addon=str(addon),fakeroot_source_sha256=addons,python=PYTHON,chromium=CHROMIUM,
        materializer_limits=dict(raw_bytes=LIMITS.raw_bytes,final_bytes=LIMITS.final_bytes,members=LIMITS.members,
            special_members=LIMITS.special_members,seconds=MATERIALIZE_SECONDS),
        nltk_pythonpath=str(R/'nltk-runtime/venv/lib/python3.12/site-packages'),nltk_data=str(R/'nltk-runtime/nltk_data'))
    if predecessor:proposal['predecessor']=predecessor
    h.put(root/'proposal.json',proposal)
    return dict(proposal_path=str(root/'proposal.json'),proposal_sha256=h.digest(proposal),
                status='offline_prepared_requires_review_and_exact_hosting_approval',allocations=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',type=Path);parser.add_argument('--proposal',type=Path);parser.add_argument('--approval',type=Path)
    parser.add_argument('--run',action='store_true');parser.add_argument('--inside',type=Path)
    parser.add_argument('--campaign-root',type=Path);parser.add_argument('--predecessor-json',type=Path)
    args=parser.parse_args()
    if args.prepare:result=prepare(args.prepare,args.campaign_root,
        json.loads(args.predecessor_json.read_text()) if args.predecessor_json else None)
    elif args.inside:return inside(args.inside)
    elif args.run:return run(args.proposal,args.approval)
    else:
        proposal=h.read(args.proposal);result=dict(status='unapproved_resource_preview',resources=proposal['resources'],limits=proposal['limits'],
            proposal_sha256=h.digest(proposal),allocations=0,source=proposal['source'])
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
