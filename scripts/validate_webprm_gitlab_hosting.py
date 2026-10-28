#!/usr/bin/env python3
"""Prospective GitLab hosting/reset owner; preparation is offline, execution requires exact approval.
Reuses reviewed CMS allocation, namespace, image-Env and lifecycle primitives. An
immutable GitLab base yields two independent private states. Native login, profile
mutation persisted in SQL, native evaluator positive/negative controls, a repository
storage marker, fresh session and full owned closure are verified. The storage marker
is not claimed as a native Git commit mutation. No model, API or download calls.
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
import urllib.request

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from openwebrl import webprm_cms_hosting as h
from openwebrl import webprm_cms_materializer as materializer
from openwebrl.webprm_gitlab_native import ORIGIN, browser_control, user_sql, reset_gate, released_account, validate_user
from openwebrl.webprm_lifecycle import OwnedEpisodeTree,_process,_processes,_subreaper
from openwebrl.webprm_gitlab_recovery import verify_pristine_base,verify_independent_copy
from openwebrl.webprm_webarena import digest, verify_source
from scripts import validate_webprm_cms_hosting as cms
from scripts.validate_webprm_cms_hosting import (PYTHON, CHROMIUM, NATIVE, R, N, bytes_new, command, container_prefix,
    prepare_controls, private_db_options, ref, service_child, terminal_rows)

PROFILE=h.GITLAB
PLAN=N/'gitlab-hosting-service-plan-20261013-v1/owned-services-plan.json'
ACQUIRED=N/'gitlab-thirdparty-oci-transfer-exception-v1'
PROBE=N/'gitlab-bringup-probe-v6/probe-result.json'
# GitLab probe: ~68 MB manifest and ~34.2 GB apparent baseline.
LIMITS=materializer.Limits(raw_bytes=90*2**30,final_bytes=90*2**30,members=30000000,special_members='fifo_and_skip_devices')
MATERIALIZE_SECONDS=2700
SOURCES=('scripts/validate_webprm_gitlab_hosting.py','scripts/validate_webprm_cms_hosting.py','scripts/submit_webprm_cms_hosting.py',
    'openwebrl/webprm_cms_hosting.py','openwebrl/webprm_cms_materializer.py','openwebrl/webprm_cms_native.py',
    'openwebrl/webprm_gitlab_native.py','openwebrl/webprm_gitlab_recovery.py','openwebrl/webprm_webarena_helpers.py','openwebrl/webprm_lifecycle.py',
    'openwebrl/jev_eval.py','openwebrl/webprm_source.py','openwebrl/webprm_webarena.py','openwebrl/__init__.py',
    'tests/test_webprm_cms_hosting.py','tests/test_validate_webprm_gitlab_hosting.py','tests/test_webprm_gitlab_native.py','tests/test_webprm_gitlab_recovery.py')


def signin_ready():
    """Owned namespace readiness must never use an inherited HTTP proxy."""
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *_args, **_kwargs):
            raise h.Closed('GitLab readiness must not redirect')
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),NoRedirect())
    with opener.open(ORIGIN+'/users/sign_in',timeout=10) as response:
        return response.status==200


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
        raise h.Closed('GitLab services require a fresh owned rootmapped network namespace')
    return context,proposal,plan


def prepare_controls(state, controls, plan):
    """Modify only the newly copied private instance; original image stays unchanged."""
    import re
    state=h.private_path(state);controls=h.private_path(controls)
    controls.mkdir(mode=0o700, exist_ok=False)
    bytes_new(controls/'controller.sh', h.reference(plan['planned_configs']['controller']).read_bytes(), 0o500)
    for target in cms.MOUNT_POINTS:
        h.private_path(state/target.lstrip('/')).mkdir(parents=True, exist_ok=True)
    config=h.private_path(state/'etc/gitlab/gitlab.rb');before=config.read_text()
    bytes_new(controls/'gitlab.rb.before',before.encode())
    updated=re.sub(r"(?m)^external_url.*$", "external_url '"+ORIGIN+"'", before)
    if updated==before: raise h.Closed('Expected original GitLab external_url missing')
    updated+="\npackage['modify_kernel_parameters'] = false\n"
    config.write_text(updated)
    hosts=h.private_path(state/'etc/hosts')
    bytes_new(controls/'hosts.before', hosts.read_bytes())
    hosts.write_text('127.0.0.1 localhost\n::1 localhost ip6-localhost ip6-loopback\n')
    h.put(controls/'adaptations.json',dict(external_url=ORIGIN,kernel_parameter_changes=False,
        gitlab_rb_before=ref(controls/'gitlab.rb.before'),gitlab_rb_after=ref(config),hosts_after=ref(hosts)))


def inside(context_path):
    import urllib.request
    context,proposal,plan=validate_context(context_path,inside=True)
    output=Path(context['instance_output']);output.mkdir(mode=0o700,exist_ok=False)
    deadline=context['deadline'];prefix=container_prefix(context,proposal)
    command(['/usr/sbin/ip','link','set','lo','up'],output/'loopback.log',deadline)
    service_logs=h.private_path(Path(context['state'])/'var/log/webprm-probe')
    if service_logs.exists():raise h.Closed('Fresh private GitLab controller log path required')
    account=released_account(proposal['native_source'],proposal['account_binding'])
    sql=user_sql(account['username'])
    psql=prefix+['gitlab-psql','-d','gitlabhq_production','-At','-v','ON_ERROR_STOP=1']
    previous=_subreaper(True);baseline=set(_processes());process=tree=None;phase_result=None
    def wait_file(name, seconds):
        stop=min(deadline,time.time()+seconds)
        while time.time()<stop:
            tree.refresh()
            if (service_logs/name).exists():return
            if not process.is_alive():raise h.Closed('GitLab services exited before '+name)
            time.sleep(2)
        raise h.Closed('GitLab service deadline: '+name)
    try:
        process=multiprocessing.get_context('fork').Process(target=service_child,
            args=(prefix+['/bin/bash','/opt/webprm-cms/controller.sh'],str(output/'services.log')))
        try:process.start()
        finally:
            if process.pid is not None:tree=OwnedEpisodeTree(process,baseline)
        wait_file('configured',600)
        if (service_logs/'reconfigure.rc').read_text().strip()!='0':raise h.Closed('GitLab configure failed')
        for _ in range(60):
            materializer.check(deadline);tree.refresh()
            try:
                if signin_ready():break
            except Exception:pass
            time.sleep(3)
        else:raise h.Closed('GitLab sign-in did not become ready')
        user=validate_user(json.loads(command(psql,output/'fixed-user.json',deadline,input_bytes=sql,separate_stderr=True)))
        common=dict(native_source=proposal['native_source'],chromium=proposal['chromium'],account_binding=proposal['account_binding'],
            instance_id=context['instance_id'],base_sha256=context['base_sha256'],mutation_name=context['mutation_name'],user=user)
        repository=h.private_path(Path(context['state'])/'var/opt/gitlab/git-data/repositories')
        if not repository.is_dir():raise h.Closed('Restored repository storage missing')
        marker=h.private_path(repository/context['media_filename'])
        if context['phase']=='first':
            base=browser_control(dict(common,phase='baseline',output=str(output/'native-baseline')))
            mutation=browser_control(dict(common,phase='mutate',output=str(output/'native-mutate')))
            observed=validate_user(json.loads(command(psql,output/'mutated-user.json',deadline,input_bytes=sql,separate_stderr=True)))
            if observed!=mutation['user']:raise h.Closed('UI mutation disagrees with GitLab persisted database')
            bytes_new(marker,context['mutation_name'].encode(),0o600)
            phase_result=dict(status='passed',baseline=base,mutation=mutation,repository_marker=ref(marker),
                native_git_commit_mutation_tested=False,instance_id=context['instance_id'])
        elif context['phase']=='second':
            earlier=h.read(h.reference(context['first_result']))
            if user!=earlier['baseline']['user']:raise h.Closed('Fresh imported DB did not restore user')
            reset=browser_control(dict(common,phase='reset',output=str(output/'native-reset')))
            gate=reset_gate(earlier['baseline'],earlier['mutation'],reset,repository_marker_absent=not os.path.lexists(marker))
            phase_result=dict(status='passed',reset=reset,gate=gate,instance_id=context['instance_id'])
        else:raise h.Closed('Unknown GitLab phase')
        h.put(output/'outcome-before-service-close.json',phase_result)
        bytes_new(service_logs/'stop',b'')
        wait_file('stopped',120);process.join(timeout=30)
        if process.is_alive():raise h.Closed('GitLab service owner did not close')
    finally:
        try:
            closure=tree.stop(20) if tree else dict(remaining_owned_processes={},child_exitcode=None)
            h.put(output/'service-closure.json',closure)
        finally:_subreaper(previous)
    if closure['remaining_owned_processes'] or closure['child_exitcode']!=0 or phase_result is None:
        raise h.Closed('GitLab namespace did not close completely')
    h.put(output/'result.json',phase_result)


def verify_reuse(context_path):
    context,proposal,_=validate_context(context_path)
    reuse=h.read(h.reference(proposal['pristine_base_reuse']))
    if os.uname().nodename!=reuse['source_node']:raise h.Closed('Preserved pristine base requires its original node')
    old=h.read(h.reference(reuse['context']))
    closure=h.read(h.reference(reuse['closure']))
    if closure.get('remaining_owned_processes')!={}:raise h.Closed('Old GitLab processes lack verified closure')
    path=h.reference(reuse['materialization'],maximum=128*2**20)
    manifest=json.loads(path.read_text());base=Path(old['scratch'])/'base'
    if str(base)!=reuse['base_path'] or old['job_id']!=reuse['job_id']:raise h.Closed('Exact old unexecuted base required')
    if 'reuse_copy_state' in context:
        state=h.private_path(context['reuse_copy_state'])
        if state.parent!=Path(context['scratch']) or state.name not in ('state-a','state-b'):
            raise h.Closed('Only fresh states in current scratch may be verified')
        other=Path(context['scratch'])/'state-a' if state.name=='state-b' else None
        result=verify_independent_copy(base,state,manifest,context['deadline'],other_state=other)
    else:result=verify_pristine_base(base,manifest,context['deadline'])
    result.update(materialization=reuse['materialization'],source_node=reuse['source_node'])
    h.put(h.private_path(context['reuse_verification_output']),result)


def reuse_check(context,proposal,attempt,label,copy_state=None):
    child=dict(context,reuse_verification_output=str(attempt/('reuse-'+label+'.json')))
    if copy_state is not None:child['reuse_copy_state']=str(copy_state)
    path=attempt/('reuse-context-'+label+'.json');h.put(path,child)
    command(['unshare','--user','--map-root-user',PYTHON,'-B',str(Path(__file__).resolve()),
        '--verify-reuse',str(path)],attempt/('reuse-'+label+'.log'),context['deadline'],timeout=1800)
    return h.read(child['reuse_verification_output'])


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
    started=time.time();reusing='pristine_base_reuse' in proposal
    if reusing:
        verified=reuse_check(context,proposal,attempt,'before')
        base=Path(verified['base'])
        receipt=dict(status='reused_verified_unexecuted_pristine_base',manifest=verified['materialization'])
        base_bytes=verified['apparent_bytes']
    else:
        base=scratch/'base'
        receipt=materializer.materialize(layers,base,attempt/'materialization',min(deadline,time.time()+MATERIALIZE_SECONDS),LIMITS)
        base_bytes=h.apparent(base,maximum_entries=PROFILE.apparent_entries)
    timing=dict(materialize_or_reverify_seconds=round(time.time()-started))
    path=h.private_path(Path(receipt['manifest']['path']))
    if h.sha(path)!=receipt['manifest']['sha256']:raise h.Closed('Materialization manifest changed')
    base_manifest=json.loads(path.read_text());content=content_sha256(base_manifest)
    new_copies=2 if reusing else 3
    if new_copies*base_bytes+context['prior_scratch_bytes']+PROFILE.limits['runtime_reserve_bytes']>PROFILE.limits['scratch_bytes']:
        raise h.Closed('Prior retained scratch plus complete new copies and reserve fails storage admission')
    h.put(attempt/'storage-admission.json',dict(verified=True,base_bytes=base_bytes,new_copies=new_copies,
        prior_scratch_bytes=context['prior_scratch_bytes'],projected_bytes=new_copies*base_bytes+context['prior_scratch_bytes']+PROFILE.limits['runtime_reserve_bytes'],
        scratch_cap_bytes=PROFILE.limits['scratch_bytes'],external_pristine_base_already_charged=reusing))
    base_sha=receipt['manifest']['sha256'];first=None
    for phase,label in (('first','a'),('second','b')):
        state=scratch/('state-'+label);controls=attempt/('controls-'+label)
        started=time.time()
        command(['unshare','--user','--map-root-user','cp','-a','--reflink=auto','--',str(base),str(state)],
            attempt/('copy-'+label+'.log'),deadline,timeout=1800)
        timing['copy_'+label+'_seconds']=round(time.time()-started)
        if reusing:reuse_check(context,proposal,attempt,'copy-'+label,copy_state=state)
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
    if reusing:reuse_check(context,proposal,attempt,'after')
    result=h.read(attempt/'instance-b/result.json')
    if result.get('gate',{}).get('fresh_reset_verified') is not True:
        raise h.Closed('Native GitLab fresh-reset gate missing')
    h.put(attempt/'validation-result.json',dict(status='passed',first=first,second=ref(attempt/'instance-b/result.json'),
        materialization=dict(receipt,materialized_content_sha256=content,base_apparent_bytes=base_bytes),timing=timing,
        original_state_equivalence_verified=False,source='third_party_digest_pinned_oci_candidate',
        live_adapter_enabled=False,model_calls=0,api_usd=0,downloads=0))


def run(proposal_path,approval_path):
    proposal,approval=h.read(proposal_path),h.read(approval_path);h.validate(proposal,approval)
    if h.profile_of(proposal) is not PROFILE:raise h.Closed('GitLab hosting profile required')
    root=h.private_path(proposal['campaign_root']);rows=h.history(root,PROFILE);job=os.environ.get('SLURM_JOB_ID','')
    if (not job.isdigit() or not rows or rows[-1]['job_id']!=job or
            rows[-1]['proposal_sha256']!=h.digest(proposal) or '/job_'+job+'/' not in Path('/proc/self/cgroup').read_text() or
            Path(__file__).resolve().parents[1]!=Path(proposal['frozen_source'])):
        raise h.Closed('Durably registered current frozen allocation required')
    budget=h.prior_budget(root,rows[:-1],terminal_rows(rows[:-1]),PROFILE)
    if rows[-1]['requested_seconds']!=budget['remaining_seconds']:raise h.Closed('Prior allocation clock changed')
    raw=subprocess.check_output(['scontrol','--json','show','job',job],text=True,timeout=15)
    clock=h.scheduler(raw,job,requested_seconds=budget['remaining_seconds'],profile=PROFILE)
    if 'pristine_base_reuse' in proposal:
        reuse=h.read(h.reference(proposal['pristine_base_reuse']))
        if os.uname().nodename!=reuse['source_node']:raise h.Closed('Recovery allocation must use the preserved base node')
    attempt=root/('attempt-'+job);attempt.mkdir(mode=0o700,exist_ok=False)
    h.put(attempt/'scheduler.json',dict(raw=raw,captured_epoch=time.time()))
    scratch=h.private_path(Path('/tmp')/('gitv-'+job+'-'+secrets.token_hex(3)));scratch.mkdir(mode=0o700)
    worker_entered=False
    try:
        if shutil.disk_usage(scratch).free<PROFILE.limits['scratch_bytes']-budget['prior_scratch_bytes']:
            raise h.Closed('Insufficient private scratch capacity')
        letters=''.join(random.SystemRandom().choice(string.ascii_letters) for _ in range(16))
        context=dict(proposal_path=str(proposal_path),approval_path=str(approval_path),proposal_sha256=h.digest(proposal),
            approval_sha256=h.sha(approval_path),job_id=job,owner_pid=os.getpid(),owner_start_ticks=_process(os.getpid())['start_ticks'],
            outer_network_namespace=os.readlink('/proc/self/ns/net'),source_script_sha256=h.sha(__file__),
            deadline=clock['work_deadline'],attempt=str(attempt),scratch=str(scratch),prior_scratch_bytes=budget['prior_scratch_bytes'],
            mutation_name='OpenWebRL GitLab reset '+secrets.token_hex(8),media_filename='webprm-reset-'+secrets.token_hex(8)+'.txt')
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
    if h.read(attempt/'validation-result.json').get('status')!='passed':raise h.Closed('GitLab validation incomplete')
    h.put(root/'completed.json',dict(status='hosting_validation_passed_pending_independent_audit',job_id=job,
        validation=ref(attempt/'validation-result.json'),closure=ref(attempt/'closure.json'),
        original_state_equivalence_verified=False,live_adapter_enabled=False))


def prepare(root,campaign_root=None,predecessor=None):
    """A successor package passes the shared campaign root so prior attempts keep their charge."""
    root=h.private_path(root);root.mkdir(mode=0o700,exist_ok=False)
    if h.sha(PLAN)!=PROFILE.plan_sha:raise h.Closed('GitLab service plan changed')
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
        'exec '+shlex.quote(PYTHON)+' -B '+shlex.quote(str(frozen/'scripts/validate_webprm_gitlab_hosting.py'))+
        ' --proposal '+shlex.quote(str(root/'proposal.json'))+' --approval "$1" --run',''])
    bytes_new(frozen/'hosting.sbatch',batch.encode())
    native=h.put(root/'native-source.json',verify_source(NATIVE))
    addon=R/'native-fakeroot-addon-focal'
    addons={str(p.relative_to(addon)):h.sha(p) for p in addon.rglob('*') if p.is_file() and not p.is_symlink()}
    proposal=dict(schema_version=1,status='prepared_not_approved',protocol=PROFILE.protocol,
        campaign_root=str(h.private_path(campaign_root) if campaign_root else root/'campaign'),
        frozen_source=str(frozen),source_sha256={name:h.sha(frozen/name) for name in SOURCES},
        resources=PROFILE.resources,limits=PROFILE.limits,source='third_party_digest_pinned_oci_candidate',
        scratch_accounting='cumulative_across_all_registered_attempts_including_other_nodes',
        original_state_equivalence_verified=False,service_plan=ref(PLAN),batch_script=ref(frozen/'hosting.sbatch'),
        acquisition_audit=ref(ACQUIRED/'completed.json'),layer_proof=ref(ACQUIRED/'attempt-356548/raw-layer-verification.json'),
        manifest=ref(ACQUIRED/'metadata/manifest.json'),blob_root=str(ACQUIRED/'attempt-356548/oci/blobs/sha256'),
        bringup_probe=ref(PROBE),native_source=str(NATIVE),native_source_receipt=native,
        account_binding=ref(N/'gitlab-native-accounts-v1/account-binding.json'),
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
    parser.add_argument('--verify-reuse',type=Path)
    parser.add_argument('--campaign-root',type=Path);parser.add_argument('--predecessor-json',type=Path)
    args=parser.parse_args()
    if args.prepare:result=prepare(args.prepare,args.campaign_root,
        json.loads(args.predecessor_json.read_text()) if args.predecessor_json else None)
    elif args.verify_reuse:return verify_reuse(args.verify_reuse)
    elif args.inside:return inside(args.inside)
    elif args.run:return run(args.proposal,args.approval)
    else:
        proposal=h.read(args.proposal);result=dict(status='unapproved_resource_preview',resources=proposal['resources'],limits=proposal['limits'],
            proposal_sha256=h.digest(proposal),allocations=0,source=proposal['source'])
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
