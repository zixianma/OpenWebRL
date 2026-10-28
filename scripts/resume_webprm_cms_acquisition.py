#!/usr/bin/env python3
"""Manually supervise bounded CMS startup recovery; never retry a transfer.

The campaign remains under the original 3,600 allocation-second cap. Each
invocation owns one salloc/srun and waits for it. A subsequent invocation needs
terminal accounting plus a changed-source repair review; it is never automatic.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import acquire_webprm_cms_original as cms

SCRIPT = Path(__file__).resolve()
PROFILE = 'cms_cumulative_startup_recovery_v1'
PREVIOUS = cms.RUNTIME/'next-stage-preparation/cms-original-acquisition-replacement-v3'
TOTAL_SECONDS = 3600
MAX_STARTUPS = 3
RESOURCES = dict(account='zixianma', partition='gpu-h200', qos='normal', nodes=1,
                 cpus=2, memory_gib=8, gpus=0, original_total_seconds=TOTAL_SECONDS,
                 prior_elapsed_seconds=9, remaining_total_seconds=3591,
                 maximum_additional_startup_allocations=MAX_STARTUPS,
                 maximum_transfer_attempts=1, api_budget_usd=0, model_provider_calls=0)
SOURCES = cms.SOURCES + ('scripts/resume_webprm_cms_acquisition.py',
                         'tests/test_resume_webprm_cms_acquisition.py')


def source_hashes():
    return {str(REPO/name): cms.sha(REPO/name) for name in SOURCES}


def terminal_accounting(snapshot, job, requested_seconds):
    """Only immutable terminal scheduler epochs reclaim a reservation."""
    try:
        if snapshot.get('errors') != [] or len(snapshot['jobs']) != 1:
            raise ValueError('One error-free terminal job required')
        row = snapshot['jobs'][0]
        def finite(key):
            value = row[key]
            if (value.get('set') is not True or value.get('infinite') is not False or
                    type(value.get('number')) is not int):
                raise ValueError('Finite integer scheduler field required')
            return value['number']
        start, end = finite('start_time'), finite('end_time')
        if (str(row['job_id']) != str(job) or row['user_id'] != os.getuid() or
                row['restart_cnt'] != 0 or
                row['job_state'] not in (['FAILED'], ['COMPLETED'], ['CANCELLED'], ['TIMEOUT']) or
                start <= 0 or end < start or finite('time_limit')*60 != requested_seconds or
                finite('cpus') != 2 or finite('node_count') != 1 or finite('memory_per_node') != 8192 or
                row['account'] != 'zixianma' or row['partition'] != 'gpu-h200' or row['qos'] != 'normal'):
            raise ValueError('Terminal identity/resources/clock mismatch')
        tres = dict(piece.split('=', 1) for piece in row['tres_alloc_str'].split(',') if '=' in piece)
        if any(k.startswith('gres/gpu') and v != '0' for k,v in tres.items()):
            raise ValueError('Unexpected GPU allocation')
        return dict(job_id=str(job), elapsed_seconds=end-start,
                    requested_seconds=requested_seconds, state=row['job_state'][0],
                    scheduler_snapshot=snapshot)
    except (KeyError, ValueError, TypeError, AttributeError) as error:
        raise cms.GateClosed('No independently verified terminal accounting; retain full reservation') from error


def prior_evidence():
    old = cms.read(PREVIOUS/'proposal.json')
    first = cms.predecessor_evidence(old['predecessor']['scheduler_snapshot'])
    auth, launch, claim = (cms.read(PREVIOUS/name) for name in ('approval.json','launch.json','run-claim.json'))
    if (auth.get('status') != 'approved' or auth.get('proposal_sha256') != cms.digest(old) or
            launch.get('proposal_sha256') != cms.digest(old) or
            claim.get('job_id') != '355691' or claim.get('nonce') != launch.get('nonce') or
            old.get('resources') != cms.REPLACEMENT_RESOURCES or
            any(PREVIOUS.glob('attempt-*')) or any(PREVIOUS.rglob('transfer-ledger.json'))):
        raise cms.GateClosed('Preserved second zero-transfer startup is unavailable or changed')
    for name, expected in old['source_sha256'].items():
        if cms.sha(PREVIOUS/'frozen-source'/Path(name).relative_to(REPO)) != expected:
            raise cms.GateClosed('Second approved source evidence changed')
    second = terminal_accounting(cms.read(PREVIOUS/'scheduler-terminal-355691.json'), '355691', 3540)
    if second['elapsed_seconds'] != 5 or second['state'] != 'FAILED':
        raise cms.GateClosed('Expected five-second second startup failure')
    second.update(root=str(PREVIOUS), transfer_attempts=0, transfer_body_bytes=0,
                  prior_proposal_sha256=cms.digest(old),
                  files_sha256={n:cms.sha(PREVIOUS/n) for n in
                                ('proposal.json','approval.json','launch.json','run-claim.json','scheduler-terminal-355691.json')})
    return [first, second]


def prepare(root):
    root = cms.checked_path(root)
    evidence = prior_evidence()
    if sum(x['elapsed_seconds'] for x in evidence) != 9:
        raise cms.GateClosed('Prior elapsed accounting changed')
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    proposal = dict(schema_version=1, status='pending_user_approval', resource_profile=PROFILE,
        campaign_root=str(root), resources=RESOURCES, limits=cms.LIMITS, predecessors=evidence,
        original=cms.read(PREVIOUS/'proposal.json')['original'], source_sha256=source_hashes(),
        behavior=dict(automatic_allocation_retries=0, automatic_network_retries=0, resume=False,
            manual_successor_requires_changed_source_and_review=True, scheduler_rounding='floor(remaining_seconds/60)*60',
            unknown_allocations_keep_full_reservation=True, one_transfer_claim_for_entire_campaign=True,
            all_predecessor_and_attempt_storage_counted=True, owned_salloc_awaits_owned_srun=True,
            scope='Official CMS archive acquisition and offline identity only; no extraction, hosting, browser, benchmark, model, or API calls.'))
    cms.put(root/'proposal.json', proposal)
    cms.put(root/'approval.json', dict(status='pending', proposal_sha256=cms.digest(proposal),
        exact_user_approval=None, startup_attempt_cap_extension_approved=False))
    return proposal


def approved(root):
    root = cms.checked_path(root)
    p, a = cms.read(root/'proposal.json'), cms.read(root/'approval.json')
    if (p.get('resource_profile') != PROFILE or p.get('campaign_root') != str(root) or
            p.get('resources') != RESOURCES or p.get('limits') != cms.LIMITS or
            p.get('predecessors') != prior_evidence() or
            p.get('original') != cms.read(PREVIOUS/'proposal.json')['original'] or
            a.get('status') != 'approved' or a.get('proposal_sha256') != cms.digest(p) or
            a.get('startup_attempt_cap_extension_approved') is not True or
            not isinstance(a.get('exact_user_approval'),str) or not a['exact_user_approval'].strip()):
        raise cms.GateClosed('Exact bounded startup-attempt extension is not approved')
    # The resource/ownership controller and inspector stay frozen. A manually
    # reviewed later fix may change only the acquisition implementation/tests.
    mutable = {str(REPO/'scripts/acquire_webprm_cms_original.py'),
               str(REPO/'tests/test_acquire_webprm_cms_original.py')}
    current = source_hashes()
    if set(current) != set(p['source_sha256']) or any(
            current[k] != v for k,v in p['source_sha256'].items() if k not in mutable):
        raise cms.GateClosed('Frozen recovery ownership/inspection source changed')
    return p


def budget(root):
    p = cms.read(root/'proposal.json')
    rows = []
    used = sum(x['elapsed_seconds'] for x in p['predecessors'])
    for path in sorted(root.glob('startup-[0-9][0-9]')):
        c = cms.read(path/'launch.json')
        requested = c['requested_seconds']
        if type(requested) is not int or requested < 60 or requested % 60 or requested > 3600-used:
            raise cms.GateClosed('Allocation reservation exceeds original cumulative budget')
        terminal = path/'terminal.json'
        if terminal.exists():
            v = cms.read(terminal)
            checked = terminal_accounting(v['scheduler_snapshot'], v['job_id'], requested)
            if checked != v or cms.read(path/'run-claim.json')['job_id'] != v['job_id']:
                raise cms.GateClosed('Terminal receipt or owned allocation identity changed')
            charge, settled = v['elapsed_seconds'], True
        else:
            charge, settled = requested, False
        used += charge
        rows.append(dict(path=str(path), requested_seconds=requested, charged_seconds=charge, settled=settled))
    if len(rows)>MAX_STARTUPS or used>TOTAL_SECONDS:
        raise cms.GateClosed('Original cumulative CPU budget or additional startup limit exhausted')
    return dict(original_seconds=TOTAL_SECONDS, prior_seconds=9, charged_seconds=used,
                remaining_seconds=TOTAL_SECONDS-used, additional_startups=len(rows), attempts=rows)


def next_request(root, repair_review=None):
    p = approved(root)
    b = budget(root)
    if (root/'transfer-claim.json').exists() or (root/'completed.json').exists():
        raise cms.GateClosed('The sole transfer attempt was already claimed; no network retry/resume')
    if any(not x['settled'] for x in b['attempts']):
        raise cms.GateClosed('Prior allocation unresolved; no overlapping or unknown retry')
    if b['additional_startups'] >= MAX_STARTUPS:
        raise cms.GateClosed('Additional startup attempt cap reached')
    current = source_hashes()
    if not b['attempts']:
        if repair_review is not None or current != p['source_sha256']:
            raise cms.GateClosed('Initial recovery must use the reviewed frozen source')
    else:
        previous = Path(b['attempts'][-1]['path'])
        failure = cms.read(previous/'failure.json')
        old = cms.read(previous/'launch.json')['source_sha256']
        review = cms.read(repair_review) if repair_review is not None else {}
        if (failure.get('stage') != 'startup_before_transfer' or failure.get('status') != 'failed' or
                not isinstance(failure.get('error_type'),str) or
                review.get('previous_startup') != str(previous) or
                review.get('previous_failure_sha256') != cms.sha(previous/'failure.json') or
                review.get('source_sha256') != current or current == old or
                not isinstance(review.get('explanation'),str) or not review['explanation'].strip() or
                review.get('offline_tests_passed') is not True or
                cms.sha(cms.checked_path(review.get('test_log_path',''))) != review.get('test_log_sha256')):
            raise cms.GateClosed('A concrete changed-source repair and passing offline review are required before manual retry')
    seconds = b['remaining_seconds']//60*60
    if seconds <= cms.LIMITS['closure_reserve_seconds']:
        raise cms.GateClosed('No whole-minute useful work remains inside original budget')
    return dict(budget=b, requested_seconds=seconds, source_sha256=current,
                repair_review=None if repair_review is None else cms.read(repair_review))


def reconcile(root, startup):
    path = cms.checked_path(root/f'startup-{startup:02d}')
    if (path/'terminal.json').exists():
        return budget(root)
    claim = cms.read(path/'run-claim.json')
    snapshot = json.loads(subprocess.check_output(['scontrol','--json','show','job',claim['job_id']],text=True,timeout=15))
    result = terminal_accounting(snapshot, claim['job_id'], cms.read(path/'launch.json')['requested_seconds'])
    cms.put(path/'terminal.json', result)
    return budget(root)


def launch(root, repair_review=None):
    plan = next_request(root, repair_review)
    number = plan['budget']['additional_startups']+1
    path = cms.checked_path(root/f'startup-{number:02d}'); path.mkdir(mode=0o700)
    nonce = secrets.token_hex(16)
    minutes = plan['requested_seconds']//60
    argv = ['salloc','--account=zixianma','--partition=gpu-h200','--qos=normal',
            '--nodes=1','--ntasks=1','--cpus-per-task=2','--mem=8G',f'--time=00:{minutes:02d}:00',
            '--job-name=wa-cms-recovery','--chdir='+str(root),'srun','--ntasks=1','--cpus-per-task=2',
            sys.executable,'-I','-B',str(SCRIPT),'--run','--root',str(root),'--startup',str(number),'--nonce',nonce]
    cms.put(path/'launch.json',dict(plan,status='submission_claimed',nonce=nonce,argv=argv,
                                  proposal_sha256=cms.digest(cms.read(root/'proposal.json')),at_epoch=time.time()))
    for name in SOURCES:
        target=cms.checked_path(path/'frozen-source'/name);target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as handle:handle.write((REPO/name).read_bytes())
    env=cms.safe_environment(path/'launch-cache')
    env={k:v for k,v in env.items() if not k.startswith('SLURM_')}
    # No unattended successor: the invoking controller owns and awaits salloc.
    result=subprocess.run(argv,env=env,cwd=root,check=False)
    cms.put(path/'owner-exit.json',dict(returncode=result.returncode,finished_epoch=time.time()))
    try:
        state=reconcile(root,number)
    except Exception as error:
        cms.put(path/'accounting-unresolved.json',dict(error_type=type(error).__name__,error=str(error),
                    reservation_retained=True,automatic_retry=False))
        raise
    print(json.dumps(dict(status='allocation_closed',budget=state,returncode=result.returncode)))
    return result.returncode


def run(root,number,nonce):
    path=cms.checked_path(root/f'startup-{number:02d}')
    stage='startup_before_transfer'
    try:
        p=approved(root);claim=cms.read(path/'launch.json')
        job=os.environ.get('SLURM_JOB_ID','')
        if (not job.isdigit() or f'/job_{job}/' not in Path('/proc/self/cgroup').read_text() or
                claim.get('nonce')!=nonce or claim.get('proposal_sha256')!=cms.digest(p) or
                claim.get('source_sha256')!=source_hashes()):
            raise cms.GateClosed('Owned recovery allocation or frozen attempt source mismatch')
        cms.put(path/'run-claim.json',dict(job_id=job,nonce=nonce,at_epoch=time.time()))
        b=budget(root)
        if b['additional_startups']!=number or number<1 or number>MAX_STARTUPS:
            raise cms.GateClosed('Concurrent or out-of-order recovery allocation')
        resources=dict(cms.RESOURCES,total_allocation_seconds_including_retries=claim['requested_seconds'])
        context=cms.saved_allocation_context(path,job,nonce,claim,resources)
        prior=sum(cms.apparent_bytes(Path(x['root'])) for x in p['predecessors'])
        retained=cms.apparent_bytes(root)+prior
        if (shutil.disk_usage(root).free<cms.LIMITS['initial_persistent_free_bytes'] or
                shutil.disk_usage('/tmp').free<cms.LIMITS['initial_temporary_free_bytes'] or
                retained+cms.ARCHIVE_BYTES+cms.LIMITS['metadata_received_bytes']+2*cms.LIMITS['receipt_reserve_bytes']>cms.LIMITS['retained_bytes']):
            raise cms.GateClosed('Cumulative retained-storage or free-space admission failed')
        # This exclusive campaign-level marker spends the one transfer attempt.
        # Even an interrupted worker cannot be silently launched a second time.
        cms.put(root/'transfer-claim.json',dict(startup=number,job_id=job,at_epoch=time.time(),resume_allowed=False))
        stage='sole_transfer_attempt'
        scratch=cms.checked_path(Path('/tmp')/f'webprm-cms-recovery-{job}-{secrets.token_hex(4)}');scratch.mkdir(mode=0o700)
        attempt=cms.checked_path(path/('attempt-'+job));attempt.mkdir(mode=0o700)
        cms.put(attempt/'context.json',dict(context,scratch=str(scratch),proposal_sha256=cms.digest(p)))
        # Count all campaign/predecessor bytes outside this attempt in live guards.
        outside=cms.apparent_bytes(root)+prior-cms.apparent_bytes(path)
        cms.own_worker(attempt,scratch,context['work_deadline'],prior_retained_bytes=outside)
        result,proof,transfer=(cms.read(attempt/name) for name in
                              ('stage-result.json','archive-identity.json','download-complete.json'))
        if (result.get('status')!='acquisition_and_internal_identity_verified' or
                result.get('archive_sha256')!=proof.get('archive_sha256') or
                result.get('archive_sha256')!=transfer.get('archive_sha256') or
                time.time()>=context['allocation_deadline'] or
                cms.apparent_bytes(root)+prior>cms.LIMITS['retained_bytes'] or
                cms.apparent_bytes(scratch)>cms.LIMITS['temporary_bytes']):
            raise cms.GateClosed('Final archive identity, cumulative storage, or allocation clock failed')
        cms.put(root/'completed.json',dict(status='acquisition_and_internal_identity_verified',job_id=job,
                    startup=number,result_path=str(attempt/'stage-result.json'),ready_to_host=False,
                    finished_epoch=time.time(),allocation_deadline=context['allocation_deadline']))
    except Exception as error:
        if not (path/'failure.json').exists():
            cms.put(path/'failure.json',dict(status='failed',stage=stage,error_type=type(error).__name__,
                    error=str(error),at_epoch=time.time(),automatic_retry=False))
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    m=p.add_mutually_exclusive_group(required=True)
    for mode in ('prepare','launch','run','reconcile','plan'):m.add_argument('--'+mode,action='store_true')
    p.add_argument('--root',type=Path,required=True);p.add_argument('--startup',type=int)
    p.add_argument('--nonce');p.add_argument('--repair-review',type=Path)
    a=p.parse_args();root=cms.checked_path(a.root)
    if a.prepare:print(json.dumps(dict(status='pending_user_approval',proposal_sha256=cms.digest(prepare(root)))))
    elif a.launch:raise SystemExit(launch(root,a.repair_review))
    elif a.run:run(root,a.startup,a.nonce)
    elif a.reconcile:print(json.dumps(reconcile(root,a.startup)))
    else:print(json.dumps(next_request(root,a.repair_review)))


if __name__=='__main__':main()
