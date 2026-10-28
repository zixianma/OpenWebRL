#!/usr/bin/env python3
"""Prospective single CMS confirmation transfer under the original cumulative cap.

Preparation is offline. Launch requires a new exact approval; this never reopens
the spent historical transfer claim or overwrites its frozen source/evidence.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts import resume_webprm_cms_acquisition as base
from scripts import acquire_webprm_cms_original as cms
from scripts import webprm_cms_confirmation_transport as transport

OLD=cms.RUNTIME/'next-stage-preparation/cms-original-acquisition-startup-recovery-v1'
OLD_PROPOSAL='07b951987e548072058a3e319c2a83ff119cef8847d9692c03594a3078c51c18'
PROFILE='cms_confirmation_single_successor_v1'
PRIOR_DOWNLOAD=133549
PRIOR_METADATA=2477
PRIOR_RESPONSES=2
# Failed worker executed only safe_environment's empty directory creation before
# receiving HTML. No inspection/extraction/cache-producing library was reached.
# Reserve 2 MiB for this known directory-only path; never report it as measured.
PRIOR_TEMPORARY_RESERVATION=2*2**20
RESOURCES=dict(base.RESOURCES,prior_elapsed_seconds=15,remaining_total_seconds=3585,
    maximum_additional_startup_allocations=2,maximum_transfer_attempts=1)
LIMITS=dict(cms.LIMITS,temporary_bytes=512*2**20-PRIOR_TEMPORARY_RESERVATION)
SOURCES=base.SOURCES+('scripts/webprm_cms_confirmation_transport.py',
    'scripts/resume_webprm_cms_confirmation.py','tests/test_webprm_cms_confirmation.py')
_budget=base.budget
_ledger=cms.TransferLedger


def prior_evidence():
    p=cms.read(OLD/'proposal.json');approval=cms.read(OLD/'approval.json')
    if (cms.digest(p)!=OLD_PROPOSAL or approval.get('status')!='approved' or
            approval.get('proposal_sha256')!=OLD_PROPOSAL):
        raise cms.GateClosed('Preserved approved predecessor identity changed')
    terminal=cms.read(OLD/'startup-01/terminal.json')
    checked=base.terminal_accounting(terminal['scheduler_snapshot'],'355870',3540)
    attempt=OLD/'startup-01/attempt-355870'
    ledger=cms.read(attempt/'transfer-ledger.json')
    closure=cms.read(attempt/'closure.json')
    failure=cms.read(attempt/'failure.json')
    if (checked!=terminal or terminal['elapsed_seconds']!=6 or terminal['state']!='FAILED' or
            closure.get('remaining_owned_processes')!={} or closure.get('child_exitcode')!=1 or
            cms.read(OLD/'transfer-claim.json').get('job_id')!='355870' or
            cms.read(OLD/'startup-01/run-claim.json').get('job_id')!='355870' or
            failure.get('reason')!='Source quota/HTML/unavailable or unbounded metadata; no retry or mirror fallback' or
            (attempt/'archive-attempt-001.partial').stat().st_size!=0 or (OLD/'completed.json').exists() or
            ledger.get('charged_bytes')!=PRIOR_DOWNLOAD or ledger.get('metadata_received_bytes')!=PRIOR_METADATA or
            ledger.get('known_received_bytes')!=PRIOR_METADATA or len(ledger.get('reservations',[]))!=PRIOR_RESPONSES or
            any(x.get('state')!='complete' for x in ledger['reservations']) or
            sum(x['received_bytes']+x['header_reserved_bytes'] for x in ledger['reservations'])!=PRIOR_DOWNLOAD):
        raise cms.GateClosed('Preserved closed confirmation failure accounting changed')
    # Original source paths are labels in the predecessor manifest. Preserve their
    # relative repo identities independently of this successor's execution path.
    prefix='/gpfs/projects/krishna/zixianma/OpenWebRL/'
    for name,expected in p['source_sha256'].items():
        if not name.startswith(prefix) or cms.sha(OLD/'frozen-source'/name[len(prefix):])!=expected:
            raise cms.GateClosed('Historical frozen source changed')
    # Parse the actual preserved page offline; never fetch a token at preparation.
    transport.confirmation_url((attempt/'response-0001-metadata.bin').read_bytes())
    files=('proposal.json','approval.json','transfer-claim.json','startup-01/launch.json',
        'startup-01/run-claim.json','startup-01/terminal.json','startup-01/owner-exit.json',
        'startup-01/attempt-355870/failure.json','startup-01/attempt-355870/closure.json',
        'startup-01/attempt-355870/transfer-ledger.json','startup-01/attempt-355870/response-0001-metadata.bin',
        'closed-attempt-independent-audit-v1.json')
    row=dict(terminal,root=str(OLD),prior_proposal_sha256=OLD_PROPOSAL,
        transfer_attempts=1,archive_body_bytes=0,charged_download_bytes=PRIOR_DOWNLOAD,
        metadata_body_bytes=PRIOR_METADATA,temporary_bytes_reserved_not_measured=PRIOR_TEMPORARY_RESERVATION,
        files_sha256={name:cms.sha(OLD/name) for name in files})
    rows=p['predecessors']+[row]
    if sum(x['elapsed_seconds'] for x in rows)!=15:
        raise cms.GateClosed('Original fifteen-second cumulative allocation accounting changed')
    return rows


def budget(root):
    value=_budget(root);value['prior_seconds']=15
    return value


def approved(root):
    p,a=cms.read(root/'proposal.json'),cms.read(root/'approval.json')
    if (p.get('resource_profile')!=PROFILE or p.get('campaign_root')!=str(root) or
            p.get('resources')!=RESOURCES or p.get('limits')!=LIMITS or
            p.get('predecessors')!=prior_evidence() or
            p.get('original')!=cms.read(OLD/'proposal.json')['original'] or
            a.get('status')!='approved' or a.get('proposal_sha256')!=cms.digest(p) or
            a.get('one_additional_confirmation_transfer_approved') is not True or
            not isinstance(a.get('exact_user_approval'),str) or not a['exact_user_approval'].strip()):
        raise cms.GateClosed('Exact additional confirmation-transfer approval required')
    current=base.source_hashes()
    mutable={str(REPO/'scripts/webprm_cms_confirmation_transport.py'),str(REPO/'tests/test_webprm_cms_confirmation.py')}
    if set(current)!=set(p['source_sha256']) or any(current[k]!=v for k,v in p['source_sha256'].items() if k not in mutable):
        raise cms.GateClosed('Frozen successor ownership/accounting/inspection source changed')
    return p


def install():
    base.PROFILE=PROFILE;base.RESOURCES=RESOURCES;base.MAX_STARTUPS=2;base.SOURCES=SOURCES
    base.SCRIPT=Path(__file__).resolve();base.prior_evidence=prior_evidence;base.budget=budget;base.approved=approved
    cms.LIMITS.clear();cms.LIMITS.update(LIMITS)
    cms.download=transport.download
    cms.TransferLedger=lambda root:transport.CumulativeLedger(root,prior_charged_bytes=PRIOR_DOWNLOAD,
        prior_metadata_bytes=PRIOR_METADATA,prior_responses=PRIOR_RESPONSES)


def prepare(root):
    root=cms.checked_path(root);evidence=prior_evidence();root.mkdir(mode=0o700,parents=True,exist_ok=False)
    proposal=dict(schema_version=1,status='pending_user_approval',resource_profile=PROFILE,
        campaign_root=str(root),resources=RESOURCES,limits=LIMITS,predecessors=evidence,
        original=cms.read(OLD/'proposal.json')['original'],source_sha256=base.source_hashes(),
        cumulative_network=dict(prior_charged_bytes=PRIOR_DOWNLOAD,prior_metadata_bytes=PRIOR_METADATA,
            prior_responses=PRIOR_RESPONSES,total_charged_bytes_cap=10000000000,
            remaining_charged_bytes=10000000000-PRIOR_DOWNLOAD,maximum_additional_transfer_attempts=1),
        cumulative_storage=dict(retained_bytes_cap=12*2**30,temporary_bytes_cap=512*2**20,
            prior_temporary_bytes_reserved_not_measured=PRIOR_TEMPORARY_RESERVATION,
            temporary_bytes_new_attempt_cap=LIMITS['temporary_bytes']),
        behavior=dict(automatic_allocation_retries=0,automatic_network_retries=0,resume=False,
            manual_startup_successor_requires_changed_source_and_review=True,
            maximum_confirmation_forms=1,confirmation_action='https://drive.usercontent.google.com/download',
            confirmation_fields=['id','export','confirm','uuid'],fixed_original_drive_id=cms.DRIVE_ID,
            allowed_archive_responses=['206 exact contiguous original ranges','200 exact full original body at offset zero'],
            full_body_reservation_expanded_before_read=True,preserve_prior_transfer_claim=True,
            original_expected_archive_bytes=cms.ARCHIVE_BYTES,independently_published_archive_checksum=None,
            scope='Original CMS acquisition and offline identity only; no extraction, hosting, browser, model or API calls.'))
    cms.put(root/'proposal.json',proposal)
    cms.put(root/'approval.json',dict(status='pending',proposal_sha256=cms.digest(proposal),
        exact_user_approval=None,one_additional_confirmation_transfer_approved=False))
    for name in SOURCES:
        target=root/'frozen-source'/name;target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as out:out.write((REPO/name).read_bytes())
    return proposal


def main():
    p=argparse.ArgumentParser(description=__doc__);m=p.add_mutually_exclusive_group(required=True)
    for mode in ('prepare','launch','run','reconcile','plan'):m.add_argument('--'+mode,action='store_true')
    p.add_argument('--root',type=Path,required=True);p.add_argument('--startup',type=int)
    p.add_argument('--nonce');p.add_argument('--repair-review',type=Path)
    a=p.parse_args();root=cms.checked_path(a.root);install()
    if a.prepare:print(json.dumps(dict(status='pending_user_approval',proposal_sha256=cms.digest(prepare(root)))))
    elif a.launch:raise SystemExit(base.launch(root,a.repair_review))
    elif a.run:base.run(root,a.startup,a.nonce)
    elif a.reconcile:print(json.dumps(base.reconcile(root,a.startup)))
    else:print(json.dumps(base.next_request(root,a.repair_review)))


if __name__=='__main__':main()
