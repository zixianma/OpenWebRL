#!/usr/bin/env python3
"""Review/submit one exact approved bounded Table5 allocation.

Preview is the default. Submission holds the new job, records it durably in the
campaign registry, and immediately releases it. Never extends the cumulative
resource contract or retries a claimed primary cell.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from scripts import launch_webprm_table5_campaign as owner

TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY','PREEMPTED','BOOT_FAIL','DEADLINE'}


def financial_handoff(proposal,approval):
    path=Path(proposal['campaign_root'])/'shared-budget.json.events.jsonl'
    if not path.exists():
        if approval.get('previous_closed_epoch_sha256') is not None:
            raise ValueError('Absent financial journal cannot bind a previous epoch')
        return
    previous='0'*64;last=None
    with path.open('rb') as stream:
        for sequence,raw in enumerate(stream):
            if not raw.endswith(b'\n'):raise ValueError('Incomplete financial tail requires recovery audit')
            event=json.loads(raw);core={k:v for k,v in event.items() if k!='event_sha256'}
            if (event.get('sequence')!=sequence or event.get('previous_sha256')!=previous
                    or event.get('event_sha256')!=owner.digest(core)):
                raise ValueError('Financial journal chain differs')
            previous=event['event_sha256'];last=event
    if last is None:raise ValueError('Empty financial journal requires recovery audit')
    expected=last['event_sha256'] if last['event']=='epoch_closed' else None
    if last['event'] not in ('header','epoch_closed') or approval.get('previous_closed_epoch_sha256')!=expected:
        raise ValueError('Financial epoch is not independently closed/bound for recovery')


def preview(proposal_path,approval_path):
    proposal=owner.read(proposal_path);approval=owner.read(approval_path)
    if (approval.get('status')!='approved' or approval.get('proposal_sha256')!=owner.digest(proposal)
            or approval.get('resources')!=proposal['resources'] or approval.get('limits')!=proposal['limits']
            or approval.get('protocol_sha256')!=proposal['protocol_sha256']
            or approval.get('ephemeral_policy_sha256')!=owner.digest(proposal['ephemeral_policy'])
            or approval.get('job_id') is not None):
        raise ValueError('Exact approved request required before submission')
    cap=owner.attempt_cap(proposal,approval)
    owner.validate_proposal(proposal)
    rows=owner.allocation_history(proposal);jobs=[r['job_id'] for r in rows]
    if approval.get('attempt_job_ids')!=jobs or len(jobs)>=cap:
        raise ValueError('Complete prior attempts and remaining allocation slot required')
    elapsed=0
    if jobs:
        root=Path(proposal['campaign_root'])
        for job in jobs:
            owner.closed_attempt_storage(proposal,job)
        raw=subprocess.check_output(['sacct','-j',','.join(jobs),'-X','--noheader','--parsable2',
            '--format=JobID,State,ElapsedRaw'],text=True,timeout=15)
        history=[line.split('|') for line in raw.strip().splitlines()]
        if (len(history)!=len(jobs) or {r[0] for r in history}!=set(jobs)
                or any(r[1].split()[0] not in TERMINAL or not r[2].isdigit() for r in history)):
            raise ValueError('Every old allocation must be terminal and accounted before release')
        elapsed=sum(int(r[2]) for r in history)
    financial_handoff(proposal,approval)
    # A cumulative multi-day campaign must still use normal-QoS allocations of
    # at most 24 hours; every successor subtracts actual prior scheduler time.
    minutes=min(1440,(proposal['resources']['total_seconds']-elapsed)//60)
    if minutes*60<=proposal['limits']['cleanup_seconds']+proposal['limits']['min_start_seconds']:
        raise ValueError('No bounded useful allocation time remains')
    batch=Path(proposal_path).parent/'campaign.sbatch'
    if proposal['source_sha256'].get(str(batch))!=owner.sha(batch):raise ValueError('Frozen batch script differs')
    argv=['sbatch','--hold','--parsable','--time='+str(minutes),'--export=ALL,TMPDIR=/tmp',str(batch),str(approval_path)]
    return proposal,approval,dict(status='approved_submission_preview',prior_job_ids=jobs,
        prior_elapsed_seconds=elapsed,allocation_minutes=minutes,argv=argv,allocations_submitted=0)


def submit(proposal_path,approval_path):
    proposal,approval,request=preview(proposal_path,approval_path)
    root=owner.checked(proposal['campaign_root']);root.mkdir(mode=0o700,exist_ok=True)
    if root.stat().st_mode & 0o077:raise ValueError('Campaign must remain private')
    fd=os.open(owner.checked(root/'submission.lock'),os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        proposal,approval,request=preview(proposal_path,approval_path)  # refresh under exclusive submission lock
        raw=subprocess.check_output(request['argv'],text=True,timeout=30).strip()
        job=raw.split(';',1)[0]
        if not job.isdigit():raise ValueError('Cannot identify held submission; preserve scheduler output: '+raw)
        receipt=Path(proposal_path).parent/('submission-'+job+'.json')
        try:
            owner.put(receipt,dict(request,job_id=job,held=True,proposal_sha256=owner.digest(proposal),
                approval_path=str(approval_path),approval_sha256=owner.sha(approval_path)))
            owner.register_allocation(proposal,approval,job)
            subprocess.check_output(['scontrol','release',job],text=True,timeout=30)
        except BaseException as error:
            cancel_code=cancel_error=record_error=None
            try:
                cancelled=subprocess.run(['scancel',job],capture_output=True,text=True,timeout=30)
                cancel_code=cancelled.returncode
            except BaseException as failure:cancel_error=type(failure).__name__
            try:
                owner.put(receipt.with_name('submission-failure-'+job+'.json'),dict(job_id=job,
                    released=False,cancel_returncode=cancel_code,cancel_error=cancel_error,
                    requires_independent_accounting=True))
            except BaseException as failure:record_error=type(failure).__name__
            raise RuntimeError(f'Held submission {job} failed; cancellation returncode={cancel_code}, '
                f'cancellation_error={cancel_error}, failure_receipt_error={record_error}') from error
        owner.put(receipt.with_name('released-'+job+'.json'),dict(job_id=job,registered=True,released=True,
            registry_path=str(root/'allocation-history'/f"{len(request['prior_job_ids'])+1:04d}-{job}.json")))
        return dict(job_id=job,status='registered_and_released',submission_receipt=str(receipt))
    finally:os.close(fd)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proposal',required=True,type=Path);parser.add_argument('--approval',required=True,type=Path)
    parser.add_argument('--submit',action='store_true');args=parser.parse_args()
    result=submit(args.proposal,args.approval) if args.submit else preview(args.proposal,args.approval)[2]
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
