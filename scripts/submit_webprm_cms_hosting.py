#!/usr/bin/env python3
"""Preview or submit an exactly approved Magento (CMS/Shopping) hosting validation job; hold/register/release."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import validate_webprm_cms_hosting as owner
h=owner.h


def submission_lineage(root, rows):
    """A crashed/ambiguous submit cannot silently regain an attempt budget.

    Registered attempts are accounted by prior_budget. An intent with no
    registered job requires independent scheduler reconciliation; this helper
    never infers that a failed sbatch transport means no job was created.
    """
    root=Path(root);registered={row['job_id'] for row in rows}
    references={}
    for path in root.glob('submission-[0-9]*.json'):
        row=h.read(path);intent=h.reference(row['intent'])
        if (intent.parent!=root or not intent.name.startswith('submission-intent-') or
                row.get('job_id') not in registered or str(intent) in references):
            raise h.Closed('Unregistered/ambiguous held submission needs independent terminal accounting')
        references[str(intent)]=row['job_id']
    if {str(p) for p in root.glob('submission-intent-*.json')}!=set(references):
        raise h.Closed('Unresolved submission intent needs independent scheduler reconciliation')


def _profile(p):
    # validate() has already rejected unknown protocols; this only selects caps/labels.
    return h.PROFILES.get(p.get('protocol'),h.CMS)


def preview(proposal_path,approval_path):
    p,a=h.read(proposal_path),h.read(approval_path);h.validate(p,a);profile=_profile(p)
    rows=h.history(p['campaign_root'],profile)
    submission_lineage(p['campaign_root'],rows)
    if len(rows)>=profile.resources['max_allocations']:
        raise h.Closed(profile.label+' hosting allocation cap exhausted')
    terminals=owner.terminal_rows(rows);budget=h.prior_budget(p['campaign_root'],rows,terminals,profile)
    batch=h.reference(p['batch_script'])
    if batch.parent!=Path(p['frozen_source']):raise h.Closed('Frozen batch path differs')
    return p,a,dict(prior_job_ids=[r['job_id'] for r in rows],terminal_rows=terminals,budget=budget,
        requested_minutes=budget['remaining_seconds']//60,batch_script=str(batch),allocations_submitted=0)


def submit(proposal_path,approval_path):
    p,a,request=preview(proposal_path,approval_path)
    root=h.private_path(p['campaign_root']);root.mkdir(mode=0o700,exist_ok=True)
    with os.fdopen(os.open(root/'submission.lock',os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600),'r+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        p,a,request=preview(proposal_path,approval_path)
        nonce=secrets.token_hex(12)
        argv=['sbatch','--hold','--parsable','--time='+str(request['requested_minutes']),
              '--comment=webprm-'+_profile(p).name+'-host:'+nonce,
              '--export=ALL,TMPDIR=/tmp,BASH_ENV=/dev/null,ENV=/dev/null,PYTHONDONTWRITEBYTECODE=1,PYTHONNOUSERSITE=1',
              request['batch_script'],str(approval_path)]
        intent=root/('submission-intent-'+nonce+'.json')
        h.put(intent,dict(request,argv=argv,nonce=nonce,proposal_sha256=h.digest(p),approval=owner.ref(approval_path)))
        raw=subprocess.check_output(argv,text=True,timeout=30).strip();job=raw.split(';',1)[0]
        if not job.isdigit():
            raise h.Closed('Held job identity unknown; reconcile scheduler comment webprm-'+_profile(p).name+'-host:'+nonce)
        receipt=root/('submission-'+job+'.json')
        try:
            h.put(receipt,dict(job_id=job,held=True,intent=owner.ref(intent),raw_scheduler_output=raw))
            held=subprocess.check_output(['scontrol','--json','show','job',job],text=True,timeout=15)
            h.register(p,a,job,held,request['terminal_rows'])
            subprocess.check_output(['scontrol','release',job],text=True,timeout=30)
        except BaseException as failure:
            cancel=None;cancel_error=record_error=None
            try:cancel=subprocess.run(['scancel',job],capture_output=True,text=True,timeout=30).returncode
            except BaseException as error:cancel_error=type(error).__name__
            try:
                h.put(root/('submission-failure-'+job+'.json'),dict(job_id=job,cancel_returncode=cancel,
                    cancel_error=cancel_error,requires_independent_terminal_accounting=True))
            except BaseException as error:record_error=type(error).__name__
            raise h.Closed(f'{_profile(p).label} held job {job} failed; cancellation={cancel}, cancellation_error={cancel_error}, receipt_error={record_error}') from failure
        h.put(root/('released-'+job+'.json'),dict(job_id=job,registered=True,released=True))
        return dict(job_id=job,status='registered_and_released',submission_receipt=str(receipt))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proposal',type=Path,required=True);parser.add_argument('--approval',type=Path,required=True)
    parser.add_argument('--submit',action='store_true');args=parser.parse_args()
    result=submit(args.proposal,args.approval) if args.submit else preview(args.proposal,args.approval)[2]
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
