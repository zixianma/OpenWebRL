#!/usr/bin/env python3
"""Prepare additive pre-actor failure audits; never activate pointers or rerun work."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
from scripts.collect_webprm_table5_campaign import Evidence, canonical


def ref(path):
    return dict(path=str(path), sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())


def put(path, value):
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())


def prepare(campaign, cohort_path, job, output):
    campaign=canonical(campaign);cohort_path=canonical(cohort_path);output=canonical(output)
    if not job.isdigit() or output.exists() or campaign in output.parents:
        raise ValueError('Fresh external review directory and exact job required')
    attempt=campaign/('attempt-'+job)
    scheduler=subprocess.check_output(['sacct','-j',job,'-X','--noheader','--parsable2',
        '--format=JobID,State,ElapsedRaw,ExitCode'],text=True,timeout=20).strip().split('|')
    if len(scheduler)!=4 or scheduler[:2]!=[job,'COMPLETED']:
        raise ValueError('Exact normally completed allocation required')
    output.mkdir(mode=0o700);(output/'source').mkdir(mode=0o700)
    source=output/'source/webprm_pre_actor_audit.py'
    shutil.copyfile(REPO/'openwebrl/webprm_pre_actor_audit.py',source)
    implementation=ref(source);spec=importlib.util.spec_from_file_location('_pinned_pre_actor_builder',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    reports=campaign/'operational-audits'/('pre-actor-readiness-'+job)
    reports.mkdir(mode=0o700,parents=True,exist_ok=True)
    sched=dict(job_id=job,state=scheduler[1],elapsed_seconds=int(scheduler[2]),exit_code=scheduler[3])
    schedpath=reports/'scheduler.json'
    if schedpath.exists():
        if json.loads(schedpath.read_text())!=sched:raise ValueError('Historical scheduler proof changed')
    else:put(schedpath,sched)
    cohort=json.loads(cohort_path.read_text());cohort=cohort.get('cohort',cohort)
    cells={cell['cell_id']:cell for cell in cohort['cells']}
    budgetpath=campaign/'shared-budget.json.events.jsonl';prefix=b''
    with budgetpath.open('rb') as stream:
        for raw in stream:
            prefix+=raw;event=json.loads(raw)
            if event.get('event')=='epoch_closed' and event.get('epoch_id')=='job-'+job:break
        else:raise ValueError('No closed financial prefix for this attempt')
    budget=dict(path=str(budgetpath),bytes=len(prefix),sha256=hashlib.sha256(prefix).hexdigest())
    brokerpath=attempt/'broker.jsonl';broker=dict(ref(brokerpath),bytes=brokerpath.stat().st_size)
    evidence=Evidence(300);cases=[]
    for path in sorted((campaign/'queue/outcomes').glob('*.json')):
        outcome=json.loads(path.read_text());cell_id=outcome['cell_id'];key=hashlib.sha256(cell_id.encode()).hexdigest()
        folder=attempt/'cells'/key
        if not (folder/'state-claim.json').exists() or outcome['result'].get('reason')!='worker_exception':continue
        cell=cells[cell_id]
        artifacts=dict(worker_plan=folder/'worker-plan.json',state_claim=folder/'state-claim.json',
            queue_claim=campaign/'queue/claims'/path.name,queue_outcome=path,
            site_instance=folder/'site/instance.json',site_closure=folder/'site/closure.json',
            copy_receipt=folder/'copy/receipt.json',context=attempt/'execution-context.json',
            controller_closure=attempt/'controller-closure.json',storage=attempt/'storage-final.json',
            financial_contract=attempt/'financial-contract.json',readiness=folder/'readiness-0.log',scheduler=schedpath)
        inputs=dict(protocol=module.PROTOCOL,cell=cell,campaign_root=str(campaign),cell_folder=str(folder),
            implementation=implementation,artifacts={name:ref(value) for name,value in artifacts.items()},
            budget=budget,broker=broker)
        stem=key+'.'+output.name
        inputpath=reports/(stem+'.input.json');put(inputpath,inputs)
        rebuilt=module.verify(evidence,inputs,cell,cohort,campaign,folder)
        auditpath=reports/(stem+'.audit.json')
        put(auditpath,dict(protocol=module.PROTOCOL,audit_verified=True,errors=[],implementation=implementation,
            input_reference=ref(inputpath),reconstruction=rebuilt))
        recordpath=reports/(stem+'.record.json');put(recordpath,dict(rebuilt['record'],audit_sha256=ref(auditpath)['sha256']))
        pointerpath=reports/(stem+'.pointer.json')
        put(pointerpath,dict(protocol=module.PROTOCOL,implementation=implementation,input=ref(inputpath),
            audit=ref(auditpath),record=ref(recordpath)))
        cases.append(dict(cell_id=cell_id,folder=str(folder),input=str(inputpath),candidate_pointer=str(pointerpath)))
    if not cases:raise ValueError('No matching pre-actor failures')
    result=dict(status='candidates_verified_not_activated',cases=cases,implementation=implementation,
        cohort=str(cohort_path),campaign_root=str(campaign),native_verdicts_created=0,raw_primary_files_changed=0,
        state_disposals=0,new_provider_calls=0,new_allocations=0)
    put(output/'preparation.json',result)
    return dict(status=result['status'],cells=len(cases),output=str(output))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('campaign','cohort','output'):parser.add_argument('--'+name,required=True,type=Path)
    parser.add_argument('--job',required=True);args=parser.parse_args()
    print(json.dumps(prepare(args.campaign,args.cohort,args.job,args.output)))
