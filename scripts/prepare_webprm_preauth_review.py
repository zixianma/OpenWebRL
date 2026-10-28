#!/usr/bin/env python3
"""Prepare immutable additive audits for the two saved CMS preauth interruptions.

No primary files, allocations, provider calls or collector activation are changed.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from scripts.audit_webprm_interrupted_cell import SOURCE_FILES

N=Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();root=args.output
    if root.exists() or root.resolve()!=root or N not in root.parents:
        raise ValueError('Fresh canonical external review directory required')
    root.mkdir(mode=0o700)
    shutil.copytree(N/'cms-table5-350-startup-repair-v7/source',root/'source')
    names=set(SOURCE_FILES)|{'openwebrl/webprm_preauth_audit.py','openwebrl/webprm_cms_site.py',
        'scripts/collect_webprm_table5_campaign.py','scripts/prepare_webprm_preauth_review.py'}
    for name in names:
        path=root/'source'/name;path.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(REPO/name,path)
    # Use a separate interpreter and frozen working directory; never mix live
    # and frozen Python package namespaces in the same verification process.
    subprocess.run([sys.executable,'-B',str(root/'source/scripts/prepare_webprm_preauth_review.py'),
                    '--verify-frozen',str(root)],cwd=root/'source',check=True)


def verify_frozen(root):
    from scripts.collect_webprm_table5_campaign import Evidence
    from openwebrl import webprm_preauth_audit as m
    from scripts import audit_webprm_interrupted_cell as legacy
    ref=legacy.ref
    def put(path,value):
        with path.open('x') as handle:json.dump(value,handle,indent=2);handle.write('\n')
    old=json.loads((N/'cms-364923-interrupted-auth-review-v1/evidence-review.json').read_text())
    campaign=N/'cms-table5-350-v3/campaign';cp=N/'cms-table5-350-v3/cohort.json'
    cohort=json.loads(cp.read_text());cohort=cohort.get('cohort',cohort);cases=[];evidence=Evidence(300)
    for case in old['cells']:
        cell=next(c for c in cohort['cells'] if c['cell_id']==case['cell_id'])
        folder=Path(case['artifacts']['claim']['path']).parent
        episode=Path(case['artifacts']['result']['path']).parent;attempt=folder.parents[1];key=folder.name
        paths=dict(plan=folder/'worker-plan.json',claim=folder/'state-claim.json',launch=folder/'worker-launch.json',
            result=episode/'result.json',watchdog=episode/'watchdog.json',phase=episode/'phase.json',
            lifecycle=episode/'lifecycle.jsonl',instance=folder/'site/instance.json',lease=folder/'lease.json',
            site_closure=folder/'site/closure.json',outer_closure=attempt/'controller-closure.json',
            copy_receipt=folder/'copy/receipt.json',contract=attempt/'financial-contract.json',
            queue_manifest=campaign/'queue/manifest.json',queue_claim=campaign/'queue/claims'/(m.digest(cell['cell_id'])+'.json'),
            queue_outcome=campaign/'queue/outcomes'/(m.digest(cell['cell_id'])+'.json'))
        deps=set(legacy.SOURCE_FILES)|{'openwebrl/webprm_preauth_audit.py','openwebrl/webprm_cms_site.py'}
        inputs=dict(protocol=m.PROTOCOL,cell=cell,campaign_root=str(campaign),cell_folder=str(folder),
            cohort=ref(cp),worker_plan=ref(paths['plan']),implementation=ref(root/'source/openwebrl/webprm_preauth_audit.py'),
            dependency_sources={p:ref(root/'source'/p) for p in sorted(deps)},
            audit_source_sha256={p:legacy.sha(root/'source'/p) for p in legacy.SOURCE_FILES},
            artifacts={k:ref(p) for k,p in paths.items()},recovery=ref(N/'cms-364923-epoch-recovery/final.json'),
            owner_closure=ref(N/'cms-364923-epoch-recovery/closure.json'),
            financial_review=ref(N/'cms-364923-epoch-recovery/receipt-audit.json'))
        ip=root/(key+'.input.json');put(ip,inputs)
        result=m.verify(evidence,inputs,cell,cohort,campaign,folder);ap=root/(key+'.audit.json')
        put(ap,dict(protocol=m.PROTOCOL,audit_verified=True,errors=[],implementation=inputs['implementation'],
            input_reference=ref(ip),reconstruction=result));rp=root/(key+'.record.json')
        put(rp,dict(result['record'],audit_sha256=ref(ap)['sha256']))
        cases.append(dict(cell_id=cell['cell_id'],folder=str(folder),input=ref(ip),audit=ref(ap),record=ref(rp)))
    evidence.unchanged()
    put(root/'preparation.json',dict(status='candidates_verified_not_activated',campaign_root=str(campaign),
        cohort=ref(cp),implementation=inputs['implementation'],cases=cases,raw_primary_files_changed=0,
        native_verdicts_created=0,state_disposals=0,new_provider_calls=0,new_allocations=0))
    print(json.dumps(dict(status='candidates_verified_not_activated',cells=len(cases))))


if __name__=='__main__':
    if len(sys.argv)==3 and sys.argv[1]=='--verify-frozen':verify_frozen(Path(sys.argv[2]))
    else:main()
