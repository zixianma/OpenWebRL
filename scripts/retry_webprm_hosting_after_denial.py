#!/usr/bin/env python3
"""Unfrozen retry adapter for exactly approved GitLab/Map frozen hosting owners.

Default is a read-only preview. Frozen modules and bytes remain unchanged. Only
submission_lineage is replaced, retaining accepted-job accounting and admitting
the strict independently reconciled non-job sidecars. A real retry requires an
explicit --submit and remains subject to normal host budget/permission controls.
Run in a fresh Python process; never use this to poll or evade an account denial.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[1]
N=Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation')
PACKAGES={
    'gitlab':dict(root=N/'gitlab-hosting-validation-v3',digest='718403bfb5334572a440a0688f960c9632f8aa605049a6c2ec54c191b785d52a',
                  source='frozen-source',owner='scripts.validate_webprm_gitlab_hosting'),
    'map':dict(root=N/'map-hosting-validation-v2',digest='3c027a93b35912f2e1d246895cc448f5b3ab956e6967165ec1cd3aa9f5ecf225',
               source='source',owner='scripts.validate_webprm_map_hosting'),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(path):
    path=Path(path)
    if not path.is_absolute() or path.resolve()!=path or any(p.is_symlink() for p in (path,*path.parents)):
        raise ValueError('Canonical nonsymlink frozen package paths required')
    return path


def load(site):
    package=PACKAGES[site];root=canonical(package['root']);frozen=canonical(root/package['source'])
    path=root/'proposal.json';proposal=json.loads(path.read_text())
    digest=hashlib.sha256(json.dumps(proposal,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if digest!=package['digest'] or proposal['frozen_source']!=str(frozen):
        raise ValueError('Exact approved hosting package changed')
    for name,expected in proposal['source_sha256'].items():
        file=canonical(frozen/name)
        if frozen not in file.parents or sha(file)!=expected:raise ValueError('Frozen hosting source changed')
    for name,module in tuple(sys.modules.items()):
        if name in ('openwebrl','scripts') or name.startswith(('openwebrl.','scripts.')):
            origin=getattr(module,'__file__',None)
            if origin and frozen not in Path(origin).resolve().parents:
                raise ValueError('Run retry adapter in a fresh process without checkout project modules')
    sys.path.insert(0,str(frozen))
    owner=importlib.import_module(package['owner'])
    submitter=importlib.import_module('scripts.submit_webprm_cms_hosting')
    h=submitter.h
    approval=root/'approved-request.json'
    if site=='map':owner.approved(proposal,h.read(approval))
    else:h.validate(proposal,h.read(approval))
    for module in (owner,submitter,h):
        if frozen not in canonical(module.__file__).parents:raise ValueError('Owner/submission modules escaped frozen source')
    helper=REPO/'openwebrl/webprm_submission_denial.py'
    spec=importlib.util.spec_from_file_location('openwebrl.webprm_submission_denial',helper)
    denial=importlib.util.module_from_spec(spec);spec.loader.exec_module(denial)
    if denial.h is not h:raise ValueError('Denial helper does not share the frozen accounting module')
    def lineage(campaign,rows):
        campaign=h.private_path(campaign);registered={row['job_id'] for row in rows};references={}
        for receipt in campaign.glob('submission-[0-9]*.json'):
            row=h.read(receipt);intent=h.reference(row['intent'])
            if (intent.parent!=campaign or not intent.name.startswith('submission-intent-') or
                    row.get('job_id') not in registered or str(intent) in references):
                raise h.Closed('Unregistered/ambiguous held submission needs independent terminal accounting')
            references[str(intent)]=row['job_id']
        denied=set()
        for receipt in campaign.glob('submission-denial-*.json'):
            row=h.read(receipt);intent=h.reference(row['intent'])
            if intent.parent!=campaign or str(intent) in references or str(intent) in denied:
                raise h.Closed('Denial cannot erase a registered or duplicate submission intent')
            denial.verified_denial(campaign,intent,receipt);denied.add(str(intent))
        if {str(p) for p in campaign.glob('submission-intent-*.json')}!=set(references)|denied:
            raise h.Closed('Unresolved submission intent needs independent scheduler reconciliation')
    # The sole replacement; workers, scientific protocol and accepted allocation
    # registration/deadline/closure code stay the unchanged frozen implementation.
    submitter.submission_lineage=lineage
    metadata=dict(site=site,proposal_sha256=digest,frozen_source=str(frozen),
        adapter_path=str(Path(__file__).resolve()),adapter_sha256=sha(__file__),
        denial_helper_path=str(helper),denial_helper_sha256=sha(helper),
        replacement='submission_lineage only; strict independently reconciled non-job sidecars')
    return submitter,path,approval,metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--site',choices=PACKAGES,required=True)
    parser.add_argument('--submit',action='store_true');args=parser.parse_args()
    submitter,proposal,approval,metadata=load(args.site)
    preview=submitter.preview(proposal,approval)[2]
    if not args.submit:
        print(json.dumps(dict(status='read_only_retry_preview',adapter=metadata,preview=preview),indent=2));return
    # A durable adapter receipt identifies the unfrozen operational overlay before
    # the unchanged frozen submitter writes a new held/register/release intent.
    import secrets
    root=Path(submitter.h.read(proposal)['campaign_root'])
    submitter.h.put(root/('retry-adapter-'+secrets.token_hex(12)+'.json'),dict(adapter=metadata,preview=preview))
    print(json.dumps(dict(adapter=metadata,result=submitter.submit(proposal,approval)),indent=2))


if __name__=='__main__':main()
