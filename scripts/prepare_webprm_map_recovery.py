#!/usr/bin/env python3
"""Freeze the explicit Map resumed-A/fresh-B recovery offline; never submit."""
import argparse
import ast
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import resume_webprm_map_hosting as owner
from scripts.validate_webprm_cms_hosting import ref,bytes_new
h=owner.h


def accounting_patch(source):
    """Patch only a NEW frozen copy; preserve defaults for every old owner."""
    start='def own(target, args, *, attempt, scratch, campaign_root, deadline, prior_scratch_bytes=0, profile=CMS):'
    if source.count(start)!=1:raise h.Closed('Reviewed own() source shape changed')
    source=source.replace(start,start.replace('profile=CMS','prior_scratch_paths=(), profile=CMS'))
    anchor='    previous=_subreaper(True);baseline=set(_processes());tree=process=None;stopped=[False];reason=None'
    helper='''    prior_scratch_paths=tuple(private_path(p) for p in prior_scratch_paths)
    if any(p==Path(scratch) or p in Path(scratch).parents or Path(scratch) in p.parents for p in prior_scratch_paths):
        raise Closed('Old/new scratch must be disjoint')
    def storage_measure():
        physical=apparent(scratch,maximum_entries=profile.apparent_entries)
        old_live=sum(apparent(p,maximum_entries=profile.apparent_entries) for p in prior_scratch_paths)
        growth=max(0,old_live-prior_scratch_bytes) if prior_scratch_paths else 0
        return dict(scratch_bytes=physical+growth,physical_scratch_bytes=physical,
                    prior_live_scratch_bytes=old_live,prior_scratch_bytes=prior_scratch_bytes,
                    total_charged_bytes=prior_scratch_bytes+physical+growth)
'''
    if source.count(anchor)!=1:raise h.Closed('Reviewed owner setup shape changed')
    source=source.replace(anchor,helper+anchor)
    source=source.replace('scratch_used=apparent(scratch,maximum_entries=profile.apparent_entries);retained=apparent(campaign_root)',
                          "scratch_used=storage_measure()['scratch_bytes'];retained=apparent(campaign_root)")
    old="""put(Path(attempt)/'storage-final.json',dict(scratch_bytes=apparent(scratch,maximum_entries=profile.apparent_entries),retained_bytes=apparent(campaign_root),
                prior_scratch_bytes=prior_scratch_bytes,measurement=f'Final apparent lstat sum; runtime guard samples every {profile.sample_seconds} seconds, not a kernel quota.'))"""
    new="""put(Path(attempt)/'storage-final.json',dict(storage_measure(),retained_bytes=apparent(campaign_root),
                measurement=f'New apparent bytes plus positive old-root growth, old charge counted once without refunds; samples every {profile.sample_seconds} seconds, not a kernel quota.'))"""
    if source.count(old)!=1:raise h.Closed('Reviewed final accounting shape changed')
    source=source.replace(old,new)
    anchor="        if not running:\n            if row['job_state']"
    replacement="""        if profile.protocol=='map_resumed_a_fresh_b_owned_validation_v1':
            if (not running and row.get('required_nodes')!='g007') or (running and row.get('nodes')!='g007'):
                raise ValueError('Retained Map node must remain g007')
        if not running:
            if row['job_state']"""
    if source.count(anchor)!=1:raise h.Closed('Reviewed native scheduler shape changed')
    source=source.replace(anchor,replacement)
    ast.parse(source)
    return source


def prepare(root):
    root=h.private_path(root);plan=h.read(root/'recovery-plan.json')
    if (root/'proposal.json').exists():raise h.Closed('Do not overwrite frozen recovery proposal')
    predecessor=owner.old.N/'map-hosting-validation-v4-vlm/proposal.json';old=h.read(predecessor);oldsource=Path(old['frozen_source'])
    for name,sha in old['source_sha256'].items():
        if h.sha(oldsource/name)!=sha:raise h.Closed('Predecessor frozen source changed')
    source=root/'source';source.mkdir(mode=0o700)
    for name in old['source_sha256']:bytes_new(source/name,(oldsource/name).read_bytes())
    # The copied native recipe and evaluator remain unchanged except a new
    # empty-origins assertion before first browser navigation.
    recipe=source/'scripts/webprm_map_recipe.py';text=recipe.read_text()
    anchor="        rep['initial_cookie_count']=len(c.cookies())"
    if text.count(anchor)!=1:raise h.Closed('Reviewed native browser source changed')
    recipe.write_text(text.replace(anchor,anchor+"\n        rep['initial_origin_count']=len(c.storage_state()['origins'])"))
    module=source/'openwebrl/webprm_cms_hosting.py';module.write_text(accounting_patch(module.read_text()))
    for name in ('scripts/resume_webprm_map_hosting.py','scripts/prepare_webprm_map_recovery.py','tests/test_resume_webprm_map_hosting.py'):
        bytes_new(source/name,(owner.REPO/name).read_bytes())
    service=dict(protocol=owner.PROTOCOL,planned_configs=dict(recipe=ref(recipe),recovery_plan=ref(root/'recovery-plan.json')),
                 recovery_phases=plan['stages'],science=plan['science'],storage=plan['storage'],
                 production_reset='Unchanged original base plus reviewed control recipe only; validation A/B never reused as production baseline.')
    bytes_new(root/'service-plan.json',(json.dumps(service,sort_keys=True,indent=2)+'\n').encode())
    batch='\n'.join(['#!/bin/bash','#SBATCH --account=vlm','#SBATCH --partition=gpu-h200','#SBATCH --qos=normal',
        '#SBATCH --nodes=1','#SBATCH --nodelist=g007','#SBATCH --ntasks=1','#SBATCH --cpus-per-task=8','#SBATCH --mem=128G',
        '#SBATCH --time=226','#SBATCH --job-name=webprm-map-recovery','#SBATCH --output='+str(root/'slurm-%j.log'),
        'set -euo pipefail','export TMPDIR=/tmp PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 BASH_ENV=/dev/null ENV=/dev/null',
        'exec '+owner.old.common.PYTHON+' -B '+str(source/'scripts/resume_webprm_map_hosting.py')+' --proposal '+str(root/'proposal.json')+' --approval "$1" --run',''])
    bytes_new(source/'hosting.sbatch',batch.encode())
    proposal=dict(old)
    proposal.update(protocol=owner.PROTOCOL,recovery_protocol=owner.PROTOCOL,resources=owner.RESOURCES,limits=owner.LIMITS,
        node=owner.NODE,previous_job_id=owner.OLD_JOB,previous_elapsed_seconds=839,next_requested_seconds=13560,prior_scratch_bytes=owner.PRIOR_BYTES,
        allocation_exception={'additional_allocations':1,'previous_maximum_allocations':1},old_state=plan['old_state'],
        service_plan=ref(root/'service-plan.json'),frozen_source=str(source),batch_script=ref(source/'hosting.sbatch'),
        predecessor_proposal=ref(predecessor),status='prepared_not_approved',
        source_sha256={name:h.sha(source/name) for name in sorted(set(old['source_sha256'])|{'scripts/resume_webprm_map_hosting.py','scripts/prepare_webprm_map_recovery.py','tests/test_resume_webprm_map_hosting.py'})})
    bytes_new(root/'proposal.json',(json.dumps(proposal,sort_keys=True,indent=2)+'\n').encode())
    return dict(status='prepared_not_approved',proposal=ref(root/'proposal.json'),proposal_sha256=h.digest(proposal),allocations_submitted=0)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    print(json.dumps(prepare(parser.parse_args().root),indent=2))
