#!/usr/bin/env python3
"""Offline Map two-fresh-state retry with explicit allocation/storage exceptions."""
import argparse
import ast
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import validate_webprm_map_hosting as owner
from scripts.validate_webprm_cms_hosting import ref,bytes_new
h=owner.h
RESOURCES=dict(owner.RESOURCES,account='vlm',max_allocations=2)
LIMITS=dict(owner.LIMITS,scratch_bytes=640*2**30)
PRIOR_BYTES=212292624932


def replace_once(text,before,after):
    if text.count(before)!=1:raise h.Closed('Reviewed frozen source shape changed: '+before[:70])
    return text.replace(before,after)


def patched_owner(text):
    text=replace_once(text,'memory_gib=128,total_seconds=14400,max_allocations=1)',
                           'memory_gib=128,total_seconds=14400,max_allocations=2)')
    text=replace_once(text,'scratch_bytes=384*2**30','scratch_bytes=640*2**30')
    text=replace_once(text,'from openwebrl.webprm_map_controls import web_database_environment',
                           'from openwebrl.webprm_map_controls import web_database_environment,private_hosts')
    text=replace_once(text,"hosts='127.0.0.1 localhost\\n::1 localhost ip6-localhost ip6-loopback\\n'",'hosts=private_hosts(os.uname().nodename)')
    text=replace_once(text,"    p=profile(proposal)\n    h.validate(proposal,approval)","""    p=profile(proposal)
    h.validate(proposal,approval)
    if (proposal.get('fresh_recovery_exception')!={'previous_job_id':'362807','previous_elapsed_seconds':839,'additional_allocations':1,'previous_scratch_bytes':384*2**30,'new_cumulative_scratch_bytes':640*2**30} or
            approval.get('allocation_and_scratch_exception_approved') is not True):
        raise h.Closed('Exact extra-slot and cumulative 640GiB scratch exception required')
    h.reference(proposal['recovery_predecessor'])
    for reference in proposal['failure_evidence'].values():h.reference(reference)""")
    text=replace_once(text,"if 3*size+LIMITS['runtime_reserve_bytes']>LIMITS['scratch_bytes']:",
                           "if context['prior_scratch_bytes']+3*size+LIMITS['runtime_reserve_bytes']>LIMITS['scratch_bytes']:")
    text=replace_once(text,"raise h.Closed('Three independent Map copies exceed scratch cap')",
                           "raise h.Closed('Prior scratch plus three independent Map copies exceed cumulative scratch cap')")
    text=replace_once(text,"scratch=str(scratch),attempt=str(attempt))",
                           "scratch=str(scratch),attempt=str(attempt),prior_scratch_bytes=budget['prior_scratch_bytes'])")
    text=replace_once(text,"if shutil.disk_usage(scratch).free<LIMITS['scratch_bytes']:",
                           "if shutil.disk_usage(scratch).free<LIMITS['scratch_bytes']-budget['prior_scratch_bytes']:")
    text=replace_once(text,"h.own(worker,(path,),attempt=attempt,scratch=scratch,campaign_root=root,deadline=clock['work_deadline'],profile=p)",
                           "h.own(worker,(path,),attempt=attempt,scratch=scratch,campaign_root=root,deadline=clock['work_deadline'],profile=p,prior_scratch_bytes=budget['prior_scratch_bytes'])")
    ast.parse(text)
    return text


def prepare(root):
    root=h.private_path(root);root.mkdir(mode=0o700,exist_ok=False)
    predecessor=owner.N/'map-hosting-validation-v4-vlm/proposal.json';old=h.read(predecessor);oldsource=Path(old['frozen_source'])
    source=root/'source';source.mkdir(mode=0o700)
    for name,sha in old['source_sha256'].items():
        if h.sha(oldsource/name)!=sha:raise h.Closed('Original frozen source changed')
        bytes_new(source/name,(oldsource/name).read_bytes())
    module=source/'scripts/validate_webprm_map_hosting.py';module.write_text(patched_owner(module.read_text()))
    # This helper contains the proven private nodename mapping only; the native
    # recipe/evaluator and fresh-state reset gate retain their frozen bytes.
    controls='openwebrl/webprm_map_controls.py';(source/controls).write_bytes((owner.REPO/controls).read_bytes())
    names=('scripts/prepare_webprm_map_fresh_recovery.py','tests/test_prepare_webprm_map_fresh_recovery.py')
    for name in names:bytes_new(source/name,(owner.REPO/name).read_bytes())
    oldattempt=Path(old['campaign_root'])/'attempt-362807'
    evidence=dict(closure=ref(oldattempt/'closure.json'),storage=ref(oldattempt/'storage-final.json'),
                  original_failure=ref(predecessor.parent/'normal-owner-stop-request-362807.json'),
                  hostname_diagnostic=ref(predecessor.parent/'hostname-repair-native-diagnostic-362807-v2.json'))
    plan=dict(protocol=owner.PROTOCOL,planned_configs=dict(recipe=ref(source/'scripts/webprm_map_recipe.py'),hostname_helper=ref(source/controls)),
        scope='Original two independent fresh-state reconstruction protocol; fresh source materialization, no dependency on old node-local paths.',
        repair='Map hostname loopback mapping matches actual nodename; preserve source, native controls and original two-fresh-state reset verifier.',
        recovery_predecessor=ref(predecessor),failure_evidence=evidence,
        accounting=dict(prior_seconds=839,prior_scratch_bytes=PRIOR_BYTES,next_minutes=226,next_seconds=13560,total_seconds=14400,
                        old_charge_preserved_if_old_tmp_absent=True,prior_plus_three_copies_plus_16GiB_admission=True),
        exceptions=dict(additional_allocations=1,previous_max_allocations=1,new_max_allocations=2,
                        previous_cumulative_scratch_bytes=384*2**30,new_cumulative_scratch_bytes=640*2**30),
        old_state_policy='No old state writes or disposal; original failed attempt, source and evidence unchanged.',
        science=dict(original_state_equivalence_verified=False,browse_equivalence_supported=False,
                     frontend_database='released_fresh_schema_reconstruction',PRM_labels_collected=0,
                     validation_states_are_production_baselines=False),
        resources=RESOURCES,limits=LIMITS)
    bytes_new(root/'service-plan.json',(json.dumps(plan,sort_keys=True,indent=2)+'\n').encode())
    batch='\n'.join(['#!/bin/bash','#SBATCH --account=vlm','#SBATCH --partition=gpu-h200','#SBATCH --qos=normal',
        '#SBATCH --nodes=1','#SBATCH --ntasks=1','#SBATCH --cpus-per-task=8','#SBATCH --mem=128G','#SBATCH --time=226',
        '#SBATCH --job-name=webprm-map-fresh-recovery','#SBATCH --output='+str(root/'slurm-%j.log'),'set -euo pipefail',
        'export TMPDIR=/tmp PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 BASH_ENV=/dev/null ENV=/dev/null',
        'exec '+owner.common.PYTHON+' -B '+str(source/'scripts/validate_webprm_map_hosting.py')+' --proposal '+str(root/'proposal.json')+' --approval "$1" --run',''])
    bytes_new(source/'hosting.sbatch',batch.encode())
    proposal=dict(old)
    proposal.update(resources=RESOURCES,limits=LIMITS,frozen_source=str(source),batch_script=ref(source/'hosting.sbatch'),service_plan=ref(root/'service-plan.json'),
        recovery_predecessor=ref(predecessor),failure_evidence=evidence,
        fresh_recovery_exception=dict(previous_job_id='362807',previous_elapsed_seconds=839,additional_allocations=1,previous_scratch_bytes=384*2**30,new_cumulative_scratch_bytes=640*2**30),
        source_sha256={name:h.sha(source/name) for name in sorted(set(old['source_sha256'])|set(names))},status='prepared_not_approved')
    bytes_new(root/'proposal.json',(json.dumps(proposal,sort_keys=True,indent=2)+'\n').encode())
    return dict(status='prepared_not_approved',proposal=ref(root/'proposal.json'),proposal_sha256=h.digest(proposal),allocations_submitted=0)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    print(json.dumps(prepare(parser.parse_args().root),indent=2))
