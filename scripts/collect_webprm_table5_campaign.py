#!/usr/bin/env python3
"""Read-only, hash-bound Table 5 collection with explicit additive audit lineage.

Never reruns an episode/auditor or infers verification from cell-outcome.json.
The existing independent audit is trusted only after its saved inputs, source,
normalized record and immutable primary artifacts are rebound. This is a
reporting consistency check, not a new native replay or website-truth audit.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.prepare_webprm_table5_campaign import TASKS,TASKS_SHA,cohort as make_cohort
from openwebrl.webprm_table5_analysis import aggregate_table5,digest,validate_cohort,_validate_records,_counts,_rates

AUDIT='table5_independent_cell_audit_v1'
ERRATUM='table5_empty_dispatch_audit_erratum_v1'
INTERRUPTION='table5_saved_interrupted_supervisor_audit_v1'
INTERRUPTION_IMPLEMENTATION='3e5c55851f007138113d1980df9c901b5c5e57e7987a5a6d61e7a54e9b2c585c'
ORIGINAL='02622744b726cb582219f149df3cd7689cdae4eff13dce5b85e048be0817b59d'
CORRECTED='c68a87879a910014b29f60267e6203c5392638226acd8a50dec210bea24f56da'
IMPLEMENTATION='c1b46c5e4e058ea36f1c990730a5ed6d0fca9a7481daff619005b0f0071c67db'
REJECTION=[dict(type='ValueError',message='Native dispatch differs from selected action')]
V10_INTERRUPTION='table5_saved_v10_interruption_disposition_candidate_v1'
V10_IMPLEMENTATION='0d85abc7239794e26e59d1c1c9df2ddc015748df9def53f7a4ddb4675f0ffd9c'
PRE_ACTOR='table5_pre_actor_readiness_failure_audit_v1'
PRE_ACTOR_IMPLEMENTATION='ef7b6166942da66667d47037808724448122995e43493fc3f398cddd76d23628'
PREAUTH='table5_saved_cms_preauth_interruption_v1'
PREAUTH_IMPLEMENTATION='2fa7fb2909f4c9628d7ba185a1f014c4e0ff6d4795d605d52652c4b8c6b0ab73'
UNICODE_ERRATUM='table5_unicode_task_identity_erratum_v1'
UNICODE_SOURCE='5e0ad65e74aaea50a5b10f2c2a9257b0620b17270a5d192b6c878ad8e23c8f07'
UNICODE_RUNNER='5eda3c453e956403a61c073853e8525cb17108fa6ac06a86f111d5cc7fbf3688'
CLOSED_WATCHDOG='table5_closed_cms_watchdog_invalid_audit_v1'
CLOSED_WATCHDOG_IMPLEMENTATION='ab61c4c82938272f2a506ba497bdbf39dcdea905b73dd09294fb94d9c2acf91a'
SHA=re.compile('[0-9a-f]{64}')


def require(condition,message):
    if not condition:raise ValueError(message)


def canonical(value):
    path=Path(value)
    roots=(Path('/tmp'),Path('/gpfs/scrubbed/zixianma/openwebrl-runtime'))
    require(path.is_absolute() and path.resolve()==path and any(r in path.parents for r in roots)
        and not any(p.is_symlink() for p in (path,*path.parents)),'Canonical private runtime evidence required')
    return path


class Evidence:
    def __init__(self,seconds=120):
        require(type(seconds) is int and 1<=seconds<=3600,'Bounded offline collection deadline required')
        self.deadline=time.monotonic()+seconds;self.cache={};self.prefixes=set();self.bytes=0

    def fingerprint(self,path):
        value=path.stat()
        return (value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns)

    def reference(self,value,root=None):
        require(isinstance(value,dict) and set(value)=={'path','sha256'} and
                SHA.fullmatch(str(value['sha256'])),'Exact evidence path/SHA256 required')
        path=canonical(value['path'])
        require(root is None or root in path.parents,'Evidence escaped its cell/campaign')
        require(time.monotonic()<self.deadline and path.is_file(),'Missing evidence or collection deadline')
        before=self.fingerprint(path);require(before[2]<=512*2**20,'Evidence file exceeds bounded collector size')
        cached=self.cache.get(str(path))
        if cached is not None:
            require(cached==(before,value['sha256']),'Evidence changed during collection')
            return path
        checksum=hashlib.sha256()
        with path.open('rb') as stream:
            while True:
                require(time.monotonic()<self.deadline,'Collection deadline exhausted')
                block=stream.read(2**20)
                if not block:break
                self.bytes+=len(block);require(self.bytes<=2**40,'Collection byte bound exceeded')
                checksum.update(block)
        require(before==self.fingerprint(path) and checksum.hexdigest()==value['sha256'],'Evidence reference changed')
        self.cache[str(path)]=(before,value['sha256']);return path

    def ref(self,path):
        path=canonical(path);require(path.is_file() and path.stat().st_size<=32*2**20,'Absent/oversized reporting JSON')
        before=self.fingerprint(path);checksum=hashlib.sha256(path.read_bytes()).hexdigest()
        require(before==self.fingerprint(path),'Reporting JSON changed while reading')
        value=dict(path=str(path),sha256=checksum);self.reference(value);return value

    def read(self,path):
        ref=self.ref(path);return json.loads(Path(ref['path']).read_text())

    def read_ref(self,ref,root=None):
        return self.read(self.reference(ref,root))

    def unchanged(self):
        for name,(before,_) in self.cache.items():
            require(self.fingerprint(Path(name))==before,'Previously collected evidence changed')

    def prefix(self,path,size,sha,root):
        path=canonical(path)
        require(root in path.parents and type(size) is int and 0<=size<=512*2**20 and
            SHA.fullmatch(str(sha)) and path.is_file() and path.stat().st_size>=size,'Invalid saved journal prefix')
        key=(str(path),size,sha)
        if key in self.prefixes:return
        checksum=hashlib.sha256();remaining=size
        with path.open('rb') as stream:
            while remaining:
                require(time.monotonic()<self.deadline,'Collection deadline exhausted')
                block=stream.read(min(2**20,remaining));require(block,'Saved journal prefix truncated')
                checksum.update(block);remaining-=len(block);self.bytes+=len(block)
        require(checksum.hexdigest()==sha,'Saved accounting/broker prefix changed')
        self.prefixes.add(key)


def record_binding(evidence,folder,cell,cohort,campaign,audit_path,record_path,original,extra):
    inputs=evidence.read(folder/'audit-input.json')
    require(inputs.get('schema_version')==1 and inputs.get('campaign_root')==str(campaign)
        and inputs.get('cell')==cell,'Audit input belongs to another primary cell')
    bound=evidence.read_ref(inputs['cohort']);bound=bound.get('cohort',bound)
    require(bound==cohort,'Audit cohort differs from fixed reporting cohort')
    plan=evidence.read_ref(inputs['worker_plan'],folder)
    require(inputs['worker_plan']['path']==str(folder/'worker-plan.json') and
        digest(plan.get('protocol_contract'))==cohort['protocol_sha256'] and
        plan.get('task_ids')==[cell['task_id']] and plan.get('actor_model')==cell['actor'] and
        plan.get('selector')==cell['selector'],'Worker task/actor/selector/protocol changed')
    sources=inputs.get('source_sha256')
    code=plan.get('code_sha256')
    require(isinstance(sources,dict) and isinstance(code,dict) and code and
        all(sources.get(name)==sha for name,sha in code.items()),'Source manifest differs from worker plan')
    for name,sha in sources.items():evidence.reference(dict(path=name,sha256=sha))
    auditors=[sha for name,sha in sources.items() if name.endswith('/scripts/audit_webprm_table5_cell.py')]
    require(len(auditors)==1 and auditors[0] in (ORIGINAL,CORRECTED,UNICODE_SOURCE),'Unreviewed original cell auditor source')
    audit=evidence.read(audit_path);refs=inputs['artifacts']
    require(audit.get('schema_version')==1 and audit.get('audit_verified') is True and audit.get('errors')==[]
        and audit.get('input_sha256')==evidence.ref(folder/'audit-input.json')['sha256']
        and audit.get('artifact_refs')==refs and audit.get('evidence_manifest_sha256')==digest(refs),
        'Verified audit is not bound to the exact primary evidence')
    paths={name:evidence.reference(value,campaign) for name,value in refs.items()}
    require({'result','watchdog','api','site_closure','closure'}<=paths.keys(),'Missing primary/closure evidence')
    result=evidence.read(paths['result']);record=evidence.read(record_path)
    calls=audit.get('call_audit',{})
    require(calls.get('verified') is True and type(calls.get('cell_unknown_reservations')) is int and
        calls['cell_unknown_reservations']>=0,
        'Independent metered-call audit is incomplete')
    for key,path_key,prefix_name in (('budget','event_path','budget'),('broker','audit_path','broker')):
        boundary=inputs[key]
        evidence.prefix(boundary[path_key],boundary['prefix_bytes'],boundary['prefix_sha256'],campaign)
        require(calls.get(prefix_name+'_prefix_bytes')==boundary['prefix_bytes'] and
            calls.get(prefix_name+'_prefix_sha256')==boundary['prefix_sha256'],'Audited accounting prefix binding differs')
    expected=dict(audit['record'],audit_sha256=evidence.ref(audit_path)['sha256'],**extra)
    require(record==expected,'Normalized record differs from its verified audit body')
    _validate_records(cohort,{cell['cell_id']:cell},[record])
    require(result.get('identity_sha256')==digest(plan) and result.get('actor_model')==cell['actor'] and
        result.get('selector')==cell['selector'] and result.get('task_id')==f"webarena_lite_{cell['task_id']:03d}" and
        result.get('attempted') is record['attempted'] and result.get('valid') is record['valid'],
        'Normalized outcome differs from the primary worker identity/disposition')
    if record['valid']:
        require({'verdict','outcome','trajectory','rollout','evaluator_inputs'}<=paths.keys(),'Valid record missing native evidence')
        verdict=evidence.read(paths['verdict']);before_close=evidence.read(paths['outcome'])
        require(verdict.get('valid') is True and verdict.get('success') in (True,False) and
            int(verdict['success'])==record['success'] and
            int(verdict.get('native_success',verdict['success']))==record['native_success'] and
            verdict.get('score')==record['success'],'Native/effective primary verdict was changed')
        require(audit.get('native_replay',{}).get('verified') is True and record['effective_semantic_status']=='valid' and
            record['native_semantic_status']==('invalid' if verdict.get('scoring_amendments') and record['native_success'] else 'valid'),
            'Native replay or effective/raw semantic classification differs')
        for key in ('success','score','native_success','native_score','scoring_amendments','evaluator_protocol'):
            if key in verdict:
                require(result.get(key)==verdict[key] and before_close.get(key)==verdict[key],
                    'Before-close and final native/effective verdicts differ')
    outcome=evidence.read(folder/'cell-outcome.json')
    require(outcome.get('cell_id')==cell['cell_id'] and outcome.get('result_path')==str(paths['result']) and
        outcome.get('result_sha256')==refs['result']['sha256'] and
        outcome.get('attempted') is record['attempted'] and outcome.get('valid') is record['valid'] and
        (not record['valid'] or outcome.get('success')==record['success']),
        'Mechanical outcome is not bound to the same primary result')
    # No trust is placed in outcome.audit_verified: that mechanical flag stays
    # false even when the later independent normalized audit is verified.
    episode=paths['result'].parent;covered=set(paths.values())
    for path in episode.rglob('*'):
        if path.is_file() and path.name!='storage-state-private.json':
            require(canonical(path) in covered,'Primary episode file omitted from immutable audit evidence')
    require(original.get('input_sha256')==audit['input_sha256'] and original.get('artifact_refs')==refs and
        original.get('evidence_manifest_sha256')==audit['evidence_manifest_sha256'],
        'Replacement changed original primary input/evidence identity')
    return record,audit,inputs,auditors[0]


def replacement(evidence,pointer,folder,cell,original):
    value=evidence.read(pointer)
    if value.get('protocol')==UNICODE_ERRATUM:
        return unicode_replacement(evidence,value,folder,cell,original)
    require(value.get('cell_id')==cell['cell_id'] and value.get('protocol')==ERRATUM and
        value.get('original_bytes_preserved') is True and value.get('primary_rerun') is False,
        'Unrecognized or resampled audit replacement')
    for name in ('original_audit','original_outcome','effective_audit','effective_record'):
        evidence.reference(value[name],folder)
    require(value['original_audit']==evidence.ref(folder/'audit.json') and
        value['original_outcome']==evidence.ref(folder/'cell-outcome.json'),
        'Replacement does not preserve the exact original rejection/outcome')
    require(original.get('audit_verified') is False and original.get('errors')==REJECTION and
        original.get('complete_for_disposal') is False,'Replacement is not the reviewed original rejection')
    audit_path=Path(value['effective_audit']['path']);record_path=Path(value['effective_record']['path'])
    require(record_path==audit_path.with_suffix('.record.json') and audit_path!=folder/'audit.json',
        'Replacement must be additive with its exact normalized sibling')
    if 'independent_audit' in value:
        evidence.reference(value['independent_audit'],folder)
        require(value['independent_audit']['sha256']==value['effective_audit']['sha256'],
            'Independent replacement replay differs from selected root audit')
    audit=evidence.read(audit_path);erratum=audit.get('erratum',{})
    require(audit.get('protocol')==ERRATUM and audit.get('original_audit_reference')==value['original_audit'] and
        audit.get('original_audit_verified') is False and audit.get('original_rejection_reproduced') is True and
        erratum.get('original_auditor_sha256')==ORIGINAL and erratum.get('primary_bytes_changed') is False and
        erratum.get('primary_rerun') is False and erratum.get('replaced_functions')==['audit_empty_native_dispatch','audit_rollout'],
        'Unreviewed audit erratum scope')
    for key,sha in (('corrected_source_reference',CORRECTED),('implementation_reference',IMPLEMENTATION)):
        require(erratum.get(key,{}).get('sha256')==sha,'Erratum source is not the reviewed implementation')
        evidence.reference(erratum[key])
    proofs=audit.get('reconstructed_rejections')
    require(isinstance(proofs,list) and proofs and all(r.get('native_conversion_rejection_reproduced') is True for r in proofs),
        'Missing reconstructed native conversion rejection')
    return audit_path,record_path,dict(audit_erratum_protocol=ERRATUM,
        original_audit_sha256=value['original_audit']['sha256'],corrected_audit_source_sha256=CORRECTED),value


def unicode_replacement(evidence,value,folder,cell,original):
    """Accept only the reviewed Unicode identity correction over unchanged evidence."""
    campaign=folder.parents[2]
    require(value.get('cell_id')==cell['cell_id'] and value.get('original_bytes_preserved') is True
        and value.get('primary_rerun') is False and value.get('disposal_performed') is False,
        'Unicode replacement changed primary evidence or state')
    require(original.get('audit_verified') is False and original.get('errors')==[
        dict(type='ValueError',message='Released task identity differs')],
        'Unicode replacement belongs to another rejection')
    require(value.get('original_audit')==evidence.ref(folder/'audit.json') and
        value.get('original_outcome')==evidence.ref(folder/'cell-outcome.json'),
        'Unicode replacement does not preserve original rejection/outcome')
    for key in ('effective_audit','effective_record','independent_audit','activation_review'):
        evidence.reference(value[key],campaign)
    source=evidence.reference(value['corrected_source']);runner=evidence.reference(value['runner'])
    original_source=evidence.reference(value['original_source'])
    require(value['corrected_source']['sha256']==UNICODE_SOURCE and value['runner']['sha256']==UNICODE_RUNNER
        and value['original_source']['sha256']==CORRECTED,'Unreviewed Unicode implementation')
    before=original_source.read_text()
    old='from scripts.prepare_webprm_table5_campaign import TASKS, TASKS_SHA, cohort as make_cohort'
    require(before.count(old)==1 and before.count("if digest(raw) != cell['task_config_sha256']:")==1 and
        source.read_text()==before.replace(old,old+', digest as cohort_digest').replace(
            "if digest(raw) != cell['task_config_sha256']:","if cohort_digest(raw) != cell['task_config_sha256']:"),
        'Unicode erratum changed more than the cohort task hash convention')
    audit=evidence.read_ref(value['effective_audit']);replay=evidence.read_ref(value['independent_audit'])
    review=evidence.read_ref(value['activation_review'])
    require(audit==replay and value['effective_audit']['sha256']==value['independent_audit']['sha256'] and
        value['effective_audit']['path']!=value['independent_audit']['path'] and
        review.get('protocol')=='table5_unicode_identity_activation_review_v1' and review.get('verified') is True and
        review.get('errors')==[] and all(review.get(k)==value[k] for k in ('corrected_source','runner','original_source')) and
        any(item.get('cell_id')==cell['cell_id'] and all(item.get(k)==value[k] for k in
            ('original_audit','original_outcome','effective_audit','effective_record','independent_audit'))
            for item in review.get('cases',[])), 'Unicode replacement lacks exact independent replay/review')
    from scripts.prepare_webprm_table5_campaign import digest as cohort_digest
    released=evidence.read_ref(dict(path=str(TASKS),sha256=TASKS_SHA))
    raw=next(t for t in released if t['task_id']==cell['task_id'])
    from openwebrl.webprm_webarena import digest as worker_digest
    require(cohort_digest(raw)==cell['task_config_sha256'] and worker_digest(raw)!=cohort_digest(raw),
        'Unicode replacement is not a genuine frozen serialization mismatch')
    audit_path=Path(value['effective_audit']['path']);record_path=Path(value['effective_record']['path'])
    require(record_path==audit_path.with_suffix('.record.json') and audit_path!=folder/'audit.json',
        'Unicode audit/record must remain additive siblings')
    return audit_path,record_path,{},value


def retained_state_binding(evidence,value,folder,cell,inputs,campaign):
    """Bind reported retention to saved closure; never inspect/delete node state."""
    retained=value.get('retained_state')
    if retained is None:return None
    require('disposal' not in value and isinstance(retained,dict) and
        retained.get('schema_version')==1 and retained.get('status')=='retained_no_disposal_claim' and
        retained.get('disposal_performed') is False,'Conflicting or unreviewed retained-state disposition')
    attempt=folder.parents[1]
    expected={'state_claim':folder/'state-claim.json','cell_closure':folder/'closure.json',
              'attempt_closure':attempt/'controller-closure.json','attempt_storage':attempt/'storage-final.json'}
    saved={}
    for name,path in expected.items():
        require(retained.get(name)==evidence.ref(path),'Retained-state reference belongs to another claim/attempt')
        saved[name]=evidence.read_ref(retained[name],campaign)
    require(inputs['artifacts'].get('state_claim')==retained['state_claim'] and
        inputs['artifacts'].get('closure')==retained['cell_closure'],'Retained claim/closure differs from audited evidence')
    claim,closure,outer,storage=(saved[k] for k in expected)
    scratch=canonical(claim.get('scratch_root',''))
    require(claim.get('cell_id')==cell['cell_id'] and claim.get('campaign_root')==str(campaign) and
        claim.get('destination_absent_at_claim') is True and
        claim.get('state_path')==str(scratch/'states'/folder.name) and storage.get('scratch')==str(scratch) and
        type(storage.get('apparent_bytes')) is int and storage['apparent_bytes']>0,
        'Retained-state claim/storage identity differs')
    require(closure.get('cell_id')==cell['cell_id'] and closure.get('broker_cell_inflight')==0 and
        closure.get('worker_remaining_processes')=={} and closure.get('service_remaining_processes')=={} and
        outer.get('remaining_owned_processes')=={},'Retained-state claim lacks closed cell/outer ownership')
    return dict(retained,state_path=claim['state_path'],scope='Saved claim and closed-owner retention accounting; '
        'no independent current node-filesystem inspection, disposal, or per-state size certification.')


def pending_disposition(evidence,folder,cell,cohort,campaign):
    """An interrupted claim is not a fabricated normalized invalid episode."""
    value=dict(cell_id=cell['cell_id'],status='pending_terminal_audit')
    attempt=folder.parents[1];stop=campaign/('operational-stop-'+attempt.name.split('-',1)[1]+'.json')
    if not stop.exists():return value
    stopped=evidence.read(stop)
    matches=[r for r in stopped.get('active_cells',[]) if r.get('cell_id')==cell['cell_id']]
    if not matches:return value
    require(matches==[dict(cell_id=cell['cell_id'],folder=str(folder))] and
        stopped.get('job_id')==attempt.name.split('-',1)[1], 'Interrupted claim differs from saved stop intent')
    outer=attempt/'controller-closure.json'
    if not outer.exists():return value
    require(evidence.read(outer).get('remaining_owned_processes')=={},'Interrupted claim outer owner is not closed')
    claim=evidence.read(folder/'state-claim.json');plan=evidence.read(folder/'worker-plan.json')
    require(claim.get('cell_id')==cell['cell_id'] and claim.get('campaign_root')==str(campaign) and
        claim.get('destination_absent_at_claim') is True and plan.get('task_ids')==[cell['task_id']] and
        plan.get('actor_model')==cell['actor'] and plan.get('selector')==cell['selector'] and
        digest(plan.get('protocol_contract'))==cohort['protocol_sha256'],'Interrupted fixed-claim identity differs')
    return dict(value,status='interrupted_missing_terminal_audit',audit_verified=False,
        normalized_outcome_created=False,stop_intent=evidence.ref(stop),attempt_closure=evidence.ref(outer),
        state_claim=evidence.ref(folder/'state-claim.json'),worker_plan=evidence.ref(folder/'worker-plan.json'))


def interruption_binding(evidence,pointer,folder,cell,cohort,campaign):
    """Select a reviewed saved failure, without manufacturing absent verdicts."""
    value=evidence.read(pointer)
    require(value.get('protocol')==INTERRUPTION and value.get('cell_id')==cell['cell_id'] and
        value.get('original_bytes_preserved') is True and value.get('primary_rerun') is False and
        value.get('disposal_performed') is False and value.get('raw_success') is None and
        value.get('effective_success') is None,'Unreviewed interruption pointer or fabricated verdict')
    require(not (folder/'audit.json').exists() and not (folder/'cell-outcome.json').exists(),
        'Interruption supplement cannot replace an existing primary audit/outcome')
    audit=evidence.read_ref(value['effective_audit']);record=evidence.read_ref(value['effective_record'])
    inputs=evidence.read_ref(value['input']);review=evidence.read_ref(value['independent_review'])
    replay=evidence.read_ref(value['independent_audit'])
    require(Path(value['effective_record']['path'])==Path(value['effective_audit']['path']).with_suffix('.record.json') and
        campaign not in Path(value['effective_audit']['path']).parents and
        value['independent_audit']['path']!=value['effective_audit']['path'] and
        value['independent_audit']['sha256']==value['effective_audit']['sha256'] and replay==audit,
        'Independent interrupted replay differs or candidate is not additive')
    require(review.get('protocol')=='table5_interruption_independent_activation_review_v1' and
        review.get('verified') is True and review.get('errors')==[] and review.get('cell_id')==cell['cell_id'] and
        all(review.get(k)==value[k] for k in ('input','effective_audit','effective_record','independent_audit')),
        'Interruption activation lacks exact independent review')
    implementation=value['implementation'];evidence.reference(implementation)
    require(implementation['sha256']==INTERRUPTION_IMPLEMENTATION and
        review.get('implementation')==implementation,'Unreviewed interruption implementation')
    source_root=Path(implementation['path']).parents[1]
    sources=inputs.get('audit_source_sha256',{})
    require(sources.get('scripts/audit_webprm_interrupted_cell.py')==INTERRUPTION_IMPLEMENTATION,
        'Interrupted input names a different auditor')
    for name,sha in sources.items():
        relative=Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts,'Escaping interruption source path')
        evidence.reference(dict(path=str(source_root/relative),sha256=sha),source_root)
    require(inputs.get('schema_version')==1 and inputs.get('protocol')==INTERRUPTION and
        inputs.get('campaign_root')==str(campaign) and inputs.get('cell_folder')==str(folder) and inputs.get('cell')==cell,
        'Interruption input belongs to another immutable primary')
    bound=evidence.read_ref(inputs['cohort']);bound=bound.get('cohort',bound)
    plan=evidence.read_ref(inputs['worker_plan'],folder)
    require(bound==cohort and inputs['worker_plan']['path']==str(folder/'worker-plan.json') and
        plan.get('task_ids')==[cell['task_id']] and plan.get('actor_model')==cell['actor'] and
        plan.get('selector')==cell['selector'] and digest(plan.get('protocol_contract'))==cohort['protocol_sha256'],
        'Interrupted fixed cohort/task/actor/selector/protocol differs')
    for name,sha in plan['code_sha256'].items():evidence.reference(dict(path=name,sha256=sha))
    refs=inputs['artifacts'];paths={k:evidence.reference(v,campaign) for k,v in refs.items()}
    require(audit.get('schema_version')==1 and audit.get('protocol')==INTERRUPTION and
        audit.get('audit_verified') is True and audit.get('errors')==[] and audit.get('input_reference')==value['input'] and
        audit.get('artifact_refs')==refs and audit.get('evidence_manifest_sha256')==digest(refs) and
        audit.get('complete_for_disposal') is False and audit.get('native_replay') is None and
        audit.get('original_result_has_no_native_verdict') is True and audit.get('all_primary_bytes_preserved') is True,
        'Saved interruption audit is not bound to complete preserved evidence')
    require(record==dict(audit['record'],audit_sha256=value['effective_audit']['sha256']) and
        record.get('valid') is False and record.get('attempted') is True and record.get('terminal') is True and
        record.get('success') is None and record.get('native_success') is None,
        'Interrupted normalized record must preserve absent native/effective verdicts')
    _validate_records(cohort,{cell['cell_id']:cell},[record])
    episode=folder/'episodes'/f"webarena_lite_{cell['task_id']:03d}"/(cell['actor']+'__'+cell['selector'])
    require(paths['result']==episode/'result.json' and paths['claim']==folder/'state-claim.json' and
        paths['outer_closure']==folder.parents[1]/'controller-closure.json' and
        paths['queue_claim']==campaign/'queue/claims'/(digest(cell['cell_id'])+'.json') and
        paths['queue_outcome']==campaign/'queue/outcomes'/(digest(cell['cell_id'])+'.json'),
        'Saved interruption evidence belongs to a different primary/queue owner')
    result=evidence.read(paths['result'])
    require(result.get('identity_sha256')==digest(plan) and result.get('attempted') is True and
        result.get('valid') is False and result.get('error_type')=='EpisodeWorkerExited' and
        result.get('error')=='Episode child exited without result (exit -15)' and
        not any(k in result for k in ('native_score','native_success','score','terminal')),
        'Saved failure identity differs or contains a verdict')
    for name in ('watchdog','site_closure','outer_closure'):
        require(evidence.read(paths[name]).get('remaining_owned_processes')=={},'Interrupted process closure differs')
    for path in episode.rglob('*'):
        if path.is_file() and path.name!='storage-state-private.json':
            require(canonical(path) in set(paths.values()),'Interrupted episode file omitted from evidence')
    recovery=evidence.read_ref(inputs['epoch_recovery']);before,after=recovery['before'],recovery['after']
    for name in ('closure_receipt','financial_receipt'):evidence.read_ref(before[name])
    calls=audit['call_audit']
    require(calls.get('verified') is True and calls.get('cell_unknown_reservations')==0 and
        calls.get('global_unknown_reservations')==0 and calls.get('policy_within_caps') is True and
        calls.get('global_charged_usd')==after['charged'] and
        audit.get('reset_and_queue',{}).get('verified') is True and
        audit['reset_and_queue'].get('queue_claim_sha256')==refs['queue_claim']['sha256'] and
        audit['reset_and_queue'].get('queue_outcome_sha256')==refs['queue_outcome']['sha256'],
        'Interrupted reset/queue/financial review is incomplete')
    for label,path,size,sha in [('budget',campaign/'shared-budget.json.events.jsonl',after['budget_prefix_bytes'],after['budget_ledger_sha256']),
            ('broker',before['audit_path'],before['audit_prefix_bytes'],before['audit_sha256'])]:
        evidence.prefix(path,size,sha,campaign)
        require(calls.get(label+'_prefix_bytes')==size and calls.get(label+'_prefix_sha256')==sha,
            'Interrupted financial replay prefix differs')
    return record,dict(cell_id=cell['cell_id'],audit=value['effective_audit'],record=value['effective_record'],
        input=value['input'],original_result=refs['result'],mechanical_queue_outcome=refs['queue_outcome'],
        replacement=True,replacement_pointer=evidence.ref(pointer),independent_review=value['independent_review'],
        unknown_reservations_preserved=0,interrupted_failure=True,disposal_performed=False,
        retained_state_claim=refs['claim'])


def v10_interruption_binding(evidence,pointer,folder,cell,cohort,campaign):
    """Bind independently reviewed closed claims while preserving absent verdicts."""
    value=evidence.read(pointer)
    require(value.get('protocol')==V10_INTERRUPTION and value.get('cell_id')==cell['cell_id'] and
        value.get('original_bytes_preserved') is True and value.get('primary_rerun') is False and
        value.get('disposal_performed') is False and value.get('raw_success') is None and
        value.get('effective_success') is None,'Unreviewed v10 interruption pointer or fabricated verdict')
    require(folder.parents[1].name=='attempt-359482' and not (folder/'audit.json').exists() and
        not (folder/'cell-outcome.json').exists(),'V10 supplement cannot replace original audit/outcome')
    candidate=evidence.read_ref(value['candidate']);replay=evidence.read_ref(value['independent_audit'])
    review=evidence.read_ref(value['independent_review']);record=evidence.read_ref(value['effective_record'])
    implementation=value['implementation'];evidence.reference(implementation)
    require(implementation['sha256']==V10_IMPLEMENTATION and candidate.get('implementation')==implementation and
        candidate==replay and value['candidate']['path']!=value['independent_audit']['path'] and
        value['candidate']['sha256']==value['independent_audit']['sha256'],
        'Independent v10 interruption replay or implementation differs')
    require(review.get('protocol')=='table5_saved_v10_interruption_activation_review_v1' and
        review.get('verified') is True and review.get('errors')==[] and review.get('cell_id')==cell['cell_id'] and
        all(review.get(k)==value[k] for k in ('candidate','effective_record','independent_audit','implementation')),
        'V10 interruption lacks exact independent activation review')
    require(candidate.get('protocol')==V10_INTERRUPTION and len(candidate.get('records',[]))==4,
        'V10 candidate protocol or fixed interruption panel differs')
    bound=evidence.read_ref(candidate['cohort']);bound=bound.get('cohort',bound)
    require(bound==cohort,'V10 interruption cohort differs')
    source_root=Path(implementation['path']).parents[1]
    require(candidate['dependency_sources'].get(implementation['path'])==implementation['sha256'],
        'V10 executing implementation is omitted from source freeze')
    for path,sha in candidate['dependency_sources'].items():evidence.reference(dict(path=path,sha256=sha),source_root)
    matches=[r for r in candidate['records'] if r.get('cell')==cell]
    require(len(matches)==1,'Missing or repeated v10 candidate cell')
    item=matches[0]
    require(item.get('audit_verified') is True and item.get('valid') is False and
        item.get('raw_native_success') is None and item.get('effective_success') is None and
        item.get('native_verdict_created') is False and item.get('normalized_table_record_created') is False and
        item.get('original_result_and_watchdog_unmodified') is True and item.get('primary_rerun') is False and
        item.get('disposal_performed') is False,'V10 candidate changes saved verdict or state')
    refs=item['artifacts'];paths={k:evidence.reference(v,folder) for k,v in refs.items()}
    for name,path in paths.items():require(path==folder/name,'V10 artifact path belongs to another cell')
    require(set(paths)=={str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file() and
        p.name!='storage-state-private.json'},'V10 primary artifact omitted or added')
    require(item['worker_plan']==refs['worker-plan.json'],'V10 worker plan reference differs')
    plan=evidence.read_ref(item['worker_plan'],folder)
    require(plan.get('task_ids')==[cell['task_id']] and plan.get('actor_model')==cell['actor'] and
        plan.get('selector')==cell['selector'] and digest(plan.get('protocol_contract'))==cohort['protocol_sha256'] and
        plan.get('code_sha256')==item['source_sha256'],'V10 worker science/source binding differs')
    for path,sha in item['source_sha256'].items():evidence.reference(dict(path=path,sha256=sha))
    claim=evidence.read(folder/'state-claim.json')
    require(claim.get('cell_id')==cell['cell_id'] and claim.get('campaign_root')==str(campaign) and
        claim.get('destination_absent_at_claim') is True,'V10 state claim identity differs')
    for key,subdir in [('queue_claim','claims'),('queue_outcome','outcomes')]:
        require(item[key]['path']==str(campaign/'queue'/subdir/(digest(cell['cell_id'])+'.json')),
            'V10 queue reference belongs to another cell')
    qclaim=evidence.read_ref(item['queue_claim'],campaign);qout=evidence.read_ref(item['queue_outcome'],campaign)
    require(qclaim.get('cell_id')==cell['cell_id'] and qout.get('cell_id')==cell['cell_id'] and
        qclaim.get('cell_sha256')==digest(cell)==qout.get('cell_sha256') and
        qclaim.get('owner')==qout.get('owner') and qout.get('result')==item['saved_queue_outcome'] and
        qout['result'].get('status')=='invalid' and qout['result'].get('reason')=='worker_exception',
        'V10 fixed queue ownership/outcome differs')
    require(candidate['outer_closure']['path']==str(folder.parents[1]/'controller-closure.json'),
        'V10 outer closure path differs')
    require(candidate['broker_closure']['path']==str(folder.parents[1]/'broker-closure.json'),
        'V10 broker owner closure belongs to another attempt')
    outer=evidence.read_ref(candidate['outer_closure'],campaign);broker=evidence.read_ref(candidate['broker_closure'],campaign)
    site=evidence.read(folder/'site/closure.json');instance=evidence.read(folder/'site/instance.json')
    require(outer.get('remaining_owned_processes')=={} and site.get('remaining_owned_processes')=={} and
        instance.get('service_pid') in site['owned_pids'] and instance['service_pid'] in outer['owned_pids'] and
        broker.get('reaped') is True and broker.get('exitcode')==0 and broker.get('pid') in outer['owned_pids'] and
        item.get('later_outer_closure_complete') is True,'V10 site/broker/outer ownership closure differs')
    episode=folder/'episodes'/f"webarena_lite_{cell['task_id']:03d}"/(cell['actor']+'__'+cell['selector'])
    result_path=episode/'result.json';watchdog_path=episode/'watchdog.json'
    require(item.get('saved_result_present')==result_path.exists() and
        not (episode/'native-evaluation/native-verdict.json').exists(),'V10 result presence or native verdict changed')
    if result_path.exists():
        result=evidence.read(result_path);watchdog=evidence.read(watchdog_path)
        require(result.get('identity_sha256')==digest(plan) and result.get('attempted') is True and
            result.get('valid') is False and result.get('error_type')=='EpisodeWatchdogTimeout' and
            result.get('success')==item['saved_result_success']==0 and result.get('watchdog')==watchdog and
            not any(k in result for k in ('native_score','native_success','score','terminal')),
            'V10 saved supervisor failure identity/verdict differs')
        remaining=watchdog['remaining_owned_processes']
        require(watchdog.get('supervisor_signal')==15 and watchdog.get('reason')=='supervisor_termination' and
            set(watchdog['owned_pids'])<=set(outer['owned_pids']) and
            set(map(int,remaining))<=set(watchdog['owned_pids']) and
            remaining==item['saved_worker_remaining_processes'] and
            (remaining=={}) is item['saved_worker_closure_complete'] and
            watchdog.get('child_exitcode')==item['saved_worker_child_exitcode'],
            'V10 original watchdog/late owner lineage differs')
    else:
        require(not watchdog_path.exists() and item.get('saved_result_success') is None and
            item.get('saved_worker_closure_complete') is None and item.get('saved_worker_remaining_processes') is None and
            item.get('saved_worker_child_exitcode') is None,'V10 absent result/watchdog was fabricated')
    calls=item['financial_audit']
    require(calls.get('verified') is True and calls.get('cell_unknown_reservations')==0 and
        calls.get('global_unknown_reservations')==0 and calls.get('policy_within_caps') is True,
        'V10 independent financial audit is incomplete')
    for label in ('financial','broker'):
        prefix=candidate[label+'_prefix']
        expected_path=campaign/'shared-budget.json.events.jsonl' if label=='financial' else folder.parents[1]/'broker.jsonl'
        require(prefix['path']==str(expected_path),'V10 financial/broker prefix belongs to another owner')
        evidence.prefix(prefix['path'],prefix['bytes'],prefix['sha256'],campaign)
        name='budget' if label=='financial' else label
        require(calls.get(name+'_prefix_bytes')==prefix['bytes'] and calls.get(name+'_prefix_sha256')==prefix['sha256'],
            'V10 financial replay prefix differs')
    prefix=candidate['financial_prefix']
    with Path(prefix['path']).open('rb') as stream:last=json.loads(stream.read(prefix['bytes']).splitlines()[-1])
    require(last.get('event')=='epoch_closed' and last.get('event_sha256')==candidate['financial_epoch_closed'],
        'V10 financial handoff was not a verified closed prefix')
    expected=dict(cell_id=cell['cell_id'],released_task_id=cell['task_id'],actor_model=cell['actor'],selector=cell['selector'],
        domain=cell['domain'],task_config_sha256=cell['task_config_sha256'],tasks_sha256=cohort['source_sha256'],
        cohort_sha256=digest(cohort),protocol_sha256=cohort['protocol_sha256'],audit_verified=True,terminal=True,
        attempted=True,valid=False,success=None,native_success=None,effective_semantic_status='not_applicable',
        native_semantic_status='not_applicable',audit_sha256=value['candidate']['sha256'])
    require(record==expected,'V10 normalized invalid record fabricated a verdict or changed fixed identity')
    _validate_records(cohort,{cell['cell_id']:cell},[record])
    return record,dict(cell_id=cell['cell_id'],audit=value['candidate'],record=value['effective_record'],
        independent_review=value['independent_review'],replacement=True,replacement_pointer=evidence.ref(pointer),
        interrupted_failure=True,unknown_reservations_preserved=0,disposal_performed=False,
        original_result_present=item['saved_result_present'],original_worker_closure_complete=item['saved_worker_closure_complete'],
        later_outer_closure=candidate['outer_closure'],mechanical_queue_outcome=item['queue_outcome'])


def pre_actor_binding(evidence,pointer,folder,cell,cohort,campaign):
    value=evidence.read(pointer)
    require(value.get('protocol')==PRE_ACTOR,'Unrecognized pre-actor review protocol')
    implementation=value['implementation'];path=evidence.reference(implementation)
    require(implementation['sha256']==PRE_ACTOR_IMPLEMENTATION,'Unreviewed pre-actor audit implementation')
    spec=importlib.util.spec_from_file_location('_webprm_pinned_pre_actor_audit',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    inputs=evidence.read_ref(value['input'],campaign)
    audit=evidence.read_ref(value['audit'],campaign)
    record=evidence.read_ref(value['record'],campaign)
    require(inputs.get('implementation')==implementation and audit.get('protocol')==PRE_ACTOR and
        audit.get('audit_verified') is True and audit.get('errors')==[] and
        audit.get('input_reference')==value['input'] and audit.get('implementation')==implementation,
        'Pre-actor report/input/source binding differs')
    # Reconstruct the invalid disposition independently of the saved report's
    # verification flag. Missing actor/verdict evidence must stay absent.
    rebuilt=module.verify(evidence,inputs,cell,cohort,campaign,folder)
    require(audit.get('reconstruction')==rebuilt and
        record==dict(rebuilt['record'],audit_sha256=value['audit']['sha256']),
        'Pre-actor report fabricates a verdict or differs from raw evidence')
    require('activation_review' in value,'Pre-actor candidate lacks activation review')
    review=evidence.read_ref(value['activation_review'],campaign)
    for source_ref in review.get('source_references',[]):
        evidence.reference(source_ref)
    require(review.get('protocol')=='table5_pre_actor_readiness_activation_review_v1' and
        review.get('verified') is True and review.get('errors')==[] and
        review.get('implementation')==implementation and
        review.get('native_verdicts_created')==review.get('state_disposals')==review.get('new_provider_calls')==0 and
        any(item.get('cell_id')==cell['cell_id'] and
            all(item.get(key)==value[key] for key in ('input','audit','record')) for item in review.get('cases',[])),
        'Pre-actor activation review differs from exact candidate/evidence')
    _validate_records(cohort,{cell['cell_id']:cell},[record])
    return record,dict(cell_id=cell['cell_id'],audit=value['audit'],record=value['record'],
        input=value['input'],implementation=implementation,activation_review=value['activation_review'],replacement=True,
        replacement_pointer=evidence.ref(pointer),pre_actor_failure=True,
        unknown_reservations_preserved=0,raw_native_verdict_present=False,disposal_performed=False)



def load_preauth_implementation(evidence,implementation):
    """Load hash-pinned verifier/dependencies without mixing package roots."""
    path=evidence.reference(implementation)
    require(implementation['sha256']==PREAUTH_IMPLEMENTATION,'Unreviewed preauthentication implementation')
    root=path.parents[1];legacy_path=root/'scripts/audit_webprm_interrupted_cell.py'
    evidence.reference(dict(path=str(legacy_path),sha256=INTERRUPTION_IMPLEMENTATION))
    def load(name,path):
        spec=importlib.util.spec_from_file_location(name,path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
    module=load('_webprm_preauth_frozen',path)
    # Bind the unchanged legacy identity helper to its own frozen source root;
    # a same-named live package may already be imported by the collector.
    module.legacy=load('_webprm_preauth_identity_frozen',legacy_path)
    return module


def preauth_binding(evidence,pointer,folder,cell,cohort,campaign):
    value=evidence.read(pointer)
    require(value.get('protocol')==PREAUTH and value.get('cell_id')==cell['cell_id']
        and value.get('original_bytes_preserved') is True and value.get('primary_rerun') is False
        and value.get('disposal_performed') is False and value.get('raw_success') is None
        and value.get('effective_success') is None,'Unreviewed preauthentication pointer or fabricated verdict')
    implementation=value['implementation'];module=load_preauth_implementation(evidence,implementation)
    inputs=evidence.read_ref(value['input']);audit=evidence.read_ref(value['audit'])
    record=evidence.read_ref(value['record']);review=evidence.read_ref(value['activation_review'])
    replay=evidence.read_ref(value['independent_audit'])
    for source in review.get('source_references',[]):evidence.reference(source)
    require(inputs.get('implementation')==implementation and audit.get('protocol')==PREAUTH
        and audit.get('audit_verified') is True and audit.get('errors')==[]
        and audit.get('input_reference')==value['input'] and audit.get('implementation')==implementation
        and value['audit']['path']!=value['independent_audit']['path']
        and value['audit']['sha256']==value['independent_audit']['sha256'] and audit==replay,
        'Preauthentication report or independent replay differs')
    bound=evidence.read_ref(inputs['cohort']);bound=bound.get('cohort',bound)
    require(bound==cohort,'Preauthentication cohort differs')
    rebuilt=module.verify(evidence,inputs,cell,cohort,campaign,folder)
    require(audit.get('reconstruction')==rebuilt and record==dict(rebuilt['record'],audit_sha256=value['audit']['sha256']),
        'Preauthentication normalized disposition differs from raw evidence')
    require(review.get('protocol')=='table5_preauth_interruption_activation_review_v1'
        and review.get('verified') is True and review.get('errors')==[]
        and review.get('implementation')==implementation
        and review.get('native_verdicts_created')==review.get('state_disposals')==review.get('new_provider_calls')==0
        and any(item.get('cell_id')==cell['cell_id'] and all(item.get(k)==value[k]
            for k in ('input','audit','record','independent_audit')) for item in review.get('cases',[])),
        'Preauthentication activation lacks an exact independent review')
    _validate_records(cohort,{cell['cell_id']:cell},[record])
    return record,dict(cell_id=cell['cell_id'],audit=value['audit'],record=value['record'],input=value['input'],
        implementation=implementation,independent_review=value['activation_review'],
        replacement=True,replacement_pointer=evidence.ref(pointer),interrupted_failure=True,
        preauth_interrupted_failure=True,unknown_reservations_preserved=0,
        raw_native_verdict_present=False,disposal_performed=False)


def closed_watchdog_binding(evidence,pointer,folder,cell,cohort,campaign):
    """Retain original watchdog failures, including later outer process closure."""
    value=evidence.read(pointer)
    require(value.get('protocol')==CLOSED_WATCHDOG and value.get('cell_id')==cell['cell_id'] and
        value.get('original_bytes_preserved') is True and value.get('primary_rerun') is False and
        value.get('disposal_performed') is False and value.get('raw_success') is None and
        value.get('effective_success') is None,'Unreviewed closed-watchdog disposition')
    require(folder.parents[1].name=='attempt-367895' and not (folder/'audit.json').exists() and
        not (folder/'cell-outcome.json').exists(),'Watchdog supplement cannot replace a primary audit/outcome')
    implementation=value['implementation'];evidence.reference(implementation,campaign)
    require(implementation['sha256']==CLOSED_WATCHDOG_IMPLEMENTATION,'Unreviewed closed-watchdog implementation')
    audit=evidence.read_ref(value['audit'],campaign);replay=evidence.read_ref(value['independent_audit'],campaign)
    record=evidence.read_ref(value['record'],campaign);review=evidence.read_ref(value['activation_review'],campaign)
    require(audit==replay and value['audit']['path']!=value['independent_audit']['path'] and
        value['audit']['sha256']==value['independent_audit']['sha256'] and audit.get('protocol')==CLOSED_WATCHDOG and
        audit.get('implementation')==implementation and audit.get('cell')==cell and
        review.get('protocol')=='table5_closed_watchdog_activation_review_v1' and review.get('verified') is True and
        review.get('errors')==[] and review.get('implementation')==implementation and
        any(item.get('cell_id')==cell['cell_id'] and all(item.get(k)==value[k] for k in
            ('audit','record','independent_audit')) for item in review.get('cases',[])),
        'Watchdog supplement lacks exact independent replay/activation review')
    for name in ('worker_source_sha256','audit_dependency_sha256'):
        for path,sha in audit[name].items():evidence.reference(dict(path=path,sha256=sha))
    bound=evidence.read_ref(audit['cohort']);bound=bound.get('cohort',bound)
    require(bound==cohort,'Closed-watchdog cohort differs')
    refs=audit['artifacts'];paths={name:evidence.reference(ref,folder) for name,ref in refs.items()}
    require(set(paths)=={str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file() and
        p.name!='storage-state-private.json'} and all(paths[k]==folder/k for k in paths),
        'Closed-watchdog primary evidence was added, omitted or cross-bound')
    plan=evidence.read(paths['worker-plan.json']);episode=folder/'episodes'/f"webarena_lite_{cell['task_id']:03d}"/(cell['actor']+'__'+cell['selector'])
    result=evidence.read(episode/'result.json');watchdog=evidence.read(episode/'watchdog.json')
    outer=evidence.read_ref(audit['outer_closure'],campaign);site=evidence.read(folder/'site/closure.json')
    require(audit['outer_closure']==evidence.ref(folder.parents[1]/'controller-closure.json') and
        outer.get('remaining_owned_processes')=={} and site.get('remaining_owned_processes')=={} and
        result.get('identity_sha256')==digest(plan) and result.get('task_id')==f"webarena_lite_{cell['task_id']:03d}" and
        result.get('arm')==cell['actor']+'__'+cell['selector'] and result.get('attempted') is True and
        result.get('valid') is False and result.get('success')==0 and result.get('error_type')=='EpisodeWatchdogTimeout' and
        result.get('watchdog')==watchdog and not any(k in result for k in ('native_score','native_success','score','terminal')) and
        not (episode/'native-evaluation/native-verdict.json').exists(),'Saved watchdog identity, verdict absence or later closure differs')
    require(audit.get('original_worker_remaining_processes')==watchdog['remaining_owned_processes'] and
        set(map(int,watchdog['remaining_owned_processes']))<=set(outer['owned_pids']) and
        watchdog['last_phase']['pid'] in outer['owned_pids'] and audit.get('later_outer_owned_closure_empty') is True,
        'Watchdog immediate failure was rewritten or lacks later owned closure')
    for key,directory in (('queue_claim','claims'),('queue_outcome','outcomes')):
        require(audit[key]==evidence.ref(campaign/'queue'/directory/(digest(cell['cell_id'])+'.json')),
            'Watchdog queue reference belongs to another cell')
    claim=evidence.read_ref(audit['queue_claim']);outcome=evidence.read_ref(audit['queue_outcome'])
    require(claim.get('cell_id')==outcome.get('cell_id')==cell['cell_id'] and
        claim.get('cell_sha256')==outcome.get('cell_sha256')==digest(cell) and claim.get('owner')==outcome.get('owner') and
        outcome['result'].get('status')=='invalid' and outcome['result'].get('reason')=='worker_exception',
        'Closed-watchdog consumed queue identity differs')
    calls=audit['financial_audit']
    require(calls.get('verified') is True and calls.get('cell_unknown_reservations')==0 and
        calls.get('policy_within_caps') is True and calls.get('api_sha256')==refs[str(episode.relative_to(folder)/'api.jsonl')]['sha256'],
        'Closed-watchdog metered-call audit is incomplete')
    for label,field in (('financial','budget'),('broker','broker')):
        prefix=audit[label+'_prefix'];evidence.prefix(prefix['path'],prefix['bytes'],prefix['sha256'],campaign)
        require(calls[field+'_prefix_bytes']==prefix['bytes'] and calls[field+'_prefix_sha256']==prefix['sha256'],
            'Closed-watchdog accounting prefix differs')
    prefix=audit['financial_prefix']
    with Path(prefix['path']).open('rb') as stream:last=json.loads(stream.read(prefix['bytes']).splitlines()[-1])
    require(last.get('event')=='epoch_closed' and last.get('epoch_id')=='job-367895' and
        last.get('event_sha256')==audit['financial_epoch_closed_sha256'],'Watchdog financial owner was not closed')
    owner=evidence.read_ref(audit['owner_closure_review'],campaign)
    require(owner.get('closed') is True and owner.get('job_id')=='367895' and
        owner.get('scheduler')=='367895|FAILED|27500|1:0' and owner.get('closure')==audit['outer_closure'] and
        owner.get('financial_epoch_closed_sha256')==audit['financial_epoch_closed_sha256'],
        'Closed-watchdog terminal scheduler/owner review differs')
    expected=dict(cell_id=cell['cell_id'],released_task_id=cell['task_id'],actor_model=cell['actor'],selector=cell['selector'],
        domain=cell['domain'],task_config_sha256=cell['task_config_sha256'],tasks_sha256=cohort['source_sha256'],
        cohort_sha256=digest(cohort),protocol_sha256=cohort['protocol_sha256'],audit_verified=True,terminal=True,
        attempted=True,valid=False,success=None,native_success=None,effective_semantic_status='not_applicable',
        native_semantic_status='not_applicable')
    require(audit['record']==expected and record==dict(expected,audit_sha256=value['audit']['sha256']) and
        all(audit.get(k) is False for k in ('primary_modified','primary_rerun','disposal_performed')),
        'Closed-watchdog normalized record fabricated a verdict or changed primary data')
    _validate_records(cohort,{cell['cell_id']:cell},[record])
    return record,dict(cell_id=cell['cell_id'],audit=value['audit'],record=value['record'],
        independent_audit=value['independent_audit'],activation_review=value['activation_review'],
        replacement=True,replacement_pointer=evidence.ref(pointer),interrupted_failure=True,
        unknown_reservations_preserved=0,raw_native_verdict_present=False,disposal_performed=False,
        original_worker_closure_complete=watchdog['remaining_owned_processes']=={},later_outer_closure=audit['outer_closure'])


def collect(campaign_root,cohort_path,*,scope_domains=('reddit',),seconds=120,bootstrap_draws=2000):
    campaign=canonical(campaign_root);evidence=Evidence(seconds)
    cohort=evidence.read(cohort_path);cohort=cohort.get('cohort',cohort)
    planned=validate_cohort(cohort)
    require(cohort['protocol_sha256'] is not None,'Reporting requires a bound scientific protocol')
    released=evidence.read_ref(dict(path=str(TASKS),sha256=TASKS_SHA))
    require(cohort['cells']==make_cohort(released)['cells'],'Cohort task membership differs from pinned released source')
    require(scope_domains and len(set(scope_domains))==len(scope_domains) and set(scope_domains)<=set(cohort['domain_counts']),
        'Explicit fixed domain reporting scope required')
    names={hashlib.sha256(key.encode()).hexdigest():cell for key,cell in planned.items()}
    pointers={}
    for pointer in sorted((campaign/'audit-replacements').glob('*')):
        require(pointer.is_file() and pointer.suffix=='.json' and pointer.stem in names,'Unexpected replacement pointer')
        pointers[pointer.stem]=canonical(pointer)
    seen={};records=[];selected=[];dispositions=[];unused=[];directory_count=prepared_count=0
    for attempt in sorted(campaign.glob('attempt-*')):
        require(attempt.is_dir() and re.fullmatch('attempt-[0-9]+',attempt.name),'Unexpected campaign attempt directory')
        for folder in sorted((attempt/'cells').glob('*')):
            require(folder.is_dir() and folder.name in names,'Unexpected/foreign primary cell directory')
            canonical(folder);cell=names[folder.name]
            directory_count+=1
            # Controllers prepare every eligible plan at allocation startup.
            # Repeated plan-only directories are not repeated primary episodes.
            files=[p for p in folder.rglob('*') if p.is_file()]
            if not files and not any(folder.iterdir()):
                prepared_count+=1
                dispositions.append(dict(cell_id=cell['cell_id'],status='empty_preparation_directory_no_primary_evidence'))
                continue
            if files==[folder/'worker-plan.json']:
                plan=evidence.read(folder/'worker-plan.json')
                require(plan.get('task_ids')==[cell['task_id']] and plan.get('actor_model')==cell['actor'] and
                    plan.get('selector')==cell['selector'] and digest(plan.get('protocol_contract'))==cohort['protocol_sha256'],
                    'Prepared worker plan has a foreign cell/protocol identity')
                prepared_count+=1
                dispositions.append(dict(cell_id=cell['cell_id'],status='prepared_only_no_primary_evidence',
                    worker_plan=evidence.ref(folder/'worker-plan.json')))
                continue
            require(cell['cell_id'] not in seen,'Duplicate/resampled primary cell across attempts')
            seen[cell['cell_id']]=str(folder)
            pointer=pointers.pop(folder.name,None)
            original_path=folder/'audit.json';record_path=folder/'audit.record.json'
            if not original_path.exists() or not (folder/'cell-outcome.json').exists():
                if pointer:
                    protocol=evidence.read(pointer).get('protocol')
                    binding=(pre_actor_binding if protocol==PRE_ACTOR else
                             preauth_binding if protocol==PREAUTH else
                             closed_watchdog_binding if protocol==CLOSED_WATCHDOG else
                             v10_interruption_binding if protocol==V10_INTERRUPTION else interruption_binding)
                    record,selection=binding(evidence,pointer,folder,cell,cohort,campaign)
                    records.append(record);selected.append(selection)
                    continue
                require(pointer is None,'Replacement points at an incomplete original audit/outcome')
                dispositions.append(pending_disposition(evidence,folder,cell,cohort,campaign))
                continue
            original=evidence.read(original_path)
            extra={};value=None;audit_path=original_path
            if pointer:
                audit_path,record_path,extra,value=replacement(evidence,pointer,folder,cell,original)
            elif original.get('audit_verified') is not True:
                require(not record_path.exists(),'Rejected audit unexpectedly has a normalized record')
                dispositions.append(dict(cell_id=cell['cell_id'],status='audit_rejected',audit=evidence.ref(original_path)))
                continue
            else:require(original.get('protocol')==AUDIT,'Unrecognized independent audit protocol')
            record,audit,inputs,auditor=record_binding(evidence,folder,cell,cohort,campaign,audit_path,record_path,original,extra)
            if value:
                expected_auditor=CORRECTED if value.get('protocol')==UNICODE_ERRATUM else ORIGINAL
                require(auditor==expected_auditor and value.get('raw_success')==record['native_success'] and
                    value.get('effective_success')==record['success'],'Replacement changed primary native/effective outcomes')
            records.append(record)
            selection=dict(cell_id=cell['cell_id'],audit=evidence.ref(audit_path),record=evidence.ref(record_path),
                input=evidence.ref(folder/'audit-input.json'),original_audit=evidence.ref(original_path),
                mechanical_outcome=evidence.ref(folder/'cell-outcome.json'),replacement=bool(pointer),
                unknown_reservations_preserved=audit['call_audit']['cell_unknown_reservations'])
            if pointer:
                selection['replacement_pointer']=evidence.ref(pointer)
                retained=retained_state_binding(evidence,value,folder,cell,inputs,campaign)
                if retained is not None:selection['retained_state']=retained
            selected.append(selection)
            for other in folder.glob('*.record.json'):
                if other!=record_path:
                    other_record=evidence.read(other)
                    require(all(other_record.get(k)==record.get(k) for k in audit['record']),
                        'Conflicting unselected normalized record')
                    unused.append(dict(reference=evidence.ref(other),reason='Unselected additive audit; exact pointer controls selection'))
    require(not pointers,'Replacement points at an absent fixed primary cell')
    full=aggregate_table5(cohort,records,bootstrap_draws=bootstrap_draws)
    cells=[c for c in cohort['cells'] if c['domain'] in scope_domains]
    indexed=_validate_records(cohort,planned,records);counts=_counts(cells,indexed)
    groups=[(actor,selector,[c for c in cells if c['actor']==actor and c['selector']==selector])
            for actor in cohort['actors'] for selector in cohort['selectors']]
    reasons=[]
    if counts['missing']:reasons.append('incomplete_fixed_domain_scope')
    if counts['effective_semantic_invalid']:reasons.append('known_invalid_effective_judgments')
    if counts['effective_semantic_unreviewed']:reasons.append('unreviewed_effective_judgments')
    if any(not _counts(group,indexed)['valid'] for _,_,group in groups):reasons.append('method_groups_without_valid_episodes')
    bounded=dict(domains=list(scope_domains),counts=counts,terminal_grid_complete=counts['missing']==0,
        performance_rates_withheld=bool(reasons),withheld_reasons=reasons,
        methods=[dict(actor_model=a,selector=s,counts=(n:=_counts(g,indexed)),rates=_rates(n,not reasons)) for a,s,g in groups])
    evidence.unchanged()
    claim_counts=dict(claimed=len(seen),audited_records=len(records),audited_valid=sum(r['valid'] for r in records),
        audited_invalid=sum(not r['valid'] for r in records),
        interrupted_missing_terminal_audit=sum(d['status']=='interrupted_missing_terminal_audit' for d in dispositions),
        bounded_scope_unclaimed=sum(c['cell_id'] not in seen for c in cells))
    return dict(schema_version=2,protocol='table5_saved_audit_collection_v2',campaign_root=str(campaign),
        cohort_reference=evidence.ref(cohort_path),cohort_sha256=digest(cohort),protocol_sha256=cohort['protocol_sha256'],
        selected=selected,unselected_additive_records=unused,dispositions=dispositions,normalized_records=records,
        full_table5=full,bounded_scope=bounded,primary_claim_counts=claim_counts,primary_directories_observed=len(seen),
        cell_directories_observed=directory_count,prepared_only_directories=prepared_count,
        replacement_count=sum(x['replacement'] for x in selected),bytes_hashed=evidence.bytes,
        scope='Saved evidence/source consistency collection; no episode rerun, native re-evaluation, independent historical process observation or website-truth certification. Mechanical audit flags do not authorize results. Missing/rejected/pending audit cells remain missing coverage; every saved primary/native/effective verdict remains unchanged.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',required=True,type=Path);parser.add_argument('--cohort',required=True,type=Path)
    parser.add_argument('--domain',action='append',default=None);parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--seconds',type=int,default=120);args=parser.parse_args()
    result=collect(args.campaign,args.cohort,scope_domains=tuple(args.domain or ['reddit']),seconds=args.seconds)
    output=canonical(args.output)
    require(canonical(args.campaign) not in output.parents,'Reporting output must stay outside immutable primary campaign')
    with os.fdopen(os.open(output,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),'wb') as stream:
        stream.write((json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n').encode());stream.flush();os.fsync(stream.fileno())
    print(json.dumps(dict(status='collected',recorded=len(result['normalized_records']),
        replacements=result['replacement_count'],full_rates_withheld=result['full_table5']['performance_rates_withheld'],
        bounded_rates_withheld=result['bounded_scope']['performance_rates_withheld'],output=str(output))))


if __name__=='__main__':main()
