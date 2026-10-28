#!/usr/bin/env python3
"""Additive offline audit of an actual saved, interrupted supervisor failure.

No worker result, provider reply, native verdict or broker terminal event is
created. Candidate records require independent review before collector use.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.audit_webprm_table5_cell import read,reference,sha,write_once
from scripts.prepare_webprm_table5_campaign import TASKS,TASKS_SHA,cohort as fixed_cohort
from openwebrl.webprm_table5_analysis import digest,validate_cohort
from openwebrl.webprm_webarena_sites import canonical_runtime_path
from openwebrl.webprm_journal_audit import audit_cell_calls,_prefix
from scripts.run_webprm_webarena_pilot import REQUIRED_CODE

PROTOCOL='table5_saved_interrupted_supervisor_audit_v1'
SOURCE_FILES=tuple(sorted(set(REQUIRED_CODE)|{'scripts/audit_webprm_interrupted_cell.py','scripts/audit_webprm_table5_cell.py',
 'scripts/prepare_webprm_table5_campaign.py','openwebrl/webprm_journal_audit.py',
 'openwebrl/webprm_table5_analysis.py','openwebrl/webprm_webarena_worker.py',
 'openwebrl/webprm_worker.py','openwebrl/webprm_webarena.py','openwebrl/webprm_source.py',
 'openwebrl/webprm_webarena_piotr.py','openwebrl/webprm_webarena_sites.py',
 'scripts/analyze_webprm_webarena_pilot.py','openwebrl/webprm_webarena_helpers.py',
 'openwebrl/webprm_webarena_helper_runtime.py','openwebrl/webprm_evaluator_transcript.py',
 'openwebrl/webprm_baselines.py','openwebrl/webprm_budget_journal.py','openwebrl/webprm_concurrent_broker.py',
 'openwebrl/webprm_ephemeral_state.py'}))


def require(ok,message):
    if not ok:raise ValueError(message)


def ref(path):return dict(path=str(path),sha256=sha(path))


def identity(inputs,plan,cohort):
    from openwebrl.webprm_webarena_worker import validate_protocol_contract,validate_frozen_sources
    from scripts.run_webprm_webarena_pilot import REQUIRED_CODE
    cell=inputs['cell'];indexed=validate_cohort(cohort)
    require(indexed.get(cell.get('cell_id'))==cell and sha(TASKS)==TASKS_SHA,'Fixed released cell/task source differs')
    tasks=read(TASKS);require(cohort['cells']==fixed_cohort(tasks)['cells'],'Released cohort membership/order differs')
    raw=next(t for t in tasks if t['task_id']==cell['task_id'])
    require(digest(raw)==cell['task_config_sha256'] and digest(validate_protocol_contract(plan))==cohort['protocol_sha256'],
            'Task or scientific protocol changed')
    require(plan.get('task_ids')==[cell['task_id']] and plan.get('tasks_sha256')==digest([raw]) and
        plan.get('actor_model')==cell['actor'] and plan.get('selector')==cell['selector'] and
        plan.get('actors')==[cell['actor']] and plan.get('selectors')==[cell['selector']] and
        plan.get('max_turns')==30 and plan.get('paper_exact_protocol') is False,'Fixed worker identity changed')
    code=plan.get('code_sha256',{});roots=[Path(p).parents[1] for p in code if p.endswith('/scripts/audit_webprm_table5_cell.py')]
    require(len(roots)==1 and all(str(roots[0]/p) in code for p in SOURCE_FILES
        if p!='scripts/audit_webprm_interrupted_cell.py'),'Original worker source freeze incomplete')
    for path,checksum in code.items():reference(dict(path=path,sha256=checksum))
    validate_frozen_sources(plan)
    if cell['selector']=='piotr_selectionarm':
        from openwebrl.webprm_webarena_piotr import validate_piotr_selectionarm
        require(plan.get('piotr_manifest')==validate_piotr_selectionarm(plan),'Piotr identity changed')
    own=inputs.get('audit_source_sha256',{})
    require(set(own)==set(SOURCE_FILES) and all(sha(REPO/p)==own[p] for p in SOURCE_FILES),
            'Executing interruption auditor source differs from reviewed input')
    return cell


def verify_supervisor(cell,plan,result,watchdog,claim,launch,instance,site,outer,stop,folder,campaign):
    """Actual legacy supervisor fields bind identity without adding absent keys."""
    execution=f"webarena_lite_{cell['task_id']:03d}";arm=cell['actor']+'__'+cell['selector']
    require(result.get('identity_sha256')==digest(plan) and result.get('task_id')==execution and
        result.get('arm')==arm and result.get('attempted') is True and result.get('valid') is False and
        type(result.get('success')) is int and result['success']==0 and result.get('error_type')=='EpisodeWorkerExited' and
        result.get('error')=='Episode child exited without result (exit -15)' and result.get('watchdog')==watchdog,
        'Saved result is not the exact interrupted supervisor failure')
    for key,value in [('actor_model',cell['actor']),('selector',cell['selector'])]:
        require(key not in result or result[key]==value,'Saved optional worker identity contradicts fixed cell')
    require(not any(k in result for k in ('native_score','native_success','score','terminal')),
            'Interrupted result unexpectedly contains a native verdict')
    require(watchdog.get('supervisor_signal')==15 and watchdog.get('child_exitcode')==-15 and
        watchdog.get('remaining_owned_processes')=={} and site.get('remaining_owned_processes')=={} and
        outer.get('remaining_owned_processes')=={},'Interrupted worker/site/outer closure incomplete')
    require(set(watchdog.get('owned_pids',[]))<=set(outer.get('owned_pids',[])) and
        instance.get('service_pid') in site.get('owned_pids',[]) and
        instance.get('service_pid') in outer.get('owned_pids',[]),'Saved process ownership identities differ')
    phase=watchdog.get('last_phase',{})
    require(phase.get('identity_sha256')==digest(plan) and phase.get('pid') in watchdog.get('owned_pids',[]),
            'Saved interrupted phase differs from owned worker')
    require(claim.get('cell_id')==cell['cell_id'] and claim.get('campaign_root')==str(campaign) and
        claim.get('destination_absent_at_claim') is True and claim.get('preserve_historical_states') is True and
        claim.get('state_path')==str(Path(claim['scratch_root'])/'states'/folder.name),'Fresh fixed primary claim differs')
    require(launch.get('cell_id')==cell['cell_id'] and launch.get('resampling_forbidden') is True,
            'Worker launch did not bind this immutable primary')
    argv=launch.get('argv',[])
    for flag,value in [('--plan',str(folder/'worker-plan.json')),('--output',str(folder/'episodes')),
                       ('--task-id',str(cell['task_id'])),('--actor',cell['actor']),('--selector',cell['selector'])]:
        require(argv.count(flag)==1 and argv[argv.index(flag)+1]==value,'Actual launched worker command differs')
    require(argv.count('--target')==1 and str(instance['service_pid'])==argv[argv.index('--target')+1] and argv.count('--execute')==1,
            'Launched worker namespace or execution mode differs')
    require(any(row==dict(cell_id=cell['cell_id'],folder=str(folder)) for row in stop.get('active_cells',[])) and
        str(plan['job_id'])==stop.get('job_id'),'Saved stop intent does not cover this fixed primary')


def verify_reset_queue(inputs,paths,plan,cohort):
    from scripts.audit_webprm_table5_cell import audit_reddit_reset
    cell=inputs['cell'];instance=read(paths['instance']);provenance=read(paths['provenance'])
    auth=read(paths['authentication']);manifests=plan['source_manifests']
    critic=plan['piotr_manifest'] if cell['selector']=='piotr_selectionarm' else manifests['viewer_reward']
    require(provenance.get('identity_sha256')==digest(plan) and provenance.get('actor_model')==cell['actor'] and
        provenance.get('original_instance')==instance and provenance.get('actor')==manifests['arbiter_harness'] and
        provenance.get('observation')==manifests['arbiter_observation'] and provenance.get('critic')==critic and
        provenance.get('paper_exact_protocol') is False and auth.get('authenticated') is True and
        auth.get('state_provenance_sha256')==provenance.get('reset_id') and
        str(instance.get('job_id'))==str(plan['job_id']) and
        re.fullmatch(r'net:\[[0-9]+\]',str(instance.get('network_namespace'))) and
        type(instance.get('service_start_ticks')) is int and instance['service_start_ticks']>0,
        'Saved partial-worker provenance/authentication/instance differs')
    # The supervisor failure result never carried reset_id. Reuse the reset
    # verifier with the genuine saved provenance projection, without editing or
    # augmenting that result, and make this source of reset identity explicit.
    reset_paths=dict(paths,state_claim=paths['claim'])
    audit_reddit_reset(inputs,reset_paths,plan,{'reset_id':provenance['reset_id']})
    queue=read(paths['queue_manifest']);claim=read(paths['queue_claim']);outcome=read(paths['queue_outcome'])
    require(queue==dict(cell_count=len(cohort['cells']),cells_sha256=digest(cohort['cells'])) and
        claim.get('cell_id')==cell['cell_id'] and claim.get('cell_sha256')==digest(cell) and
        re.fullmatch('[0-9a-f]{32}',str(claim.get('owner'))) and
        all(outcome.get(k)==claim.get(k) for k in ('cell_id','cell_sha256','owner')) and
        outcome.get('result') in [dict(status='invalid',reason='worker_exception',error_type=kind)
                                  for kind in ('RuntimeError','ValueError')] and
        type(claim.get('epoch')) in (int,float) and type(outcome.get('epoch')) in (int,float) and
        claim['epoch']<=outcome['epoch'],'Durable fixed queue claim/outcome differs')
    return dict(verified=True,reset_identity_source='saved_worker_provenance',reset_id=provenance['reset_id'],
        queue_claim_sha256=sha(paths['queue_claim']),queue_outcome_sha256=sha(paths['queue_outcome']))


def verify(inputs):
    require(inputs.get('schema_version')==1 and inputs.get('protocol')==PROTOCOL,'Explicit interruption audit contract required')
    campaign=canonical_runtime_path(inputs['campaign_root']);folder=canonical_runtime_path(inputs['cell_folder'])
    require(campaign in folder.parents,'Interrupted cell escaped campaign')
    cohort=read(reference(inputs['cohort']));cohort=cohort.get('cohort',cohort)
    plan=read(reference(inputs['worker_plan'],root=campaign));cell=identity(inputs,plan,cohort)
    require(inputs['financial_contract'].get('cohort_sha256')==digest(cohort) and
        inputs['financial_contract'].get('protocol_sha256')==cohort['protocol_sha256'],
        'Recovered financial contract differs from fixed scientific cohort')
    require(folder.name==hashlib.sha256(cell['cell_id'].encode()).hexdigest() and
        inputs['worker_plan']['path']==str(folder/'worker-plan.json'),'Claim directory/plan differs')
    refs=inputs['artifacts'];paths={k:reference(v,root=campaign) for k,v in refs.items()}
    required={'result','watchdog','api','claim','launch','instance','site_closure','outer_closure','stop_intent','rollout',
        'lease','copy_receipt','provenance','authentication','queue_manifest','queue_claim','queue_outcome'}
    require(required<=paths.keys(),'Missing actual interruption evidence')
    episode=folder/'episodes'/f"webarena_lite_{cell['task_id']:03d}"/(cell['actor']+'__'+cell['selector'])
    expected=dict(result=episode/'result.json',watchdog=episode/'watchdog.json',api=episode/'api.jsonl',
        rollout=episode/'rollout.json',claim=folder/'state-claim.json',launch=folder/'worker-launch.json',
        instance=folder/'site/instance.json',site_closure=folder/'site/closure.json',
        outer_closure=folder.parents[1]/'controller-closure.json',
        stop_intent=campaign/f"operational-stop-{plan['job_id']}.json",lease=folder/'lease.json',
        copy_receipt=folder/'copy/receipt.json',provenance=episode/'provenance.json',authentication=episode/'authentication.json',
        queue_manifest=campaign/'queue/manifest.json',
        queue_claim=campaign/'queue/claims'/(digest(cell['cell_id'])+'.json'),
        queue_outcome=campaign/'queue/outcomes'/(digest(cell['cell_id'])+'.json'))
    require(all(paths[k]==p for k,p in expected.items()),'Named evidence belongs to another primary/owner path')
    saved={k:read(paths[k]) for k in required-{'api'}}
    verify_supervisor(cell,plan,saved['result'],saved['watchdog'],saved['claim'],saved['launch'],saved['instance'],
        saved['site_closure'],saved['outer_closure'],saved['stop_intent'],folder,campaign)
    reset=verify_reset_queue(inputs,paths,plan,cohort)
    require(type(saved['result'].get('turns')) is int and 0<=saved['result']['turns']<=plan['max_turns'] and
        len(saved['rollout'])==saved['result']['turns'],'Saved partial turn accounting differs')
    covered=set(paths.values())
    for path in paths['result'].parent.rglob('*'):
        if path.is_file() and path.name!='storage-state-private.json':
            require(path in covered,'Partial episode file omitted from interruption manifest')
            require(path.name not in ('verdict.json','outcome.json','terminal.json'),'Interruption cannot conceal a saved native verdict')
    recovery=read(reference(inputs['epoch_recovery']));before,after=recovery['before'],recovery['after']
    require(recovery.get('status')=='stopped_epoch_closed_with_independent_proofs' and
        recovery.get('provider_calls')==0 and recovery.get('refunds')==0 and recovery.get('old_outcomes_and_claims_untouched') is True and
        after.get('active_epoch_id') is None and after.get('unsettled_reservations')==0,'Epoch recovery/accounting incomplete')
    closure_ref=before['closure_receipt'];finance_ref=before['financial_receipt']
    closure=read(reference(closure_ref));finance=read(reference(finance_ref))
    require(closure.get('verified') is True and closure.get('errors')==[] and closure.get('all_owned_processes_reaped') is True and
        closure.get('scheduler_terminal') is True and closure.get('process_closure_evidence')==refs['outer_closure'] and
        closure.get('epoch_id')=='job-'+str(plan['job_id']),'Independent recovered-owner closure differs')
    scheduler=read(reference(closure['scheduler_evidence']));reference(closure['stop_accounting_audit'])
    jobs=scheduler.get('jobs',[])
    require(scheduler.get('errors')==[] and len(jobs)==1 and str(jobs[0].get('job_id'))==str(plan['job_id']) and
        jobs[0].get('state',{}).get('current')==[closure['scheduler_state']] and
        closure['scheduler_state'] in ('FAILED','CANCELLED','TIMEOUT','COMPLETED','OUT_OF_MEMORY','NODE_FAIL') and
        jobs[0].get('time',{}).get('elapsed')==closure['elapsed_seconds'] and
        type(closure['elapsed_seconds']) is int and closure['elapsed_seconds']>0,
        'Saved scheduler terminal/elapsed identity differs')
    require(finance.get('verified') is True and finance.get('errors')==[] and finance.get('unknown_reservations')==0 and
        finance.get('unknown_reservations_retained') is True and finance.get('independent_audit')==closure['stop_accounting_audit'],
        'Independent recovered financial audit differs')
    require(before['audit_path']==str(folder.parents[1]/'broker.jsonl') and
        all(finance.get(k)==before.get(k) for k in ('audit_path','audit_prefix_bytes','audit_sha256','budget_prefix_bytes','budget_ledger_sha256')),
        'Recovered prefix bindings differ')
    budget_path=campaign/'shared-budget.json.events.jsonl'
    events=list(_prefix(budget_path,after['budget_prefix_bytes'],after['budget_ledger_sha256']))
    last=events[-1][0]
    require(last.get('event')=='epoch_closed' and last.get('closure')=='audited_recovery' and
        last.get('epoch_id')==closure['epoch_id'] and last.get('event_sha256')==recovery['previous_closed_epoch_sha256'] and
        last.get('recovery_evidence')==[closure_ref,finance_ref] and
        all(last.get(k)==before.get(k) for k in ('audit_path','audit_prefix_bytes','audit_sha256')),
        'Actual durable recovered-epoch event differs from saved proof')
    broker=list(_prefix(before['audit_path'],before['audit_prefix_bytes'],before['audit_sha256']))
    broker_cell=f"webarena_lite_{cell['task_id']:03d}/"+cell['actor']+'__'+cell['selector']
    groups={}
    for event,_,_ in broker:
        if event.get('cell_id')==broker_cell:groups.setdefault(event['request_id'],[]).append(event)
    unfinished=[ident for ident,rows in groups.items() if [r['event'] for r in rows]==['dispatch_started']]
    require(len(unfinished)<=1 and all(groups[i][0].get('local') is True for i in unfinished),
            'Only one explicitly recorded interrupted local dispatch is supported')
    calls=audit_cell_calls(plan,broker_cell,paths['api'],before['audit_path'],budget_path,
        after['budget_prefix_bytes'],after['budget_ledger_sha256'],inputs['financial_contract'],
        broker_prefix_bytes=before['audit_prefix_bytes'],broker_prefix_sha256=before['audit_sha256'],
        interrupted_local_request_ids=tuple(unfinished))
    require(calls['global_unknown_reservations']==0 and calls['cell_unknown_reservations']==0 and
        calls['policy_within_caps'] and calls['global_charged_usd']==after['charged'] and
        calls['broker_policy_sha256s']==[closure['broker_policy_sha256']],'Recovered exact call/cost totals differ')
    for name,saved_name in [('global_dispatches','dispatches'),('global_paid_dispatches','paid_dispatches'),
                            ('global_local_dispatches','local_dispatches')]:
        require(calls[name]==after[saved_name]==finance[saved_name],'Recovered cumulative dispatch totals differ')
    # Recheck closed evidence after interpretation; later unrelated financial appends
    # are allowed only beyond the independently verified fixed journal prefix.
    for value in refs.values():reference(value,root=campaign)
    record=dict(cell_id=cell['cell_id'],released_task_id=cell['task_id'],actor_model=cell['actor'],selector=cell['selector'],
        domain=cell['domain'],task_config_sha256=cell['task_config_sha256'],tasks_sha256=cohort['source_sha256'],
        cohort_sha256=digest(cohort),protocol_sha256=cohort['protocol_sha256'],audit_verified=True,terminal=True,
        attempted=True,valid=False,success=None,native_success=None,effective_semantic_status='not_applicable',
        native_semantic_status='not_applicable')
    return dict(record=record,call_audit={k:v for k,v in calls.items() if k not in ('pairs','broker_only_failures')},
        complete_for_disposal=False,native_replay=None,reset_and_queue=reset,unfinished_local_dispatches=unfinished,
        original_result_has_no_native_verdict=True,all_primary_bytes_preserved=True)


def audit(input_path,output_path):
    inputs=read(input_path);output_path=canonical_runtime_path(str(output_path))
    require(canonical_runtime_path(inputs['campaign_root']) not in output_path.parents,
            'Unactivated interruption review must stay outside primary campaign')
    report=dict(schema_version=1,protocol=PROTOCOL,input_reference=ref(input_path),
        evidence_manifest_sha256=digest(inputs['artifacts']),artifact_refs=inputs['artifacts'],
        audit_verified=False,complete_for_disposal=False,errors=[],activation='candidate_pending_independent_review',
        scope='Saved interrupted supervisor/identity/closure and recovered financial evidence only. No native success judgment, provider response, broker terminal event, rerun, refund or disposal is created.')
    try:report.update(verify(inputs));report['audit_verified']=True
    except Exception as error:report['errors']=[dict(type=type(error).__name__,message=str(error)[:1000])]
    write_once(output_path,report)
    if report['audit_verified']:write_once(output_path.with_suffix('.record.json'),dict(report['record'],audit_sha256=sha(output_path)))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();report=audit(a.input,a.output);print(json.dumps({k:report[k] for k in ('audit_verified','errors','activation')}))
    return 0 if report['audit_verified'] else 2


if __name__=='__main__':raise SystemExit(main())
