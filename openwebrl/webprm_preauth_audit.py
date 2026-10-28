"""Additive audit of a saved CMS worker interrupted before authentication.

This certifies an attempted, invalid execution with no native verdict. It never
creates episode evidence, submits work, calls a provider, or disposes of state.
"""
from pathlib import Path
import hashlib
import re

from scripts import audit_webprm_interrupted_cell as legacy
from openwebrl.webprm_journal_audit import audit_cell_calls, _prefix
from openwebrl.webprm_table5_analysis import digest
from openwebrl import webprm_cms_site as cms

PROTOCOL='table5_saved_cms_preauth_interruption_v1'


def require(ok,message):
    if not ok:raise ValueError(message)


def verify(evidence,inputs,cell,cohort,campaign,folder):
    require(inputs.get('protocol')==PROTOCOL and inputs.get('cell')==cell and cell['domain']=='shopping_admin'
        and inputs.get('campaign_root')==str(campaign) and inputs.get('cell_folder')==str(folder),
        'Explicit fixed CMS preauthentication audit required')
    require(folder.name==hashlib.sha256(cell['cell_id'].encode()).hexdigest() and campaign in folder.parents,
        'Foreign primary folder')
    required_sources=set(legacy.SOURCE_FILES)|{'openwebrl/webprm_preauth_audit.py','openwebrl/webprm_cms_site.py'}
    require(set(inputs['dependency_sources'])==required_sources,'Complete frozen audit dependencies required')
    for name,ref in inputs['dependency_sources'].items():
        require(ref==legacy.ref(legacy.REPO/name),'Executing preauthentication dependencies differ')
        evidence.reference(ref)
    implementation=inputs['implementation'];evidence.reference(implementation)
    require(implementation['sha256']==legacy.sha(Path(__file__)),'Executing audit implementation differs')
    require(inputs['dependency_sources'].get('openwebrl/webprm_preauth_audit.py')==implementation,
        'Executing preauthentication implementation omitted from freeze')
    refs=inputs['artifacts'];paths={k:evidence.reference(v,campaign) for k,v in refs.items()}
    episode=folder/'episodes'/f"webarena_lite_{cell['task_id']:03d}"/(cell['actor']+'__'+cell['selector'])
    attempt=folder.parents[1]
    expected=dict(plan=folder/'worker-plan.json',claim=folder/'state-claim.json',launch=folder/'worker-launch.json',
        result=episode/'result.json',watchdog=episode/'watchdog.json',phase=episode/'phase.json',
        lifecycle=episode/'lifecycle.jsonl',instance=folder/'site/instance.json',lease=folder/'lease.json',
        site_closure=folder/'site/closure.json',outer_closure=attempt/'controller-closure.json',
        copy_receipt=folder/'copy/receipt.json',contract=attempt/'financial-contract.json',
        queue_manifest=campaign/'queue/manifest.json',queue_claim=campaign/'queue/claims'/(digest(cell['cell_id'])+'.json'),
        queue_outcome=campaign/'queue/outcomes'/(digest(cell['cell_id'])+'.json'))
    require(paths==expected,'Missing or cross-bound preauthentication evidence')
    require(set(p.name for p in episode.iterdir())=={'result.json','watchdog.json','phase.json','lifecycle.jsonl'},
        'Preauthentication episode contains unreviewed evidence or a later worker phase')
    require(not (folder/'audit.json').exists() and not (folder/'cell-outcome.json').exists(),
        'Cannot replace an existing primary audit/outcome')
    values={k:evidence.read(p) for k,p in paths.items() if k!='lifecycle'};plan=values['plan']
    legacy.identity(inputs,plan,cohort)
    require(inputs['worker_plan']==refs['plan'],'Worker identity reference differs')
    result,wd,phase=values['result'],values['watchdog'],values['phase']
    require(result.get('identity_sha256')==digest(plan) and result.get('task_id')==f"webarena_lite_{cell['task_id']:03d}"
        and result.get('arm')==cell['actor']+'__'+cell['selector'] and result.get('attempted') is True
        and result.get('valid') is False and type(result.get('success')) is int and result['success']==0
        and result.get('turns')==0 and result.get('error_type')=='EpisodeWorkerExited'
        and result.get('error')=='Episode child exited without result (exit -15)' and result.get('watchdog')==wd
        and not any(k in result for k in ('native_success','native_score','score','terminal','reset_id')),
        'Saved supervisor failure or absent verdict differs')
    require(phase==wd.get('last_phase') and phase.get('phase')=='native_authentication'
        and phase.get('identity_sha256')==digest(plan) and phase.get('pid') in wd.get('owned_pids',[])
        and result['started_at']<=phase['started_at']<=result['finished_at']
        and wd.get('supervisor_signal')==15 and wd.get('child_exitcode')==-15,
        'Worker did not stop in its owned preauthentication phase')
    site,outer=values['site_closure'],values['outer_closure'];instance=values['instance'];claim=values['claim']
    require(all(v.get('remaining_owned_processes')=={} for v in (wd,site,outer))
        and set(wd['owned_pids'])<=set(outer['owned_pids'])
        and instance['service_pid'] in site['owned_pids'] and instance['service_pid'] in outer['owned_pids'],
        'Saved worker/site/controller process closure differs')
    require(claim.get('cell_id')==cell['cell_id'] and claim.get('campaign_root')==str(campaign)
        and claim.get('destination_absent_at_claim') is True and claim.get('preserve_historical_states') is True
        and claim.get('state_path')==str(Path(claim['scratch_root'])/'states'/folder.name)
        and instance.get('state_path')==claim['state_path'] and str(instance.get('job_id'))==str(plan['job_id'])
        and instance.get('domain')=='shopping_admin' and instance.get('scope')=='original'
        and instance.get('base_identity_sha256')==digest(instance['base_identity'])
        and instance.get('state_provenance_sha256')==digest(instance['state_provenance'])
        and instance['state_provenance']['copy_receipt']==refs['copy_receipt'],
        'Consumed primary claim or saved site identity differs')
    cms.validate_instance(instance)
    lease=values['lease'];launch=values['launch'];argv=launch.get('argv',[])
    require(lease==dict(task_id=f"webarena_lite_{cell['task_id']:03d}",actor_model=cell['actor'],selector=cell['selector'],
        instance_receipt=str(paths['instance']),instance_receipt_sha256=refs['instance']['sha256'])
        and launch.get('cell_id')==cell['cell_id'] and launch.get('resampling_forbidden') is True,
        'Worker launch/lease differs')
    for flag,value in [('--target',str(instance['service_pid'])),('--plan',str(paths['plan'])),
        ('--output',str(folder/'episodes')),('--task-id',str(cell['task_id'])),('--actor',cell['actor']),
        ('--selector',cell['selector']),('--lease',str(paths['lease']))]:
        require(argv.count(flag)==1 and argv[argv.index(flag)+1]==value,'Launched namespace/worker identity differs')
    require(argv.count('--execute')==1,'Saved command did not execute its worker')
    qc,qo=values['queue_claim'],values['queue_outcome']
    require(values['queue_manifest']==dict(cell_count=len(cohort['cells']),cells_sha256=digest(cohort['cells']))
        and qc.get('cell_id')==cell['cell_id'] and qc.get('cell_sha256')==digest(cell)
        and re.fullmatch('[0-9a-f]{32}',str(qc.get('owner')))
        and all(qo.get(k)==qc[k] for k in ('cell_id','cell_sha256','owner'))
        and qo.get('result') in [dict(status='invalid',reason='worker_exception',error_type=t)
            for t in ('RuntimeError','ValueError')]
        and qc['epoch']<=qo['epoch'],'Durable queue failure/owner differs')
    recovery=evidence.read_ref(inputs['recovery']);closed=evidence.read_ref(inputs['owner_closure'])
    finance=evidence.read_ref(inputs['financial_review']);scheduler=evidence.read_ref(closed['scheduler_evidence'])
    require(recovery.get('status')=='stopped_epoch_closed_with_independent_proofs'
        and recovery.get('new_provider_calls')==0 and recovery.get('new_allocations')==0
        and closed.get('verified') is True and closed.get('errors')==[] and closed.get('scheduler_terminal') is True
        and closed.get('scheduler_state') in ('FAILED','CANCELLED','TIMEOUT','COMPLETED')
        and closed.get('process_closure_evidence')==refs['outer_closure']
        and closed.get('all_owned_processes_reaped') is True and closed.get('job_id')==str(plan['job_id']),
        'Closed owning allocation/recovery proof differs')
    require(set(scheduler)=={'job_id','state','elapsed_seconds','exit_code'}
        and scheduler['job_id']==str(plan['job_id']) and scheduler['state']==closed['scheduler_state']
        and type(scheduler['elapsed_seconds']) is int and scheduler['elapsed_seconds']>0,
        'Terminal scheduler identity differs')
    broker_closure=evidence.read_ref(closed['broker_closure_evidence'])
    require(broker_closure.get('reaped') is True and broker_closure.get('pid') in outer['owned_pids'],
        'Saved broker closure is not empty')
    before,after=recovery['before'],recovery['after'];budget=campaign/'shared-budget.json.events.jsonl';broker=attempt/'broker.jsonl'
    require(finance.get('verified') is True and finance.get('errors')==[] and finance.get('unknown_reservations')==0
        and finance['audit_path']==str(broker) and finance['budget_prefix_bytes']==before['budget_prefix_bytes']
        and finance['budget_ledger_sha256']==before['budget_ledger_sha256'] and after.get('active_epoch_id') is None
        and after.get('unsettled_reservations')==0,'Closed financial prefix differs')
    events=list(_prefix(budget,after['budget_prefix_bytes'],after['budget_ledger_sha256']));last=events[-1][0]
    require(last.get('event')=='epoch_closed' and last.get('epoch_id')=='job-'+str(plan['job_id'])
        and last.get('event_sha256')==recovery['previous_closed_epoch_sha256']
        and last.get('recovery_evidence')==[inputs['owner_closure'],inputs['financial_review']],
        'Durable recovered closure event differs')
    broker_cell=f"webarena_lite_{cell['task_id']:03d}/"+cell['actor']+'__'+cell['selector']
    calls=audit_cell_calls(plan,broker_cell,episode/'api.jsonl',broker,budget,
        after['budget_prefix_bytes'],after['budget_ledger_sha256'],values['contract'],
        broker_prefix_bytes=finance['audit_prefix_bytes'],broker_prefix_sha256=finance['audit_sha256'])
    require(calls['verified'] and calls['paid_calls']==calls['local_calls']==calls['failed_calls']==0
        and calls['global_unknown_reservations']==0 and calls['cell_unknown_reservations']==0
        and calls['policy_within_caps'] is True and calls['global_charged_usd']==after['charged']
        and calls['global_dispatches']==after['dispatches']==finance['dispatches'],
        'Preauthentication zero-call classification or financial replay differs')
    for ref in refs.values():evidence.reference(ref,campaign)
    record=dict(cell_id=cell['cell_id'],released_task_id=cell['task_id'],actor_model=cell['actor'],selector=cell['selector'],
        domain=cell['domain'],task_config_sha256=cell['task_config_sha256'],tasks_sha256=cohort['source_sha256'],
        cohort_sha256=digest(cohort),protocol_sha256=cohort['protocol_sha256'],audit_verified=True,terminal=True,
        attempted=True,valid=False,success=None,native_success=None,effective_semantic_status='not_applicable',
        native_semantic_status='not_applicable')
    return dict(record=record,call_audit={k:v for k,v in calls.items() if k not in ('pairs','broker_only_failures')},
        complete_for_disposal=False,native_replay=None,raw_native_verdict_present=False,
        original_bytes_preserved=True,primary_rerun=False,disposal_performed=False,
        authentication_verified=False,reset_identity_source='saved_instance_and_copy_only_no_authenticated_episode_provenance')
