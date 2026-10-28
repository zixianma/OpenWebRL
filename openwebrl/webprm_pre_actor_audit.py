"""Read-only reconstruction of closed, pre-actor readiness failures.

No native verdict, successful episode, resampling permission or disposal
permission is created. A consumed fixed claim becomes a terminal invalid
disposition only after source, queue, process and financial evidence agree.
"""
from pathlib import Path
import hashlib
import json
import importlib.util

from openwebrl.webprm_table5_analysis import digest, _validate_records

PROTOCOL = 'table5_pre_actor_readiness_failure_audit_v1'


def require(value, message):
    if not value:
        raise ValueError(message)


def load_financial_verifier(path):
    spec = importlib.util.spec_from_file_location('_webprm_closed_pre_actor_finance', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.audit_cell_calls


def verify(evidence, inputs, cell, cohort, campaign, folder):
    require(inputs.get('protocol') == PROTOCOL and inputs.get('cell') == cell and
            inputs.get('campaign_root') == str(campaign) and inputs.get('cell_folder') == str(folder),
            'Pre-actor audit identity differs')
    refs = inputs['artifacts']
    paths = {name: evidence.reference(ref, campaign) for name, ref in refs.items()}
    attempt = folder.parents[1]
    expected = dict(worker_plan=folder/'worker-plan.json', state_claim=folder/'state-claim.json',
                    queue_claim=campaign/'queue/claims'/(digest(cell['cell_id'])+'.json'),
                    queue_outcome=campaign/'queue/outcomes'/(digest(cell['cell_id'])+'.json'),
                    site_instance=folder/'site/instance.json', site_closure=folder/'site/closure.json',
                    copy_receipt=folder/'copy/receipt.json', context=attempt/'execution-context.json',
                    controller_closure=attempt/'controller-closure.json', storage=attempt/'storage-final.json',
                    financial_contract=attempt/'financial-contract.json', readiness=folder/'readiness-0.log',
                    scheduler=campaign/'operational-audits'/('pre-actor-readiness-'+attempt.name.removeprefix('attempt-'))/'scheduler.json')
    require(paths == expected, 'Pre-actor evidence paths differ from fixed primary')
    plan = evidence.read(paths['worker_plan']); context = evidence.read(paths['context'])
    claim = evidence.read(paths['state_claim']); queued = evidence.read(paths['queue_claim'])
    outcome = evidence.read(paths['queue_outcome']); instance = evidence.read(paths['site_instance'])
    require(plan.get('task_ids') == [cell['task_id']] and plan.get('actor_model') == cell['actor'] and
            plan.get('selector') == cell['selector'] and digest(plan.get('protocol_contract')) == cohort['protocol_sha256'],
            'Pre-actor worker task, actor, selector or protocol differs')
    require(context.get('job_id') == attempt.name.removeprefix('attempt-') and
            context['proposal'].get('cohort_sha256') == digest(cohort), 'Closed attempt cohort/job differs')
    for path, sha in plan['code_sha256'].items():
        evidence.reference(dict(path=path, sha256=sha))
    # The standalone financial verifier must be the same implementation pinned
    # by the worker plan, not an unbound reporting-time replacement.
    pinned = [(path, sha) for path, sha in plan['code_sha256'].items() if path.endswith('/webprm_journal_audit.py')]
    require(len(pinned) == 1, 'Exact frozen financial verification implementation required')
    financial_path = evidence.reference(dict(path=pinned[0][0], sha256=pinned[0][1]))
    audit_cell_calls = load_financial_verifier(financial_path)
    require(claim.get('cell_id') == cell['cell_id'] and claim.get('campaign_root') == str(campaign) and
            claim.get('destination_absent_at_claim') is True and
            claim.get('state_path') == str(Path(context['scratch'])/'states'/folder.name) and
            claim.get('policy_sha256') == digest(context['proposal']['ephemeral_policy']),
            'Fresh state claim differs')
    require(queued.get('cell_id') == outcome.get('cell_id') == cell['cell_id'] and
            queued.get('cell_sha256') == outcome.get('cell_sha256') == digest(cell) and
            queued.get('owner') == outcome.get('owner') and isinstance(queued.get('owner'), str) and
            outcome.get('result') == dict(status='invalid', reason='worker_exception', error_type='TimeoutError') and
            queued['epoch'] <= outcome['epoch'] < context['deadline'], 'Raw queue failure/owner differs')
    require(instance.get('job_id') == context['job_id'] and instance.get('domain') == cell['domain'] and
            instance.get('state_path') == claim['state_path'], 'Website instance differs from failed state')
    copy = evidence.read(paths['copy_receipt'])
    require(copy.get('status') == 'completed' and copy.get('returncode') == 0 and
            copy.get('state_path') == claim['state_path'], 'Failed primary never finished its state copy')
    for name in ('site_closure', 'controller_closure'):
        require(evidence.read(paths[name]).get('remaining_owned_processes') == {}, 'Actual process closure is incomplete')
    scheduler = evidence.read(paths['scheduler'])
    require(scheduler.get('job_id') == context['job_id'] and scheduler.get('state') == 'COMPLETED' and
            type(scheduler.get('elapsed_seconds')) is int and scheduler['elapsed_seconds'] > 0 and
            evidence.read(paths['controller_closure']).get('child_exitcode') == 0,
            'Exact terminal scheduler and normal controller closure required')
    storage = evidence.read(paths['storage'])
    require(storage.get('scratch') == context['scratch'] and
            0 <= storage['apparent_bytes'] <= storage['cap_bytes'] and
            0 <= storage['campaign_apparent_bytes'] <= storage['campaign_cap_bytes'], 'Closed storage accounting differs')
    require(not (folder/'worker-launch.json').exists() and not (folder/'cell-outcome.json').exists() and
            not (folder/'disposal-intent.json').exists() and not (folder/'disposal-receipt.json').exists(),
            'Pre-actor disposition conflicts with launched worker or disposal')
    for path in (folder/'episodes').rglob('*'):
        require(not path.is_file() or path.name == 'api.jsonl' and path.stat().st_size == 0,
                'Pre-actor cell contains actor/terminal evidence')
    budget = inputs['budget']; broker = inputs['broker']
    require(budget['path'] == str(campaign/'shared-budget.json.events.jsonl') and
            broker['path'] == str(attempt/'broker.jsonl'), 'Financial evidence belongs to another owner')
    for boundary in (budget, broker):
        evidence.prefix(boundary['path'], boundary['bytes'], boundary['sha256'], campaign)
    with Path(budget['path']).open('rb') as stream:
        last = json.loads(stream.read(budget['bytes']).splitlines()[-1])
    require(last.get('event') == 'epoch_closed' and last.get('epoch_id') == 'job-'+context['job_id'],
            'Financial prefix does not close this allocation epoch')
    api = folder/'episodes'/f"webarena_lite_{cell['task_id']:03d}"/(cell['actor']+'__'+cell['selector'])/'api.jsonl'
    calls = audit_cell_calls(plan, f"webarena_lite_{cell['task_id']:03d}/"+cell['actor']+'__'+cell['selector'],
        api, broker['path'], budget['path'], budget['bytes'], budget['sha256'],
        evidence.read(paths['financial_contract']), broker_prefix_bytes=broker['bytes'],
        broker_prefix_sha256=broker['sha256'])
    require(calls['verified'] and calls['paid_calls'] == calls['local_calls'] == calls['failed_calls'] == 0 and
            calls['global_unknown_reservations'] == 0, 'Pre-actor cell has dispatched or unresolved requests')
    record = dict(cell_id=cell['cell_id'], released_task_id=cell['task_id'], actor_model=cell['actor'],
        selector=cell['selector'], domain=cell['domain'], task_config_sha256=cell['task_config_sha256'],
        tasks_sha256=cohort['source_sha256'], cohort_sha256=digest(cohort), protocol_sha256=cohort['protocol_sha256'],
        audit_verified=True, terminal=True, attempted=True, valid=False, success=None, native_success=None,
        effective_semantic_status='not_applicable', native_semantic_status='not_applicable')
    _validate_records(cohort, {cell['cell_id']: cell}, [dict(record, audit_sha256='0'*64)])
    return dict(record=record, call_audit={k:v for k,v in calls.items() if k != 'pairs'},
                raw_native_verdict_present=False, raw_queue_preserved=True, complete_for_disposal=False)
