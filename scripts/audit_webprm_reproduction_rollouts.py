#!/usr/bin/env python3
"""Offline audit of paired rollout identities, paid attempts and native verdicts.

Uses the frozen native replay implementation. Never starts a browser, provider,
service, allocation or retry. Scheduler accounting must be supplied separately.
"""
from __future__ import annotations
import argparse
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import sys


def require(condition, message):
    if not condition: raise ValueError(message)


def read(path):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path and not path.is_symlink(), 'Canonical evidence path required')
    require(path.stat().st_size <= 512*1024**2, 'Evidence file exceeds audit bound')
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def events(path):
    path = Path(path)
    if not path.exists(): return []
    require(path.stat().st_size <= 512*1024**2, 'Journal exceeds audit bound')
    raw = path.read_bytes()
    require(not raw or raw.endswith(b'\n'), 'Incomplete journal tail')
    return [json.loads(line) for line in raw.splitlines()]


def audit_calls(plan, cell_id, journal, broker, budget, used):
    """Join actual requests/responses to unique broker dispatches and money."""
    mini, large = 'gpt-4o-mini-2024-07-18', 'gpt-4o-2024-05-13'
    actor = cell_id.split('/')[-1].split('__')[0]
    routes = {'policy': actor, 'state_diff': mini, 'terminal': mini, 'native_fuzzy': large}
    from openwebrl.webprm_arbiter_harness import load_arbiter_harness
    from openwebrl.webprm_webarena_worker import single_action_system_prompt
    harness = load_arbiter_harness(plan['arbiter_source'])
    terminal_system, _ = harness.build_terminal_prompt(intent='', observation='', current_url='', trajectory='',
        state_diff='', previous_terminal_action=None)
    systems = dict(terminal=terminal_system+plan['reproduction_diagnostic']['terminal_system_suffix'],
                   state_diff=harness.state_diff_system_prompt,
                   policy=single_action_system_prompt(harness.policy_system_prompt, version=plan['no_search_prompt_version']))
    starts, completed, failed = {}, {}, {}
    for row in broker:
        require(row.get('cell_id') == cell_id and row.get('identity_sha256') == digest(plan)
                and row.get('local') is False and row.get('model') == routes.get(row.get('purpose')),
                'Broker cell/model/route identity differs')
        kind = row.get('event'); ident = row.get('request_id')
        target = {'dispatch_started': starts, 'dispatch_completed': completed, 'dispatch_failed': failed}.get(kind)
        require(target is not None and ident not in target, 'Unknown/duplicate broker event')
        target[ident] = row
        if kind != 'dispatch_started':
            require(ident in starts and all(row.get(k) == starts[ident].get(k) for k in
                ('purpose','model','local','payload_sha256','policy_sha256')), 'Broker completion has no matching start')
    require(set(completed).isdisjoint(failed) and set(completed)|set(failed) == set(starts),
            'Closed broker has an unresolved or contradictory dispatch')
    require(len(starts) <= 90, 'Per-episode dispatch cap exceeded')
    pairs = []; pending = None; matched = set()
    for row in journal:
        kind = row.get('event')
        if kind == 'request':
            require(pending is None, 'Overlapping API requests in serial episode')
            ident = row.get('reservation'); payload = row.get('payload', {}); purpose = row.get('purpose')
            require(type(ident) is int and 0 <= ident < len(budget['calls']) and ident not in used,
                    'Missing/duplicate paid reservation')
            require(row.get('local') is False and payload.get('model') == routes.get(purpose), 'Paid route differs')
            if purpose != 'native_fuzzy':
                require(payload.get('temperature') == 1. and payload.get('top_p') == .9
                        and payload.get('max_tokens') == 2048 and payload.get('n',1) == 1,
                        'Actor/helper sampling changed')
                require(payload.get('messages',[{}])[0] == dict(role='system',content=systems[purpose]),
                        'Actual policy/helper system prompt differs')
                if purpose == 'policy':
                    require(payload.get('response_format') == {'type':'json_object'}, 'Policy output envelope changed')
            wanted = [value for request_id,value in starts.items() if request_id not in matched
                      and value.get('purpose') == purpose and value.get('payload_sha256') == digest(payload)]
            settled = [v for v in wanted if completed.get(v['request_id'],{}).get('reservation_id') == ident]
            require(len(settled) <= 1 and (settled or wanted), 'API request lacks its actual broker dispatch')
            dispatch = (settled or wanted)[0]; matched.add(dispatch['request_id']); used.add(ident)
            require(budget['calls'][ident]['model'] == payload['model'], 'Reservation model differs')
            pending = dict(request=row, response=None, broker=completed.get(dispatch['request_id'], dispatch))
        elif kind in ('response', 'response_error'):
            require(pending is not None and row.get('reservation') == pending['request']['reservation'],
                    'API reply lacks its request')
            pending['response'] = row if kind == 'response' else None
            pairs.append(pending); pending = None
        else: raise ValueError('Unknown API journal event')
    if pending is not None: pairs.append(pending)
    # Starts without journal entries are conservative failed admissions, e.g.
    # a refused budget reservation. They still consume the dispatch cap.
    require(set(starts)-matched <= set(failed), 'Unjournaled broker dispatch is not an explicit failure')
    for pair in pairs:
        request, response, dispatch = pair['request'], pair['response'], pair['broker']
        financial = budget['calls'][request['reservation']]
        if financial['state'] == 'reserved': continue
        require(response is not None and financial['state'] == 'settled', 'Settled request has no response')
        actual = response['response']; usage = actual.get('usage', {})
        require(actual.get('model') == request['payload']['model'] and financial.get('usage') == usage,
                'Provider model or metered usage differs')
        require(all(type(usage.get(k)) is int and usage[k] >= 0 for k in ('prompt_tokens','completion_tokens')),
                'Invalid token usage')
        rates = [Decimal(str(x)) for x in plan['prices_usd_per_token'][actual['model']]]
        charge = usage['prompt_tokens']*rates[0]+usage['completion_tokens']*rates[1]
        require(math.isclose(float(charge), financial['charged'], rel_tol=0, abs_tol=1e-9), 'Token charge differs')
        if dispatch['event'] == 'dispatch_completed':
            require(dispatch.get('reservation_id') == request['reservation'] and dispatch.get('reservation_state') == 'settled',
                    'Broker settlement differs')
    return dict(pairs=pairs, dispatches=len(starts), journal_requests=len(pairs),
                failures=len(failed), incomplete_reservations=sum(budget['calls'][p['request']['reservation']]['state']=='reserved' for p in pairs))


def reset_proof(folder, owned, cell, result, seen):
    """Audit retained original image/build/copy chain without restoring state."""
    from openwebrl.webprm_reddit_provenance import (ORIGINAL_MANIFEST, ORIGINAL_CONFIG,
        validate_original_verification, validate_original_download)
    state = read(folder/'copy/receipt.json'); instance = read(folder/'site/instance.json'); lease = read(folder/'lease.json')
    build_path = Path(owned['output'])/'original-build/receipt.json'; build = read(build_path)
    state_path = str(Path(owned['scratch'])/f"state-{cell['index']:02d}")
    require(state_path not in seen, 'Fresh-state path reused'); seen.add(state_path)
    require(state.get('status') == 'completed' and state.get('returncode') == 0
            and state.get('state_path') == state_path and state.get('source_sandbox_path') == build.get('sandbox_path')
            and state.get('argv') == ['cp','-a','--reflink=auto','--',build['sandbox_path'],state_path]
            and state.get('build_receipt_path') == str(build_path) and state.get('build_receipt_sha256') == sha(build_path),
            'Fresh copy/build linkage differs')
    require(build.get('status') == 'completed' and build.get('returncode') == 0
            and build.get('source_manifest_sha256') == ORIGINAL_MANIFEST and build.get('config_sha256') == ORIGINAL_CONFIG,
            'Original build identity differs')
    require(instance.get('state_path') == state_path and instance.get('scope') == 'original'
            and instance.get('image_sha256') == ORIGINAL_MANIFEST and instance.get('config_sha256') == ORIGINAL_CONFIG
            and instance.get('verified_original_layers') == 29 and instance.get('job_id') == owned['job_id']
            and instance.get('state_provenance_path') == str(folder/'copy/receipt.json')
            and instance.get('state_provenance_sha256') == sha(folder/'copy/receipt.json')
            and result.get('reset_id') == sha(folder/'copy/receipt.json'), 'Instance/reset differs')
    require(lease == dict(task_id=f"webarena_lite_{cell['task_id']:03d}", actor_model=cell['actor'], selector='no_search',
        instance_receipt=str(folder/'site/instance.json'),instance_receipt_sha256=sha(folder/'site/instance.json')),
        'Fresh instance lease differs')
    validate_original_verification(state['original_verification_path'], state['original_verification_sha256'])
    validate_original_download(build['download_receipt_path'], build['download_receipt_sha256'])
    return instance


def native_paths(episode):
    """Mirror actual durable locations, including the native-evaluation folder."""
    episode = Path(episode)
    files = dict(trajectory='trajectory.json', verdict='native-verdict.json', terminal='terminal.json',
                 outcome='outcome-before-close.json', terminal_screenshot='terminal.png')
    paths = {key:episode/'native-evaluation'/name for key,name in files.items()}
    paths.update({key:episode/name for key,name in dict(rollout='rollout.json', provenance='provenance.json',
        authentication='authentication.json', evaluator_inputs='native-evaluator-transcript.json',
        result='result.json', native_fuzzy='native-fuzzy.jsonl').items()})
    return paths


def audit_parity_actor(plan, view, rollout, calls, *, actor_model):
    """Replay the actual actor, including every prompt, filter and past label."""
    from copy import deepcopy
    from openwebrl.webprm_webarena_worker import ViewerActor
    pending = [pair for pair in calls['pairs'] if pair['request']['purpose'] != 'native_fuzzy']
    def request(config, payload, journal, **kwargs):
        require(bool(pending), 'Actor replay requested an extra provider call')
        pair = pending.pop(0)
        require(pair['request']['purpose'] == kwargs['purpose'] and pair['request']['payload'] == payload,
                'Replayed actor prompt/model/sampling differs from actual paid request')
        require(pair['response'] is not None, 'Actor replay needs the saved actual response')
        return deepcopy(pair['response']['response'])
    require(actor_model in plan['actors'], 'Replay actor is outside the frozen actor list')
    actor_config = dict(plan, actor_model=actor_model, selector='no_search')
    actor = ViewerActor(actor_config, '/tmp/unused-parity-audit-journal', request=request)
    history = []
    for row in rollout:
        proposal = row['proposal']; context = proposal['context']
        observation = dict(text=context['text_observation'], url=context['current_url'],
                           last_action_error=history[-1].get('error') if history else None)
        require(actor.propose(view, observation, deepcopy(history)) == proposal,
                'Replayed actor candidate rejection/history/proposal differs')
        selected = row['selected_index']
        history.append(dict(proposal['candidates'][selected], error=row.get('execution_error'))
            if selected is not None else dict(action='[no action executed]', thought='', error=row['execution_error']))
    require(not pending, 'Saved auxiliary/policy calls were omitted from actor replay')


def audit_time(scheduler, proposal, approval):
    elapsed=scheduler.get('elapsed_seconds');limit=proposal['resources']['total_seconds']
    prior=approval.get('prior_scheduler_seconds',0);cap=approval.get('original_total_cap_seconds',14400)
    require(type(elapsed) is int and type(prior) is int and type(cap) is int
            and 0<=elapsed<=limit and prior>=0 and cap==14400 and prior+limit<=cap,
            'Exact attempt and cumulative scheduler caps required; failed time cannot reset')
    return dict(prior_scheduler_seconds=prior,current_elapsed_seconds=elapsed,
                cumulative_elapsed_seconds=prior+elapsed,current_limit_seconds=limit,original_cap_seconds=cap)


def audit(root, scheduler):
    root = Path(root).resolve(); proposal = read(root/'proposal.json')
    sys.path.insert(0, str(root/'source'))
    from scripts import launch_webprm_reproduction_rollouts as launcher
    if hasattr(launcher, 'configure_study'):
        launcher.configure_study(proposal.get('study', 'stopping_rule_20_v1'))
    else:
        require(proposal.get('study', 'stopping_rule_20_v1') == 'stopping_rule_20_v1',
                'Frozen legacy controller cannot audit a prospective study')
    from scripts.launch_webprm_reproduction_rollouts import validate, approval_check, cells, DIAGNOSTIC
    from scripts.run_webprm_webarena_pilot import configured_tasks, validate_plan
    from scripts.audit_webprm_table5_cell import audit_native
    approval=read(root/'approved-request.json')
    validate(proposal); approval_check(proposal, approval)
    limits = proposal['limits']
    prior_limits = (launcher.prior_limits() if hasattr(launcher, 'prior_limits') else
                    dict(charged_usd=0, paid_requests=0, scheduler_seconds=0, scratch_bytes=0, durable_bytes=0))
    times=audit_time(scheduler,proposal,approval)
    if proposal.get('study') in ('viewer_parity_8_v1', 'viewer_parity_4_v1'):
        require(times['prior_scheduler_seconds'] == prior_limits['scheduler_seconds'],
                'Parity successor omitted the completed pilot scheduler charges')
    elif times['prior_scheduler_seconds']:
        replacement=approval['replacement_request'];require(sha(replacement['path'])==replacement['sha256'],'Replacement request changed')
        request=read(replacement['path']);prior_ref=request['failed_attempt_audit']
        require(sha(prior_ref['path'])==prior_ref['sha256'],'Prior failure audit changed')
        prior=read(prior_ref['path'])
        require(prior['audit_verified'] is True and prior['owned_processes_closed'] is True
                and prior['job_id']==approval['prior_job_id']
                and prior['elapsed_scheduler_seconds']==times['prior_scheduler_seconds']
                and prior['primary_claims']==0 and prior['paid_requests']==0 and prior['charged_usd']==0,
                'Replacement cannot omit prior time, calls or primary claims')
    campaign = root/'campaign'; allocation = read(campaign/'allocation.json'); job = allocation['job_id']
    output = campaign/('attempt-'+job); owned = read(output/'execution-context.json')
    require(owned['proposal'] == proposal and owned['approval'] == allocation['approval'], 'Controller execution identity differs')
    require(scheduler['job_id'] == job and scheduler['state'] in ('COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL')
            and times['current_elapsed_seconds']<=proposal['resources']['total_seconds'],
            'Exact terminal scheduler accounting within total time cap required')
    closure = read(output/'controller-closure.json')
    require(closure.get('remaining_owned_processes') == {}, 'Unclosed controller processes')
    budget_path = campaign/'shared-budget.json'
    budget = read(budget_path) if budget_path.exists() else dict(cap=limits['api_usd'],charged=0.,calls=[])
    require(budget['cap'] == limits['api_usd'] and math.isfinite(budget['charged'])
            and 0 <= budget['charged'] <= limits['api_usd'], 'Shared API cap differs')
    used=set(); seen_states=set(); records=[]; dispatches=0; previous_finish=0
    for cell in cells():
        folder = output/f"cell-{cell['index']:02d}"
        if not (folder/'finished.json').exists():
            records.append(dict(**cell, recorded=False, claimed=(folder/'claim.json').exists(), valid=False, success=0)); continue
        require(read(folder/'claim.json') == cell and read(folder/'finished.json')['cell'] == cell, 'Fixed cell claim changed')
        plan = read(folder/'plan.json'); validate_plan(plan)
        template = read(root/('plan-'+cell['condition']+'.json'))
        mutable = {'authorization_status','job_id','deadline','api_usd_cap','broker_socket'}
        require({k:v for k,v in plan.items() if k not in mutable} == {k:v for k,v in template.items() if k not in mutable}
                and plan['api_usd_cap'] == limits['api_usd'] and plan['job_id'] == job and plan['deadline'] == owned['deadline'],
                'Dispatched science/condition/budget differs')
        task = next(task for task,raw in configured_tasks(plan) if raw['task_id'] == cell['task_id'])
        name = cell['actor']+'__no_search'; episode = folder/'episodes'/task.execution_id/name
        result = read(episode/'result.json')
        require(result.get('identity_sha256') == digest(plan) and result.get('actor_model') == cell['actor']
                and result.get('selector') == 'no_search' and result.get('task_id') == task.execution_id
                and type(result.get('valid')) is bool and type(result.get('attempted')) is bool
                and sha(episode/'result.json') == read(folder/'finished.json')['result_sha256'], 'Saved outcome identity differs')
        watchdog = read(episode/'watchdog.json'); site = read(folder/'site/closure.json')
        require(watchdog == result.get('watchdog') and watchdog.get('remaining_owned_processes') == {}
                and site.get('remaining_owned_processes') == {} and read(folder/'broker-closure.json').get('reaped') is True,
                'Worker/site/broker closure is incomplete')
        require(read(folder/'copy/request.json')['started_epoch'] >= previous_finish, 'Site copies overlap preceding services')
        previous_finish = site['finished_epoch']
        calls = audit_calls(plan,task.execution_id+'/'+name,events(episode/'api.jsonl'),events(folder/'broker.jsonl'),budget,used)
        dispatches += calls['dispatches']
        native = None
        if result['valid']:
            require(calls['failures'] == 0 and calls['incomplete_reservations'] == 0,
                    'Valid episode has failed or unknown paid calls')
            instance = reset_proof(folder,owned,cell,result,seen_states)
            require(read(episode/'provenance.json').get('reproduction_diagnostic') == template['reproduction_diagnostic'],
                    'Saved diagnostic provenance differs')
            paths = native_paths(episode)
            outcome, native = audit_native(task,plan,paths,result,calls,instance)
            require(native.get('verified') is True, 'Native evaluator replay failed')
            if plan.get('viewer_parity_diagnostic') is not None:
                audit_parity_actor(plan, task.actor_view, read(episode/'rollout.json'), calls, actor_model=cell['actor'])
        else:
            require(result.get('error') or result.get('error_type') or watchdog.get('reason'), 'Invalid outcome lacks failure evidence')
            outcome = dict(success=0,native_success=None)
        records.append(dict(**cell,recorded=True,claimed=True,attempted=result['attempted'],valid=result['valid'],
                            success=outcome['success'],native_success=outcome['native_success'],native_replay=native,
                            paid_dispatches=calls['dispatches'],unknown_reservations=calls['incomplete_reservations']))
    require(dispatches <= limits['paid_requests'] and dispatches+prior_limits['paid_requests'] <= 1800,
            'Total dispatch cap exceeded')
    require(used == set(range(len(budget['calls']))), 'Extra or unaudited paid reservations')
    charged = Decimal(0)
    for index,row in enumerate(budget['calls']):
        require(row['id'] == index and row['state'] in ('reserved','settled'), 'Financial reservation state differs')
        amount = row['reserved'] if row['state'] == 'reserved' else row['charged']
        require(type(amount) in (int,float) and math.isfinite(amount) and amount >= 0, 'Invalid charged/reserved amount')
        charged += Decimal(str(amount))
    require(math.isclose(float(charged),budget['charged'],rel_tol=0,abs_tol=1e-8), 'Budget aggregate differs')
    overlay = read(campaign/'budget-reservation-overlay.json'); parent_ref = overlay['parent']['parent_budget']
    require(overlay['parent'] == read(DIAGNOSTIC/'budget-reservation-overlay.json')
            and overlay['closed_diagnostic_charged'] == read(DIAGNOSTIC/'budget.json')['charged']
            and overlay['reserved_existing_cap_usd'] == limits['api_usd'] and overlay['additional_budget_usd'] == 0
            and overlay['child_budget'] == str(budget_path), 'API funding overlay differs')
    require(overlay.get('closed_rollout_charged',0) == prior_limits['charged_usd'], 'Closed rollout charges omitted')
    require(sha(parent_ref['path']) == parent_ref['sha256'], 'Parent campaign budget changed')
    combined = read(parent_ref['path'])['charged']+overlay['closed_diagnostic_charged']+prior_limits['charged_usd']+budget['charged']
    require(combined <= 250, 'Combined original API cap exceeded')
    complete = all(r['recorded'] for r in records)
    if complete:
        require(closure.get('child_exitcode') == 0 and read(output/'completed.json').get('episodes') == len(cells()),
                'Fixed outcomes lack controller completion')
        require(read(output/'storage-final.json')['apparent_bytes']+prior_limits['scratch_bytes'] <= limits['scratch_bytes'],
                'Cumulative retained scratch cap exceeded')
    groups=[]
    for actor in dict.fromkeys(cell['actor'] for cell in cells()):
        for condition in dict.fromkeys(cell['condition'] for cell in cells()):
            rows=[r for r in records if r['actor']==actor and r['condition']==condition]
            valid=sum(r['valid'] for r in rows); success=sum(r['success'] for r in rows)
            groups.append(dict(actor=actor,condition=condition,planned=len(rows),recorded=sum(r['recorded'] for r in rows),
                               valid=valid,successes=success,overall_percent=100*success/len(rows),
                               valid_only_percent=None if not valid else 100*success/valid))
    return dict(schema_version=1,audit_verified=True,verified_complete=complete,job_id=job,
                scope='Saved evidence/source/accounting consistency and native evaluator replay; not independent website truth or historical paper parity',
                proposal_sha256=digest(proposal),auditor_sha256=sha(__file__),records=records,groups=groups,
                paid_dispatches=dispatches,charged_usd=budget['charged'],combined_charged_usd=combined,
                scheduler=scheduler,time_accounting=times,all_time_cap_seconds=14400,resources=proposal['resources'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--scheduler',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    value=audit(args.root,read(args.scheduler))
    path=args.output
    require(path.is_absolute() and path.resolve()==path and any(root in path.parents for root in
            (Path('/gpfs/scrubbed/zixianma/openwebrl-runtime'),Path('/tmp'))), 'Audit output outside approved roots')
    with path.open('x') as stream:json.dump(value,stream,indent=2,sort_keys=True);stream.write('\n')
    print(json.dumps({k:v for k,v in value.items() if k not in ('records','groups')}))


if __name__=='__main__':main()
