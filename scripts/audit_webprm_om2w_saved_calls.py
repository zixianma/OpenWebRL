#!/usr/bin/env python3
"""Read-only OM2W request/response and terminal-judge evidence reconciliation.

The frozen financial auditor performs the checks. Exact immutable financial and
broker prefixes are cached to avoid rereading the same large files for each
cell. One uncached call must reproduce the cached output exactly. No provider,
browser, scheduler mutation, result relabeling or resampling occurs.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

REPO=Path(__file__).resolve().parents[1]


def require(value, message):
    if not value: raise ValueError(message)


def read(path): return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):h.update(block)
    return h.hexdigest()


def ref(path):return dict(path=str(path),sha256=sha(path))


def signature(path):
    s=Path(path).stat()
    return s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns


def expected_dispositions(proposal):
    """Retain the original audit scope; admit only the exact new control contract."""
    if 'candidate_controls' not in proposal:
        return 450
    from scripts.launch_webprm_om2w_table5 import validate_proposal
    validate_proposal(proposal)
    return proposal['stage_cells']


def judge_binding(task, folder, pairs, *, judge_messages, judge_trajectory, model, tokens):
    """Rebuild the actual prompt/image and join the exact paid provider reply."""
    record=read(folder/'om2w-judge.json')
    fallback=read(folder/'protocol-fallback.json') if (folder/'protocol-fallback.json').exists() else None
    trajectory=judge_trajectory(read(folder/'rollout.json'),fallback,record['termination'])
    payload=dict(model=model,messages=judge_messages(task.actor_view['intent'],trajectory,(folder/'judge-final.jpg').read_bytes()),
                 max_completion_tokens=tokens)
    candidates=[p for p in pairs if p['request']['purpose']=='om2w_judge']
    require(len(candidates)==1,'Valid outcome must have exactly one actual paid judge request')
    pair=candidates[0]
    require(pair['request']['payload']==payload and pair['response'] is not None and pair['broker'] is not None,
            'Actual judge prompt/image/payload differs from saved task and actions')
    response=pair['response']['response'];choices=response.get('choices') or []
    require(response.get('model')==model and len(choices)==1 and choices[0].get('finish_reason')=='stop'
            and choices[0].get('message',{}).get('content')==record['raw'] and response.get('usage')==record['usage'],
            'Saved judge verdict is not the actual settled provider response')
    return pair['request']['reservation']


def audit_local_preflight(plan, groups, api, replay, financial):
    """Join repeated startup IDs by immutable epoch, retaining global accounting.

    The frozen helper selects by cell ID across the cumulative journal. Startup
    diagnostics reuse that ID after node failure; primary cells never do.
    """
    policies={g['dispatch_started']['policy_sha256'] for g in groups.values()}
    epochs=[e for e,row in replay['epochs'].items() if row['broker_policy_sha256'] in policies]
    require(len(policies)==len(epochs)==1,'Startup diagnostic epoch is ambiguous')
    target={k:v for k,v in replay['dispatches'].items()
            if v['cell_id']=='local_preflight' and v['epoch_id']==epochs[0]}
    require(set(target)==set(groups),'Startup financial and broker dispatch sets differ')
    chunks,api_sha,api_bytes=financial._api_chunks(api)
    require(len(chunks)==len(groups),'Startup API/dispatch counts differ')
    pairs=[]
    for (ident,group),chunk in zip(groups.items(),chunks):
        dispatch=target[ident];meta={k:v for k,v in dispatch.items() if k not in ('epoch_id','reservation_id')}
        require(set(group)=={'dispatch_started','dispatch_completed'},'Startup request did not complete cleanly')
        for event in group.values():
            require(event.get('identity_sha256')==financial._digest(plan)
                    and all(event.get(k)==v for k,v in meta.items()),'Startup durable identity mismatch')
        request=chunk['request'];response=chunk['response'];done=group['dispatch_completed']
        payload=request.get('payload')
        require(dispatch['local'] is True and dispatch['reservation_id'] is None
                and request.get('local') is True and request.get('reservation') is None
                and isinstance(payload,dict) and financial._digest(payload)==dispatch['payload_sha256']
                and payload.get('model')==dispatch['model'] and request.get('purpose')==dispatch['purpose'],
                'Startup request is paid or differs from durable admission')
        require(response is not None and chunk['error'] is None
                and done.get('journal_sha256')==chunk['terminal_prefix_sha256']
                and done.get('reservation_state')=='local' and done.get('reservation_id') is None
                and done.get('budget_owner')=='webprm_worker.completion'
                and response['response'].get('model')==dispatch['model']
                and response['response'].get('usage')==done.get('usage'),
                'Startup response/receipt mismatch')
        pairs.append(dict(request=request,response=response,error=None,broker=done,request_id=ident))
    return dict(verified=True,policy_within_caps=replay['charged']<=replay['cap'] and replay['overruns']==0,
        scope='Startup diagnostics joined by exact epoch; full cumulative financial replay retained',
        epoch_id=epochs[0],paid_calls=0,local_calls=len(target),api_request_count=len(chunks),
        cell_unknown_reservations=0,pairs=pairs,api_sha256=api_sha,api_bytes=api_bytes)


def audit(package, timeout=300):
    started=time.monotonic();deadline=started+timeout
    def check():require(time.monotonic()<deadline,'Bounded offline call audit deadline reached')
    package=Path(package).resolve();proposal=read(package/'proposal.json');campaign=Path(proposal['campaign_root'])
    roots=[Path(p).parents[1] for p in proposal['source_sha256'] if p.endswith('/openwebrl/webprm_journal_audit.py')]
    require(len(roots)==1,'One frozen financial implementation required')
    source=roots[0];sys.path.insert(0,str(source))
    from openwebrl import webprm_journal_audit as financial
    from openwebrl.webprm_webarena import digest
    from openwebrl.webprm_om2w_table5 import (load_tasks,OM2WTask,judge_messages,judge_trajectory,JUDGE,JUDGE_MAX_COMPLETION_TOKENS)
    for relative in ('openwebrl/webprm_journal_audit.py','openwebrl/webprm_worker.py','openwebrl/webprm_om2w_table5.py'):
        require(sha(source/relative)==sha(Path(financial.__file__).parents[1]/relative),
                'Executing dependencies differ from frozen financial/judge implementation')
    # The collector's additive node-failure activation checks were prepared
    # after launch. Load that separately; it does not modify raw outcomes.
    spec=importlib.util.spec_from_file_location('om2w_final_collector',REPO/'scripts/collect_webprm_om2w_table5.py')
    collector=importlib.util.module_from_spec(spec);spec.loader.exec_module(collector)
    rows=collector.stage_records(package);check()
    expected=expected_dispositions(proposal)
    require(len(rows)==expected and all(result is not None and state in ('recorded','audited_interrupted') for _,result,state in rows),
            'All fixed study dispositions must be reconciled first')
    tasks=load_tasks(proposal['task_file']);cell_map={cell['cell_id']:(cell,result,state) for cell,result,state in rows}
    pointers=[read(p) for p in (campaign/'forensic-dispositions').glob('*.json')]
    reviewed_refs={tuple(sorted(p['financial_helper_archive'].items())) for p in pointers}
    reviewed_ref=reviewed=None
    if not pointers and proposal.get('repair_of') is not None:
        # The stopped predecessor retains unfinished request reservations. Load
        # only the explicitly reviewed additive helper; verify cumulative replay
        # against the original frozen financial implementation below.
        from scripts.launch_webprm_om2w_table5 import repair_parent
        require(repair_parent(proposal) is not None, 'Interrupted accounting requires verified repair lineage')
        reviewed_ref = proposal.get('interruption_financial_helper')
        require(isinstance(reviewed_ref, dict) and set(reviewed_ref) == {'path', 'sha256'}
                and proposal['source_sha256'].get(reviewed_ref['path']) == reviewed_ref['sha256']
                and sha(reviewed_ref['path']) == reviewed_ref['sha256'],
                'Repair interruption helper must be immutable and source bound')
        reviewed_spec=importlib.util.spec_from_file_location('openwebrl._reviewed_om2w_financial',reviewed_ref['path'])
        reviewed=importlib.util.module_from_spec(reviewed_spec);sys.modules[reviewed_spec.name]=reviewed;reviewed_spec.loader.exec_module(reviewed)
    elif not pointers and 'candidate_controls' in proposal:
        pass  # Fresh study has no fabricated historical interruption helper.
    else:
        require(len(reviewed_refs)==1,'One already reviewed additive interruption helper required')
        reviewed_ref=dict(next(iter(reviewed_refs)))
        require(sha(reviewed_ref['path'])==reviewed_ref['sha256'],'Reviewed interruption helper changed')
        reviewed_spec=importlib.util.spec_from_file_location('openwebrl._reviewed_om2w_financial',reviewed_ref['path'])
        reviewed=importlib.util.module_from_spec(reviewed_spec);sys.modules[reviewed_spec.name]=reviewed;reviewed_spec.loader.exec_module(reviewed)
    journal=campaign/'shared-budget.json.events.jsonl';journal_hash=sha(journal);journal_size=journal.stat().st_size
    fingerprints={str(journal):signature(journal)};cache={};offsets=set();contexts={};plan_by_hash={}
    for attempt in sorted(campaign.glob('attempt-*')):
        check();context=read(attempt/'execution-context.json');contexts[attempt.name]=context
        broker=attempt/'broker.jsonl';require(broker.exists(),'Attempt broker journal missing')
        key=(str(broker),broker.stat().st_size,sha(broker))
        cached=list(financial._prefix(*key));cache[key]=cached;fingerprints[str(broker)]=signature(broker)
        for event,_,_ in cached:
            if event.get('event')=='dispatch_completed' and event.get('local') is False:offsets.add(event['budget_prefix_bytes'])
        for path in attempt.glob('cells/*/worker-plan.json'):
            plan=read(path);plan_by_hash[digest(plan)]=plan
    contract=read(sorted(campaign.glob('attempt-*'))[-1]/'financial-contract.json')
    replay=financial._financial(journal,journal_size,journal_hash,contract,offsets);check()
    require(reviewed is None or reviewed._financial(journal,journal_size,journal_hash,contract,offsets)==replay,
            'Reviewed interruption helper changes global financial replay')
    require(replay['active_epoch'] is None and replay['overruns']==0 and replay['charged']<=replay['cap'],
            'Final financial epoch is open or outside cap')
    def cached_financial(path,size,checksum,actual_contract,requested):
        require((str(path),size,checksum)==(str(journal),journal_size,journal_hash) and actual_contract==contract
                and set(requested)<=offsets,'Cache requested an unverified financial prefix')
        return replay
    def cached_prefix(path,size,checksum):
        key=(str(path),size,checksum)
        require(key in cache,'Cache requested an unverified broker prefix')
        yield from cache[key]
    counters=Counter();records=[];diagnostics=[];covered=set();paid=set();judge_reservations=set();equivalence=False;review_equivalence=False;reviewed_cells=0
    for attempt in sorted(campaign.glob('attempt-*')):
        check();broker=attempt/'broker.jsonl'
        broker_key=next(key for key in cache if key[0]==str(broker));events=[r for r,_,_ in cache[broker_key]]
        target_groups={}
        for event in events:
            if event.get('event') in ('dispatch_started','dispatch_completed','dispatch_failed'):
                target_groups.setdefault(event['cell_id'],{}).setdefault(event['request_id'],{})[event['event']]=event
        for cell_id,groups in target_groups.items():
            check();interrupted_local=[];interrupted_paid=[]
            for ident,group in groups.items():
                require('dispatch_started' in group,'Broker terminal record has no start')
                if not {'dispatch_completed','dispatch_failed'} & group.keys():
                    stopped_repair = False
                    if proposal.get('repair_of') is not None and cell_id in cell_map:
                        _, saved, disposition = cell_map[cell_id]
                        watchdog = saved.get('watchdog', {}) if saved else {}
                        stopped_repair = (disposition == 'recorded' and saved.get('valid') is False
                            and saved.get('success') == 0 and watchdog.get('supervisor_signal') == 15
                            and watchdog.get('remaining_owned_processes') == {}
                            and watchdog.get('child_exitcode') in (0, -15))
                    require(cell_id in cell_map and (cell_map[cell_id][2]=='audited_interrupted' or stopped_repair),
                            'Unfinished request lacks an exact reviewed closed-invalid disposition')
                    (interrupted_local if group['dispatch_started']['local'] else interrupted_paid).append(ident)
            if cell_id=='local_preflight':
                identity=next(iter(groups.values()))['dispatch_started']['identity_sha256']
                require(identity in plan_by_hash,'Startup diagnostic lacks its exact frozen assigned plan')
                plan=plan_by_hash[identity];api=attempt/'local-preflight-api.jsonl';folder=None
            else:
                require(cell_id in cell_map,'Broker contains an out-of-cohort cell')
                cell,result,state=cell_map[cell_id];key=hashlib.sha256(cell_id.encode()).hexdigest()
                assigned=attempt/'cells'/key;plan=read(assigned/'worker-plan.json')
                folder=assigned/'episodes'/('om2w_'+cell['task_id'])/(cell['actor']+'__'+cell['selector']);api=folder/'api.jsonl'
                require(cell_id not in covered,'A primary cell was dispatched in two attempts')
                covered.add(cell_id)
            kwargs=dict(broker_prefix_bytes=broker_key[1],broker_prefix_sha256=broker_key[2])
            helper=financial
            if interrupted_local or interrupted_paid:
                require(reviewed is not None,'Interrupted request requires independently reviewed additive evidence')
                helper=reviewed;reviewed_cells+=1
                kwargs.update(interrupted_local_request_ids=interrupted_local,interrupted_paid_request_ids=interrupted_paid)
            args=(plan,cell_id,api,broker,journal,journal_size,journal_hash,contract)
            fingerprints[str(api)]=signature(api)
            control=(helper is financial and not equivalence or helper is reviewed and not review_equivalence)
            if control and cell_id!='local_preflight':
                uncached=helper.audit_cell_calls(*args,**kwargs);check()
            else:uncached=None
            if cell_id=='local_preflight':
                calls=audit_local_preflight(plan,groups,api,replay,financial)
                diagnostics.append({k:v for k,v in calls.items() if k!='pairs'})
            else:
                try:
                    with patch.object(helper,'_financial',cached_financial),patch.object(helper,'_prefix',cached_prefix):
                        calls=helper.audit_cell_calls(*args,**kwargs)
                except ValueError as exc:
                    raise ValueError(f'{attempt.name}/{cell_id}: {exc}') from exc
            if uncached is not None:
                require(calls==uncached,'Cached and uncached financial audits differ')
                if helper is financial:equivalence=True
                else:review_equivalence=True
            require(calls['verified'] is True and calls['policy_within_caps'],'Per-cell financial evidence failed')
            for pair in calls['pairs']:
                reservation=pair['request']['reservation']
                if reservation is not None:
                    require(reservation not in paid,'Duplicate paid reservation across episode journals');paid.add(reservation)
            counters.update(paid=calls['paid_calls'],local=calls['local_calls'],api=calls['api_request_count'],unknown=calls['cell_unknown_reservations'])
            if folder is not None:
                cell,result,state=cell_map[cell_id];judge_id=None
                if result['valid']:
                    judge_id=judge_binding(OM2WTask.from_record(tasks[cell['task_id']]),folder,calls['pairs'],
                        judge_messages=judge_messages,judge_trajectory=judge_trajectory,model=JUDGE,tokens=JUDGE_MAX_COMPLETION_TOKENS)
                    require(judge_id not in judge_reservations,'Repeated judge reservation');judge_reservations.add(judge_id)
                records.append(dict(cell_id=cell_id,valid=result['valid'],success=result['success'],state=state,
                    financial={k:v for k,v in calls.items() if k not in ('pairs','broker_only_failures')},judge_reservation=judge_id))
    require(equivalence,'No uncached/cached equivalence control completed')
    require(not reviewed_cells or review_equivalence,'No reviewed-helper cache equivalence control completed')
    require(paid==set(replay['calls']),'Financial journal contains unmatched paid reservations')
    require(counters['paid']==replay['counts'][False] and counters['local']==replay['counts'][True],
            'Not every paid/local dispatch was reconciled')
    # Assigned cells which failed before a model call still need their original
    # outcome; the collector already verifies these closed invalid dispositions.
    zero_call=[cell_id for cell_id,(cell,result,state) in cell_map.items() if cell_id not in covered]
    require(all(not cell_map[cell_id][1]['valid'] for cell_id in zero_call),'Valid outcome lacks actual model calls')
    for path,before in fingerprints.items():require(signature(path)==before,'Evidence changed during audit')
    require(len(judge_reservations)==sum(result['valid'] for _,result,_ in rows),'Valid judge coverage differs')
    return dict(schema_version=1,status='saved_call_and_judge_bindings_verified',audit_verified=True,
        full_completion_audit_verified=False,scope='Actual saved financial/request/response/judge consistency under the frozen native protocol; final validity and closure review are separate',
        created_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-started,
        proposal=ref(package/'proposal.json'),auditor=ref(Path(__file__).resolve()),collector=ref(REPO/'scripts/collect_webprm_om2w_table5.py'),
        frozen_financial_helper=ref(Path(financial.__file__).resolve()),financial_journal=dict(path=str(journal),sha256=journal_hash,bytes=journal_size),
        reviewed_interruption_helper=reviewed_ref,reviewed_helper_cells=reviewed_cells,reviewed_helper_cache_equivalence_verified=review_equivalence,
        cached_uncached_equivalence_verified=equivalence,planned_cells=len(rows),cells_with_calls=len(covered),zero_call_invalid_cells=zero_call,
        actual_paid_dispatches=counters['paid'],actual_local_dispatches=counters['local'],actual_api_requests=counters['api'],
        unknown_reservations_retained=replay['unknown'],charged_usd=str(replay['charged']),cap_usd=str(replay['cap']),
        valid_native_judges_bound=len(judge_reservations),startup_diagnostics=diagnostics,records=records)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--package',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--timeout',type=int,default=300);args=p.parse_args()
    require(0<args.timeout<=600,'Audit timeout must be bounded by 600 seconds')
    target=args.output
    require(target.is_absolute() and target.resolve()==target and any(root in target.parents for root in
        (Path('/tmp'),Path('/gpfs/scrubbed/zixianma/openwebrl-runtime'))),'Output must remain in private approved runtime')
    result=audit(args.package,args.timeout)
    with target.open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('records','zero_call_invalid_cells')}))


if __name__=='__main__':main()
