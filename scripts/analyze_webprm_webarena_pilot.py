#!/usr/bin/env python3
"""Audit saved five-selector Reddit outcomes and cumulative campaign accounting.

Reads only durable files; never calls a browser, model, scheduler or provider.
The aggregate contains no raw tasks, screenshots, credentials or host paths.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import re
from pathlib import Path
import sys
from types import SimpleNamespace

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from openwebrl.webprm_webarena import digest

ROWS=('no_search','gpt4omini','shepherd8b','arbiter7b','piotr_selectionarm')
ACTOR='gpt-4o-mini-2024-07-18'
JUDGE='gpt-4o-2024-05-13'
PRICES={ACTOR:(.15/1e6,.60/1e6),JUDGE:(5/1e6,15/1e6)}
CALL_GROUPS=('actor_aux','selector','native_judge','other')
CALL_COUNTS=('requests','complete_responses','broker_started','broker_completed','matched_complete')
SELECTOR_PURPOSES={'gpt4omini':('selector_pair',),'arbiter7b':('selector_pair',),
    'shepherd8b':('selector_checklist','selector_pointwise'),'piotr_selectionarm':('selector_piotr',)}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def proposal_digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',', ':'),allow_nan=False).encode()).hexdigest()


def events(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def summarize(records, launched=(), call_evidence=None, planned_rows=ROWS):
    if tuple(planned_rows) not in (ROWS,('no_search',)):
        raise ValueError('Only the fixed comparison or separate NoSearch diagnostic is supported')
    launched=set(launched)
    if launched-set(planned_rows):raise ValueError('Unplanned launched cell')
    indexed={}
    for record in records:
        name=record['selector']
        if name not in planned_rows or name in indexed:
            raise ValueError('Duplicate or unplanned fixed cell')
        if (type(record.get('attempted')) is not bool or type(record.get('valid')) is not bool
                or type(record.get('success')) is not int or record['success'] not in (0,1)
                or record['valid'] and not record['attempted'] or record['success'] and not record['valid']):
            raise ValueError('Invalid outcome accounting')
        indexed[name]=record
    rows=[]
    for name in planned_rows:
        result=indexed.get(name)
        attempted=int(bool(result and result['attempted']))
        valid=int(bool(result and result['valid']))
        success=result['success'] if result else 0
        nested=result.get('error') if result and isinstance(result.get('error'),dict) else {}
        unknown=int(result is None and name in launched)
        rows.append(dict(selector=name,planned=1,recorded=int(result is not None),attempted=attempted,
            valid=valid,invalid=attempted-valid,missing=1-int(result is not None),successes=success,
            launched=int(name in launched),attempted_unknown=unknown,
            recorded_unattempted=int(result is not None and not result['attempted']),
            not_launched=int(result is None and name not in launched),
            overall_success_pct=100.*success if attempted else None,
            valid_only_success_pct=100.*success/valid if valid else None,
            error_type=(result.get('error_type') or nested.get('type')) if result else None,
            error_stage=nested.get('stage'),watchdog_reason=result.get('watchdog',{}).get('reason') if result else None))
        calls=(call_evidence or {}).get(name,empty_call_evidence(name))
        rows[-1].update(call_evidence=calls,guided_selector_exercised=calls['guided_selector_exercised'])
    totals={key:sum(row[key] for row in rows) for key in ('planned','recorded','attempted','valid','invalid','missing','successes',
        'launched','attempted_unknown','recorded_unattempted','not_launched')}
    guided=all(row['guided_selector_exercised'] is True for row in rows if row['selector']!='no_search')
    return dict(status='fixed_attempts_recorded' if totals['attempted']==len(planned_rows) else 'partial',
        all_native_verdicts_valid=totals['valid']==len(planned_rows),all_guided_selectors_exercised=guided,
        all_mechanics_valid=False,rows=rows,totals=totals,
        mechanics_semantics='True only after native replay validates all five verdicts, every row has a matched complete actor/aux response, all four guided selectors have matched complete scoring responses, and all registered controllers are closed. Counts do not prove all actions or candidate translations correct.',
        count_semantics='attempted counts confirmed results only; attempted_unknown means launched with no durable result; missing counts absent result files.',
        overall_success_pct=100.*totals['successes']/len(planned_rows) if totals['attempted']==len(planned_rows) else None,
        valid_only_success_pct=100.*totals['successes']/totals['valid'] if totals['valid'] else None)


def empty_call_evidence(name):
    return dict(journal_present=False,by_group={group:{key:0 for key in CALL_COUNTS} for group in CALL_GROUPS},
        by_purpose={},guided_selector_exercised=None if name=='no_search' else False)


def audit_call_evidence(folder, name, plan, broker_events):
    """Count durable episode calls, excluding startup probes and other cells.

    A complete response means one nonempty stop choice from the requested model.
    Matched completion also needs a unique completed broker dispatch with the
    same cell, plan, payload, mode and reservation. This does not replay selectors.
    """
    from openwebrl.webprm_broker import payload_sha256
    evidence=empty_call_evidence(name)
    journal=folder/'api.jsonl';evidence['journal_present']=journal.exists()
    def bucket(purpose):
        if purpose in ('policy','state_diff','terminal'):return 'actor_aux'
        if purpose in ('selector_pair','selector_checklist','selector_pointwise','selector_piotr'):return 'selector'
        return 'native_judge' if purpose=='native_fuzzy' else 'other'
    def increment(purpose,key):
        purpose=purpose if isinstance(purpose,str) else 'unspecified'
        evidence['by_group'][bucket(purpose)][key]+=1
        evidence['by_purpose'].setdefault(purpose,{k:0 for k in CALL_COUNTS})[key]+=1
    identity=dict(identity_sha256=digest(plan),cell_id='webarena_lite_005/'+ACTOR+'__'+name)
    routes={purpose:(ACTOR,False) for purpose in ('policy','state_diff','terminal')}
    routes['native_fuzzy']=(JUDGE,False)
    if name in ('gpt4omini','arbiter7b'):
        routes['selector_pair']=(ACTOR,False) if name=='gpt4omini' else (plan.get('arbiter_model'),True)
    elif name=='shepherd8b':
        routes.update({purpose:(plan.get('shepherd_model'),True) for purpose in SELECTOR_PURPOSES[name]})
    elif name=='piotr_selectionarm':routes['selector_piotr']=(plan.get('piotr_model'),True)
    own=[event for event in broker_events if all(event.get(k)==v for k,v in identity.items())]
    starts={event['request_id']:event for event in own if event.get('event')=='dispatch_started'}
    for event in starts.values():increment(event.get('purpose'),'broker_started')
    completions={}
    for event in own:
        if event.get('event')!='dispatch_completed':continue
        start=starts.get(event.get('request_id'))
        if (not start or any(event.get(k)!=start.get(k) for k in
                ('model','purpose','local','payload_sha256')) or event.get('reservation_state')!=
                ('local' if start.get('local') else 'settled')):continue
        # Duplicated completion records cannot provide additional coverage.
        if event['request_id'] in completions:continue
        completions[event['request_id']]=event
        increment(event.get('purpose'),'broker_completed')
    pending=None;used=set()
    for event in events(journal):
        if event.get('event')=='request':
            pending=event;increment(event.get('purpose'),'requests')
        elif event.get('event')=='response_error':pending=None
        elif event.get('event')=='response' and pending:
            request=pending;pending=None
            response=event.get('response',{});choices=response.get('choices',[])
            if (event.get('reservation')!=request.get('reservation') or
                    response.get('model')!=request.get('payload',{}).get('model') or len(choices)!=1 or
                    choices[0].get('finish_reason')!='stop' or
                    not isinstance(choices[0].get('message',{}).get('content'),str) or
                    not choices[0]['message']['content'].strip()):continue
            purpose=request.get('purpose');increment(purpose,'complete_responses')
            if (request['payload']['model'],request.get('local'))!=routes.get(purpose):continue
            expected=dict(model=request['payload']['model'],purpose=purpose,local=request.get('local'),
                payload_sha256=payload_sha256(request['payload']),reservation_id=request.get('reservation'))
            match=next((ident for ident,value in completions.items() if ident not in used and
                        all(value.get(k)==v for k,v in expected.items())),None)
            if match is not None:used.add(match);increment(purpose,'matched_complete')
    if name!='no_search':
        evidence['guided_selector_exercised']=all(
            evidence['by_purpose'].get(purpose,{}).get('matched_complete',0)>0
            for purpose in SELECTOR_PURPOSES[name])
    return evidence


def replay_native_verdict(plan, trajectory, terminal, response, payload):
    """Replay only pinned string scoring on saved text; no provider/browser."""
    from openwebrl.webprm_webarena import TASK_FILE,load_native_evaluator,verify_source
    from openwebrl.webprm_webarena_episode import TaskEnvelope
    from openwebrl.webprm_webarena_fuzzy import NativeFuzzyJudge,validate_fuzzy_configuration
    selected=validate_fuzzy_configuration(plan)
    expected_rows=(['no_search'] if plan.get('diagnostic_protocol')=='reddit_no_search_guarded_browser_v1' else list(ROWS))
    if (selected['model']!=JUDGE or plan['task_ids']!=[5] or plan['actors']!=[ACTOR]
            or plan['selectors']!=expected_rows or verify_source(plan['arbiter_source'])!=plan['source_manifest']):
        raise ValueError('Saved approved one-task protocol differs')
    raw=next(row for row in read(Path(plan['arbiter_source'])/TASK_FILE) if row['task_id']==5)
    if digest([raw])!=plan['tasks_sha256']:raise ValueError('Saved task membership changed')
    task=TaskEnvelope.from_release(raw,plan['endpoints'])
    if task.native_config()['eval']['eval_types']!=['string_match'] or terminal.get('termination')!='stop':
        raise ValueError('This text-only audit requires the actual task5 terminal answer')
    choices=response.get('choices',[])
    if (response.get('model')!=JUDGE or len(choices)!=1 or choices[0].get('finish_reason')!='stop'
            or not isinstance(choices[0].get('message',{}).get('content'),str)
            or not choices[0]['message']['content'].strip()):
        raise ValueError('Saved judge response text is incomplete')
    usage=response.get('usage',{})
    if any(type(usage.get(key)) is not int or usage[key]<=0 for key in ('prompt_tokens','completion_tokens')):
        raise ValueError('Saved judge usage is invalid')
    native=NativeFuzzyJudge(plan['arbiter_source'],enabled=False,model=selected['model'],
        protocol=selected['protocol'],model_change_approved=selected['model_change_approved'],
        max_input_tokens=plan['fuzzy_max_input_tokens'],max_calls=plan['fuzzy_max_calls'],deadline=plan['deadline'])
    proxy=SimpleNamespace(enabled=True,provider_calls=0,
        manifest=dict(native.manifest,enabled=True,budget_owner='webprm_worker.completion'))
    def saved_response(**request):
        if request.pop('context_length',None)!=0 or request.pop('model',None)!='gpt-4-1106-preview':
            raise ValueError('Pinned native request contract differs')
        request['model']=JUDGE
        if request!=payload:raise ValueError('Saved fuzzy prompt does not match task and terminal answer')
        proxy.provider_calls+=1
        if proxy.provider_calls!=1:raise ValueError('Saved one-task judge call count differs')
        return choices[0]['message']['content'],usage
    native._native['generate_from_openai_chat_completion']=saved_response
    proxy.fuzzy_match=native._native['llm_fuzzy_match']
    proxy.ua_match=native._native['llm_ua_match']
    try:
        from openwebrl.webprm_webarena import NATIVE_EVALUATOR_PROTOCOL
        verdict=load_native_evaluator(plan['arbiter_source'],fuzzy_judge=proxy,
            scoring_protocol=plan.get('evaluator_protocol',NATIVE_EVALUATOR_PROTOCOL)).evaluate(
            task.native_config(),trajectory,termination=terminal['termination'])
    except (AssertionError,KeyError,TypeError) as error:
        raise ValueError('Saved native scoring evidence is malformed') from error
    if proxy.provider_calls!=1:raise ValueError('Task5 requires one saved fuzzy response')
    return task,verdict


def audit_reset_evidence(attempt, folder, result, plan, name):
    """Verify durable identity links only, never the closed site's live inode/PID."""
    from openwebrl.webprm_reddit_provenance import ORIGINAL_MANIFEST,ORIGINAL_CONFIG,original_layers
    def linked(path,expected):
        if sha(path)!=expected:raise ValueError('Reset provenance receipt hash differs')
        return read(path)
    lease=read(attempt/('lease-'+name+'.json'))
    if any(lease.get(k)!=v for k,v in dict(task_id='webarena_lite_005',actor_model=ACTOR,selector=name).items()):
        raise ValueError('Reset lease belongs to another cell')
    instance=linked(lease['instance_receipt'],lease['instance_receipt_sha256'])
    provenance=read(folder/'provenance.json');auth=read(folder/'authentication.json')
    reset=result.get('reset_id')
    if (instance.get('scope')!='original' or instance.get('image_sha256')!=ORIGINAL_MANIFEST
            or instance.get('config_sha256')!=ORIGINAL_CONFIG or instance.get('verified_original_layers')!=29
            or instance.get('endpoint','').rstrip('/')!=plan['endpoints']['reddit'].rstrip('/')
            or provenance.get('identity_sha256')!=digest(plan) or provenance.get('original_instance')!=instance
            or provenance.get('reset_id')!=reset or instance.get('state_provenance_sha256')!=reset
            or auth.get('authenticated') is not True or auth.get('state_provenance_sha256')!=reset):
        raise ValueError('Saved reset/authentication/provenance identities differ')
    state=linked(instance['state_provenance_path'],reset)
    build=linked(instance['build_receipt_path'],instance['build_receipt_sha256'])
    verification=linked(state['original_verification_path'],state['original_verification_sha256'])
    download=linked(build['download_receipt_path'],build['download_receipt_sha256'])
    if (state.get('kind')!='original_reddit_state_copy' or state.get('status')!='completed' or state.get('returncode')!=0
            or state.get('state_path')!=instance['state_path'] or state.get('source_sandbox_path')!=build['sandbox_path']
            or state['state_path']==build['sandbox_path'] or state.get('build_receipt_path')!=instance['build_receipt_path']
            or state.get('build_receipt_sha256')!=instance['build_receipt_sha256']
            or state.get('argv')!=['cp','-a','--reflink=auto','--',build['sandbox_path'],instance['state_path']]
            or build.get('kind')!='original_reddit_sandbox_build' or build.get('status')!='completed'
            or build.get('returncode')!=0 or build.get('source_manifest_sha256')!=ORIGINAL_MANIFEST
            or build.get('config_sha256')!=ORIGINAL_CONFIG
            or build.get('argv')!=['apptainer','build','--disable-cache','--sandbox',build['sandbox_path'],download['oci_reference']]
            or download.get('status')!='complete' or download.get('source_manifest_digest')!='sha256:'+ORIGINAL_MANIFEST
            or download.get('config_digest')!='sha256:'+ORIGINAL_CONFIG
            or verification.get('status')!='verified' or verification.get('all_original_layers_verified') is not True
            or verification.get('source_manifest_digest')!='sha256:'+ORIGINAL_MANIFEST
            or verification.get('config_sha256')!=ORIGINAL_CONFIG
            or instance.get('original_verification_sha256')!=state['original_verification_sha256']):
        raise ValueError('Saved original build/copy/verification chain differs')
    layers=verification.get('layers',[]);expected_layers=original_layers()
    if len(layers)!=29 or any(row.get('verified') is not True or any(row.get(k)!=v for k,v in expected.items())
                              for row,expected in zip(layers,expected_layers)):
        raise ValueError('Saved original layer identity differs')
    return instance['state_path'],instance['build_receipt_sha256']


def audit_terminal_action(folder, result, plan, trajectory, rollout):
    """Cross-check actor termination or an explicitly executed protocol fallback.

    This authenticates recorded consistency, not live browser execution. The
    fallback is never inferred merely because the saved answer happens to be N/A.
    """
    from openwebrl.webprm_arbiter_harness import load_arbiter_harness,native_execution_code
    from openwebrl.webprm_source import ARBITER_REVISION
    from openwebrl.webprm_webarena_worker import NA_TURN_LIMIT,turn_limit_protocol,validate_protocol_contract
    harness=load_arbiter_harness(plan['arbiter_source'])
    fallback_path=folder/'protocol-fallback.json'
    fallback=result.get('protocol_fallback')
    terminal_action=trajectory[-1]
    if fallback is None:
        if fallback_path.exists() or terminal_action.get('origin') or terminal_action.get('protocol'):
            raise ValueError('Undeclared protocol-origin terminal evidence')
        last=rollout[-1];chosen=last.get('selected_index');pool=last.get('proposal',{}).get('candidates',[])
        if (type(chosen) is not int or not 0<=chosen<len(pool) or last.get('execution_error')
                or pool[chosen].get('action')!=terminal_action.get('raw_action')):
            raise ValueError('Saved terminal action differs from executed rollout')
        parsed=harness.parse_action(pool[chosen]['action'])
        if parsed['kind']!='send_message_to_user' or parsed['args'][0]!=terminal_action['answer']:
            raise ValueError('Saved terminal literal answer differs')
        return
    validate_protocol_contract(plan)
    if (turn_limit_protocol(plan)!=NA_TURN_LIMIT or result.get('turn_limit_protocol')!=NA_TURN_LIMIT
            or not isinstance(fallback,dict) or not fallback_path.exists() or read(fallback_path)!=fallback
            or type(plan.get('max_turns')) is not int or plan['max_turns']!=30
            or len(rollout)!=30 or result.get('turns')!=30):
        raise ValueError('Protocol fallback lacks its explicit full-turn contract')
    action='send_msg_to_user [N/A]';origin='released_viewer_normal_turn_limit'
    expected=dict(schema_version=1,protocol=NA_TURN_LIMIT,origin=origin,termination_reason='turn_limit',
        action=action,answer='N/A',actor_selected=False,actor_turns=30,source_revision=ARBITER_REVISION,
        source_file='viewer/src/server/service/executors/step_loop_executor.py',status='executed')
    if (any(fallback.get(key)!=value for key,value in expected.items()) or
            type(fallback.get('actor_selected')) is not bool or
            terminal_action!=dict(action_type='send_msg_to_user',answer='N/A',raw_action=action,
                origin=origin,protocol=NA_TURN_LIMIT) or
            fallback.get('pre_fallback_trajectory_sha256')!=digest(trajectory[:-1])):
        raise ValueError('Fallback provenance or pre-fallback trajectory hash differs')
    # Every actor-selected action must still match the pre-fallback trajectory;
    # no earlier actual answer may be replaced, even on the final actor turn.
    cursor=1
    for turn,row in enumerate(rollout):
        if row.get('turn')!=turn:raise ValueError('Fallback actor turn order differs')
        chosen=row.get('selected_index');pool=row.get('proposal',{}).get('candidates',[])
        if chosen is None:
            if pool or row.get('native_dispatch') is not None:
                raise ValueError('Fallback empty-pool actor evidence differs')
            continue
        if type(chosen) is not int or not 0<=chosen<len(pool):
            raise ValueError('Fallback actor selected index differs')
        parsed=harness.parse_action(pool[chosen]['action'])
        if parsed['kind']=='send_message_to_user':
            raise ValueError('Fallback cannot overwrite an actual actor terminal answer')
        if (cursor+1>=len(trajectory)-1 or trajectory[cursor].get('raw_action')!=pool[chosen]['action']
                or trajectory[cursor].get('action_type')!=parsed['kind']
                or trajectory[cursor].get('args')!=list(parsed['args'])
                or trajectory[cursor].get('kwargs')!=parsed['kwargs']
                or trajectory[cursor+1].get('info',{}).get('execution_error')!=row.get('execution_error')):
            raise ValueError('Fallback pre-trajectory differs from actual actor rollout')
        cursor+=2
    if cursor!=len(trajectory)-1 or 'observation' not in trajectory[-2]:
        raise ValueError('Fallback does not follow the recorded final actor state')
    before=fallback.get('chat_messages_before');after=fallback.get('chat_messages_after')
    appended=fallback.get('appended_messages')
    if (not isinstance(before,list) or not isinstance(after,list) or
            any(not isinstance(row,dict) for row in before+after) or
            after[:len(before)]!=before or after[len(before):]!=appended or
            not isinstance(appended,list) or len(appended)!=1 or
            appended[0].get('role')!='assistant' or appended[0].get('message')!='N/A'):
        raise ValueError('Fallback lacks its matching newly appended assistant chat')
    dispatch=fallback.get('native_dispatch')
    if not isinstance(dispatch,dict) or dispatch.get('action')!=action:
        raise ValueError('Fallback lacks actual native dispatch evidence')
    code=native_execution_code(plan['arbiter_source'],harness.parse_action(action),
        dispatch.get('element_bounds',{}),key_mapping=harness.key_mapping)
    if dispatch.get('code')!=code or dispatch.get('code_sha256')!=hashlib.sha256(code.encode()).hexdigest():
        raise ValueError('Fallback native execution code differs')


def audit_valid_cell(folder, result, plan, broker_events):
    native=folder/'native-evaluation'
    verdict=read(native/'native-verdict.json')
    if (verdict.get('valid') is not True or verdict.get('score')!=result['score']
            or verdict.get('success')!=bool(result['success']) or result['score']!=result['success']):
        raise ValueError('Native verdict does not support saved outcome')
    terminal=read(native/'terminal.json')
    if result['terminal']!=terminal or sha(native/'terminal.png')!=terminal['screenshot_sha256']:
        raise ValueError('Terminal evidence changed')
    from PIL import Image
    with Image.open(native/'terminal.png') as image:image.verify()
    trajectory=read(native/'trajectory.json')
    if not trajectory or 'observation' not in trajectory[0]:raise ValueError('Native trajectory absent')
    rollout=read(folder/'rollout.json')
    if len(rollout)!=result['turns'] or not rollout:raise ValueError('Rollout turn count differs')
    if read(folder/'provenance.json')['identity_sha256']!=digest(plan):raise ValueError('Cell provenance differs')
    fuzzy=events(folder/'native-fuzzy.jsonl')
    judgments=[event for event in fuzzy if event['event']=='native_verdict']
    responses=[event['response'] for event in fuzzy if event['event']=='response']
    if (len(judgments)!=1 or judgments[0]['score']!=verdict.get('native_score',result['success']) or len(responses)!=1
            or responses[0].get('model')!=JUDGE or len(responses[0].get('choices',[]))!=1
            or responses[0]['choices'][0].get('finish_reason')!='stop'):
        raise ValueError('Complete approved fuzzy-judge evidence absent')
    response=responses[0]
    receipt=response.get('_webprm_broker',{})
    reservation=receipt.get('reservation_id')
    # The broker deliberately projects these fields; the completion journal
    # retains additional provider metadata (id/object/created/etc.).
    projected_response={key:response.get(key) for key in ('model','choices','usage')}
    journal=events(folder/'api.jsonl')
    requests=[event for event in journal if event['event']=='request' and event.get('reservation')==reservation]
    received=[event for event in journal if event['event']=='response' and event.get('reservation')==reservation]
    if (type(reservation) is not int or receipt.get('reservation_state')!='settled'
            or receipt.get('identity_sha256')!=digest(plan) or len(requests)!=1 or len(received)!=1
            or requests[0].get('local') is not False or requests[0].get('purpose')!='native_fuzzy'
            or requests[0]['payload'].get('model')!=JUDGE
            or {key:received[0]['response'].get(key) for key in projected_response}!=projected_response):
        raise ValueError('Native judge lacks matching metered provider evidence')
    payload=requests[0]['payload']
    from openwebrl.webprm_broker import payload_sha256
    expected=dict(identity_sha256=digest(plan),cell_id='webarena_lite_005/'+result['arm'],
        purpose='native_fuzzy',model=JUDGE,local=False,payload_sha256=payload_sha256(payload))
    if any(receipt.get(k)!=v for k,v in expected.items()):raise ValueError('Fuzzy broker receipt identity differs')
    matching=[row for row in broker_events if row.get('request_id')==receipt.get('request_id')]
    started=[row for row in matching if row['event']=='dispatch_started']
    completed=[row for row in matching if row['event']=='dispatch_completed']
    if (len(started)!=1 or len(completed)!=1 or any(started[0].get(k)!=v for k,v in expected.items())
            or any(completed[0].get(k)!=v for k,v in receipt.items())):
        raise ValueError('Saved judge lacks matching completed broker dispatch')
    task,replayed=replay_native_verdict(plan,trajectory,terminal,response,payload)
    if replayed!=verdict:raise ValueError('Saved native verdict differs from offline replay')
    for key,value in dict(raw_config_sha256=task.raw_config_sha256,
                          configured_config_sha256=task.configured_config_sha256).items():
        if result.get(key)!=value:raise ValueError('Saved native task identity differs')
    audit_terminal_action(folder,result,plan,trajectory,rollout)
    before=read(native/'outcome-before-close.json')
    for key in ('task_id','raw_config_sha256','configured_config_sha256','valid','score','success','terminal'):
        if before.get(key)!=result.get(key):raise ValueError('Durable before-close outcome differs')
    if 'evaluator_protocol' in verdict:
        for key in ('evaluator_protocol','native_score','native_success','native_evaluator_details','scoring_amendments'):
            if before.get(key)!=verdict[key] or result.get(key)!=verdict[key]:
                raise ValueError('Prospective native/effective outcome provenance differs')
    return nonanswer_numeric_reference_positive(task.native_config(), trajectory, verdict)


def nonanswer_numeric_reference_positive(task, trajectory, verdict):
    """Flag the observed N/A/numeric-reference contradiction, without regrading.

    This narrow deterministic diagnostic is not a general semantic evaluator.
    It never applies to URL/DOM checks or an explicit unachievable reference.
    """
    evaluation = task.get('eval', {})
    references = evaluation.get('reference_answers', {}).get('fuzzy_match', [])
    if isinstance(references, str):
        references = [references]
    if not isinstance(references, list) or not references:
        return False
    numeric = all(isinstance(value, str) and re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)', value.strip())
                  for value in references)
    answer = trajectory[-1].get('answer') if trajectory else None
    return bool(evaluation.get('eval_types') == ['string_match'] and numeric
                and isinstance(answer, str) and answer.strip().strip('\"\'').lower() == 'n/a'
                and verdict.get('success') is True and verdict.get('score') == 1)


def attach_outcome_quality(summary, checks):
    """Keep native counts; withhold performance rates when this defect occurs."""
    flagged = [name for name, flag in checks.items() if flag]
    summary['outcome_quality'] = dict(
        diagnostic='nonanswer_numeric_reference_positive_v1',
        checked_valid_cells=len(checks), flagged_cells=len(flagged), flagged_selectors=flagged,
        performance_rates_withheld=bool(flagged), raw_native_results_unchanged=True,
        limitation='Narrow N/A-versus-numeric-reference check; absence of flags does not establish judge accuracy.')
    if not flagged:
        return
    summary['native_reported_rates'] = {key: summary[key] for key in ('overall_success_pct', 'valid_only_success_pct')}
    summary['overall_success_pct'] = summary['valid_only_success_pct'] = None
    for row in summary['rows']:
        row['native_reported_rates'] = {key: row[key] for key in ('overall_success_pct', 'valid_only_success_pct')}
        row['overall_success_pct'] = row['valid_only_success_pct'] = None
        row['nonanswer_numeric_reference_positive'] = checks.get(row['selector'])


def audit_ledger(ledger, paid_requests, paid_responses, expected_cap=15):
    if expected_cap not in (2,15) or ledger['cap']!=expected_cap:raise ValueError('API cap changed')
    ids=set(range(len(ledger['calls'])))
    if (any(type(key) is not int for key in paid_requests) or set(paid_requests)-ids
            or set(paid_responses)-set(paid_requests)):
        raise ValueError('Paid journal reservation missing from shared ledger')
    settled=reserved=0.
    for position,call in enumerate(ledger['calls']):
        if call['id']!=position or call['model'] not in PRICES:
            raise ValueError('Ledger identity/model mismatch')
        if position in paid_requests and paid_requests[position]!=call['model']:
            raise ValueError('Ledger request model mismatch')
        if not math.isfinite(call['reserved']) or call['reserved']<0:raise ValueError('Invalid reservation')
        if call['state']=='reserved':reserved+=call['reserved'];continue
        if call['state'] not in ('settled','overrun'):raise ValueError('Unknown ledger state')
        usage=call['usage'];rates=PRICES[call['model']]
        if any(type(usage.get(k)) is not int or usage[k]<0 for k in ('prompt_tokens','completion_tokens')):
            raise ValueError('Invalid token usage')
        cost=usage['prompt_tokens']*rates[0]+usage['completion_tokens']*rates[1]
        if (not math.isclose(cost,call['charged'],abs_tol=1e-9) or paid_requests.get(position)!=call['model']
                or paid_responses.get(position)!=usage):raise ValueError('Settled charge lacks matching request/response')
        settled+=cost
    if (not math.isclose(settled+reserved,ledger['charged'],abs_tol=1e-8)
            or not 0<=ledger['charged']<=expected_cap):raise ValueError('API ledger total differs or exceeds cap')
    return dict(cap_usd=expected_cap,settled_usd=settled,retained_reservations_usd=reserved,
        charged_usd=ledger['charged'],reservations=len(ledger['calls']),halt_reason=ledger.get('halt_reason'))


V3_PROFILE='reddit_corrected_v3_90m_400local'
V3_RESOURCES=dict(gpus=1,gpu_type='H200',cpus=8,memory_gib=120,total_seconds=5400)
V3_LIMITS=dict(api_usd=15.,paid_requests=575,local_requests=400,total_requests=975,
    scratch_bytes=300*1024**3,runtime_reserve_bytes=16*1024**3,max_turns=30,max_candidates=5)
V4_PROFILE='reddit_guarded_v4_90m_400local'
V4_RESOURCES=dict(V3_RESOURCES)
V4_LIMITS=dict(V3_LIMITS,paid_requests=600,total_requests=1000)
CPU_PROFILE='reddit_no_search_guarded_v4_cpu_45m'
CPU_RESOURCES=dict(gpus=0,gpu_type=None,cpus=8,memory_gib=16,total_seconds=2700)
CPU_LIMITS=dict(V4_LIMITS,api_usd=2.,paid_requests=98,local_requests=0,total_requests=98,scratch_bytes=128*1024**3)


def approved_campaign_profile(context):
    proposal=context['proposal'];approval=context['approval']
    profile=proposal.get('resource_profile')
    if profile is None:
        if approval.get('resource_profile') is not None:
            raise ValueError('Approval resource profile differs')
        return ('legacy_v2',575,320)
    expected_resources=CPU_RESOURCES if profile==CPU_PROFILE else V4_RESOURCES if profile==V4_PROFILE else V3_RESOURCES
    expected_limits=CPU_LIMITS if profile==CPU_PROFILE else V4_LIMITS if profile==V4_PROFILE else V3_LIMITS
    if (profile not in (V3_PROFILE,V4_PROFILE,CPU_PROFILE) or approval.get('resource_profile')!=profile or
            approval.get('status')!='approved' or
            approval.get('turn_limit_protocol')!='released_viewer_na_fallback_v1' or
            approval.get('turn_limit_protocol_approved') is not True or
            any(proposal_digest(record.get(key))!=proposal_digest(expected)
                for record in (proposal,approval)
                for key,expected in [('resources',expected_resources),('limits',expected_limits)])):
        raise ValueError('Corrected campaign lacks its exact approved resource/protocol profile')
    descriptor=proposal.get('piotr_representation')
    if descriptor is not None:
        from openwebrl.webprm_webarena_piotr import REPRESENTATION_V2
        if (descriptor!=REPRESENTATION_V2 or approval.get('piotr_representation')!=descriptor
                or approval.get('piotr_representation_approved') is not True):
            raise ValueError('Descriptor-v2 lacks its explicit representation approval')
    elif approval.get('piotr_representation') is not None:
        raise ValueError('Descriptor approval differs from proposal')
    if profile in (V4_PROFILE,CPU_PROFILE):
        from openwebrl.webprm_webarena import ABSTENTION_EVALUATOR_PROTOCOL
        from openwebrl.webprm_webarena_worker import ENVELOPE_NO_SEARCH_PROMPT
        for key,value in (('evaluator_protocol',ABSTENTION_EVALUATOR_PROTOCOL),('no_search_prompt_version',ENVELOPE_NO_SEARCH_PROMPT)):
            if (proposal.get(key)!=value or approval.get(key)!=value or approval.get(key+'_approved') is not True):
                raise ValueError('Guarded v4 scoring/NoSearch lacks explicit proposal and approval')
    if profile==CPU_PROFILE and (proposal.get('rows')!=['no_search']
            or proposal.get('diagnostic_protocol')!='reddit_no_search_guarded_browser_v1'
            or proposal.get('maximum_allocation_attempts')!=1):
        raise ValueError('CPU diagnostic membership/attempt cap differs')
    return (profile,expected_limits['paid_requests'],expected_limits['local_requests'])


def audit(campaign):
    campaign=Path(campaign);records=[];seen_rows=set();request_ids=set();counts=Counter()
    paid_requests={};paid_responses={};reset_ids=set();state_paths=set();closed=True;evidence={};attempts=0
    call_evidence={};campaign_profile=None;startup_fuzzy={};outcome_quality={}
    for attempt in sorted(campaign.glob('attempt-*')):
        if not (attempt/'execution-context.json').exists():
            if (attempt/'broker.jsonl').exists() or list(attempt.glob('worker-launch-*.json')):
                raise ValueError('Execution evidence lacks attempt context')
            if (attempt/'preexecution-rejection.json').exists():attempts+=1
            continue
        attempts+=1;context=read(attempt/'execution-context.json')
        if (context['proposal_sha256']!=proposal_digest(context['proposal']) or
                context['approval']['proposal_sha256']!=context['proposal_sha256']):
            raise ValueError('Attempt approval/proposal identity differs')
        profile=approved_campaign_profile(context)
        if campaign_profile is not None and profile!=campaign_profile:
            raise ValueError('Distinct protocol/resource cohorts must use separate campaigns')
        campaign_profile=profile
        closure=attempt/'controller-closure.json'
        closed &= closure.exists() and read(closure).get('remaining_owned_processes')=={}
        for command in attempt.glob('*-command.json'):
            argv=read(command).get('argv',[])
            if 'sglang.launch_server' in argv and '--skip-server-warmup' not in argv:
                counts['conservative_internal_warmups']+=1
        broker_events=events(attempt/'broker.jsonl')
        for event in broker_events:
            if event['event']=='dispatch_started':
                if event['request_id'] in request_ids or type(event['local']) is not bool:
                    raise ValueError('Duplicate/malformed dispatch')
                request_ids.add(event['request_id']);counts['local_dispatches' if event['local'] else 'paid_dispatches']+=1
        for journal in [attempt/'local-preflight-api.jsonl',attempt/'native-fuzzy-preflight-api.jsonl',
                        *attempt.glob('episodes/*/*/api.jsonl')]:
            pending=None
            for event in events(journal):
                if event['event']=='request':
                    pending=event
                    if type(event.get('local')) is not bool:raise ValueError('Malformed journal request mode')
                    counts['journal_local_requests' if event['local'] else 'journal_paid_requests']+=1
                    if not event['local']:
                        reservation=event['reservation']
                        if reservation in paid_requests:raise ValueError('Duplicate API reservation')
                        paid_requests[reservation]=event['payload']['model']
                elif event['event']=='response' and pending and not pending['local']:
                    if event['reservation']!=pending['reservation']:raise ValueError('API response reservation differs')
                    response=event['response']
                    if response.get('model')==pending['payload']['model']:
                        paid_responses[event['reservation']]=response.get('usage')
                    pending=None
                elif event['event']=='response_error':pending=None
        plan_path=attempt/'worker-plan.json'
        if not plan_path.exists():
            if list(attempt.glob('worker-launch-*.json')):raise ValueError('Launched cell lacks frozen worker plan')
            if (attempt/'native-fuzzy-preflight-api.jsonl').exists() or (attempt/'native-fuzzy-preflight').exists():
                raise ValueError('Startup judge evidence lacks frozen worker plan')
            continue
        plan=read(plan_path)
        if plan.get('piotr_representation')!=context['proposal'].get('piotr_representation'):
            raise ValueError('Worker Piotr representation differs from approved proposal')
        from openwebrl.webprm_webarena_worker import NA_TURN_LIMIT,turn_limit_protocol
        if (turn_limit_protocol(plan)==NA_TURN_LIMIT)!=(profile[0] in (V3_PROFILE,V4_PROFILE,CPU_PROFILE)):
            raise ValueError('Worker fallback protocol differs from approved campaign profile')
        if any(key in plan for key in ('protocol','protocol_contract','turn_limit_protocol')):
            from openwebrl.webprm_webarena_worker import validate_protocol_contract
            validate_protocol_contract(plan)
        if profile[0] in (V4_PROFILE,CPU_PROFILE) and any(plan.get(key)!=context['proposal'].get(key)
                for key in ('no_search_prompt_version','evaluator_protocol')):
            raise ValueError('Guarded v4 worker prompt/scoring differs from approved proposal')
        if profile[0] in (V3_PROFILE,V4_PROFILE,CPU_PROFILE):
            from scripts.audit_webprm_fuzzy_preflight import audit_native_fuzzy_preflight
            startup_fuzzy[attempt.name]=audit_native_fuzzy_preflight(attempt,plan,context['proposal'],broker_events,
                primary_launched=bool(list(attempt.glob('worker-launch-*.json'))))
        planned_rows=('no_search',) if profile[0]==CPU_PROFILE else ROWS
        if profile[0]==CPU_PROFILE and (plan.get('selectors')!=['no_search']
                or plan.get('diagnostic_protocol')!='reddit_no_search_guarded_browser_v1'
                or len(list(campaign.glob('attempt-*/execution-context.json')))!=1):
            raise ValueError('CPU diagnostic exceeded fixed membership/one-attempt scope')
        for name in planned_rows:
            launch=attempt/('worker-launch-'+name+'.json')
            if not launch.exists():continue
            if read(launch).get('selector')!=name:raise ValueError('Worker launch selector differs')
            if name in seen_rows:raise ValueError('Primary cell resampled across attempts')
            seen_rows.add(name)
            folder=attempt/'episodes/webarena_lite_005'/(ACTOR+'__'+name)
            call_evidence[name]=audit_call_evidence(folder,name,plan,broker_events)
            result_path=folder/'result.json'
            if not result_path.exists():continue
            result=read(result_path)
            if (result['identity_sha256']!=digest(plan) or result['arm']!=ACTOR+'__'+name
                    or result['task_id']!='webarena_lite_005'):
                raise ValueError('Wrong fixed cell identity')
            watchdog=read(folder/'watchdog.json')
            if watchdog!=result.get('watchdog') or watchdog.get('remaining_owned_processes')!={}:
                raise ValueError('Episode process closure unverified')
            if result.get('reset_id'):
                if result['reset_id'] in reset_ids:raise ValueError('Fresh state reused across selectors')
                reset_ids.add(result['reset_id'])
            if result['valid']:
                if not result.get('reset_id'):raise ValueError('Valid episode lacks fresh-state identity')
                state_path,_=audit_reset_evidence(attempt,folder,result,plan,name)
                if state_path in state_paths:raise ValueError('Fresh state path reused across selectors')
                state_paths.add(state_path)
                outcome_quality[name]=audit_valid_cell(folder,result,plan,broker_events)
            records.append(dict(result,selector=name));evidence[name]=sha(result_path)
    local=counts['local_dispatches']+counts['conservative_internal_warmups']
    _,paid_cap,local_cap=campaign_profile or ('legacy_v2',575,320)
    if counts['paid_dispatches']>paid_cap or local>local_cap:raise ValueError('Request cap exceeded')
    if (counts['journal_paid_requests']>counts['paid_dispatches'] or
            counts['journal_local_requests']>counts['local_dispatches']):
        raise ValueError('Completion requests lack broker dispatch accounting')
    ledger_path=campaign/'shared-budget.json'
    if not ledger_path.exists() and (counts['paid_dispatches'] or paid_requests or paid_responses):
        raise ValueError('Paid campaign lost its shared API ledger')
    cpu_only=campaign_profile and campaign_profile[0]==CPU_PROFILE
    expected_cap=2 if cpu_only else 15
    ledger=read(ledger_path) if ledger_path.exists() else dict(cap=expected_cap,charged=0,calls=[])
    summary=summarize(records,seen_rows,call_evidence,planned_rows=('no_search',) if cpu_only else ROWS)
    summary['all_mechanics_valid']=(summary['all_native_verdicts_valid'] and
        summary['all_guided_selectors_exercised'] and closed and all(
            row['call_evidence']['by_group']['actor_aux']['matched_complete']>0 for row in summary['rows']))
    attach_outcome_quality(summary,outcome_quality)
    if campaign_profile and campaign_profile[0] in (V4_PROFILE,CPU_PROFILE):
        summary['scoring_protocol']='native_with_explicit_na_abstention_guard_v1'
        summary['native_positive_count']=sum(bool(record.get('native_success')) for record in records if record['valid'])
        summary['effective_positive_count']=summary['totals']['successes']
        summary['scoring_amendment_count']=sum(bool(record.get('scoring_amendments')) for record in records if record['valid'])
        for row in summary['rows']:
            record=next((r for r in records if r['selector']==row['selector']),{})
            row['native_positive_count']=int(bool(record.get('native_success')))
            row['effective_positive_count']=row['successes']
    if cpu_only:
        summary['overall_success_pct']=summary['valid_only_success_pct']=None
        for row in summary['rows']:row['overall_success_pct']=row['valid_only_success_pct']=None
        summary['diagnostic_only']=True
    return dict(schema_version=3,scope=('One fixed NoSearch prompt/scoring browser validation; not a performance estimate'
        if cpu_only else 'Five-row one-task mechanics pilot; not Table5 performance reproduction'),
        actor=ACTOR,judge=JUDGE,judge_substitution_disclosed=True,paper_exact_protocol=False,
        **summary,allocation_attempts_registered=attempts,all_registered_controllers_closed=closed,
        requests=dict(counts,charged_local_requests=local,paid_cap=paid_cap,local_cap=local_cap),
        api=audit_ledger(ledger,paid_requests,paid_responses,expected_cap=expected_cap),result_sha256=evidence,
        native_fuzzy_preflights=startup_fuzzy,
        audit_limits=['Saved text/native-verdict replay only; no live browser or website reexecution.',
            'Selector exercise means completed scoring dispatches, not offline replay of selector parsing or a guarantee of action/grounding correctness. Terminal-only episodes remain valid without critic coverage.',
            'Reset receipts verify recorded hash/identity links, not current node-local filesystem bytes or live PID/inode identity.',
            'Allocation time and retained-storage accounting require the separate controller/scheduler audit.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=audit(args.campaign)
    path=args.output
    if (not path.is_absolute() or path.resolve()!=path or path.is_symlink()
            or not any(root in path.parents for root in (REPO,Path('/tmp'),
                Path('/gpfs/scrubbed/zixianma/openwebrl-runtime')))):
        raise ValueError('Aggregate output requires a fresh canonical approved path')
    with args.output.open('x') as output:json.dump(result,output,indent=2);output.write('\n')
    print(json.dumps(result,indent=2))
