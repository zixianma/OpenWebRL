#!/usr/bin/env python3
"""Offline evidence audit for one fixed Table 5 reconstruction episode.

Never opens a browser/provider/scheduler, restores a database, or deletes state.
The owner supplies terminal process observations; this audit checks their saved
bindings and consistency, not independent historical process or website truth.
Other site reset chains remain disabled until independently implemented.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openwebrl.webprm_webarena import digest
from openwebrl.webprm_webarena_sites import canonical_runtime_path

SCHEMA = 'table5_independent_cell_audit_v1'
BASE_ARTIFACTS = {'result', 'watchdog', 'api', 'site_closure', 'closure'}
VALID_ARTIFACTS = {'rollout', 'provenance', 'authentication', 'trajectory', 'terminal',
                   'verdict', 'outcome', 'evaluator_inputs', 'terminal_screenshot',
                   'lease', 'instance', 'copy_receipt', 'state_claim'}


def sha(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


def read(path, cap=32*2**20):
    path = canonical_runtime_path(str(path))
    if not path.is_file() or path.stat().st_size > cap:
        raise ValueError('Absent or oversized evidence file')
    return json.loads(path.read_text())


def reference(value, *, root=None):
    if (not isinstance(value, dict) or set(value) != {'path', 'sha256'} or
            not re.fullmatch('[0-9a-f]{64}', str(value.get('sha256', '')))):
        raise ValueError('Exact path and SHA256 evidence reference required')
    path = canonical_runtime_path(value['path'])
    if root is not None and root not in path.parents:
        raise ValueError('Cell artifact escaped its durable campaign')
    if not path.is_file() or sha(path) != value['sha256']:
        raise ValueError('Evidence reference changed or missing')
    return path


def read_lines(path):
    path = canonical_runtime_path(str(path))
    if path.stat().st_size > 512*2**20:
        raise ValueError('Cell journal exceeds offline audit bound')
    with path.open('rb') as stream:
        rows = []
        for line in stream:
            if not line.endswith(b'\n'):
                raise ValueError('Incomplete cell journal tail')
            rows.append(json.loads(line))
    return rows


def audit_identity(inputs, plan, cohort):
    from scripts.prepare_webprm_table5_campaign import TASKS, TASKS_SHA, cohort as make_cohort, digest as cohort_digest
    from openwebrl.webprm_table5_analysis import validate_cohort
    from openwebrl.webprm_webarena_worker import validate_protocol_contract, validate_frozen_sources
    from openwebrl.webprm_webarena_episode import TaskEnvelope
    indexed = validate_cohort(cohort)
    cell = inputs['cell']
    if indexed.get(cell.get('cell_id')) != cell or sha(TASKS) != TASKS_SHA:
        raise ValueError('Cell differs from the pinned released cohort')
    released = read(TASKS)
    rebuilt = make_cohort(released)
    if cohort['cells'] != rebuilt['cells']:
        raise ValueError('Cohort members/order differ from the released task file')
    raw = next(t for t in released if t['task_id'] == cell['task_id'])
    # Cohort task hashes escape Unicode; worker/config hashes below use UTF-8.
    # Preserve both existing identities rather than treating them as aliases.
    if cohort_digest(raw) != cell['task_config_sha256']:
        raise ValueError('Released task identity differs')
    protocol = validate_protocol_contract(plan)
    if digest(protocol) != cohort['protocol_sha256']:
        raise ValueError('Shared scientific protocol differs from cohort')
    if (plan.get('task_ids') != [cell['task_id']] or plan.get('tasks_sha256') != digest([raw]) or
            plan.get('actor_model') != cell['actor'] or plan.get('selector') != cell['selector'] or
            plan.get('actors') != [cell['actor']] or plan.get('selectors') != [cell['selector']]):
        raise ValueError('Worker plan changed fixed cell membership')
    validate_frozen_sources(plan)
    audit_code_freeze(inputs,plan)
    if cell['selector'] == 'piotr_selectionarm':
        from openwebrl.webprm_webarena_piotr import validate_piotr_selectionarm
        if plan.get('piotr_manifest') != validate_piotr_selectionarm(plan):
            raise ValueError('Released Piotr source/checkpoint/representation identity differs')
    if plan.get('max_turns') != 30 or plan.get('paper_exact_protocol') is not False:
        raise ValueError('Frozen 30-turn disclosed reconstruction required')
    return TaskEnvelope.from_release(raw, plan['endpoints'])


def audit_code_freeze(inputs,plan):
    from scripts.run_webprm_webarena_pilot import REQUIRED_CODE
    code = plan.get('code_sha256')
    if not isinstance(code, dict) or not code or any(inputs['source_sha256'].get(k) != v for k, v in code.items()):
        raise ValueError('Complete frozen implementation references required')
    required = set(REQUIRED_CODE) | {'scripts/audit_webprm_table5_cell.py',
        'scripts/analyze_webprm_webarena_pilot.py','scripts/prepare_webprm_table5_campaign.py',
        'openwebrl/webprm_webarena_helpers.py','openwebrl/webprm_webarena_helper_runtime.py',
        'openwebrl/webprm_evaluator_transcript.py','openwebrl/webprm_journal_audit.py',
        'openwebrl/webprm_budget_journal.py','openwebrl/webprm_concurrent_broker.py',
        'openwebrl/webprm_ephemeral_state.py','openwebrl/webprm_table5_analysis.py'}
    matches = [Path(p).parents[1] for p in code if p.endswith('/scripts/audit_webprm_table5_cell.py')]
    if len(matches) != 1 or any(str(matches[0]/p) not in code for p in required):
        raise ValueError('Required worker/helper/transcript/auditor source freeze is incomplete')
    executing = Path(__file__).resolve().parents[1]
    if any(sha(executing/p) != code[str(matches[0]/p)] for p in required):
        raise ValueError('Executing offline audit dependencies differ from the frozen implementation')
    for path, expected in inputs['source_sha256'].items():
        reference(dict(path=path, sha256=expected))


def audit_closure(inputs, paths, result):
    watchdog, site, closure = (read(paths[k]) for k in ('watchdog', 'site_closure', 'closure'))
    cell = inputs['cell']
    if (watchdog != result.get('watchdog') or watchdog.get('remaining_owned_processes') != {} or
            site.get('remaining_owned_processes') != {} or
            closure.get('cell_id') != cell['cell_id'] or
            closure.get('worker_remaining_processes') != {} or closure.get('service_remaining_processes') != {} or
            type(closure.get('broker_cell_inflight')) is not int or closure['broker_cell_inflight'] != 0 or
            closure.get('worker_watchdog_sha256') != sha(paths['watchdog']) or
            closure.get('site_closure_sha256') != sha(paths['site_closure'])):
        raise ValueError('Actual saved worker/service/request closure is incomplete')
    status = read(reference(closure['broker_status_reference']))
    active = status.get('active_cell_ids')
    if (not isinstance(active, list) or inputs['broker']['cell_id'] in active or
            status.get('active_cells') != len(active)):
        raise ValueError('Broker snapshot does not establish this cell is inactive')
    if (status.get('budget',{}).get('budget_prefix_bytes') != inputs['budget']['prefix_bytes'] or
            status.get('budget',{}).get('budget_ledger_sha256') != inputs['budget']['prefix_sha256']):
        raise ValueError('Broker closure and audited financial prefix differ')
    return closure


def audit_reddit_reset(inputs, paths, plan, result):
    from openwebrl.webprm_reddit_provenance import (ORIGINAL_MANIFEST, ORIGINAL_CONFIG, original_layers,
        validate_original_verification,validate_original_download)
    if inputs['cell']['domain'] != 'reddit':
        raise ValueError('This domain has no completed offline original-reset auditor')
    instance, lease, state, claim = (read(paths[k]) for k in ('instance','lease','copy_receipt','state_claim'))
    def linked(path, checksum):
        return read(reference(dict(path=path, sha256=checksum)))
    ident = f"webarena_lite_{inputs['cell']['task_id']:03d}"
    if (any(lease.get(k) != v for k,v in dict(task_id=ident, actor_model=inputs['cell']['actor'],
            selector=inputs['cell']['selector']).items()) or
            reference(dict(path=lease['instance_receipt'],sha256=lease['instance_receipt_sha256'])) != paths['instance'] or
            instance.get('scope') != 'original' or instance.get('image_sha256') != ORIGINAL_MANIFEST or
            instance.get('config_sha256') != ORIGINAL_CONFIG or instance.get('verified_original_layers') != 29 or
            instance.get('endpoint','').rstrip('/') != plan['endpoints']['reddit'].rstrip('/') or
            instance.get('state_provenance_path') != str(paths['copy_receipt']) or
            instance.get('state_provenance_sha256') != sha(paths['copy_receipt']) or
            result.get('reset_id') != sha(paths['copy_receipt'])):
        raise ValueError('Original instance/lease/reset identity differs')
    from openwebrl.webprm_ephemeral_state import PROTOCOL as CLAIM_PROTOCOL
    expected_state = Path(claim['scratch_root'])/'states'/hashlib.sha256(inputs['cell']['cell_id'].encode()).hexdigest()
    if (claim.get('protocol') != CLAIM_PROTOCOL or claim.get('cell_id') != inputs['cell']['cell_id'] or
            claim.get('destination_absent_at_claim') is not True or claim.get('preserve_historical_states') is not True or
            claim.get('state_path') != str(expected_state) or instance.get('state_path') != str(expected_state) or
            claim.get('campaign_root') != inputs['campaign_root']):
        raise ValueError('Original reset lacks its unique prospective state claim')
    build = linked(instance['build_receipt_path'],instance['build_receipt_sha256'])
    verification = linked(state['original_verification_path'],state['original_verification_sha256'])
    download = linked(build['download_receipt_path'],build['download_receipt_sha256'])
    validate_original_verification(state['original_verification_path'],state['original_verification_sha256'])
    validate_original_download(build['download_receipt_path'],build['download_receipt_sha256'])
    site = read(paths['site_closure'])
    if type(instance.get('service_pid')) is not int or instance['service_pid'] not in site.get('owned_pids',[]):
        raise ValueError('Service closure belongs to another original instance')
    if (state.get('kind') != 'original_reddit_state_copy' or state.get('status') != 'completed' or state.get('returncode') != 0 or
            state.get('state_path') != str(expected_state) or state.get('source_sandbox_path') != build.get('sandbox_path') or
            claim.get('base_path') != build.get('sandbox_path') or state['state_path'] == build.get('sandbox_path') or
            state.get('build_receipt_path') != instance['build_receipt_path'] or
            state.get('build_receipt_sha256') != instance['build_receipt_sha256'] or
            state.get('argv') != ['cp','-a','--reflink=auto','--',build['sandbox_path'],str(expected_state)] or
            build.get('kind') != 'original_reddit_sandbox_build' or build.get('status') != 'completed' or build.get('returncode') != 0 or
            build.get('source_manifest_sha256') != ORIGINAL_MANIFEST or build.get('config_sha256') != ORIGINAL_CONFIG or
            build.get('argv') != ['apptainer','build','--disable-cache','--sandbox',build['sandbox_path'],download['oci_reference']] or
            download.get('status') != 'complete' or download.get('source_manifest_digest') != 'sha256:'+ORIGINAL_MANIFEST or
            download.get('config_digest') != 'sha256:'+ORIGINAL_CONFIG or
            verification.get('status') != 'verified' or verification.get('all_original_layers_verified') is not True or
            verification.get('source_manifest_digest') != 'sha256:'+ORIGINAL_MANIFEST or verification.get('config_sha256') != ORIGINAL_CONFIG or
            instance.get('original_verification_sha256') != state['original_verification_sha256']):
        raise ValueError('Original image/build/copy verification chain differs')
    layers = verification.get('layers', [])
    if len(layers) != 29 or any(row.get('verified') is not True or any(row.get(k) != v for k,v in expected.items())
            for row, expected in zip(layers, original_layers())):
        raise ValueError('Original layer identity differs')
    return instance


def saved_fuzzy_proxy(plan, pairs, native_events):
    """Replay exact native judge prompts against metered saved responses only."""
    from openwebrl.webprm_webarena_fuzzy import NativeFuzzyJudge, validate_fuzzy_configuration
    selected = validate_fuzzy_configuration(plan)
    native = NativeFuzzyJudge(plan['arbiter_source'],enabled=False,model=selected['model'],
        protocol=selected['protocol'],model_change_approved=selected['model_change_approved'],
        max_input_tokens=plan['fuzzy_max_input_tokens'],max_calls=plan['fuzzy_max_calls'],deadline=plan['deadline'])
    pending = [p for p in pairs if p.get('request',{}).get('purpose') == 'native_fuzzy']
    saved = [e['response'] for e in native_events if e.get('event') == 'response']
    if len(saved) != len(pending) or any(p.get('response') is None for p in pending):
        raise ValueError('Native judge evidence lacks its exact metered responses')
    proxy = SimpleNamespace(enabled=True,provider_calls=0,
        manifest=dict(native.manifest,enabled=True,budget_owner='webprm_worker.completion'))
    def reply(**request):
        i = proxy.provider_calls
        if i >= len(pending) or request.pop('context_length',None) != 0 or request.pop('model',None) != 'gpt-4-1106-preview':
            raise ValueError('Unrecorded native fuzzy request')
        request['model'] = selected['model']
        pair, response = pending[i], saved[i]
        actual = pair['response']['response']
        receipt = response.get('_webprm_broker',{})
        projection = ('model','choices','usage')
        choices = response.get('choices',[])
        if (request != pair['request']['payload'] or any(response.get(k) != actual.get(k) for k in projection) or
                response.get('model') != selected['model'] or len(choices) != 1 or choices[0].get('finish_reason') != 'stop' or
                not isinstance(choices[0].get('message',{}).get('content'),str) or
                not choices[0]['message']['content'].strip() or
                receipt.get('reservation_id') != pair['request'].get('reservation') or
                receipt.get('request_id') != pair['broker'].get('request_id') or
                any(pair['broker'].get(k) != v for k,v in receipt.items())):
            raise ValueError('Saved fuzzy prompt/response/broker receipt changed')
        proxy.provider_calls += 1
        return choices[0]['message']['content'], response['usage']
    native._native['generate_from_openai_chat_completion'] = reply
    proxy.fuzzy_match = native._native['llm_fuzzy_match']; proxy.ua_match = native._native['llm_ua_match']
    return proxy, len(pending)


def audit_empty_native_dispatch(plan,trajectory,cursor,parsed,harness,row):
    """Replay an explicitly saved rejection before native code was generated.

    The worker clears its dispatch map before conversion and fills it only after
    conversion succeeds. An empty map is therefore admissible only when the
    preceding saved bounds reproduce the exact recorded parsing exception.
    """
    from openwebrl.webprm_arbiter_harness import native_execution_code,ActionParsingError
    preceding=trajectory[cursor-1] if 0<cursor<=len(trajectory) else None
    observation=preceding.get('observation') if isinstance(preceding,dict) else None
    if (row.get('native_dispatch')!={} or not isinstance(observation,dict) or
            not isinstance(observation.get('element_bounds'),dict) or
            not isinstance(row.get('execution_error'),str)):
        raise ValueError('Empty native dispatch lacks saved pre-dispatch rejection evidence')
    try:
        native_execution_code(plan['arbiter_source'],parsed,observation['element_bounds'],
                              key_mapping=harness.key_mapping)
    except ActionParsingError as error:
        if row['execution_error']!=type(error).__name__+': '+str(error):
            raise ValueError('Saved pre-dispatch rejection differs from native conversion') from error
    else:
        raise ValueError('Empty native dispatch cannot hide an executable action')


def audit_rollout(plan,trajectory,rollout,result):
    """Verify every saved selected action and state transition, including failures."""
    from openwebrl.webprm_arbiter_harness import load_arbiter_harness,native_execution_code
    harness=load_arbiter_harness(plan['arbiter_source'])
    if not rollout or not 0<len(rollout)<=plan['max_turns'] or result.get('turns')!=len(rollout):
        raise ValueError('Saved rollout exceeds its frozen turn limit')
    cursor=1
    for turn,row in enumerate(rollout):
        if row.get('turn')!=turn:raise ValueError('Rollout turn order differs')
        pool=row.get('proposal',{}).get('candidates');chosen=row.get('selected_index')
        if not isinstance(pool,list):raise ValueError('Original candidate pool missing')
        if chosen is None:
            if pool or row.get('native_dispatch') is not None:raise ValueError('Empty-pool execution contradicts rollout')
            continue
        if type(chosen) is not int or not 0<=chosen<len(pool):raise ValueError('Selected original candidate index differs')
        action=pool[chosen]['action'];parsed=harness.parse_action(action)
        dispatch=row.get('native_dispatch')
        if dispatch=={}:
            audit_empty_native_dispatch(plan,trajectory,cursor,parsed,harness,row)
        elif dispatch is not None:
            if not isinstance(dispatch,dict) or dispatch.get('action')!=action:
                raise ValueError('Native dispatch differs from selected action')
            code=native_execution_code(plan['arbiter_source'],parsed,dispatch.get('element_bounds',{}),key_mapping=harness.key_mapping)
            if dispatch.get('code')!=code or dispatch.get('code_sha256')!=hashlib.sha256(code.encode()).hexdigest():
                raise ValueError('Saved native execution code differs')
        elif not row.get('execution_error'):
            raise ValueError('Apparently executed action lacks native dispatch evidence')
        if parsed['kind']=='send_message_to_user':
            expected=dict(action_type='send_msg_to_user',answer=parsed['args'][0],raw_action=action)
            if cursor>=len(trajectory) or trajectory[cursor]!=expected or row.get('execution_error') or turn!=len(rollout)-1:
                raise ValueError('Actual terminal action differs from full rollout')
            cursor+=1
        else:
            expected=dict(action_type=parsed['kind'],raw_action=action,args=list(parsed['args']),kwargs=parsed['kwargs'])
            if (cursor+1>=len(trajectory) or trajectory[cursor]!=expected or 'observation' not in trajectory[cursor+1] or
                    trajectory[cursor+1].get('info',{}).get('execution_error')!=row.get('execution_error') or
                    trajectory[cursor+1].get('info',{}).get('native_dispatch')!=dispatch):
                raise ValueError('Earlier selected action/state differs from evaluator trajectory')
            cursor+=2
    if cursor!=len(trajectory)-(1 if result.get('protocol_fallback') is not None else 0):
        raise ValueError('Evaluator trajectory contains missing or extra actions')


def audit_native(task, plan, paths, result, calls, instance):
    from openwebrl.webprm_evaluator_transcript import replay_native_evaluation
    from scripts.analyze_webprm_webarena_pilot import audit_terminal_action
    from openwebrl.webprm_webarena import evaluation_requirements, evaluator_protocol, NATIVE_EVALUATOR_PROTOCOL
    trajectory, verdict, terminal, outcome, rollout, provenance, auth = (
        read(paths[k]) for k in ('trajectory','verdict','terminal','outcome','rollout','provenance','authentication'))
    transcript = read(paths['evaluator_inputs'])
    scoring = evaluator_protocol(plan.get('evaluator_protocol',NATIVE_EVALUATOR_PROTOCOL))
    if (transcript.get('binding',{}).get('scoring_protocol') != scoring or
            verdict.get('evaluator_protocol',NATIVE_EVALUATOR_PROTOCOL) != scoring or
            transcript.get('binding',{}).get('termination') != terminal.get('termination') or
            verdict.get('termination') != terminal.get('termination') or
            scoring != NATIVE_EVALUATOR_PROTOCOL and not
            {'evaluator_protocol','native_score','native_success','native_evaluator_details','scoring_amendments'} <= verdict.keys()):
        raise ValueError('Saved evaluator protocol/termination differs from the frozen worker')
    manifests = plan.get('source_manifests', {})
    if (verdict.get('valid') is not True or verdict.get('score') != result.get('score') or
            verdict.get('success') != bool(result.get('success')) or result.get('score') != result.get('success') or
            result.get('terminal') != terminal or terminal.get('screenshot_sha256') != sha(paths['terminal_screenshot']) or
            not trajectory or 'observation' not in trajectory[0] or not rollout or len(rollout) != result.get('turns') or
            provenance.get('identity_sha256') != digest(plan) or provenance.get('original_instance') != instance or
            provenance.get('reset_id') != result.get('reset_id') or
            provenance.get('actor') != manifests.get('arbiter_harness') or
            provenance.get('observation') != manifests.get('arbiter_observation') or
            auth.get('authenticated') is not True or auth.get('state_provenance_sha256') != result.get('reset_id')):
        raise ValueError('Native outcome/trajectory/screenshot/provenance differs')
    selector = plan.get('selector')
    expected_critic = (manifests.get('viewer_reward') if selector in ('gpt4omini','arbiter7b') else
                       manifests.get('shepherd') if selector == 'shepherd8b' else
                       plan.get('piotr_manifest') if selector == 'piotr_selectionarm' else None)
    if provenance.get('critic') != expected_critic:
        raise ValueError('Saved critic source/representation differs from the frozen worker')
    requirements = evaluation_requirements(task.native_config())
    if requirements['helper_functions_required']:
        from openwebrl.webprm_webarena_helper_runtime import HELPER_RUNTIME_PROTOCOL
        helper = verdict.get('native_helper_manifest', {})
        if (helper != provenance.get('deployment_helpers') or helper.get('runtime_protocol') != HELPER_RUNTIME_PROTOCOL or
                helper.get('state_provenance_sha256') != result['reset_id'] or
                helper.get('configured_task_sha256') != task.configured_config_sha256 or
                helper.get('owned_site') != task.native_config()['sites'][0]):
            raise ValueError('Native helper runtime/reset/task binding differs')
    from PIL import Image
    with Image.open(paths['terminal_screenshot']) as image:
        image.verify()
    for key, value in dict(task_id=task.execution_id,released_task_id=task.released_task_id,
            raw_config_sha256=task.raw_config_sha256,configured_config_sha256=task.configured_config_sha256).items():
        if result.get(key) != value or outcome.get(key) != value:
            raise ValueError('Native outcome task/configuration identity differs')
    for key in ('valid','success','score','terminal','evaluator_protocol','native_score','native_success',
                'native_evaluator_details','scoring_amendments','observation_transcript'):
        if key in verdict and (outcome.get(key) != verdict[key] or result.get(key) != verdict[key]):
            raise ValueError('Before-close/raw/effective outcome provenance differs')
    audit_rollout(plan,trajectory,rollout,result)
    if 'observation' not in trajectory[-1] or result.get('protocol_fallback') is not None:
        audit_terminal_action(paths['result'].parent,result,plan,trajectory,rollout)
    judge, expected_calls = None, 0
    if requirements['paid_fuzzy_judge_may_be_required']:
        if 'native_fuzzy' not in paths:
            raise ValueError('Native fuzzy journal absent')
        judge, expected_calls = saved_fuzzy_proxy(plan,calls['pairs'],read_lines(paths['native_fuzzy']))
    proof = replay_native_evaluation(plan['arbiter_source'],task.native_config(),trajectory,
        transcript,verdict,fuzzy_judge=judge)
    if judge is not None and judge.provider_calls != expected_calls:
        raise ValueError('Unused or additional native judge evidence')
    native_success = int(verdict.get('native_success',verdict['success']))
    amended = bool(verdict.get('scoring_amendments'))
    return dict(success=int(verdict['success']),native_success=native_success,
        effective_semantic_status='valid',native_semantic_status='invalid' if amended and native_success else 'valid'), proof


def verify(inputs):
    if inputs.get('schema_version') != 1:
        raise ValueError('Explicit cell audit input schema required')
    campaign = canonical_runtime_path(inputs['campaign_root'])
    raw_cohort = read(reference(inputs['cohort']))
    cohort = raw_cohort.get('cohort',raw_cohort)
    plan = read(reference(inputs['worker_plan'],root=campaign))
    task = audit_identity(inputs,plan,cohort)
    refs = inputs['artifacts']
    if not BASE_ARTIFACTS <= refs.keys():
        raise ValueError('Incomplete terminal attempt/closure evidence')
    paths = {k:reference(v,root=campaign) for k,v in refs.items()}
    result = read(paths['result'])
    cell = inputs['cell']
    if (result.get('identity_sha256') != digest(plan) or result.get('actor_model') != cell['actor'] or
            result.get('selector') != cell['selector'] or result.get('task_id') != task.execution_id or
            type(result.get('valid')) is not bool or type(result.get('attempted')) is not bool):
        raise ValueError('Worker result fixed identity or accounting differs')
    audit_closure(inputs,paths,result)
    broker, budget = inputs['broker'], inputs['budget']
    expected_broker = task.execution_id+'/'+cell['actor']+'__'+cell['selector']
    if (broker['cell_id'] != expected_broker or budget['financial_contract']['cohort_sha256'] != digest(cohort) or
            budget['financial_contract']['protocol_sha256'] != cohort['protocol_sha256']):
        raise ValueError('Cumulative financial/cell identity differs')
    from openwebrl.webprm_journal_audit import audit_cell_calls
    calls = audit_cell_calls(plan,expected_broker,paths['api'],broker['audit_path'],budget['event_path'],
        budget['prefix_bytes'],budget['prefix_sha256'],budget['financial_contract'],
        broker_prefix_bytes=broker['prefix_bytes'],broker_prefix_sha256=broker['prefix_sha256'])
    if calls.get('verified') is not True:
        raise ValueError('Metered request audit failed')
    status = read(reference(read(paths['closure'])['broker_status_reference']))
    if calls.get('broker_policy_sha256s') != [status.get('policy_sha256')]:
        raise ValueError('Saved broker closure belongs to another financial epoch')
    if not result['attempted'] and (calls.get('api_request_count',0) or calls.get('paid_calls',0) or
            calls.get('local_calls',0) or calls.get('broker_only_failures') or 'verdict' in paths or 'rollout' in paths):
        raise ValueError('Unattempted result contradicts recorded model/episode activity')
    outcome = dict(success=None,native_success=None,effective_semantic_status='not_applicable',native_semantic_status='not_applicable')
    replay = None
    if result['valid']:
        if not result['attempted'] or not VALID_ARTIFACTS <= paths.keys():
            raise ValueError('Valid episode lacks complete native evidence')
        instance = audit_reddit_reset(inputs,paths,plan,result)
        outcome,replay = audit_native(task,plan,paths,result,calls,instance)
    elif not (result.get('error_type') or result.get('error') or result['watchdog'].get('reason')):
        raise ValueError('Invalid attempt lacks an explicit recorded failure')
    folder = paths['result'].parent
    covered = set(paths.values())
    for path in folder.rglob('*'):
        if path.is_file() and path.name != 'storage-state-private.json' and reference(
                dict(path=str(path),sha256=sha(path)),root=campaign) not in covered:
            raise ValueError('Saved episode evidence omitted from artifact manifest')
    record = dict(cell_id=cell['cell_id'],released_task_id=cell['task_id'],actor_model=cell['actor'],
        selector=cell['selector'],domain=cell['domain'],task_config_sha256=cell['task_config_sha256'],
        tasks_sha256=cohort['source_sha256'],cohort_sha256=digest(cohort),protocol_sha256=cohort['protocol_sha256'],
        audit_verified=True,terminal=True,attempted=result['attempted'],valid=result['valid'],**outcome)
    return dict(record=record,native_replay=replay,
        call_audit={k:v for k,v in calls.items() if k not in ('pairs','broker_only_failures')},
        complete_for_disposal=bool(result['valid'] and replay and replay['verified']))


def write_once(path, value):
    path = canonical_runtime_path(str(path))
    fd = os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as stream:
        stream.write((json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode())
        stream.flush();os.fsync(stream.fileno())
    fd = os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def audit_cell(input_path, output_path):
    inputs = read(input_path)
    output_path = canonical_runtime_path(str(output_path))
    if canonical_runtime_path(inputs['campaign_root']) not in output_path.parents:
        raise ValueError('Cell audit output must remain in its durable campaign')
    result_reference=inputs.get('artifacts',{}).get('result',{})
    if result_reference.get('path'):
        episode=canonical_runtime_path(result_reference['path']).parent
        if output_path.parent==episode or episode in output_path.parents:
            raise ValueError('Audit outputs must stay outside the immutable episode evidence directory')
    report = dict(schema_version=1,protocol=SCHEMA,input_sha256=sha(input_path),
        artifact_refs=inputs.get('artifacts',{}),evidence_manifest_sha256=digest(inputs.get('artifacts',{})),
        audit_verified=False,complete_for_disposal=False,errors=[],
        scope='Offline evidence/source/accounting consistency and pinned evaluator replay. Saved process closure is checked, not independently observed; GPU numerical internals, website truth and judge accuracy are not certified.')
    try:
        report.update(verify(inputs));report['audit_verified']=True
    except Exception as error:
        report['errors']=[dict(type=type(error).__name__,message=str(error)[:500])]
    write_once(output_path,report)
    record = None
    if report['audit_verified']:
        record = dict(report['record'],audit_sha256=sha(output_path))
        write_once(output_path.with_suffix('.record.json'),record)
    return dict(report,normalized_record=record,audit_path=str(output_path),audit_sha256=sha(output_path))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();result=audit_cell(args.input,args.output)
    print(json.dumps({k:result[k] for k in ('audit_verified','complete_for_disposal','errors','audit_path','audit_sha256')}))
    return 0 if result['audit_verified'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
