"""One approval-bound native WebArena cell; no site startup or allocation.

The owning controller supplies a fresh private Reddit lease and the sole host
broker. Original coordinator/browser/post-action code runs through explicit
transport and persistence adapters. Non-Arbiter methods are disclosed matched
controls, not recovered historical WebShepherd or no-search policy runners.
"""
from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
import re
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(os.environ.get('WEBPRM_NATIVE_SOURCE_ROOT', '/gpfs/projects/krishna/zixianma/OpenWebRL'))
sys.path.insert(0, str(PROJECT))

from openwebrl.webprm_lifecycle import (EPISODE_WATCHDOG, mark_episode_phase,
    run_supervised_episode, save_outcome_and_close)
from openwebrl.webprm_native_release import reference
from openwebrl.webprm_webarena import (TASK_FILE, EvaluationNotReady, digest,
    evaluation_requirements, verify_source)
from openwebrl.webprm_webarena_episode import TaskEnvelope
from openwebrl.webprm_webarena_worker import (cell_name, claim_lease, authenticated_state,
    broker_request, episode_evaluator, ViewerSelector, single_action_system_prompt,
    ENVELOPE_NO_SEARCH_PROMPT)
from openwebrl.webprm_native_piotr_v11 import NativePiotrSelector, RECEIPT as V11_RECEIPT

PROTOCOL = 'released_coordinator_original_browser_table5_v9'
TASK_IDS = (0, 2, 4, 8, 9, 10, 13, 14, 17, 21, 22, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 54, 55, 56, 58, 59, 62, 63, 65, 66, 67, 68, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 86, 88, 89, 93, 94, 95, 96, 97, 98, 99, 100, 101, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 145, 148, 149, 150, 151, 152, 153, 154, 156, 157, 160, 162, 163, 164)  # v9: Table 5 Shopping/CMS/Reddit/GitLab single-site tasks
METHODS = ('no_search','arbiter7b','shepherd8b','piotr_selectionarm')  # v10: Piotr SelectionARM
ACTOR = 'gpt-4o-2024-08-06'  # 05-13 shut down by OpenAI 2026-10-23
MINI = 'gpt-4o-2024-08-06'  # v8: released 'gpt-4o-mini' helper spec bound to gpt-4o (user: "just use gpt-4o")


def module_from_reference(name, ref):
    path=Path(ref['path'])
    if not path.is_absolute() or path.resolve()!=path or hashlib.sha256(path.read_bytes()).hexdigest()!=ref['sha256']:
        raise EvaluationNotReady('Native component source identity changed')
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,path)
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    elif Path(sys.modules[name].__file__).resolve()!=path:
        raise EvaluationNotReady('Conflicting native component import')
    return sys.modules[name]


def write(path,value):
    from openwebrl.jev_eval import write_json
    write_json(path,value)
    return reference(path)


def normalize_trajectory(raw, action_types):
    """Lossless evaluator projection, without inventing a terminal action.

    The original coordinator ends in the terminal control action without calling
    the browser or appending a new state. Reject any post-terminal state/action;
    the prospective runner must preserve this boundary without trimming evidence.
    """
    result=[]
    for index,entry in enumerate(raw):
        if index%2==0:
            obs=entry['observation'];info=entry['info']
            result.append(dict(observation={k:v for k,v in obs.items() if k!='image'},
                info=dict(fail_error=info.get('fail_error',''),observation_metadata=deepcopy(info['observation_metadata']))))
        else:
            action={k:(v.tolist() if hasattr(v,'tolist') else v) for k,v in entry.items()}
            kind=action_types(int(action['action_type']))
            action['action_type']='send_msg_to_user' if kind==action_types.SEND_MESSAGE_TO_USER else kind.name.lower()
            result.append(action)
    if any(index%2 and entry.get('action_type')=='send_msg_to_user' and index!=len(result)-1
           for index,entry in enumerate(result)):
        raise EvaluationNotReady('Original terminal control cannot have a following browser state or action')
    if result and len(result)%2==0 and result[-1].get('action_type')!='send_msg_to_user':
        raise EvaluationNotReady('Nonterminal browser action lacks its actual following state')
    return result


JSON_FENCE = __import__('re').compile(r'\A\s*```(?:json)?[ \t]*\n?(.*?)\n?\s*```\s*\Z', __import__('re').S)


def strip_json_fence(text):
    """v7: return the body of a whole-reply ```json fence; any other text is unchanged."""
    match = JSON_FENCE.match(text) if isinstance(text, str) else None
    return match.group(1) if match else text


def json_mode(value):
    """v7: native actor policy requests ask for a JSON object (OM2W protocol uses the same)."""
    if value['purpose'] == 'native_policy':
        value['payload']['response_format'] = {'type': 'json_object'}
        value['policy_json_mode'] = 'v7_user_directed_json_object'
    return value


def validate_single_action_response(text):
    parsed=json.loads(text)
    if not isinstance(parsed,dict) or not isinstance(parsed.get('BLOCKS'),list) or len(parsed['BLOCKS'])!=1:
        raise EvaluationNotReady('Single-action control requires exactly one raw BLOCK')


# ---- v8 (2026-10-25, user: "ok just use gpt-4o. and go for it"); see v8_operational_changes. ----
V8_VIEWPORT = {'width': 1280, 'height': 720}  # v11: user-chosen 1280x720 (WebArena browser_env default, envs.py:73; was the released service's 1024x2490, run_management.py:172)
V8_REWARD_DECODING = dict(temperature=0., top_p=1.)  # released reward_lm_config temperature 0 / README do_sample=False
V8_TERMINAL = re.compile(r'\A\s*send_msg_to_user\b(.*)\Z', re.S)
V8_BRACKETED = re.compile(r'\A\s*send_msg_to_user\s*\[.*\]\s*\Z', re.S)
V8_RECEIPT = dict(viewport=V8_VIEWPORT, compatibility_geometry=True, auxiliary_wire_model=MINI,
    reward_decoding=V8_REWARD_DECODING, terminal_policy='stop_competes_with_selector_candidates_except_no_search',
    bracketless_terminal='send_msg_to_user [text]')


def v8_bracket(action):
    """v8: execute a bracketless terminal message as `send_msg_to_user [text]`; every other string is unchanged."""
    if not isinstance(action, str):
        return action
    match = V8_TERMINAL.match(action)
    if match is None or V8_BRACKETED.match(action):
        return action
    text = match.group(1).strip()
    return f'send_msg_to_user [{text}]' if text else action


def v8_terminal(action):
    """Bracket-normalize a detector terminal BLOCK; None/empty actions pass through unchanged."""
    if isinstance(action, dict) and isinstance(action.get('action'), str):
        return dict(action, action=v8_bracket(action['action']))
    return action


def v8_plain(value):
    """JSON form of released dataclass values (the released tournament returns a BlockInfo winner)."""
    from dataclasses import is_dataclass
    if is_dataclass(value):
        return v8_plain(asdict(value))
    if isinstance(value, dict):
        return {str(k): v8_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [v8_plain(v) for v in value]
    return value


def v8_action(selected):
    return selected.get('action') if isinstance(selected, dict) else getattr(selected, 'action', None)


def v8_pool(candidates, terminal):
    """Policy candidates plus the detector stop action (unless an identical action is already a candidate)."""
    pool = deepcopy(list(candidates))
    if not terminal or not isinstance(terminal.get('action'), str) or not terminal['action'].strip():
        return pool, None
    for index, candidate in enumerate(pool):
        if candidate.get('action') == terminal['action']:
            return pool, index
    pool.append(dict(thought=terminal.get('thought', ''), action=terminal['action'],
                     confidence=terminal.get('confidence', 0.0), status='v8_terminal_detector_candidate',
                     reasoning=None, meaning=None))
    return pool, len(pool)-1


def v8_decision_adapter(method, select_control, record):
    """Run the released CoordinatorAgent.execute_step with the v8 terminal policy.

    arbiter7b/shepherd8b: the released terminal detection still runs, but its action is held back and added to
    the tournament / WebShepherd pool instead of short-circuiting it. If the policy produced no candidates the
    stop action is the only option and is selected. no_search: unchanged released override (gpt-4o detector).
    select_control(pool, inputs, context) -> (index, trace) for the non-tournament methods.
    """
    async def adapter(*, native, agent, context, inputs):
        stash = {}
        released_detect = type(agent)._detect_terminal_if_needed
        async def detect(*args, **kwargs):
            action = v8_terminal(await released_detect(agent, *args, **kwargs))
            if method in ('no_search', 'piotr_selectionarm') or not action:  # v10: Piotr keeps the released override
                return action
            stash['terminal'] = deepcopy(action)
            return None
        released_tournament = native.coordinator.run_tournament
        async def tournament(**kwargs):
            pool, stop_index = v8_pool(kwargs['candidates'], stash.get('terminal'))
            if method == 'arbiter7b':
                outcome = await released_tournament(**dict(kwargs, candidates=pool))
                record(inputs['current_step'], dict(method=method, candidates=pool, v8_stop_index=stop_index,
                    v8_terminal=stash.get('terminal'), tournament_success=bool(outcome.get('success')),
                    final_winner=v8_plain(outcome.get('final_winner')) if outcome.get('success') else None,
                    adaptation='v8_stop_competes_in_released_tournament'))
                return outcome
            index, trace = await select_control(pool, inputs, context)
            record(inputs['current_step'], dict(method=method, candidates=pool, selected_index=index, trace=trace,
                v8_stop_index=stop_index, v8_terminal=stash.get('terminal'),
                adaptation='shared_released_policy_matched_control' if method == 'no_search' else
                    'v10_piotr_selectionarm_released_five_block_pool' if method == 'piotr_selectionarm' else 'v8_stop_competes_with_shepherd_pool'))
            return dict(success=True, final_winner=deepcopy(pool[index]))
        agent._detect_terminal_if_needed = detect
        native.coordinator.run_tournament = tournament
        try:
            decision = await agent.execute_step(**inputs)
        finally:
            native.coordinator.run_tournament = released_tournament
            del agent._detect_terminal_if_needed
        if method != 'no_search' and not v8_action(decision.get('selected_action')) and stash.get('terminal'):
            decision = dict(decision, selected_action=deepcopy(stash['terminal']), v8_only_stop_candidate=True)
        return decision
    return adapter


# ---- v9 (2026-10-26): paper Best-of-N candidate protocol; four Table 5 sites. ----
V9_SAMPLES = 20  # WebShepherd App. F.2 "we sample 20 output sequences"
V9_TOP_N = 5  # WebArbiter Sec. 5.2.1 "N=5 candidate actions"; WebShepherd "top-n most frequent actions"
V9_SAMPLING = dict(n=V9_SAMPLES, temperature=1.0, top_p=1.0)  # "nucleus sampling with a temperature of 1.0"
V9_RECEIPT = dict(samples=V9_SAMPLES, top_n=V9_TOP_N, sampling=V9_SAMPLING, prompt='disclosed_single_action_envelope_v2',
    ranking='frequency_desc_then_first_sampled', validity='released_filter_candidates',
    confidence='sample_share_of_valid', methods=['arbiter7b', 'shepherd8b'], no_search='unchanged_v8')


def v9_aggregate(response, observation_text, sampling, choice_text):
    """Turn one n=20 policy response into the released sampler's BLOCKS text plus a full record.

    choice_text(single_choice_response) -> text validates one choice exactly like a one-choice reply.
    """
    from collections import Counter
    choices = response.get('choices')
    if not isinstance(choices, list) or len(choices) != V9_SAMPLES:
        raise EvaluationNotReady('v9 policy reply must carry exactly 20 samples')
    usage = response.get('usage')
    samples, valid = [], []
    for index, choice in enumerate(choices):
        row = dict(index=index, finish_reason=choice.get('finish_reason') if isinstance(choice, dict) else None)
        try:
            text = strip_json_fence(choice_text(dict(choices=[choice], usage=usage)))
            parsed = json.loads(text)
            blocks = parsed.get('BLOCKS') if isinstance(parsed, dict) else None
            if not isinstance(blocks, list) or len(blocks) != 1 or not isinstance(blocks[0], dict):
                raise ValueError('not exactly one BLOCK')
            thought = (blocks[0].get('thought') or '').strip() if isinstance(blocks[0].get('thought'), str) else ''
            action = (blocks[0].get('action') or '').strip() if isinstance(blocks[0].get('action'), str) else ''
            row.update(thought=thought, action=action)
            block = sampling.BlockInfo(thought=thought, action=action, confidence=None)
            if thought and action and sampling.filter_candidates([block], observation_text):
                row['status'] = 'valid'
                valid.append(row)
            else:
                row['status'] = 'rejected_by_released_filter'
        except Exception as error:
            row.update(status='unparsed', error_type=type(error).__name__)
        samples.append(row)
    counts = Counter(row['action'] for row in valid)
    first = {}
    for row in valid:
        first.setdefault(row['action'], row)
    ranked = sorted(counts, key=lambda action: (-counts[action], first[action]['index']))[:V9_TOP_N]
    blocks = [dict(thought=first[action]['thought'], action=action, confidence=counts[action]/len(valid),
                   frequency=counts[action]) for action in ranked]
    record = dict(protocol=V9_RECEIPT, samples=samples, valid_samples=len(valid), distinct_valid_actions=len(counts),
                  candidates=[dict(action=b['action'], frequency=b['frequency'], first_sample=first[b['action']]['index'])
                              for b in blocks])
    return json.dumps(dict(BLOCKS=blocks)), record


def _run_episode(task_record, arm, config):
    task=TaskEnvelope.from_release(task_record['released'],config['endpoints'])
    method=config['selector'];folder=Path(config['output'])/task.execution_id/arm
    result=dict(task_id=task.execution_id,arm=arm,actor_model=config['actor_model'],selector=method,
        identity_sha256=config['identity_sha256'],attempted=False,valid=False,success=0,started_at=time.time(),
        protocol=PROTOCOL,paper_exact_protocol=False)
    try:
        if config.get('authorization_status')!='approved' or config.get('api_usd_cap')!=8.0:  # v9 package cap
            raise EvaluationNotReady('Exact positive v9 authorization required')
        if str(config.get('job_id'))!=os.environ.get('SLURM_JOB_ID'):
            raise EvaluationNotReady('Worker allocation identity changed')
        for path,expected in config['code_sha256'].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:
                raise EvaluationNotReady('Frozen native worker dependency changed')
        components=config['native_components']
        native_module=module_from_reference('_webprm_native_episode_v1',components['episode'])
        broker_module=module_from_reference('_webprm_native_broker_v2',components['broker'])
        native=native_module.load_native(config['arbiter_source'])
        request=broker_request(config,task,method)
        instance,reset_id=claim_lease(task,method,config)
        config=dict(config,reset_id=reset_id)
        from openwebrl.webprm_runtime import prepare_browser_runtime
        prepare_browser_runtime()
        result['attempted']=True
        mark_episode_phase(config,'native_authentication')
        from openwebrl.webprm_webarena_sites import site_authentication
        state,auth=site_authentication(task,config,instance,config.get('chromium',str(native_module.CHROMIUM)),
            folder,reddit_authenticate=authenticated_state)
        helpers=None
        if evaluation_requirements(task.native_config())['helper_functions_required']:
            from openwebrl.webprm_webarena_helper_runtime import owned_native_helpers
            helpers=owned_native_helpers(task,config,instance)
        evaluator=episode_evaluator(task,config,request,folder,deployment_helpers=helpers)
        Spec=native.tools['engine'].ModelSpec
        v9_domain=task.native_config()['sites'][0]  # v9: the task's own Table 5 site
        if len(task.native_config()['sites'])!=1:raise EvaluationNotReady('v9 admits single-site tasks only')
        v9_observation={}
        helper=Spec('openai','gpt-4o-mini','chat',dict(max_tokens=2048,context_length=8192,temperature=1.,top_p=.9))
        critic=config.get('arbiter_model') if method=='arbiter7b' else config.get('shepherd_model') if method=='shepherd8b' else None
        context=native.coordinator.RunContext.create(run_id=config['identity_sha256'][:16]+'-'+task.execution_id+'-'+method,
            intent=task.actor_view['intent'],domain=v9_domain,domain_knowledge=native.history.load_domain_knowledge(v9_domain) or '',
            policy_model=Spec('openai',ACTOR,'chat'),reward_model=Spec('openai',critic,'chat') if critic else None)
        routes={('openai',ACTOR,'chat'):dict(purpose='native_policy',allowed_gen_configs=[None]),
                ('openai','gpt-4o-mini','chat'):dict(purpose='native_auxiliary',allowed_gen_configs=[helper.gen_config])}
        local=[]
        if method=='arbiter7b':
            routes[('openai',critic,'chat')]=dict(purpose='native_reward',allowed_gen_configs=[None]);local.append(critic)
        config_receipt=dict(routes=[dict(provider=k[0],model=k[1],mode=k[2],**v) for k,v in routes.items()],
            local_models=local,declared_default_model=asdict(helper),wire_model_aliases={'gpt-4o-mini':MINI},
            max_attempts=512,selector=method,no_search_prompt_version=ENVELOPE_NO_SEARCH_PROMPT if method=='no_search' else None,v8=V8_RECEIPT,
            v9=V9_RECEIPT if method in ('arbiter7b','shepherd8b') else None,v11=V11_RECEIPT if method=='piotr_selectionarm' else None)
        context_ref=write(folder/'native-context.json',dict(run_id=context.run_id,intent=context.intent,domain=context.domain,
            domain_knowledge=context.domain_knowledge,policy_model=asdict(context.policy_model),
            reward_model=asdict(context.reward_model) if context.reward_model else None))
        bridge_ref=write(folder/'native-bridge-config.json',config_receipt)
        class CellBridge(broker_module.NativeBrokerBridge):
            def request(self,engine,system,user,model=None):
                value=json_mode(super().request(engine,system,user,model))
                if value['purpose']=='native_reward':
                    value['payload'].update(V8_REWARD_DECODING)  # v8 greedy WebArbiter-7B
                    value['reward_decoding']='v8_greedy_released_reward_lm_config'
                if value['purpose']=='native_policy' and method=='piotr_selectionarm':  # v10: released 5-BLOCK prompt
                    value['upstream_unadapted_payload']=deepcopy(value['payload'])
                    value['policy_adaptation']='v10_released_five_block_prompt_unchanged'
                elif value['purpose']=='native_policy':
                    value['upstream_unadapted_payload']=deepcopy(value['payload'])
                    value['payload']['messages'][0]['content']=single_action_system_prompt(
                        value['payload']['messages'][0]['content'],version=ENVELOPE_NO_SEARCH_PROMPT)
                    value['policy_adaptation']='disclosed_single_action_envelope_v2'
                    if method!='no_search':
                        value['payload'].update(V9_SAMPLING)
                        value['policy_adaptation']='v9_best_of_20_top5_frequency'
                return value
            def response_text(self,response):
                purpose=response.get('_webprm_broker',{}).get('purpose')
                if purpose=='native_policy' and method in ('arbiter7b','shepherd8b'):
                    if 'text' not in v9_observation:raise EvaluationNotReady('v9 policy reply without its observation')
                    single=super().response_text
                    def choice_text(value):return single(value)[0]
                    usage=deepcopy(response.get('usage'))
                    total=dict(choices=[dict(finish_reason='stop',message=dict(role='assistant',content='{}'))],usage=usage)
                    single(total)  # usage schema/reconciliation exactly as a one-choice reply
                    text,record=v9_aggregate(response,v9_observation['text'],native.tools['sampling'],choice_text)
                    v9_observation['sequence']=v9_observation.get('sequence',0)+1
                    write(folder/f"v9-samples-{v9_observation['sequence']:03d}.json",dict(record,step=v9_observation.get('step'),
                        request_id=response.get('_webprm_broker',{}).get('request_id')))
                    return text,usage
                text,usage=super().response_text(response)
                if purpose in ('native_policy','native_auxiliary'):
                    text=strip_json_fence(text)
                if purpose=='native_policy' and method=='no_search':
                    validate_single_action_response(text)
                return text,usage
        bridge=CellBridge(native.tools['engine'].LMEngine,routes=routes,client=request,config=config,
            cell_id=task.execution_id+'/'+arm,local_models=local,declared_default_model=helper,
            wire_model_aliases={'gpt-4o-mini':MINI},evidence_root=folder/'native-models',max_attempts=512)
        shepherd=ViewerSelector(method,config,folder/'requests-unused.jsonl',request=request) if method=='shepherd8b' else None
        piotr=NativePiotrSelector(config,folder/'requests-piotr.jsonl',request=request) if method=='piotr_selectionarm' else None
        async def select_control(candidates,inputs,context):
            if method=='no_search':
                if len(candidates)!=1:raise EvaluationNotReady('Single-action control must return one admissible candidate')
                return 0,[]
            if method=='piotr_selectionarm':  # v10: the frame is the browser's current (pre-action) capture
                return await asyncio.to_thread(piotr.select,candidates,dict(intent=context.intent,current_url=inputs['current_url'],
                    observation_text=inputs['observation_text']),deepcopy(browser.observations[-1]),inputs['current_step'])
            proposal=dict(candidates=deepcopy(candidates),terminal=False,context=dict(intent=context.intent,start_url=task.actor_view['start_url'],
                text_observation=inputs['observation_text'],current_url=inputs['current_url'],
                trajectory=inputs['trajectory_text'],state_diff=inputs['state_diff']))
            return await asyncio.to_thread(shepherd.select,proposal,inputs['current_step'])
        v8_adapter=v8_decision_adapter(method,select_control,
            lambda step,value:write(folder/f'selector-{step:03d}.json',value))
        async def decision_adapter(*,native,agent,context,inputs):
            v9_observation.update(text=inputs['observation_text'],step=inputs['current_step'])
            decision=await v8_adapter(native=native,agent=agent,context=context,inputs=inputs)
            if piotr:piotr.note_decision(decision)
            return decision
        v8_normalized=[]
        def evaluate(configured,raw,page,terminal):
            mark_episode_phase(config,'native_evaluation')
            raw_projection=[]
            for index,entry in enumerate(raw):
                if index%2==0:
                    observation=entry['observation'];info=entry['info']
                    raw_projection.append(dict(observation={k:v for k,v in observation.items() if k!='image'},
                        image=browser.observations[index//2]['image'],info=dict(fail_error=info.get('fail_error',''),
                            observation_metadata=deepcopy(info['observation_metadata']))))
                else:raw_projection.append(native_module.plain(entry))
            write(folder/'native-raw-trajectory.json',raw_projection)
            normalized=normalize_trajectory(raw,native.actions.ActionTypes)
            write(folder/'native-evaluator-trajectory.json',normalized)
            verdict=evaluator.evaluate(task.native_config(),normalized,page,termination='stop' if terminal else 'turn_limit')
            write(folder/'native-evaluation.json',verdict)
            return verdict
        resolved=task.native_config();resolved['storage_state']=str(folder/'storage-state-private.json') if state is not None else None
        if state is not None and not Path(resolved['storage_state']).exists():
            write(Path(resolved['storage_state']),state)
            os.chmod(resolved['storage_state'],0o600)
        class PhasedNativeBrowser(native_module.NativeBrowser):
            async def start(self,*args):
                mark_episode_phase(config,'native_observation')
                return await super().start(*args)
            async def execute(self,action_string):
                mark_episode_phase(config,'browser_step')
                if piotr:piotr.note_execution(action_string,deepcopy(self.observations[-1]))  # v10: pre-action grounding
                executed=v8_bracket(action_string)
                if executed!=action_string:v8_normalized.append(dict(selected=action_string,executed=executed))
                return await super().execute(executed)
            async def close(self):
                mark_episode_phase(config,'cleanup')
                return await super().close()
        browser=PhasedNativeBrowser(native,folder/'native-browser',viewport=dict(V8_VIEWPORT),compatibility_geometry=True,
            allowed_origins=['{0.scheme}://{0.netloc}'.format(__import__('urllib.parse').parse.urlsplit(instance['endpoint']))],chromium=config.get('chromium',native_module.CHROMIUM),
            launch_parent=config.get('browser_tmp_root'))
        episode=asyncio.run(native_module.run_episode(task=resolved,context=context,browser=browser,
            completion_scope=bridge.installed(),evaluate=evaluate,evidence_root=folder/'native-episode',max_steps=30,native=native,
            decision_adapter=decision_adapter))
        native_ref=reference(folder/'native-episode/result.json')
        verdict=episode.get('verdict')
        valid=bool(episode.get('valid') and isinstance(verdict,dict) and verdict.get('valid')
            and episode.get('closure',{}).get('closed') and not episode.get('closure',{}).get('errors'))
        result.update(valid=valid,success=int(valid and verdict['success']),
            native_success=int(valid and verdict.get('native_success',verdict['success'])),
            native_evaluation=verdict,native_episode=native_ref,native_context=context_ref,native_bridge_config=bridge_ref,
            reset_id=reset_id,authentication=json.loads((folder/'authentication.json').read_text()),authentication_policy=auth,original_instance=instance,
            turns=sum(not row.get('fallback',False) for row in episode.get('selected_actions',[])),
            executed_actions=len(episode.get('selected_actions',[])),termination=episode.get('status'),
            terminal=episode.get('terminal'),raw_effective_verdicts_preserved=True,
            v10_piotr_ungroundable_history_actions=piotr.ungroundable if piotr else None,v8_terminal_bracket_normalizations=v8_normalized,
            method_interpretation='released_tournament_coordinator' if method=='arbiter7b' else
                'disclosed_single_action_control' if method=='no_search' else
                'v10_piotr_selectionarm_released_five_block_pool' if method=='piotr_selectionarm' else 'WebShepherd_critic_best_of_20_top5_frequency')
        if not valid:result.update(error_type=episode.get('error_type','NativeEpisodeInvalid'),error=episode.get('error','Native episode/closure/verdict incomplete'))
    except Exception as error:
        result.update(error_type=type(error).__name__,error=str(error))
    finally:
        save_outcome_and_close(result,folder,None,config)
    return result


def run_episode(task_record,selector,config):
    if selector not in METHODS or config['actor_model']!=ACTOR or task_record['released']['task_id'] not in TASK_IDS:
        raise EvaluationNotReady('Cell outside fixed v9 cohort')
    if config.get('protocol')!=PROTOCOL or config.get('max_turns')!=30:
        raise EvaluationNotReady('Native coordinator protocol/turn cap changed')
    arm=cell_name(config['actor_model'],selector)
    return run_supervised_episode(task_record,arm,dict(config,selector=selector,cell_name=arm),_run_episode)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--execute',action='store_true');parser.add_argument('--task-id',type=int,required=True)
    parser.add_argument('--actor',required=True);parser.add_argument('--selector',choices=METHODS,required=True)
    parser.add_argument('--lease',type=Path,required=True);args=parser.parse_args()
    plan=json.loads(args.plan.read_text());identity=digest(plan)
    if not args.execute:raise EvaluationNotReady('Native cell worker is execution-only; use controller preview')
    if args.task_id not in TASK_IDS or plan.get('task_ids')!=[args.task_id] or plan.get('actors')!=[ACTOR] or args.actor!=ACTOR or plan.get('selectors')!=[args.selector]:
        raise EvaluationNotReady('Worker CLI differs from exact assigned plan')
    if verify_source(plan['arbiter_source'])!=plan['source_manifest']:raise EvaluationNotReady('Native task/evaluator source changed')
    rows=json.loads((Path(plan['arbiter_source'])/TASK_FILE).read_text())
    raw=next(x for x in rows if x['task_id']==args.task_id)
    if digest([raw])!=plan['tasks_sha256']:raise EvaluationNotReady('Released task identity changed')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    write(output/'manifest.json',dict(identity=plan,identity_sha256=identity))
    task=TaskEnvelope.from_release(raw,plan['endpoints'])
    config=dict(plan,output=str(output),actor_model=args.actor,identity_sha256=identity,
        episode_watchdog=dict(EPISODE_WATCHDOG),lease_path=str(args.lease.resolve()),
        lease_sha256=hashlib.sha256(args.lease.read_bytes()).hexdigest())
    result=run_episode(dict(task_id=task.execution_id,released=raw),args.selector,config)
    print(json.dumps({k:result[k] for k in ['task_id','arm','attempted','valid','success']}))


if __name__=='__main__':main()
