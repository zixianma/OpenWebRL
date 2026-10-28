"""Budgeted BrowserGym pilot with native actor/critic interfaces.

This is a supported-interface OM2W adaptation, not an exact paper reproduction.
"""
from __future__ import annotations
import ast
import base64
from collections import deque
import fcntl
import hashlib
import io
import ipaddress
import json
import os
from pathlib import Path
import random
import time
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen, build_opener, ProxyHandler, HTTPRedirectHandler

from openwebrl.webprm_baselines import ARMS, candidate_pool
from openwebrl.webprm_source import load_shepherd, load_arbiter, load_arbiter_causal_renderer
from openwebrl.jev_eval import write_json, append_json, judge_messages, judge_protocol, parse_verdict, JUDGE_SOURCE
from openwebrl.webprm_runtime import load_api_credentials, prepare_browser_runtime, live_task_type
from openwebrl.webprm_lifecycle import mark_episode_phase, run_supervised_episode, save_outcome_and_close

CHROMIUM = '/gpfs/scrubbed/zixianma/openwebrl-runtime/browsers/chromium-1208/chrome-linux64/chrome'

ACTOR = 'gpt-4o-mini-2024-07-18'
JUDGE = 'o4-mini-2025-04-16'
PILOT_ARMS = ARMS
PRICES = {ACTOR: (.15e-6, .60e-6), JUDGE: (1.10e-6, 4.40e-6)}
NO_ADMISSIBLE_ACTION = '__webprm_no_admissible_native_action__'


class _NoLocalRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError('Local critic endpoints must not redirect')


def loopback_urlopen(request, *, timeout):
    """Contact an owned loopback service without inheriting an external proxy.

    External provider calls still use the ordinary proxy-aware urlopen path.
    Reject redirects so a local endpoint cannot turn this into a proxy bypass
    for another destination.
    """
    url = request.full_url if isinstance(request, Request) else request
    parsed = urlsplit(url)
    host = parsed.hostname
    try:
        loopback = host == 'localhost' or ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = False
    if parsed.scheme not in ('http', 'https') or not loopback or parsed.username or parsed.password:
        raise ValueError('Direct local transport requires an uncredentialed loopback URL')
    return build_opener(ProxyHandler({}), _NoLocalRedirect()).open(request, timeout=timeout)


def arm_mode(arm):
    if arm not in ARMS:
        raise ValueError('Unknown pilot arm')
    family = 'arbiter' if arm.startswith('arbiter') else 'shepherd'
    mode = 'unguided' if arm.endswith('unguided') else 'random' if arm.endswith('random5') else 'guided'
    return family, mode



class SharedBudget:
    """Durable atomic reservations across processes; unknown calls retain cost."""
    def __init__(self, path, cap=30.):
        self.path = Path(path)
        self.cap = cap
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def change(self, callback):
        with self.path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            data = json.loads(self.path.read_text()) if self.path.exists() else dict(cap=self.cap, charged=0., calls=[])
            if data['cap'] != self.cap:
                raise ValueError('Budget cap cannot change across retries')
            value = callback(data)
            write_json(self.path, data)
            directory=os.open(str(self.path.parent),os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            return value

    def reserve(self, amount, model):
        def update(data):
            if data.get('halt_reason') or amount < 0 or data['charged'] + amount > data['cap']:
                raise RuntimeError('Shared API budget exhausted')
            ident = len(data['calls'])
            data['charged'] += amount
            data['calls'].append(dict(id=ident, model=model, reserved=amount, state='reserved'))
            return ident
        return self.change(update)

    def settle(self, ident, actual, usage):
        def update(data):
            row = data['calls'][ident]
            if row['state'] != 'reserved':
                raise RuntimeError('Reservation already settled')
            if actual > row['reserved']:
                data['halt_reason']='observed_usage_exceeded_reservation'
                data['charged']+=actual-row['reserved']
                row.update(state='overrun',charged=actual,usage=usage)
                return False
            data['charged'] -= row['reserved'] - actual
            row.update(state='settled', charged=actual, usage=usage)
            return True
        if not self.change(update):
            raise RuntimeError('Observed usage exceeded reservation; all paid calls halted')


def safe_action(text, action_set):
    tree = ast.parse(text.strip(), mode='eval')
    call = tree.body
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
        raise ValueError('One named action call required')
    if call.func.id not in action_set.action_set:
        raise ValueError('Unsupported BrowserGym action')
    for node in call.args:
        ast.literal_eval(node)
    for keyword in call.keywords:
        if keyword.arg is None:
            raise ValueError('Expanded action keywords forbidden')
        ast.literal_eval(keyword.value)
    action_set.to_python_code(text.strip())
    return text.strip()


def native_tournament(candidates, compare):
    """Viewer TournamentRunner ordering: leftover bye precedes round winners."""
    queue = deque(candidates)
    trace = []
    while len(queue) > 1:
        pairs = []
        while len(queue) >= 2:
            pairs.append((queue.popleft(), queue.popleft()))
        for left, right in pairs:
            verdict = compare(left, right)
            if verdict not in (1, 2):
                raise ValueError('Arbiter verdict missing or invalid')
            queue.append((left, right)[verdict - 1])
            trace.append(dict(left=left, right=right, verdict=verdict))
    return queue[0], trace


def input_token_upper_bound(messages):
    images=0
    def stripped(value):
        nonlocal images
        if isinstance(value,list):
            return [stripped(item) for item in value]
        if isinstance(value,dict):
            if value.get('type')=='image_url':
                images+=1
                return {'type':'image_url','image_url':{'url':'image_payload_excluded','detail':value.get('image_url',{}).get('detail','auto')}}
            return {key:stripped(item) for key,item in value.items()}
        return value
    text=stripped(messages)
    return len(json.dumps(text,ensure_ascii=False).encode())+images*25000+4096


def completion(config, payload, journal, *, local=False, endpoint=None, purpose=None, budget=None):
    mark_episode_phase(config, 'critic_request' if local else 'judge_request' if payload['model'] == JUDGE else 'actor_request',
                       purpose=purpose, model=payload['model'])
    if time.time() >= config['deadline']:
        raise TimeoutError('Pilot deadline reached before request')
    # Fail before reserving money or logging request intent. Direct workers must
    # load credentials too; they cannot depend on a particular CLI parent.
    key = 'local' if local else load_api_credentials(config.get('env_file'))
    injected_budget = budget is not None
    budget = budget if injected_budget else SharedBudget(config['budget_path'], cap=config.get('api_usd_cap', 30.))
    reservation = None
    if not local:
        judge = payload['model'] == JUDGE
        rates = config.get('prices_usd_per_token', PRICES)[payload['model']]
        if injected_budget:
            from decimal import Decimal
            rates = [Decimal(str(rate)) for rate in rates]
        # UTF8 bytes upper-bound text tokenization; images have explicit allowance.
        input_bound = input_token_upper_bound(payload['messages'])
        output_bound = payload.get('max_completion_tokens', payload.get('max_tokens', 1024)) * payload.get('n', 1)
        reservation = budget.reserve(2*(input_bound*rates[0] + output_bound*rates[1]), payload['model'])
    append_json(journal, dict(event='request', time=time.time(), local=local,
                             reservation=reservation, payload=payload, purpose=purpose))
    url = (endpoint if local else config.get('api_base', 'https://api.openai.com/v1')).rstrip('/') + '/chat/completions'
    request = Request(url, data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json', 'Authorization': 'Bearer '+key})
    try:
        open_request = loopback_urlopen if local else urlopen
        with open_request(request, timeout=max(1,min(90,config['deadline']-time.time()))) as response:
            result = json.load(response)
    except HTTPError as error:
        # HTTP status alone hid a local context-length rejection. Preserve a
        # bounded response body and the original exception chain; do not turn
        # failed or ambiguous provider calls into uncharged budget entries.
        body_limit = 16384
        body = error.read(body_limit + 1)
        body_truncated = len(body) > body_limit
        body = body[:body_limit].decode('utf-8', errors='replace')
        if not local:
            body = body.replace(key, '[REDACTED_API_KEY]')
        append_json(journal, dict(event='response_error', time=time.time(), local=local,
            reservation=reservation, model=payload['model'], purpose=purpose,
            status_code=error.code, error_body=body, body_truncated=body_truncated))
        detail = body.strip()[:2048]
        reason = f'{error.reason}: {detail}' if detail else str(error.reason)
        raise HTTPError(error.url, error.code, reason, error.headers,
                        io.BytesIO(body.encode())) from error
    append_json(journal, dict(event='response', time=time.time(), reservation=reservation, response=result))
    if not isinstance(result, dict) or result.get('model') != payload['model']:
        raise RuntimeError('Returned model identity differs from requested model; reservation remains charged')
    if not local:
        usage = result.get('usage')
        if not isinstance(usage, dict) or any(type(usage.get(key)) is not int or usage[key] < 0
                for key in ('prompt_tokens', 'completion_tokens')):
            raise RuntimeError('Missing or invalid usage; reservation remains charged')
        budget.settle(reservation, usage['prompt_tokens']*rates[0]+usage['completion_tokens']*rates[1], usage)
    if local and any(choice.get('finish_reason') != 'stop' for choice in result.get('choices', [])):
        raise ValueError('Incomplete critic response; partial checklist scores are not usable')
    return result


def single_response_text(response):
    choices = response.get('choices', [])
    if len(choices) != 1 or choices[0].get('finish_reason') != 'stop':
        raise ValueError('Incomplete auxiliary actor response')
    text = choices[0].get('message', {}).get('content')
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Empty auxiliary actor response')
    return text


def arbiter_auxiliary(harness, config, journal, context, previous_tree, previous_terminal, action_error,
                      *, actor_model=ACTOR, request=None):
    """Released state-diff and terminal stages, shared by all three conditions."""
    state_diff = ''
    request = request or completion
    if previous_tree is not None:
        messages = [dict(role='system', content=harness.state_diff_system_prompt),
                    dict(role='user', content=harness.render_state_diff_prompt(
                        prev_text=previous_tree, cur_text=context['text_observation']))]
        response = request(config, dict(model=actor_model, messages=messages, temperature=1.,
                              top_p=.9, max_tokens=2048), journal, purpose='state_diff')
        state_diff = single_response_text(response)
    # Execution exceptions are facts unavailable to the two-tree native diff
    # tool. Preserve them explicitly; do not infer success from a changed page.
    if action_error:
        state_diff += '\nLast action execution error: ' + action_error
    system, user = harness.build_terminal_prompt(intent=context['intent'],
        observation=context['text_observation'], current_url=context['current_url'],
        trajectory=context['trajectory'], state_diff=state_diff,
        previous_terminal_action=previous_terminal)
    response = request(config, dict(model=actor_model, messages=[dict(role='system', content=system),
                          dict(role='user', content=user)], temperature=1., top_p=.9, max_tokens=2048),
                          journal, purpose='terminal')
    block = harness.parse_terminal_response(single_response_text(response)).terminal_action
    if block is None:
        # The released parser treats malformed detection text as not complete.
        # Keep its actual None result, without inventing a reflexion or answer.
        return state_diff, None
    # The released step loop selects any nonempty detector action. The ordinary
    # action executor, not the detector's name, decides whether it terminates.
    return state_diff, dict(thought=block.thought, action=block.action, confidence=block.confidence)


def screenshot(obs, path):
    from PIL import Image
    buffer = io.BytesIO()
    Image.fromarray(obs['screenshot']).convert('RGB').save(buffer, format='JPEG', quality=80)
    path.write_bytes(buffer.getvalue())
    return base64.b64encode(buffer.getvalue()).decode()


def prepare_browser():
    runtime = prepare_browser_runtime()
    # BrowserGym's chat window starts its own Chromium without launch kwargs.
    # Point both launches at the same existing executable; no download occurs.
    cache = (runtime / 'playwright' if os.environ.get('WEBPRM_BROWSER_TMP_ROOT')
             else Path('/tmp/openwebrl-webprm-browser-cache'))
    directory = cache / 'chromium-1117/chrome-linux'
    # Validate before mkdir: even an otherwise correct final executable link
    # cannot make a parent directory symlink safe to write through.
    if (directory.resolve() != directory or any(parent.is_symlink()
            for parent in (directory, *directory.parents))):
        raise ValueError('Browser executable cache parents must not be symlinks')
    directory.mkdir(parents=True, exist_ok=True)
    link = directory / 'chrome'
    try:
        link.symlink_to(CHROMIUM)
    except FileExistsError:
        if link.resolve() != Path(CHROMIUM).resolve():
            raise ValueError('Browser cache executable mismatch')
    os.environ['PLAYWRIGHT_BROWSERS_PATH'] = str(cache)
    os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
    os.environ['MESA_LOADER_DRIVER_OVERRIDE'] = 'llvmpipe'


def run_episode(task, arm, config):
    return run_supervised_episode(task, arm, config, _run_episode)


def _run_episode(task, arm, config):
    prepare_browser()
    from browsergym.core.env import BrowserEnv
    from browsergym.core.action.highlevel import HighLevelActionSet
    from browsergym.utils.obs import flatten_axtree_to_str
    actions = HighLevelActionSet(subsets=['chat', 'tab', 'nav', 'bid', 'infeas'],
                                 multiaction=False, strict=False, demo_mode='default')
    family, mode = arm_mode(arm)
    folder = Path(config['output']) / task['task_id'] / arm
    folder.mkdir(parents=True, exist_ok=True)
    journal = folder / 'requests.jsonl'
    result = dict(task_id=task['task_id'], arm=arm, success=0, valid=False,
                  attempted=False, identity_sha256=config['identity_sha256'], started_at=time.time())
    if (folder / 'result.json').exists():
        previous=json.loads((folder / 'result.json').read_text())
        if previous.get('identity_sha256')!=config['identity_sha256']:
            raise ValueError('Episode resume identity changed')
        return previous
    env = None
    history, rollout = [], []
    try:
        load_api_credentials(config.get('env_file'))
        repo=Path(__file__).resolve().parents[1]
        for source,expected in config['code_sha256'].items():
            if hashlib.sha256((repo/source).read_bytes()).hexdigest()!=expected:
                raise ValueError('Frozen code changed after controller preflight')
        if time.time() >= config['deadline']:
            raise TimeoutError('Pilot deadline reached')
        native_bounds = {}
        def guarded_mapping(action):
            if action == NO_ADMISSIBLE_ACTION:
                raise ValueError('No admissible native action in fixed actor samples; no action executed')
            if family == 'arbiter':
                from openwebrl.webprm_arbiter_harness import load_arbiter_harness,native_execution_code
                native = load_arbiter_harness(config['arbiter_source'])
                return native_execution_code(config['arbiter_source'],native.parse_action(action),
                                             native_bounds,key_mapping=native.key_mapping)
            return actions.to_python_code(safe_action(action,actions))
        result['attempted'] = True
        env = BrowserEnv(task_entrypoint=live_task_type(), task_kwargs=dict(start_url=task['start_url'], goal=task['intent'],
                         deadline=config['deadline'], navigation_ms=config.get('navigation_ms', 30000)),
                         action_mapping=guarded_mapping, headless=True, slow_mo=0, pw_chromium_kwargs={'executable_path':config.get('chromium_executable', CHROMIUM)})
        mark_episode_phase(config, 'browser_reset')
        obs, reset_info = env.reset(seed=config['seed'])
        mark_episode_phase(config, 'episode_setup')
        write_json(folder / 'browser-start.json', dict(info=reset_info,
                   navigation_wait_until='domcontentloaded', navigation_ms=config.get('navigation_ms', 30000)))
        shepherd = load_shepherd(config['shepherd_source']) if family == 'shepherd' and mode == 'guided' else None
        arbiter = load_arbiter(config['arbiter_source']) if family == 'arbiter' and mode == 'guided' else None
        shepherd_actor = None
        if family == 'shepherd':
            from openwebrl.webprm_shepherd_harness import load_shepherd_actor
            shepherd_actor = load_shepherd_actor(config['shepherd_actor_source'], actions,expected_sha256=config['source_manifests']['shepherd_actor_sha256'])
        harness = None
        if family == 'arbiter':
            from openwebrl.webprm_arbiter_harness import load_arbiter_harness, to_browsergym_actions
            harness = load_arbiter_harness(config['arbiter_source'])
            from openwebrl.webprm_arbiter_harness import load_native_observation_processor
            native_obs = load_native_observation_processor(config['arbiter_source'],viewport_size=env.task.viewport)
        renderer = load_arbiter_causal_renderer(config['arbiter_benchmark']) if arbiter else None
        provenance = dict(actor_model=ACTOR, judge_model=JUDGE,
                          judge_source_sha256=hashlib.sha256(JUDGE_SOURCE.read_bytes()).hexdigest(),
                          identity_sha256=config['identity_sha256'])
        if shepherd_actor:
            provenance['actor_prompt_origin'] = 'native_author_WebShepherd_BrowserGym_demo'
            provenance['shepherd_actor'] = shepherd_actor.manifest
        if shepherd:
            if shepherd.manifest!=config['source_manifests']['shepherd']:
                raise ValueError('Shepherd source changed after preflight')
            provenance['shepherd'] = shepherd.manifest
        if harness:
            if harness.manifest!=config['source_manifests']['arbiter_harness'] or native_obs.manifest!=config['source_manifests']['arbiter_observation']:
                raise ValueError('Arbiter native harness source changed after preflight')
            provenance['actor_prompt_origin'] = 'native_Arbiter_viewer_policy_observation_execution_BrowserGym_lifecycle'
            provenance['arbiter_observation'] = native_obs.manifest
            provenance['arbiter_harness'] = harness.manifest
        if arbiter:
            if arbiter.manifest!=config['source_manifests']['arbiter'] or renderer.manifest!=config['source_manifests']['arbiter_prompt']:
                raise ValueError('Arbiter reward source changed after preflight')
            provenance['arbiter'] = arbiter.manifest
            provenance['arbiter_prompt'] = renderer.manifest
        write_json(folder / 'provenance.json', provenance)
        checklist = None
        terminal = 'turn_limit'
        previous_tree = previous_terminal = None
        for turn in range(30):
            if time.time() >= config['deadline']:
                raise TimeoutError('Pilot deadline reached')
            mark_episode_phase(config, 'native_observation' if harness else 'observation_processing', turn=turn)
            if harness:
                tree, native_bounds = native_obs.observe(env.page)
            else:
                tree = shepherd_actor.preprocess(obs)['axtree_txt']
            url = obs['url']
            screenshot(obs, folder / f'step_{turn:02d}.jpg')
            context = dict(intent=task['intent'], trajectory='\n'.join(
                f"Thought {i+1}: {h['thought']}\nAction {i+1}: {h['action']}" +
                (f"\nExecution error: {h['action_error']}" if h.get('action_error') else '') for i,h in enumerate(history)),
                current_url=url, text_observation=tree, start_url=task['start_url'])
            terminal_candidate = None
            state_diff = ''
            if shepherd_actor:
                messages = shepherd_actor.messages(shepherd_actor.preprocess(obs),
                    history=[h['raw_response'] for h in history], model=ACTOR)
            if harness:
                state_diff, previous_terminal = arbiter_auxiliary(harness, config, journal, context,
                    previous_tree, previous_terminal, obs.get('last_action_error'))
                if previous_terminal and previous_terminal['action']:
                    terminal_candidate = dict(thought=previous_terminal['thought'],
                                              action=previous_terminal['action'])
                messages = [dict(role='system',content=harness.policy_system_prompt),
                    dict(role='user',content=harness.render_policy_prompt(
                        objective=task['intent'],observation=tree,url=url,
                        trajectory=context['trajectory'],state_diff=state_diff,domain_knowledge=None))]
            samples, rejected = [], []
            if terminal_candidate:
                samples = [terminal_candidate]
            else:
                raw = completion(config, dict(model=ACTOR, messages=messages, temperature=1., top_p=.9 if harness else 1.,
                                 n=1 if harness or mode == 'unguided' else 20, max_tokens=2048 if harness else 512,
                                 **({'response_format':{'type':'json_object'}} if harness else {})), journal, purpose='policy')
                if shepherd_actor:
                    from openwebrl.webprm_shepherd_harness import classify_shepherd_choices
                    samples, rejected = classify_shepherd_choices(raw['choices'], actions)
                else:
                    choice = raw['choices'][0]
                    if len(raw['choices']) != 1 or choice.get('finish_reason') != 'stop':
                        raise ValueError('Incomplete Arbiter policy response')
                    native = harness.parse_policy_response(choice['message']['content'])
                    for block in native.candidates:
                        samples.append(dict(thought=block.thought,action=block.action))
            write_json(folder / f'candidates_{turn:02d}.json', dict(accepted=samples, rejected=rejected,
                policy='native_per_sample_filter_no_resampling', native_terminal=previous_terminal))
            pool = ([dict(action=sample['action'],first_sample=i,frequency=1) for i,sample in enumerate(samples)]
                    if harness else candidate_pool([sample['action'] for sample in samples]) if samples else [])
            for candidate in pool:
                candidate['thought'] = samples[candidate['first_sample']]['thought']
                candidate['raw_response'] = samples[candidate['first_sample']].get('raw_response')
            trace = []
            if not pool:
                # Like native BrowserGym, a malformed action consumes a turn
                # and returns an execution error; it does not kill the episode.
                selected = dict(action=NO_ADMISSIBLE_ACTION, thought='',
                    raw_response='No admissible action. '+str([r['error'] for r in rejected]))
            elif terminal_candidate or mode == 'unguided':
                selected = pool[0]
            elif mode == 'random':
                selected = random.Random(config['seed']+turn).choice(pool)
            elif shepherd:
                if checklist is None:
                    msg = shepherd.get_messages(context, 'checklist_generation', 'web_shepherd')
                    response = completion(config, dict(model=config['shepherd_model'], messages=msg, temperature=0., max_tokens=1024), journal, local=True, endpoint=config['shepherd_endpoint'])
                    text = response['choices'][0]['message']['content']
                    checklist = text.split('<answer>')[-1].split('</answer>')[0].strip() if '<answer>' in text else text.split('[CHECKLISTS]')[-1].strip()
                for candidate in pool:
                    critic_context = dict(context,trajectory='\n'.join(f"Thought {i+1}: {h['thought']}\nAction {i+1}: {h['action']}" for i,h in enumerate(history[-10:])))
                    msg = shepherd.get_messages(dict(critic_context, checklist=checklist, thought=candidate['thought'], action=candidate['action']),
                        'judge_progress', 'web_shepherd', text_obs=True, image_obs=False)
                    response = completion(config, dict(model=config['shepherd_model'], messages=msg, temperature=0., max_tokens=2048, logprobs=True, top_logprobs=10), journal, local=True, endpoint=config['shepherd_endpoint'])
                    tokens = response['choices'][0].get('logprobs', {}).get('content')
                    if not tokens or any('top_logprobs' not in t for t in tokens):
                        raise ValueError('Shepherd label probabilities unavailable')
                    text = response['choices'][0]['message']['content']
                    if '<answer>' not in text and 'CHECKLIST EVALUATION:' not in text:
                        raise ValueError('Shepherd answer labels unavailable')
                    probs = shepherd.get_judge_probs(tokens)
                    if not probs or any(sum(p.values()) == 0 for p in probs):
                        raise ValueError('Shepherd probability parser fallback rejected')
                    candidate['score'] = shepherd.get_score([probs], 'prob')[0]
                    trace.append(dict(candidate=candidate.copy(), probabilities=probs))
                selected = max(pool, key=lambda c: (c['score'], c['frequency'], -c['first_sample']))
            else:
                def compare(left, right):
                    prompt = renderer.render(**context, thought1=left['thought'], action1=left['action'], thought2=right['thought'], action2=right['action'])
                    response = completion(config, dict(model=config['arbiter_model'], messages=[dict(role='user',content=prompt)], temperature=0., max_tokens=2048), journal, local=True, endpoint=config['arbiter_endpoint'])
                    return arbiter.parse_answer(response['choices'][0]['message']['content'])
                selected, trace = native_tournament(pool, compare)
            record = dict(turn=turn, context=context, raw_samples=samples, candidates=pool,
                          selected=selected, selection_trace=trace, checklist=checklist,
                          rejected_samples=rejected, state_diff=state_diff, native_terminal=previous_terminal)
            rollout.append(record)
            write_json(folder / 'rollout.json', rollout)
            history.append(dict(action=selected['action'], thought=selected['thought'],raw_response=selected.get('raw_response')))
            dispatched = [selected['action']]
            if isinstance(dispatched,str):
                dispatched=[dispatched]
            record['dispatched_actions'] = dispatched
            for dispatch in dispatched:
                mark_episode_phase(config, 'browser_step', turn=turn)
                obs, reward, terminated, truncated, info = env.step(dispatch)
                if terminated or truncated:
                    break
            record['action_error'] = obs.get('last_action_error')
            history[-1]['action_error'] = record['action_error']
            previous_tree = tree
            write_json(folder / 'rollout.json', rollout)
            if terminated or truncated or any(action.startswith('send_msg_to_user') or action.startswith('report_infeasible(') or action.startswith('stop') for action in dispatched):
                terminal = 'agent_stopped'
                break
        mark_episode_phase(config, 'terminal_processing')
        final = screenshot(obs, folder / 'terminal.jpg')
        messages = judge_messages(task, history, final, terminal)
        trajectory = '\n\n'.join(f"Thought {i}: {h['thought']}\nAction {i}: {h['action']}" for i,h in enumerate(history,1))
        trajectory += f'\n\nThought {len(history)+1}: \nAction {len(history)+1}: Agent stopped: {terminal} (this is not a success verdict)'
        messages[1]['content'][0]['text'] = judge_protocol()['USER_PROMPT'].format(task=task['intent'], thoughts_and_actions=trajectory)
        judged = completion(config, dict(model=JUDGE, messages=messages, max_completion_tokens=4096, seed=42), journal)
        if judged['choices'][0].get('finish_reason') != 'stop':
            raise ValueError('Incomplete terminal judge response')
        text = judged['choices'][0]['message']['content'] or ''
        verdict = parse_verdict(text)
        write_json(folder / 'verdict.json', dict(model=judged['model'], raw=text, parsed=verdict, terminal=terminal))
        if verdict is None:
            raise ValueError('Unparseable terminal judge verdict')
        result.update(success=int(verdict), valid=True, turns=len(history), terminal=terminal)
    except Exception as exc:
        result.update(error_type=type(exc).__name__, error=str(exc), turns=len(history))
    finally:
        save_outcome_and_close(result, folder, env, config)
    return result


def fixture(output,*,shepherd_actor_source='/tmp/openwebrl-webshepherd-demo.py',arbiter_source='/tmp/openwebrl-webarbiter-upstream'):
    prepare_browser()
    """Execute real BrowserGym data-page actions with no provider calls."""
    from browsergym.core.env import BrowserEnv
    from browsergym.core.task import OpenEndedTask
    from browsergym.core.action.highlevel import HighLevelActionSet
    from browsergym.utils.obs import flatten_axtree_to_str
    actions = HighLevelActionSet(multiaction=False, strict=True)
    from urllib.parse import quote
    html = "<label>Name<input id='name'></label><button onclick=\"document.body.dataset.done=document.getElementById('name').value\">Save</button>"
    url = 'data:text/html,' + quote(html)
    env = BrowserEnv(task_entrypoint=OpenEndedTask, task_kwargs=dict(start_url=url,goal='Enter Ada and save'),
                     action_mapping=actions.to_python_code, slow_mo=0, headless=True, pw_chromium_kwargs={'executable_path':CHROMIUM})
    try:
        obs,_ = env.reset(seed=42)
        tree = flatten_axtree_to_str(obs['axtree_object'])
        import re
        input_bid = re.search(r'\[([^\]]+)\] textbox',tree).group(1)
        button_bid = re.search(r'\[([^\]]+)\] button',tree).group(1)
        for action in [f"fill('{input_bid}', 'Ada')",f"click('{button_bid}')"]:
            obs,*_ = env.step(safe_action(action, actions))
            if obs.get('last_action_error'):
                raise RuntimeError(obs['last_action_error'])
        from openwebrl.webprm_arbiter_harness import load_arbiter_harness, to_browsergym_actions
        native = load_arbiter_harness(arbiter_source)
        from openwebrl.webprm_arbiter_harness import load_native_observation_processor,native_execution_code
        processor=load_native_observation_processor(arbiter_source,viewport_size=env.task.viewport)
        native_tree,bounds=processor.observe(env.page)
        native_input=re.search(r'\[([^\]]+)\] textbox',native_tree).group(1)
        native_button=re.search(r'\[([^\]]+)\] button',native_tree).group(1)
        env.action_mapping=lambda action:native_execution_code(arbiter_source,native.parse_action(action),bounds,key_mapping=native.key_mapping)
        dispatched=[f'type [{native_input}] Ada',f'click [{native_button}]']
        for action in dispatched:
            obs,*_=env.step(action)
            if obs.get('last_action_error'):
                raise RuntimeError(obs['last_action_error'])
        if env.page.evaluate('document.body.dataset.done') != 'Ada':
            raise AssertionError('BrowserGym action execution failed')
        original_url=env.page.url
        obs,*_=env.step('new_tab')
        if obs.get('last_action_error') or len(env.page.context.pages)!=2:
            raise AssertionError('Native new_tab failed')
        obs,*_=env.step('tab_focus [0]')
        if obs.get('last_action_error') or env.page.url!=original_url:
            raise AssertionError('Native tab_focus failed')
        obs,*_=env.step('tab_focus [1]')
        obs,*_=env.step('close_tab')
        if obs.get('last_action_error') or env.page.url!=original_url or len(env.page.context.pages)!=1:
            raise AssertionError('Native close_tab lifecycle bridge failed')
        obs,*_=env.step('send_msg_to_user [Ada saved]')
        if obs.get('last_action_error') or not any(message.get('role')=='assistant' and 'Ada saved' in message.get('message','') for message in obs['chat_messages']):
            raise AssertionError('Native send_msg_to_user did not reach terminal chat')
        target=Path(output);target.mkdir(parents=True,exist_ok=True)
        screenshot(obs,target/'fixture.jpg')
        from openwebrl.webprm_shepherd_harness import load_shepherd_actor
        actor=load_shepherd_actor(shepherd_actor_source,actions)
        messages=actor.messages(actor.preprocess(obs),model=ACTOR)
        write_json(target/'fixture.json',dict(passed=True,observation=tree,provider_calls=0,
                   native_arbiter_tab_stop_passed=True,native_arbiter_type_actions=dispatched,native_arbiter_observation=native_tree,
                   native_arbiter_bounds=bounds,native_arbiter_source=processor.manifest, native_shepherd_prompt=messages))
    finally:
        env.close()
