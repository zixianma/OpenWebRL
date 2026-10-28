"""Table 5-style selector comparison on live Online-Mind2Web; nothing starts at import.

Same viewer actor/candidate/selector protocol as the WebArena Table 5 worker
(webprm_webarena_worker: two actor snapshots, NoSearch one-BLOCK envelope, five
guided slots, Shepherd/Arbiter/GPT-4o-mini/Piotr selectors, released 30-turn N/A
fallback), executed on live public websites. There is no site hosting, reset or
programmatic evaluator: every terminal state is scored once by the established
OM2W/AgentTrek terminal-success judge (o4-mini, 4,096 completion tokens). Initial
navigation failures and other errors are recorded invalid attempts, never replaced.
"""
from __future__ import annotations

import base64
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from types import MappingProxyType

from .jev_eval import JUDGE_SOURCE, judge_protocol, parse_verdict
from .webprm_webarena import EvaluationNotReady, digest

PROTOCOL = 'om2w_table5_viewer_reconstruction_v3'  # v2: desktop UA + title retry; v3: HTTP>=400 start pages proceed
CAMPAIGN_PROTOCOL = 'om2w_table5_live_queue_v1'
JUDGE = 'o4-mini-2025-04-16'
JUDGE_PURPOSE = 'om2w_judge'
JUDGE_MAX_COMPLETION_TOKENS = 4096
JUDGE_PRICES = (1.10e-6, 4.40e-6)
LIVE_RESET_ID = 'om2w_live_public_website_no_reset'
TASK_FILE = 'openwebrl/data/eval/online-mind2web.jsonl'
TASK_FILE_SHA256 = '8343c23be98d6d63856e9b53ff3884222be099cc0f55ad5edb475176f54317ed'
COHORTS_FILE = 'openwebrl/docs/arm_c2_full300_cohorts.json'
COHORTS_FILE_SHA256 = None  # checked against the hash recorded at package build
SEED = 42
MAX_TURNS = 30
DESKTOP_USER_AGENT = ('Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/134.0.0.0 Safari/537.36')
NAVIGATION_MS = 30000
# Playwright's page.screenshot waits for web fonts (document.fonts.ready). On ad-heavy live pages the
# fonts never settle: fixtures 358449/358468 timed out at 5 s and 30 s (terminal judge and Piotr
# pre-action screenshots), and a login-node check of flightaware.com still had fonts 'loading' after
# 10 s. The viewport is therefore captured with CDP Page.captureScreenshot, as BrowserGym already does
# for observations; it does not wait for fonts (same check: 40 s on that page, then succeeded).
SCREENSHOT_METHOD = 'cdp_page_capture_screenshot_viewport'


def cdp_viewport_screenshot(page, *, type='png', path=None, quality=None):
    """Viewport PNG/JPEG bytes via CDP, without Playwright's web-font wait."""
    params = dict(format='jpeg' if type == 'jpeg' else 'png')
    if quality is not None:
        params['quality'] = quality
    session = page.context.new_cdp_session(page)
    try:
        data = base64.b64decode(session.send('Page.captureScreenshot', params)['data'])
    finally:
        session.detach()
    if path is not None:
        Path(path).write_bytes(data)
    return data


def use_cdp_screenshots():
    """In this worker process, route viewport page.screenshot calls through CDP capture."""
    from playwright.sync_api import Page
    original = Page.screenshot
    if getattr(original, '_webprm_cdp', False):
        return

    def screenshot(self, *, type='png', path=None, quality=None, **kwargs):
        if kwargs.get('full_page') or kwargs.get('clip') is not None or kwargs.get('mask'):
            return original(self, type=type, path=path, quality=quality, **kwargs)
        return cdp_viewport_screenshot(self, type=type, path=path, quality=quality)
    screenshot._webprm_cdp = True
    Page.screenshot = screenshot
VIEWPORT = MappingProxyType({'width': 1280, 'height': 720})
# Prespecified, disjoint stages: the first ten tasks of the repeated 100-task
# stratum, its remaining 90, then the other 200. Each stage is its own approved
# package (own cap and claim store); the union is the full 300-task cohort.
STAGES = MappingProxyType({'pilot10': (0, 10), 'next90': (10, 100), 'rest200': (100, 300)})
CANDIDATE_CONTROLS = 'om2w_random_candidate_control_8x6_v1'
CONTROL_SELECTORS = ('no_search', 'random5', 'gpt4omini', 'shepherd8b', 'arbiter7b', 'piotr_selectionarm')
FOCUSED_CONTROL_SELECTORS = ('no_search', 'random5', 'arbiter7b', 'piotr_selectionarm')


def candidate_control_spec(task_count=8, *, focused=False):
    """Prospective development sample, selected by fixed order without reading outcomes."""
    if type(task_count) is not int:
        raise ValueError('Fixed study size must be an integer')
    if type(focused) is not bool or (focused and task_count != 100):
        raise ValueError('Focused controls require exactly100 fresh tasks')
    if focused:
        spec = candidate_control_spec(100)
        spec.update(identity='om2w_random_candidate_control_100x4_v1',
            selectors=list(FOCUSED_CONTROL_SELECTORS),
            scope='400 one-attempt paired episodes; four approved methods, no first5; separate from prior cohorts')
        return spec
    if task_count == 100:
        spec = candidate_control_spec()
        spec.update(identity='om2w_random_candidate_control_100x6_v1',
            task_indices=list(range(108, 208)),
            selection='next100 tasks of the pinned holdout after the disjoint eight-task diagnostic; no outcome selection',
            scope='600 one-attempt paired episodes; first5 omitted; separate from prior diagnostic, next90 and paper reproduction')
        return spec
    if task_count != 8:
        raise ValueError('Only the fixed eight-task diagnostic or fresh100-task comparison is defined')
    return dict(identity=CANDIDATE_CONTROLS, actor='gpt-4o-2024-05-13',
        task_indices=list(range(100, 108)), selectors=list(CONTROL_SELECTORS),
        selection='first eight tasks of the pinned 200-task holdout, without outcome selection',
        comparisons=['guided versus random5: selection policy given the same generation protocol',
                     'random5 versus no_search: combined candidate-generation and random-selection policy; generation alone is not isolated'],
        candidate_coupling='fresh candidates at each actual state; divergent trajectories are not matched-state counterfactuals',
        scope='48 one-attempt development episodes; first5 omitted at user request; separate from primary cohorts and paper reproduction',
        unchanged='native actor/helper/observation/30-turn/judge protocols; no filter/history intervention',
        random_selection='existing deterministic seed42+turn draw over surviving candidates; no extra inference')


def candidate_control_ids(cohort, task_count=8, *, focused=False):
    spec = candidate_control_spec(task_count, focused=focused)
    return [c['cell_id'] for c in cohort['cells'] if c['task_index'] in spec['task_indices']]


def candidate_control_envelope(task_count=8, *, focused=False):
    candidate_control_spec(task_count, focused=focused)  # Reject arbitrary size/membership variants.
    attempts = 1 if task_count == 8 else 2
    paid = task_count * (4*91 if focused else 5*91 + 211)
    # Keep the existing11-call local startup validation; no paid call is added.
    local = task_count * 30 * ((4+1) if focused else (6+4+1)) + 11*attempts
    return dict(resources=dict(gpus=1, gpu_type='H200', cpus=8, memory_gib=240,
                               total_seconds=(8 if task_count == 8 else 48)*3600),
                limits=dict(paid_requests=paid, local_requests=local, total_requests=paid+local,
                            max_workers=4, maximum_attempts=attempts))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _copy(value):
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


@dataclass(frozen=True)
class OM2WTask:
    """TaskEnvelope-compatible view; only intent/start URL reach the actor."""
    execution_id: str
    released_task_id: str
    raw_config_sha256: str
    configured_config_sha256: str
    actor_view: MappingProxyType
    requires_storage_state: bool
    _record_json: str = field(repr=False)

    @classmethod
    def from_record(cls, record):
        raw = _copy(record)
        meta = raw.get('metadata') or {}
        task_id, intent, start = meta.get('task_id'), meta.get('intent'), meta.get('start_url')
        if not isinstance(task_id, str) or not task_id or any(c in task_id for c in '/\\\0') or task_id in ('.', '..'):
            raise ValueError('OM2W task requires a path-safe string task_id')
        if not isinstance(intent, str) or not intent.strip() or not isinstance(start, str) or not start.startswith(('http://', 'https://')):
            raise ValueError('OM2W task requires an intent and an absolute start URL')
        if meta.get('storage_state') or meta.get('require_login'):
            raise ValueError('Authenticated OM2W tasks are not part of this protocol')
        view = MappingProxyType({'intent': intent, 'start_url': start, 'sites': ('online_mind2web',)})
        return cls('om2w_' + task_id, task_id, digest(raw), digest(raw), view, False,
                   json.dumps(raw, ensure_ascii=False, allow_nan=False))

    def native_config(self):
        """Evaluator-only descriptor; no reference answer exists or reaches the actor."""
        return dict(task_id=self.released_task_id, storage_state=None,
                    eval=dict(eval_types=['om2w_terminal_judge'], reference_answers=None, program_html=[]))

    def record(self):
        return json.loads(self._record_json)


def load_tasks(path):
    if sha(path) != TASK_FILE_SHA256:
        raise ValueError('Pinned Online-Mind2Web 300-task file changed')
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    tasks = {row['metadata']['task_id']: row for row in rows}
    if len(rows) != 300 or len(tasks) != 300:
        raise ValueError('Expected 300 unique Online-Mind2Web tasks')
    return tasks


def task_order(task_path, cohorts_path):
    """Repeated 100-task stratum (its frozen order) first, then the holdout 200."""
    tasks = load_tasks(task_path)
    cohorts = json.loads(Path(cohorts_path).read_text())
    if cohorts.get('task_file_sha256') != TASK_FILE_SHA256:
        raise ValueError('Frozen OM2W strata refer to a different task file')
    first = cohorts['cohorts']['checkpoint_selection_100']['task_ids']
    rest = cohorts['cohorts']['checkpoint_selection_holdout_200']['task_ids']
    order = list(first) + list(rest)
    if len(first) != 100 or len(rest) != 200 or set(order) != set(tasks) or len(set(order)) != 300:
        raise ValueError('Frozen 100/200 strata must partition the 300 tasks')
    return order, tasks


def cohort(task_path, cohorts_path, actors, selectors):
    """Fixed task-major cell order; condition order within a task is a per-task seeded shuffle.

    Task-major order keeps the ten conditions of one task contemporaneous on the
    same live site. The shuffle prevents a fixed condition always running first.
    """
    order, tasks = task_order(task_path, cohorts_path)
    conditions = [(a, s) for a in actors for s in selectors]
    cells = []
    for index, task_id in enumerate(order):
        stage = next(name for name, (lo, hi) in STAGES.items() if lo <= index < hi)
        shuffled = list(conditions)
        random.Random(f'{SEED}:{task_id}').shuffle(shuffled)
        for actor, selector in shuffled:
            cells.append(dict(cell_id=f'om2w_{task_id}/{actor}__{selector}', domain='om2w-' + stage,
                              stage=stage, task_index=index, task_id=task_id, actor=actor, selector=selector,
                              task_config_sha256=digest(tasks[task_id])))
    return dict(schema_version=1, protocol=CAMPAIGN_PROTOCOL, task_file_sha256=TASK_FILE_SHA256,
                actors=list(actors), selectors=list(selectors), stages={k: list(v) for k, v in STAGES.items()},
                task_order=order, cells=cells, cell_count=len(cells))


def protocol_contract(arbiter_source):
    """Same viewer actor/selector contract as Table 5 v4 (scoring section replaced)."""
    from .webprm_webarena_worker import ENVELOPE_NO_SEARCH_PROMPT, NA_TURN_LIMIT, protocol_definition
    from .webprm_webarena_piotr import REPRESENTATION_V2
    base = protocol_definition(arbiter_source, turn_limit=NA_TURN_LIMIT, piotr_representation=REPRESENTATION_V2,
                               no_search_prompt=ENVELOPE_NO_SEARCH_PROMPT)
    prompts = judge_protocol()
    base.update(identity=PROTOCOL, viewer_actor_selector_identity=base['identity'],
        benchmark='online_mind2web_live_300', task_file_sha256=TASK_FILE_SHA256, seed=SEED, max_turns=MAX_TURNS,
        site='live_public_websites_no_hosting_no_reset',
        screenshot_method=SCREENSHOT_METHOD,
        browsergym_observation='raw_page_metadata; actor uses pinned native observer separately',
        browser_repairs_v2=dict(user_agent=DESKTOP_USER_AGENT, user_agent_reason='probe 359515: HeadlessChrome UA 25/84 vs desktop 8/84 initial HTTP 403',
                                page_title='retry through navigation (5 tries, domcontentloaded wait), else empty metadata title; pilot 359462 v1 invalidated 7/49'),
        initial_navigation=dict(wait_until='domcontentloaded', timeout_ms=NAVIGATION_MS,
                                http_status_at_least_400='proceed_recorded_judge_scores_outcome (v3, as OpenWebRL harness)',
                                navigation_exception='invalid_attempt_never_replaced'),
        viewport=dict(VIEWPORT), observation='pinned_native_accessibility_tree_current_viewport_with_geometry_restoration',
        evaluator=dict(model=JUDGE, purpose=JUDGE_PURPOSE, prompt_source='openwebrl/eval/reward_online_mind2web.py',
            prompt_sha256=digest(prompts), max_completion_tokens=JUDGE_MAX_COMPLETION_TOKENS,
            image='final_viewport_jpeg_detail_high', parse='reward_online_mind2web._parse_verdict_identical',
            trajectory='selected_thought_and_action_per_executed_turn_plus_agent_stopped_line',
            seed='omitted_by_broker_payload_allowlist', single_call_no_retry=True,
            incomplete_or_unparseable='invalid_attempt'))
    return base


def worker_template(table5_template, arbiter_source, code_sha256, env_file):
    """Derive the OM2W cell template from a frozen Table 5 template (no WebArena evaluator fields)."""
    keep = ('actors', 'selectors', 'arbiter_source', 'shepherd_source', 'piotr_checkpoint', 'piotr_manifest',
            'piotr_model', 'piotr_revision', 'piotr_source', 'piotr_representation', 'source_manifest',
            'source_manifests', 'no_search_prompt_version', 'turn_limit_protocol', 'paper_exact_protocol',
            'prices_usd_per_token', 'price_provenance')
    template = {key: deepcopy(table5_template[key]) for key in keep}
    template['prices_usd_per_token'][JUDGE] = list(JUDGE_PRICES)
    template['price_provenance'][JUDGE] = dict(source='openwebrl/webprm_worker.py PRICES (OM2W pilot ledger)',
                                                usd_per_token=list(JUDGE_PRICES))
    contract = protocol_contract(arbiter_source)
    template.update(schema_version=1, protocol=PROTOCOL, protocol_contract=contract, seed=SEED, max_turns=MAX_TURNS,
        authorization_status='pending', api_usd_cap=0, deadline=0, job_id=None, broker_socket=None,
        budget_path=None, code_sha256=dict(code_sha256), env_file=env_file, navigation_ms=NAVIGATION_MS,
        judge_model=JUDGE, judge_max_completion_tokens=JUDGE_MAX_COMPLETION_TOKENS)
    return template


def routes(table5_routes, table5_cell_routes, actors, *, candidate_controls=False):
    """Table 5 actor/critic routes with the OM2W judge replacing the WebArena fuzzy judge."""
    out = [deepcopy(r) for r in table5_routes if r['purpose'] != 'native_fuzzy']
    out.append(dict(local=False, model=JUDGE, purpose=JUDGE_PURPOSE, max_input_tokens=120000,
                    max_output_tokens=JUDGE_MAX_COMPLETION_TOKENS))
    cells = {actor: {selector: [row for row in rows if row[1] != 'native_fuzzy'] + [[JUDGE, JUDGE_PURPOSE]]
                     for selector, rows in table5_cell_routes[actor].items()} for actor in actors}
    if candidate_controls:
        for actor in actors:
            for selector in ('random5',):
                cells[actor][selector] = deepcopy(cells[actor]['no_search'])
    return out, cells


def judge_trajectory(rollout, fallback, termination):
    """Thought/action text for the judge from actually executed (selected) turns only."""
    rows = []
    for record in rollout:
        index = record.get('selected_index')
        if index is None:
            rows.append(('', '[no action executed] ' + str(record.get('execution_error') or '')))
            continue
        candidate = record['proposal']['candidates'][index]
        rows.append((candidate.get('thought') or '', candidate['action']))
    if fallback and fallback.get('status') == 'executed':
        rows.append(('', fallback['action']))
    text = '\n\n'.join(f'Thought {i}: {t}\nAction {i}: {a}' for i, (t, a) in enumerate(rows, 1))
    stop = len(rows) + 1
    return text + f'\n\nThought {stop}: \nAction {stop}: Agent stopped: {termination} (this is not a success verdict)'


def judge_messages(intent, trajectory, jpeg):
    prompts = judge_protocol()
    return [dict(role='system', content=prompts['SYSTEM_PROMPT']),
            dict(role='user', content=[
                dict(type='text', text=prompts['USER_PROMPT'].format(task=intent, thoughts_and_actions=trajectory)),
                dict(type='image_url', image_url=dict(url='data:image/jpeg;base64,' + base64.b64encode(jpeg).decode(),
                                                      detail='high'))])]


def om2w_evaluator(task, config, request, folder):
    """evaluate(native_config, entries, page, termination) -> one judged verdict (no retry)."""
    folder = Path(folder)

    def evaluate(native_config, entries, page, *, termination):
        from .webprm_webarena_worker import write_json
        rollout = json.loads((folder / 'rollout.json').read_text()) if (folder / 'rollout.json').exists() else []
        fallback_path = folder / 'protocol-fallback.json'
        fallback = json.loads(fallback_path.read_text()) if fallback_path.exists() else None
        jpeg = page.screenshot(type='jpeg')
        with (folder / 'judge-final.jpg').open('xb') as stream:
            stream.write(jpeg); stream.flush(); os.fsync(stream.fileno())
        trajectory = judge_trajectory(rollout, fallback, termination)
        payload = dict(model=JUDGE, messages=judge_messages(task.actor_view['intent'], trajectory, jpeg),
                       max_completion_tokens=JUDGE_MAX_COMPLETION_TOKENS)
        response = request(config, payload, folder / 'requests.jsonl', purpose=JUDGE_PURPOSE)
        choices = response.get('choices') or []
        finish = choices[0].get('finish_reason') if len(choices) == 1 else None
        text = (choices[0].get('message') or {}).get('content') if len(choices) == 1 else None
        verdict = parse_verdict(text) if finish == 'stop' and isinstance(text, str) else None
        record = dict(model=response.get('model'), finish_reason=finish, raw=text, parsed=verdict,
                      termination=termination, trajectory_sha256=hashlib.sha256(trajectory.encode()).hexdigest(),
                      final_screenshot_sha256=hashlib.sha256(jpeg).hexdigest(), url=page.url,
                      usage=response.get('usage'))
        write_json(folder / 'om2w-judge.json', record)
        if response.get('model') != JUDGE or verdict is None:
            raise EvaluationNotReady('Incomplete or unparseable OM2W terminal judgment')
        return dict(valid=True, score=int(verdict), success=bool(verdict), judge=record)
    return evaluate


def validate_execution_config(config):
    from scripts.check_webprm_reddit_reset import safe_path
    from .webprm_webarena_worker import ACTOR, ACTOR_SNAPSHOTS, SELECTORS, validate_frozen_sources
    if config.get('authorization_status') != 'approved':
        raise EvaluationNotReady('An explicitly approved execution plan is required')
    if str(config.get('job_id')) != os.environ.get('SLURM_JOB_ID'):
        raise EvaluationNotReady('Execution must stay inside its approved existing allocation')
    cap, deadline = config.get('api_usd_cap'), config.get('deadline')
    if type(cap) not in (int, float) or not math.isfinite(cap) or cap <= 0:
        raise EvaluationNotReady('Explicit finite positive approved API cap required')
    if type(deadline) not in (int, float) or not math.isfinite(deadline) or deadline <= time.time():
        raise EvaluationNotReady('Explicit unexpired finite allocation deadline required')
    if config.get('max_turns') != MAX_TURNS or config.get('seed') != SEED:
        raise EvaluationNotReady('Common 30-turn protocol and seed 42 required')
    if config.get('actor_model') not in ACTOR_SNAPSHOTS or config.get('selector') not in SELECTORS:
        raise EvaluationNotReady('Unknown actor or selector')
    if config.get('protocol') != PROTOCOL or config.get('protocol_contract') != protocol_contract(config['arbiter_source']):
        raise EvaluationNotReady('Frozen OM2W viewer protocol contract differs')
    paid = {config['actor_model'], JUDGE} | ({ACTOR} if config['selector'] == 'gpt4omini' else set())
    for model in paid:
        rates = config.get('prices_usd_per_token', {}).get(model)
        if (not isinstance(rates, (list, tuple)) or len(rates) != 2 or
                any(type(r) not in (int, float) or not math.isfinite(r) or r <= 0 for r in rates)):
            raise EvaluationNotReady('Explicit finite positive model prices required')
    for name in ('output', 'budget_path', 'broker_socket'):
        if not config.get(name):
            raise EvaluationNotReady('Missing explicit runtime path: ' + name)
        safe_path(config[name])
    critic = {'arbiter7b': 'arbiter', 'shepherd8b': 'shepherd', 'piotr_selectionarm': 'piotr'}.get(config['selector'])
    if critic and any(not isinstance(config.get(critic + s), str) or not config[critic + s] for s in ('_model', '_endpoint')):
        raise EvaluationNotReady('Explicit frozen critic model and broker route required')
    if config['selector'] == 'piotr_selectionarm':
        from .webprm_webarena_piotr import validate_piotr_selectionarm
        if config.get('piotr_manifest') != validate_piotr_selectionarm(config):
            raise EvaluationNotReady('Frozen Piotr source/checkpoint manifest required')
    validate_frozen_sources(config)


TRANSIENT_NAVIGATION_ERRORS = ('Execution context was destroyed', 'Frame was detached',
                               'Frame has been detached', 'Target page, context or browser has been closed')


def _stable_title(page, tries=5):
    """Page.title() races live navigations (pilot 359462: 5/24 cells invalid). Titles are
    metadata only — the actor reads the pinned native observer — so wait, retry, then ''."""
    for left in reversed(range(tries)):
        try:
            return page.title()
        except Exception as error:
            if left == 0 or not any(s in str(error) for s in TRANSIENT_NAVIGATION_ERRORS):
                if any(s in str(error) for s in TRANSIENT_NAVIGATION_ERRORS):
                    return ''
                raise
            try:
                page.wait_for_load_state('domcontentloaded', timeout=3000)
            except Exception:
                pass
            time.sleep(0.5)


def navigation_tolerant_browser_env():
    """BrowserEnv whose raw-page observation survives navigation during title reads."""
    from browsergym.core.env import BrowserEnv

    class NavigationTolerantBrowserEnv(BrowserEnv):
        def _get_obs(self):
            if not self.use_raw_page_output:
                return super()._get_obs()
            import numpy as np
            pages = list(self.context.pages)
            return {
                'page': self.page,
                'chat_messages': tuple(deepcopy(self.chat.messages)),
                'goal': _legacy_goal(self.goal_object),
                'goal_object': tuple(deepcopy(self.goal_object)),
                'open_pages_urls': tuple(page.url for page in pages),
                'open_pages_titles': tuple(_stable_title(page) for page in pages),
                'active_page_index': np.asarray([pages.index(self.page)]),
                'url': self.page.url,
                'last_action': self.last_action,
                'last_action_error': self.last_action_error,
                'elapsed_time': np.asarray([time.time() - self.start_time]),
            }
    return NavigationTolerantBrowserEnv


def _legacy_goal(goal_object):
    from browsergym.core.env import _try_to_extract_legacy_goal
    return _try_to_extract_legacy_goal(goal_object)


def run_episode(task_record, selector, config, *, validate=True):
    from .webprm_lifecycle import run_supervised_episode
    from .webprm_webarena_worker import cell_name
    if validate:
        validate_execution_config(dict(config, selector=selector))
    task = OM2WTask.from_record(task_record)
    cell = cell_name(config['actor_model'], selector)
    return run_supervised_episode(dict(task_id=task.execution_id, record=task_record), cell,
                                  dict(config, selector=selector, cell_name=cell), _run_episode)


def _run_episode(task_record, cell, config):
    from .webprm_lifecycle import save_outcome_and_close, mark_episode_phase
    from .webprm_webarena_episode import EpisodeHooks, open_authenticated_browser
    from .webprm_webarena_worker import ViewerActor, ViewerSelector, broker_request, run_loop, write_json
    from .webprm_arbiter_harness import load_native_observation_processor, native_execution_code
    from .webprm_worker import CHROMIUM, prepare_browser
    task = OM2WTask.from_record(task_record['record'])
    folder = Path(config['output']) / task.execution_id / cell
    folder.mkdir(parents=True, exist_ok=True)
    env = None
    result = dict(task_id=task.execution_id, released_task_id=task.released_task_id, arm=cell, valid=False,
                  success=0, attempted=False, actor_model=config['actor_model'], selector=config['selector'],
                  identity_sha256=config['identity_sha256'])
    try:
        for path, expected in config['code_sha256'].items():
            if sha(path) != expected:
                raise EvaluationNotReady('Frozen worker code changed')
        request = config.get('_request_override') or broker_request(config, task, config['selector'])
        config = dict(config, reset_id=LIVE_RESET_ID)
        prepare_browser(); use_cdp_screenshots()
        result['attempted'] = True
        actor = ViewerActor(config, folder / 'requests.jsonl', request=request)
        selector = ViewerSelector(config['selector'], config, folder / 'requests.jsonl', request=request)
        bounds, dispatch_evidence = {}, {}

        def mapping(action):
            code = native_execution_code(config['arbiter_source'], actor.harness.parse_action(action),
                                         bounds, key_mapping=actor.harness.key_mapping)
            dispatch_evidence.update(action=action, code=code, code_sha256=hashlib.sha256(code.encode()).hexdigest(),
                                     element_bounds=deepcopy(bounds))
            return code
        from .webprm_runtime import live_task_type
        viewport = dict(VIEWPORT)
        mark_episode_phase(config, 'browser_construct')
        env = open_authenticated_browser(task,
            hooks=EpisodeHooks(dependency=lambda _: dict(ready=True, reset_id=LIVE_RESET_ID, authenticated=False)),
            browser_factory=navigation_tolerant_browser_env(), task_entrypoint=live_task_type(), action_mapping=mapping,
            task_options=dict(deadline=config['deadline'], navigation_ms=config.get('navigation_ms', NAVIGATION_MS),
                              reject_http_errors=False),
            # The viewer observer below owns the actor's AX tree and geometry.
            # BrowserGym's full DOM/AX/screenshot extraction is discarded and can
            # stall on live ad frames; retain its executor and raw chat/error metadata.
            browser_options=dict(headless=True, slow_mo=0, viewport=deepcopy(viewport), use_raw_page_output=True,
                pw_chromium_kwargs={'executable_path': config.get('chromium', CHROMIUM)},
                # Headless Chromium advertises "HeadlessChrome"; probe 359515 (first 100 cohort
                # start pages): 25/84 HTTP 403 with it vs 8/84 with a desktop Chrome UA.
                pw_context_kwargs={'user_agent': config.get('user_agent', DESKTOP_USER_AGENT)}))
        native_observation = load_native_observation_processor(config['arbiter_source'], viewport_size=viewport)

        def observe(environment, observation):
            text, observed = native_observation.observe(environment.page)
            bounds.clear(); bounds.update(observed)
            return dict(text=text, url=environment.page.url, last_action_error=observation.get('last_action_error'),
                        element_bounds=deepcopy(bounds), viewport=dict(environment.page.viewport_size))
        evaluate = om2w_evaluator(task, config, request, folder)
        write_json(folder / 'provenance.json', dict(actor=actor.harness.manifest, observation=native_observation.manifest,
            critic=selector.reward.manifest if selector.reward else selector.shepherd.manifest if selector.shepherd
                else selector.piotr.manifest if selector.piotr else None,
            critic_template='pinned_online_viewer' if selector.reward else config['selector'],
            reset_id=LIVE_RESET_ID, actor_model=config['actor_model'], judge_model=JUDGE,
            judge_source_sha256=sha(JUDGE_SOURCE), task_config_sha256=task.raw_config_sha256,
            paper_exact_protocol=False, identity_sha256=config['identity_sha256']))
        result = run_loop(task, config['selector'], config, env, actor, selector, observe, evaluate, folder,
                          dispatch_evidence=dispatch_evidence)
        env = None  # score_and_close owns normal cleanup.
    except Exception as error:
        if (folder / 'result.json').exists():
            result = json.loads((folder / 'result.json').read_text())
        result.update(error_type=type(error).__name__, error=str(error)[:4000])
    finally:
        if env is not None or result.get('cleanup_status') is None:
            save_outcome_and_close(result, folder, env, config)
    return result


def cell_outcome(result_path, cell, identity):
    """Durable assigned result plus closed watchdog; exit status alone proves nothing."""
    result = json.loads(Path(result_path).read_text())
    watchdog = json.loads((Path(result_path).parent / 'watchdog.json').read_text())
    if (result.get('identity_sha256') != identity or result.get('task_id') != 'om2w_' + cell['task_id']
            or result.get('arm') != cell['actor'] + '__' + cell['selector']
            or result.get('actor_model') != cell['actor'] or result.get('selector') != cell['selector']
            or type(result.get('attempted')) is not bool or type(result.get('valid')) is not bool
            or type(result.get('success')) is not int or result['success'] not in (0, 1)
            or (result['success'] and not result['valid']) or watchdog != result.get('watchdog')
            or watchdog.get('remaining_owned_processes') != {}):
        raise ValueError('Durable assigned result or worker closure is incomplete')
    return dict(cell_id=cell['cell_id'], result_path=str(result_path), result_sha256=sha(result_path),
                attempted=result['attempted'], valid=result['valid'], success=result['success'],
                turns=result.get('turns'), error_type=result.get('error_type'))
