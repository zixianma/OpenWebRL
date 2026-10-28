"""v12 (2026-10-28, user: "ah yes try C"): OpenWebRL-4B-SFT actor in its own screenshot + coordinate action space on
the hosted WebArena sites, scored by the released WebArena evaluator; no_search vs Piotr SelectionARM.

The agent loop is OpenWebRL's own (frozen snapshot of generate_browser.generate_turn_sample and the crossbench
selector classes used for the 2026-10-18..28 full-set selector evaluations). Only the plumbing differs:

* browser: the snapshot's WebEnv runs in the worker process (instead of an env_server subprocess) so the native
  evaluator can read the live terminal page; observations/infos are JSON round-tripped exactly as env_server does.
* model transport: actor /generate and ARM /select reach the host's loopback servers through job-scratch UNIX
  sockets (owrl_uds_relay.py), because the worker runs inside the site's network namespace.
* termination: an OpenWebRL ``done(response)`` becomes WebArena's terminal ``send_msg_to_user`` answer; any other
  agent-caused ending (step limit, format failure, context limit) gets the released WebArbiter max-step fallback
  answer "N/A" (as v11's native coordinator), then the evaluator reads the final page as usual. Infrastructure endings
  (browser/model/wall-clock) are invalid cells, never scored.
"""
from __future__ import annotations

import asyncio
from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

V12_ACTOR = 'openwebrl-sft-4b'
ACTOR_PATH = '/gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT'
ACTOR_REVISION = '15e777db2ddba2e0e82080ebccd3ad8d215b7f0a'
PIOTR_SELECTION = '/gpfs/scrubbed/zixianma/checkpoints/web/arm/selection/81b452d800d9f859687074f82680dd5257e02d89'
PIOTR_SGLANG_DIR = '/gpfs/scrubbed/zixianma/checkpoints/web/arm/selection-sglang-81b452d'
PLAYWRIGHT_BROWSERS = '/gpfs/scrubbed/zixianma/openwebrl-runtime/browsers'
CHROMIUM = PLAYWRIGHT_BROWSERS + '/chromium-1208/chrome-linux64/chrome'
VIEWPORT = dict(width=1280, height=1000)
NA_ANSWER = 'N/A'
SEED = 42
SAMPLING = dict(temperature=1.0, top_p=0.9, max_new_tokens=1024)
CANDIDATES = {'no_search': 1, 'piotr_selectionarm': 5}
SELECTOR_KIND = {'no_search': 'baseline', 'piotr_selectionarm': 'selection'}
MAX_STEPS = 30
TASK_TIMEOUT_SECONDS = 1800
EVALUATION_TIMEOUT_MS = 30000
ACTOR_URL = 'http://127.0.0.1:30000/generate'  # host part ignored by the UNIX-socket transport
SELECT_URL = 'http://127.0.0.1:30001'
PROTOCOL = dict(
    identity='native_table5_v12_owrl',
    user_decision='ah yes try C (2026-10-28): openwebrl-sft alone vs openwebrl-sft + Piotr SelectionARM in the OpenWebRL action space',
    actor=dict(name=V12_ACTOR, path=ACTOR_PATH, revision=ACTOR_REVISION,
               serving='sglang, one H200, mem-fraction .45, CUDA graphs max batch 48, --random-seed 4200, context 32768'),
    selector=dict(no_search='crossbench kind baseline: 1 sample per step',
                  piotr_selectionarm='crossbench kind selection: 5 samples per step, Piotr SelectionARM picks one '
                                     '(serve_arm_sglang /select, greedy, JSON schema, 128 tokens; first-candidate fallback)',
                  model=PIOTR_SELECTION, sglang_dir=PIOTR_SGLANG_DIR,
                  serving='sglang on the actor GPU, mem-fraction .30, behind serve_arm_sglang /select'),
    sampling=dict(SAMPLING, top_k='omitted, native generate endpoint default', seed=SEED,
                  candidate_seeds='crossbench candidate_seed(42, task_id, turn, j): candidate 0 shares its seed with no_search'),
    loop='openwebrl generate_turn_sample (frozen snapshot), browser_response_format_mode browser_env, full history, '
         '1 screenshot in context, 30 steps, max 3 consecutive parse failures, 1024 new tokens, 32768 context',
    viewport=dict(VIEWPORT, note='user-chosen 2026-10-28: OpenWebRL native 1280x1000 (env/config.yaml since 5ae7e95; all Piotr ARM '
                                 'training states are 1280x1000), not v11\'s WebArena 1280x720; coordinates 0..1000-normalized'),
    browser='OpenWebRL WebEnv in the worker process (Playwright 1.58 bundled chromium-1208, headless, as crossbench), '
            'storage state from the native site login',
    termination=dict(done='done(response) -> send_msg_to_user [response], termination stop',
                     agent_endings='step_limit/format_failure/context_limit -> released max-step fallback answer N/A',
                     invalid='browser_env/model_server/wall_timeout/contract/unknown endings: cell invalid, not scored'),
    evaluator='released WebArena evaluator (native loader, v11 abstention protocol), live terminal page, '
              f'page default timeout {EVALUATION_TIMEOUT_MS} ms during evaluation',
    fuzzy='retired gpt-4-1106-preview: cell invalid (as v11); offline gpt-4.1 re-score is a labelled sensitivity analysis',
    max_steps=MAX_STEPS)


def protocol_sha256():
    return hashlib.sha256(json.dumps(PROTOCOL, sort_keys=True).encode()).hexdigest()


def eval_args_kwargs(root):
    """crossbench_selector_eval.eval_args_kwargs values for a 30-step, full-history plan (judge fields unused)."""
    return dict(sglang_router_ip='127.0.0.1', sglang_router_port=30000, hf_checkpoint=ACTOR_PATH,
                max_steps=MAX_STEPS, context_num_screenshots=1, rollout_max_context_len=32768,
                judge_api_model='unused', judge_max_attached_imgs=0, judge_api_mode='served',
                judge_timeout_secs=180, browser_response_format_mode='browser_env', turn_history_reasoning_mode='full',
                browser_include_tool_response=1, inference_step_timeout_secs=180,
                task_timeout_secs=TASK_TIMEOUT_SECONDS, path_to_save_generated_samples=str(Path(root) / 'samples'))


def roundtrip(value):
    """JSON round trip of an env_server observation/info (screenshot bytes kept as bytes)."""
    screenshot = value.get('screenshot') if isinstance(value, dict) else None
    data = {k: v for k, v in value.items() if k != 'screenshot'} if isinstance(value, dict) else value
    data = json.loads(json.dumps(data))
    if isinstance(value, dict) and 'screenshot' in value:
        data['screenshot'] = screenshot
    return data


class InProcessWebArenaEnv:
    """LocalProcessWebEnv's interface over an in-process snapshot WebEnv; exit() is deferred to close()."""

    def __init__(self, *, storage_state, mark, record_dir, viewport=VIEWPORT):
        self.storage_state, self.mark, self.viewport = storage_state, mark, dict(viewport)
        self.record_dir = Path(record_dir)
        self.record_dir.mkdir(parents=True, exist_ok=True)
        self.web, self.steps, self.answer = None, [], None
        self.start_url = None
        self.exit_requested = False
        self.closed = False
        self._cached = None

    async def initialize(self, task_id, task_data, env_config, tool_list=None, policy=''):
        from openwebrl.env.web_env import WebEnv
        self.mark('browser_reset')
        self.start_url = task_data['start_url']
        config = env_config
        self.web = WebEnv(width=self.viewport['width'], height=self.viewport['height'], dpr=config['dpr'],
                          max_retries=config['max_retries'], wait_timeout=config['wait_timeout'],
                          screenshot_timeout=config.get('screenshot_timeout'), start_url=task_data['start_url'],
                          resize_output_coords=config['resize_output_coords'], resize_scale=config['resize_scale'],
                          image_patch_size=config['image_patch_size'], tool_list=tool_list or [], policy=policy,
                          storage_state=self.storage_state)
        await self.web.setup()                       # env_server /reset: setup() then reset()
        observation, info = await self.web.reset()
        self._save_screenshot(0, observation.get('screenshot'))
        self._cached = (roundtrip(observation), roundtrip(info))

    async def setup(self):
        pass

    async def reset(self, url=None, auth_info=None):
        if self._cached is None:
            raise RuntimeError('reset() without initialize()')
        value, self._cached = self._cached, None
        return value

    def _save_screenshot(self, index, data):
        if isinstance(data, (bytes, bytearray)) and data:
            (self.record_dir / f'observation-{index:03d}.png').write_bytes(bytes(data))

    async def step(self, action_list):
        self.mark('browser_step')
        recorded = deepcopy(action_list)              # WebEnv rescales point_2d in place
        row = dict(index=len(self.steps), actions=recorded, url=None, tool_responses=None, terminated=None,
                   screenshot=False, error=None)
        self.steps.append(row)
        for action in recorded:
            if isinstance(action, dict) and action.get('name') == 'done':
                self.answer = str((action.get('args') or {}).get('response', ''))
        observation, reward, terminated, truncated, info = await self.web.step(action_list)
        row.update(terminated=bool(terminated), tool_responses=deepcopy(info.get('tool_responses')),
                   env_message=info.get('env_message'))
        if not observation.get('screenshot'):
            row['error'] = 'Action execution failed without a screenshot.'
            raise ValueError('Action execution failed without a screenshot.')
        row.update(url=observation.get('active_tab_url'), screenshot=True)
        self._save_screenshot(len(self.steps), observation.get('screenshot'))
        return roundtrip(observation), reward, terminated, truncated, roundtrip(info)

    async def exit(self):
        self.exit_requested = True                    # the evaluator still needs the live terminal page

    async def close(self):
        if self.closed:
            return
        self.closed = True
        if self.web is not None:
            await self.web.exit()


class SyncPage:
    """Synchronous page facade (url/goto/content/evaluate/is_closed/context.browser.is_connected) for the released
    evaluator, running each call on the episode's event-loop thread against the env's current active page."""

    def __init__(self, env, loop, timeout=120):
        self.env, self.loop, self.timeout = env, loop, timeout
        self.calls = []

    def _run(self, name, fn):
        async def call():
            value = fn()
            if inspect.isawaitable(value):
                value = await value
            return value
        self.calls.append(name)
        return asyncio.run_coroutine_threadsafe(call(), self.loop).result(self.timeout)

    @property
    def _page(self):
        return self.env.web.page

    @property
    def url(self):
        return self._run('url', lambda: self._page.url)

    def is_closed(self):
        return self._run('is_closed', lambda: self._page.is_closed())

    @property
    def context(self):
        connected = lambda: self._run('is_connected', lambda: self._page.context.browser.is_connected())
        return SimpleNamespace(browser=SimpleNamespace(is_connected=connected))

    def goto(self, url, *args, **kwargs):
        self._run('goto', lambda: self._page.goto(url, *args, **kwargs))
        return None

    def content(self, *args, **kwargs):
        return self._run('content', lambda: self._page.content(*args, **kwargs))

    def evaluate(self, expression, *args, **kwargs):
        return self._run('evaluate', lambda: self._page.evaluate(expression, *args, **kwargs))

    def set_default_timeout(self, milliseconds):
        self._run('set_default_timeout', lambda: self._page.set_default_timeout(milliseconds))


def _state(url):
    return dict(observation=dict(text='', url=url or ''), info=dict(fail_error='', observation_metadata={}))


def project_trajectory(start_url, steps, answer, *, done):
    """WebArena evaluator trajectory [state, action, state, ..., send_msg_to_user]; no state follows the terminal.

    ``steps`` are InProcessWebArenaEnv rows. On ``done`` the done call becomes the terminal answer; other calls of
    that step (if any) stay one browser action whose following state is that step's observation.
    """
    if done and (answer is None or not steps or not any(a.get('name') == 'done' for a in steps[-1]['actions'])):
        raise ValueError('done ending without an executed done call')
    trajectory = [_state(start_url)]
    for index, step in enumerate(steps):
        calls = step['actions']
        if done and index == len(steps) - 1:
            calls = [a for a in calls if a.get('name') != 'done']
            if calls:
                trajectory += [dict(action_type='owrl_tool_calls', owrl_calls=deepcopy(calls)), _state(step['url'])]
            break
        trajectory += [dict(action_type='owrl_tool_calls', owrl_calls=deepcopy(calls)), _state(step['url'])]
    trajectory.append(dict(action_type='send_msg_to_user', answer=answer if done else NA_ANSWER,
                           owrl_terminal='done' if done else 'released_max_step_fallback_na'))
    return trajectory


def outcome_of(samples):
    """(ending, valid, detail) of the generator's final turn via openwebrl.outcome_validity."""
    from openwebrl.outcome_validity import classify_terminate_reason
    last = samples[-1]
    meta = last.metadata or {}
    outcome = classify_terminate_reason(last.status, meta.get('terminate_reason'))
    return outcome.ending, outcome.valid, str(meta.get('terminate_reason', ''))


def uds_async_client(socket_path, timeout=None):
    import httpx
    return httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(uds=str(socket_path)), trust_env=False,
                             timeout=httpx.Timeout(timeout, connect=10))


def uds_selection_request(socket_path, mark=None):
    """crossbench request_selection_result over the /select UNIX socket (retry only pre-submission connect errors)."""
    async def request(endpoint, payload, timeout, connect_timeout=10, raw_path=False):
        import httpx
        if mark is not None:
            mark('critic_request')
        url = endpoint if raw_path else endpoint + '/select'
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(uds=str(socket_path)), trust_env=False,
                                             timeout=httpx.Timeout(timeout, connect=min(timeout, connect_timeout))) as client:
                    response = await client.post(url, json=payload)
                    response.raise_for_status()
                    return response.json()
            except (httpx.ConnectTimeout, httpx.ConnectError):
                if attempt == 2:
                    raise
                await asyncio.sleep(.2)
    return request
