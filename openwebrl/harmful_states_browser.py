"""Native screenshot/action browser adapter restricted to the local fixture.

This adapter starts no model or judge. The normal BrowserAdapter prompt and
tool parser remain in generate_browser; only browser lifetime, network scope,
operation accounting and durable observations are supplied here.
"""
from __future__ import annotations

import asyncio
from copy import deepcopy
import hashlib
import math
import time
from urllib.parse import urlsplit


PUBLIC_PATHS = frozenset(('/', '/action', '/api/observation'))


def allowed_url(url, origin):
    try:
        parsed, base = urlsplit(url), urlsplit(origin)
        return (parsed.scheme == 'http' and parsed.hostname == '127.0.0.1'
                and parsed.netloc == base.netloc and not parsed.username
                and not parsed.password and (parsed.path or '/') in PUBLIC_PATHS)
    except (TypeError, ValueError):
        return False


def browser_environment_class(base_class=None):
    """Inject a small base double for CPU tests; use the real WebEnv by default."""
    if base_class is None:
        from openwebrl.env.web_env import WebEnv
        base_class = WebEnv

    class HarmfulStatesBrowser(base_class):
        def __init__(self, url, policy, tool_list, journal, max_operations=30,
                     deadline_unix=None):
            parsed = urlsplit(url)
            if (not allowed_url(url, url) or parsed.path not in ('', '/')
                    or parsed.query or parsed.fragment):
                raise ValueError('A loopback fixture root URL is required')
            if not isinstance(policy, str) or not policy.strip() or not callable(journal):
                raise ValueError('A browser policy and callable journal are required')
            if type(max_operations) is not int or not 1 <= max_operations <= 30:
                raise ValueError('Operation limit must be between 1 and 30')
            if deadline_unix is not None and (not math.isfinite(deadline_unix)
                                              or deadline_unix <= time.time()):
                raise ValueError('Episode deadline must be a future Unix timestamp')
            super().__init__(width=1280, height=720, dpr=1, max_retries=1,
                wait_timeout=10000, screenshot_timeout=15000, start_url=url,
                resize_output_coords=True, resize_scale=1000, image_patch_size=32,
                tool_list=tool_list, policy=policy, storage_state=None)
            self.origin = f'{parsed.scheme}://{parsed.netloc}'
            self.journal, self.max_operations = journal, max_operations
            self.deadline_unix = deadline_unix
            self.boundary_reason = None
            self.counters = dict(browser_step_calls=0, operations=0,
                failed_operations=0, wait_operations=0, done_operations=0,
                blocked_requests=0, operation_seconds=0.)
            self.operation_receipts = []
            self._capture_index = 0
            self._exiting = None
            self._closed = False
            self.playwright = None
            self.browser_args = ['--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage',
                                 '--disable-background-networking']

        async def setup(self):
            from playwright.async_api import async_playwright
            from openwebrl.env.browser_runtime import browser_process_environment
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(headless=True, timeout=30000,
                args=self.browser_args, env=browser_process_environment(self.browser_args))

        async def _route(self, route):
            if allowed_url(route.request.url, self.origin):
                await route.continue_()
            else:
                self.counters['blocked_requests'] += 1
                self.journal('network_blocked', dict(url=route.request.url))
                await route.abort('blockedbyclient')

        async def _websocket(self, socket):
            self.counters['blocked_requests'] += 1
            self.journal('websocket_blocked', dict(url=socket.url))
            await socket.close()

        async def _initialize_context(self, enable_recording=False, start_url=None, auth_info=None):
            target = start_url or self.start_url
            if enable_recording or auth_info or not allowed_url(target, self.origin):
                raise ValueError('Only a fresh unauthenticated local browser is allowed')
            if self.context is not None:
                await self.context.close()
            self.context = await self.browser.new_context(
                viewport={'width': 1280, 'height': 720}, device_scale_factor=1,
                locale='en-US', timezone_id='UTC', service_workers='block')
            self.context.set_default_timeout(10000)
            await self.context.route('**/*', self._route)
            if not callable(getattr(self.context, 'route_web_socket', None)):
                raise RuntimeError('Playwright websocket routing support is required')
            await self.context.route_web_socket('**/*', self._websocket)
            self.page = await self.context.new_page()
            await self.page.goto(target, timeout=15000, wait_until='domcontentloaded')
            self.setup_global_page_listener()
            self.setup_dialog_interceptor()

        async def _observation(self):
            screenshot = await self.get_screenshot()
            if not screenshot:
                raise RuntimeError('Browser screenshot is missing')
            self._capture_index += 1
            artifact = None
            if callable(getattr(self.journal, 'blob', None)):
                artifact = self.journal.blob(f'capture-{self._capture_index:05d}.png', screenshot)
            self.journal('browser_observation', dict(capture=self._capture_index,
                screenshot_sha256=hashlib.sha256(screenshot).hexdigest(),
                screenshot_artifact=artifact, operations=self.counters['operations'],
                boundary_reason=self.boundary_reason))
            return dict(screenshot=screenshot, a11ytree=await self.get_a11ytree(),
                screen_size=await self.get_screen_size(), all_tab_url=await self.get_all_tab_urls(),
                active_tab_url=await self.get_active_tab_url())

        async def reset(self, url=None, auth_info=None):
            if self._closed or self.boundary_reason:
                raise RuntimeError('Episode is closed; create a fresh environment')
            await self._initialize_context(start_url=url or self.start_url, auth_info=auth_info)
            return await self._observation(), dict(tool_list=self.tool_list, policy=self.policy,
                env_message='Browser ready', tool_responses=[])

        def _remaining(self):
            return 30. if self.deadline_unix is None else self.deadline_unix-time.time()

        async def execute_single_action(self, action):
            if self._closed or self.boundary_reason:
                raise RuntimeError('No operations may execute after an episode boundary')
            if self.counters['operations'] >= self.max_operations:
                self.boundary_reason = 'operation_limit'
                raise RuntimeError('Browser operation budget exhausted')
            if self._remaining() <= 0:
                self.boundary_reason = 'episode_deadline'
                raise RuntimeError('Episode deadline reached')
            if not isinstance(action, dict) or not isinstance(action.get('name'), str):
                raise ValueError('Malformed browser operation')
            name = action['name']
            self.counters['operations'] += 1
            if name == 'wait': self.counters['wait_operations'] += 1
            if name == 'done': self.counters['done_operations'] += 1
            receipt = dict(operation=self.counters['operations'], action=deepcopy(action),
                           status='dispatched', started_unix=time.time())
            self.journal('operation_dispatched', deepcopy(receipt))
            started = time.monotonic()
            cancellation = None
            try:
                if name == 'call_user':
                    self.boundary_reason = 'requires_user_input'
                    flag, message = False, 'No additional user input is available.'
                elif name == 'goto_url' and not allowed_url(action.get('args', {}).get('url'), self.origin):
                    self.counters['blocked_requests'] += 1
                    flag, message = False, 'Navigation outside the task application is unavailable.'
                else:
                    flag, message = await asyncio.wait_for(
                        super().execute_single_action(deepcopy(action)), min(30., self._remaining()))
                receipt.update(status='returned', success=bool(flag), tool_response=message)
            except BaseException as exc:
                flag, message = False, 'Browser operation failed: '+type(exc).__name__
                receipt.update(status='error', success=False, error_type=type(exc).__name__)
                if not isinstance(exc, Exception): cancellation = exc
            finally:
                receipt['seconds'] = time.monotonic()-started
                self.counters['operation_seconds'] += receipt['seconds']
            if not flag: self.counters['failed_operations'] += 1
            if name == 'done' and flag: self.boundary_reason = 'done'
            if self._remaining() <= 0: self.boundary_reason = self.boundary_reason or 'episode_deadline'
            if self.counters['operations'] >= self.max_operations:
                self.boundary_reason = self.boundary_reason or 'operation_limit'
            self.operation_receipts.append(receipt)
            self.journal('operation_finished', deepcopy(receipt))
            if cancellation is not None: raise cancellation
            return flag, message

        async def step(self, actions):
            if self._closed or self.boundary_reason:
                raise RuntimeError('Episode has already ended')
            if not isinstance(actions, list) or not actions:
                raise ValueError('A nonempty browser operation list is required')
            self.counters['browser_step_calls'] += 1
            self.action_history.append(deepcopy(actions))
            responses = []
            for action in actions:
                if self._remaining() <= 0:
                    self.boundary_reason = 'episode_deadline'
                    break
                flag, message = await self.execute_single_action(action)
                responses.append(dict(tool_name=action['name'], tool_response=message))
                if self.boundary_reason: break
            return await self._observation(), 0., self.boundary_reason == 'done', self.boundary_reason not in (None, 'done'), dict(
                tool_list=self.tool_list, policy=self.policy, tool_responses=responses,
                env_message='Browser operation completed', boundary_reason=self.boundary_reason)

        async def _close(self):
            self._closed = True
            for owner, method in ((self.context, 'close'), (self.browser, 'close'), (self.playwright, 'stop')):
                if owner is not None:
                    try:
                        await asyncio.wait_for(getattr(owner, method)(), 15)
                    except Exception as exc:
                        self.journal('browser_cleanup_error', dict(resource=method, error_type=type(exc).__name__))

        async def exit(self):
            if self._exiting is None:
                self._exiting = asyncio.create_task(self._close())
            await asyncio.shield(self._exiting)

    return HarmfulStatesBrowser


async def make_environment(url, policy, tool_list, journal, max_operations=30, deadline_unix=None):
    env = browser_environment_class()(url, policy, tool_list, journal, max_operations, deadline_unix)
    try:
        await env.setup()
    except BaseException:
        await env.exit()
        raise
    return env
