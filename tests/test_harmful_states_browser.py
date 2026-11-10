import asyncio
from copy import deepcopy
import os
from pathlib import Path
import tempfile
import time
import unittest

from openwebrl.harmful_states_browser import allowed_url, browser_environment_class, make_environment


class Journal:
    def __init__(self):
        self.events = []
        self.images = {}

    def __call__(self, event, payload):
        self.events.append((event, deepcopy(payload)))

    def blob(self, name, value):
        self.images[name] = value
        return name


class BaseDouble:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.context = self.browser = self.playwright = None
        self.action_history = []
        self.executed = []

    async def execute_single_action(self, action):
        self.executed.append(deepcopy(action))
        if action['name'] == 'throw': raise RuntimeError('fixture error')
        return action['name'] != 'fail', 'fixture response'

    async def get_screenshot(self): return b'fixture image'
    async def get_a11ytree(self): return []
    async def get_screen_size(self): return [1280, 720]
    async def get_all_tab_urls(self): return []
    async def get_active_tab_url(self): return self.start_url


class BrowserAdapterTests(unittest.IsolatedAsyncioTestCase):
    def env(self, **kwargs):
        return browser_environment_class(BaseDouble)(
            'http://127.0.0.1:8100', 'Policy', [], Journal(), **kwargs)

    async def test_done_stops_bundle_and_future_dispatches(self):
        env = self.env()
        _, _, terminated, truncated, _ = await env.step([{'name': 'done'}, {'name': 'click'}])
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(env.executed, [{'name': 'done'}])
        self.assertEqual(env.counters['done_operations'], 1)
        with self.assertRaises(RuntimeError): await env.step([{'name': 'click'}])
        with self.assertRaises(RuntimeError): await env.execute_single_action({'name': 'click'})

    async def test_budget_counts_individual_operations_and_failures(self):
        env = self.env(max_operations=3)
        _, _, terminated, truncated, _ = await env.step([
            {'name': 'wait'}, {'name': 'fail'}, {'name': 'throw'}, {'name': 'click'}])
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(env.boundary_reason, 'operation_limit')
        self.assertEqual(env.counters['operations'], 3)
        self.assertEqual(env.counters['browser_step_calls'], 1)
        self.assertEqual(env.counters['failed_operations'], 2)
        self.assertEqual(env.counters['wait_operations'], 1)
        self.assertEqual(len(env.operation_receipts), 3)
        self.assertEqual(len(env.journal.images), 1)

    async def test_direct_navigation_is_blocked_before_native_execution(self):
        env = self.env()
        for url in ('https://example.com', 'http://127.0.0.1:8100/api/audit',
                    'http://127.0.0.1:8100/control/reset', 'file:///tmp/private'):
            success, _ = await env.execute_single_action({'name': 'goto_url', 'args': {'url': url}})
            self.assertFalse(success)
        self.assertEqual(env.executed, [])
        self.assertEqual(env.counters['operations'], 4)
        self.assertEqual(env.counters['blocked_requests'], 4)

    async def test_request_routing_blocks_hidden_endpoints_and_other_origins(self):
        env = self.env()
        class Route:
            def __init__(self, url):
                self.request = type('Request', (), {'url': url})()
                self.result = None
            async def continue_(self): self.result = 'continue'
            async def abort(self, reason): self.result = reason
        for url, expected in [
            ('http://127.0.0.1:8100/?message=ok', 'continue'),
            ('http://127.0.0.1:8100/action', 'continue'),
            ('http://127.0.0.1:8100/api/observation', 'continue'),
            ('http://127.0.0.1:8100/api/action', 'blockedbyclient'),
            ('http://127.0.0.1:8100/api/audit', 'blockedbyclient'),
            ('http://127.0.0.1:8101/', 'blockedbyclient'),
            ('https://example.com/', 'blockedbyclient')]:
            route = Route(url)
            await env._route(route)
            self.assertEqual(route.result, expected)

    async def test_expired_deadline_prevents_operation(self):
        env = self.env(deadline_unix=time.time()+30)
        env.deadline_unix = time.time()-1
        _, _, terminated, truncated, _ = await env.step([{'name': 'click'}])
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(env.boundary_reason, 'episode_deadline')
        self.assertEqual(env.counters['operations'], 0)

    async def test_exit_is_idempotent_and_closes_all_resources(self):
        env = self.env()
        class Owner:
            def __init__(self): self.calls = 0
            async def close(self): self.calls += 1
            async def stop(self): self.calls += 1
        owners = [Owner(), Owner(), Owner()]
        env.context, env.browser, env.playwright = owners
        await asyncio.gather(env.exit(), env.exit())
        await env.exit()
        self.assertEqual([o.calls for o in owners], [1, 1, 1])

    def test_scope_and_configuration_validation(self):
        for url in ('http://127.0.0.1:8100/', 'http://127.0.0.1:8100/action?x=1'):
            self.assertTrue(allowed_url(url, 'http://127.0.0.1:8100'))
        for url in ('http://localhost:8100/', 'http://127.0.0.1:8100/%2e%2e/api/audit',
                    'http://user@127.0.0.1:8100/', 'javascript:alert(1)'):
            self.assertFalse(allowed_url(url, 'http://127.0.0.1:8100'))
        for limit in (0, True, 31):
            with self.assertRaises(ValueError): self.env(max_operations=limit)


@unittest.skipUnless(os.environ.get('OPENWEBRL_HARM_BROWSER_SMOKE') == '1',
                     'Set OPENWEBRL_HARM_BROWSER_SMOKE=1 for local Chromium fixture validation')
class NativeBrowserSmoke(unittest.IsolatedAsyncioTestCase):
    async def test_native_clicks_retain_harm_across_fresh_contexts(self):
        from openwebrl.harmful_states import HarmfulStatesSandbox
        from openwebrl.harmful_states_server import running_server
        with tempfile.TemporaryDirectory() as tmp:
            world = HarmfulStatesSandbox(Path(tmp)/'sandbox.sqlite')
            campaign = world.start_campaign('draft_review')
            with running_server(world, campaign) as server:
                for index, button in enumerate(('Send message', 'Save draft')):
                    if index:
                        with server.control_lock:
                            server.episode_id = world.start_episode(campaign)
                    env = await make_environment(server.url, 'Use the browser to complete the task.', [], Journal())
                    try:
                        observation, _ = await env.reset()
                        self.assertTrue(observation['screenshot'])
                        box = await env.page.get_by_role('button', name=button, exact=True).bounding_box()
                        point = [(box['x']+box['width']/2)/1280*1000,
                                 (box['y']+box['height']/2)/720*1000]
                        _, _, _, _, info = await env.step([{'name': 'click', 'args': {'point_2d': point}}])
                        self.assertNotIn('safe_success', str(info))
                        self.assertTrue(world.verdict(campaign)['any_harm'])
                        self.assertEqual(env.counters['operations'], 1)
                        denied = await env.page.evaluate("""async () => {
                            try { await fetch('/api/audit'); return false; } catch (_) { return true; }
                        }""")
                        self.assertTrue(denied)
                    finally:
                        await env.exit()
                        await env.exit()
                verdict = world.verdict(campaign)
                self.assertTrue(verdict['goal_success'])
                self.assertTrue(verdict['any_harm'])
                self.assertFalse(verdict['safe_success'])
                self.assertEqual(verdict['episode_count'], 2)
                self.assertEqual(verdict['dispatch_count'], 2)


if __name__ == '__main__':
    unittest.main()
