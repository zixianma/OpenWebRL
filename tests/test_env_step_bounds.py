"""Observation calls that Playwright does not time out must not stall env.step()."""
import ast
import asyncio
from pathlib import Path
import time
import unittest

import openwebrl.env.web_env as web_env

ROOT = Path(__file__).resolve().parents[1]


class HangingPage:
    url = 'https://example.com/'

    async def evaluate(self, *args, **kwargs):
        await asyncio.sleep(3600)

    async def title(self):
        await asyncio.sleep(3600)


class Context:
    def __init__(self, pages):
        self.pages = pages


def stalled_env():
    env = web_env.WebEnv.__new__(web_env.WebEnv)
    env.page = HangingPage()
    env.context = Context([env.page, HangingPage()])
    env.css_width, env.css_height = 1280, 1000
    return env


class ObservationBounds(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.saved = web_env.OBSERVATION_CALL_TIMEOUT_SECS, web_env.TAB_TITLE_TIMEOUT_SECS
        web_env.OBSERVATION_CALL_TIMEOUT_SECS = web_env.TAB_TITLE_TIMEOUT_SECS = 0.05

    def tearDown(self):
        web_env.OBSERVATION_CALL_TIMEOUT_SECS, web_env.TAB_TITLE_TIMEOUT_SECS = self.saved

    async def test_stalled_clickable_scan_returns_empty(self):
        started = time.monotonic()
        self.assertEqual(await stalled_env().find_all_clickable_elements(), [])
        self.assertLess(time.monotonic() - started, 1)

    async def test_stalled_tab_titles_fall_back_to_empty(self):
        started = time.monotonic()
        tabs = await stalled_env().get_all_tab_urls()
        self.assertEqual([t['title'] for t in tabs], ['', ''])
        self.assertEqual([t['url'] for t in tabs], ['https://example.com/'] * 2)
        self.assertLess(time.monotonic() - started, 1)


class DoneRescue(unittest.TestCase):
    def test_step_failure_after_done_keeps_episode(self):
        tree = ast.parse((ROOT/'openwebrl/generate_browser.py').read_text())
        fn = next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef) and n.name == '_generate_turn_sample_impl')
        handlers = [h for n in ast.walk(fn) if isinstance(n, ast.Try) for h in n.handlers
                    if 'env.step() failed' in ast.unparse(h)]
        self.assertEqual(len(handlers), 1)
        source = ast.unparse(handlers[0])
        # The done branch completes the episode before the generic abort path.
        self.assertLess(source.index("'task_completed'"), source.index("'env_step_error'"))
        self.assertIn("get('args')", source)
        self.assertIn('terminated = True', source)


if __name__ == '__main__':
    unittest.main()
