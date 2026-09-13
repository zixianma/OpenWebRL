import asyncio
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import diagnose_browser_scaling as diagnostic


class BrowserDiagnosticTest(unittest.TestCase):
    def test_comparisons_use_same_urls_and_attempt_count(self):
        specs = diagnostic.trial_specs()
        self.assertEqual(specs[0], (1, 0.0, 6))
        self.assertEqual([x[0] for x in specs[1:]], [32,32,64,64,96,96])
        self.assertTrue(all(x[2] == 96 for x in specs[1:]))
        self.assertEqual(len(diagnostic.URLS), 6)

    def test_skips_are_not_counted_as_browser_failures(self):
        result = diagnostic.summarize([
            {'url':'a','success':True}, {'url':'a','success':False},
            {'url':'b','skipped':True}])
        self.assertEqual(result['success_rate'], 0.5)
        self.assertEqual(result['attempts'], 2)
        self.assertEqual(result['skipped'], 1)

    def test_challenge_pages_are_separate_from_screenshot_failures(self):
        self.assertTrue(diagnostic.suspected_challenge('GTmetrix Performing security verification'))
        self.assertFalse(diagnostic.suspected_challenge('VELUX daylight products'))
        result = diagnostic.summarize([{'url':'a','success':True,'suspected_challenge':True}])
        self.assertEqual(result['success_rate'],1)
        self.assertEqual(result['suspected_challenges'],1)

    def test_rl_requires_a_complete_reliable_burst_case(self):
        good = {'concurrency_limit':64,'launch_interval_seconds':0,'attempts':96,'skipped':0,'success_rate':0.98}
        self.assertEqual(diagnostic.confirmation_candidate([good]),64)
        for change in [{'launch_interval_seconds':0.25},{'success_rate':0.9},{'skipped':1},{'attempts':80}]:
            self.assertIsNone(diagnostic.confirmation_candidate([good|change]))

    def test_screenshot_failure_preserves_both_navigations_and_cleans_up(self):
        class Page:
            async def goto(self, url, **kwargs):
                return None
        class Env:
            def __init__(self, **kwargs):
                self.url = kwargs['start_url']
                self.page = Page()
            async def _initialize_context(self):
                await self.page.goto(self.url)
            async def setup(self):
                await self._initialize_context()
            async def reset(self):
                await self._initialize_context()
                await self.get_screenshot()
            async def get_screenshot(self):
                raise TimeoutError('Page.screenshot: Timeout 30000ms exceeded')
            async def exit(self):
                closed.append(True)
        closed = []
        modules = {
            'playwright.async_api': SimpleNamespace(Page=Page),
            'openwebrl.env.web_env': SimpleNamespace(WebEnv=Env),
            'yaml': SimpleNamespace(safe_load=json.loads),
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = root/'openwebrl/env/config.yaml'; cfg.parent.mkdir(parents=True)
            cfg.write_text(json.dumps({k:1 for k in ('width','height','dpr','max_retries',
                'wait_timeout','screenshot_timeout','resize_output_coords','resize_scale','image_patch_size')}))
            output = root/'result.json'
            with patch.object(diagnostic, 'SOURCE', root), patch.dict(sys.modules, modules), patch.object(sys,'path',sys.path.copy()), patch.object(diagnostic.asyncio, 'sleep', new_callable=AsyncMock) as sleep:
                asyncio.run(diagnostic.browser_worker({'url':'https://example.com', 'hold_seconds':20}, output))
            sleep.assert_awaited_once_with(20)
            row = json.loads(output.read_text())
            self.assertFalse(row['success'])
            self.assertEqual(row['failure_stage'],'screenshot')
            self.assertEqual(len(row['navigation_calls']),2)
            self.assertEqual(closed,[True])
            self.assertIn('cleanup',row['timings'])
            self.assertLessEqual(row['setup_complete_time'], row['reset_started_time'])


if __name__ == '__main__':
    unittest.main()
