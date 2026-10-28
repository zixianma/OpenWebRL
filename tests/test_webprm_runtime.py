import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from openwebrl import webprm_runtime

from openwebrl.webprm_runtime import load_api_credentials, prepare_browser_runtime, navigation_timeout_ms
from openwebrl.webprm_worker import CHROMIUM, completion, prepare_browser


class RuntimeRegressionTests(unittest.TestCase):
    def test_missing_credentials_cannot_reserve_or_log_request(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as directory, patch.dict(os.environ, {}, clear=True):
            root = Path(directory)
            config = dict(deadline=time.time()+30, budget_path=str(root/'budget.json'))
            with self.assertRaisesRegex(ValueError, 'OPENAI_API_KEY is absent'):
                completion(config, dict(model='gpt-4o-mini-2024-07-18', messages=[]), root/'requests.jsonl')
            self.assertFalse((root/'budget.json').exists())
            self.assertFalse((root/'requests.jsonl').exists())

    def test_explicit_env_file_works_without_controller_parent(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as directory, patch.dict(os.environ, {}, clear=True):
            path = Path(directory)/'fixture.env'
            path.write_text('OPENAI_API_KEY=fixture-no-provider-credential\n')
            self.assertEqual(load_api_credentials(path), 'fixture-no-provider-credential')
            self.assertEqual(load_api_credentials(), 'fixture-no-provider-credential')

    def test_long_temp_and_home_cache_settings_are_replaced(self):
        old_temp = tempfile.tempdir
        try:
            with patch.dict(os.environ, dict(TMPDIR='/path/'+'long/'*60,
                            XDG_CACHE_HOME='/gpfs/home/zixianma/.cache')), patch.object(webprm_runtime, '_PROCESS_RUNTIME', None):
                os.environ.pop('WEBPRM_BROWSER_TMP_ROOT', None)
                root = prepare_browser_runtime()
                self.assertEqual(root.parent, Path('/tmp'))
                self.assertLess(len(str(root)), 40)
                self.assertEqual(Path(tempfile.gettempdir()), root)
                for name in ('XDG_CACHE_HOME','XDG_CONFIG_HOME','XDG_DATA_HOME'):
                    self.assertEqual(Path(os.environ[name]).parent, root)
        finally:
            tempfile.tempdir = old_temp

    def test_campaign_root_owns_browser_profiles_and_caches(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as folder, patch.object(webprm_runtime, '_PROCESS_RUNTIME', None), \
                patch.object(tempfile, 'tempdir', None), patch.dict(os.environ, WEBPRM_BROWSER_TMP_ROOT=folder):
            root = prepare_browser_runtime()
            self.assertEqual(root.parent, Path(folder))
            self.assertEqual(prepare_browser_runtime(), root)
            self.assertEqual(Path(os.environ['TMPDIR']), root)
            for name in ('XDG_CACHE_HOME', 'XDG_CONFIG_HOME', 'XDG_DATA_HOME'):
                self.assertEqual(Path(os.environ[name]).parent, root)
            other = Path(folder) / 'other'; other.mkdir()
            with patch.dict(os.environ, WEBPRM_BROWSER_TMP_ROOT=str(other)), self.assertRaisesRegex(ValueError, 'changed'):
                prepare_browser_runtime()

    def test_campaign_root_rejects_home_relative_missing_and_symlink_paths_before_creation(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as folder, patch.object(webprm_runtime, '_PROCESS_RUNTIME', None):
            base = Path(folder)
            (base/'protected').symlink_to('/gpfs/home/zixianma')
            (base/'local-link').symlink_to(base)
            for value in ['/gpfs/home/zixianma/forbidden', 'relative', str(base/'missing'),
                          str(base/'protected'/'forbidden'), str(base/'local-link')]:
                with self.subTest(value=value), patch.dict(os.environ, WEBPRM_BROWSER_TMP_ROOT=value), \
                        self.assertRaises(ValueError):
                    prepare_browser_runtime()
            self.assertIsNone(webprm_runtime._PROCESS_RUNTIME)
            self.assertFalse((base/'missing').exists())

    def test_reused_runtime_rejects_existing_cache_symlink_to_home(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as folder, patch.object(tempfile, 'tempdir', None), \
                patch.dict(os.environ, WEBPRM_BROWSER_TMP_ROOT=folder):
            root=Path(folder)/'wprm-synthetic';root.mkdir()
            (root/'cache').symlink_to('/gpfs/home/zixianma')
            with patch.object(webprm_runtime, '_PROCESS_RUNTIME', (os.getpid(),root)), \
                    self.assertRaisesRegex(ValueError, 'symlink'):
                prepare_browser_runtime()

    def test_navigation_cannot_extend_episode_deadline(self):
        with patch('openwebrl.webprm_runtime.time.time', return_value=100):
            self.assertEqual(navigation_timeout_ms(103), 3000)
            self.assertEqual(navigation_timeout_ms(200), 30000)
            with self.assertRaises(TimeoutError):
                navigation_timeout_ms(99)

    def test_released_multi_tab_start_urls_open_separately_and_focus_first(self):
        events = []

        class Context:
            def __init__(self):
                self.pages = []

            def new_page(self):
                page = Page(self)
                self.pages.append(page)
                return page

        class Page:
            def __init__(self, context):
                self.context = context
                self.url = 'about:blank'

            def goto(self, url, **options):
                events.append((url, options['timeout']))
                self.url = url
                return type('Response', (), {'status': 200})()

            def bring_to_front(self):
                events.append(('focus', self))

            def wait_for_load_state(self, state, **options):
                events.append(('loaded', self.url, state, options['timeout']))

            def evaluate(self, script):
                self.binding_script = script

        context = Context()
        first = context.new_page()
        urls = ['https://fixture.invalid/one', 'https://fixture.invalid/two',
                'https://fixture.invalid/three']
        task = webprm_runtime.live_task_type()(42, ' |AND| '.join(urls),
            goal='Compare the open tabs', deadline=110)
        with patch('openwebrl.webprm_runtime.time.time',
                   side_effect=[100, 103, 108, 108.5, 109, 109.5]):
            goal, info = task.setup(first)
        self.assertEqual([page.url for page in context.pages], urls)
        self.assertEqual(events[:3], list(zip(urls, [10000, 7000, 2000])))
        self.assertEqual(events[3:6], [('loaded', url, 'load', timeout)
                                     for url, timeout in zip(urls, [1500, 1000, 500])])
        self.assertEqual(events[6], ('focus', first))
        self.assertIn('browsergym_page_activated', first.binding_script)
        self.assertEqual(goal, task.goal)
        self.assertEqual([n['final_url'] for n in info['initial_navigations']], urls)
        self.assertEqual(info['initial_navigation']['final_url'], urls[0])

    def test_multi_tab_navigation_failure_is_not_retried_or_ignored(self):
        from types import SimpleNamespace
        first = SimpleNamespace(url='https://fixture.invalid/one')
        second = SimpleNamespace(url='https://fixture.invalid/two')
        first.goto = lambda *args, **kwargs: SimpleNamespace(status=200)
        second.goto = lambda *args, **kwargs: SimpleNamespace(status=404)
        from unittest.mock import Mock
        context = SimpleNamespace(new_page=Mock(return_value=second))
        first.context = context
        task = webprm_runtime.live_task_type()(42,
            'https://fixture.invalid/one |AND| https://fixture.invalid/two |AND| https://fixture.invalid/three',
            deadline=time.time()+30)
        with self.assertRaisesRegex(RuntimeError, 'HTTP status 404'):
            task.setup(first)
        context.new_page.assert_called_once_with()

    def test_multi_tab_load_timeout_cannot_be_ignored_or_retried(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        first = SimpleNamespace(url='https://fixture.invalid/one',
            goto=Mock(return_value=SimpleNamespace(status=200)),
            wait_for_load_state=Mock(side_effect=TimeoutError('pending initial resource')),
            bring_to_front=Mock())
        second = SimpleNamespace(url='https://fixture.invalid/two',
            goto=Mock(return_value=SimpleNamespace(status=200)), wait_for_load_state=Mock())
        first.context = SimpleNamespace(pages=[first, second], new_page=Mock(return_value=second))
        task = webprm_runtime.live_task_type()(42,
            'https://fixture.invalid/one |AND| https://fixture.invalid/two', deadline=110)
        with patch('openwebrl.webprm_runtime.time.time', return_value=109), \
                self.assertRaisesRegex(TimeoutError, 'pending initial resource'):
            task.setup(first)
        first.wait_for_load_state.assert_called_once_with('load', timeout=1000)
        second.wait_for_load_state.assert_not_called()
        first.bring_to_front.assert_not_called()

    def test_campaign_playwright_cache_stays_under_owned_runtime(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as folder, patch.object(webprm_runtime, '_PROCESS_RUNTIME', None), \
                patch.object(tempfile, 'tempdir', None), patch.dict(os.environ, WEBPRM_BROWSER_TMP_ROOT=folder):
            prepare_browser()
            cache = Path(os.environ['PLAYWRIGHT_BROWSERS_PATH'])
            self.assertEqual(cache.parent.parent, Path(folder))
            self.assertEqual((cache/'chromium-1117/chrome-linux/chrome').resolve(), Path(CHROMIUM).resolve())
            prepare_browser()  # Reuse only this process's validated cache.

    def test_campaign_playwright_parent_symlink_rejected_before_mkdir(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as folder, patch.dict(os.environ, WEBPRM_BROWSER_TMP_ROOT=folder):
            root = Path(folder)/'owned';root.mkdir()
            protected_fixture = Path(folder)/'protected-fixture';protected_fixture.mkdir()
            (root/'playwright').symlink_to(protected_fixture)
            with patch('openwebrl.webprm_worker.prepare_browser_runtime', return_value=root), \
                    self.assertRaisesRegex(ValueError, 'parents'):
                prepare_browser()
            self.assertEqual(list(protected_fixture.iterdir()), [])

    def test_legacy_playwright_parent_symlink_rejected_before_mkdir(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as folder, patch.dict(os.environ):
            os.environ.pop('WEBPRM_BROWSER_TMP_ROOT', None)
            protected_fixture = Path(folder)/'protected-fixture';protected_fixture.mkdir()
            redirected = Path(folder)/'redirected';redirected.symlink_to(protected_fixture)
            real_path = Path
            def paths(value):
                return redirected if str(value) == '/tmp/openwebrl-webprm-browser-cache' else real_path(value)
            with patch('openwebrl.webprm_worker.prepare_browser_runtime', return_value=Path(folder)), \
                    patch('openwebrl.webprm_worker.Path', side_effect=paths), \
                    self.assertRaisesRegex(ValueError, 'parents'):
                prepare_browser()
            self.assertEqual(list(protected_fixture.iterdir()), [])


if __name__ == '__main__':
    unittest.main()
