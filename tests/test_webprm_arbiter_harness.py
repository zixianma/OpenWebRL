import ast
import json
import logging
import re
from enum import IntEnum
from pathlib import Path
from types import SimpleNamespace
import unittest

from openwebrl.webprm_arbiter_harness import load_arbiter_harness, load_native_observation_processor, to_browsergym_actions, native_scroll_code, native_execution_code, ActionParsingError
from openwebrl.webprm_source import _extract

ROOT = Path('/tmp/openwebrl-webarbiter-upstream')


@unittest.skipUnless(ROOT.exists(), 'Reviewed upstream checkout unavailable')
class ArbiterHarness(unittest.TestCase):
    def setUp(self):
        self.api = load_arbiter_harness(ROOT)

    def test_prompt_and_parser(self):
        prompt = self.api.render_policy_prompt(objective='Find book', observation='[1] textbox',
                    url='https://example.org', trajectory='', state_diff='')
        self.assertIn('Find book', prompt)
        self.assertIn('5 independent choices', self.api.policy_system_prompt)
        raw = json.dumps({'BLOCKS': [{'thought': 'Search', 'action': 'type [1] book', 'confidence': .9}]})
        result = self.api.parse_policy_response(raw)
        self.assertTrue(result.is_valid)
        self.assertEqual(result.candidates[0].action, 'type [1] book')
        self.assertFalse(self.api.parse_policy_response('```json\n' + raw + '\n```').is_valid)

    def test_auxiliary_prompts_match_complete_upstream_modules(self):
        diff, terminal = {}, {}
        for filename, namespace in [('observation_p.py', diff), ('terminal_p.py', terminal)]:
            path = ROOT / 'viewer/src/prompts' / filename
            exec(compile(path.read_text(), str(path), 'exec'), namespace)
        self.assertEqual(self.api.state_diff_system_prompt, diff['system_text_diff'])
        self.assertEqual(self.api.render_state_diff_prompt('before [12]', 'after [24]'),
                         diff['render_diff_user_prompt']('before [12]', 'after [24]'))
        args = dict(intent='Read saved name', observation='[2] StaticText Ada',
                    current_url='https://example.org/saved', trajectory='type [1] Ada',
                    state_diff='The saved name is now Ada.',
                    previous_terminal_action={'thought': 'No saved name yet', 'action': None,
                                              'confidence': .8})
        self.assertEqual(self.api.build_terminal_prompt(**args),
                         (terminal['system_text'], terminal['render_terminal_prompt'](**args)))
        self.assertIn('DO NOT report ID changes', self.api.state_diff_system_prompt)
        self.assertIn('send_msg_to_user', self.api.build_terminal_prompt(**args)[0])
        for filename in ['prompts/observation_p.py', 'prompts/terminal_p.py',
                         'server/core/tools/state_diff.py', 'server/core/tools/terminal.py',
                         'server/engine.py']:
            self.assertIn('viewer/src/' + filename, self.api.manifest['files'])

    def test_terminal_parser_matches_complete_upstream_class(self):
        path = ROOT / 'viewer/src/server/core/tools/terminal.py'
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                   and n.name == 'TerminalDetector')
        ns = dict(json=json, re=re, BlockInfo=SimpleNamespace, ModelSpec=object,
                  TerminalDetectionCommand=object, logger=logging.getLogger(__name__),
                  TerminalDetectionResult=lambda **kw: SimpleNamespace(terminal_action=kw.get('terminal_action')))
        future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[future, cls], type_ignores=[])),
                     str(path), 'exec'), ns)
        reference = object.__new__(ns['TerminalDetector'])
        for response in [
                '{"thought":"Name is saved","action":"send_msg_to_user [Ada]","confidence":0.9}',
                '{"thought":"No evidence","action":null,"confidence":0.3}',
                '```json\n{"thought":"No evidence","action":"null"}\n```',
                '{{"thought":"Name is saved","action":"send_msg_to_user [Ada]"}}',
                '{"thought":"Broken confidence","action":null,"confidence":"invalid"}',
                'malformed']:
            expected = reference._parse_detection_result(response).terminal_action
            actual = self.api.parse_terminal_response(response).terminal_action
            self.assertEqual(vars(actual) if actual else None, vars(expected) if expected else None)

    def test_native_type_and_translation(self):
        parsed = self.api.parse_action('type [1] book')
        self.assertEqual(parsed['kwargs']['text'], 'book\n')
        self.assertEqual(to_browsergym_actions(parsed, {'1': 'a2'}),
                         ["fill('a2', 'book')", "press('a2', \"Enter\")"])
        bracketed = self.api.parse_action('type [1] [book]')
        self.assertEqual(bracketed['kwargs']['text'], '[book]\n')
        with self.assertRaises(ActionParsingError):
            self.api.parse_action('type [1] [book] [0]')

    def test_actions(self):
        self.assertEqual(to_browsergym_actions(self.api.parse_action('goto [https://example.org]')),
                         ["goto('https://example.org')"])
        self.assertEqual(to_browsergym_actions(self.api.parse_action('tab_focus [2]')), ['tab_focus(2)'])
        with self.assertRaises(ActionParsingError):
            to_browsergym_actions(self.api.parse_action('click [Book]'))
        with self.assertRaises(ActionParsingError):
            to_browsergym_actions(self.api.parse_action('click [1]'), {'2': 'a2'})

    def test_native_scroll_execution(self):
        class Page:
            def __init__(self):
                self.calls = []
            def evaluate(self, code):
                self.calls.append(code)
        native, adapter = Page(), Page()
        self.api.execute_scroll('down', native)
        exec(native_scroll_code(ROOT, 'down'), {'page': adapter})
        self.assertEqual(native.calls, adapter.calls)
        self.assertIn('window.innerHeight', adapter.calls[0])
        self.assertEqual(to_browsergym_actions(self.api.parse_action('scroll [down]')),
                         ["native_scroll('down')"])

    def test_native_coordinate_typing_events(self):
        from types import SimpleNamespace
        events = []
        page = SimpleNamespace(viewport_size={'width': 1280, 'height': 720},
                               mouse=SimpleNamespace(click=lambda x, y: events.append(('click', x, y))),
                               keyboard=SimpleNamespace(press=lambda key: events.append(('press', key)),
                                                        type=lambda text: events.append(('type', text))),
                               evaluate=lambda code: 'Linux', wait_for_timeout=lambda ms: None)
        page.context = SimpleNamespace(pages=[page])
        parsed = self.api.parse_action('type [12] hello')
        code = native_execution_code(ROOT, parsed, {'12': [10, 20, 100, 40]}, key_mapping=self.api.key_mapping)
        exec(code, {'page': page})
        self.assertEqual(events, [('click', 60.0, 40.0), ('press', 'Control+A'),
                                  ('press', 'Backspace'), ('type', 'hello'), ('press', 'Enter')])

    def test_navigation_click_matches_upstream_without_js_bridge(self):
        """A navigation can destroy JS context after a successful native click."""
        upstream = dict(IntEnum=IntEnum, Action=dict, Page=object,
                        BrowserContext=object, ObservationProcessor=object)
        _extract(ROOT / 'viewer/src/browser_env/actions.py', upstream,
                 names={'ActionTypes', 'execute_mouse_click', 'execute_action'})
        def make_page():
            events = []
            def evaluate(code):
                raise RuntimeError('Execution context was destroyed, most likely because of a navigation')
            page = SimpleNamespace(viewport_size={'width': 1280, 'height': 720},
                mouse=SimpleNamespace(click=lambda x, y: events.append(('click', x, y))),
                wait_for_timeout=lambda ms: events.append(('wait', ms)), evaluate=evaluate)
            page.context = SimpleNamespace(pages=[page])
            return page, events
        native_page, native_events = make_page()
        adapted_page, adapted_events = make_page()
        processor = SimpleNamespace(get_element_center=lambda eid: (60 / 1280, 40 / 720))
        upstream['execute_action'](
            {'action_type': upstream['ActionTypes'].CLICK, 'element_id': '12'},
            native_page, native_page.context, processor)
        exec(native_execution_code(ROOT, self.api.parse_action('click [12]'),
                                   {'12': [10, 20, 100, 40]}), {'page': adapted_page})
        self.assertEqual(adapted_events, native_events)

    def test_tab_activation_retries_only_bridge_on_navigation(self):
        calls = []
        def evaluate(code):
            calls.append('activate')
            if calls.count('activate') == 1:
                raise RuntimeError('Execution context was destroyed, most likely because of a navigation')
        target = SimpleNamespace(viewport_size={'width': 1280, 'height': 720},
                                 bring_to_front=lambda: calls.append('focus'),
                                 wait_for_timeout=lambda ms: None,
                                 wait_for_load_state=lambda state: calls.append(state),
                                 evaluate=evaluate)
        original = SimpleNamespace(viewport_size={'width': 1280, 'height': 720})
        context = SimpleNamespace(pages=[original, target])
        original.context = target.context = context
        exec(native_execution_code(ROOT, self.api.parse_action('tab_focus [1]'), {}),
             {'page': original})
        self.assertEqual(calls, ['focus', 'domcontentloaded', 'activate',
                                'domcontentloaded', 'activate'])

    def test_tab_activation_preserves_non_navigation_errors(self):
        def evaluate(code):
            raise RuntimeError('Page closed')
        target = SimpleNamespace(viewport_size={'width': 1280, 'height': 720},
                                 bring_to_front=lambda: None, wait_for_timeout=lambda ms: None,
                                 wait_for_load_state=lambda state: None, evaluate=evaluate)
        original = SimpleNamespace(viewport_size={'width': 1280, 'height': 720})
        context = SimpleNamespace(pages=[original, target])
        original.context = target.context = context
        with self.assertRaisesRegex(RuntimeError, 'Page closed'):
            exec(native_execution_code(ROOT, self.api.parse_action('tab_focus [1]'), {}),
                 {'page': original})

    def test_popup_viewport_is_restored_before_activation(self):
        calls = []
        target = SimpleNamespace(viewport_size={'width': 600, 'height': 600},
            bring_to_front=lambda: calls.append('focus'),
            wait_for_timeout=lambda ms: None,
            wait_for_load_state=lambda state: calls.append(state),
            evaluate=lambda code: calls.append(('activate', dict(target.viewport_size))))
        def resize(viewport):
            calls.append(('resize', dict(viewport)))
            target.viewport_size = dict(viewport)
        target.set_viewport_size = resize
        original = SimpleNamespace(viewport_size={'width': 1280, 'height': 720})
        context = SimpleNamespace(pages=[original, target])
        original.context = target.context = context
        exec(native_execution_code(ROOT, self.api.parse_action('tab_focus [1]'), {}),
             {'page': original})
        self.assertEqual(calls, ['focus', ('resize', {'width': 1280, 'height': 720}),
            'domcontentloaded', ('activate', {'width': 1280, 'height': 720})])


@unittest.skipUnless(ROOT.exists(), 'Reviewed upstream checkout unavailable')
class NativeObservationGeometry(unittest.TestCase):
    def setUp(self):
        self.viewport = {'width': 1280, 'height': 720}
        self.native = load_native_observation_processor(ROOT, self.viewport)
        self.native.processor.process = lambda page, client: str(page.geometry)
        self.native.processor.obs_nodes_info = {'12': {'union_bound': [950, 650, 200, 40]}}

    def tearDown(self):
        self.native.close()

    def page(self, *, reject_override=False):
        # Model Chromium's per-session deduplication of an identical override,
        # including the independent screenshot session resetting screen bounds.
        class Page:
            def __init__(page):
                page.viewport_size = dict(self.viewport)
                page.geometry = dict(screenWidth=800, screenHeight=600,
                                     innerWidth=1280, innerHeight=720, devicePixelRatio=1)
                page.clients = []
                page.context = SimpleNamespace(new_cdp_session=page.new_cdp_session)

            def evaluate(page, expression):
                return dict(page.geometry)

            def new_cdp_session(page, target):
                assert target is page
                class Client:
                    last_override = None
                    detached = False

                    def send(client, method, params=None):
                        if method == 'Emulation.clearDeviceMetricsOverride':
                            client.last_override = None
                            page.geometry.update(screenWidth=800, screenHeight=600)
                        elif method == 'Emulation.setDeviceMetricsOverride':
                            if params != client.last_override and not reject_override:
                                page.geometry.update(screenWidth=params['screenWidth'],
                                                     screenHeight=params['screenHeight'])
                            client.last_override = params
                        else:
                            raise AssertionError(method)

                    def detach(client):
                        client.detached = True
                        page.geometry.update(screenWidth=800, screenHeight=600)
                client = Client()
                page.clients.append(client)
                return client
        return Page()

    def test_geometry_survives_observation_and_recovers_after_screenshot(self):
        first, second = self.page(), self.page()
        for page in [first, first, second, first]:
            # BrowserGym's screenshot capture resets the shared screen metrics.
            page.geometry.update(screenWidth=800, screenHeight=600)
            text, bounds = self.native.observe(page)
            self.assertEqual(page.geometry['screenWidth'], 1280)
            self.assertEqual(page.geometry['screenHeight'], 720)
            self.assertEqual(self.native.last_geometry, page.geometry)
            self.assertEqual(bounds, {'12': [950, 650, 200, 40]})
            self.assertEqual(text, str(page.geometry))
            self.assertEqual(len(page.clients), 1)
            self.assertFalse(page.clients[0].detached)
        self.native.close()
        self.assertTrue(first.clients[0].detached)
        self.assertTrue(second.clients[0].detached)

    def test_failed_alignment_releases_session_without_processing_observation(self):
        page = self.page(reject_override=True)
        self.native.processor.process = lambda *args: self.fail('Unaligned geometry was processed')
        with self.assertRaisesRegex(ValueError, 'screen and viewport could not be aligned'):
            self.native.observe(page)
        self.assertTrue(page.clients[0].detached)
        self.assertEqual(self.native.last_geometry, {})

    def test_changed_viewport_is_rejected(self):
        page = self.page()
        page.viewport_size = {'width': 1000, 'height': 720}
        with self.assertRaisesRegex(ValueError, 'differs from BrowserGym page viewport'):
            self.native.observe(page)
        self.assertTrue(page.clients[0].detached)


if __name__ == '__main__':
    unittest.main()
