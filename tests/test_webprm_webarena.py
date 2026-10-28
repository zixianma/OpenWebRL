"""Actual pinned native definitions, synthetic terminal pages, zero provider calls."""
from copy import deepcopy
import builtins
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from openwebrl.webprm_webarena import (
    DOMAINS, EvaluationNotReady, SOURCE_HASHES, load_native_evaluator,
    prepare_manifest, substitute_endpoints, validate_endpoints,
    verify_source,
)

SOURCE = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-baselines-20261009/sources/arbiter')


def trajectory(answer='42'):
    return [{'observation': {'text': 'Synthetic controlled page'}, 'info': {}},
            {'action_type': 'send_msg_to_user', 'answer': answer}]


def config(kind='string_match'):
    return {'intent': 'Synthetic contract fixture', 'eval': {
        'eval_types': [kind], 'reference_answers': {'exact_match': '42'},
        'reference_url': 'http://127.0.0.1:8123/goal',
        'program_html': [{'url': 'last', 'locator': '',
                          'required_contents': {'exact_match': '<body>42</body>'}}]}}


class PreparationTests(unittest.TestCase):
    def test_endpoint_replacement_is_explicit_recursive_and_preserves_task(self):
        raw = {'start_url': '__SHOPPING__', 'eval': {'reference_url': '__SHOPPING__/goal'}}
        saved = deepcopy(raw)
        result = substitute_endpoints(raw, {'shopping': 'http://example.invalid:7770/'})
        self.assertEqual(result['eval']['reference_url'], 'http://example.invalid:7770/goal')
        self.assertEqual(raw, saved)
        with self.assertRaises(EvaluationNotReady):
            substitute_endpoints(raw, {})

    def test_credentials_and_implicit_urls_rejected(self):
        for url in ['http://user:secret@host', 'http://host/?key=secret', 'http://host/#secret',
                    'file:///tmp/site', 'localhost:7770', 'http://bad port', 'http://host:bad']:
            with self.subTest(url=url), self.assertRaises(ValueError):
                validate_endpoints({'shopping': url})

    def test_mutated_released_source_is_rejected(self):
        changed = dict(SOURCE_HASHES)
        changed[next(iter(changed))] = '0' * 64
        with patch('openwebrl.webprm_webarena.source_manifest', return_value={'files': changed}):
            with self.assertRaisesRegex(ValueError, 'modified'):
                verify_source('/tmp/unused-pinned-source-fixture')

    @unittest.skipUnless(SOURCE.exists(), 'Pinned source checkout not installed')
    def test_released_inventory_and_prospective_pilot(self):
        manifest, groups, configured = prepare_manifest(SOURCE)
        self.assertEqual({k: len(v) for k, v in groups.items()}, {
            'released_full_165': 165, 'released_single_site_155': 155,
            'mechanical_first_per_domain_5': 5})
        self.assertEqual([t['task_id'] for t in groups['mechanical_first_per_domain_5']], [4, 0, 5, 9, 1])
        self.assertEqual(manifest['single_site_counts'], {
            'shopping': 45, 'shopping_admin': 35, 'reddit': 19, 'gitlab': 30, 'map': 26})
        self.assertFalse(manifest['paper_table_5_exact_task_membership_verified'])
        self.assertEqual(manifest['evaluation_inventory'], {'fuzzy_judge_tasks': 40, 'helper_function_tasks': 14})
        self.assertEqual(configured, [])
        self.assertEqual(manifest['source']['files'], SOURCE_HASHES)

    @unittest.skipUnless(SOURCE.exists(), 'Pinned source checkout not installed')
    def test_all_task_endpoints_can_be_prepared_without_readiness_claim(self):
        manifest, _, configured = prepare_manifest(SOURCE, {
            domain: f'http://{domain.replace("_", "-")}.invalid:8080' for domain in DOMAINS})
        self.assertEqual(len(configured), 165)
        self.assertFalse(manifest['website_readiness_verified'])
        self.assertFalse(manifest['reset_and_login_verified'])


@unittest.skipUnless(SOURCE.exists(), 'Pinned source checkout not installed')
class NativeEvaluatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.native = load_native_evaluator(SOURCE)

    def setUp(self):
        # No source or adapter may accidentally make a socket connection.
        self.network = patch('socket.socket.connect', side_effect=AssertionError('Network forbidden'))
        self.network.start()
        self.addCleanup(self.network.stop)

    def test_native_exact_positive_negative_and_quote_normalization(self):
        for answer, expected in [('42', True), ('"42"', True), ('43', False)]:
            result = self.native.evaluate(config(), trajectory(answer))
            self.assertEqual(result['success'], expected)
            self.assertEqual(result['model_provider_calls'], 0)

    def test_native_url_or_and_localhost_normalization(self):
        task = config('url_match')
        task['eval']['reference_url'] += ' |OR| http://example.invalid/alternate'
        self.assertTrue(self.native.evaluate(task, trajectory(), SimpleNamespace(url='http://localhost:8123/goal/'))['success'])
        self.assertFalse(self.native.evaluate(task, trajectory(), SimpleNamespace(url='http://localhost:8123/not-goal'))['success'])

    def test_url_and_html_accept_real_final_state_at_declared_normal_termination(self):
        final_state_trajectory = [trajectory()[0], {'action_type': 'click'}, trajectory()[0]]
        saved = deepcopy(final_state_trajectory)
        page = SimpleNamespace(url='http://127.0.0.1:8123/goal', content=lambda: '<body>42</body>')
        for kind in ['url_match', 'program_html']:
            for termination in ['stop', 'turn_limit']:
                result = self.native.evaluate(config(kind), final_state_trajectory, page, termination=termination)
                self.assertTrue(result['success'])
                self.assertEqual(result['termination'], termination)
        self.assertEqual(final_state_trajectory, saved)

    def test_final_state_requires_normal_termination_and_does_not_invent_answer(self):
        state_only = [trajectory()[0]]
        page = SimpleNamespace(url='http://127.0.0.1:8123/goal')
        for termination in [None, 'browser_crash', 'deadline', 'provider_error']:
            with self.assertRaises(EvaluationNotReady):
                self.native.evaluate(config('url_match'), state_only, page, termination=termination)
        with self.assertRaisesRegex(EvaluationNotReady, 'actual terminal answer'):
            self.native.evaluate(config(), state_only, page, termination='turn_limit')
        with self.assertRaises(EvaluationNotReady):
            self.native.evaluate(config('url_match'), [trajectory()[0], {'action_type': 'click'}],
                                 page, termination='turn_limit')

    def test_empty_url_alternative_cannot_create_vacuous_substring_success(self):
        task = config('url_match')
        task['eval'].update(reference_url='http://goal.invalid |OR| ', url_note='GOLD in PRED')
        with self.assertRaisesRegex(EvaluationNotReady, 'empty match'):
            self.native.evaluate(task, trajectory(), SimpleNamespace(url='http://wrong.invalid'))

    def test_native_html_exact_and_combined_product(self):
        task = config('program_html')
        page = SimpleNamespace(url='http://127.0.0.1:8123/goal', content=lambda: '<body>42</body>')
        self.assertTrue(self.native.evaluate(task, trajectory(), page)['success'])
        task['eval']['eval_types'] += ['string_match', 'url_match']
        self.assertEqual(len(self.native.evaluate(task, trajectory(), page)['evaluator_details']), 3)
        self.assertFalse(self.native.evaluate(task, trajectory('wrong'), page)['success'])

    def test_native_javascript_locator_contract(self):
        task = config('program_html')
        task['eval']['program_html'][0].update(locator='document.body.textContent', required_contents={'exact_match': '42'})
        expressions = []
        page = SimpleNamespace(url='http://fixture.invalid', evaluate=lambda expression: expressions.append(expression) or '42')
        self.assertTrue(self.native.evaluate(task, trajectory(), page)['success'])
        self.assertEqual(expressions, ['() => document.body.textContent'])

    @staticmethod
    def javascript_page(evaluate):
        return SimpleNamespace(url='http://fixture.invalid', evaluate=evaluate,
                               is_closed=lambda: False,
                               context=SimpleNamespace(browser=SimpleNamespace(is_connected=lambda: True)))

    @staticmethod
    def javascript_config(expected='42'):
        task = config('program_html')
        task['eval']['program_html'][0].update(
            locator='document.body.textContent', required_contents={'exact_match': expected})
        return task

    def test_missing_browser_capabilities_cannot_be_scored_even_for_empty_text(self):
        page = SimpleNamespace(url='http://fixture.invalid')
        for locator, url, capability in [('document.body.textContent', 'last', 'evaluate'),
                                         ('', 'last', 'content'),
                                         ('document.body.textContent', 'http://goal.invalid', 'goto')]:
            task = self.javascript_config('')
            task['eval']['program_html'][0].update(locator=locator, url=url)
            with self.subTest(capability=capability), self.assertRaises(EvaluationNotReady):
                self.native.evaluate(task, trajectory(), page)

    def test_closed_browser_is_invalid_before_dispatch(self):
        page = self.javascript_page(lambda expression: '')
        page.is_closed = lambda: True
        with self.assertRaisesRegex(EvaluationNotReady, 'closed'):
            self.native.evaluate(self.javascript_config(''), trajectory(), page)

    def test_string_only_evaluation_does_not_depend_on_browser_liveness(self):
        page = SimpleNamespace(is_closed=lambda: True)
        self.assertTrue(self.native.evaluate(config(), trajectory(), page)['success'])

    def test_transport_and_unknown_errors_cannot_be_native_empty_text_success(self):
        from playwright.sync_api import Error
        errors = [Error('Page.evaluate: Target page, context or browser has been closed'),
                  Error('Page.evaluate: Execution context was destroyed'),
                  ConnectionError('transport disconnected'),
                  AttributeError('missing evaluator implementation'),
                  TypeError("TypeError: Cannot read properties of null (reading 'textContent')")]
        for error in errors:
            def fail(expression, error=error):
                raise error
            for expected in ['', '42']:
                with self.subTest(error=type(error).__name__, expected=expected):
                    with self.assertRaisesRegex(EvaluationNotReady, 'native result discarded'):
                        self.native.evaluate(self.javascript_config(expected), trajectory(), self.javascript_page(fail))

    def test_recognized_dom_error_on_live_browser_preserves_native_semantics(self):
        from playwright.sync_api import Error
        def missing_dom(expression):
            raise Error("Page.evaluate: TypeError: Cannot read properties of null (reading 'textContent')")
        page = self.javascript_page(missing_dom)
        result = self.native.evaluate(self.javascript_config('42'), trajectory(), page)
        self.assertTrue(result['valid'])
        self.assertFalse(result['success'])
        self.assertEqual(len(result['native_javascript_errors']), 1)
        # The released native evaluator substitutes empty text on this *healthy*
        # JavaScript failure. Preserve its exact-match semantics, disclose the
        # diagnostic, and distinguish it from unavailable browser transport.
        empty = self.native.evaluate(self.javascript_config(''), trajectory(), page)
        self.assertTrue(empty['success'])
        self.assertEqual(len(empty['native_javascript_errors']), 1)

    def test_javascript_error_without_verified_liveness_is_unavailable(self):
        from playwright.sync_api import Error
        def missing_dom(expression):
            raise Error("Page.evaluate: TypeError: Cannot read properties of null (reading 'textContent')")
        with self.assertRaises(EvaluationNotReady):
            self.native.evaluate(self.javascript_config(), trajectory(),
                                 SimpleNamespace(url='http://fixture.invalid', evaluate=missing_dom))

    def test_browser_disconnect_during_evaluation_discards_apparent_success(self):
        connected = [True]
        def evaluate(expression):
            connected[0] = False
            return '42'
        page = self.javascript_page(evaluate)
        page.context.browser.is_connected = lambda: connected[0]
        with self.assertRaisesRegex(EvaluationNotReady, 'disconnected'):
            self.native.evaluate(self.javascript_config(), trajectory(), page)

    def test_content_and_navigation_errors_are_unavailable(self):
        def fail(*args, **kwargs):
            raise ConnectionError('browser transport unavailable')
        page = SimpleNamespace(url='http://fixture.invalid', content=fail, goto=fail)
        with self.assertRaisesRegex(EvaluationNotReady, 'content failed'):
            self.native.evaluate(config('program_html'), trajectory(), page)
        task = config('program_html')
        task['eval']['program_html'][0]['url'] = 'http://goal.invalid'
        with self.assertRaisesRegex(EvaluationNotReady, 'goto failed'):
            self.native.evaluate(task, trajectory(), page)

    def test_async_page_methods_and_hidden_awaitable_results_are_rejected(self):
        async def asynchronous(expression):
            return '42'
        for evaluate in [asynchronous, lambda expression: asynchronous(expression)]:
            with self.assertRaises(EvaluationNotReady):
                self.native.evaluate(self.javascript_config(), trajectory(), self.javascript_page(evaluate))

    def test_native_prep_action_swallow_cannot_hide_transport_failure(self):
        task = self.javascript_config()
        task['eval']['program_html'][0]['prep_actions'] = ['document.button.click()']
        def evaluate(expression):
            if expression == '() => document.button.click()':
                raise ConnectionError('prep action lost browser connection')
            return '42'
        with self.assertRaisesRegex(EvaluationNotReady, 'native result discarded'):
            self.native.evaluate(task, trajectory(), self.javascript_page(evaluate))

    def test_native_numeric_contract(self):
        task = config()
        task['eval']['reference_answers'] = {'required_values': ['>= 40', '< 50']}
        self.assertTrue(self.native.evaluate(task, trajectory('42'))['success'])
        self.assertFalse(self.native.evaluate(task, trajectory('100'))['success'])

    def test_empty_or_unknown_evaluation_cannot_pass_vacuously(self):
        for eval_types in [[], ['unknown'], ['string_match', 'string_match']]:
            task = config()
            task['eval']['eval_types'] = eval_types
            with self.assertRaises(EvaluationNotReady):
                self.native.evaluate(task, trajectory())
        for refs in [{}, {'unknown_match': '42'}, {'must_include': []}]:
            task = config()
            task['eval']['reference_answers'] = refs
            with self.assertRaises(EvaluationNotReady):
                self.native.evaluate(task, trajectory())

    def test_missing_terminal_answer_and_stale_answer_fail_closed(self):
        for malformed in [[], trajectory()[:-1], [trajectory()[0], {'action_type': 'click'}],
                          trajectory() + [trajectory()[0], {'action_type': 'send_msg_to_user'}]]:
            with self.assertRaises(EvaluationNotReady):
                self.native.evaluate(config(), malformed)

    def test_fuzzy_and_na_fallback_disabled_before_any_provider_import(self):
        for value in [['the meaning of life'], 'N/A']:
            task = config()
            task['eval']['reference_answers'] = {'fuzzy_match': value}
            with self.assertRaisesRegex(EvaluationNotReady, 'GPT-4'):
                self.native.evaluate(task, trajectory())

    def test_explicit_fuzzy_adapter_has_per_evaluation_call_accounting(self):
        class Judge:
            enabled = True
            provider_calls = 0
            manifest = {'scope': 'synthetic injected judge, no network'}
            def fuzzy_match(self, *args, **kwargs):
                self.provider_calls += 1
                return 1.
            def ua_match(self, *args, **kwargs):
                self.provider_calls += 1
                return 1.
        judge = Judge()
        native = load_native_evaluator(SOURCE, fuzzy_judge=judge)
        task = config()
        task['eval']['reference_answers'] = {'fuzzy_match': ['42']}
        for _ in range(2):
            result = native.evaluate(task, trajectory())
            self.assertTrue(result['success'])
            self.assertEqual(result['model_provider_calls'], 1)
            self.assertEqual(result['fuzzy_judge_manifest'], judge.manifest)
        self.assertEqual(judge.provider_calls, 2)
        judge.enabled = False
        with self.assertRaises(EvaluationNotReady):
            load_native_evaluator(SOURCE, fuzzy_judge=judge)

    def test_missing_native_tokenizer_never_falls_back_to_substring_matching(self):
        task = config()
        task['eval']['reference_answers'] = {'must_include': ['0']}
        original_import = builtins.__import__

        def missing_nltk(name, *args, **kwargs):
            if name.startswith('nltk'):
                raise ImportError('Controlled absent tokenizer dependency')
            return original_import(name, *args, **kwargs)

        with patch('builtins.__import__', side_effect=missing_nltk):
            with self.assertRaisesRegex(EvaluationNotReady, 'no replacement tokenizer'):
                self.native.evaluate(task, trajectory('100'))

    def test_helper_and_unresolved_endpoint_fail_closed(self):
        task = config('program_html')
        task['eval']['program_html'][0]['url'] = 'func:shopping_get_latest_order_url()'
        with self.assertRaisesRegex(EvaluationNotReady, 'helper-function'):
            self.native.evaluate(task, trajectory(), SimpleNamespace(url='http://fixture.invalid'))
        task = config('url_match')
        task['eval']['reference_url'] = '__GITLAB__/goal'
        with self.assertRaisesRegex(EvaluationNotReady, 'Unresolved'):
            self.native.evaluate(task, trajectory(), SimpleNamespace(url='http://fixture.invalid'))

    def test_html_empty_or_multiple_requirements_rejected(self):
        for requirements in [{}, {'exact_match': '42', 'must_include': ['42']}]:
            task = config('program_html')
            task['eval']['program_html'][0]['required_contents'] = requirements
            with self.assertRaises(EvaluationNotReady):
                self.native.evaluate(task, trajectory(), SimpleNamespace(url='http://fixture.invalid'))


if __name__ == '__main__':
    unittest.main()
