"""Synthetic episode-boundary contracts; no live tasks, browsers or providers."""
import base64
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from openwebrl.webprm_webarena import EvaluationNotReady, digest, load_native_evaluator
from openwebrl.webprm_webarena_episode import (
    EpisodeHooks, NativeTrajectory, TaskEnvelope, open_authenticated_browser,
    score_and_close,
)


PNG = base64.b64decode(
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/l9sAAAAASUVORK5CYII=')
SOURCE = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-baselines-20261009/sources/arbiter')


def raw_task():
    return dict(task_id=5, intent='Synthetic fixture task', start_url='__REDDIT__',
                sites=['reddit'], storage_state='./.auth/reddit_state.json',
                intent_template='Do not leak this annotation',
                future_private_field={'answer': 'unpublished reference'},
                eval={'eval_types': ['string_match'],
                      'reference_answers': {'exact_match': '42'}})


def task():
    return TaskEnvelope.from_release(raw_task(), {'reddit': 'http://fixture.invalid'})


def trajectory(answer='42'):
    result = NativeTrajectory({'url': 'http://fixture.invalid', 'text': 'Actual synthetic state'}, {})
    result.record_executed_terminal({'action_type': 'send_msg_to_user', 'answer': answer})
    return result


class FakeEnv:
    def __init__(self, events, fail_close=False):
        self.events = events
        self.fail_close = fail_close
        self.closed = False
        self.page = SimpleNamespace(url='http://fixture.invalid')

    def close(self):
        self.events.append('close')
        self.closed = True
        if self.fail_close:
            raise RuntimeError('Synthetic cleanup failure')


class BoundaryTests(unittest.TestCase):
    def setUp(self):
        self.network = patch('socket.socket.connect', side_effect=AssertionError('Network forbidden'))
        self.network.start()
        self.addCleanup(self.network.stop)
        self.temp = TemporaryDirectory(prefix='webprm-episode-test-', dir='/tmp')
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name) / 'evidence'

    def test_integer_identity_hashes_and_allowlisted_immutable_actor_view(self):
        raw = raw_task()
        original = deepcopy(raw)
        wrapped = TaskEnvelope.from_release(raw, {'reddit': 'http://fixture.invalid'})
        self.assertEqual(wrapped.execution_id, 'webarena_lite_005')
        self.assertEqual((self.output / wrapped.execution_id).name, 'webarena_lite_005')
        self.assertEqual(wrapped.released_task_id, 5)
        self.assertEqual(wrapped.raw_config_sha256, digest(original))
        self.assertEqual(wrapped.configured_config_sha256, digest(wrapped.native_config()))
        self.assertEqual(set(wrapped.actor_view), {'intent', 'start_url', 'sites'})
        with self.assertRaises(TypeError):
            wrapped.actor_view['intent'] = 'Changed'
        with self.assertRaises(TypeError):
            wrapped.actor_view['sites'][0] = 'shopping'
        raw['eval']['reference_answers']['exact_match'] = 'changed input'
        detached = wrapped.native_config()
        detached['eval']['reference_answers']['exact_match'] = 'changed output'
        self.assertEqual(wrapped.native_config()['eval']['reference_answers']['exact_match'], '42')
        self.assertNotIn('unpublished reference', repr(wrapped))

    def test_unsafe_or_ambiguous_task_ids_are_rejected(self):
        for ident in ('../../home', '5', True, -1, None):
            raw = raw_task()
            raw['task_id'] = ident
            with self.subTest(ident=ident), self.assertRaises(ValueError):
                TaskEnvelope.from_release(raw, {'reddit': 'http://fixture.invalid'})

    def test_default_callbacks_block_and_actor_callbacks_never_receive_eval(self):
        default = EpisodeHooks()
        for name in ('dependency', 'policy', 'selector', 'snapshot', 'evaluate'):
            with self.subTest(name=name), self.assertRaises(EvaluationNotReady):
                getattr(default, name)()
        seen = []
        def policy(view, observation):
            seen.append((view, observation))
            return ['candidate']
        def selector(view, observation, candidates):
            seen.append((view, observation))
            self.assertEqual(candidates, ['candidate'])
            return 0
        hooks = EpisodeHooks(policy=policy, selector=selector)
        candidates = hooks.propose(task(), {'text': 'current observation'})
        self.assertEqual(hooks.select(task(), {}, candidates), 0)
        for view, _ in seen:
            self.assertEqual(set(view), {'intent', 'start_url', 'sites'})
            self.assertNotIn('42', str(view))

    def test_auth_state_enters_context_before_browser_construction(self):
        events = []
        state = {'cookies': [{'name': 'synthetic', 'value': 'private'}], 'origins': []}
        def guard(wrapped):
            events.append('guard')
            return {'ready': True, 'reset_id': 'synthetic-reset-1', 'authenticated': True}
        def factory(**kwargs):
            events.append('factory')
            self.assertEqual(kwargs['pw_context_kwargs']['storage_state'], state)
            self.assertEqual(kwargs['pw_context_kwargs']['locale'], 'en-US')
            self.assertEqual(kwargs['task_kwargs'], {'deadline': 100, 'start_url': 'http://fixture.invalid', 'goal': 'Synthetic fixture task'})
            self.assertNotIn('eval', kwargs['task_kwargs'])
            kwargs['pw_context_kwargs']['storage_state']['cookies'][0]['value'] = 'factory mutation'
            return 'synthetic-browser'
        result = open_authenticated_browser(task(), hooks=EpisodeHooks(dependency=guard),
                    browser_factory=factory, task_entrypoint=object, action_mapping=lambda x: x,
                    storage_state=state, task_options={'deadline': 100},
                    browser_options={'pw_context_kwargs': {'locale': 'en-US'}})
        self.assertEqual(result, 'synthetic-browser')
        self.assertEqual(events, ['guard', 'factory'])
        self.assertEqual(state['cookies'][0]['value'], 'private')

    def test_missing_auth_or_reset_never_constructs_browser(self):
        factory_calls = []
        for proof, storage in [({'ready': True}, {'cookies': [], 'origins': []}),
                               ({'ready': True, 'reset_id': 'one'}, {'cookies': [], 'origins': []}),
                               ({'ready': True, 'reset_id': 'one', 'authenticated': True}, None)]:
            with self.subTest(proof=proof), self.assertRaises(EvaluationNotReady):
                open_authenticated_browser(task(), hooks=EpisodeHooks(dependency=lambda _: proof),
                    browser_factory=lambda **kw: factory_calls.append(kw), task_entrypoint=object,
                    action_mapping=lambda x: x, storage_state=storage)
        self.assertEqual(factory_calls, [])

    def test_browser_options_cannot_smuggle_alternative_task_or_auth(self):
        hooks = EpisodeHooks(dependency=lambda _: {'ready': True, 'reset_id': 'one', 'authenticated': True})
        common = dict(hooks=hooks, browser_factory=lambda **kwargs: None, task_entrypoint=object,
                      action_mapping=lambda x: x, storage_state={'cookies': [], 'origins': []})
        for options in ({'task_kwargs': {'goal': 'other'}}, {'pw_context_kwargs': {'storage_state': {}}}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                open_authenticated_browser(task(), browser_options=options, **common)
        with self.assertRaises(ValueError):
            open_authenticated_browser(task(), task_options={'goal': 'other'}, **common)

    def test_trajectory_records_actual_alternation_without_aliases(self):
        initial = {'text': 'before'}
        book = NativeTrajectory(initial, {'reset_id': 'one'})
        action = {'action_type': 'click', 'element_id': '7'}
        book.record_executed_action(action, {'text': 'after'}, {'execution_error': None})
        initial['text'] = 'changed'
        action['element_id'] = 'changed'
        book.record_executed_terminal({'action_type': 'send_msg_to_user', 'answer': ''})
        self.assertEqual(book.termination, 'stop')
        self.assertEqual(len(book.entries), 4)
        self.assertEqual(book.entries[0]['observation']['text'], 'before')
        self.assertEqual(book.entries[1]['element_id'], '7')
        self.assertEqual(book.entries[-1]['answer'], '')
        exposed = book.entries
        exposed[-1]['answer'] = 'forged'
        self.assertEqual(book.entries[-1]['answer'], '')
        with self.assertRaises(ValueError):
            book.record_executed_action({'action_type': 'click'}, {}, {})

    def test_half_steps_bad_termination_and_synthetic_answers_are_rejected(self):
        book = NativeTrajectory({}, {})
        with self.assertRaises(TypeError):
            book.record_executed_action({'action_type': 'click'}, None, {})
        self.assertEqual(len(book.entries), 1)
        with self.assertRaises(ValueError):
            book.record_executed_terminal({'action_type': 'send_msg_to_user'})
        with self.assertRaises(ValueError):
            book.record_executed_action({'action_type': 'send_msg_to_user', 'answer': 'x'}, {}, {})
        with self.assertRaises(ValueError):
            book.finish_at_state('browser_crash')
        book.finish_at_state('turn_limit')
        self.assertEqual(len(book.entries), 1)
        self.assertNotIn('answer', book.entries[-1])

    def snapshot(self, env):
        def capture(page):
            self.assertFalse(env.closed)
            env.events.append('snapshot')
            return {'url': page.url, 'screenshot': PNG}
        return capture

    def test_evidence_then_native_scoring_then_close_preserves_terminal_before_navigation(self):
        events = []
        env = FakeEnv(events)
        def evaluate(config, entries, page, *, termination):
            self.assertFalse(env.closed)
            self.assertEqual(termination, 'stop')
            self.assertEqual(entries[-1]['answer'], '42')
            self.assertEqual(config['eval']['reference_answers']['exact_match'], '42')
            self.assertEqual((self.output / 'terminal.png').read_bytes(), PNG)
            events.append('evaluate')
            page.url = 'http://fixture.invalid/evaluator-navigation'
            return {'valid': True, 'success': False, 'score': 0.0, 'source': {'fixture': True}}
        original_close = env.close
        def close():
            saved = json.loads((self.output / 'outcome-before-close.json').read_text())
            self.assertTrue(saved['valid'])
            original_close()
        env.close = close
        result = score_and_close(task(), trajectory(), env, self.output,
                    hooks=EpisodeHooks(snapshot=self.snapshot(env), evaluate=evaluate))
        self.assertEqual(events, ['snapshot', 'evaluate', 'close'])
        self.assertTrue(result['valid'])
        self.assertFalse(result['success'])
        self.assertEqual(result['score'], 0.0)
        self.assertEqual(result['terminal']['url'], 'http://fixture.invalid')
        self.assertEqual(result['cleanup_status'], 'completed')

    def test_missing_evaluator_is_invalid_and_close_failure_does_not_hide_it(self):
        env = FakeEnv([], fail_close=True)
        result = score_and_close(task(), trajectory(), env, self.output,
                                 hooks=EpisodeHooks(snapshot=self.snapshot(env)))
        self.assertFalse(result['valid'])
        self.assertIsNone(result['score'])
        self.assertEqual(result['error']['type'], 'EvaluationNotReady')
        self.assertEqual(result['error']['stage'], 'native_evaluation')
        self.assertEqual(result['cleanup_status'], 'error')
        self.assertTrue(env.closed)

    def test_snapshot_or_scoring_exception_stays_invalid_and_closes(self):
        for stage in ('snapshot', 'evaluate'):
            env = FakeEnv([])
            def fail(*args, **kwargs):
                raise RuntimeError('Synthetic browser/dependency failure')
            hooks = EpisodeHooks(snapshot=fail if stage == 'snapshot' else self.snapshot(env),
                                 evaluate=fail)
            result = score_and_close(task(), trajectory(), env, self.output / stage, hooks=hooks)
            self.assertFalse(result['valid'])
            self.assertIsNone(result['score'])
            self.assertTrue(env.closed)
            self.assertTrue((self.output / stage / 'result.json').is_file())

    def test_close_failure_preserves_completed_native_verdict(self):
        env = FakeEnv([], fail_close=True)
        result = score_and_close(task(), trajectory(), env, self.output,
                    hooks=EpisodeHooks(snapshot=self.snapshot(env),
                        evaluate=lambda *a, **k: {'valid': True, 'success': True, 'score': 1.0}))
        self.assertTrue(result['valid'])
        self.assertEqual(result['cleanup_status'], 'error')

    def test_unfinished_episode_and_reused_output_never_call_evaluator(self):
        calls = []
        hooks = EpisodeHooks(evaluate=lambda *a, **k: calls.append(1))
        env = FakeEnv([])
        first = score_and_close(task(), NativeTrajectory({}, {}), env, self.output, hooks=hooks)
        self.assertEqual(first['error']['stage'], 'trajectory')
        env2 = FakeEnv([])
        second = score_and_close(task(), trajectory(), env2, self.output, hooks=hooks)
        self.assertEqual(second['error']['stage'], 'evidence_directory')
        self.assertEqual(calls, [])
        self.assertTrue(env2.closed)

    def test_symlink_to_forbidden_home_is_rejected_without_writing(self):
        link = Path(self.temp.name) / 'forbidden-home-link'
        link.symlink_to('/gpfs/home/zixianma', target_is_directory=True)
        env = FakeEnv([])
        result = score_and_close(task(), trajectory(), env, link / 'not-created-by-test')
        self.assertFalse(result['valid'])
        self.assertEqual(result['error']['stage'], 'evidence_directory')
        self.assertEqual(result['error']['type'], 'ValueError')
        self.assertTrue(env.closed)

    @unittest.skipUnless(SOURCE.exists(), 'Pinned native evaluator source not installed')
    def test_actual_native_evaluator_scores_answer_and_rejects_missing_answer(self):
        native = load_native_evaluator(SOURCE)
        for label, book, expected in [('correct', trajectory('42'), True),
                                       ('wrong', trajectory('43'), False),
                                       ('no-answer', NativeTrajectory({}, {}), None)]:
            if label == 'no-answer':
                book.finish_at_state('turn_limit')
            env = FakeEnv([])
            result = score_and_close(task(), book, env, self.output / label,
                        hooks=EpisodeHooks(snapshot=self.snapshot(env), evaluate=native.evaluate))
            self.assertEqual(result['valid'], expected is not None)
            if expected is None:
                self.assertEqual(result['error']['type'], 'EvaluationNotReady')
                self.assertIsNone(result['score'])
            else:
                self.assertEqual(result['success'], expected)

    @unittest.skipUnless(SOURCE.exists(), 'Pinned native evaluator source not installed')
    def test_actual_native_final_state_url_html_and_closed_page(self):
        raw = raw_task()
        raw['eval'] = {'eval_types': ['url_match', 'program_html'],
                       'reference_url': '__REDDIT__',
                       'program_html': [{'url': 'last', 'locator': '',
                                         'required_contents': {'exact_match': '<body>42</body>'}}]}
        wrapped = TaskEnvelope.from_release(raw, {'reddit': 'http://fixture.invalid'})
        native = load_native_evaluator(SOURCE)
        for closed in (False, True):
            env = FakeEnv([])
            env.page.content = lambda: '<body>42</body>'
            env.page.is_closed = lambda: closed
            book = NativeTrajectory({'url': 'http://fixture.invalid'}, {})
            book.record_executed_action({'action_type': 'click'}, {'url': 'http://fixture.invalid'}, {})
            book.finish_at_state('turn_limit')
            result = score_and_close(wrapped, book, env, self.output / str(closed),
                        hooks=EpisodeHooks(snapshot=self.snapshot(env), evaluate=native.evaluate))
            self.assertEqual(result['valid'], not closed)
            if closed:
                self.assertIsNone(result['score'])
                self.assertEqual(result['error']['type'], 'EvaluationNotReady')
            else:
                self.assertTrue(result['success'])
                self.assertEqual(len(book.entries), 3)


if __name__ == '__main__':
    unittest.main()
