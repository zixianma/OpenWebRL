"""Pinned native fuzzy prompts/verdicts with fake transport: no provider calls."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from openwebrl.webprm_webarena import EvaluationNotReady, load_native_evaluator
from openwebrl.webprm_webarena_fuzzy import (MODEL, BrokerBudgetedTransport,
    NativeFuzzyJudge, SingleAttemptTransport, NATIVE_PROTOCOL, REPLACEMENT_MODEL,
    REPLACEMENT_PROTOCOL, validate_fuzzy_configuration)

SOURCE = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-baselines-20261009/sources/arbiter')


def completion(text='correct', **changes):
    result = {'model': MODEL, 'choices': [{'finish_reason': 'stop',
              'message': {'role': 'assistant', 'content': text}}],
              'usage': {'prompt_tokens': 120, 'completion_tokens': 2}}
    result.update(changes)
    return result


class Budget:
    """Offline accounting double; real atomic monetary caps are caller-owned."""
    def __init__(self):
        self.reserved = []
        self.settled = []

    def reserve(self, record):
        self.reserved.append(deepcopy(record))
        return str(len(self.reserved))

    def settle(self, reservation, usage):
        self.settled.append((reservation, deepcopy(usage)))


@unittest.skipUnless(SOURCE.exists(), 'Pinned source checkout not installed')
class FuzzyTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp(prefix='webprm-native-fuzzy-test-', dir='/tmp'))
        self.budget = Budget()
        self.requests = []
        self.broker_configs = []
        self.response = completion()
        self.network = patch('socket.socket.connect', side_effect=AssertionError('Network forbidden'))
        self.network.start()
        self.addCleanup(self.network.stop)

    def send(self, request):
        self.assertTrue(self.budget.reserved)
        self.assertEqual(self.events()[-1]['event'], 'dispatch')
        self.requests.append(deepcopy(request))
        return deepcopy(self.response)

    def judge(self, **changes):
        options = dict(enabled=True, transport=SingleAttemptTransport(self.send),
                       budget=self.budget, journal_path=self.directory / 'calls.jsonl',
                       count_input_tokens=lambda request: 200, max_input_tokens=256,
                       max_calls=20, deadline=time.time() + 60)
        options.update(changes)
        return NativeFuzzyJudge(SOURCE, **options)

    def events(self):
        path = self.directory / 'calls.jsonl'
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def test_default_and_incomplete_configuration_never_dispatch(self):
        disabled = NativeFuzzyJudge(SOURCE)
        with self.assertRaises(EvaluationNotReady):
            disabled.fuzzy_match('answer', 'reference', 'question')
        self.assertFalse(disabled.enabled)
        self.assertEqual(disabled.provider_calls, 0)
        self.assertEqual(self.events(), [])
        for changes in [dict(transport=None), dict(budget=None), dict(count_input_tokens=None),
                        dict(max_input_tokens=0), dict(max_calls=True), dict(deadline=0),
                        dict(journal_path=None), dict(enabled=1)]:
            with self.subTest(changes=changes), self.assertRaises(EvaluationNotReady):
                self.judge(**changes)
        self.assertEqual(self.requests, [])

    def test_exact_native_request_and_durable_accounting_order(self):
        judge = self.judge()
        self.assertEqual(judge.fuzzy_match('answer', 'reference', 'question'), 1)
        expected_user = (
            'Help a teacher to grade the answer of a student given a question. Keep in mind that the student may use different phrasing or wording to answer the question. The goal is to evaluate whether the answer is semantically equivalent to the reference answer.\n'
            'question: question\nreference answer: reference\n'
            "all the string 'N/A' that you see is a special sequence that means 'not achievable'\n"
            "student answer: answer\nConclude the judgement by 'correct', 'incorrect', or 'partially correct'. Only output one of these options, and nothing else.")
        self.assertEqual(self.requests, [dict(model=MODEL, temperature=0, top_p=1.0,
            max_tokens=768, context_length=0, messages=[
                {'role': 'system', 'content': 'You are a helpful assistant'},
                {'role': 'user', 'content': expected_user}])])
        self.assertEqual([row['event'] for row in self.events()],
                         ['reserved_request', 'dispatch', 'response', 'settled', 'native_verdict'])
        self.assertEqual(judge.provider_calls, 1)
        self.assertEqual(self.budget.reserved[0]['request'], self.requests[0])
        self.assertEqual(self.budget.reserved[0]['max_output_tokens'], 768)
        self.assertEqual(self.budget.settled[0][1]['prompt_tokens'], 120)
        self.assertFalse(judge.manifest['alternate_model_allowed'])

    def test_native_verdict_semantics_are_preserved_including_substrings(self):
        judge = self.judge()
        # "not correct" is a native substring success, not a new adapter rule.
        for answer, expected in [('Correct', 1), ('INCORRECT', 0), ('partially correct', 0),
                                 ('not correct', 1)]:
            self.response = completion(answer)
            self.assertEqual(judge.fuzzy_match('p', 'r', 'q'), expected)
        for answer, expected in [('same', 1), ('different', 0), ('same but different', 0)]:
            self.response = completion(answer)
            self.assertEqual(judge.ua_match('p', 'r', 'q'), expected)
        self.response = completion('undetermined')
        with self.assertRaisesRegex(EvaluationNotReady, 'verdict'):
            judge.fuzzy_match('p', 'r', 'q')
        with self.assertRaisesRegex(EvaluationNotReady, 'verdict'):
            judge.ua_match('p', 'r', 'q')
        self.assertEqual(len(self.budget.settled), 9)

    def test_model_substitution_requires_explicit_protocol_and_approval(self):
        for changes in [dict(model=REPLACEMENT_MODEL),
                        dict(model=REPLACEMENT_MODEL, protocol=REPLACEMENT_PROTOCOL),
                        dict(model=REPLACEMENT_MODEL, model_change_approved=True),
                        dict(model=MODEL, protocol=REPLACEMENT_PROTOCOL),
                        dict(model='other', protocol=REPLACEMENT_PROTOCOL, model_change_approved=True),
                        dict(model_change_approved=1)]:
            with self.subTest(changes=changes), self.assertRaises(EvaluationNotReady):
                self.judge(**changes)
        self.assertEqual(self.requests, [])

    def test_approved_substitution_changes_only_model_and_preserves_native_verdict(self):
        native = self.judge()
        self.assertEqual(native.fuzzy_match('answer', 'reference', 'question'), 1)
        native_request = deepcopy(self.requests[-1])
        self.response = completion('partially correct', model=REPLACEMENT_MODEL)
        replacement = self.judge(model=REPLACEMENT_MODEL, protocol=REPLACEMENT_PROTOCOL,
                                 model_change_approved=True)
        self.assertEqual(replacement.fuzzy_match('answer', 'reference', 'question'), 0)
        expected = dict(native_request, model=REPLACEMENT_MODEL)
        self.assertEqual(self.requests[-1], expected)
        self.assertEqual(self.budget.reserved[-1]['request'], expected)
        self.assertEqual(replacement.manifest['model'], REPLACEMENT_MODEL)
        self.assertEqual(replacement.manifest['native_model'], MODEL)
        self.assertTrue(replacement.manifest['model_changed'])
        self.assertFalse(replacement.manifest['automatic_model_fallback'])
        # A provider returning the original or a floating alias cannot be
        # represented as the explicitly selected replacement model.
        self.response = completion()
        with self.assertRaises(EvaluationNotReady):
            replacement.fuzzy_match('p', 'r', 'q')

    def test_preflight_configuration_does_not_construct_transport(self):
        config = dict(fuzzy_enabled=True, fuzzy_model=MODEL, fuzzy_protocol=NATIVE_PROTOCOL,
                      fuzzy_max_input_tokens=8192, fuzzy_max_calls=1)
        self.assertFalse(validate_fuzzy_configuration(config)['model_changed'])
        for changes in [dict(fuzzy_enabled=False), dict(fuzzy_max_calls=True),
                        dict(fuzzy_max_input_tokens=0), dict(fuzzy_model='unknown'),
                        dict(fuzzy_model=REPLACEMENT_MODEL, fuzzy_protocol=REPLACEMENT_PROTOCOL)]:
            with self.subTest(changes=changes), self.assertRaises(EvaluationNotReady):
                validate_fuzzy_configuration(dict(config, **changes))
        replacement = dict(config, fuzzy_model=REPLACEMENT_MODEL, fuzzy_protocol=REPLACEMENT_PROTOCOL,
                           fuzzy_model_change_approved=True)
        self.assertTrue(validate_fuzzy_configuration(replacement)['model_changed'])
        self.assertEqual(self.requests, [])

    def test_caps_budget_denials_and_elapsed_deadline_stop_before_transport(self):
        for count in [257, -1, True]:
            judge = self.judge(count_input_tokens=lambda request, count=count: count)
            with self.assertRaisesRegex(EvaluationNotReady, 'input-token'):
                judge.fuzzy_match('p', 'r', 'q')
        judge = self.judge(deadline=time.time() + 10)
        with patch('openwebrl.webprm_webarena_fuzzy.time.time', return_value=time.time() + 20):
            with self.assertRaisesRegex(EvaluationNotReady, 'deadline'):
                judge.fuzzy_match('p', 'r', 'q')
        with patch.object(self.budget, 'reserve', side_effect=EvaluationNotReady('Shared cap exhausted')):
            with self.assertRaisesRegex(EvaluationNotReady, 'Shared cap'):
                self.judge().fuzzy_match('p', 'r', 'q')
        self.assertEqual(self.requests, [])
        self.assertEqual(self.budget.reserved, [])
        judge = self.judge(max_calls=1)
        judge.fuzzy_match('p', 'r', 'q')
        with self.assertRaisesRegex(EvaluationNotReady, 'call cap'):
            judge.ua_match('p', 'r', 'q')
        self.assertEqual(len(self.requests), 1)

    def test_transport_failure_retains_reservation_and_never_retries(self):
        calls = []
        def fail(request):
            calls.append(request)
            raise OSError('Synthetic connection failure')
        judge = self.judge(transport=SingleAttemptTransport(fail))
        with self.assertRaisesRegex(EvaluationNotReady, 'no retry'):
            judge.fuzzy_match('p', 'r', 'q')
        self.assertEqual(len(calls), 1)
        self.assertEqual(judge.provider_calls, 1)
        self.assertEqual(len(self.budget.reserved), 1)
        self.assertEqual(self.budget.settled, [])
        self.assertFalse(self.events()[-1]['automatic_retry'])
        self.assertNotIn('Synthetic connection', (self.directory / 'calls.jsonl').read_text())

    def test_journal_failure_prevents_dispatch(self):
        judge = self.judge()
        with patch('openwebrl.webprm_webarena_fuzzy._append', side_effect=OSError('Journal unavailable')):
            with self.assertRaises(OSError):
                judge.fuzzy_match('p', 'r', 'q')
        self.assertEqual(judge.provider_calls, 0)
        self.assertEqual(self.requests, [])
        self.assertEqual(len(self.budget.reserved), 1)

    def test_invalid_responses_cannot_become_scored_failures(self):
        variants = [completion(model='gpt-4o'), completion(choices=[]),
                    completion(choices=[{'finish_reason': 'length', 'message': {'content': 'correct'}}]),
                    completion(''), completion(usage={}),
                    completion(usage={'prompt_tokens': True, 'completion_tokens': 2})]
        for response in variants:
            with self.subTest(response=response):
                self.response = response
                judge = self.judge()
                with self.assertRaises(EvaluationNotReady):
                    judge.fuzzy_match('p', 'r', 'q')
                self.assertEqual(judge.provider_calls, 1)
        self.assertEqual(self.budget.settled, [])

    def test_usage_overrun_is_accounted_and_halts_further_calls(self):
        self.response = completion(usage={'prompt_tokens': 201, 'completion_tokens': 2})
        judge = self.judge()
        with self.assertRaisesRegex(EvaluationNotReady, 'reservation'):
            judge.fuzzy_match('p', 'r', 'q')
        self.assertEqual(self.budget.settled[0][1]['prompt_tokens'], 201)
        with self.assertRaisesRegex(EvaluationNotReady, 'halted'):
            judge.fuzzy_match('p', 'r', 'q')
        self.assertEqual(judge.provider_calls, 1)

    def test_accounting_failure_halts_further_calls(self):
        judge = self.judge()
        with patch.object(self.budget, 'settle', side_effect=OSError('Accounting unavailable')):
            with self.assertRaises(EvaluationNotReady):
                judge.fuzzy_match('p', 'r', 'q')
        with self.assertRaisesRegex(EvaluationNotReady, 'halted'):
            judge.fuzzy_match('p', 'r', 'q')
        self.assertEqual(judge.provider_calls, 1)

    def test_callbacks_cannot_mutate_the_native_request(self):
        def count(request):
            request['model'] = 'replacement'
            request['messages'].clear()
            return 200
        original = self.budget.reserve
        def reserve(record):
            result = original(record)
            record['request']['model'] = 'replacement'
            record['request']['messages'].clear()
            return result
        self.budget.reserve = reserve
        self.judge(count_input_tokens=count).fuzzy_match('p', 'r', 'q')
        self.assertEqual(self.requests[0]['model'], MODEL)
        self.assertEqual(len(self.requests[0]['messages']), 2)

    def test_private_nonsymlink_journal_and_no_provider_import(self):
        target = self.directory / 'target'
        target.mkdir()
        link = self.directory / 'linked'
        link.symlink_to(target, target_is_directory=True)
        for path in ['/gpfs/home/zixianma/forbidden-fuzzy.jsonl', 'relative.jsonl', link / 'calls.jsonl']:
            with self.subTest(path=path), self.assertRaises(EvaluationNotReady):
                self.judge(journal_path=path)
        before = set(sys.modules)
        self.judge().fuzzy_match('p', 'r', 'q')
        new_modules = set(sys.modules) - before
        self.assertFalse(any(name.startswith(('openai', 'webarbiter.src.llms')) for name in new_modules))

    def test_retries_async_and_optimized_native_execution_are_rejected(self):
        async def send(request):
            return completion()
        for options in [dict(send=send), dict(send=self.send, automatic_retries=1)]:
            with self.assertRaises(EvaluationNotReady):
                SingleAttemptTransport(**options)
        fake_sys = type('Sys', (), {'flags': type('Flags', (), {'optimize': 1})()})()
        with patch('openwebrl.webprm_webarena_fuzzy.sys', fake_sys):
            with self.assertRaisesRegex(EvaluationNotReady, 'optimization'):
                self.judge()

    def test_actual_native_string_evaluator_tracks_per_evaluation_calls(self):
        judge = self.judge()
        native = load_native_evaluator(SOURCE, fuzzy_judge=judge)
        task = {'intent': 'Synthetic question', 'eval': {'eval_types': ['string_match'],
                'reference_answers': {'fuzzy_match': ['Synthetic reference']}}}
        trajectory = [{'observation': {'text': 'Synthetic state'}, 'info': {}},
                      {'action_type': 'send_msg_to_user', 'answer': 'Synthetic answer'}]
        for text, expected in [('correct', True), ('incorrect', False)]:
            self.response = completion(text)
            result = native.evaluate(task, trajectory)
            self.assertTrue(result['valid'])
            self.assertEqual(result['success'], expected)
            self.assertEqual(result['model_provider_calls'], 1)
        task['eval']['reference_answers']['fuzzy_match'] = 'N/A'
        task['eval']['string_note'] = 'Synthetic impossible constraint'
        trajectory[-1]['answer'] = 'N/A'
        result = native.evaluate(task, trajectory)
        self.assertTrue(result['success'])
        self.assertEqual(result['model_provider_calls'], 0)
        trajectory[-1]['answer'] = 'The synthetic constraint prevents completion'
        self.response = completion('same')
        self.assertTrue(native.evaluate(task, trajectory)['success'])
        self.assertIn('actual unachievable reason: Synthetic impossible constraint',
                      self.requests[-1]['messages'][1]['content'])
        with self.assertRaisesRegex(EvaluationNotReady, 'disabled'):
            load_native_evaluator(SOURCE).evaluate(task, trajectory)

    def broker(self, alter=None, *, model=MODEL):
        from openwebrl.webprm_broker import payload_sha256
        def client(config, payload, journal, *, purpose):
            self.broker_configs.append(deepcopy(config))
            self.requests.append(deepcopy(payload))
            self.assertNotIn('context_length', payload)
            self.assertEqual(purpose, 'native_fuzzy')
            self.assertEqual(self.events()[-1]['event'], 'dispatch')
            response = completion(model=model)
            response['_webprm_broker'] = dict(schema_version=1, budget_owner='webprm_worker.completion',
                model=model, purpose=purpose, local=False, reservation_id=0, reservation_state='settled',
                usage=response['usage'], cell_id='synthetic', request_id='synthetic-request',
                identity_sha256='a' * 64, journal_sha256='b' * 64, budget_ledger_sha256='c' * 64,
                policy_sha256='d' * 64, payload_sha256=payload_sha256(payload))
            if alter:
                alter(response['_webprm_broker'])
            return response
        client._webprm_broker_client = True
        client._webprm_budget_owner = 'webprm_worker.completion'
        return BrokerBudgetedTransport(client, {'deadline': time.time() + 60}, self.directory / 'broker-client.jsonl')

    def test_broker_is_sole_budget_owner_and_http_payload_keeps_native_parameters(self):
        judge = self.judge(transport=self.broker(), budget=None)
        self.assertEqual(judge.fuzzy_match('café', '参考', 'Which résumé?'), 1)
        self.assertEqual(self.budget.reserved, [])
        self.assertEqual(self.budget.settled, [])
        self.assertEqual(judge.provider_calls, 1)
        self.assertEqual(self.requests[0]['max_tokens'], 768)
        self.assertEqual(self.requests[0]['temperature'], 0)
        self.assertEqual(self.requests[0]['top_p'], 1.0)
        self.assertEqual([row['event'] for row in self.events()],
                         ['broker_request', 'dispatch', 'response', 'broker_settled', 'native_verdict'])
        with self.assertRaisesRegex(EvaluationNotReady, 'sole'):
            self.judge(transport=self.broker())
        with self.assertRaisesRegex(EvaluationNotReady, 'broker client'):
            BrokerBudgetedTransport(lambda *a, **k: completion(), {}, self.directory / 'unused')

    def test_broker_receipt_mismatch_halts_without_duplicate_accounting(self):
        for change in [dict(reservation_state='reserved'), dict(payload_sha256='0' * 64),
                       dict(purpose='actor'), dict(local=True), dict(budget_ledger_sha256=None),
                       dict(usage={'prompt_tokens': 1, 'completion_tokens': 2})]:
            with self.subTest(change=change):
                broker = self.broker(lambda receipt: receipt.update(change))
                judge = self.judge(transport=broker, budget=None)
                with self.assertRaisesRegex(EvaluationNotReady, 'budget receipt'):
                    judge.fuzzy_match('p', 'r', 'q')
                with self.assertRaisesRegex(EvaluationNotReady, 'halted'):
                    judge.fuzzy_match('p', 'r', 'q')
                self.assertEqual(judge.provider_calls, 1)
        self.assertEqual(self.budget.reserved, [])
        self.assertEqual(self.budget.settled, [])

    def test_replacement_broker_receipt_must_record_actual_model(self):
        options = dict(budget=None, model=REPLACEMENT_MODEL,
                       protocol=REPLACEMENT_PROTOCOL, model_change_approved=True)
        judge = self.judge(transport=self.broker(model=REPLACEMENT_MODEL), **options)
        self.assertEqual(judge.fuzzy_match('p', 'r', 'q'), 1)
        self.assertEqual(self.requests[-1]['model'], REPLACEMENT_MODEL)
        bad = self.broker(lambda receipt: receipt.update(model=MODEL), model=REPLACEMENT_MODEL)
        with self.assertRaisesRegex(EvaluationNotReady, 'budget receipt'):
            self.judge(transport=bad, **options).fuzzy_match('p', 'r', 'q')
        self.assertEqual(self.budget.reserved, [])

    def test_each_broker_request_uses_earlier_deadline_without_mutating_cell(self):
        now = time.time()
        for cell_deadline, judge_deadline in [(now + 60, now + 10), (now + 10, now + 60)]:
            with self.subTest(cell_deadline=cell_deadline, judge_deadline=judge_deadline):
                transport = self.broker()
                transport.config['deadline'] = cell_deadline
                original = deepcopy(transport.config)
                judge = self.judge(transport=transport, budget=None, deadline=judge_deadline)
                self.assertEqual(judge.fuzzy_match('p', 'r', 'q'), 1)
                self.assertEqual(self.broker_configs[-1]['deadline'], min(cell_deadline, judge_deadline))
                self.assertEqual(transport.config, original)


if __name__ == '__main__':
    unittest.main()
