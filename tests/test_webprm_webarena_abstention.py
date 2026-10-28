"""Prospective scoring amendment; actual pinned evaluators, no provider calls."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from openwebrl.webprm_webarena import (load_native_evaluator, EvaluationNotReady,
    ABSTENTION_EVALUATOR_PROTOCOL, NATIVE_EVALUATOR_PROTOCOL)
from openwebrl.webprm_webarena_worker import (protocol_definition, validate_protocol_contract,
    NA_TURN_LIMIT, ENVELOPE_NO_SEARCH_PROMPT)

SOURCE = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-baselines-20261009/sources/arbiter')


class Judge:
    enabled = True
    manifest = {'offline_double': True}

    def __init__(self, score=1):
        self.provider_calls = 0
        self.score = score
        self.calls = []

    def fuzzy_match(self, pred, ref, intent):
        self.provider_calls += 1
        self.calls.append(('fuzzy', pred, ref, intent))
        return self.score

    def ua_match(self, pred, ref, intent):
        self.provider_calls += 1
        self.calls.append(('ua', pred, ref, intent))
        return self.score


def task(reference=None):
    return dict(intent='How many qualifying items?', eval=dict(eval_types=['string_match'],
        reference_answers=dict(fuzzy_match=['0'] if reference is None else reference),
        string_note='The requested operation is unavailable.'))


def trajectory(answer):
    return [dict(observation=dict(text='Synthetic controlled page'), info={}),
        dict(action_type='send_msg_to_user', answer=answer)]


@unittest.skipUnless(SOURCE.exists(), 'Pinned source unavailable')
class AbstentionTests(unittest.TestCase):
    def setUp(self):
        self.network = patch('socket.socket.connect', side_effect=AssertionError('Network forbidden'))
        self.network.start(); self.addCleanup(self.network.stop)
        self.judge = Judge()
        self.native = load_native_evaluator(SOURCE, fuzzy_judge=self.judge)
        self.guarded = load_native_evaluator(SOURCE, fuzzy_judge=self.judge,
            scoring_protocol=ABSTENTION_EVALUATOR_PROTOCOL)

    def test_legacy_result_and_judge_request_are_unchanged(self):
        result = self.native.evaluate(task(), trajectory('N/A'))
        self.assertEqual(result['score'], 1)
        self.assertNotIn('native_score', result)
        self.assertNotIn('evaluator_protocol', result)
        self.assertEqual(self.judge.calls, [('fuzzy', 'n/a', '0', 'How many qualifying items?')])

    def test_raw_native_positive_retained_and_effective_abstention_is_failure(self):
        saved_task, saved_trajectory = task(), trajectory('N/A')
        originals = deepcopy((saved_task, saved_trajectory))
        result = self.guarded.evaluate(saved_task, saved_trajectory, termination='turn_limit')
        self.assertEqual(result['native_score'], 1)
        self.assertTrue(result['native_success'])
        self.assertEqual(result['native_evaluator_details'], [{'eval_type': 'string_match', 'score': 1}])
        self.assertEqual(result['score'], 0)
        self.assertFalse(result['success'])
        self.assertTrue(result['valid'])
        self.assertEqual(result['model_provider_calls'], 1)
        self.assertEqual(len(result['scoring_amendments']), 1)
        self.assertEqual((saved_task, saved_trajectory), originals)

    def test_same_rule_for_actor_final_answer_and_protocol_fallback(self):
        for termination in ('stop', 'turn_limit'):
            for answer in ('N/A', 'n/a', 'N/a', '"N/A"', "'N/A'"):
                with self.subTest(termination=termination, answer=answer):
                    result = self.guarded.evaluate(task(), trajectory(answer), termination=termination)
                    self.assertEqual((result['score'], result['native_score']), (0, 1))

    def test_ordinary_non_numeric_references_receive_the_same_guard(self):
        result = self.guarded.evaluate(task(['blue triangle', 'blue triangular object']), trajectory('N/A'))
        self.assertEqual((result['score'], result['native_score']), (0, 1))

    def test_no_silent_broadening_of_native_normalization_or_semantics(self):
        # Exact native normalization lowercases and removes paired outer quotes;
        # it does not trim whitespace, rewrite phrases, or infer numeric answers.
        for answer in ('0', 'zero', 'There are no qualifying items.', '1', ' N/A ', 'N/A because unavailable'):
            with self.subTest(answer=answer):
                result = self.guarded.evaluate(task(), trajectory(answer))
                self.assertEqual(result['score'], result['native_score'])
                self.assertEqual(result['scoring_amendments'], [])

    def test_native_negative_stays_negative_and_raw_is_saved(self):
        self.judge.score = 0
        result = self.guarded.evaluate(task(), trajectory('N/A'))
        self.assertEqual((result['score'], result['native_score']), (0, 0))
        self.assertEqual(len(result['scoring_amendments']), 1)

    def test_legitimate_unachievable_exact_reference_does_not_call_judge(self):
        for answer in ('N/A', 'n/a', '"N/A"'):
            result = self.guarded.evaluate(task('N/A'), trajectory(answer))
            self.assertEqual((result['score'], result['native_score']), (1, 1))
            self.assertEqual(result['model_provider_calls'], 0)
            self.assertEqual(result['scoring_amendments'], [])
        self.assertEqual(self.judge.calls, [])

    def test_legitimate_unachievable_explanation_keeps_native_ua_route(self):
        result = self.guarded.evaluate(task('N/A'), trajectory('The requested operation is unavailable.'))
        self.assertEqual(result['score'], 1)
        self.assertEqual(self.judge.calls[0][0], 'ua')
        self.assertEqual(result['scoring_amendments'], [])

    def test_explicit_na_among_ordinary_reference_alternatives_is_not_overridden(self):
        for refs in (['N/A'], ['0', 'n/a'], ['"N/A"']):
            result = self.guarded.evaluate(task(refs), trajectory('N/A'))
            self.assertEqual(result['score'], 1)
            self.assertEqual(result['scoring_amendments'], [])

    def test_url_and_dom_fuzzy_use_native_results_even_with_na_final_answer(self):
        page = SimpleNamespace(url='http://fixture.invalid/goal', content=lambda: 'N/A')
        for kind in ('url_match', 'program_html'):
            sample = task()
            sample['eval'].update(eval_types=[kind], reference_url=page.url,
                program_html=[dict(url='last', locator='', required_contents=dict(fuzzy_match=['0']))])
            result = self.guarded.evaluate(sample, trajectory('N/A'), page)
            self.assertEqual(result['score'], 1)
            self.assertEqual(result['scoring_amendments'], [])
        self.assertEqual(self.judge.calls[-1][0:3], ('fuzzy', 'N/A', '0'))

    def test_combined_native_component_product_is_retained_separately(self):
        sample = task()
        sample['eval'].update(eval_types=['string_match', 'url_match'], reference_url='http://fixture.invalid/goal')
        result = self.guarded.evaluate(sample, trajectory('N/A'), SimpleNamespace(url='http://fixture.invalid/goal'))
        self.assertEqual((result['score'], result['native_score']), (0, 1))
        self.assertEqual(result['evaluator_details'][1], result['native_evaluator_details'][1])

    def test_v4_contract_explicit_and_unknown_or_legacy_guard_rejected(self):
        with self.assertRaises(EvaluationNotReady):
            load_native_evaluator(SOURCE, scoring_protocol='implicit_fix')
        with self.assertRaises(EvaluationNotReady):
            protocol_definition(SOURCE, scoring_protocol=ABSTENTION_EVALUATOR_PROTOCOL)
        contract = protocol_definition(SOURCE, turn_limit=NA_TURN_LIMIT,
            no_search_prompt=ENVELOPE_NO_SEARCH_PROMPT, scoring_protocol=ABSTENTION_EVALUATOR_PROTOCOL)
        sample = dict(arbiter_source=str(SOURCE), turn_limit_protocol=NA_TURN_LIMIT,
            no_search_prompt_version=ENVELOPE_NO_SEARCH_PROMPT,
            evaluator_protocol=ABSTENTION_EVALUATOR_PROTOCOL, protocol=contract['identity'], protocol_contract=contract)
        self.assertEqual(contract['identity'], 'webarena_table5_viewer_reconstruction_v4')
        self.assertEqual(validate_protocol_contract(sample), contract)
        with self.assertRaises(EvaluationNotReady):
            validate_protocol_contract(dict(sample, evaluator_protocol=NATIVE_EVALUATOR_PROTOCOL))


if __name__ == '__main__':
    unittest.main()
