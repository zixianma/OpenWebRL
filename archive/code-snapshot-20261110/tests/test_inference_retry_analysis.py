import itertools
import math
import unittest

from openwebrl.inference_retry_analysis import summarize_retries, summarize_task


def episode(success, cost, steps=0):
    return dict(success=success, costs=dict(tokens=cost, steps=steps))


class RetryAnalysisTests(unittest.TestCase):
    def test_two_episode_exact_enumeration(self):
        rows = [episode(False, 10, 4), episode(True, 2, 1)]
        one, two = summarize_task(rows, [episode(True, 3)])['actor']
        self.assertEqual(one['success'], .5)
        self.assertEqual(one['run_all_costs'], dict(tokens=6, steps=2.5))
        self.assertEqual(one['oracle_stop_costs'], one['run_all_costs'])
        self.assertEqual(two['success'], 1)
        self.assertEqual(two['run_all_costs'], dict(tokens=12, steps=5))
        self.assertEqual(two['oracle_stop_costs'], dict(tokens=7, steps=3))

    def test_costly_failure_preserves_outcome_cost_dependence(self):
        rows = [episode(False, 100), episode(True, 1), episode(True, 1)]
        value = summarize_retries(rows, 3)
        self.assertEqual(value['run_all_costs']['tokens'], 102)
        self.assertAlmostEqual(value['oracle_stop_costs']['tokens'], 103 / 3)
        # mean-cost * mean-attempts would incorrectly give 34 * 4/3.
        self.assertNotAlmostEqual(value['oracle_stop_costs']['tokens'], 34 * 4 / 3)

    def test_no_successes_pay_all_and_real_zero_cost_survives(self):
        rows = [episode(False, 0), episode(False, 4), episode(False, 8)]
        for k in range(1, 4):
            value = summarize_retries(rows, k)
            self.assertEqual(value['success'], 0)
            self.assertEqual(value['run_all_costs']['tokens'], 4 * k)
            self.assertEqual(value['oracle_stop_costs'], value['run_all_costs'])
            self.assertEqual(value['oracle_stop_costs']['steps'], 0)

    def test_all_successes_pay_only_first_episode(self):
        rows = [episode(True, 0), episode(True, 4), episode(True, 8)]
        for k in range(1, 4):
            value = summarize_retries(rows, k)
            self.assertEqual(value['success'], 1)
            self.assertEqual(value['oracle_stop_costs']['tokens'], 4)
            self.assertEqual(value['run_all_costs']['tokens'], 4 * k)

    def test_brute_three_episode_sequences(self):
        rows = [episode(False, 3), episode(True, 7), episode(False, 11)]
        for k in range(1, 4):
            sequences = list(itertools.permutations(rows, k))
            all_cost, stop_cost, wins = [], [], []
            for sequence in sequences:
                all_cost.append(sum(row['costs']['tokens'] for row in sequence))
                stopping_index = next((i for i, row in enumerate(sequence)
                                       if row['success']), k - 1)
                stop_cost.append(sum(row['costs']['tokens']
                                     for row in sequence[:stopping_index + 1]))
                wins.append(any(row['success'] for row in sequence))
            value = summarize_retries(rows, k)
            self.assertAlmostEqual(value['success'], sum(wins) / len(wins))
            self.assertAlmostEqual(value['run_all_costs']['tokens'], sum(all_cost) / len(all_cost))
            self.assertAlmostEqual(value['oracle_stop_costs']['tokens'], sum(stop_cost) / len(stop_cost))

    def test_permutation_symmetry_and_guided_is_not_oracle_pooled(self):
        rows = [episode(False, 20), episode(True, 5), episode(False, 0)]
        guided = [episode(False, 2), episode(True, 8)]
        expected = summarize_task(rows, guided)
        self.assertEqual(expected['guided'], dict(success=.5, costs=dict(tokens=5, steps=0)))
        for reordered in itertools.permutations(rows):
            self.assertEqual(summarize_task(reordered, list(reversed(guided))), expected)

    def test_pass_at_k_matches_combinatorial_estimator_at_limit(self):
        rows = [episode(i < 3, i) for i in range(8)]
        for point in summarize_task(rows, [episode(True, 1)])['actor']:
            k = point['k']
            expected = 1 - math.comb(5, k) / math.comb(8, k) if k <= 5 else 1
            self.assertAlmostEqual(point['success'], expected)

    def test_invalid_sizes_and_k(self):
        for rows in ([], [episode(False, 1)] * 9, None):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                summarize_retries(rows, 1)
        for k in (0, 2, True, 1.0, None):
            with self.subTest(k=k), self.assertRaises(ValueError):
                summarize_retries([episode(False, 1)], k)

    def test_invalid_success_and_costs(self):
        for value in (0, 1, None, 'true'):
            with self.subTest(success=value), self.assertRaises(ValueError):
                summarize_retries([episode(value, 1)], 1)
        for value in (None, True, -1, float('nan'), float('inf'), -float('inf'), '1'):
            with self.subTest(cost=value), self.assertRaises(ValueError):
                summarize_retries([episode(False, value)], 1)

    def test_missing_or_different_metric_schemas_are_not_zero_filled(self):
        good = episode(False, 1)
        for bad in ({'success': False}, {'success': False, 'costs': {}},
                    {'success': False, 'costs': {'tokens': 1}}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                summarize_task([good], [bad])
        with self.assertRaises(ValueError):
            summarize_retries([good, {'success': True, 'costs': {'tokens': 1}}], 1)


if __name__ == '__main__':
    unittest.main()
