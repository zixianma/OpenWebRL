import unittest
from openwebrl.webprm_baselines import ARMS, candidate_pool, scalar_select, tournament, prepare_plan, paired_summary


class SelectionTests(unittest.TestCase):
    def test_analysis_keeps_invalid_denominator_and_paired_validity(self):
        rows = [dict(task_id=i, arm=a, success=int(a == 'shepherd8b'), valid=True)
                for i in ('a', 'b') for a in ARMS]
        rows[0]['valid'] = False
        result = paired_summary(rows, ['a', 'b'], 100)
        self.assertEqual(result['methods']['shepherd_unguided']['tasks'], 2)
        self.assertEqual(result['methods']['shepherd_unguided']['valid'], 1)
        first = result['comparisons'][0]
        self.assertEqual(first['delta_pp'], 100)
        self.assertEqual(first['common_valid_tasks'], 1)
        self.assertIsNone(first['relative_gain_percent'])
        with self.assertRaises(ValueError):
            paired_summary(rows[:-1], ['a', 'b'], 100)
        with self.assertRaises(ValueError):
            paired_summary(rows + [rows[0]], ['a', 'b'], 100)

    def test_frequency_pool_preserves_action_semantics(self):
        a, b = 'fill("1", "New York")', 'fill("1", "new york")'
        pool = candidate_pool([a, b, b, a, 'click("3")'])
        self.assertEqual([c['action'] for c in pool], [a, b, 'click("3")'])
        winner, _ = scalar_select(pool, lambda c: 1.0)
        self.assertEqual(winner['action'], a)

    def test_invalid_score_cannot_execute(self):
        with self.assertRaises(ValueError):
            scalar_select(candidate_pool(['click("1")']), lambda c: float('nan'))

    def test_tournament_byes_and_complete_comparison_accounting(self):
        pool = candidate_pool([str(i) for i in range(5)])
        winner, trace = tournament(pool, lambda a, b: int(int(b['action']) > int(a['action'])), 42)
        self.assertEqual(winner['action'], '4')
        self.assertEqual(len(trace), 4)
        with self.assertRaises(ValueError):
            tournament(pool, lambda a, b: 'Answer: A', 42)

    def test_plan_is_order_independent_and_not_authorization(self):
        tasks = [dict(task_id=str(i)) for i in range(300)]
        a = prepare_plan(tasks, 'source')
        b = prepare_plan(list(reversed(tasks)), 'source')
        self.assertEqual(a, b)
        self.assertEqual(a['pilot_episodes'], 72)
        self.assertEqual(a['status'], 'prepared_not_launchable')
        self.assertFalse(a['proposal']['full300_authorized'])
        with self.assertRaises(ValueError):
            prepare_plan(tasks[:-1], 'source')


if __name__ == '__main__':
    unittest.main()
