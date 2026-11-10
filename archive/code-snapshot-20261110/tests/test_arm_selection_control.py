import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1]/'scripts/prepare_arm_selection_control.py'
spec = importlib.util.spec_from_file_location('selection_control', path)
control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)


class SelectionControlTests(unittest.TestCase):
    def test_schedule_is_invariant_to_input_order_and_has_six_fresh_attempts(self):
        ids = [f'task{i}' for i in range(29)]
        schedule = control.schedule(ids)
        self.assertEqual(schedule, control.schedule(list(reversed(ids))))
        seeds = set()
        for row in schedule:
            self.assertEqual({r['mode'] for r in row['attempts']}, set(control.MODES))
            for attempt in row['attempts']:
                self.assertNotIn(attempt['seed'], seeds)
                seeds.add(attempt['seed'])
                self.assertEqual(attempt['phase'], 'selection_control')
        counts = [sum(r['shard'] == i for r in schedule) for i in range(4)]
        self.assertLessEqual(max(counts)-min(counts), 1)

    def test_random_control_matches_sites_without_excluding_known_successes(self):
        rows = {str(i): {'metadata': {'start_url': f'https://site{i%2}.test/page'}} for i in range(10)}
        selected = ['0', '2', '1']
        random_ids = control.matched_random(rows, list(rows), selected)
        self.assertEqual(control.Counter(control.host(rows[t]) for t in selected),
                         control.Counter(control.host(rows[t]) for t in random_ids))
        # Selecting every eligible task is valid and must keep all successes.
        self.assertEqual(set(control.matched_random(rows, list(rows), list(rows))), set(rows))

    def test_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError):
            control.schedule(['a', 'a'])

    def test_invalid_screen_excluded_and_native_format_failure_is_valid_failure(self):
        screen = [dict(task_id=t, attempt=i, valid=t != 'invalid', reward=-1 if i == 0 else 0,
                       policy_id=control.POLICY, judge_id=control.JUDGE, phase='screen', mode='actor',
                       selector_calls=0) for t in ['eligible', 'invalid'] for i in range(5)]
        guided = [dict(task_id='eligible', attempt=0, valid=True, reward=1, policy_id=control.POLICY,
                       judge_id=control.JUDGE, phase='guided', mode='arm', selector_revision=control.ARM_REVISION,
                       candidate_count=5, candidate_representation='full', actor_requests=5,
                       actor_output_tokens=100, selector_calls=1, selector_fallback_turns=0)]
        _, eligible, rescued = control.validate_records(screen, guided, ['eligible', 'invalid'])
        self.assertEqual(eligible, ['eligible'])
        self.assertEqual(rescued['successes'], 1)
        for changed in [guided + guided, [], [dict(guided[0], judge_id='other')]]:
            with self.assertRaises(ValueError):
                control.validate_records(screen, changed, ['eligible', 'invalid'])


if __name__ == '__main__':
    unittest.main()
