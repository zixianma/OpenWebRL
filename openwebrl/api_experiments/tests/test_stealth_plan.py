"""Offline checks for matched task selection and proposed allowance arithmetic."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'reproduce/stealth_plan.py'
SPEC = importlib.util.spec_from_file_location('stealth_plan', SCRIPT)
planner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(planner)


class StealthPlanTests(unittest.TestCase):
    def setUp(self):
        self.tasks = [dict(task_id=f't{i}', benchmark='online-mind2web',
                           start_url=f'https://host{i % 30}.example/task/{i}') for i in range(300)]

    def test_pairing_and_order_independence(self):
        plan = planner.build(self.tasks)
        self.assertEqual(plan, planner.build(list(reversed(self.tasks))))
        self.assertEqual(plan['selection']['distinct_hosts'], 20)
        self.assertEqual(len({row['task_id'] for row in plan['pairs']}), 20)
        for row in plan['pairs']:
            self.assertEqual(set(row['worker_arms']), {'luna_single_high', 'parallel5_select_high'})
        self.assertEqual(sum(row['worker_arms'][0] == 'luna_single_high' for row in plan['pairs']), 10)

    def test_allowances(self):
        limits = planner.build(self.tasks)['proposed_limits']
        self.assertEqual(limits['total_usd'], '16')
        self.assertEqual(limits['browser_creates_all_attempts'], 42)
        self.assertEqual(limits['generation_calls_all_attempts'], 4200)
        self.assertEqual(limits['model_http_calls_including_input_counters_all_attempts'], 8400)
        self.assertEqual(limits['terra_calls_all_attempts'], 40)
        self.assertEqual(limits['resources']['seconds_all_attempts'], 14400)
        self.assertIn('not a provider-enforced hard cap', limits['browser_allowance_kind'])

    def test_full_roster_coverage(self):
        plan = planner.build(self.tasks, count=300)
        self.assertEqual({p['task_id'] for p in plan['pairs']}, {t['task_id'] for t in self.tasks})

    def test_malformed_inputs(self):
        for kwargs in [dict(count=True), dict(count=301), dict(seed=True), dict(single_usd='NaN'),
                       dict(browser_usd='0'), dict(judge_usd='Infinity')]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                planner.build(self.tasks, **kwargs)
        for tasks in [self.tasks[:299], self.tasks + [self.tasks[0]],
                      [{**t, 'task_id': 'same'} for t in self.tasks],
                      [{**t, 'start_url': 'bad'} for t in self.tasks]]:
            with self.assertRaises(ValueError):
                planner.build(tasks)

    def test_cli_pins_identities_and_preserves_existing_output(self):
        with tempfile.TemporaryDirectory(prefix='stealth-plan-', dir='/tmp') as directory:
            root = Path(directory)
            tasks, roster, output = (root / name for name in ('tasks.json', 'roster.json', 'plan.json'))
            tasks.write_text(json.dumps(self.tasks))
            roster.write_text(json.dumps([dict(task_id=t['task_id']) for t in self.tasks]))
            command = [sys.executable, '-B', str(SCRIPT), '--tasks', str(tasks),
                       '--expected-roster', str(roster), '--output', str(output)]
            first = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            frozen = output.read_bytes()
            self.assertEqual(json.loads(frozen)['status'], 'prepared_not_authorized')
            self.assertIn('reference_roster_sha256', json.loads(frozen))
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
            self.assertEqual(output.read_bytes(), frozen)
            changed = list(self.tasks)
            changed[0] = {**changed[0], 'task_id': 'foreign-task'}
            tasks.write_text(json.dumps(changed))
            command[-1] = str(root / 'new-plan.json')
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
            self.assertFalse((root / 'new-plan.json').exists())


if __name__ == '__main__':
    unittest.main()
