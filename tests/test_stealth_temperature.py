"""Temperature identity must agree with the actual frozen generation code."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import evaluate_stealth_temperature as m


class TemperatureTest(unittest.TestCase):
    def plan(self, declared, actual, requested):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            (source / 'openwebrl').mkdir()
            (source / 'reference_manifest.json').write_text(json.dumps(
                {'paper_om2w_benchmark': {'temperature': declared}}))
            (source / 'openwebrl/eval_benchmark.py').write_text(
                f'SAMPLING = dict(temperature={actual}, top_p=.95, top_k=20)\n')
            base = {'completed_training_iterations': 80, 'environment': {'JUDGE_MODEL': 'o4-mini'},
                    'command': ['bash', 'launch', '--wandb-group', 'old']}
            with patch.object(m, 'build_plan', return_value=base):
                return m.temperature_plan(source, 'checkpoint', 'output', '42', requested)

    def test_temperature_changes_have_distinct_and_correct_identity(self):
        zero = self.plan(0, 0, 0)
        sampled = self.plan(.6, .6, .6)
        self.assertNotEqual(zero['wandb_run_id'], sampled['wandb_run_id'])
        for plan in (zero, sampled):
            self.assertEqual(plan['environment']['WANDB_RUN_ID'], plan['wandb_run_id'])
            self.assertEqual(plan['environment']['JUDGE_MODEL'], 'o4-mini')
        self.assertIn('actor T=0,', zero['protocol'])
        self.assertIn('actor T=0.6,', sampled['protocol'])

    def test_declared_and_executed_temperature_must_both_match(self):
        for declared, actual in [(0, .6), (.6, 0), (.6, .6)]:
            with self.subTest(declared=declared, actual=actual):
                with self.assertRaisesRegex(ValueError, 'differs from the frozen source'):
                    self.plan(declared, actual, 0)


if __name__ == '__main__':
    unittest.main()
