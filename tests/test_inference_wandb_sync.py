import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from sync_sol_inference_wandb import snapshot


class ProgressMetrics(unittest.TestCase):
    def test_partial_results_preserve_all_three_denominators(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            results = root / 'shard-0/results'
            results.mkdir(parents=True)
            for i, (valid, reward) in enumerate(((True, 1), (True, 0), (False, None))):
                (results / f'{i}.json').write_text(json.dumps(dict(task_id=str(i), valid=valid, reward=reward)))
            values = snapshot(root)
            self.assertEqual(values['progress/completed_tasks'], 3)
            self.assertEqual(values['eval/invalid_tasks'], 1)
            self.assertEqual(values['eval/success_rate_all_scheduled'], 1 / 300)
            self.assertEqual(values['eval/success_rate_completed'], 1 / 3)
            self.assertEqual(values['eval/success_rate_valid'], 1 / 2)
            self.assertFalse(values['eval/complete'])

    def test_duplicate_tasks_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for segment in ('pilot-0', 'shard-0'):
                results = root / segment / 'results'
                results.mkdir(parents=True)
                (results / 'one.json').write_text(json.dumps(dict(task_id='same', valid=True, reward=1)))
            with self.assertRaisesRegex(ValueError, 'Duplicate task'):
                snapshot(root)


if __name__ == '__main__':
    unittest.main()
