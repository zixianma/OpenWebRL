"""Verify public data pins and exact campaign source identities without private data."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('api_public_tasks', ROOT / 'reproduce/tasks.py')
TASKS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TASKS)


class PublicTaskTests(unittest.TestCase):
    def test_public_pins_and_benchmark_reference_mapping(self):
        for benchmark, count in [('online-mind2web', 300), ('webvoyager', 595), ('deepshop', 150)]:
            source = TASKS.dataset_rows(benchmark)
            self.assertEqual(len(source), count)
            tasks = TASKS.load_tasks(benchmark)
            self.assertEqual([x['task_id'] for x in tasks], list(source))
            for task in tasks:
                if benchmark != 'online-mind2web':
                    self.assertEqual(task['native_metadata'], source[task['native_task_id']])

    def test_campaign_order_is_preserved_when_selecting_ids(self):
        rows = TASKS.load_tasks('browsercode_claude')
        selected = [rows[-1]['task_id'], rows[0]['task_id']]
        self.assertEqual([x['task_id'] for x in TASKS.load_tasks('browsercode_claude', selected)], selected)
        with self.assertRaises(ValueError):
            TASKS.load_tasks('browsercode_claude', selected + selected)
        with self.assertRaises(ValueError):
            TASKS.load_tasks('browsercode_claude', ['missing-task'])

    def test_modified_benchmark_is_rejected_before_normalization(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / 'openwebrl/data/eval'
            directory.mkdir(parents=True)
            (directory / 'online-mind2web.jsonl').write_text('{}\n')
            with self.assertRaisesRegex(ValueError, 'benchmark source changed'):
                TASKS.dataset_rows('online-mind2web', root)
