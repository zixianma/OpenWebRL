"""Curation boundaries that previously changed site coverage or leaked repeats."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location('rebuild', Path(__file__).resolve().parents[1] / 'scripts/rebuild_arm_task_pool.py')
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)
UPSTREAM = m.ROOT / 'curation-v3-20260929/upstream'


class TaskPoolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.upstream, _ = m.load_upstream(UPSTREAM)

    def test_same_instruction_different_site_is_preserved(self):
        rows = [dict(task_id=str(i), task_name='Find the contact page.', website=site) for i, site in enumerate(
            ['https://one.org', 'https://two.org', 'https://one.org'])]
        kept, dropped = m.dedup_exact(rows, self.upstream)
        self.assertEqual([r['task_id'] for r in kept], ['0', '1'])
        self.assertEqual(dropped[0]['representative_id'], '0')

    def test_different_paths_and_punctuation_are_not_exact_duplicates(self):
        rows = [dict(task_id=str(i), task_name=t, website=u) for i, (t, u) in enumerate([
            ('Find C++ docs', 'https://site.org/a'), ('Find C docs', 'https://site.org/a'),
            ('Find C++ docs', 'https://site.org/b')])]
        self.assertEqual(len(m.dedup_exact(rows, self.upstream)[0]), 3)

    def test_released_url_alias_normalization_used_before_dedup(self):
        rows = [dict(task_id=str(i), task_name='Find books', website=u) for i, u in enumerate(
            ['https://www.books.google.com', 'https://books.google.com'])]
        self.assertEqual(len(m.dedup_exact(rows, self.upstream)[0]), 1)

    def test_released_blacklist_keeps_exact_host_semantics(self):
        rows = [dict(task_id=str(i), website=u) for i, u in enumerate([
            'https://www.books.google.com', 'https://example.org', 'https://www.example.org'])]
        kept, dropped = self.upstream.filter_records_by_host_blacklist(rows, {'books.google.com', 'example.org'})
        self.assertEqual([r['task_id'] for r in kept], ['2'])
        self.assertEqual(len(dropped), 2)

    def test_frozen_outputs_cannot_be_replaced(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'rows.jsonl'
            m.frozen_rows(p, [dict(task_id='1')])
            with self.assertRaises(FileExistsError):
                m.frozen_rows(p, [dict(task_id='2')])


if __name__ == '__main__':
    unittest.main()
