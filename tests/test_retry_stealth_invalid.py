import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from retry_stealth_invalid import is_valid, select_retry, merge_records


def record(score, **kwargs):
    return dict(status='completed', remove_sample=False, reward={'judge': score}, **kwargs)


class RetryTests(unittest.TestCase):
    def test_unflagged_missing_verdict_is_invalid(self):
        self.assertFalse(is_valid(record(None)))
        self.assertTrue(is_valid(record(0)))

    def test_select_only_invalid_and_missing_preserving_task_order(self):
        tasks = [{'metadata': {'task_id': key}} for key in ['failure', 'invalid', 'missing', 'success']]
        records = {'failure': record(0), 'invalid': record(None), 'success': record(1)}
        self.assertEqual([t['metadata']['task_id'] for t in select_retry(tasks, records)], ['invalid', 'missing'])

    def test_merge_retains_valid_failure_and_counts_retry_failure(self):
        original = {'oldfailure': record(0), 'oldsuccess': record(1), 'bad': record(None)}
        retry = {'bad': record(0), 'missing': record(1)}
        result = merge_records(original, retry, ['bad', 'missing'], planned=4)
        self.assertEqual(result['successes'], 2)
        self.assertEqual(result['valid'], 4)
        self.assertEqual(result['overall_pct'], 50)
        self.assertEqual(original['bad']['reward']['judge'], None)

    def test_merge_cannot_replace_valid_failure_or_omit_a_retry(self):
        with self.assertRaisesRegex(ValueError, 'originally valid'):
            merge_records({'validfailure': record(0)}, {'validfailure': record(1)}, ['validfailure'], 1)
        with self.assertRaisesRegex(ValueError, 'exact selected'):
            merge_records({}, {'one': record(1)}, ['one', 'two'], 2)


if __name__ == '__main__':
    unittest.main()
