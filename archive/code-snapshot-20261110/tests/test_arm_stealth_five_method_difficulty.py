"""Protect published bin boundaries and mean-of-ratios aggregation."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from report_arm_stealth_five_method_difficulty import difficulty_from_reference_length, summarize_repeats


class DifficultySummaryTests(unittest.TestCase):
    def test_published_human_step_boundaries(self):
        for length, band in [(1, 'easy'), (5, 'easy'), (6, 'medium'), (10, 'medium'),
                             (11, 'hard'), (12, 'hard')]:
            self.assertEqual(difficulty_from_reference_length(length), band)
        for invalid in [0, -1, 1.5, True]:
            with self.assertRaises(ValueError):
                difficulty_from_reference_length(invalid)

    def test_valid_only_averages_ratios_instead_of_pooling(self):
        groups = [dict(tasks=4, successes=s, valid=v, invalid=4-v, overall=s/4, valid_only=s/v)
                  for s, v in [(1, 1), (1, 2), (2, 4)]]
        result = summarize_repeats(groups)
        self.assertAlmostEqual(result['valid_only_mean'], 2/3)
        self.assertNotAlmostEqual(result['valid_only_mean'], 4/7)
        self.assertEqual(result['valid_by_repeat'], [1, 2, 4])


if __name__ == '__main__':
    unittest.main()
