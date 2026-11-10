"""Regression checks against treating correlated repeats as independent tasks."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from analyze_arm_stealth_overall_ci import overall_intervals


class OverallClusterIntervalTests(unittest.TestCase):
    def test_copying_repeat_does_not_narrow_task_interval(self):
        one = np.random.default_rng(31).integers(0, 2, size=(40, 2, 1))
        single = overall_intervals(one, resamples=800, seed=7)
        triples = overall_intervals(np.repeat(one, 3, axis=2), resamples=800, seed=7)
        np.testing.assert_allclose(triples, single, atol=1e-12, rtol=0)

    def test_rejects_fractional_outcomes(self):
        with self.assertRaisesRegex(ValueError, 'binary'):
            overall_intervals(np.full((20, 1, 3), .5), resamples=100)


if __name__ == '__main__':
    unittest.main()
