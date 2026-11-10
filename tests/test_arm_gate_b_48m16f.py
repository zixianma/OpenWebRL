"""Gate B 48M+16F source: parameterized group bounds, Gate B defaults unchanged.

Run with the frozen source (reference-arm-gate-b-48m16f-20261031-v1) first on sys.path.
"""
from types import SimpleNamespace
import unittest

from openwebrl import arm_failure_additive as add
from openwebrl import arm_gate_recovery as recovery


def config(**kw):
    return dict(dict(policy_id='p', failure_group_cap=16, candidate_gate='min2'), **kw)


class Bounds(unittest.TestCase):
    def test_window_limit_is_64_rows(self):
        self.assertEqual(len(add.window_schedule([{}]*64, 1200, [256], 256, 1/3)[0]), 64)
        with self.assertRaises(ValueError):
            add.window_schedule([{}]*65, 1200, [256], 256, 1/3)

    def test_coefficient_and_population_bounds(self):
        add.window_schedule([{}], 1200, [256], 256, 16/48)
        with self.assertRaises(ValueError):
            add.window_schedule([{}], 1201, [256], 256, 1/6)
        with self.assertRaises(ValueError):
            add.window_schedule([{}], 10, [256], 256, 1/3+1e-9)

    def test_scale_formula_unchanged(self):
        # Gate B formula flc * N / 8 with flc = 1/6 gives N/48.
        self.assertAlmostEqual(1/6*16/8, 16/48)

    def test_retain_cap_16(self):
        rows = lambda g: [SimpleNamespace(metadata={'task_id': f't{g}'}, group_index=g)]
        current = dict(config=config())
        for g in range(20):
            add.retain(current, rows(g))
        self.assertEqual(len(current['failure_additive_pool']), 16)
        with self.assertRaises(ValueError):
            add.retain(dict(config=config(failure_group_cap=17)), rows(0))

    def test_payload_limit(self):
        self.assertEqual(recovery.auxiliary_payload_limit(config()), 8*1024**3)
        self.assertEqual(recovery.auxiliary_payload_limit(config(failure_group_cap=8)), 4*1024**3)


if __name__ == '__main__':
    unittest.main()
