"""Prevent greedy dedup from degenerating into all-neighbor suppression."""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from dedup_arm_task_pool import choose_block


class GreedyTests(unittest.TestCase):
    def test_rejected_middle_neighbor_does_not_reject_later_candidate(self):
        # A~B, B~C, A !~ C => greedy must retain A and C.
        sim = np.array([[1, .97, .90], [.97, 1, .97], [.90, .97, 1]])
        kept, decisions = choose_block(np.zeros(3), np.zeros(3, dtype=int), sim, .95, 10)
        self.assertEqual(kept, [0, 2])
        self.assertEqual(decisions[1]['nearest_index'], 10)

    def test_reference_rejection_cannot_suppress_following_candidate(self):
        kept, decisions = choose_block(np.array([.97, .2]), np.array([3, 4]), np.array([[1, .99], [.99, 1]]), .95, 8)
        self.assertEqual(kept, [1])
        self.assertEqual(decisions[0]['nearest_index'], 3)

    def test_threshold_is_inclusive(self):
        kept, _ = choose_block(np.array([.95, .94999]), np.array([0, 0]), np.eye(2), .95, 10)
        self.assertEqual(kept, [1])

    def test_matches_naive_greedy_on_random_normalized_vectors(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(75, 12)); x /= np.linalg.norm(x, axis=1, keepdims=True)
        expected = list(range(7))
        for i in range(7, len(x)):
            if max(x[i] @ x[expected].T) < .4:
                expected.append(i)
        actual = list(range(7))
        for start in range(7, len(x), 9):
            b = x[start:start+9]; sim = b @ x[actual].T
            idx = sim.argmax(axis=1)
            kept, _ = choose_block(sim.max(axis=1), np.asarray(actual)[idx], b @ b.T, .4, start)
            actual.extend(start+i for i in kept)
        self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
