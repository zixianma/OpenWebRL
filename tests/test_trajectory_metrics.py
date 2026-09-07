import unittest
from types import SimpleNamespace
from slime.utils.trajectory_metrics import collection_metrics, flat_trajectory_metrics, trajectory_metrics


def turn(trajectory, index, reward, invalid=False):
    return SimpleNamespace(group_index=0, index=trajectory*1000+index,
        metadata={'trajectory_id': trajectory, 'turn_index': index},
        reward=reward, remove_sample=invalid, status='completed',
        get_reward_value=lambda args: reward)


class TrajectoryMetricsTest(unittest.TestCase):
    def test_unequal_lengths_and_shuffled_turns(self):
        samples = [turn(1, i, 1) for i in range(9)] + [turn(2, 0, 0)]
        metrics = flat_trajectory_metrics(None, samples[::-1])
        self.assertEqual(metrics['trajectories'], 2)
        self.assertEqual(metrics['success_rate_valid'], .5)
        self.assertEqual(metrics['turns_mean'], 5)

    def test_invalid_sibling_and_missing_reward(self):
        groups = [[turn(1, 0, 1, True), turn(1, 1, 1)], [turn(2, 0, None)], [turn(3, 0, 1)]]
        metrics = trajectory_metrics(None, groups)
        self.assertEqual(metrics['invalid_trajectories'], 2)
        self.assertEqual(metrics['success_rate_valid'], 1)
        self.assertAlmostEqual(metrics['success_rate_all_completed'], 1/3)

    def test_pre_filter_denominator(self):
        mixed = [[turn(1, 0, 1)], [turn(2, 0, 0)]]
        rejected = [[turn(3, 0, 0)], [turn(4, 0, 0)]]
        metrics = collection_metrics(None, [mixed, rejected], [mixed], 20)
        self.assertEqual(metrics['rollout/task/completed/success_rate_valid'], .25)
        self.assertEqual(metrics['rollout/task/accepted/success_rate_valid'], .5)
        self.assertEqual(metrics['rollout/sampling/acceptance_rate'], .5)
        self.assertEqual(metrics['perf/seconds_per_accepted_group'], 20)

    def test_no_valid_rewards_is_not_zero_success(self):
        metrics = trajectory_metrics(None, [[turn(1, 0, float('nan'))]])
        self.assertNotIn('success_rate_valid', metrics)
        self.assertEqual(metrics['invalid_rate'], 1)
        self.assertNotIn('reward_mean', trajectory_metrics(None, []))

if __name__ == '__main__':
    unittest.main()
