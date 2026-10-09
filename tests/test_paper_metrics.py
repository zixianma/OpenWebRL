import unittest
from slime.utils.paper_metrics import pending_paper_metrics


class PaperMetricsTest(unittest.TestCase):
    def test_percent_and_zero_based_axis_ignore_bad_trainer_iteration(self):
        history = [
            {'rollout/iteration': 1, 'rollout/step': 0, 'rollout/raw_reward_mean': .38},
            {'rollout/iteration': 0, 'rollout/step': 0, 'rollout/raw_reward': .39},
            {'rollout/iteration': 2, 'rollout/step': 1, 'rollout/raw_reward_mean': .33},
        ]
        self.assertEqual(pending_paper_metrics(history, set()), [
            {'paper/iteration': 0, 'paper/selected_batch_reward_pct': 39.},
            {'paper/iteration': 0, 'paper/training_reward_pct': 38.},
            {'paper/iteration': 1, 'paper/training_reward_pct': 33.},
        ])

    def test_replay_and_restart_do_not_duplicate_series(self):
        row = {'rollout/iteration': 1, 'rollout/raw_reward_mean': .38}
        points = pending_paper_metrics([row, row], set())
        self.assertEqual(len(points), 1)
        self.assertEqual(pending_paper_metrics([row] + points, set()), [])
        self.assertEqual(pending_paper_metrics([row], {(0, 'paper/training_reward_pct')}), [])

    def test_optimizer_updates_and_normalized_rewards_are_not_paper_rewards(self):
        self.assertEqual(pending_paper_metrics([
            {'train/step': 7, 'train/loss': .1},
            {'rollout/step': 0, 'rollout/rewards': -.1},
        ], set()), [])


if __name__ == '__main__':
    unittest.main()
