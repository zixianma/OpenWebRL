import unittest

from scripts.sync_training_rewards import pending_rewards
from slime.utils.training_reward_metrics import REWARD_AXIS, training_reward_metrics


class TrainingRewardMetricsTest(unittest.TestCase):
    def test_reward_preserves_turn_and_task_weighting(self):
        result = training_reward_metrics({
            "rollout/iteration": 2, "rollout/raw_reward_mean": 0.3,
            "rollout/task/accepted/reward_mean": 0.6, "rollout/rewards": -0.1,
        })
        self.assertEqual(result, {
            REWARD_AXIS: 2, "train/reward": 0.3, "train/task_reward_accepted": 0.6,
        })
        self.assertEqual(training_reward_metrics({"train/step": 3, "train/abs_advantage": 0.8}), {})

    def test_replay_duplicate_and_optimizer_rows_do_not_create_reward_points(self):
        history = [
            {"rollout/iteration": 1, "rollout/raw_reward_mean": 0.3,
             "rollout/task/accepted/reward_mean": 0.6},
            {"rollout/iteration": 1, "rollout/raw_reward_mean": 0.3},
            {"train/step": 0, "train/loss": 0.1},
        ]
        points = pending_rewards(history, set())
        self.assertEqual(len(points), 1)
        self.assertEqual(points[0]["train/task_reward_accepted"], 0.6)
        self.assertEqual(pending_rewards(history, {1}), [])
        self.assertEqual(pending_rewards(history + points, set()), [])

    def test_new_rollout_adds_one_point(self):
        history = [
            {"rollout/iteration": 1, "rollout/raw_reward_mean": 0.3},
            {"rollout/iteration": 2, "rollout/raw_reward_mean": 0.5},
        ]
        self.assertEqual(pending_rewards(history, {1}), [{REWARD_AXIS: 2, "train/reward": 0.5}])


if __name__ == "__main__":
    unittest.main()
