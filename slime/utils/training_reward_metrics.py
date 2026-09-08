"""Training reward charts measured once per collected rollout, before PPO updates.

`train/reward` is the accepted training turns' unnormalized mean reward. Task
metrics count trajectories equally and retain the source's validity convention.
Never repeat these measurements on every optimizer step: PPO reuses the batch.
"""
import math

REWARD_AXIS = "train/reward_iteration"
REWARD_SOURCES = {
    "train/reward": "rollout/raw_reward_mean",
    "train/reward_std": "rollout/raw_reward_std",
    "train/judge_reward": "rollout/judge_reward_mean",
    "train/format_reward": "rollout/format_reward_mean",
    "train/task_reward_accepted": "rollout/task/accepted/reward_mean",
    "train/task_reward_completed": "rollout/task/completed/reward_mean",
    "train/task_success_rate": "rollout/task/completed/success_rate_all_completed",
    "train/task_success_rate_valid": "rollout/task/completed/success_rate_valid",
    "train/task_invalid_rate": "rollout/task/completed/invalid_rate",
}


def training_reward_metrics(metrics):
    """Copy available measured values; ignore optimizer and normalized rewards."""
    if "rollout/raw_reward_mean" not in metrics or "rollout/iteration" not in metrics:
        return {}
    result = {}
    for target, source in REWARD_SOURCES.items():
        value = metrics.get(source)
        if isinstance(value, (int, float)) and math.isfinite(value):
            result[target] = value
    if "train/reward" not in result:
        return {}
    result[REWARD_AXIS] = int(metrics["rollout/iteration"])
    return result


def define_training_reward_metrics(run):
    run.define_metric(REWARD_AXIS, hidden=True)
    for metric in REWARD_SOURCES:
        run.define_metric(metric, step_metric=REWARD_AXIS, step_sync=False)
