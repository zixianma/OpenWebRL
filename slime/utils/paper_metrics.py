"""Reward diagnostics on the paper's displayed iteration/percentage axes.

The collector mean is the closest documented released metric. The authors'
figure-export code is unavailable, so exact figure aggregation is unverified.
"""
import math

PAPER_AXIS = 'paper/iteration'
PAPER_METRICS = ('paper/training_reward_pct', 'paper/selected_batch_reward_pct')


def collection_paper_metrics(metrics, rollout_id):
    value = metrics.get('rollout/raw_reward_mean')
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return {}
    return {PAPER_AXIS: int(rollout_id), PAPER_METRICS[0]: 100 * value}


def trainer_paper_metrics(metrics, rollout_id):
    value = metrics.get('rollout/raw_reward')
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return {}
    return {PAPER_AXIS: int(rollout_id), PAPER_METRICS[1]: 100 * value}


def define_paper_metrics(run):
    run.define_metric(PAPER_AXIS, hidden=True)
    for metric in PAPER_METRICS:
        run.define_metric(metric, step_metric=PAPER_AXIS, step_sync=False)


def pending_paper_metrics(history, published):
    """Backfill each measured series independently, deduplicating replay rows."""
    existing = set(published)
    candidates = {}
    for row in history:
        for metric in PAPER_METRICS:
            if metric in row and PAPER_AXIS in row:
                existing.add((int(row[PAPER_AXIS]), metric))
        if 'rollout/raw_reward_mean' in row and 'rollout/iteration' in row:
            point = collection_paper_metrics(row, int(row['rollout/iteration']) - 1)
        elif 'rollout/raw_reward' in row and 'rollout/step' in row:
            # This bridge requires the baseline's wandb_always_use_train_step=False.
            point = trainer_paper_metrics(row, int(row['rollout/step']))
        else:
            continue
        for metric in PAPER_METRICS:
            if metric in point:
                candidates[(point[PAPER_AXIS], metric)] = point
    return [point for key, point in sorted(candidates.items()) if key not in existing]
