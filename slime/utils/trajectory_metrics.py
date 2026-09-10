"""Trajectory-level telemetry. Never weight task outcomes by browser turn count.

Completed pre-filter groups include rejected groups, but exclude in-flight tasks
cancelled when the accepted batch fills. These are training diagnostics, not an
unbiased held-out evaluation. Missing/invalid rewards never count as failures in
the valid-only rate; success_rate_all_completed includes them in its denominator.
"""
from collections import defaultdict
import math


def trajectory_metrics(args, trajectories):
    trajectories = list(trajectories)
    rewards = []
    turns = []
    for trajectory in trajectories:
        samples = trajectory if isinstance(trajectory, list) else [trajectory]
        turns.append(len(samples))
        if not samples or any(s.remove_sample or getattr(s.status, 'value', s.status) == 'aborted' for s in samples):
            continue
        terminal = max(samples, key=lambda s: (s.metadata or {}).get('turn_index', 0))
        if terminal.reward is None:
            continue
        reward = terminal.get_reward_value(args)
        if reward is not None and math.isfinite(float(reward)):
            rewards.append(float(reward))
    total = len(trajectories)
    successes = sum(r == 1 for r in rewards)
    result = {'trajectories': total, 'valid_trajectories': len(rewards),
              'invalid_trajectories': total - len(rewards), 'successes': successes}
    if total:
        result.update(invalid_rate=(total-len(rewards))/total,
                      success_rate_all_completed=successes/total,
                      turns_mean=sum(turns)/total, turns_max=max(turns))
    if rewards:
        result.update(success_rate_valid=successes/len(rewards), reward_mean=sum(rewards)/len(rewards))
    return result


def flat_trajectory_metrics(args, samples):
    grouped = defaultdict(list)
    for ordinal, sample in enumerate(samples):
        metadata = sample.metadata or {}
        identity = metadata.get('trajectory_id', sample.index)
        if identity is None:
            identity = ('missing_id', ordinal)
        grouped[(sample.group_index, identity)].append(sample)
    return trajectory_metrics(args, grouped.values())


def collection_metrics(args, completed_groups, accepted_groups, elapsed):
    metrics = {}
    for name, groups in [('completed', completed_groups), ('accepted', accepted_groups)]:
        metrics.update({f'rollout/task/{name}/{k}': v for k, v in
                        trajectory_metrics(args, (t for g in groups for t in g)).items()})
    completed, accepted = len(completed_groups), len(accepted_groups)
    metrics.update({'rollout/sampling/completed_groups': completed,
                    'rollout/sampling/accepted_groups': accepted})
    if completed:
        metrics['rollout/sampling/acceptance_rate'] = accepted/completed
    if accepted:
        metrics['rollout/sampling/completed_groups_per_accepted_group'] = completed/accepted
        metrics['perf/seconds_per_accepted_group'] = elapsed/accepted
    if elapsed > 0:
        metrics['perf/completed_trajectories_per_second'] = sum(map(len, completed_groups))/elapsed
    from slime.utils.rollout_archive import maybe_archive_completed_groups
    metrics.update(maybe_archive_completed_groups(args, completed_groups, accepted_groups))
    return metrics
