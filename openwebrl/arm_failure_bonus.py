"""Opt-in ARM supervision on valid zero-outcome actor groups.

The actor still executes candidate zero. Terminal rewards and the native GRPO
normalizer are unchanged; only the dynamic admission rule is extended.
"""
from collections import Counter
import math

from slime.rollout.filter_hub.base_types import DynamicFilterOutput
from slime.rollout.filter_hub.dynamic_sampling_filters import check_reward_nonempty_nonzero_std


def _rows(trajectory):
    return trajectory if isinstance(trajectory, list) else [trajectory]


def invalid_trajectory_reason(args, trajectory, current):
    rows = _rows(trajectory)
    if not rows:
        return 'empty_trajectory'
    terminal = rows[-1]
    metadata = terminal.metadata or {}
    reason = metadata.get('terminate_reason')
    status = getattr(terminal.status, 'value', terminal.status)
    reward = metadata.get('reward', {})
    if any(s.remove_sample or (s.metadata or {}).get('arm_calibration_excluded') for s in rows):
        return 'removed_or_excluded'
    if any(s.reward is None or not math.isfinite(s.get_reward_value(args))
           or s.get_reward_value(args) != 0 for s in rows):
        return 'missing_or_nonzero_reward'
    if any(s.response_length <= 0 or (s.loss_mask is not None and not any(s.loss_mask)) for s in rows):
        return 'no_trainable_response'
    ids = {str((s.metadata or {}).get('trajectory_id', s.index)) for s in rows}
    tasks = {str((s.metadata or {}).get('task_id', '')) for s in rows}
    if len(ids) != 1 or len(tasks) != 1 or not next(iter(tasks)):
        return 'inconsistent_trajectory_identity'
    if next(iter(tasks)) in current.get('excluded_ids', set()):
        return 'evaluation_overlap'
    if metadata.get('judge_timeout') or reward.get('judge_timeout'):
        return 'judge_timeout'
    if reward.get('combined') != 0 or reward.get('judge_prompt_variant') != 'action_history':
        return 'missing_native_reward_provenance'
    # Baseline max-step failures are deterministic zeros, not judge verdicts.
    if status == 'failed' and reason == 'max_steps_exhausted':
        return None
    if status != 'completed' or reason != 'task_completed':
        return 'invalid_termination'
    # Native API errors and unparsable verdicts can also return zero. Do not
    # reclassify them as policy failures in the newly admitted pool.
    if reward.get('judge') != 0 or 'NOT SUCCESS' not in str(reward.get('judge_text', '')):
        return 'unverified_judge_failure'
    return None


def invalid_group_reason(args, samples, current):
    """Validate terminal failures independently of the availability/sign of labels."""
    if len(samples) != 5:
        return 'requires_five_trajectories'
    identities = [str((_rows(t)[-1].metadata or {}).get('trajectory_id', _rows(t)[-1].index))
                  for t in samples if _rows(t)]
    if len(set(identities)) != 5:
        return 'duplicate_or_empty_trajectory'
    rows = [s for t in samples for s in _rows(t)]
    if len({s.group_index for s in rows}) != 1 or len({(s.metadata or {}).get('task_id') for s in rows}) != 1:
        return 'inconsistent_group_identity'
    for trajectory in samples:
        if reason := invalid_trajectory_reason(args, trajectory, current):
            return reason
    return None


def filter_groups(args, samples, **kwargs):
    from openwebrl.arm_turn_bonus import state, unit_bonus
    current = state(args)
    if not current['config'].get('admit_all_failure_groups'):
        raise ValueError('All-failure filter requires an explicit experiment configuration')
    native = check_reward_nonempty_nonzero_std(args, samples, **kwargs)
    if native.keep or native.reason != 'zero_std_0.0':
        return native
    if reason := invalid_group_reason(args, samples, current):
        return DynamicFilterOutput(keep=False, reason='arm_failure_' + reason)
    rows = [s for t in samples for s in _rows(t)]
    # Admit based on usable supervision, never on whether ARM picked the actor.
    units = [unit_bonus(s, current['config']['policy_id']) for s in rows]
    if not any(units):
        return DynamicFilterOutput(keep=False, reason='arm_failure_no_usable_labels')
    for s in rows:
        s.metadata['arm_all_failure_group'] = True
    return DynamicFilterOutput(keep=True, reason='arm_all_failure', samples=samples)


def annotate_calibration(report, samples, normalized_rewards, policy_id):
    from openwebrl.arm_turn_bonus import panel, unit_bonus
    failed = [(s, a) for s, a in zip(samples, normalized_rewards)
              if (s.metadata or {}).get('arm_all_failure_group')]
    mixed = [(s, a) for s, a in zip(samples, normalized_rewards)
             if not (s.metadata or {}).get('arm_all_failure_group')]
    if any(a != 0 for _, a in failed):
        raise ValueError('All-failure native outcome advantages must be exactly zero')
    units = [unit_bonus(s, policy_id) for s, _ in failed]
    report.update(
        all_failure_experiment=True,
        all_failure_groups=len({s.group_index for s, _ in failed}),
        all_failure_rows=len(failed),
        all_failure_usable_labels=sum(u != 0 for u in units),
        all_failure_positive_turns=sum(u > 0 for u in units),
        all_failure_negative_turns=sum(u < 0 for u in units),
        all_failure_zero_signal_turns=sum(u == 0 for u in units),
        all_failure_turn_fraction=len(failed)/len(samples) if samples else 0.,
        all_failure_bonus_rms=.5*math.sqrt(sum(u*u for u in units)/len(units)) if units else 0.,
        mixed_outcome_groups=len({s.group_index for s, _ in mixed}),
        mixed_outcome_panel=panel([s for s, _ in mixed], [a for _, a in mixed], policy_id) if mixed else None,
        all_failure_termination_counts=dict(Counter((s.metadata or {}).get('terminate_reason') for s, _ in failed)),
    )


def calibration_decision(report):
    """Preserve baseline scale calibration on mixed groups; audit new signal."""
    from openwebrl.arm_turn_bonus_runtime import calibration_decision as native_decision
    mixed = report.get('mixed_outcome_panel')
    # No division by zero, nor automatic approval of a pure proxy-reward batch.
    # Keep sample-count and coverage requirements on the whole retained batch.
    # Applying the 100-label threshold separately to mixed groups would turn a
    # changed group mixture into an artificial calibration failure.
    decision = native_decision(dict(report, all_failure_experiment=False))
    scale_decision = native_decision(mixed) if mixed else None
    decision['checks']['scale'] = bool(scale_decision and scale_decision['checks']['scale'])
    decision['suggested_beta'] = scale_decision['suggested_beta'] if scale_decision else None
    decision['checks'].update(
        all_failure_groups=report['all_failure_groups'] > 0,
        all_failure_labels=report['all_failure_usable_labels'] > 0,
    )
    decision['passed'] = all(decision['checks'].values())
    decision['rule'] = ('Keep beta=0.5 and q=0.2. Apply original label/task/coverage gates to the full retained '
                        'batch and the scale gate to mixed-outcome groups; require usable supervision in at least one valid all-failure group. '
                        'Log all-failure advantage scale separately; never normalize it by zero outcome variance.')
    from openwebrl.arm_continuation_guard import apply_adjacent_count_support
    return apply_adjacent_count_support(report, decision)
