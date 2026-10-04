"""Original-bonus ablation: outcome weighting on ordinary mixed groups only.

Pure scalar transform, executed before DP splitting/epoch trimming. Never adds
failure groups, changes raw outcomes, or creates an auxiliary loss.
"""
from collections import defaultdict
import math
import statistics


def validate_config(config):
    if config.get('advantage_mode') not in ('original_bonus', 'outcome_reweight'):
        raise ValueError('Explicit original-bonus or outcome-reweight mode required')
    if (config.get('k') != 5 or config.get('beta') != .5
            or config.get('scored_fraction') != .2
            or config.get('candidate_gate') != 'min2'
            or config.get('credit_assignment', 'response_index') != 'response_index'):
        raise ValueError('Preserve matched relaxed K5/q20/beta0.5/min2/index credit')
    if (config.get('admit_all_failure_groups') or config.get('additive_failure_groups')
            or config.get('failure_loss_coefficient', 0) or config.get('failure_group_cap', 0)
            or config.get('max_failure_groups', 0) or config.get('failure_ablation')):
        raise ValueError('No all-failure group or auxiliary signal in this ablation')
    if config.get('reweight_lambda') != .5:
        raise ValueError('Predeclared lambda must remain0.5')


def trainable(sample):
    meta = sample.metadata or {}
    return (not sample.remove_sample and not meta.get('arm_calibration_excluded')
            and (sample.loss_mask is None or any(sample.loss_mask)))


def unit(sample, policy_id, candidate_gate='min2'):
    if not trainable(sample):
        return 0.
    label = (sample.metadata or {}).get('arm_turn_bonus', {})
    if not label.get('eligible'):
        return 0.
    selected = label.get('selected_index')
    if (label.get('policy_id') != policy_id or label.get('executed_index') != 0
            or type(selected) is not int or not 0 <= selected < 5
            or label.get('candidate_gate', 'distinct5') != candidate_gate
            or label.get('credit_assignment', 'response_index') != 'response_index'):
        raise ValueError('Original-bonus label policy/gate/credit mismatch')
    classes = label.get('action_class_ids')
    minimum = 2 if candidate_gate == 'min2' else 5
    if candidate_gate not in ('min2', 'distinct5'):
        raise ValueError('Unknown candidate gate')
    if candidate_gate == 'min2' and classes is None:
        raise ValueError('Relaxed labels need action-class provenance')
    if classes is not None and (len(classes) != 5 or len(set(classes)) < minimum):
        raise ValueError('Label does not meet the declared distinct-action gate')
    value = float(selected == 0)-.2
    if label.get('unit_bonus') != value:
        raise ValueError('Stored unit signal mismatch')
    return value


def transform(samples, raw_rewards, advantages, policy_id, lam=.5, mode='outcome_reweight', candidate_gate='min2'):
    if (len(samples) != len(raw_rewards) or len(samples) != len(advantages)
            or not math.isfinite(lam) or not 0 <= lam <= 1
            or mode not in ('original_bonus', 'outcome_reweight')):
        raise ValueError('Invalid weighting inputs')
    if any(not math.isfinite(a) for a in advantages):
        raise ValueError('Nonfinite native advantage')
    groups = defaultdict(list)
    trajectories = defaultdict(list)
    us = [unit(s, policy_id, candidate_gate) for s in samples]
    result = [a+.5*u for a, u in zip(advantages, us)]
    weights = [1.] * len(samples)
    for j, sample in enumerate(samples):
        meta = sample.metadata or {}
        if not hasattr(sample, 'group_index') or 'trajectory_id' not in meta:
            raise ValueError('Group/trajectory identity required before DP slicing')
        key = (sample.group_index, str(meta['trajectory_id']))
        groups[sample.group_index].append(j)
        if trainable(sample):
            trajectories[key].append(j)
    fallback = {g for g, js in groups.items() if any(raw_rewards[j] not in (0, 1) for j in js)}
    mean_error = 0.
    for (group, _), js in trajectories.items():
        A = advantages[js[0]]
        if any(not math.isclose(advantages[j], A, rel_tol=1e-8, abs_tol=1e-8) for j in js):
            raise ValueError('Expected one native outcome advantage per trajectory')
        if mode == 'original_bonus' or group in fallback:
            continue
        sign = 1 if A > 0 else -1 if A < 0 else 0
        v = [math.exp(lam*sign*us[j]) for j in js]
        mean = statistics.mean(v)
        for j, value in zip(js, v):
            weights[j] = value/mean
            result[j] = A*weights[j]
        mean_error = max(mean_error, abs(statistics.mean(result[j] for j in js)-A))
    active = [j for j, sample in enumerate(samples) if trainable(sample)]
    denominator = sum(advantages[j]**2 for j in active)
    changed = sum((result[j]-advantages[j])**2 for j in active)
    rw = sorted(weights[j] for js in trajectories.values() for j in js
                if samples[j].group_index not in fallback)
    report = dict(mode=mode, candidate_gate=candidate_gate, lambda_value=lam, rows=len(active), groups=len(groups),
        fallback_groups=len(fallback), fallback_rows=sum(samples[j].group_index in fallback for j in active),
        perturbation_to_outcome_rms=math.sqrt(changed/denominator) if denominator else None,
        minimum_weight=min(rw, default=1.), maximum_weight=max(rw, default=1.),
        mean_advantage_error=mean_error, all_failure_auxiliary=False)
    return result, report


def post_process_rewards(args, samples, raw_rewards, normalized_rewards):
    """Wrap the frozen original hook's calibration without altering collection."""
    from pathlib import Path
    import os
    from openwebrl import arm_turn_bonus as parent
    state = parent.state(args)
    config = state['config']
    validate_config(config)
    if os.environ.get('OPENWEBRL_ARM_FAILURE_AUX_MANIFEST'):
        raise ValueError('An auxiliary failure manifest must not be enabled')
    if getattr(args, 'dynamic_sampling_filter_path', '') != 'slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonempty_nonzero_std':
        raise ValueError('Original native mixed-group filter must be retained')
    raw, original = parent.post_process_rewards(args, samples, raw_rewards, normalized_rewards)
    # Preserve parent calibration stop/shadow behavior and object identity.
    import json
    calibration = json.loads((Path(config['output'])/'calibration.json').read_text())
    if calibration['applied_beta'] == 0:
        return raw, original
    proposed, report = transform(samples, raw_rewards, normalized_rewards, config['policy_id'],
                                 config['reweight_lambda'], config['advantage_mode'], config['candidate_gate'])
    parent.write_json(Path(config['output'])/'reweighting.json', report)
    try:
        import wandb
        if wandb.run is not None:
            wandb.log({'arm_reweight/'+k:v for k,v in report.items() if isinstance(v,(int,float))})
    except Exception as exc:
        print('[ARM reweight telemetry unavailable] '+type(exc).__name__, flush=True)
    return raw, proposed
