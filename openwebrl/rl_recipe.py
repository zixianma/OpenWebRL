"""CPU-only primitives for outcome validation and optional browser RL experiments.

No model/server imports: these routines also work in offline dataset tooling.
"""
import json
import math
import re
import statistics
from collections import defaultdict


def parse_judge_verdict(text):
    """Read a structured verdict or a standalone final verdict, never rationale substrings."""
    cleaned = text.strip()
    if cleaned.startswith('```'):
        cleaned = re.sub(r'^```(?:json)?\s*|\s*```$', '', cleaned)
    try:
        obj = json.loads(cleaned)
    except (ValueError, TypeError):
        obj = None
    if isinstance(obj, dict):
        verdict = obj.get('verdict')
    else:
        last = cleaned.splitlines()[-1] if cleaned else ''
        last = last.replace('**', '').strip()
        match = re.fullmatch(r'(?:(?:final\s+)?verdict\s*:\s*)?(SUCCESS|NOT SUCCESS)[.!]?', last, re.I)
        verdict = match.group(1) if match else None
    if not isinstance(verdict, str):
        return None
    return {'SUCCESS': 1.0, 'NOT SUCCESS': 0.0}.get(verdict.strip().upper())


def valid_trajectory(turns):
    return bool(turns) and not any(t.remove_sample for t in turns) and turns[-1].reward is not None


def trajectory_advantages(samples, args):
    """Return GRPO advantages with invalid trajectories excluded from group statistics."""
    groups = defaultdict(lambda: defaultdict(list))
    for s in samples:
        groups[s.group_index][s.metadata.get('trajectory_id', s.index)].append(s)
    result = {id(s): 0.0 for s in samples}
    for trajectories in groups.values():
        valid = [ts for ts in trajectories.values() if valid_trajectory(ts)]
        rewards = [ts[-1].get_reward_value(args) for ts in valid]
        if not rewards:
            continue
        mean = statistics.mean(rewards)
        scale = statistics.stdev(rewards) + 1e-6 if len(rewards) > 1 and args.grpo_std_normalization else 1.0
        for ts, reward in zip(valid, rewards):
            for s in ts:
                result[id(s)] = (reward - mean) / scale
    return result


def state_advantages(args, samples):
    """Opt-in reward postprocessor using a frozen, pre-action value provider.

    Provider receives only prefix state records (no responses, labels or terminal metadata)
    and must return one success probability per record. Train it on earlier rollouts or
    task-disjoint folds. This implements Monte Carlo R-V, not a PPO critic/GAE.
    """
    if args.advantage_estimator != 'grpo' or not args.rewards_normalization:
        raise ValueError('state_advantages requires normalized GRPO')
    mix = float(getattr(args, 'browser_state_advantage_mix', 0.0))
    if not 0 <= mix <= 1:
        raise ValueError('browser_state_advantage_mix must be in [0, 1]')
    invalid = {(s.group_index, s.metadata.get('trajectory_id', s.index)) for s in samples if s.remove_sample}
    for s in samples:
        if (s.group_index, s.metadata.get('trajectory_id', s.index)) in invalid:
            s.remove_sample = True
    base = trajectory_advantages(samples, args)
    raw = [s.get_reward_value(args) for s in samples]
    if mix == 0:
        return raw, [base[id(s)] for s in samples]
    import importlib
    module_name, _, function_name = args.browser_value_provider_path.rpartition('.')
    provider = getattr(importlib.import_module(module_name), function_name)
    valid = [s for s in samples if not s.remove_sample]
    if not valid:
        return raw, [0.0] * len(samples)
    if any('turn_index' not in s.metadata for s in valid):
        raise ValueError('state_advantages requires turn-level browser samples')
    states = [{'prompt': s.prompt, 'images': (s.multimodal_inputs or {}).get('images', []),
               'turn_index': s.metadata['turn_index']} for s in valid]
    values = list(provider(args, states))
    if len(values) != len(valid) or any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise ValueError('value provider must return one finite probability in [0, 1] per state')
    out = {id(s): 0.0 for s in samples}
    for s, value in zip(valid, values):
        reward = s.get_reward_value(args)
        # Keep the separate protocol penalty on its original GRPO scale.
        out[id(s)] = base[id(s)] if reward < 0 else (1-mix)*base[id(s)] + mix*(reward-value)
    return raw, [out[id(s)] for s in samples]


def observation_memory(turns, max_entries=8, max_chars=4000):
    """Bounded record of executed actions and environment observations, not model claims.

    This is a conservative memory baseline: no inferred facts or progress labels.
    Full reasoning history remains available to the policy.
    """
    if max_entries <= 0 or max_chars < 2:
        raise ValueError('memory limits must be positive (max_chars >= 2)')
    entries = []
    for s in turns:
        for feedback in s.metadata.get('step_tool_responses', []):
            entries.append({'step': s.metadata.get('turn_index'),
                            'tool': feedback.get('tool_name', ''),
                            'observation': str(feedback.get('tool_response', ''))[:1000]})
    entries = entries[-max_entries:]
    while entries and len(json.dumps(entries, ensure_ascii=False)) > max_chars:
        entries.pop(0)
    return json.dumps(entries, ensure_ascii=False)
