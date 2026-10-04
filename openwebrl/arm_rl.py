"""ARM RL eligibility and exact, differentiable component objectives.

These functions operate on response-token log probabilities. They do not merge
teacher samples into GRPO groups or silently change optimizer batch sizes.
The distributed training integration must satisfy the same component contract.
"""
from dataclasses import dataclass
import hashlib
import json
import math
import statistics


def stable_seed(*parts):
    data = json.dumps(parts, ensure_ascii=False, separators=(',', ':')).encode()
    return int.from_bytes(hashlib.sha256(data).digest()[:4], 'big') & 0x7fffffff


def outcome_reward(*, format_error_failed, format_valid, judge_success, valid=True):
    if not valid:
        return None
    if format_error_failed:
        return -1.0
    return float(bool(format_valid and judge_success))


def group_advantages(rewards):
    """One value per trajectory; sample std, as in the preserved baseline."""
    if not rewards or any(r is None or not math.isfinite(r) for r in rewards):
        raise ValueError('Filter invalid trajectories before GRPO normalization')
    mean = statistics.mean(rewards)
    scale = statistics.stdev(rewards) + 1e-6 if len(rewards) > 1 else 1.0
    return [(r - mean) / scale for r in rewards]


def rescue_eligibility(group, *, policy_id, judge_id, excluded_task_ids=(), excluded_intents=()):
    """Strict admission: exactly five distinct, valid zero-reward trajectories."""
    if group.get('task_id') in excluded_task_ids:
        return 'evaluation_task'
    intent = ' '.join(str(group.get('intent', '')).casefold().split())
    if not intent or intent in excluded_intents:
        return 'missing_or_evaluation_intent'
    if not policy_id or group.get('policy_id') != policy_id:
        return 'policy_mismatch'
    if not judge_id or group.get('judge_id') != judge_id:
        return 'judge_mismatch'
    trajectories = group.get('trajectories', [])
    if len(trajectories) != 5:
        return 'not_five'
    ids = [t.get('trajectory_id') for t in trajectories]
    if None in ids or len(set(ids)) != 5:
        return 'duplicate_or_missing_trajectory_id'
    if any(t.get('policy_id') != policy_id or t.get('judge_id') != judge_id for t in trajectories):
        return 'trajectory_provenance_mismatch'
    if any(t.get('valid') is not True or t.get('reward') != 0.0 for t in trajectories):
        return 'not_five_valid_zero_rewards'
    return None


def admit_rescue(group, rescued, **identity):
    reason = rescue_eligibility(group, **identity)
    if reason:
        return reason
    if any(rescued.get(k) != group.get(k) for k in ('task_id', 'policy_id', 'judge_id')):
        return 'rescue_provenance_mismatch'
    if rescued.get('valid') is not True or rescued.get('reward') != 1.0:
        return 'rescue_not_successful'
    turns = rescued.get('turns', [])
    if not turns or any(t.get('executed') is not True or t.get('fallback') or
                        t.get('selected_index') is None for t in turns):
        return 'missing_execution_or_selection'
    return None


def local_advantages(action_classes, winner):
    """Each class is a canonical, schema-validated action sequence, not text."""
    if type(winner) is not int or not 0 <= winner < len(action_classes):
        raise ValueError('Invalid ARM winner; fallback is not a label')
    if any(not isinstance(key, str) or not key for key in action_classes):
        raise ValueError('All candidates must have validated executable actions')
    losers = {}
    for i, key in enumerate(action_classes):
        if key != action_classes[winner]:
            losers.setdefault(key, i)
    if not losers:
        return None
    result = [0.0] * len(action_classes)
    result[winner] = 1.0
    for i in losers.values():
        result[i] = -1.0 / len(losers)
    return result


def action_token_mask(response, token_ids, tokenizer, spans):
    """Select complete tool-call spans and im_end; fail on token-ID mismatch.

    Callers validate tool schemas and supply their exact character spans.
    Newline/padding and historical reasoning are excluded. A token straddling
    reasoning/action is rejected rather than silently supervising reasoning.
    """
    if not response.endswith('<|im_end|>\n') or not spans:
        raise ValueError('Missing normal completion or executable action')
    if any(a < 0 or b <= a or b > len(response) for a, b in spans):
        raise ValueError('Invalid action spans')
    boundary = len(response) - len('<|im_end|>\n')
    regions = [*spans, (boundary, boundary + len('<|im_end|>'))]
    encoded = tokenizer.encode(response, add_special_tokens=False)
    # tokenizers.Tokenizer gives exact offsets including registered special tokens.
    if list(encoded.ids) != list(token_ids):
        raise ValueError('Response re-tokenization differs from sampled IDs')
    mask = []
    for a, b in encoded.offsets:
        overlaps = [(u, v) for u, v in regions if max(a, u) < min(b, v)]
        if overlaps and not any(u <= a < b <= v for u, v in overlaps):
            raise ValueError('A response token crosses a supervision boundary')
        mask.append(int(bool(overlaps)))
    if not any(mask) or not any(mask[i] for i, (a, b) in enumerate(encoded.offsets)
                                if a == boundary and b > boundary):
        raise ValueError('Completion token not aligned')
    return mask


@dataclass
class LossRow:
    logp: object
    mask: object
    old_logp: object = None
    advantage: float = 0.0
    old_logp_provenance: str = ''


def _mean(row, values):
    import torch
    mask = torch.as_tensor(row.mask, device=values.device, dtype=values.dtype)
    if values.ndim != 1 or mask.shape != values.shape or not bool(((mask == 0) | (mask == 1)).all()):
        raise ValueError('Response mask must be a matching binary vector')
    if not bool(mask.sum() > 0) or not bool(torch.isfinite(values).all()):
        raise ValueError('Empty mask or nonfinite response values')
    return (mask * values).sum() / mask.sum()


def clipped_row(row):
    import torch
    if row.old_logp is None or row.old_logp.shape != row.logp.shape:
        raise ValueError('Matching old-policy log probabilities required')
    if not math.isfinite(row.advantage):
        raise ValueError('Nonfinite advantage')
    ratio = (row.logp - row.old_logp.detach()).exp()
    return -_mean(row, torch.minimum(ratio * row.advantage,
                                    ratio.clamp(.8, 1.28) * row.advantage))


def component_losses(outcome_rows, *, demo_rows=(), local_groups=(), eta=0., lam=0.):
    """One optimizer window: outcome mean + eta*demo mean + lam*state mean.

    Input log probabilities are temperature .8 for PPO and T=1 for demo CE.
    Every local group's list retains zero-weight duplicate rows for audit. Its
    positive mass is +1, negative mass -1; there is no further divide by K.
    Coefficient zero skips auxiliary validation/graphs entirely (baseline identity).
    """
    if not outcome_rows or not all(math.isfinite(x) and x >= 0 for x in (eta, lam)):
        raise ValueError('Require an outcome batch and nonnegative finite coefficients')
    outcome = sum(clipped_row(r) for r in outcome_rows) / len(outcome_rows)
    zero = outcome.detach() * 0
    demo, local = zero, zero
    if eta and demo_rows:
        demo = -sum(_mean(r, r.logp) for r in demo_rows) / len(demo_rows)
    if lam and local_groups:
        terms = []
        for rows in local_groups:
            advantages = [r.advantage for r in rows]
            if (not math.isclose(sum(a for a in advantages if a > 0), 1.) or
                    not math.isclose(sum(a for a in advantages if a < 0), -1.)):
                raise ValueError('Local state needs positive mass +1 and negative mass -1')
            if any(r.old_logp_provenance != 'frozen_actor_teacher_forced' for r in rows if r.advantage):
                raise ValueError('Recompute old logps: generator appends synthetic completion logps of zero')
            terms.append(sum(clipped_row(r) for r in rows if r.advantage))
        local = sum(terms) / len(terms)
    return dict(total=outcome + eta * demo + lam * local,
                outcome=outcome, demo=demo, local=local)


def calibration(outcome_gradient, auxiliary_gradient, cap=.2):
    """Training-only pilot coefficient; freeze it for subsequent updates."""
    import torch
    if not 0 < cap <= 1:
        raise ValueError('Invalid gradient ratio cap')
    a, b = outcome_gradient.detach().double().flatten(), auxiliary_gradient.detach().double().flatten()
    if a.shape != b.shape or not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
        raise ValueError('Need matching finite gradients')
    na, nb = a.norm().item(), b.norm().item()
    if na <= 1e-12 or nb <= 1e-12:
        raise ValueError('Calibration requires nonzero outcome and auxiliary gradients')
    return dict(coefficient=cap * na / nb, cosine=(a @ b).item() / (na * nb),
                outcome_norm=na, auxiliary_norm=nb, initial_scaled_ratio=cap)


def optimizer_windows(outcomes, *, demos=(), outcome_batch_size=256, demo_cap=32):
    """Auxiliary exposure cannot create optimizer steps or change the RL tail."""
    if outcome_batch_size < 1 or demo_cap < 0:
        raise ValueError('Invalid window sizes')
    windows = []
    for offset in range(0, len(outcomes), outcome_batch_size):
        windows.append(dict(outcome=list(outcomes[offset:offset + outcome_batch_size]),
                            demo=list(demos[:demo_cap])))
    return windows
