"""One episode-validity rule shared by RL rollouts, branch collection and MolmoWeb evaluation.

Rule: an episode is invalid when infrastructure ended it; anything the agent
caused is a valid outcome (scored by that system's own judge, usually 0 unless
the judge says success).

Valid endings      done, step_limit, action_limit, decision_limit, context_limit,
                   format_failure, requires_user_input
Invalid endings    browser_env    browser step/navigation/observation/reset failure
                   model_server   actor or selector server abort, timeout or outage
                   wall_timeout   whole-episode wall-clock limit
                   capture        final observation could not be captured
                   judge          judge timeout, error or missing verdict
                   contract       budget/contract/input guard tripped
                   unknown        no recognized signal (treated as invalid)

Each ``classify_*`` maps one system's saved record to ``Outcome``; nothing here
changes how a run executes.
"""
from __future__ import annotations

from dataclasses import dataclass

VALID_ENDINGS = frozenset({'done', 'step_limit', 'action_limit', 'decision_limit', 'context_limit',
                           'format_failure', 'requires_user_input'})
INVALID_ENDINGS = frozenset({'browser_env', 'model_server', 'wall_timeout', 'capture', 'judge', 'contract', 'unknown'})


@dataclass(frozen=True)
class Outcome:
    ending: str
    detail: str = ''

    @property
    def valid(self) -> bool:
        return self.ending in VALID_ENDINGS


def _status(value) -> str:
    """'Status.ABORTED', Status.ABORTED or 'aborted' -> 'aborted'."""
    value = getattr(value, 'value', value)
    return str(value or '').rsplit('.', 1)[-1].lower()


def classify_terminate_reason(status, terminate_reason, *, judge_failed=False) -> Outcome:
    """Shared mapping of generate_browser.py's (status, terminate_reason)."""
    status, reason = _status(status), str(terminate_reason or '')
    low = reason.lower()
    if low == 'task_completed':
        return Outcome('judge', 'judge timeout or invalid verdict') if judge_failed else Outcome('done')
    if low == 'max_steps_exhausted':
        return Outcome('step_limit')
    if low in ('generation_length_limit', 'rollout_max_context_length_exceeded') or 'must be below' in low:
        return Outcome('context_limit', reason)
    if low == 'format_error_failed':
        return Outcome('format_failure')
    if low == 'env_step_error' or 'env_server' in low or 'sandbox exec timed out' in low or 'not healthy within' in low:
        return Outcome('browser_env', reason)
    if 'rollout_task_timeout' in low:
        return Outcome('wall_timeout', reason)
    if low == 'generation_abort' or low.startswith('generation_error'):
        return Outcome('model_server', reason)
    if low == 'no_generated_turns':
        return Outcome('contract', reason)
    if status == 'completed':
        return Outcome('judge', 'judge timeout or invalid verdict') if judge_failed else Outcome('done')
    return Outcome('unknown', f'{status}:{reason}')


def classify_rl_trajectory(turns) -> Outcome:
    """RL rollout archive trajectory: list of turn dicts with status/metadata."""
    last = max(turns, key=lambda s: (s.get('metadata') or {}).get('turn_index', 0))
    meta = last.get('metadata') or {}
    reward = meta.get('reward') or {}
    return classify_terminate_reason(last.get('status'), meta.get('terminate_reason'),
                                     judge_failed=bool(reward.get('judge_timeout') or meta.get('judge_invalid')))


def classify_branch(result, rollout=None) -> Outcome:
    """Branch-collection result.json plus its last saved rollout record (status, terminate_reason).

    The controlled browser enforces the 30-operation action limit inside env.step,
    so an action-limit ending appears as env_step_error and must be checked first.
    """
    boundary = result.get('policy_boundary_reason')
    terminal = result.get('browser_terminal_reason')
    if result.get('reason') == 'judge_budget_halt':
        return Outcome('judge', 'judge budget halt')
    if boundary == 'episode_deadline':
        return Outcome('wall_timeout', 'episode_deadline')
    if boundary in ('action_limit', 'decision_limit', 'requires_user_input'):
        return Outcome(boundary)
    if terminal in ('action_limit', 'requires_user_input'):
        return Outcome(terminal)
    rollout = rollout or {}
    outcome = classify_terminate_reason(rollout.get('status'), rollout.get('terminate_reason'))
    if outcome.ending in ('done', 'unknown') and result.get('reason') == 'missing_fresh_terminal_evidence':
        return Outcome('capture', 'final observation missing')
    if outcome.ending == 'unknown' and result.get('reason') == 'infrastructure_or_input_budget_invalid':
        return Outcome('contract', result['reason'])
    if outcome.valid and result.get('reward') is None:
        return Outcome('judge', 'no score recorded')
    return outcome


_MODEL_SERVER_MARKERS = ('connection', '503', 'overloaded', 'timed out', 'timeout', 'unavailable',
                         'disconnect', 'reset', 'refused', 'bad gateway', '502', '500')


def classify_molmoweb(worker, judge=None, prediction_error='') -> Outcome:
    """MolmoWeb worker-result.json (+ judge-result.json, + ACTION_PREDICTION_ERROR message)."""
    status = worker.get('status')
    invalid = worker.get('invalid_reason') or {}
    if status == 'completed_timeout' or invalid.get('classification') == 'episode_wall_timeout':
        return Outcome('wall_timeout')
    if status == 'completed_invalid':
        return Outcome('browser_env', f"{invalid.get('classification')}:{invalid.get('phase')}")
    errors = (worker.get('episode') or {}).get('error') or []
    if 'ACTION_PREDICTION_ERROR' in errors:
        low = str(prediction_error).lower()
        if not low or any(m in low for m in _MODEL_SERVER_MARKERS) or 'draw is unavailable' in low:
            return Outcome('model_server', prediction_error[:120])
        return Outcome('format_failure', prediction_error[:120])
    if status == 'completed' and worker.get('valid') is True:
        if judge is not None and not (judge.get('native') or {}).get('judged'):
            return Outcome('judge', 'native verdict missing')
        return Outcome('done')
    if status == 'budget_blocked':
        return Outcome('contract', 'budget_blocked')
    return Outcome('unknown', f"{status}:{worker.get('error_type')}")
