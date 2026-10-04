"""o4-mini/AgentTrek evaluation with durable per-task archives.

This public template uses the same protocol and persistence as the frozen
stealth runner. Local GPT-4.1 monitoring remains in eval_monitor.py.
"""
import asyncio
from copy import copy
import json
import os
import uuid
from pathlib import Path

SAMPLING = dict(temperature=0.6, top_p=0.95, top_k=20,
                max_new_tokens=4096, repetition_penalty=1.0)
_provider_blocked = False


class BrowserCreditsExhausted(RuntimeError):
    """Account-wide failure: stop the cohort instead of counting task failures."""


def credits_exhausted(reason):
    text = str(reason).lower()
    return '402' in text and 'credits' in text and 'browser session' in text


def stop_for_credits():
    global _provider_blocked
    _provider_blocked = True
    directory = os.environ.get('OPENWEBRL_EVAL_ROLLOUT_DIR')
    if directory:
        root = Path(directory).parent
        root.mkdir(parents=True, exist_ok=True)
        (root/'provider-blocked.json').write_text(json.dumps(dict(
            reason='Browser Use credit exhaustion (HTTP402)',
            requires_user=True, verified_complete=False))+'\n')
    raise BrowserCreditsExhausted('Browser Use credit exhaustion (HTTP402); preserve partial results')



def save_completed_task(args, turns):
    """Keep the legacy summary used by subset retries, alongside full archives."""
    directory = os.environ.get('OPENWEBRL_BENCHMARK_RESULTS_DIR')
    if not directory or not turns:
        return
    last = next((s for s in turns if s.metadata.get('is_last_turn')), turns[-1])
    meta = last.metadata
    record = {
        'task_id': meta.get('task_id'), 'intent': meta.get('intent'),
        'trajectory_id': meta.get('trajectory_id'),
        'status': getattr(last.status, 'value', str(last.status)),
        'remove_sample': any(s.remove_sample for s in turns),
        'terminate_reason': meta.get('terminate_reason'),
        'total_steps': meta.get('total_steps', len(turns)),
        'reward': meta.get('reward'), 'transport_reward': last.reward,
        'judge_invalid': meta.get('judge_invalid', False),
        'judge_model': args.judge_api_model,
        'judge_prompt_variant': getattr(args, 'judge_prompt_variant', None),
        'responses': [s.response for s in turns],
    }
    target = Path(directory) / (uuid.uuid4().hex + '.json')
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(target)


def configure(args, sampling_params):
    result = copy(args)
    result.max_steps = 30
    result.rollout_max_context_len = 32768
    result.context_num_screenshots = 1
    result.turn_history_reasoning_mode = 'full'
    result.browser_response_format_mode = 'browser_env'
    result.browser_include_tool_response = 1
    result.judge_api_model = 'o4-mini'
    result.judge_prompt_variant = 'agenttrek'
    result.judge_timeout_secs = 120.0
    result.inference_step_timeout_secs = 120.0
    result.task_timeout_secs = 600.0
    result.rollout_temperature = SAMPLING['temperature']
    result.rollout_max_response_len = SAMPLING['max_new_tokens']
    if getattr(args, 'browser_action_selector', None) is not None:
        raise ValueError('Stealth comparison requires a standalone actor')
    return result, dict(sampling_params, **SAMPLING)


async def generate(args, sample, sampling_params, evaluation=False):
    if not evaluation:
        raise ValueError('Stealth benchmark is evaluation-only')
    if _provider_blocked:
        raise BrowserCreditsExhausted('Browser Use account is blocked; no further session requests')
    from openwebrl.generate_browser import generate_turn_sample
    from openwebrl.eval.reward_online_mind2web import reward_func
    from openwebrl.eval_monitor import persist_task
    from slime.utils.types import Sample

    eval_args, sampling = configure(args, sampling_params)
    turns, error_type = [], None
    try:
        turns = await generate_turn_sample(eval_args, sample, sampling)
        if any(credits_exhausted((getattr(s, 'metadata', None) or {}).get('terminate_reason', ''))
               for s in turns):
            stop_for_credits()
        if turns and not any(s.status == Sample.Status.ABORTED for s in turns):
            rewards = await reward_func(eval_args, turns)
            for turn, reward in zip(turns, rewards, strict=True):
                # A missing verdict must not trigger generic GPT-4.1 rejudging.
                turn.reward = reward if reward is not None else 0.0
                if reward is None:
                    turn.remove_sample = True
                    turn.metadata['judge_invalid'] = True
                    turn.metadata['benchmark_judge_reward'] = None
    except BaseException as exc:
        error_type = type(exc).__name__
        if credits_exhausted(exc):
            error_type = 'BrowserCreditsExhausted'
            stop_for_credits()
        raise
    finally:
        await persist_task(eval_args, sample, turns, error_type)
        await asyncio.to_thread(save_completed_task, eval_args, turns)
    return turns
