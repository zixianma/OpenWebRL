"""Paper deterministic monitoring protocol; independent of training horizons."""
import asyncio
from copy import copy
import logging
import os

logger = logging.getLogger(__name__)


async def generate(args, sample, sampling_params, evaluation=False):
    if not evaluation:
        raise ValueError('The monitoring generator is evaluation-only')
    from openwebrl.generate_browser import generate_turn_sample
    from openwebrl.reward_browser import reward_func
    from slime.utils.types import Sample

    eval_args = copy(args)
    eval_args.max_steps = 30
    eval_args.inference_step_timeout_secs = 120.0
    eval_args.judge_timeout_secs = 120.0
    turns = await generate_turn_sample(eval_args, sample, sampling_params)
    if turns and not any(s.status == Sample.Status.ABORTED for s in turns):
        rewards = await reward_func(eval_args, turns)
        for turn, reward in zip(turns, rewards, strict=True):
            turn.reward = reward
    # Eval retains all completed trajectories until logging and debug saving.
    # Reuse the lossless collection mapping before their CPU images accumulate.
    if os.environ.get("OPENWEBRL_MULTIMODAL_STORAGE_DIR"):
        from slime.utils.rollout_transport import file_back_completed_group

        mapped = await asyncio.to_thread(file_back_completed_group, turns)
        logger.info("[EvalStorage] mapped_completed_turns=%d", mapped)
    return turns
