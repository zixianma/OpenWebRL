"""Paper deterministic monitoring protocol; independent of training horizons."""
import asyncio
from copy import copy
import logging
import os
from pathlib import Path

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
    turns = []
    error_type = None
    try:
        turns = await generate_turn_sample(eval_args, sample, sampling_params)
        if turns and not any(s.status == Sample.Status.ABORTED for s in turns):
            rewards = await reward_func(eval_args, turns)
            for turn, reward in zip(turns, rewards, strict=True):
                turn.reward = reward
    except BaseException as exc:
        error_type = type(exc).__name__
        raise
    finally:
        await persist_task(eval_args, sample, turns, error_type)
    return turns


async def persist_task(args, sample, turns, error_type=None):
    """Preserve completed, invalid, and judge-error attempts without changing rewards."""
    # Eval retains all completed trajectories until logging and debug saving.
    # Reuse the lossless collection mapping before their CPU images accumulate.
    if os.environ.get("OPENWEBRL_MULTIMODAL_STORAGE_DIR"):
        from slime.utils.rollout_transport import file_back_completed_group

        mapped = await asyncio.to_thread(file_back_completed_group, turns)
        logger.info("[EvalStorage] mapped_completed_turns=%d", mapped)
    # Keep a lossless, task-addressable copy alongside the aggregate recovery
    # archive.  Judge retries can then reuse these trajectories without
    # reopening browsers or loading the actor checkpoint.
    rollout_dir = os.environ.get("OPENWEBRL_EVAL_ROLLOUT_DIR")
    task_id = sample.metadata.get("task_id") if sample is not None else None
    if rollout_dir:
        if task_id is None:
            raise ValueError('Per-task evaluation persistence requires a task ID')
        await asyncio.to_thread(save_task, args, Path(rollout_dir), str(task_id), turns, error_type)


def save_task(args, directory, task_id, turns, error_type):
    import hashlib
    import json
    import torch
    from slime.utils.trajectory_metrics import trajectory_metrics

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'{hashlib.sha256(task_id.encode()).hexdigest()}.pt'
    if path.exists():
        raise ValueError('Evaluation attempted the same task twice in one output directory')
    payload = dict(task_id=task_id, turns=[turn.to_dict() for turn in turns], error_type=error_type)
    temporary = path.with_suffix('.pt.partial')
    with temporary.open('xb') as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    terminal = max(turns, key=lambda s: (s.metadata or {}).get('turn_index', 0)) if turns else None
    record = dict(task_id=task_id, rollout_file=path.name, turns=len(turns),
        error_type=error_type, metrics=trajectory_metrics(args, [turns]),
        judge_model=getattr(args, 'judge_api_model', None),
        judge_prompt_variant=getattr(args, 'judge_prompt_variant', None),
        terminal_status=getattr(terminal.status, 'value', terminal.status) if terminal else None,
        reward_metadata=(terminal.metadata or {}).get('reward', {}) if terminal else {})
    temporary = path.with_suffix('.json.partial')
    with temporary.open('x') as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path.with_suffix('.json'))
