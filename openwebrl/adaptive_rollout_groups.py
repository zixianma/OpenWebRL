"""Ordinary eight-rollout and five-then-three rollout allocation.

This module only schedules native trajectory generation. It does not select
actions, change rewards, normalize advantages, or update the actor.

Frozen-collector integration (not installed by importing this module):

* Reserve eight unique native Sample slots via ``n_samples_per_prompt=8``.
* In ``generate_and_rm_group``, keep the native evaluation and early-abort
  paths. After assigning native session IDs, call ``generate_rollout_group``
  instead of the training gather. Pass an async closure that calls the native
  ``generate_and_rm(args, sample, params, evaluation=False)``. Each call must
  initialize a fresh browser and use the same frozen actor throughout a group.
* Inject ``state.group_sampling_seeds`` only when native deterministic
  inference is enabled, ``lambda: state.aborted``, and
  ``lambda sample: sample.get_reward_value(args)``. Reject ``args.group_rm``:
  adaptive allocation requires per-trajectory rewards before phase two.
* Return the actual five/eight trajectories to the native dynamic filter.
  Downstream collection/normalization must support variable group sizes and
  retain native group identities; never pad missing slots with fake samples.
  Count physical trajectories from generated records, not the reserved size.

The adaptive gate uses native validity flags and rewards, including native
format failures of -1. It does not add a judge-quality gate or rescore outcomes.
Training launcher/collector integration and a GPU smoke remain separate checks.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any


UNIFORM_EIGHT = "uniform_8"
ADAPTIVE_FIVE_THEN_THREE = "adaptive_5_then_3"
METADATA_KEY = "adaptive_rollout_group"


def _rows(trajectory: Any) -> list[Any]:
    return trajectory if isinstance(trajectory, list) else [trajectory]


def _status(sample: Any) -> str | None:
    status = getattr(sample, "status", None)
    return getattr(status, "value", status)


def failure_extension_reason(
    base_trajectories: Sequence[Any],
    *,
    reward_value: Callable[[Any], Any] | None = None,
) -> str:
    """Return ``all_five_valid_failures`` iff three extra slots are eligible.

    An invalid trajectory anywhere in the five takes precedence over a success.
    Reward extraction is injectable to preserve the native ``reward_key`` path.
    Unexpected extractor errors propagate instead of silently authorizing work.
    """
    if len(base_trajectories) != 5:
        raise ValueError("Adaptive eligibility requires exactly five trajectories")
    read_reward = reward_value or (lambda sample: sample.reward)
    terminal_rewards = []
    for trajectory in base_trajectories:
        rows = _rows(trajectory)
        if not rows:
            return "base_contains_invalid"
        for sample in rows:
            if (
                sample is None
                or getattr(sample, "remove_sample", False)
                or _status(sample) not in {"completed", "failed", "truncated"}
                or getattr(sample, "reward", None) is None
            ):
                return "base_contains_invalid"
            value = read_reward(sample)
            if value is None or not math.isfinite(float(value)):
                return "base_contains_invalid"
        terminal_rewards.append(float(read_reward(rows[-1])))
    if any(value > 0 for value in terminal_rewards):
        return "base_contains_success"
    return "all_five_valid_failures"


async def _gather_owned(awaitables: Sequence[Awaitable[Any]]) -> list[Any]:
    """Propagate errors/cancellation and await cleanup of all sibling tasks."""
    tasks = [asyncio.create_task(awaitable) for awaitable in awaitables]
    try:
        return list(await asyncio.gather(*tasks))
    except BaseException:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


async def generate_rollout_group(
    group: Sequence[Any],
    sampling_params: Mapping[str, Any],
    *,
    mode: str,
    generate_one: Callable[[Any, dict[str, Any]], Awaitable[Any]],
    sampling_seeds: Sequence[int] | None = None,
    is_aborted: Callable[[], bool] | None = None,
    reward_value: Callable[[Any], Any] | None = None,
    evaluation: bool = False,
) -> list[Any]:
    """Run actual native trajectories, preserving slot order, indices and seeds.

    ``generate_one`` owns browser and actor semantics; it returns a native
    Sample or a list of native turn Samples. The helper never synthesizes a
    trajectory. Evaluation runs all supplied slots without adaptive annotations.
    Abort state raises CancelledError; native callers should retain their own
    pre-generation abort branch. A returned native ABORTED trajectory simply
    makes the adaptive five ineligible for extension.
    """
    if mode not in {UNIFORM_EIGHT, ADAPTIVE_FIVE_THEN_THREE}:
        raise ValueError(f"Unknown ordinary rollout mode: {mode!r}")
    if not evaluation:
        if len(group) != 8:
            raise ValueError("Training requires eight reserved native Sample slots")
        indices = [sample.index for sample in group]
        if any(index is None for index in indices) or len(set(indices)) != 8:
            raise ValueError("Reserved native sample indices must be unique and non-null")
        group_indices = {sample.group_index for sample in group}
        if len(group_indices) != 1 or None in group_indices:
            raise ValueError("Reserved samples must share one non-null native group_index")
    if sampling_seeds is not None and len(sampling_seeds) < len(group):
        raise ValueError("Native seed vector must cover all reserved slots")

    def check_abort() -> None:
        if is_aborted is not None and is_aborted():
            raise asyncio.CancelledError("Native rollout state aborted")

    async def run_slots(slots: range) -> list[Any]:
        check_abort()
        calls = []
        for slot in slots:
            params = dict(sampling_params)
            if sampling_seeds is not None:
                params["sampling_seed"] = sampling_seeds[slot]
            calls.append(generate_one(group[slot], params))
        result = await _gather_owned(calls)
        check_abort()
        return result

    if evaluation:
        return await run_slots(range(len(group)))

    if mode == UNIFORM_EIGHT:
        base_count = 8
        result = await run_slots(range(8))
        extra_allocated = 0
        reason = "uniform_eight"
    else:
        base_count = 5
        result = await run_slots(range(5))
        reason = failure_extension_reason(result, reward_value=reward_value)
        extra_allocated = 3 if reason == "all_five_valid_failures" else 0
        if extra_allocated:
            # Phase two starts after all five outcomes exist. Always collect all
            # three extras; a success cannot cancel its sibling trajectories.
            result.extend(await run_slots(range(5, 8)))

    for slot, trajectory in enumerate(result):
        for sample in _rows(trajectory):
            if sample is None:
                continue
            metadata = dict(getattr(sample, "metadata", None) or {})
            metadata[METADATA_KEY] = {
                "version": 1,
                "mode": mode,
                "reserved_count": 8,
                "base_count": base_count,
                "extra_allocated": extra_allocated,
                "generated_count": len(result),
                "reason": reason,
                "slot": slot,
                "phase": "base" if slot < base_count else "extra",
                "reserved_sample_index": group[slot].index,
                "group_index": group[slot].group_index,
                "sampling_seed": (
                    sampling_seeds[slot] if sampling_seeds is not None
                    else sampling_params.get("sampling_seed")
                ),
            }
            sample.metadata = metadata
    return result
