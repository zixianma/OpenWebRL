"""CPU-only ordinary rollout allocation checks; no native GPU imports."""

import asyncio
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
import json
import unittest

from openwebrl.adaptive_rollout_groups import (
    ADAPTIVE_FIVE_THEN_THREE as ADAPTIVE,
    METADATA_KEY,
    UNIFORM_EIGHT as UNIFORM,
    failure_extension_reason,
    generate_rollout_group,
)


class Status(Enum):
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


@dataclass
class Sample:
    index: int
    group_index: int = 19
    reward: object = 0.0
    status: object = Status.COMPLETED
    remove_sample: bool = False
    metadata: dict = field(default_factory=lambda: {"task_id": "task-a"})


def slots():
    return [Sample(index=100 + i) for i in range(8)]


def trajectories(rewards=(0, 0, 0, 0, 0)):
    return [[Sample(1000 + 10 * i), Sample(1001 + 10 * i, reward=reward)]
            for i, reward in enumerate(rewards)]


class EligibilityTests(unittest.TestCase):
    def test_zero_and_format_negative_failures_are_eligible(self):
        for rewards in [(0,) * 5, (-1,) * 5, (0, -1, 0, -1, 0)]:
            group = trajectories(rewards)
            for trajectory in group:
                trajectory[-1].status = Status.FAILED
            self.assertEqual(failure_extension_reason(group), "all_five_valid_failures")

    def test_mixed_and_all_success_do_not_extend(self):
        for rewards in [(0, 0, 1, 0, 0), (1,) * 5]:
            self.assertEqual(failure_extension_reason(trajectories(rewards)),
                             "base_contains_success")

    def test_invalid_anywhere_never_triggers_extension(self):
        mutations = [
            lambda group: setattr(group[2][0], "remove_sample", True),
            lambda group: setattr(group[2][-1], "remove_sample", True),
            lambda group: setattr(group[2][0], "reward", None),
            lambda group: setattr(group[2][-1], "reward", None),
            lambda group: setattr(group[2][0], "status", Status.ABORTED),
            lambda group: setattr(group[2][-1], "status", "aborted"),
            lambda group: setattr(group[2][-1], "status", "pending"),
            lambda group: setattr(group[2][-1], "reward", float("nan")),
            lambda group: setattr(group[2][0], "reward", float("inf")),
            lambda group: group.__setitem__(2, []),
            lambda group: group[2].__setitem__(0, None),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                group = trajectories()
                mutation(group)
                self.assertEqual(failure_extension_reason(group), "base_contains_invalid")

    def test_invalid_takes_precedence_over_success(self):
        group = trajectories((1, 0, 0, 0, 0))
        group[-1][-1].reward = None
        self.assertEqual(failure_extension_reason(group), "base_contains_invalid")

    def test_reward_key_and_native_single_sample_trajectory(self):
        group = [Sample(i, reward={"combined": -1}) for i in range(5)]
        read = lambda sample: sample.reward["combined"]
        self.assertEqual(failure_extension_reason(group, reward_value=read),
                         "all_five_valid_failures")
        group[-1].reward["combined"] = None
        self.assertEqual(failure_extension_reason(group, reward_value=read),
                         "base_contains_invalid")

    def test_wrong_base_size_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "exactly five"):
            failure_extension_reason(trajectories()[:4])


class AsyncAllocationTests(unittest.IsolatedAsyncioTestCase):
    async def test_phases_seeds_indices_and_all_extras_after_early_success(self):
        group = slots()
        original = deepcopy(group)
        parameters = {"temperature": 0.8, "sampling_seed": 999}
        seeds = list(range(700, 708))
        calls, finished = [], []
        first_five_started = asyncio.Event()
        release_fifth = asyncio.Event()
        extras_started = asyncio.Event()
        release_last_extra = asyncio.Event()

        async def generate(sample, params):
            slot = sample.index - 100
            calls.append((slot, params.copy()))
            params["temperature"] = 999  # Each callback must own its dict.
            if slot == 4:
                first_five_started.set()
                await release_fifth.wait()
            if slot >= 5:
                self.assertEqual(set(range(5)) - set(finished), set())
                if slot == 7:
                    extras_started.set()
                    await release_last_extra.wait()
            finished.append(slot)
            # Native multi-turn indices differ from reserved trajectory indices.
            return [Sample(sample.index * 10, reward=1 if slot == 5 else 0),
                    Sample(sample.index * 10 + 1, reward=1 if slot == 5 else 0)]

        task = asyncio.create_task(generate_rollout_group(
            group, parameters, mode=ADAPTIVE, generate_one=generate,
            sampling_seeds=seeds))
        await first_five_started.wait()
        self.assertEqual([slot for slot, _ in calls], list(range(5)))
        release_fifth.set()
        await extras_started.wait()
        self.assertFalse(task.done())
        release_last_extra.set()
        result = await task
        self.assertEqual(len(result), 8)
        self.assertEqual([params["sampling_seed"] for _, params in calls], seeds)
        self.assertTrue(all(params["temperature"] == 0.8 for _, params in calls))
        self.assertEqual(group, original)
        self.assertEqual(parameters, {"temperature": 0.8, "sampling_seed": 999})
        for slot, trajectory in enumerate(result):
            self.assertEqual([sample.index for sample in trajectory],
                             [(100 + slot) * 10, (100 + slot) * 10 + 1])
            for sample in trajectory:
                self.assertEqual(sample.group_index, 19)
                record = sample.metadata[METADATA_KEY]
                self.assertEqual(record["slot"], slot)
                self.assertEqual(record["sampling_seed"], seeds[slot])
                self.assertEqual(record["extra_allocated"], 3)
                self.assertEqual(record["generated_count"], 8)
                self.assertEqual(record["reserved_sample_index"], 100 + slot)
                json.dumps(record)

    async def test_no_extension_for_success_invalid_or_native_abort(self):
        for label in ["mixed", "all_success", "removed", "null", "aborted"]:
            with self.subTest(label=label):
                calls = []

                async def generate(sample, params):
                    calls.append(sample.index)
                    result = deepcopy(sample)
                    if label == "all_success" or (label == "mixed" and sample.index == 102):
                        result.reward = 1
                    if sample.index == 102:
                        if label == "removed":
                            result.remove_sample = True
                        if label == "null":
                            result.reward = None
                        if label == "aborted":
                            result.status = Status.ABORTED
                    return [result]

                group = slots()
                result = await generate_rollout_group(
                    group, {}, mode=ADAPTIVE, generate_one=generate)
                self.assertEqual(calls, list(range(100, 105)))
                self.assertEqual(len(result), 5)
                self.assertFalse(any(METADATA_KEY in sample.metadata for sample in group[5:]))
                self.assertEqual(result[0][0].metadata[METADATA_KEY]["extra_allocated"], 0)

    async def test_uniform_eight_start_together_and_ignore_eligibility(self):
        calls = []
        all_started = asyncio.Event()

        async def generate(sample, params):
            calls.append(sample.index)
            if len(calls) == 8:
                all_started.set()
            await all_started.wait()
            sample.reward = None  # Uniform mode has no failure gate.
            return sample

        result = await asyncio.wait_for(generate_rollout_group(
            slots(), {}, mode=UNIFORM, generate_one=generate), timeout=2)
        self.assertEqual(len(result), 8)
        self.assertEqual(result[0].metadata[METADATA_KEY]["reason"], "uniform_eight")
        self.assertEqual(result[0].metadata[METADATA_KEY]["base_count"], 8)

    async def test_evaluation_bypasses_size_gate_and_annotations(self):
        group = slots()[:2]

        async def generate(sample, params):
            self.assertEqual(params["sampling_seed"], 400 + sample.index - 100)
            return sample

        result = await generate_rollout_group(
            group, {}, mode=ADAPTIVE, generate_one=generate,
            sampling_seeds=[400, 401], evaluation=True)
        self.assertEqual(result, group)
        self.assertFalse(any(METADATA_KEY in sample.metadata for sample in result))

    async def test_exceptions_cancel_and_await_owned_siblings(self):
        cleaned = asyncio.Event()

        async def generate(sample, params):
            if sample.index == 100:
                await asyncio.sleep(0)
                raise RuntimeError("native generator failed")
            try:
                await asyncio.Event().wait()
            finally:
                cleaned.set()

        with self.assertRaisesRegex(RuntimeError, "native generator failed"):
            await generate_rollout_group(slots(), {}, mode=ADAPTIVE, generate_one=generate)
        self.assertTrue(cleaned.is_set())

    async def test_external_cancellation_propagates_and_cleans_all_children(self):
        started, cleaned = [], []
        all_started = asyncio.Event()

        async def generate(sample, params):
            started.append(sample.index)
            if len(started) == 5:
                all_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaned.append(sample.index)

        task = asyncio.create_task(generate_rollout_group(
            slots(), {}, mode=ADAPTIVE, generate_one=generate))
        await all_started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertCountEqual(cleaned, started)

    async def test_state_abort_before_start_or_between_phases_never_runs_extras(self):
        for initial_abort in [True, False]:
            calls = []
            state = {"aborted": initial_abort}

            async def generate(sample, params):
                calls.append(sample.index)
                if len(calls) == 5:
                    state["aborted"] = True
                return sample

            with self.assertRaises(asyncio.CancelledError):
                await generate_rollout_group(
                    slots(), {}, mode=ADAPTIVE, generate_one=generate,
                    is_aborted=lambda: state["aborted"])
            self.assertEqual(len(calls), 0 if initial_abort else 5)

    async def test_bad_reserved_identity_or_seed_count_rejected_before_work(self):
        async def generate(sample, params):
            self.fail("Invalid reservation must not generate")

        mutations = [
            lambda group: group.pop(),
            lambda group: setattr(group[-1], "index", group[0].index),
            lambda group: setattr(group[-1], "group_index", 20),
            lambda group: setattr(group[-1], "index", None),
        ]
        for mutate in mutations:
            group = slots()
            mutate(group)
            with self.assertRaises(ValueError):
                await generate_rollout_group(group, {}, mode=ADAPTIVE, generate_one=generate)
        with self.assertRaisesRegex(ValueError, "seed vector"):
            await generate_rollout_group(
                slots(), {}, mode=ADAPTIVE, generate_one=generate, sampling_seeds=[0] * 5)


if __name__ == "__main__":
    unittest.main()
