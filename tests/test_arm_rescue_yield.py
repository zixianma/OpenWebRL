import asyncio
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from openwebrl.arm_rescue_yield import eligible_tasks, summarize_retries, MeteredSelector, run_pilot
from slime.utils.eval_config import EvalDatasetConfig


def record(task='a', attempt=0, mode='actor', valid=True, reward=0.):
    return dict(task_id=task, attempt=attempt, mode=mode, valid=valid, reward=reward,
        policy_id='frozen', judge_id='gpt-4.1/action_history', actor_output_tokens=10,
        actor_requests=1, elapsed_seconds=1., selector_calls=int(mode=='arm'), selector_fallback_turns=0)


class Eligibility(unittest.TestCase):
    def test_invalid_success_and_format_errors_are_not_five_failures(self):
        rows = [record(task, i) for task in ['good', 'invalid', 'success', 'format'] for i in range(5)]
        rows[5]['valid'] = False
        rows[10]['reward'] = 1.
        rows[15]['reward'] = -1.
        self.assertEqual(eligible_tasks(rows, ['good', 'invalid', 'success', 'format'], 'frozen'), ['good'])

    def test_incomplete_duplicate_and_wrong_policy_screens_fail(self):
        rows = [record(attempt=i) for i in range(5)]
        for modified in [rows[:-1], [rows[0], *rows[1:4], rows[0]],
                         [dict(rows[0], policy_id='other'), *rows[1:]]]:
            with self.subTest(rows=modified), self.assertRaises(ValueError):
                eligible_tasks(modified, ['a'], 'frozen')


class YieldMetrics(unittest.TestCase):
    def test_one_retry_and_five_retry_controls_are_distinct(self):
        rows = [record(task=t, mode=m) for t in ['a', 'b']
                for m in ['arm', *[f'actor{i}' for i in range(5)]]]
        rows[0]['reward'] = 1.  # ARM rescues a.
        rows[-1]['reward'] = 1.  # Actor's fifth retry rescues b, not its first.
        summary = summarize_retries(rows, ['a', 'b'])
        self.assertEqual(summary['by_mode']['arm']['successes'], 1)
        self.assertEqual(summary['by_mode']['actor0']['successes'], 0)
        self.assertEqual(summary['paired']['actor5_any_success'], 1)
        self.assertEqual(summary['paired']['arm_only_vs_actor5'], 1)
        self.assertEqual(summary['actor5_output_tokens'], 100)

    def test_missing_retry_cannot_silently_shrink_denominator(self):
        with self.assertRaises(ValueError):
            summarize_retries([record(mode='arm')], ['a'])
        self.assertIsNone(summarize_retries([], [])['by_mode']['arm']['success_rate'])


class CostAccounting(unittest.IsolatedAsyncioTestCase):
    async def test_all_candidate_tokens_count_not_just_executed_tokens(self):
        class Selector:
            mode = 'selection'
            async def __call__(self, infer, **kwargs):
                outputs = await asyncio.gather(*(infer(i) for i in range(5)))
                return outputs[2], {'fallback': None}
        async def infer(i):
            return ('text', list(range(i+1)), [], 'stop')
        meter = MeteredSelector(Selector())
        await meter(infer=infer)
        self.assertEqual((meter.actor_requests, meter.actor_output_tokens, meter.selector_calls), (5, 15, 1))

    async def test_failed_smoke_never_scales_to_screening(self):
        fake = ModuleType('slime.rollout.sglang_rollout')
        calls = []
        async def evaluate(args, rollout_id, dataset):
            calls.append(dataset.name)
            return {dataset.name: {}}
        fake.eval_rollout_single_dataset = evaluate
        output_module = ModuleType('slime.rollout.base_types')
        output_module.RolloutFnEvalOutput = SimpleNamespace
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tasks = root/'tasks.jsonl'
            tasks.write_text(json.dumps({'prompt':'task', 'metadata': {'task_id': 'a'}})+'\n')
            config = root/'config.json'
            config.write_text(json.dumps(dict(tasks=str(tasks), output=str(root), seed=42,
                policy_id='frozen', max_rescue_tasks=8)))
            args = SimpleNamespace(eval_datasets=[EvalDatasetConfig(name='screen', path=str(tasks),
                n_samples_per_eval_prompt=5)])
            with patch.dict(sys.modules, {'slime.rollout.sglang_rollout': fake,
                    'slime.rollout.base_types': output_module}), patch.dict(os.environ,
                    OPENWEBRL_RESCUE_YIELD_CONFIG=str(config)):
                with self.assertRaisesRegex(RuntimeError, 'smoke pair'):
                    await run_pilot(args, 0)
            self.assertEqual(calls, ['arm-rescue-smoke'])
            self.assertFalse(json.loads((root/'smoke-gate.json').read_text())['passed'])


if __name__ == '__main__':
    unittest.main()
