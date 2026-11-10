"""Offline safety checks: no browser, GPU or paid API requests."""
import asyncio
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from openwebrl.browser_actor_protocol import browser_harness_protocol
from openwebrl.luna_qwen_metrics import digest
from openwebrl.luna_qwen_policy import Budget
from openwebrl.reasoning_actor_policy import metric_prices
from openwebrl.sol_serial_parallel_eval import (LIMITS, RESOURCES, MeteredSolAPI,
    SingleAttemptJudge, paired_schedule, validate_plan)


def plan_fixture():
    tasks = [str(i) for i in range(100)]
    protocol = dict(actor='gpt-6.1-sol', serial_effort='high', serial_max_output_tokens=12288,
        parallel_effort='low', parallel_max_output_tokens=4096, selector_effort='medium',
        selector_max_output_tokens=2048, candidates=3, resize_output_coords=False,
        browser_coordinate_space='viewport_pixels', judge='o4-mini-2025-04-16', max_steps=30,
        **browser_harness_protocol())
    return dict(protocol=protocol, family='luna', resources=RESOURCES, limits=LIMITS, workers=6,
                output='/tmp/unexecuted-sol100', pricing=metric_prices('gpt-6.1-sol'),
                task_ids=tasks, seed=17, schedule=paired_schedule(tasks, 17), wandb_project='openwebrl-evals')


def high_request():
    return dict(model='gpt-6.1-sol', reasoning={'effort': 'high'}, max_output_tokens=12288,
        store=False, service_tier='default', tool_choice='required', parallel_tool_calls=False,
        include=['reasoning.encrypted_content'],
        input=[dict(role='user', content='test')], tools=[])


class FrozenProtocolTests(unittest.TestCase):
    def test_randomized_pairing_is_stable_and_complete(self):
        plan = plan_fixture()
        self.assertEqual(validate_plan(plan, dict(output=plan['output'], plan_sha256=digest(plan))), 'luna')
        self.assertEqual(plan['schedule'], paired_schedule(plan['task_ids'][::-1], 17))
        self.assertEqual({tuple(r['modes']) for r in plan['schedule']},
                         {('serial_high', 'parallel3_low'), ('parallel3_low', 'serial_high')})

    def test_no_budget_or_protocol_expansion(self):
        for key, value in [('max_steps', 31), ('candidates', 4), ('serial_effort', 'medium')]:
            plan = deepcopy(plan_fixture())
            plan['protocol'][key] = value
            with self.assertRaises(ValueError):
                validate_plan(plan, dict(output=plan['output'], plan_sha256=digest(plan)))


class MeteringTests(unittest.IsolatedAsyncioTestCase):
    async def test_failed_request_remains_reserved_and_halts_without_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            create = AsyncMock(side_effect=TimeoutError('no response'))
            budget = Budget(Path(tmp) / 'budget', 5., 2)
            meter = MeteredSolAPI(SimpleNamespace(responses=SimpleNamespace(create=create)),
                                  budget, Path(tmp) / 'episode', metric_prices('gpt-6.1-sol'))
            with self.assertRaises(TimeoutError):
                await meter.create('actor_high', **high_request())
            state = json.loads((budget.root / 'ledger.json').read_text())
            self.assertEqual(state['calls'], 1)
            self.assertGreater(state['charged_or_reserved_usd'], 0)
            self.assertTrue((budget.root / 'halt.json').exists())
            with self.assertRaises(RuntimeError):
                await meter.create('actor_high', **high_request())
            self.assertEqual(create.await_count, 1)
            receipt = json.loads(next((Path(tmp) / 'episode' / 'requests').glob('*.json')).read_text())
            self.assertEqual(receipt['status'], 'usage_unknown')

    async def test_settled_usage_counts_reasoning_as_subset(self):
        with tempfile.TemporaryDirectory() as tmp:
            usage = dict(input_tokens=20, output_tokens=10,
                         input_tokens_details=dict(cached_tokens=0, cache_write_tokens=0),
                         output_tokens_details=dict(reasoning_tokens=6))
            response = SimpleNamespace(model='gpt-6.1-sol', status='completed',
                usage=SimpleNamespace(model_dump=lambda: usage), model_dump=lambda: dict(model='gpt-6.1-sol'))
            create = AsyncMock(return_value=response)
            budget = Budget(Path(tmp) / 'budget', 5., 2)
            meter = MeteredSolAPI(SimpleNamespace(responses=SimpleNamespace(create=create)),
                                  budget, Path(tmp) / 'episode', metric_prices('gpt-6.1-sol'))
            await meter.create('actor_high', **high_request())
            state = json.loads((budget.root / 'ledger.json').read_text())
            self.assertAlmostEqual(state['charged_or_reserved_usd'], (20*2 + 10*10)/1e6)
            self.assertEqual(state['calls'], 1)

    async def test_invalid_settings_rejected_before_reservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            budget = Budget(Path(tmp) / 'budget', 5., 2)
            meter = MeteredSolAPI(None, budget, Path(tmp) / 'episode', metric_prices('gpt-6.1-sol'))
            request = high_request()
            request['reasoning'] = {'effort': 'max'}
            with self.assertRaises(ValueError):
                await meter.create('actor_high', **request)
            self.assertFalse((budget.root / 'ledger.json').exists())

    async def test_judge_cannot_physically_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / 'episode'
            (directory / 'requests').mkdir(parents=True)
            (directory / 'requests' / 'judge-00001.json').write_text('{}')
            judge = SingleAttemptJudge(None, None)
            judge.directory = directory
            with self.assertRaisesRegex(RuntimeError, 'retry'):
                await judge.create()


if __name__ == '__main__':
    unittest.main()
