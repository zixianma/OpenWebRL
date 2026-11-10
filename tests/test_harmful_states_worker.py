import asyncio
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from openwebrl.harmful_states import HarmfulStatesSandbox
from openwebrl.controlled_sft_worker import AttemptJournal
from openwebrl.harmful_states_worker import (
    BrowserPolicy, BudgetExhausted, DurableBudget, EpisodeBoundary, SAMPLING, browser_factory, schedule, usage_total,
)


class WorkerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.budget = DurableBudget(self.root/'budget', time.time()+60)
        self.world = HarmfulStatesSandbox(self.root/'world.sqlite')
        self.campaign = self.world.start_campaign('draft_review')
        self.episode = self.world.start_episode(self.campaign)
        self.journal = AttemptJournal(self.root/'events')

    def test_schedule_and_durable_failed_request_budget(self):
        rows = schedule()
        self.assertEqual(len(rows), 45)
        self.assertEqual(len({r['key'] for r in rows}), 45)
        self.budget.reserve('actor_requests', 1649)
        restored = DurableBudget(self.root/'budget', time.time()+60)
        self.assertEqual(restored.counts['actor_requests'], 1649)
        with self.assertRaises(BudgetExhausted):
            restored.reserve('actor_requests', 5)
        self.assertEqual(restored.counts['actor_requests'], 1649)

    async def policy(self, selector_raw='{"selection": 3}', fail_index=None):
        self.finished = []
        async def actor(url, payload, timeout):
            index = len(self.finished)
            self.finished.append(index)
            await asyncio.sleep(.01)
            if index == fail_index:
                raise TimeoutError('synthetic transport timeout')
            return dict(text=f'<think>proposal{index}</think><tool_call>{{"name":"done","arguments":{{}}}}</tool_call>',
                meta_info=dict(prompt_tokens=100, completion_tokens=12, cached_tokens=5,
                    output_token_logprobs=[[0.,42]], finish_reason={'type':'stop'}))
        async def selector(url, payload, timeout):
            self.assertEqual(len(payload['candidates']), 5)
            return dict(raw=selector_raw, input_tokens=600, output_tokens=5, seconds=.01)
        policy = BrowserPolicy(config=dict(max_steps=10, selector_endpoint='http://127.0.0.1:1234'),
            item=dict(case_id='draft_review',seed=43,method='selection5'),episode=0,
            sandbox=self.world,campaign=self.campaign,episode_id=self.episode,budget=self.budget,
            journal=self.journal,actor_post=actor,selector_post=selector)
        policy.set_training_context({'prompt_tokens':[1]*100})
        return policy

    async def call(self, policy):
        fake_generation = SimpleNamespace(_ensure_im_end_w_new_line=lambda text,tokens,logps:text)
        with patch.dict('sys.modules', {'openwebrl.generate_browser':fake_generation}):
            return await policy(infer=None,url='http://127.0.0.1:1235/generate',input_text=policy.prompt,
                sampling_params=dict(SAMPLING),images=['image'], observation=dict(screenshot=b'png',active_tab_url='http://127.0.0.1:1236'),
                history=[],task='Save a draft',task_id='draft_review',turn=0,timeout=30)

    async def test_five_metered_proposals_do_not_execute_backend_actions(self):
        policy = await self.policy()
        # Use the selector parser's canonical 1-based index schema.
        result, metadata = await self.call(policy)
        self.assertEqual(self.budget.counts['actor_requests'], 5)
        self.assertEqual(self.budget.counts['selector_requests'], 1)
        self.assertIn('proposal2', result[0])
        self.assertEqual(metadata['selected_index'], 2)
        self.assertEqual(self.world.verdict(self.campaign)['dispatch_count'], 0)
        events=[json.loads(p.read_text()) for p in sorted((self.root/'events').glob('[0-9]*.json'))]
        usage = usage_total(events, 'actor')
        self.assertEqual(usage['input_tokens'], 500)
        self.assertEqual(usage['output_tokens'], 60)
        self.assertEqual(usage['unknown_usage_requests'], 0)

    async def test_bad_selector_executes_nothing_and_stays_charged(self):
        policy = await self.policy(selector_raw='not JSON')
        with self.assertRaises(ValueError):
            await self.call(policy)
        self.assertEqual(self.budget.counts['actor_requests'],5)
        self.assertEqual(self.budget.counts['selector_requests'],1)
        self.assertEqual(self.world.verdict(self.campaign)['dispatch_count'],0)
        self.assertIsNotNone(policy.error)

    async def test_all_siblings_finish_when_one_request_fails(self):
        policy = await self.policy(fail_index=1)
        with self.assertRaises(TimeoutError):
            await self.call(policy)
        self.assertEqual(len(self.finished),5)
        events=[json.loads(p.read_text()) for p in sorted((self.root/'events').glob('[0-9]*.json'))]
        self.assertEqual(usage_total(events,'actor')['unknown_usage_requests'],1)
        self.assertEqual(self.budget.counts['selector_requests'],0)

    async def test_browser_limit_prevents_more_model_work(self):
        policy = await self.policy()
        policy.environment = SimpleNamespace(boundary_reason='operation_limit')
        with self.assertRaises(EpisodeBoundary):
            await self.call(policy)
        self.assertEqual(self.budget.counts['actor_requests'],0)
        self.assertIsNone(policy.error)

    async def test_factory_binds_controller_and_preserves_policy_text(self):
        policy = await self.policy()
        environment = SimpleNamespace(boundary_reason=None)
        async def make_environment(url, policy_text, tools, journal, **kwargs):
            self.assertEqual(policy_text, 'native GUI policy')
            return environment
        original = object()
        generation = SimpleNamespace(_create_env=original,
            _load_local_resources=lambda *args: ({'intent':'Save draft'}, [], 'native GUI policy'))
        module = SimpleNamespace(make_environment=make_environment)
        holder = {}
        with patch.dict('sys.modules', {'openwebrl.harmful_states_browser':module}):
            with browser_factory(generation, server=SimpleNamespace(url='http://127.0.0.1:1234'),
                    journal=self.journal,config=dict(max_browser_operations_per_episode=30,deadline_unix=time.time()+60),
                    holder=holder,policy=policy):
                result, task = await generation._create_env('draft_review', {})
        self.assertIs(result, environment)
        self.assertIs(policy.environment, environment)
        self.assertIs(generation._create_env, original)


if __name__ == '__main__':
    unittest.main()
