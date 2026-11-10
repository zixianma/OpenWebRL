import asyncio
import json
from pathlib import Path
import tempfile
import unittest

from openwebrl.harmful_native_pilot import (
    BrowserReply, BudgetStop, CampaignLedger, Ledger, METERS, ModelReply, Observation, ProtocolError,
    cold_actor_flops, feature_sidecar_flops, parse_proposal, run_policy, schedule,
)


def call(name="write", **args):
    return '<think>Use observed UI.</think><tool_call>' + json.dumps(
        {"name": name, "arguments": args}) + '</tool_call><|im_end|>'


def caps(**changes):
    return dict({k: 10000 for k in METERS}, **changes)


class World:
    def __init__(self, recoverable=False):
        self.executed, self.observations, self.recoverable = [], 0, recoverable

    async def observe(self):
        self.observations += 1
        return Observation("same-world", "http://127.0.0.1:8888/orders", "a" * 64, "Orders")

    async def execute(self, action):
        self.executed.append(action)
        return BrowserReply("native UI response", "same-world",
            "recoverable_tool_failure" if self.recoverable else "returned")


class Actor:
    def __init__(self, responses=None, fail_at=None):
        self.requests, self.responses, self.fail_at = [], responses, fail_at

    def bound(self, request):
        return dict(input_tokens=10, output_tokens=10, modeled_flops=100, actor_proposals=1)

    async def generate(self, request):
        index = len(self.requests)
        self.requests.append(json.loads(request))
        await asyncio.sleep(0)
        if index == self.fail_at:
            raise TimeoutError("transport unknown")
        text = self.responses[index] if self.responses else call(message=f"candidate{index}")
        return ModelReply(text, 2, 1, 10)


class Selector:
    def __init__(self, response='{"selection":3}'):
        self.response, self.requests = response, []

    def bound(self, request, count):
        assert count == 5
        return dict(input_tokens=50, output_tokens=10, modeled_flops=200, selector_calls=1)

    async def select(self, request, candidates):
        self.requests.append((json.loads(request), json.loads(candidates)))
        return ModelReply(self.response, 20, 1, 100)


class ParserTests(unittest.TestCase):
    def test_native_actor_sequence_and_literal_markup(self):
        text = call(message='literal </tool_call> </think> `python`')
        self.assertEqual(parse_proposal(text, 'http://127.0.0.1:8888')[0]['arguments']['message'],
                         'literal </tool_call> </think> `python`')
        two = call(message='one').removesuffix('<|im_end|>') + '<tool_call>{"name":"press_keys","arguments":{"keys":["Enter"]}}</tool_call>'
        self.assertEqual(len(parse_proposal(two, 'http://127.0.0.1:8888')), 2)
        implicit = call(message='native prefill').removeprefix('<think>')
        self.assertEqual(parse_proposal(implicit, 'http://127.0.0.1:8888')[0]['name'], 'write')

    def test_reject_executable_python_batches_and_unknown_tools(self):
        bad = ["import os; os.system('touch /tmp/no')", "```python\npage.click('Cancel')\n```",
               '<tool_call>{"name":"evaluate","arguments":{"script":"1+1"}}</tool_call>',
               '<tool_call>{"name":"write","arguments":{"message":"a"},"name":"done"}</tool_call>',
               call('click', point_2d=[float('nan'), 2]), call('click', point_2d=[1, 2], clicks=2),
               call('goto_url', url='javascript:alert(1)'), call('goto_url', url='https://outside.test'),
               call('press_keys', keys=['Exec']), call('wait', seconds=999),
               call('done', response='finished').removesuffix('<|im_end|>') + call(message='after'),
               '[{"name":"write","arguments":{"message":"a"}}]']
        for text in bad:
            with self.subTest(text=text), self.assertRaises(ProtocolError):
                parse_proposal(text, 'http://127.0.0.1:8888')

    def test_fixed_panel_is_18_unique_worlds(self):
        rows = schedule()
        self.assertEqual(len(rows), 18)
        self.assertEqual(len({r['key'] for r in rows}), 18)
        self.assertEqual({r['task_id'] for r in rows}, {470, 471, 473})

    def test_sidecar_charges_feature_forwards_not_only_head(self):
        repo = Path(__file__).resolve().parents[1]
        # Small architecture fixture retains full GQA/MLP/vision accounting.
        config = dict(text_config=dict(hidden_size=8, num_attention_heads=2, head_dim=4,
            num_key_value_heads=1, intermediate_size=16, num_hidden_layers=2, vocab_size=32),
            vision_config=dict(hidden_size=8, intermediate_size=16, depth=2, spatial_merge_size=2,
                in_channels=3, temporal_patch_size=2, patch_size=2, out_hidden_size=8,
                deepstack_visual_indexes=[1]))
        cost = feature_sidecar_flops(config, 10, [3]*5, [1,4,4], d_model=8)
        self.assertEqual(cost['total'], sum(v for k,v in cost.items() if k != 'total'))
        self.assertGreater(cost['total'], cost['head_dense'])
        longer = feature_sidecar_flops(config, 10, [6]*5, [1,4,4], d_model=8)
        self.assertGreater(longer['candidate_decoder'], cost['candidate_decoder'])
        self.assertEqual(longer['prefix_decoder'], cost['prefix_decoder'])
        self.assertGreater(cold_actor_flops(config, 10, 3, [1,4,4])['total'], 0)


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'ledger.jsonl'

    def test_unknown_cost_survives_restart_and_deadline_cannot_reset(self):
        led = Ledger(self.path, caps(), 100, clock=lambda: 1)
        led.reserve(dict(input_tokens=8, actor_proposals=1), {'role': 'actor'})
        led.close()
        restored = Ledger(self.path, caps(), 100, clock=lambda: 1)
        self.assertEqual(restored.totals()['charged']['input_tokens'], 8)
        self.assertEqual(restored.totals()['metered']['input_tokens'], 0)
        self.assertEqual(restored.totals()['unknown_reservations'], 1)
        restored.close()
        with self.assertRaises(ProtocolError):
            Ledger(self.path, caps(), 200, clock=lambda: 1)

    def test_two_writers_and_oversettlement_rejected(self):
        led = Ledger(self.path, caps(), 100, clock=lambda: 1)
        self.addCleanup(led.close)
        with self.assertRaises(BlockingIOError):
            Ledger(self.path, caps(), 100, clock=lambda: 1)
        rid = led.reserve(dict(input_tokens=8), {})
        with self.assertRaises(ProtocolError):
            led.settle(rid, dict(input_tokens=9))
        self.assertEqual(led.totals()['charged']['input_tokens'], 9)
        self.assertEqual(led.events[-1]['kind'], 'usage_overrun')

    def test_compute_browser_deadlines_are_independent(self):
        led = Ledger(self.path, caps(modeled_flops=20, browser_operations=1), 100, clock=lambda: 1)
        self.addCleanup(led.close)
        with self.assertRaisesRegex(BudgetStop, 'modeled_flops'):
            led.reserve(dict(modeled_flops=21), {})
        with self.assertRaisesRegex(BudgetStop, 'browser_operations'):
            led.reserve(dict(browser_operations=2), {})
        led.clock = lambda: 100
        with self.assertRaisesRegex(BudgetStop, 'wall_deadline'):
            led.reserve({}, {})

    def test_clock_rewind_cannot_extend_work_or_restart_cutoff(self):
        wall, mono = [10], [1000]
        led = Ledger(self.path, caps(), 20, clock=lambda: wall[0], monotonic=lambda: mono[0])
        self.assertEqual(led.remaining_seconds(), 10)
        wall[0], mono[0] = 1, 1009
        self.assertEqual(led.remaining_seconds(), 1)
        led.close()
        restored = Ledger(self.path, caps(), 20, clock=lambda: wall[0], monotonic=lambda: mono[0])
        self.addCleanup(restored.close)
        self.assertEqual(restored.remaining_seconds(), 1)
        mono[0] = 1010
        with self.assertRaises(BudgetStop):
            restored.reserve({}, {})

    def test_home_symlink_and_preserved_native_evidence_rejected(self):
        from openwebrl.harmful_native_pilot import PRESERVED_NATIVE
        link = Path(self.temp.name) / 'escape'
        link.symlink_to('/gpfs/home/zixianma', target_is_directory=True)
        for path in (link / 'forbidden.jsonl', PRESERVED_NATIVE / 'forbidden.jsonl'):
            with self.assertRaises(ProtocolError):
                Ledger(path, caps(), 100)

    def test_global_cap_shared_across_worlds_and_overrun_actuals_preserved(self):
        global_budget = Ledger(Path(self.temp.name)/'global', caps(actor_proposals=1), 100, clock=lambda: 1)
        first = Ledger(Path(self.temp.name)/'one', caps(), 90, clock=lambda: 1)
        second = Ledger(Path(self.temp.name)/'two', caps(), 90, clock=lambda: 1)
        for obj in (global_budget, first, second): self.addCleanup(obj.close)
        a = CampaignLedger(first, global_budget, world_key='one', world_slots=['one','two'])
        b = CampaignLedger(second, global_budget, world_key='two', world_slots=['one','two'])
        rid = a.reserve(dict(actor_proposals=1, input_tokens=2), {'role':'actor'})
        with self.assertRaises(BudgetStop):
            b.reserve(dict(actor_proposals=1), {'role':'actor'})
        with self.assertRaises(ProtocolError):
            a.settle(rid, dict(actor_proposals=1, input_tokens=3))
        self.assertEqual(a.totals()['charged']['input_tokens'], 3)
        self.assertEqual(global_budget.totals()['charged']['input_tokens'], 3)
        with self.assertRaises(ProtocolError):
            CampaignLedger(second, global_budget, world_key='one', world_slots=['one','two'])
        with self.assertRaises(ProtocolError):
            CampaignLedger(second, global_budget, world_key='three', world_slots=['one','two','three'])


class ControllerTests(unittest.IsolatedAsyncioTestCase):
    def linked(self, ledger, directory):
        campaign = Ledger(Path(directory)/'global', caps(), 100, clock=lambda: 1)
        self.addCleanup(campaign.close)
        return CampaignLedger(ledger, campaign, world_key='fixture', world_slots=['fixture'])

    async def run_case(self, method='ordinary', actor=None, selector=None, world=None, limits=None, decisions=1):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.ledger = Ledger(Path(temp.name) / 'ledger.jsonl', limits or caps(), 100, clock=lambda: 1)
        self.addCleanup(self.ledger.close)
        self.ledger = self.linked(self.ledger, temp.name)
        self.actor, self.selector, self.world = actor or Actor(), selector or Selector(), world or World()
        return await run_policy(item=dict(task_id=470, seed=43, method=method), task='Cancel requested order',
            world=self.world, actor=self.actor, selector=self.selector, ledger=self.ledger,
            attempt_decisions=decisions)

    async def test_selection_charges_all_five_and_executes_only_choice(self):
        result = await self.run_case('selection5')
        self.assertEqual(len(self.actor.requests), 5)
        self.assertEqual(len(self.selector.requests), 1)
        self.assertEqual([a['arguments']['message'] for a in self.world.executed], ['candidate2'])
        self.assertEqual(result['accounting']['metered']['input_tokens'], 30)
        self.assertEqual(result['accounting']['metered']['modeled_flops'], 150)
        states = [r['state'] for r in self.actor.requests]
        self.assertTrue(all(s == states[0] for s in states))
        self.assertEqual(self.selector.requests[0][0], states[0])
        self.assertIsNone(result['safe_success'])

    async def test_retry_keeps_world_and_entire_prior_action_history(self):
        actor = Actor([call(message='wrong'), call(message='correct'), call('done', response='complete')])
        result = await self.run_case('retries5', actor=actor)
        self.assertEqual(result['attempts'], 3)
        self.assertEqual([a['arguments']['message'] for a in self.world.executed], ['wrong', 'correct'])
        self.assertEqual(len(actor.requests[2]['state']['executed_history']), 2)
        self.assertEqual({r['state']['observation']['world_id'] for r in actor.requests}, {'same-world'})
        self.assertNotIn('goal_success', actor.requests[2]['state'])

    async def test_done_stops_all_arms_without_post_success_retry(self):
        for method in ('ordinary', 'retries5', 'selection5'):
            with self.subTest(method=method):
                actor = Actor([call('done', response='finished')] * 5)
                result = await self.run_case(method, actor=actor, decisions=5)
                self.assertEqual(result['stop_reason'], 'actor_done_unverified')
                self.assertEqual(result['attempts'], 1)
                self.assertEqual(len(actor.requests), 5 if method == 'selection5' else 1)
                self.assertEqual(self.world.executed, [])

    async def test_recoverable_browser_failure_retries_without_reset(self):
        result = await self.run_case('retries5', world=World(recoverable=True), decisions=10)
        self.assertEqual(result['attempts'], 5)
        self.assertEqual(result['returned_browser_actions'], 5)

    async def test_failed_proposals_all_awaited_charged_and_no_browser_action(self):
        result = await self.run_case('selection5', actor=Actor(fail_at=2))
        self.assertEqual(len(self.actor.requests), 5)
        self.assertEqual(result['accounting']['charged']['actor_proposals'], 5)
        self.assertEqual(result['accounting']['unknown_reservations'], 1)
        self.assertEqual(self.world.executed, [])
        self.assertEqual(self.selector.requests, [])

    async def test_invalid_selector_never_falls_back(self):
        for response in ('{"selection":0}', '{"selection":true}', '{}', 'bad',
                         '{"selection":1,"selection":3}'):
            result = await self.run_case('selection5', selector=Selector(response))
            self.assertEqual(result['accounting']['charged']['selector_calls'], 1)
            self.assertEqual(self.world.executed, [])

    async def test_full_candidate_panel_budget_checked_before_any_model_dispatch(self):
        result = await self.run_case('selection5', limits=caps(modeled_flops=699))
        self.assertEqual(self.actor.requests, [])
        self.assertEqual(self.world.executed, [])
        self.assertIn('modeled_flops', result['stop_reason'])

    async def test_browser_cap_prevents_more_inference_and_never_resets_between_retries(self):
        result = await self.run_case('retries5', limits=caps(browser_operations=2))
        self.assertEqual(len(self.actor.requests), 2)
        self.assertEqual(len(self.world.executed), 2)
        self.assertIn('browser_operations', result['stop_reason'])

    async def test_uncertain_browser_dispatch_is_charged_and_not_retried(self):
        class UnknownWorld(World):
            async def execute(self, action):
                self.executed.append(action)
                raise TimeoutError('native side effect may have happened')
        result = await self.run_case('retries5', world=UnknownWorld(), decisions=5)
        self.assertEqual(len(self.actor.requests), 1)
        self.assertEqual(result['accounting']['charged']['browser_operations'], 1)
        self.assertEqual(result['accounting']['unknown_reservations'], 1)
        self.assertEqual(result['attempted_browser_actions'], 1)
        self.assertEqual(result['returned_browser_actions'], 0)
        self.assertTrue(result['unknown_native_effect_possible'])

    async def test_world_change_rejected(self):
        class ChangedWorld(World):
            async def observe(self):
                obs = await super().observe()
                return obs if self.observations == 1 else Observation('reset', obs.url, obs.screenshot_sha256, obs.visible_text)
        result = await self.run_case('retries5', world=ChangedWorld())
        self.assertEqual(len(self.actor.requests), 1)
        self.assertIn('invalid', result['stop_reason'])

    async def test_deadline_crossed_in_browser_reservation_dispatches_nothing(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        now = [1]
        led = Ledger(Path(temp.name)/'budget', caps(), 100, clock=lambda: now[0])
        self.addCleanup(led.close)
        original = led.reserve
        def reserve(upper, context):
            rid = original(upper, context)
            if context.get('role') == 'browser':
                now[0] = 100
            return rid
        led.reserve = reserve
        linked = self.linked(led, temp.name)
        world = World()
        result = await run_policy(item=dict(task_id=470, seed=43, method='ordinary'), task='task',
            world=world, actor=Actor(), selector=None, ledger=linked, attempt_decisions=1)
        self.assertEqual(world.executed, [])
        self.assertEqual(result['stop_reason'], 'budget:wall_deadline')

    async def test_initial_observation_timeout_is_retained_ungraded(self):
        class StalledWorld(World):
            async def observe(self):
                await asyncio.sleep(10)
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        led = Ledger(Path(temp.name)/'budget', caps(), 100, clock=lambda: 1)
        self.addCleanup(led.close)
        linked = self.linked(led, temp.name)
        result = await run_policy(item=dict(task_id=470, seed=43, method='ordinary'), task='task',
            world=StalledWorld(), actor=Actor(), selector=None, ledger=linked, call_timeout=.01)
        self.assertIn('TimeoutError', result['stop_reason'])
        self.assertIsNone(result['safe_success'])
        self.assertEqual(led.events[-1]['kind'], 'policy_stopped')


if __name__ == '__main__':
    unittest.main()
