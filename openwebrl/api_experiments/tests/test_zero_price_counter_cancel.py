"""Offline cancellation integration: real model, ledger and native HTTP leases."""
from copy import deepcopy
import contextlib
import io
import importlib.util
import json
from pathlib import Path
import tempfile
import sys
import types
import unittest
from unittest.mock import patch

API = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


fixtures = module('_counter_actor_fixtures', API/'tests/test_cached_luna_model.py')
fix = module('_counter_cleanup', API/'reproduce/zero_price_counter_cancel.py')
common = module('_counter_common', API/'reproduce/common.py')


class CounterCleanupTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(dir='/tmp', prefix='portable-counter-offline-')
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.actor_module = fixtures.load_actor()

    def actor(self, failure=TimeoutError, arm='luna_n3_cached_high30', mutate=None):
        original = fixtures.budget.ReservationBudget
        scoped = fix.actor_budget_class(original, self.root)
        pool = scoped(self.root/'budget/model', '1', 10)
        gate = fixtures.budget.SharedHTTPGate(self.root/'http-gate', 8)
        def client(payload, role):
            if mutate:
                mutate(pool, gate)
            if failure:
                raise failure('fixture cancellation; no provider called')
            return dict(object='response.input_tokens', input_tokens=23)
        actor = self.actor_module.CachedLunaModel(arm, self.root/'episodes/task', client,
            pool, http_gate=gate, model_name='gpt-6-luna', reasoning_effort='high',
            max_output_tokens=12288)
        actor.current_group_id = 'g'*32
        return actor, pool, gate

    def count(self, actor):
        return actor._request(dict(model='gpt-6-luna', input=[dict(role='user', content='actual observation')]),
                              'input_token_count', count=True)

    def test_real_timeout_preserves_unknown_id_call_and_zero_hold_then_next_task_admits(self):
        base = self.root
        for arm in ('luna_n3_cached_high30', 'luna_n5_cached_high60'):
            with self.subTest(arm=arm):
                self.root = base / arm
                self.root.mkdir()
                actor, pool, gate = self.actor(arm=arm)
                with self.assertRaises(TimeoutError):
                    self.count(actor)
                state = pool.snapshot()
                ident, row = next(iter(state['requests'].items()))
                self.assertEqual((state['calls'], row['status'], row['reserved_usd'],
                                  row['charged_or_reserved_usd']), (1, 'usage_unknown', '0', '0'))
                self.assertFalse(pool.halted())
                self.assertEqual(gate.snapshot()['occupied_count'], 0)
                before = deepcopy(row)
                events = [json.loads(x)['event'] for x in (pool.root/'events.jsonl').read_text().splitlines()]
                self.assertEqual(events, ['reserve', 'usage_unknown'])
                actor.client = lambda *_: dict(object='response.input_tokens', input_tokens=23)
                self.count(actor)
                self.assertEqual(pool.snapshot()['calls'], 2)
                self.assertEqual(pool.snapshot()['requests'][ident], before)
                with self.assertRaisesRegex(ValueError, 'Duplicate'):
                    pool.reserve(ident, 0)
        self.root = base

    def test_cancelled_error_uses_same_released_identity_guard(self):
        from asyncio import CancelledError
        actor, pool, _ = self.actor(failure=CancelledError)
        with self.assertRaises(CancelledError):
            self.count(actor)
        self.assertFalse(pool.halted())
        self.assertEqual(pool.snapshot()['unknown_calls'], 1)

    def test_billable_timeout_keeps_full_reserve_and_fatal_halt(self):
        actor, pool, _ = self.actor()
        with self.assertRaises(TimeoutError):
            actor._request(dict(model='gpt-6-luna', input=[]), 'candidate_0', bound=1000)
        row = next(iter(pool.snapshot()['requests'].values()))
        self.assertEqual(row['status'], 'usage_unknown')
        self.assertGreater(float(row['reserved_usd']), 0)
        self.assertEqual(row['charged_or_reserved_usd'], row['reserved_usd'])
        self.assertTrue(pool.halted())

    def test_non_cancellation_or_malformed_counter_remains_fatal(self):
        actor, pool, _ = self.actor(failure=ValueError)
        with self.assertRaises(ValueError):
            self.count(actor)
        self.assertTrue(pool.halted())
        self.assertEqual(pool.snapshot()['unknown_calls'], 1)

    def test_missing_release_evidence_cannot_suppress_native_halt(self):
        actor, pool, _ = self.actor()
        original = fix.read_regular
        def read(path):
            if Path(path).name == 'released.json':
                raise ValueError('fixture missing immutable release')
            return original(path)
        with patch.object(fix, 'read_regular', side_effect=read), self.assertRaises(TimeoutError):
            self.count(actor)
        self.assertTrue(pool.halted())

    def test_foreign_identity_remains_fatal(self):
        actor, pool, _ = self.actor()
        current = fix.live_identity()
        current['thread_id'] += 1
        with patch.object(fix, 'live_identity', return_value=current), self.assertRaises(TimeoutError):
            self.count(actor)
        self.assertTrue(pool.halted())

    def test_changed_canonical_payload_remains_fatal(self):
        def mutate(pool, gate):
            row = next(iter(pool.snapshot()['requests'].values()))
            path = Path(row['metadata']['request_path'])
            raw = json.loads(path.read_text())
            raw['request']['input'][0]['content'] = 'changed after durable admission'
            path.write_text(json.dumps(raw))
        actor, pool, _ = self.actor(mutate=mutate)
        with self.assertRaises(TimeoutError):
            self.count(actor)
        self.assertTrue(pool.halted())
        self.assertEqual(pool.snapshot()['unknown_calls'], 1)

    def test_suppression_does_not_mask_later_protocol_or_cleanup_halt(self):
        actor, pool, _ = self.actor()
        with self.assertRaises(TimeoutError):
            self.count(actor)
        self.assertFalse(pool.halted())
        pool.halt('Native candidate/selector parse failure; no unchanged retry')
        self.assertTrue(pool.halted())

    def test_factory_rejects_judge_browser_and_foreign_pools(self):
        cls = fix.actor_budget_class(fixtures.budget.ReservationBudget, self.root)
        for name in ('judge', 'browser', 'foreign'):
            with self.assertRaisesRegex(ValueError, 'own actor pool'):
                cls(self.root/'budget'/name, '1', 10)
            self.assertFalse((self.root/'budget'/name).exists())

    def plan(self):
        runtime = self.root/'runtime'
        runtime.mkdir()
        (runtime/'cached-selection-experiment.json').write_text('{}')
        return dict(runtime=str(runtime), provider='luna', arm='luna_n3_cached_high30',
            cached_selection_protocol='browsercode-fixed-cached-selection-v1',
            zero_price_counter_cancellation=common.counter_cancellation_binding())

    def test_source_binding_is_checked_before_sdk_or_output_creation(self):
        plan = self.plan()
        common.validate_counter_cancellation(plan)
        plan['zero_price_counter_cancellation']['source_sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'cleanup source changed'):
            common.execute(None, plan, [])
        self.assertFalse((self.root/'output').exists())

    def test_runtime_marker_prevents_removing_plan_guard(self):
        plan = self.plan()
        plan.pop('zero_price_counter_cancellation')
        plan.pop('cached_selection_protocol')
        with self.assertRaisesRegex(ValueError, 'cleanup source changed'):
            common.validate_counter_cancellation(plan)

    def test_cached_dry_plans_bind_cleanup_source_for_both_widths(self):
        portable = module('_counter_portable_fixtures', API/'tests/test_cached_selection_portable.py')
        fixture = portable.CachedPortableTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        for arm in ('luna_n3_cached_high30', 'luna_n5_cached_high60'):
            runtime = fixture.runtime(arm)
            portable.p.configure(runtime, arm)
            plan, tasks = portable.c.prepare(fixture.args(runtime, arm), 'luna')
            self.assertEqual(plan['zero_price_counter_cancellation'], common.counter_cancellation_binding())
            self.assertEqual(len(tasks), 1045)
            self.assertFalse(Path(plan['output']).exists())

    def test_frontdoor_records_timeout_invalid_and_continues_next_task(self):
        plan = self.plan()
        root = self.root / 'output'
        root.mkdir()
        plan.update(output=str(root), harness='browsercode', browser_start_cap=3)
        args = types.SimpleNamespace(model_usd='1', judge_usd='1', model_calls=10,
            judge_calls=10, wandb_mode='disabled', arm='luna_n3_cached_high30')
        clients = []
        class Client:
            def __init__(self, **kwargs):
                self.calls, self.closed = 0, False
                clients.append(self)
            def __call__(self, payload, role):
                self.calls += 1
                if self.calls == 1:
                    raise TimeoutError('offline episode deadline')
                return dict(object='response.input_tokens', input_tokens=23)
            def close(self):
                self.closed = True
        class BrowserBudget:
            def __init__(self, root, *caps):
                self.root = root
                root.mkdir(parents=True)
        worker = types.SimpleNamespace(ReservationBudget=fixtures.budget.ReservationBudget,
            Budget=BrowserBudget, PairedModel=self.actor_module.CachedLunaModel,
            preflight=lambda *args: None)
        observed_pools = []
        def episode(output, task, cell, actor, judge, budgets, slot):
            observed_pools.append(budgets)
            model = worker.PairedModel(args.arm, output/'episodes'/task['task_id'],
                actor, budgets['actor'], model_name='gpt-6-luna',
                reasoning_effort='high', max_output_tokens=12288)
            model.current_group_id = 'h'*32
            try:
                self.count(model)
                return dict(task_id=task['task_id'], valid=True, reward=0)
            except TimeoutError:
                return dict(task_id=task['task_id'], valid=False, reward=None)
        worker.episode = episode
        cleaned = []
        env = types.SimpleNamespace(cleanup_registered_groups=lambda output: cleaned.append(output))
        with patch.dict(sys.modules, {'openai': types.SimpleNamespace(OpenAI=Client)}), \
                patch.object(common.importlib, 'import_module', return_value=fixtures.budget), \
                patch.object(common.signal, 'signal'), contextlib.redirect_stdout(io.StringIO()):
            code = common._execute_worker(args, plan, [dict(task_id='first'), dict(task_id='second')],
                root, {'OPENAI_API_KEY': 'offline-test'}, 'OPENAI_API_KEY', None, worker, env, None)
        self.assertEqual(code, 1)  # First timeout remains invalid; no synthetic success.
        result = json.loads((root/'summary.json').read_text())
        self.assertEqual((result['recorded'], result['valid'], result['invalid']), (2, 1, 1))
        self.assertEqual(result['all_attempt_pools']['actor']['calls'], 2)
        actor_pool = observed_pools[0]['actor']
        self.assertFalse(actor_pool.halted())
        self.assertEqual(actor_pool.snapshot()['unknown_calls'], 1)
        self.assertIs(type(observed_pools[0]['judge']), fixtures.budget.ReservationBudget)
        self.assertTrue(all(client.closed for client in clients))
        self.assertEqual(cleaned, [root])

    def test_other_protocols_keep_original_budget_type(self):
        runtime = self.root/'runtime'
        runtime.mkdir()
        plan = dict(runtime=str(runtime), provider='luna')
        original = fixtures.budget.ReservationBudget
        self.assertIs(common.cached_actor_budget_type(original, self.root, plan), original)


if __name__ == '__main__':
    unittest.main(verbosity=2)
