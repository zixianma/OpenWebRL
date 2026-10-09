import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from openwebrl.rollout_sampling_budget import (
    BudgetLedger, BudgetedJudge, EXPECTED_CAPS, atomic, reserve_browser, wrap,
)


class FakeClient:
    max_retries = 0

    def __init__(self, error=False):
        self.calls = []
        self.error = error
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise RuntimeError('synthetic provider failure')
        return SimpleNamespace(usage=SimpleNamespace(model_dump=lambda: dict(
            prompt_tokens=100, completion_tokens=20)))

    async def close(self):
        self.closed = True


class SamplingBudgetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.control = Path(self.tmp.name)
        atomic(self.control / 'approval-request.json', dict(approved=True, **EXPECTED_CAPS))
        self.ledger = BudgetLedger(self.control)

    def state(self):
        return json.loads((self.ledger.root / 'ledger.json').read_text())

    def test_requires_explicit_exact_approval_and_no_sdk_retries(self):
        for override in (dict(approved=False), dict(judge_cap_usd=51),
                         dict(judge_call_cap=6001), dict(browser_dispatch_cap=12001),
                         dict(browser_concurrency=65)):
            atomic(self.control / 'approval-request.json', dict(approved=True, **EXPECTED_CAPS) | override)
            with self.assertRaises((ValueError, PermissionError)):
                BudgetLedger(self.control)
        client = FakeClient()
        client.max_retries = 2
        with self.assertRaises(ValueError):
            BudgetedJudge(client, self.control)

    def test_unchanged_request_and_usage_settlement(self):
        client = FakeClient()
        guard = BudgetedJudge(client, self.control)
        kwargs = dict(model='gpt-4.1', messages=[dict(role='user', content='judge this')], temperature=0)
        asyncio.run(guard.create(**kwargs))
        self.assertEqual(client.calls, [kwargs])
        self.assertEqual(self.state()['calls'], 1)
        self.assertAlmostEqual(self.state()['charged_or_reserved_usd'], .00036)
        asyncio.run(guard.close())
        self.assertTrue(client.closed)

    def test_failures_and_restarts_preserve_charge(self):
        client = FakeClient(error=True)
        guard = BudgetedJudge(client, self.control)
        kwargs = dict(model='gpt-4.1', messages=[])
        with self.assertRaises(RuntimeError):
            asyncio.run(guard.create(**kwargs))
        initial = self.state()['charged_or_reserved_usd']
        replacement = BudgetLedger(self.control)
        replacement.reserve(.5)
        self.assertAlmostEqual(self.state()['charged_or_reserved_usd'], initial + .5)
        self.assertEqual(self.state()['calls'], 2)

    def test_received_usage_survives_interruption_before_settlement(self):
        guard = BudgetedJudge(FakeClient(), self.control)
        with patch.object(guard, 'settle', side_effect=RuntimeError('interrupted settlement')):
            with self.assertRaises(RuntimeError):
                asyncio.run(guard.create(model='gpt-4.1', messages=[]))
        receipt = json.loads((guard.root / 'request-000001.json').read_text())
        evidence = json.loads((guard.root / 'usage-000001.json').read_text())
        self.assertEqual(receipt['status'], 'reserved')
        self.assertEqual(evidence['request_id'], receipt['request_id'])
        self.assertEqual(evidence['reserved_usd'], receipt['reserved_usd'])
        self.assertEqual(evidence['usage'], dict(prompt_tokens=100, completion_tokens=20))
        self.assertEqual(self.state()['charged_or_reserved_usd'], receipt['reserved_usd'])

    def test_spend_and_call_and_browser_limits_fail_before_dispatch(self):
        for field, limit, action in (
            ('charged_or_reserved_usd', EXPECTED_CAPS['judge_cap_usd'], lambda: self.ledger.reserve(.01)),
            ('calls', EXPECTED_CAPS['judge_call_cap'], lambda: self.ledger.reserve(.01)),
            ('browser_dispatches', 12000, lambda: self.ledger.reserve_browser()),
        ):
            with tempfile.TemporaryDirectory() as directory:
                control = Path(directory)
                atomic(control / 'approval-request.json', dict(approved=True, **EXPECTED_CAPS))
                previous = self.ledger
                self.ledger = BudgetLedger(control)
                state = self.ledger._state()
                state[field] = limit
                atomic(self.ledger.root / 'ledger.json', state)
                with self.assertRaises(RuntimeError):
                    action()
                self.assertEqual(self.state()[field], limit)
                self.assertTrue((self.ledger.root / 'halt.json').exists())
                with self.assertRaises(RuntimeError):
                    self.ledger.reserve_browser()
                self.ledger = previous

    def test_invalid_or_duplicate_receipts_never_refund(self):
        ident = self.ledger.reserve(.5)
        before = self.state()
        with self.assertRaises(RuntimeError):
            self.ledger.settle(ident, .4, dict(prompt_tokens=1, completion_tokens=1))
        self.assertEqual(before, self.state())

    def test_inflight_reply_settles_after_halt_without_reopening_dispatch(self):
        ident = self.ledger.reserve(.5)
        unresolved_charge = EXPECTED_CAPS['judge_cap_usd'] - .6
        unresolved = self.ledger.reserve(unresolved_charge)
        with self.assertRaises(RuntimeError):
            self.ledger.reserve(.2)
        halt = (self.ledger.root / 'halt.json').read_bytes()
        self.ledger.settle(ident, .5, dict(prompt_tokens=100, completion_tokens=20))
        self.assertAlmostEqual(self.state()['charged_or_reserved_usd'], unresolved_charge + .00036)
        self.assertEqual(self.state()['calls'], 2)
        self.assertEqual((self.ledger.root / 'halt.json').read_bytes(), halt)
        receipt = json.loads((self.ledger.root / f'request-{unresolved:06d}.json').read_text())
        self.assertEqual(receipt['status'], 'reserved')
        self.assertEqual(receipt['charged_usd'], unresolved_charge)
        for action in (lambda: self.ledger.reserve(.01), self.ledger.reserve_browser):
            with self.assertRaises(RuntimeError):
                action()
        before = self.state()
        with self.assertRaises(RuntimeError):
            self.ledger.settle(ident, .5, dict(prompt_tokens=100, completion_tokens=20))
        self.assertEqual(self.state(), before)

    def test_double_settlement_halts_without_second_refund(self):
        ident = self.ledger.reserve(.5)
        usage = dict(prompt_tokens=100, completion_tokens=20)
        self.ledger.settle(ident, .5, usage)
        before = self.state()
        with self.assertRaises(RuntimeError):
            self.ledger.settle(ident, .5, usage)
        self.assertEqual(before, self.state())

    def test_invalid_usage_retains_reservation(self):
        ident = self.ledger.reserve(.5)
        with self.assertRaises(RuntimeError):
            self.ledger.settle(ident, .5, dict(prompt_tokens=-1, completion_tokens=3))
        self.assertEqual(self.state()['charged_or_reserved_usd'], .5)

    def test_excess_actual_usage_is_charged_then_halts(self):
        ident = self.ledger.reserve(.01)
        with self.assertRaises(RuntimeError):
            self.ledger.settle(ident, .01, dict(prompt_tokens=10000, completion_tokens=0))
        self.assertAlmostEqual(self.state()['charged_or_reserved_usd'], .02)

    def test_approval_revocation_and_ledger_loss_fail_closed(self):
        self.ledger.reserve_browser(1, 2, True)
        atomic(self.control / 'approval-request.json', dict(approved=False, **EXPECTED_CAPS))
        with self.assertRaises(PermissionError):
            self.ledger.reserve_browser()
        atomic(self.control / 'approval-request.json', dict(approved=True, **EXPECTED_CAPS))
        (self.ledger.root / 'ledger.json').rename(self.ledger.root / 'preserved-ledger.json')
        with self.assertRaises(RuntimeError):
            self.ledger.reserve_browser()

    def test_browser_attempts_count_even_repeated_ids_and_evaluations(self):
        with patch.dict(os.environ, OPENWEBRL_SAMPLING_CONTROL=str(self.control)):
            reserve_browser(10, 3)
            reserve_browser(10, 3)
            reserve_browser('eval-0', None, True)
            self.assertIsInstance(wrap(FakeClient()), BudgetedJudge)
        self.assertEqual(self.state()['browser_dispatches'], 3)
        receipt = json.loads((self.ledger.root / 'browser-000003.json').read_text())
        self.assertTrue(receipt['evaluation'])

    def test_concurrent_reservations_have_distinct_durable_ids(self):
        with ThreadPoolExecutor(max_workers=8) as pool:
            ids = list(pool.map(lambda i: BudgetLedger(self.control).reserve_browser(i), range(32)))
        self.assertEqual(sorted(ids), list(range(1, 33)))
        self.assertEqual(self.state()['browser_dispatches'], 32)

    def test_unpriced_model_never_calls_provider(self):
        client = FakeClient()
        guard = BudgetedJudge(client, self.control)
        with self.assertRaises(ValueError):
            asyncio.run(guard.create(model='other', messages=[]))
        self.assertEqual(client.calls, [])


if __name__ == '__main__':
    unittest.main()
