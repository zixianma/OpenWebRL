"""Durable browser and GPT-4.1 caps for one approved sampling-control run.

The caller preserves the native judge protocol and disables SDK retries. Each
explicit HTTP attempt reserves separately. Unsettled, failed, and interrupted
attempts remain charged across replacements; this module never retries.
"""
import fcntl
import json
import math
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace


EXPECTED_CAPS = dict(judge_cap_usd=100, judge_call_cap=8000,
                     browser_dispatch_cap=12000, browser_concurrency=64)


def atomic(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent,
                                     prefix=path.name + '.', suffix='.tmp',
                                     delete=False) as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    temporary.replace(path)


class BudgetLedger:
    def __init__(self, control):
        self.control = Path(control)
        self._approval()
        self.root = self.control / 'judge-budget'
        self.root.mkdir(exist_ok=True)

    def _approval(self):
        approval = json.loads((self.control / 'approval-request.json').read_text())
        if approval.get('approved') is not True:
            raise PermissionError('Sampling-control budget is not approved')
        if any(isinstance(approval.get(key), bool) or approval.get(key) != value
               for key, value in EXPECTED_CAPS.items()):
            raise ValueError('Sampling-control approval differs from exact prepared caps')

    def _halt(self, reason, **details):
        path = self.root / 'halt.json'
        # Preserve the first cause; later failures must not erase it.
        if not path.exists():
            atomic(path, dict(reason=reason, **details))
        raise RuntimeError('Sampling-control budget halted: ' + reason)

    def _state(self, *, allow_halted_settlement=False):
        self._approval()
        if (self.root / 'halt.json').exists() and not allow_halted_settlement:
            raise RuntimeError('Sampling-control budget is halted')
        path = self.root / 'ledger.json'
        if not path.exists():
            # A lost ledger cannot silently restart counters from zero.
            if any(self.root.glob('request-*.json')) or any(self.root.glob('browser-*.json')):
                self._halt('missing_ledger')
            return dict(schema_version=1, caps=EXPECTED_CAPS.copy(), calls=0,
                        browser_dispatches=0, charged_or_reserved_usd=0.)
        try:
            state = json.loads(path.read_text())
            if state['schema_version'] != 1 or state['caps'] != EXPECTED_CAPS:
                raise ValueError('Ledger protocol or caps changed')
            for key in ('calls', 'browser_dispatches'):
                if type(state[key]) is not int or state[key] < 0:
                    raise ValueError('Invalid count')
            charge = state['charged_or_reserved_usd']
            if isinstance(charge, bool) or not isinstance(charge, (int, float)) or not math.isfinite(charge) or charge < 0:
                raise ValueError('Invalid charge')
            if state['calls'] > EXPECTED_CAPS['judge_call_cap'] or state['browser_dispatches'] > EXPECTED_CAPS['browser_dispatch_cap'] or charge > EXPECTED_CAPS['judge_cap_usd']:
                raise ValueError('Ledger already exceeds approved caps')
        except (KeyError, TypeError, ValueError):
            self._halt('invalid_ledger')
        return state

    def _lock(self):
        stream = (self.root / 'ledger.lock').open('a')
        fcntl.flock(stream, fcntl.LOCK_EX)
        return stream

    def reserve(self, cost):
        if isinstance(cost, bool) or not isinstance(cost, (int, float)) or not math.isfinite(cost) or cost <= 0:
            raise ValueError('Invalid judge reservation')
        with self._lock():
            state = self._state()
            if state['calls'] >= EXPECTED_CAPS['judge_call_cap']:
                self._halt('judge_call_limit', state=state)
            if state['charged_or_reserved_usd'] + cost > EXPECTED_CAPS['judge_cap_usd']:
                self._halt('judge_spend_limit', state=state)
            state['calls'] += 1
            state['charged_or_reserved_usd'] += cost
            ident = state['calls']
            receipt = self.root / f'request-{ident:06d}.json'
            if receipt.exists():
                self._halt('duplicate_reservation_receipt', request_id=ident)
            # Charge first: an interruption before receipt creation cannot
            # leave a dispatched request outside the cumulative counters.
            atomic(self.root / 'ledger.json', state)
            atomic(receipt, dict(request_id=ident, status='reserved',
                                 reserved_usd=cost, charged_usd=cost))
            return ident

    def settle(self, ident, reserved, usage):
        with self._lock():
            # A halt forbids new dispatches, but a previously dispatched reply
            # can still supply confirmed usage. Preserve the halt and reconcile
            # only its existing reservation; interrupted replies stay charged.
            state = self._state(allow_halted_settlement=True)
            try:
                if type(ident) is not int or not 0 < ident <= state['calls']:
                    raise ValueError('Invalid request ID')
                path = self.root / f'request-{ident:06d}.json'
                receipt = json.loads(path.read_text())
                if receipt['request_id'] != ident or receipt['status'] != 'reserved' or receipt['reserved_usd'] != reserved or receipt['charged_usd'] != reserved:
                    raise ValueError('Invalid or already settled receipt')
                for key in ('prompt_tokens', 'completion_tokens'):
                    if type(usage[key]) is not int or usage[key] < 0:
                        raise ValueError('Invalid usage')
                cost = (usage['prompt_tokens'] * 2 + usage['completion_tokens'] * 8) / 1e6
            except (OSError, KeyError, TypeError, ValueError):
                self._halt('invalid_settlement_receipt_or_usage', request_id=ident)
            state['charged_or_reserved_usd'] += cost - reserved
            # Avoid a tiny negative caused by floating-point cancellation.
            if -1e-10 < state['charged_or_reserved_usd'] < 0:
                state['charged_or_reserved_usd'] = 0.
            atomic(self.root / 'ledger.json', state)
            atomic(path, dict(receipt, status='received', charged_usd=cost,
                              usage={k: usage[k] for k in ('prompt_tokens', 'completion_tokens')}))
            if cost > reserved or state['charged_or_reserved_usd'] > EXPECTED_CAPS['judge_cap_usd']:
                self._halt('reservation_exceeded', state=state, request_id=ident)

    def reserve_browser(self, sample_index=None, group_index=None, evaluation=False):
        if type(evaluation) is not bool:
            raise ValueError('Evaluation flag must be boolean')
        for ident in (sample_index, group_index):
            if ident is not None and type(ident) not in (int, str):
                raise ValueError('Browser identities must be integers, strings or null')
        with self._lock():
            state = self._state()
            if state['browser_dispatches'] >= EXPECTED_CAPS['browser_dispatch_cap']:
                self._halt('browser_dispatch_limit', state=state)
            state['browser_dispatches'] += 1
            ident = state['browser_dispatches']
            path = self.root / f'browser-{ident:06d}.json'
            if path.exists():
                self._halt('duplicate_browser_receipt', dispatch_id=ident)
            atomic(self.root / 'ledger.json', state)
            atomic(path, dict(dispatch_id=ident, status='reserved',
                              sample_index=sample_index, group_index=group_index,
                              evaluation=evaluation))
            return ident


class BudgetedJudge(BudgetLedger):
    def __init__(self, client, control):
        if getattr(client, 'max_retries', None) != 0:
            raise ValueError('Sampling-control judge requires SDK max_retries=0')
        super().__init__(control)
        self.client = client
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    @staticmethod
    def upper_cost(messages):
        text_bytes, images = 0, 0
        for message in messages:
            content = message.get('content', '')
            if isinstance(content, str):
                text_bytes += len(content.encode())
            else:
                for item in content:
                    if item.get('type') == 'image_url':
                        images += 1
                    else:
                        text_bytes += len(json.dumps(item).encode())
        # Reuse the approved expanded-baseline GPT-4.1 ledger rates ($2/$8
        # per million tokens) and conservative uncached-input/full32K-output
        # reservation. Do not change the native judge's completion settings.
        return (text_bytes + images * 20000 + 4096) * 2 / 1e6 + 32768 * 8 / 1e6

    async def create(self, **kwargs):
        if kwargs.get('model') != 'gpt-4.1':
            raise ValueError('Unpriced judge model')
        reservation = self.upper_cost(kwargs['messages'])
        ident = self.reserve(reservation)
        # Provider errors/cancellation leave the durable reservation charged.
        reply = await self.client.chat.completions.create(**kwargs)
        usage = reply.usage.model_dump() if getattr(reply, 'usage', None) is not None else None
        if isinstance(usage, dict) and all(
                type(usage.get(key)) is int and usage[key] >= 0
                for key in ('prompt_tokens', 'completion_tokens')):
            # Preserve confirmed usage before settlement so a halt or process
            # interruption cannot erase the evidence needed for reconciliation.
            atomic(self.root / f'usage-{ident:06d}.json', dict(
                request_id=ident, model=kwargs['model'], reserved_usd=reservation,
                usage={key: usage[key] for key in ('prompt_tokens', 'completion_tokens')}))
        self.settle(ident, reservation, usage)
        return reply

    async def close(self):
        await self.client.close()


def _control():
    control = os.environ.get('OPENWEBRL_SAMPLING_CONTROL')
    if not control:
        raise ValueError('Missing sampling-control ledger directory')
    return control


def wrap(client):
    return BudgetedJudge(client, _control())


def reserve_browser(sample_index=None, group_index=None, evaluation=False):
    return BudgetLedger(_control()).reserve_browser(sample_index, group_index, evaluation)
