"""Shared judge-spend ledger for the expanded outcome-only experiment.

Wrap the existing client without changing prompts, completion settings or the
native reward parser. No request is sent while approval is absent. Failed or
interrupted calls retain their reservation, including across job replacements.
"""
import fcntl
import json
import os
from pathlib import Path
from types import SimpleNamespace


def atomic(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


class BudgetedJudge:
    def __init__(self, client, control):
        self.client = client
        self.control = Path(control)
        approval = json.loads((self.control/'approval-request.json').read_text())
        if approval.get('approved') is not True:
            raise PermissionError('Expanded baseline judge budget is not approved')
        self.cap = float(approval['judge_cap_usd'])
        if self.cap != 200:
            raise ValueError('Recheck the prepared ledger for a changed budget')
        self.root = self.control/'judge-budget'
        self.root.mkdir(exist_ok=True)
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
        # Same $2/$8 per-million GPT-4.1 ledger rates as the completed task
        # profiling/rescue runs. Reserve uncached input and the full32K output
        # capacity; do not impose a new completion limit on the native judge.
        return (text_bytes + images*20000 + 4096)*2/1e6 + 32768*8/1e6

    def reserve(self, cost):
        with (self.root/'ledger.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            p = self.root/'ledger.json'
            state = json.loads(p.read_text()) if p.exists() else dict(calls=0, charged_or_reserved_usd=0.)
            if (self.root/'halt.json').exists() or state['charged_or_reserved_usd'] + cost > self.cap:
                atomic(self.root/'halt.json', dict(reason='judge_budget_limit', state=state))
                raise RuntimeError('Expanded baseline judge budget exhausted')
            state['calls'] += 1
            state['charged_or_reserved_usd'] += cost
            atomic(p, state)
            ident = state['calls']
            atomic(self.root/f'{ident:06d}.json', dict(status='reserved', charged_usd=cost))
            return ident

    def settle(self, ident, reserved, usage):
        cost = (usage['prompt_tokens']*2 + usage['completion_tokens']*8)/1e6
        with (self.root/'ledger.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            p = self.root/'ledger.json'
            state = json.loads(p.read_text())
            state['charged_or_reserved_usd'] += cost - reserved
            atomic(p, state)
            atomic(self.root/f'{ident:06d}.json', dict(status='received', charged_usd=cost, usage=usage))
            if cost > reserved or state['charged_or_reserved_usd'] > self.cap:
                atomic(self.root/'halt.json', dict(reason='reservation_exceeded', state=state))
                raise RuntimeError('Judge usage exceeded conservative reservation')

    async def create(self, **kwargs):
        if kwargs.get('model') != 'gpt-4.1':
            raise ValueError('Unpriced judge model')
        reservation = self.upper_cost(kwargs['messages'])
        ident = self.reserve(reservation)
        # SDK retries are disabled at client construction. The native three
        # explicit attempts each receive their own durable reservation.
        reply = await self.client.chat.completions.create(**kwargs)
        self.settle(ident, reservation, reply.usage.model_dump())
        return reply

    async def close(self):
        await self.client.close()


def wrap(client):
    control = os.environ.get('OPENWEBRL_EXPANDED_BASELINE_CONTROL')
    if not control:
        raise ValueError('Missing expanded-baseline ledger directory')
    return BudgetedJudge(client, control)
