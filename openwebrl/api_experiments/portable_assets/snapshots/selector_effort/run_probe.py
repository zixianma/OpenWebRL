#!/usr/bin/env python3
"""Bound, one-attempt, saved-snapshot selector effort diagnostic. No browser code."""
from __future__ import annotations
import argparse
import asyncio
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'om2w100-v1/source-history-transport-v1/openwebrl'
PRICING = dict(luna_input=2., luna_cached=.1, luna_cache_write=2.5, luna_output=10.)
CAPS = dict(sol_calls=200, sol_usd=5., browsers=0, judge_calls=0,
            candidate_calls=0, concurrency=2, wall_seconds=3600)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def audited_helpers():
    modules = []
    for filename in ('responses_actor_history.py', 'luna_qwen_metrics.py'):
        spec = importlib.util.spec_from_file_location('audited_' + filename[:-3], SOURCE / filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules.append(module)
    return modules[0].response_input_bound, modules[1].api_cost, modules[1].tokens


def validate_request(request, effort):
    expected = dict(model='gpt-6.1-sol', reasoning={'effort': effort},
                    max_output_tokens=2048, store=False, service_tier='default')
    if any(request.get(k) != v for k, v in expected.items()):
        raise ValueError('Unapproved model/settings')
    if set(request) != set(expected) | {'input', 'text'}:
        raise ValueError('Unexpected request field')
    fmt = request['text']['format']
    schema = fmt.get('schema', {})
    if (fmt.get('type') != 'json_schema' or fmt.get('strict') is not True
            or schema.get('type') != 'object'
            or schema.get('required') != ['selected_index']
            or schema.get('additionalProperties') is not False
            or schema.get('properties') != {'selected_index': {'type': 'integer', 'enum': [0, 1, 2]}}):
        raise ValueError('Selector schema changed')
    if not isinstance(request['input'], list) or not request['input']:
        raise ValueError('Missing saved selector input')


def load_inputs(root):
    plan_path, manifest_path = root / 'plan.json', root / 'request-manifest-private.json'
    plan = json.loads(plan_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    entries = manifest
    if len(entries) != 100 or len({e['state_id'] for e in entries}) != 100:
        raise ValueError('Expected 100 distinct saved snapshots')
    bounds, _, _ = audited_helpers()
    loaded = []
    for entry in entries:
        if sorted(entry['order']) != ['low', 'medium']:
            raise ValueError('Invalid paired ordering')
        if len(entry['candidate_actions']) != 3 or type(entry['original_selected_index']) is not int or entry['original_selected_index'] not in range(3):
            raise ValueError('Invalid saved candidate identities')
        for field in ('decision', 'original_selector_receipt'):
            if sha(entry[field]) != entry[field + '_sha256']:
                raise ValueError('Saved source hash mismatch')
        decision = json.loads(Path(entry['decision']).read_text())
        original = json.loads(Path(entry['original_selector_receipt']).read_text())
        if (decision.get('status') != 'accepted' or decision.get('task_id') != entry['task_id']
                or decision.get('turn') != entry['turn']
                or decision.get('selected_index') != entry['original_selected_index']
                or canonical(decision.get('candidates')) != canonical(entry['candidate_actions'])
                or original.get('physical_role') != 'selector'
                or parse_response(original['response']) != entry['original_selected_index']):
            raise ValueError('Original saved decision/selector identity differs')
        requests = {}
        for effort in ('low', 'medium'):
            binding = entry['requests'][effort]
            path = root / binding['path']
            if not path.resolve().is_relative_to(root.resolve()) or sha(path) != binding['sha256']:
                raise ValueError('Request binding mismatch')
            request = json.loads(path.read_text())
            validate_request(request, effort)
            bound = bounds(request['input'], [])
            if bound > 272000:
                raise ValueError('Input exceeds frozen price tier')
            if effort == 'medium' and canonical(request) != canonical(original['request']):
                raise ValueError('Medium control differs from saved original request')
            requests[effort] = dict(request=request, binding=binding, bound=bound,
                reservation=(bound * 2.5 + 2048 * 10.) / 1e6)
        left, right = (dict(requests[e]['request']) for e in ('low', 'medium'))
        left.pop('reasoning'); right.pop('reasoning')
        if canonical(left) != canonical(right):
            raise ValueError('Paired selector input/settings changed')
        loaded.append((entry, requests))
    return plan, manifest, loaded


def preflight(root, require_approval=False):
    plan, manifest, loaded = load_inputs(root)
    bindings = dict(plan_sha256=sha(root / 'plan.json'),
                    runner_sha256=sha(Path(__file__)),
                    request_manifest_sha256=sha(root / 'request-manifest-private.json'),
                    audited_sources={name: sha(SOURCE / name) for name in
                                     ('responses_actor_history.py', 'luna_qwen_metrics.py')})
    if plan.get('caps') != CAPS or plan.get('pricing') != PRICING:
        raise ValueError('Plan budget/pricing differs from executable caps')
    if plan.get('audited_sources') != bindings['audited_sources']:
        raise ValueError('Audited source hash differs')
    if require_approval:
        approval = json.loads((root / 'approval.json').read_text())
        if approval.get('approved') is not True or approval.get('caps') != CAPS:
            raise ValueError('Explicit approved caps required')
        if any(approval.get(k) != v for k, v in bindings.items()):
            raise ValueError('Approval not bound to exact executable inputs')
    return dict(status='ready' if require_approval else 'pending_approval', **bindings,
                snapshots=len(loaded), requests=2 * len(loaded),
                sum_conservative_request_reservations_usd=sum(r['reservation'] for _, rs in loaded for r in rs.values()),
                maximum_request_reservation_usd=max(r['reservation'] for _, rs in loaded for r in rs.values())), loaded


def parse_response(response):
    if response.get('status') != 'completed' or not (response.get('model') == 'gpt-6.1-sol' or str(response.get('model', '')).startswith('gpt-6.1-sol-')):
        raise ValueError('Incomplete/unexpected model response')
    texts = [p['text'] for item in response.get('output', []) if item.get('type') == 'message'
             for p in item.get('content', []) if p.get('type') == 'output_text']
    if len(texts) != 1:
        raise ValueError('Expected one structured selection')
    value = json.loads(texts[0])
    if set(value) != {'selected_index'} or type(value['selected_index']) is not int or value['selected_index'] not in range(3):
        raise ValueError('Invalid selected index')
    return value['selected_index']


class Budget:
    def __init__(self, root):
        self.root = root
        self.calls = 0
        self.known_lower = self.known_upper = self.reserved = 0.
        self.halted = False
        self.started = time.monotonic()

    def record(self):
        save(self.root / 'run/budget.json', dict(calls=self.calls, known_lower_usd=self.known_lower,
            known_upper_usd=self.known_upper, unknown_or_inflight_reserved_usd=self.reserved,
            bounded_upper_usd=self.known_upper + self.reserved, halted=self.halted, caps=CAPS))

    def reserve(self, amount):
        if self.halted or time.monotonic() - self.started >= CAPS['wall_seconds']:
            raise RuntimeError('Halted or wall cap reached')
        if self.calls >= CAPS['sol_calls'] or self.known_upper + self.reserved + amount > CAPS['sol_usd'] + 1e-10:
            raise RuntimeError('Call/dollar cap reached; no request dispatched')
        self.calls += 1
        self.reserved += amount
        self.record()
        return self.calls

    def settle(self, amount, cost):
        if cost['upper_usd'] > amount + 1e-10:
            self.halted = True
            self.record()
            raise RuntimeError('Usage exceeds conservative reservation')
        self.reserved -= amount
        self.known_lower += cost['lower_usd']
        self.known_upper += cost['upper_usd']
        self.record()


async def perform(root, client, entry, effort, binding, budget):
    _, api_cost, tokens = audited_helpers()
    reservation = binding['reservation']
    ident = budget.reserve(reservation)
    path = root / f'run/receipts/{ident:03d}.json'
    receipt = dict(snapshot_id=entry['state_id'], effort=effort, request_id=ident,
        request_sha256=binding['binding']['sha256'], request_path=binding['binding']['path'],
        reserved_usd=reservation, started_epoch=time.time(), status='reserved')
    save(path, receipt)
    started = time.monotonic()
    try:
        response = await client.responses.create(**binding['request'])
        raw = response.model_dump()
        receipt.update(response=raw, status='received', usage=raw.get('usage'))
        cost = api_cost(raw['usage'], 'selector', PRICING)
        budget.settle(reservation, cost)
        receipt.update(cost=cost, tokens=tokens(raw['usage'], 'api'))
        index = parse_response(raw)
        receipt.update(status='accepted', selected_index=index,
            selected_action=entry['candidate_actions'][index], original_selected_index=entry['original_selected_index'],
            original_index_agreement=index == entry['original_selected_index'],
            original_action_agreement=canonical(entry['candidate_actions'][index]) == canonical(entry['candidate_actions'][entry['original_selected_index']]))
    except BaseException as exc:
        budget.halted = True
        receipt.update(error_type=type(exc).__name__, http_status=getattr(exc, 'status_code', None))
        if 'usage' not in receipt or receipt['usage'] is None:
            receipt['status'] = 'usage_unknown'
        save(root / 'run/halt.json', dict(request_id=ident, error_type=type(exc).__name__, reason='Review required; no automatic retry'))
        raise
    finally:
        receipt['seconds'] = time.monotonic() - started
        save(path, receipt)
        budget.record()
    return receipt


def credentials():
    values = {}
    for line in Path('/gpfs/projects/krishna/zixianma/OpenWebRL/.env').read_text().splitlines():
        if '=' in line and not line.lstrip().startswith('#'):
            key, value = line.split('=', 1)
            values[key.strip()] = value.strip().strip('"\'')
    key = values.get('OPENAI_API_KEY')
    if not key:
        raise ValueError('Missing API credential')
    return key, values.get('OPENAI_BASE_URL')


async def run(root):
    proof, loaded = preflight(root, require_approval=True)
    # Exclusive durable start disallows both duplicate writers and automatic resumes.
    (root / 'run').mkdir(exist_ok=True)
    with (root / 'run/started.json').open('x') as handle:
        json.dump(dict(bindings=proof, started_epoch=time.time()), handle)
    from openai import AsyncOpenAI
    key, base_url = credentials()
    client = AsyncOpenAI(api_key=key, base_url=base_url, max_retries=0, timeout=90.)
    budget = Budget(root)
    rows = []
    try:
        async with asyncio.timeout(CAPS['wall_seconds']):
            # Each snapshot's two efforts are issued serially in a frozen randomized
            # order; at most two independent snapshot pairs run concurrently.
            semaphore = asyncio.Semaphore(2)
            async def pair(entry, requests):
                async with semaphore:
                    order = entry['order']
                    result = {}
                    for effort in order:
                        result[effort] = await perform(root, client, entry, effort, requests[effort], budget)
                    low, medium = result['low'], result['medium']
                    record = dict(snapshot_id=entry['state_id'], order=order,
                        low_selected_index=low['selected_index'], medium_selected_index=medium['selected_index'],
                        paired_index_agreement=low['selected_index'] == medium['selected_index'],
                        paired_action_agreement=canonical(low['selected_action']) == canonical(medium['selected_action']))
                    save(root / 'run/pairs' / (entry['state_id'] + '.json'), record)
                    rows.append(record)
            queue = asyncio.Queue()
            for item in loaded:
                queue.put_nowait(item)
            async def worker():
                while not budget.halted and not queue.empty():
                    entry, requests = queue.get_nowait()
                    try:
                        await pair(entry, requests)
                    except BaseException as exc:
                        budget.halted = True
                        return exc
            results = await asyncio.gather(worker(), worker(), return_exceptions=True)
            failures = [result for result in results if isinstance(result, BaseException)]
            if failures:
                raise failures[0]
        save(root / 'run/completion.json', dict(worker_complete=True, pairs=len(rows), calls=budget.calls,
            paired_index_agreement=sum(r['paired_index_agreement'] for r in rows),
            paired_action_agreement=sum(r['paired_action_agreement'] for r in rows), independently_verified=False))
    except BaseException:
        budget.halted = True
        budget.record()
        raise
    finally:
        await client.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.run:
        asyncio.run(run(args.root))
    else:
        proof, _ = preflight(args.root)
        save(args.root / 'preflight.json', proof)
        print(json.dumps(proof, indent=2))


if __name__ == '__main__':
    main()
