"""Explicit model, effort and price identities for the approved API actor study."""
from __future__ import annotations

import json
from pathlib import Path
import time

from openwebrl.luna_qwen_metrics import PRICING, api_cost, write_json
from openwebrl.luna_qwen_policy import Policy, actor_tools
from openwebrl.responses_actor_history import response_input_bound

MODELS = {'luna_high': 'gpt-6-luna', 'sol61_high': 'gpt-6.1-sol'}
PRICES = {
    'gpt-6-luna': dict(input=.10, cached=.01, cache_write=.125, output=.50),
    'gpt-6.1-sol': dict(input=2., cached=.10, cache_write=2.50, output=10.),
}


def metric_prices(model):
    # The inherited accounting helper calls its non-judge rate fields luna_*.
    # Bind all four fields to this exact model; receipts retain its real name.
    return dict(PRICING, **{'luna_' + k: v for k, v in PRICES[model].items()})


class MeteredActorAPI:
    def __init__(self, client, budget, directory, model, pricing):
        if model not in PRICES or pricing != metric_prices(model):
            raise ValueError('Unknown model or incorrect model-specific prices')
        self.client, self.budget, self.directory = client, budget, Path(directory)
        self.model, self.pricing = model, pricing

    async def create(self, role, **kwargs):
        if (role != 'actor' or kwargs.get('model') != self.model
                or kwargs.get('reasoning') != {'effort': 'high'}
                or kwargs.get('max_output_tokens') != 4096
                or kwargs.get('service_tier') != 'default'
                or kwargs.get('store') is not False
                or kwargs.get('tool_choice') != 'required'
                or kwargs.get('parallel_tool_calls') is not False
                or any(k in kwargs for k in ('temperature', 'top_p'))):
            raise ValueError('Request differs from approved high-reasoning actor protocol')
        input_bound = response_input_bound(kwargs['input'], kwargs.get('tools', []))
        if input_bound > 272000:
            raise ValueError('Conservative request size exceeds frozen pricing tier')
        upper = (input_bound * self.pricing['luna_cache_write']
                 + 4096 * self.pricing['luna_output']) / 1e6
        ident = self.budget.update(reserve=upper)
        path = self.directory / 'requests' / f'actor-{ident:05d}.json'
        row = dict(provider=self.model, role=role, status='reserved', reserved_usd=upper,
                   request=kwargs, started_epoch=time.time())
        write_json(path, row)
        started = time.monotonic()
        try:
            response = await self.client.responses.create(**kwargs)
            row.update(response=response.model_dump(), usage=response.usage.model_dump(), status='received')
            self.budget.update(settle=(upper, api_cost(row['usage'], role, self.pricing)['upper_usd']))
            if not (response.model == self.model or response.model.startswith(self.model + '-')):
                raise ValueError('Provider substituted the requested actor model')
            if response.status != 'completed':
                raise ValueError('Incomplete actor response; preserve receipt and halt for review')
            return response
        except BaseException as exc:
            row['error_type'] = type(exc).__name__
            row['http_status'] = getattr(exc, 'status_code', None)
            # Provider error payloads remain private with the original request.
            if getattr(exc, 'body', None) is not None:
                row['provider_error_body'] = exc.body
            if row['http_status'] in (400, 401, 403, 404) or isinstance(exc, ValueError):
                write_json(self.budget.root / 'halt.json', dict(reason='API configuration or protocol',
                    error_type=type(exc).__name__, model=self.model))
            if 'usage' not in row:
                row['status'] = 'usage_unknown'
            raise
        finally:
            row['seconds'] = time.monotonic() - started
            write_json(path, row)


class ActorPolicy(Policy):
    def __init__(self, mode, directory, api, builder, qwen_config, preprocessor):
        if mode not in MODELS or api.model != MODELS[mode]:
            raise ValueError('Actor mode/model identity mismatch')
        super().__init__('luna', directory, api, builder, qwen_config, preprocessor)
        self.arm_mode = mode

    async def __call__(self, *, observation, turn, task_id=None, **unused):
        if self.messages is None:
            raise ValueError('Missing structured observation hook')
        (self.root / 'screenshots').mkdir(parents=True, exist_ok=True)
        (self.root / 'screenshots' / f'{turn:03d}.png').write_bytes(observation['screenshot'])
        tools = actor_tools(self.tools)
        native_input = self.actor_history.prepare(self.messages, turn=turn, task_id=task_id)
        response = await self.api.create('actor', model=self.api.model, reasoning={'effort': 'high'},
            max_output_tokens=4096, store=False, service_tier='default',
            input=native_input, tools=tools,
            tool_choice='required', parallel_tool_calls=False)
        action = self.actor_history.accept(response.output, {t['name'] for t in tools})
        text = '</think>\n<tool_call>\n' + json.dumps(action) + '\n</tool_call><|im_end|>\n'
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        return (text, ids, [0.] * len(ids), 'stop'), dict(mode=self.arm_mode,
            actor_model=self.api.model, reasoning_effort='high', selected_index=None,
            browser_coordinate_space=self.browser_coordinate_space)
