"""Qwen proposals, a Luna index selector, and a Luna-only browser actor."""
from __future__ import annotations

import asyncio
import base64
import fcntl
import hashlib
import json
import random
import time
from pathlib import Path

from openwebrl.arm_inference import candidate_seed, split_response, selection_messages, parse_selection
from openwebrl.arm_inference_flops import decoder_flops, vision_flops, image_geometry
from openwebrl.luna_qwen_metrics import SEED, api_cost, tokens, write_json
from openwebrl.responses_actor_history import ResponsesActorHistory, response_input_bound

MODEL = 'gpt-6-luna'


def response_messages(messages):
    """Preserve roles/text; replace Qwen screenshot markers with API images."""
    result = []
    marker = 'screenshot:\n<|vision_start|><|image_pad|><|vision_end|>'
    for message in messages:
        content = message['content']
        if isinstance(content, str):
            result.append(dict(role=message['role'], content=content.replace(marker, 'Screenshot:')))
            continue
        parts = []
        for part in content:
            if part['type'] == 'text':
                parts.append(dict(type='input_text', text=part['text'].replace(marker, 'Screenshot:')))
            elif part['type'] == 'image_url':
                value = part['image_url']
                parts.append(dict(type='input_image', image_url=value['url'] if isinstance(value, dict) else value,
                                  detail='high'))
            else:
                raise ValueError('Unexpected browser observation modality')
        result.append(dict(role=message['role'], content=parts))
    return result


def actor_tools(tools):
    result = []
    for tool in tools:
        function = tool.get('function', tool)
        result.append(dict(type='function', name=function['name'],
                           description=function.get('description', ''),
                           parameters=function.get('parameters', {'type': 'object', 'properties': {}}), strict=False))
    if not result:
        raise ValueError('No browser tools')
    return result


class Budget:
    """Reservations survive interrupted requests and every process restart."""
    def __init__(self, root, max_usd, max_calls):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_usd, self.max_calls = max_usd, max_calls

    def update(self, reserve=None, settle=None):
        with (self.root / 'owner.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            file = self.root / 'ledger.json'
            state = json.loads(file.read_text()) if file.exists() else dict(calls=0, charged_or_reserved_usd=0.)
            if reserve is not None:
                if (self.root / 'halt.json').exists() or state['calls'] >= self.max_calls or state['charged_or_reserved_usd'] + reserve > self.max_usd:
                    write_json(self.root / 'halt.json', dict(reason='budget', ledger=state))
                    raise RuntimeError('API budget exhausted; stop, preserving results')
                state['calls'] += 1
                state['charged_or_reserved_usd'] += reserve
            if settle is not None:
                reserved, actual_upper = settle
                state['charged_or_reserved_usd'] += actual_upper - reserved
                if actual_upper > reserved or state['charged_or_reserved_usd'] > self.max_usd:
                    write_json(self.root / 'halt.json', dict(reason='unexpected_usage', ledger=state))
            write_json(file, state)
            return state['calls']


class MeteredAPI:
    def __init__(self, client, budget, directory):
        self.client, self.budget, self.directory = client, budget, Path(directory)

    async def create(self, role, **kwargs):
        if kwargs.get('model') != MODEL:
            raise ValueError('Unexpected actor/selector model')
        # Conservative UTF-8 byte bound, plus a screenshot reserve. The API
        # receipt, including hidden reasoning and cache writes, replaces it.
        input_bound = response_input_bound(kwargs['input'], kwargs.get('tools', []))
        if input_bound > 272000:
            raise ValueError('Conservative request size exceeds frozen price tier')
        upper = (input_bound * .125 + kwargs['max_output_tokens'] * .5) / 1e6
        ident = self.budget.update(reserve=upper)
        path = self.directory / 'requests' / f'luna-{ident:05d}.json'
        row = dict(provider='luna', role=role, status='reserved', reserved_usd=upper,
                   request=kwargs, started_epoch=time.time())
        write_json(path, row)
        started = time.monotonic()
        try:
            response = await self.client.responses.create(**kwargs)
            row.update(response=response.model_dump(), usage=response.usage.model_dump(), status='received')
            self.budget.update(settle=(upper, api_cost(row['usage'], role)['upper_usd']))
            if not (response.model == MODEL or response.model.startswith(MODEL + '-')):
                raise ValueError('Provider substituted the requested model')
            if response.status != 'completed':
                raise ValueError('Incomplete API response; do not execute a partial action')
            return response
        except BaseException as exc:
            row['error_type'] = type(exc).__name__
            if getattr(exc, 'status_code', None) in (400, 401, 403, 404) or isinstance(exc, ValueError):
                write_json(self.budget.root / 'halt.json', dict(reason='API configuration or protocol', error_type=type(exc).__name__))
            if 'usage' not in row:
                row['status'] = 'usage_unknown'
            raise
        finally:
            row['seconds'] = time.monotonic() - started
            write_json(path, row)


class Policy:
    def __init__(self, mode, directory, api, builder, qwen_config, preprocessor):
        self.mode, self.root, self.api, self.builder = mode, Path(directory), api, builder
        self.config, self.preprocessor = qwen_config, preprocessor
        self.count = dict(qwen=1, qwen_luna5=5, qwen_luna10=10, luna=0, sft_luna5=5)[mode]
        self.request_count = 0
        self.local_batch_seconds = 0.
        self.messages = self.tools = self.tokenizer = None
        self.actor_history = ResponsesActorHistory() if mode == 'luna' else None
        # Native API browser tools advertise viewport pixels. Qwen's released
        # policy instead emits [0,1000] coordinates. Bind the environment to
        # the actor's contract; never rewrite actions or conversation history.
        self.browser_coordinate_space = 'viewport_pixels' if mode == 'luna' else 'normalized_1000'

    def set_context(self, messages, tools, tokenizer, prompt_tokens):
        self.messages, self.tools, self.tokenizer = messages, tools, tokenizer
        self.prompt_tokens = prompt_tokens

    async def qwen(self, url, text, params, images, timeout):
        from openwebrl import generate_browser as generation
        ident = self.request_count
        self.request_count += 1
        path = self.root / 'requests' / f'qwen-{ident:05d}.json'
        row = dict(provider='qwen', role='actor', status='reserved', sampling=params,
                   prompt_sha256=hashlib.sha256(text.encode()).hexdigest(), started_epoch=time.time())
        write_json(path, row)
        started = time.monotonic()
        try:
            payload = dict(text=text, sampling_params=params, return_logprob=True)
            if images:
                payload['image_data'] = images
            response = await asyncio.wait_for(generation.post(url, payload), timeout)
            meta = response['meta_info']
            row.update(status='received', usage={k: v for k, v in meta.items() if 'logprob' not in k})
            counts = tokens(row['usage'], 'qwen')
            # Unknown KV hits cannot be substituted with a measured zero.
            cached = counts['cached_input']
            if cached is None:
                raise ValueError('Qwen server omitted KV-cache usage')
            p, g = counts['input'], counts['output']
            if g < 1:
                raise ValueError('Empty Qwen generation')
            cold = decoder_flops(self.config['text_config'], p, g, min(cached, p - 1))
            vision = 0
            for data in images:
                data = data.get('url') if isinstance(data, dict) else data
                geometry = image_geometry(data, max_pixels=self.preprocessor['size']['longest_edge'],
                                          min_pixels=self.preprocessor['size']['shortest_edge'])
                vision += vision_flops(self.config['vision_config'], geometry['grid'])['total']
            row['flops'] = dict(lower=cold['total'], upper=cold['total'] + math_ceil_div(p - min(cached, p-1), 4096) * vision)
            pairs = meta['output_token_logprobs']
            ids, logps = [v[1] for v in pairs], [v[0] for v in pairs]
            if not ids:
                raise ValueError('Empty Qwen generation')
            reply = generation._ensure_im_end_w_new_line(response['text'], ids, logps)
            row['output_text'] = response['text']
            return reply, ids, logps, meta.get('finish_reason', {}).get('type', 'stop')
        except BaseException as exc:
            row['error_type'] = type(exc).__name__
            if 'usage' not in row:
                row['status'] = 'usage_unknown'
            raise
        finally:
            row['seconds'] = time.monotonic() - started
            write_json(path, row)

    async def __call__(self, *, infer, url, input_text, sampling_params, images,
                       observation, history, task, task_id, turn, timeout):
        if self.messages is None:
            raise ValueError('Missing structured observation hook')
        (self.root / 'screenshots').mkdir(parents=True, exist_ok=True)
        (self.root / 'screenshots' / f'{turn:03d}.png').write_bytes(observation['screenshot'])
        common = dict(model=MODEL, reasoning={'effort': 'medium'}, max_output_tokens=4096,
                      store=False, service_tier='default')
        if self.mode == 'luna':
            tools = actor_tools(self.tools)
            native_input = self.actor_history.prepare(self.messages, turn=turn, task_id=task_id)
            response = await self.api.create('actor', **common, input=native_input,
                tools=tools, tool_choice='required', parallel_tool_calls=False)
            action = self.actor_history.accept(response.output, {t['name'] for t in tools})
            # Framework token IDs are Qwen serialization only, never billed or
            # reported as Luna output tokens. No hidden reasoning is fabricated.
            text = '</think>\n<tool_call>\n' + json.dumps(action) + '\n</tool_call><|im_end|>\n'
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            return (text, ids, [0.] * len(ids), 'stop'), dict(mode='luna', selected_index=None,
                browser_coordinate_space=self.browser_coordinate_space)
        # SGLang rejects input + requested output >= context_length, and its
        # scheduler likewise reserves one token. Equality is not admissible.
        remaining = 32768 - self.prompt_tokens - 1
        if remaining < 1:
            raise ValueError('Actor context exhausted')
        params = [dict(sampling_params, max_new_tokens=min(sampling_params['max_new_tokens'], remaining),
                       sampling_seed=candidate_seed(SEED, task_id, turn, i)) for i in range(self.count)]
        started = time.monotonic()
        # Await every proposal even when a sibling fails, preserving receipts
        # and preventing background work from spilling into the next episode.
        outputs = await asyncio.gather(*(self.qwen(url, input_text, p, images, timeout) for p in params), return_exceptions=True)
        self.local_batch_seconds += time.monotonic() - started
        errors = [o for o in outputs if isinstance(o, BaseException)]
        if errors:
            raise errors[0]
        candidates = [split_response(o[0]) for o in outputs]
        order = list(range(self.count))
        random.Random(candidate_seed(SEED, task_id, turn, 999)).shuffle(order)
        index = 0
        if self.count > 1:
            messages = selection_messages(self.builder, task, observation.get('active_tab_url', ''),
                [split_response(h) for h in history], [candidates[i] for i in order], observation['screenshot'])
            response = await self.api.create('selector', **common, input=response_messages(messages),
                text={'format': {'type': 'json_schema', 'name': 'action_selection', 'strict': True,
                    'schema': {'type': 'object', 'properties': {'selection': {'type': 'integer',
                        'enum': list(range(1, self.count + 1))}}, 'required': ['selection'], 'additionalProperties': False}}})
            index = order[parse_selection(response.output_text, self.count)]
        write_json(self.root / 'decisions' / f'{turn:03d}.json', dict(task_id=task_id, turn=turn,
            mode=self.mode, candidates=candidates, display_order=order, chosen_original_index=index,
            candidate_seeds=[p['sampling_seed'] for p in params], finish_types=[o[3] for o in outputs]))
        return outputs[index], dict(mode=self.mode, selected_index=index)


def math_ceil_div(value, divisor):
    return (value + divisor - 1) // divisor
