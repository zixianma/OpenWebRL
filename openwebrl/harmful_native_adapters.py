"""Trusted native GUI and local inference bridges; importing does no work.

No service/allocation is started here. The caller must bind exact frozen source,
model, capture, resource approval and ownership receipts. Offline tests exercise
dispatch and receipt validation; live prompt/feature parity is a separate gate.
"""
from __future__ import annotations

import asyncio
import base64
from dataclasses import asdict, dataclass
import hashlib
import io
import json
from pathlib import Path
import re
import uuid
from urllib.parse import urlsplit

from openwebrl.harmful_native_pilot import (
    BrowserReply, ModelReply, Observation, PROSPECTIVE_NATIVE, ProtocolError,
    SAMPLING, canonical, cold_actor_flops, digest, feature_sidecar_flops, origin, parse_action,
)


def private_path(path):
    value = Path(path).absolute()
    if any(p.is_symlink() for p in (value, *value.parents)):
        raise ProtocolError('Symlink ancestry forbidden')
    result = value.resolve()
    if not any(result.is_relative_to(p) for p in (PROSPECTIVE_NATIVE, Path('/tmp'))):
        raise ProtocolError('New private native runtime/tmp path required')
    return result


class ScreenshotStore:
    """Content-addressed exact PNGs; native observations contain no evaluator data."""
    def __init__(self, root):
        self.root, self.records = private_path(root), {}
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, world_id, native):
        raw = native['screenshot']
        if not isinstance(raw, bytes) or not raw.startswith(b'\x89PNG\r\n\x1a\n'):
            raise ProtocolError('Exact native PNG required')
        sha = hashlib.sha256(raw).hexdigest()
        path = private_path(self.root / (sha + '.png'))
        if path.exists():
            if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
                raise ProtocolError('Stored screenshot changed')
        else:
            with path.open('xb') as f:
                f.write(raw)
        obs = Observation(world_id, native['active_tab_url'], sha, native.get('visible_text', ''))
        self.records[digest(asdict(obs))] = dict(native)
        return obs

    def get(self, observation):
        result = dict(self.records[digest(observation)])
        if hashlib.sha256(result['screenshot']).hexdigest() != observation['screenshot_sha256']:
            raise ProtocolError('Observation/image identity mismatch')
        return result


class NativeGUI:
    """Operate a preinitialized, owned Playwright page without a reset method.

    Count each exposed tool call in the policy ledger; subcalls are recorded in
    the adapter receipt callback. Native HTTP/database/SMTP effects are measured
    by independent capture. A browser timeout is uncertain, never recoverable.
    """
    def __init__(self, *, page, world_id, application_url, screenshots, journal):
        self.page, self.context, self.world_id = page, page.context, world_id
        self.application_url, self.allowed = application_url, origin(application_url)
        self.screenshots, self.journal, self.guarded = screenshots, journal, False

    async def install_guards(self):
        if not hasattr(self.context, 'route_web_socket'):
            raise ProtocolError('Browser cannot enforce websocket egress restriction')
        if self.context.service_workers:
            raise ProtocolError('Native context must be created with service_workers=block')
        async def route(request):
            try:
                allowed = origin(request.request.url) == self.allowed
            except (ProtocolError, ValueError):
                allowed = False
            self.journal('network_route', {'url': request.request.url, 'allowed': allowed})
            await (request.continue_() if allowed else request.abort())
        async def websocket(ws):
            self.journal('websocket_blocked', {'url': ws.url})
            await ws.close()
        await self.context.route('**/*', route)
        await self.context.route_web_socket('**/*', websocket)
        self.guarded = True

    def check(self):
        if not self.guarded or origin(self.page.url) != self.allowed:
            raise ProtocolError('Native page/origin guard unavailable')
        if self.context.pages != [self.page] or self.context.service_workers:
            raise ProtocolError('Unreviewed popup or service worker; preserve as unknown')

    async def observe(self):
        self.check()
        viewport = self.page.viewport_size
        if not isinstance(viewport, dict) or min(viewport['width'], viewport['height']) < 1:
            raise ProtocolError('Fixed viewport required')
        raw = await self.page.screenshot(type='png', animations='disabled')
        title = await self.page.title()
        text = await self.page.locator('body').inner_text()
        native = dict(screenshot=raw, screen_size=[viewport['width'], viewport['height']],
            active_tab_url=self.page.url, visible_text=text,
            all_tab_url=[dict(index=0, active=True, url=self.page.url, title=title)])
        self.check()
        return self.screenshots.put(self.world_id, native)

    async def execute(self, action):
        self.check()
        checked = parse_action(canonical(action), self.application_url)
        name, args = checked['name'], checked['arguments']
        if name == 'done':
            raise ProtocolError('DONE is a controller boundary, never a browser mutation')
        viewport = self.page.viewport_size
        def xy(point):
            return point[0] * viewport['width'] / 1000, point[1] * viewport['height'] / 1000
        async def invoke(label, fn, *pos, **kw):
            self.journal('gui_subcall_dispatched', {'method': label, 'arguments': [pos, kw]})
            result = await fn(*pos, **kw)
            self.journal('gui_subcall_returned', {'method': label})
            return result
        if name in ('click', 'hover'):
            point = xy(args['point_2d'])
            if name == 'click':
                await invoke('mouse.click', self.page.mouse.click, *point, button=args.get('button', 'left'), click_count=1)
            else:
                await invoke('mouse.move', self.page.mouse.move, *point)
        elif name == 'write':
            await invoke('keyboard.press', self.page.keyboard.press, 'Control+A')
            await invoke('keyboard.press', self.page.keyboard.press, 'Backspace')
            await invoke('keyboard.insert_text', self.page.keyboard.insert_text, args['message'])
        elif name == 'press_keys':
            keys = ['+'.join(args['keys'])] if args.get('is_hotkey', False) else args['keys']
            for key in keys:
                await invoke('keyboard.press', self.page.keyboard.press, key)
        elif name == 'scroll':
            if 'point_2d' in args:
                await invoke('mouse.move', self.page.mouse.move, *xy(args['point_2d']))
            direction = args.get('direction', 'down'); amount = args.get('amount', .5)
            dx = (1 if direction == 'right' else -1) * viewport['width'] * amount if direction in ('left', 'right') else 0
            dy = (1 if direction == 'down' else -1) * viewport['height'] * amount if direction in ('up', 'down') else 0
            await invoke('mouse.wheel', self.page.mouse.wheel, dx, dy)
        elif name == 'wait':
            await invoke('wait', asyncio.sleep, args.get('seconds', 3))
        elif name == 'goto_url':
            await invoke('page.goto', self.page.goto, args['url'], wait_until='domcontentloaded')
        elif name == 'go_back':
            await invoke('page.go_back', self.page.go_back, wait_until='domcontentloaded')
        else:
            raise ProtocolError('Unsupported typed dispatch')
        self.check()
        return BrowserReply('Browser tool returned; task completion is unverified.', self.world_id)


@dataclass(frozen=True)
class TokenizedReply(ModelReply):
    token_ids: tuple[int, ...] = ()
    prompt_sha256: str = ''
    screenshot_sha256: str = ''


@dataclass
class PreparedPrompt:
    text: str
    prefix_inputs: object
    image_data: str
    image_grid: list[int]
    input_tokens: int
    screenshot_sha256: str


class NativePromptBuilder:
    """Use the existing BrowserAdapter chat/tool mapping and full chosen history.

    The adapter and processor are injected loaded local objects. Model choices
    cannot change the system prompt/tools. Only selected assistant responses and
    observed tool feedback enter history; the latest screenshot is retained,
    matching the existing context_num_screenshots=1 native policy.
    """
    def __init__(self, *, adapter, processor, policy, tools, screenshots, ledger):
        self.adapter, self.processor, self.policy, self.tools = adapter, processor, policy, tools
        self.screenshots, self.ledger, self.cache = screenshots, ledger, {}
        if not policy.strip() or not tools:
            raise ProtocolError('Exact original actor policy/tools required')

    def prepare(self, state_json):
        key = digest(state_json)
        if key in self.cache:
            return self.cache[key]
        state = json.loads(state_json)
        messages = [{'role': 'system', 'content': self.policy}]
        raw_by_sha, responses, first = {}, [], True
        latest = state['observation']
        observations = [i for i,e in enumerate(self.ledger.events) if e['kind'] == 'observation_received']
        if not observations or self.ledger.events[observations[-1]]['data']['observation'] != latest:
            raise ProtocolError('Current observation is not the latest retained native observation')
        for index, event in enumerate(self.ledger.events):
            data = event['data']
            if event['kind'] == 'model_reply' and data['role'] == 'actor':
                raw_by_sha[digest(data['reply']['text'])] = data['reply']['text']
            elif event['kind'] == 'observation_received':
                observed = data['observation']
                native = self.screenshots.get(observed)
                msg = (self.adapter._get_latest_user_message(state['task'], native, step_id=0) if first else
                       self.adapter._convert_tool_responses_to_msgs(responses, observation=native)[0])
                if index != observations[-1]:
                    msg = json.loads(json.dumps(msg))
                    msg['content'] = [x for x in msg['content'] if x['type'] != 'image_url']
                    for part in msg['content']:
                        if part['type'] == 'text':
                            part['text'] = part['text'].replace('screenshot:\n<|vision_start|><|image_pad|><|vision_end|>', '')
                messages.append(msg); first, responses = False, []
            elif event['kind'] == 'selected':
                text = raw_by_sha[data['proposal_sha256']].removesuffix('<|im_end|>').rstrip()
                if not text.lstrip().startswith('<think>'):
                    text = '<think>\n' + text
                messages.append({'role': 'assistant', 'content': text})
            elif event['kind'] == 'browser_reply':
                responses.append({'tool_response': data['response']['text']})
        if not messages or messages[-1]['role'] != 'user':
            raise ProtocolError('Prompt does not end at the current native observation')
        text_messages, images = self.adapter._process_multimodal_messages(messages)
        if len(images) != 1:
            raise ProtocolError('Exactly the latest screenshot must enter the model')
        text = self.adapter.token_handler.apply_chat_template(text_messages, self.tools,
            add_generation_prompt=True, enable_thinking=False)
        # Exact equivalent of native generate_browser's assistant-history restoration.
        contents = iter(m['content'] for m in messages if m['role'] == 'assistant')
        pattern = re.compile(r'(<\|im_start\|>assistant\s*\n)(.*?)(<\|im_end\|>)', re.S)
        text = pattern.sub(lambda m: m[1] + next(contents) + m[3], text,
                           count=sum(m['role'] == 'assistant' for m in messages))
        text = text.rstrip() + ('\n' if text.rstrip().endswith('<think>') else '\n<think>\n')
        from PIL import Image
        raw = self.screenshots.get(latest)['screenshot']
        prefix = self.processor(text=[text], images=[Image.open(io.BytesIO(raw)).convert('RGB')], return_tensors='pt')
        count = int(prefix['input_ids'].shape[1])
        if count + SAMPLING['max_new_tokens'] > 32768:
            raise ProtocolError('Full response does not fit context; never truncate')
        grid = prefix['image_grid_thw'].tolist()
        if len(grid) != 1:
            raise ProtocolError('One exact screenshot grid required')
        value = PreparedPrompt(text, prefix, images[0], grid[0], count, latest['screenshot_sha256'])
        self.cache[key] = value
        return value


def local_endpoint(url):
    value = urlsplit(url)
    if (value.scheme != 'http' or value.hostname != '127.0.0.1' or not value.port or
            value.username or value.password or value.path not in ('', '/') or value.query or value.fragment):
        raise ProtocolError('Explicit loopback-only local model endpoint required')
    return url.rstrip('/')


class LocalActor:
    def __init__(self, *, endpoint, client, prompts, config, journal):
        self.endpoint, self.client = local_endpoint(endpoint), client
        self.prompts, self.config, self.tainted = prompts, config, False
        self.journal = journal

    def bound(self, request):
        data = json.loads(request); prompt = self.prompts.prepare(canonical(data['state']))
        return dict(input_tokens=prompt.input_tokens, output_tokens=SAMPLING['max_new_tokens'],
            modeled_flops=cold_actor_flops(self.config, prompt.input_tokens, SAMPLING['max_new_tokens'], prompt.image_grid)['total'],
            actor_proposals=1)

    async def generate(self, request):
        if self.tainted:
            raise ProtocolError('Uncertain local inference must be drained/closed before reuse')
        data = json.loads(request); prompt = self.prompts.prepare(canonical(data['state']))
        rid = uuid.uuid4().hex
        payload = dict(rid=rid, text=prompt.text, image_data=[prompt.image_data], return_logprob=True,
                       sampling_params=dict(SAMPLING, sampling_seed=data['seed']))
        try:
            response = await self.client.post(self.endpoint + '/generate', json=payload)
            response.raise_for_status(); value = response.json(); meta = value['meta_info']
            self.journal('actor_transport_received', {'request_id': rid, 'value': value,
                'prompt_sha256': hashlib.sha256(prompt.text.encode()).hexdigest(),
                'processor_input_ids': prompt.prefix_inputs['input_ids'].tolist()[0]})
            ids = tuple(pair[1] for pair in meta['output_token_logprobs'])
            if (meta['prompt_tokens'] != prompt.input_tokens or meta['completion_tokens'] != len(ids) or
                    not ids or any(type(x) is not int or x < 0 for x in ids) or
                    meta.get('finish_reason', {}).get('type') not in ('stop', 'length')):
                raise ProtocolError('Actual actor token/context/retraction receipt differs')
            usable = meta['finish_reason']['type'] == 'stop' and meta.get('total_retractions', 0) == 0
            if not usable:
                self.tainted = True
            return TokenizedReply(text=value['text'], input_tokens=prompt.input_tokens, output_tokens=len(ids),
                modeled_flops=cold_actor_flops(self.config, prompt.input_tokens, len(ids), prompt.image_grid)['total'],
                cached_input_tokens=meta.get('cached_tokens'), usable=usable,
                usage_complete=meta.get('total_retractions', 0) == 0, token_ids=ids,
                prompt_sha256=hashlib.sha256(prompt.text.encode()).hexdigest(), screenshot_sha256=prompt.screenshot_sha256)
        except BaseException:
            self.tainted = True
            # Abort signalling is not proof of GPU drain. The campaign must stop
            # this owned service on an unknown request before further inference.
            try:
                await asyncio.wait_for(self.client.post(self.endpoint + '/abort_request',
                    json={'rid': rid, 'abort_all': False}), 2)
            except BaseException:
                pass
            raise


class LocalHead:
    """Async client for an owned local feature-scoring process, no model fallback."""
    def __init__(self, *, endpoint, client, prompts, config, scorer_identity):
        self.endpoint, self.client = local_endpoint(endpoint), client
        self.prompts, self.config, self.scorer_identity = prompts, config, scorer_identity
        self.tainted = False

    def bound(self, state, count):
        if count != 5:
            raise ProtocolError('Five candidates required')
        prompt = self.prompts.prepare(state); lengths = [SAMPLING['max_new_tokens'] + 1] * 5
        return dict(input_tokens=prompt.input_tokens + sum(lengths), output_tokens=0, selector_calls=1,
            modeled_flops=feature_sidecar_flops(self.config, prompt.input_tokens, lengths, prompt.image_grid)['total'])

    async def select(self, state, candidates):
        if self.tainted:
            raise ProtocolError('Unknown head request requires owned-service closure')
        prompt = self.prompts.prepare(state); values = json.loads(candidates)
        sha = hashlib.sha256(prompt.text.encode()).hexdigest()
        if len(values) != 5 or any(v.get('prompt_sha256') != sha or
                v.get('screenshot_sha256') != prompt.screenshot_sha256 or not v.get('token_ids') for v in values):
            raise ProtocolError('Head requires all five exact actor token receipts from this state')
        try:
            response = await self.client.post(self.endpoint + '/score', json=dict(
                prompt=prompt.text, image_data=prompt.image_data, prompt_tokens=prompt.input_tokens,
                candidates=[v['token_ids'] for v in values], scorer_identity=self.scorer_identity))
            response.raise_for_status(); result = response.json()
            if result.get('scorer_identity') != self.scorer_identity:
                raise ProtocolError('Head checkpoint/source identity changed')
            return ModelReply(**result['reply'])
        except BaseException:
            self.tainted = True
            raise


class FrozenFeatureEngine:
    """Synchronous owned-process scoring handler around already loaded local weights.

    GPU/HTTP serving ownership, timeout kill/drain and approved allocation checks
    belong to the launcher. Never run this blocking handler in the policy loop.
    The constructor performs no loading or inference; score() performs the actual
    prompt+five-candidate feature work and head inference when explicitly called.
    """
    def __init__(self, *, model, processor, head, config, identity, device):
        self.model, self.processor, self.head = model, processor, head
        self.config, self.identity, self.device = config, identity, device
        h = head.config
        if h.listwise or tuple(h.layers) != (18,27,36) or tuple(h.pools) != ('thought_mean','action_mean','last') or h.d_model != 512:
            raise ProtocolError('Frozen pointwise teacher-label head configuration differs')

    def score(self, request):
        if request.get('scorer_identity') != self.identity or len(request['candidates']) != 5:
            raise ProtocolError('Scorer identity or fixed candidate count differs')
        import torch
        from PIL import Image
        from openwebrl.selection_head_features import candidate_tokens_from_ids, extract
        raw = base64.b64decode(request['image_data'].split(',', 1)[-1], validate=True)
        prefix = self.processor(text=[request['prompt']], images=[Image.open(io.BytesIO(raw)).convert('RGB')], return_tensors='pt').to(self.device)
        p = int(prefix['input_ids'].shape[1])
        if p != request['prompt_tokens'] or p + SAMPLING['max_new_tokens'] > 32768:
            raise ProtocolError('Actor/sidecar tokenization mismatch')
        candidates = [candidate_tokens_from_ids(self.processor.tokenizer, ids) for ids in request['candidates']]
        lengths = [len(ids) for ids, _ in candidates]
        grid = prefix['image_grid_thw'].tolist()[0]
        cost = feature_sidecar_flops(self.config, p, lengths, grid)
        self.model.eval(); self.head.eval()
        with torch.cuda.device(self.device), torch.inference_mode():
            stream = torch.cuda.current_stream(self.device)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record(stream)
            state, features = extract(self.model, prefix, candidates, self.head.config.layers,
                pools=self.head.config.pools, reuse_prefix=True, batch_size=1,
                pad_id=self.processor.tokenizer.pad_token_id or 0)
            logits, _ = self.head(features.flatten(1)[None], state.flatten()[None])
            if not torch.isfinite(logits).all():
                raise ProtocolError('Nonfinite head logits')
            selection = int(logits.argmax(-1).item()) + 1
            end.record(stream); torch.cuda.synchronize(self.device)
        reply = ModelReply(canonical({'selection': selection}), p + sum(lengths), 0, cost['total'],
                           None, start.elapsed_time(end) / 1000)
        return {'scorer_identity': self.identity, 'reply': asdict(reply), 'modeled_flops_components': cost,
                'features_recomputed': True, 'actor_hidden_states_reused': False}
