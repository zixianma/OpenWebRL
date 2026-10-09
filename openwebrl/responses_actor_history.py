"""Stateless native Responses history for the single-action browser actor.

Framework XML is an execution adapter only. Native output items, including
opaque encrypted reasoning, are copied unchanged and never decoded or logged
here. Browser feedback is attached to the actual preceding function call ID.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re

API_SYSTEM_PROMPT = Path(__file__).parent / 'env/prompts/system_prompt_api_browser.md'
SCREENSHOT_MARKER = 'screenshot:\n<|vision_start|><|image_pad|><|vision_end|>'


def response_input_bound(items, tools):
    """Conservative UTF-8/JSON byte bound, including opaque replay bytes.

Image payloads are charged by the existing 32K-token image allowance, never as
base64 text as well. All other fields (including encrypted_content, call IDs,
arguments, summaries and assistant phases) count toward the same 272K cap.
"""
    images = 0

    def without_image_payload(value):
        nonlocal images
        if isinstance(value, dict):
            if value.get('type') == 'input_image':
                if not (isinstance(value.get('image_url'), str) and value['image_url']):
                    raise ValueError('Expected a nonempty browser image URL')
                images += 1
                return {k: without_image_payload(v) for k, v in value.items() if k != 'image_url'}
            return {k: without_image_payload(v) for k, v in value.items()}
        if isinstance(value, list):
            return [without_image_payload(v) for v in value]
        return value

    counted = without_image_payload(items)
    # One UTF-8 byte per token is a deliberately loose text bound. JSON escaping
    # and the existing framing allowance make the reservation more conservative.
    return (len(json.dumps(counted, allow_nan=False).encode())
            + len(json.dumps(tools, allow_nan=False).encode()) + 4096 + images * 32768)


def _framework_parts(message):
    content = message.get('content')
    if isinstance(content, str):
        return [dict(type='text', text=content)]
    if not isinstance(content, list) or not content:
        raise ValueError('Browser message must contain text and optional screenshot')
    if any(not isinstance(p, dict) or p.get('type') not in ('text', 'image_url') for p in content):
        raise ValueError('Unexpected browser observation modality')
    if any(not isinstance(p.get('text'), str) for p in content if p['type'] == 'text'):
        raise ValueError('Browser observation text must be a string')
    if sum(p['type'] == 'image_url' for p in content) > 1:
        raise ValueError('Browser actor permits only the current screenshot')
    return deepcopy(content)


def _framework_identity(messages):
    """Compare the growing framework prefix, allowing its old-image removal."""
    result = []
    for message in messages:
        parts = _framework_parts(message)
        result.append(dict(role=message['role'], content=[p['text'].replace(SCREENSHOT_MARKER, '')
            for p in parts if p['type'] == 'text']))
    return result


def _api_content(message, *, tool_result=False):
    parts = _framework_parts(message)
    texts = [p for p in parts if p['type'] == 'text']
    if not texts:
        raise ValueError('Missing browser observation text')
    if tool_result:
        # The adapter emits one feedback block containing actual tool text and
        # the ensuing observation. Do not invent a success result or a new user
        # instruction when the expected execution feedback is missing.
        if (len(texts) != 1 or not texts[0]['text'].startswith('<tool_response>\n')
                or not texts[0]['text'].endswith('\n</tool_response>')):
            raise ValueError('Missing single-action browser tool response')
        texts[0]['text'] = texts[0]['text'][len('<tool_response>\n'):-len('\n</tool_response>')]
        if not texts[0]['text'].strip():
            raise ValueError('Empty browser tool response')
    result = []
    for part in parts:
        if part['type'] == 'text':
            result.append(dict(type='input_text', text=part['text'].replace(SCREENSHOT_MARKER, 'Screenshot:')))
        else:
            value = part['image_url']
            url = value.get('url') if isinstance(value, dict) else value
            if not isinstance(url, str) or not url:
                raise ValueError('Missing current browser screenshot URL')
            result.append(dict(type='input_image', image_url=url, detail='high'))
    return result


def _drop_old_input_images(items):
    for item in items:
        # These are locally created inputs. Never alter a returned output item.
        key = ('content' if item.get('role') == 'user' else
               'output' if item.get('type') == 'function_call_output' else None)
        if key and isinstance(item.get(key), list):
            item[key] = [p for p in item[key] if p.get('type') != 'input_image']


def _framework_action(message):
    if message.get('role') != 'assistant' or not isinstance(message.get('content'), str):
        raise ValueError('Missing framework action for native pending call')
    calls = re.findall(r'<tool_call>\s*(.*?)\s*</tool_call>', message['content'], re.S)
    if len(calls) != 1:
        raise ValueError('Framework must contain exactly one matching action')
    try:
        action = json.loads(calls[0])
    except (TypeError, ValueError):
        raise ValueError('Malformed framework action') from None
    if not isinstance(action, dict) or set(action) != {'name', 'arguments'}:
        raise ValueError('Malformed framework action')
    return action


class ResponsesActorHistory:
    """One policy/task's native history, with a checked two-phase turn update."""
    def __init__(self, system_prompt_path=None):
        self.system_prompt_path = Path(system_prompt_path) if system_prompt_path is not None else API_SYSTEM_PROMPT
        self.reset()

    def reset(self):
        self._items = []
        self._framework = None
        self._pending = None
        self._prepared = None
        self._task_id = None
        self._next_turn = 0

    def prepare(self, messages, *, turn, task_id=None):
        if turn == 0 and self._framework is not None and task_id is not None and task_id != self._task_id:
            self.reset()
        if (type(turn) is not int or turn != self._next_turn or self._prepared is not None
                or (self._framework is not None and task_id != self._task_id)):
            raise ValueError('Native actor conversation turn/task is unmatched')
        if not isinstance(messages, list) or not messages or messages[-1].get('role') != 'user':
            raise ValueError('Missing latest framework user observation')
        framework = _framework_identity(messages)
        if self._framework is None:
            if any(m.get('role') not in ('system', 'developer') for m in messages[:-1]):
                raise ValueError('Cannot reconstruct an earlier native actor conversation')
            try:
                prompt = self.system_prompt_path.read_text(encoding='utf-8')
            except (OSError, UnicodeError):
                raise ValueError('Required API browser system prompt is unavailable') from None
            if not prompt.strip():
                raise ValueError('Required API browser system prompt is empty')
            items = [dict(role='system', content=prompt),
                     dict(role='user', content=_api_content(messages[-1]))]
        else:
            if (len(framework) != len(self._framework) + 2 or framework[:-2] != self._framework
                    or self._pending is None):
                raise ValueError('Framework history does not extend the native pending call')
            action = _framework_action(messages[-2])
            if action != {k: self._pending[k] for k in ('name', 'arguments')}:
                raise ValueError('Framework action differs from native pending call')
            items = deepcopy(self._items)
            _drop_old_input_images(items)
            items.append(dict(type='function_call_output', call_id=self._pending['call_id'],
                              output=_api_content(messages[-1], tool_result=True)))
        # Commit only once a complete, valid native response is accepted. The
        # caller/API receives a separate copy, so it cannot mutate our history.
        self._prepared = dict(items=items, framework=framework, task_id=task_id)
        return deepcopy(items)

    def accept(self, output, allowed_names):
        if self._prepared is None:
            raise ValueError('No prepared native actor request')
        try:
            items = [deepcopy(item if isinstance(item, dict) else item.model_dump()) for item in output]
        except (TypeError, AttributeError):
            raise ValueError('Native response output must contain serializable items') from None
        if not items or any(not isinstance(item, dict) or not isinstance(item.get('type'), str) for item in items):
            raise ValueError('Native response output must contain typed items')
        calls = [item for item in items if item['type'] == 'function_call']
        if len(calls) != 1 or calls[0].get('name') not in allowed_names:
            raise ValueError('Actor must emit exactly one permitted browser action')
        call = calls[0]
        call_id = call.get('call_id')
        if (not isinstance(call_id, str) or not call_id
                or any(item.get('call_id') == call_id for item in self._items)):
            raise ValueError('Native actor function call ID is missing or reused')
        try:
            arguments = json.loads(call['arguments'])
            if not isinstance(arguments, dict):
                raise ValueError()
            json.dumps(arguments, allow_nan=False)
            json.dumps(items, allow_nan=False)
        except (TypeError, ValueError, KeyError):
            raise ValueError('Native actor arguments/items are not valid JSON objects') from None
        self._items = self._prepared['items'] + items
        self._framework = self._prepared['framework']
        self._task_id = self._prepared['task_id']
        self._pending = dict(name=call['name'], arguments=arguments, call_id=call_id)
        self._next_turn += 1
        self._prepared = None
        return deepcopy({k: self._pending[k] for k in ('name', 'arguments')})
