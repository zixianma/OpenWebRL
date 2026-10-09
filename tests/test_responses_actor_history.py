"""Offline native Responses replay, task isolation, and conservative accounting."""
import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
from openai.types.responses import ResponseFunctionToolCall, ResponseReasoningItem

from openwebrl.luna_qwen_policy import Policy, Budget, MeteredAPI
from openwebrl.reasoning_actor_policy import ActorPolicy, MeteredActorAPI, metric_prices
from openwebrl.responses_actor_history import (ResponsesActorHistory, SCREENSHOT_MARKER,
                                             response_input_bound)


def observation(text='Goal and current page', image='FIRST', tool=False):
    text += '\n' + SCREENSHOT_MARKER
    if tool:
        text = '<tool_response>\n' + text + '\n</tool_response>'
    return dict(role='user', content=[dict(type='text', text=text),
        dict(type='image_url', image_url='data:image/png;base64,' + image)])


def outputs(number=0):
    return [ResponseReasoningItem(id=f'rs_{number}', type='reasoning', summary=[],
                encrypted_content=f'synthetic-opaque-state-{number}', status='completed'),
            ResponseFunctionToolCall(id=f'fc_{number}', type='function_call', call_id=f'call_{number}',
                name='click', arguments='{"point_2d":[640,360]}', status='completed')]


def assistant(action):
    return dict(role='assistant', content='<think>\n</think><tool_call>' + json.dumps(action) + '</tool_call>')


def strip_images(messages):
    result = deepcopy(messages)
    for message in result:
        if isinstance(message['content'], list):
            message['content'] = [p for p in message['content'] if p['type'] != 'image_url']
            for part in message['content']:
                part['text'] = part['text'].replace(SCREENSHOT_MARKER, '')
    return result


def image_count(items):
    return sum(part.get('type') == 'input_image' for item in items
               for part in (item.get('content', item.get('output', [])) or []) if isinstance(part, dict))


def test_three_turn_native_outputs_call_ids_errors_images_and_no_mutation():
    history = ResponsesActorHistory()
    messages = [dict(role='system', content=''), observation()]
    request = history.prepare(messages, turn=0, task_id='a')
    assert request[0]['content'].strip()
    assert 'exactly one native function call' in request[0]['content']
    assert '<tool_call>' not in request[0]['content']
    native = outputs()
    original = [item.model_dump() for item in native]
    action = history.accept(native, {'click'})
    request[-1]['content'][0]['text'] = 'outside mutation'
    messages = strip_images(messages) + [assistant(action), observation('ERROR: target unavailable', 'SECOND', True)]
    second = history.prepare(messages, turn=1, task_id='a')
    assert second[2:4] == original
    assert second[-1]['type'] == 'function_call_output' and second[-1]['call_id'] == 'call_0'
    assert 'ERROR: target unavailable' in second[-1]['output'][0]['text']
    assert image_count(second) == 1
    assert 'Goal and current page' in second[1]['content'][0]['text']
    action = history.accept(outputs(1), {'click'})
    messages = strip_images(messages) + [assistant(action), observation('Click succeeded', 'THIRD', True)]
    third = history.prepare(messages, turn=2, task_id='a')
    assert third[2:4] == original
    assert third[5:7] == [item.model_dump() for item in outputs(1)]
    assert image_count(third) == 1 and third[-1]['call_id'] == 'call_1'
    assert [item.model_dump() for item in native] == original
    assert '<tool_call>' not in json.dumps(third)


@pytest.mark.parametrize('problem', ['changed_task', 'changed_action', 'missing_feedback', 'earlier_context'])
def test_mismatched_history_rejected_before_next_request(problem):
    history = ResponsesActorHistory()
    messages = [observation()]
    history.prepare(messages, turn=0, task_id='a')
    action = history.accept(outputs(), {'click'})
    messages += [assistant(action), observation('actual feedback', tool=True)]
    task = 'a'
    if problem == 'changed_task': task = 'b'
    if problem == 'changed_action': messages[-2] = assistant(dict(name='click', arguments={'point_2d': [1, 2]}))
    if problem == 'missing_feedback': messages[-1] = observation()
    if problem == 'earlier_context': messages[0]['content'][0]['text'] += 'altered'
    with pytest.raises(ValueError): history.prepare(messages, turn=1, task_id=task)


@pytest.mark.parametrize('problem', ['no_call', 'multi_call', 'missing_id', 'unknown_tool', 'bad_arguments'])
def test_invalid_response_never_becomes_executable(problem):
    history = ResponsesActorHistory()
    history.prepare([observation()], turn=0, task_id='a')
    native = [item.model_dump() for item in outputs()]
    if problem == 'no_call': native = native[:1]
    if problem == 'multi_call': native.append(dict(native[-1], call_id='other'))
    if problem == 'missing_id': native[-1].pop('call_id')
    if problem == 'unknown_tool': native[-1]['name'] = 'unregistered'
    if problem == 'bad_arguments': native[-1]['arguments'] = '[]'
    with pytest.raises(ValueError): history.accept(native, {'click'})


def test_new_task_starts_clean_and_requires_nonempty_prompt(tmp_path):
    prompt = tmp_path / 'prompt.md'
    prompt.write_text('Reviewed native API policy')
    history = ResponsesActorHistory(prompt)
    history.prepare([observation()], turn=0, task_id='a')
    history.accept(outputs(), {'click'})
    request = history.prepare([observation('Different task')], turn=0, task_id='b')
    assert len(request) == 2 and 'call_0' not in json.dumps(request)
    prompt.write_text('   ')
    with pytest.raises(ValueError, match='empty'):
        ResponsesActorHistory(prompt).prepare([observation()], turn=0)
    with pytest.raises(ValueError, match='unavailable'):
        ResponsesActorHistory(tmp_path/'absent.md').prepare([observation()], turn=0)


def test_size_bound_counts_native_opaque_state_but_not_image_base64():
    items = [dict(type='reasoning', encrypted_content='x'*100),
             dict(type='function_call_output', call_id='call', output=[
                 dict(type='input_text', text='error'), dict(type='input_image', image_url='data:image/png;base64,X')])]
    initial = response_input_bound(items, [])
    changed = deepcopy(items)
    changed[0]['encrypted_content'] += 'x'*100
    assert response_input_bound(changed, []) == initial + 100
    changed[-1]['output'][-1]['image_url'] += 'X'*100000
    assert response_input_bound(changed, []) == initial + 100


@pytest.mark.parametrize('mode', ['luna', 'luna_high', 'sol61_high'])
def test_both_policy_paths_replay_native_state_with_exact_effort(mode, tmp_path):
    captured = []
    async def create(role, **kwargs):
        captured.append((role, deepcopy(kwargs)))
        return SimpleNamespace(output=outputs(len(captured)-1))
    api = SimpleNamespace(create=create, model='gpt-6.1-sol' if mode == 'sol61_high' else 'gpt-6-luna')
    cls = Policy if mode == 'luna' else ActorPolicy
    policy = cls(mode, tmp_path, api, None, {}, {})
    tokenizer = SimpleNamespace(encode=lambda text, **kwargs: list(text.encode()))
    tools = [dict(name='click', parameters={'type':'object'})]
    messages = [dict(role='system', content=''), observation()]
    async def run():
        nonlocal messages
        for turn in range(2):
            policy.set_context(messages, tools, tokenizer, 100)
            generated, _ = await policy(infer=None, url='unused', input_text='unused', sampling_params={},
                images=[], observation={'screenshot': b'offline'}, history=[], task='goal', task_id='a', turn=turn, timeout=10)
            messages = strip_images(messages) + [dict(role='assistant', content=generated[0]),
                observation('Execution failed: target not available', f'NEXT{turn}', True)]
    asyncio.run(run())
    request = captured[1][1]
    assert request['model'] == api.model
    assert request['reasoning'] == {'effort': 'medium' if mode == 'luna' else 'high'}
    assert request['max_output_tokens'] == 4096 and request['store'] is False
    assert request['parallel_tool_calls'] is False and request['tool_choice'] == 'required'
    assert request['input'][2:4] == [item.model_dump() for item in outputs()]
    assert request['input'][-1]['call_id'] == 'call_0' and image_count(request['input']) == 1
    assert not {'temperature', 'top_p', 'previous_response_id'}.intersection(request)


@pytest.mark.parametrize('high', [False, True])
def test_oversized_native_state_fails_before_api_or_reservation(tmp_path, high):
    budget = Budget(tmp_path/'budget', 1., 10)
    api = (MeteredActorAPI(None, budget, tmp_path, 'gpt-6.1-sol', metric_prices('gpt-6.1-sol')) if high
           else MeteredAPI(None, budget, tmp_path))
    kwargs = dict(model='gpt-6.1-sol' if high else 'gpt-6-luna', reasoning={'effort':'high'},
        max_output_tokens=4096, service_tier='default', store=False, tool_choice='required', parallel_tool_calls=False,
        input=[dict(type='reasoning', encrypted_content='x'*272000)], tools=[])
    with pytest.raises(ValueError, match='Conservative request size'):
        asyncio.run(api.create('actor', **kwargs))
    assert not (budget.root/'ledger.json').exists()
