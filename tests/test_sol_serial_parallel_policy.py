"""No network: candidate isolation, concurrency, history lineage and hard errors."""
import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from openwebrl.sol_serial_parallel_policy import MODEL, SolSerialParallelPolicy, selector_input


class Response:
    def __init__(self, index=0, selected=None, arguments=None):
        self.model, self.status = MODEL, 'completed'
        self.output = [dict(type='reasoning', encrypted_content=f'private-{index}'),
            dict(type='function_call', name='click', call_id=f'call-{index}',
                 arguments=json.dumps(arguments if arguments is not None else {'x': index}))]
        self.output_text = json.dumps({'selected_index': selected})

    def model_dump(self):
        return deepcopy(vars(self))


TOOLS = [dict(type='function', name='click', parameters={'type': 'object',
    'properties': {'x': {'type': 'integer'}}, 'required': ['x'], 'additionalProperties': False})]


def policy(tmp_path, api, mode='parallel3_low'):
    value = SolSerialParallelPolicy(mode, tmp_path, api)
    value.set_context([dict(role='user', content='TASK + screenshot')], TOOLS,
                      SimpleNamespace(encode=lambda text, **kw: [1, 2]), 10)
    return value


def run(value):
    return asyncio.run(value(observation={'screenshot': b'image'}, turn=0, task_id='task'))


def test_parallel_isolated_inputs_concurrency_and_selected_history(tmp_path):
    class API:
        def __init__(self):
            self.calls, self.started, self.done = [], 0, 0
            self.ready = asyncio.Event()

        async def create(self, role, **kwargs):
            self.calls.append((role, deepcopy(kwargs)))
            if role == 'selector':
                assert self.done == 3
                assert 'private-' not in json.dumps(kwargs)
                assert 'selected_index' in kwargs['text']['format']['schema']['properties']
                return Response(selected=2)
            index = int(role[-1])
            self.started += 1
            if self.started == 3:
                self.ready.set()
            await asyncio.wait_for(self.ready.wait(), 1)
            # Each request owns an independent copy. No contamination is seen
            # by sibling requests or the prepared actor-history buffer.
            kwargs['input'].append({'role': 'user', 'content': f'MUTATED-{index}'})
            self.done += 1
            return Response(index)

    api = API()
    value = policy(tmp_path, api)
    _, metadata = run(value)
    assert metadata['selected_index'] == 2
    actor_calls = [kw for role, kw in api.calls if role != 'selector']
    assert actor_calls[0]['input'] == actor_calls[1]['input'] == actor_calls[2]['input']
    assert all(kw['reasoning'] == {'effort': 'low'} and kw['max_output_tokens'] == 4096 for kw in actor_calls)
    assert all(kw['include'] == ['reasoning.encrypted_content'] for kw in actor_calls)
    serialized = json.dumps(value.actor_history._items)
    assert 'private-2' in serialized and 'private-0' not in serialized and 'MUTATED' not in serialized
    assert value.actor_history._pending['call_id'] == 'call-2'
    row = json.loads((tmp_path / 'decisions/000.json').read_text())
    assert row['status'] == 'accepted' and len(row['candidate_responses']) == 3


def test_candidate_failure_drains_siblings_and_never_selects(tmp_path):
    class API:
        def __init__(self):
            self.finished = []

        async def create(self, role, **kwargs):
            if role == 'actor_low_0':
                raise RuntimeError('provider failure')
            assert role != 'selector'
            await asyncio.sleep(.01)
            self.finished.append(role)
            return Response(int(role[-1]))

    api = API()
    value = policy(tmp_path, api)
    with pytest.raises(RuntimeError, match='provider failure'):
        run(value)
    assert sorted(api.finished) == ['actor_low_1', 'actor_low_2']
    assert value.actor_history._next_turn == 0
    row = json.loads((tmp_path / 'decisions/000.json').read_text())
    assert row['status'] == 'failed' and len(row['candidate_responses']) == 3


@pytest.mark.parametrize('selected', [-1, 3, True, '1'])
def test_invalid_selection_stops_without_history_commit(tmp_path, selected):
    class API:
        def halt(self, error):
            (tmp_path / 'global-halt.json').write_text(type(error).__name__)

        async def create(self, role, **kwargs):
            return Response(selected=selected) if role == 'selector' else Response(int(role[-1]))

    value = policy(tmp_path, API())
    with pytest.raises(ValueError, match='Invalid selector index'):
        run(value)
    assert value.actor_history._next_turn == 0
    assert (tmp_path / 'global-halt.json').read_text() == 'ValueError'
    assert json.loads((tmp_path / 'decisions/000.json').read_text())['selector_response']


def test_invalid_native_schema_never_reaches_selector(tmp_path):
    class API:
        def halt(self, error):
            (tmp_path / 'global-halt.json').write_text(type(error).__name__)

        async def create(self, role, **kwargs):
            assert role != 'selector'
            return Response(int(role[-1]), arguments={'x': 'not an integer'})

    value = policy(tmp_path, API())
    with pytest.raises(Exception, match='not of type'):
        run(value)
    assert value.actor_history._next_turn == 0
    assert (tmp_path / 'global-halt.json').read_text() == 'ValidationError'


def test_duplicate_candidates_retained_and_serial_protocol(tmp_path):
    class API:
        def __init__(self):
            self.calls = []

        async def create(self, role, **kwargs):
            self.calls.append((role, kwargs))
            return Response(selected=1) if role == 'selector' else Response(0)

    api = API()
    value = policy(tmp_path / 'parallel', api)
    run(value)
    assert len(api.calls) == 4
    row = json.loads((tmp_path / 'parallel/decisions/000.json').read_text())
    assert row['candidates'][0] == row['candidates'][1] == row['candidates'][2]
    api = API()
    value = policy(tmp_path / 'serial', api, 'serial_high')
    run(value)
    assert len(api.calls) == 1
    role, kw = api.calls[0]
    assert role == 'actor_high' and kw['max_output_tokens'] == 12288
    assert kw['reasoning'] == {'effort': 'high'} and kw['parallel_tool_calls'] is False


def test_selector_projection_removes_all_private_generated_text():
    native = [dict(role='system', content='actor policy'),
        dict(role='user', content='visible task'),
        dict(type='reasoning', encrypted_content='secret'),
        dict(type='message', role='assistant', content='private prose'),
        dict(type='function_call', name='click', arguments='{"x":1}', call_id='old'),
        dict(type='function_call_output', call_id='old', output='page result')]
    result = selector_input(native, [{'name': 'click', 'arguments': {'x': 2}}] * 3)
    text = json.dumps(result)
    assert 'secret' not in text and 'private prose' not in text and 'actor policy' not in text
    assert 'visible task' in text and 'page result' in text and 'executed_action' in text


def test_next_turn_feedback_bound_only_to_selected_native_call(tmp_path):
    class API:
        def __init__(self):
            self.turn = 0
            self.calls = []

        async def create(self, role, **kwargs):
            self.calls.append((role, deepcopy(kwargs)))
            if role == 'selector':
                self.turn += 1
                return Response(selected=1)
            return Response(10 * self.turn + int(role[-1]))

    api = API()
    value = policy(tmp_path, api)

    async def exercise():
        reply, _ = await value(observation={'screenshot': b'first'}, turn=0, task_id='task')
        messages = value.messages + [dict(role='assistant', content=reply[0]),
            dict(role='user', content='<tool_response>\nactual feedback\n</tool_response>')]
        value.set_context(messages, TOOLS, value.tokenizer, 20)
        await value(observation={'screenshot': b'second'}, turn=1, task_id='task')

    asyncio.run(exercise())
    next_input = api.calls[4][1]['input']
    assert next_input[-1]['type'] == 'function_call_output'
    assert next_input[-1]['call_id'] == 'call-1'
    assert 'actual feedback' in json.dumps(next_input[-1])
    encoded = json.dumps(next_input)
    assert 'private-1' in encoded and 'private-0' not in encoded and 'private-2' not in encoded
    assert value.actor_history._pending['call_id'] == 'call-11'
