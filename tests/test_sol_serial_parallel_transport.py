"""Offline regression for native reasoning round trips after HTTP 400."""
from copy import deepcopy

import pytest

from openwebrl.sol_serial_parallel_transport import actor_transport_input


def test_reasoning_roundtrip_preserves_opaque_state_without_response_status():
    items = [dict(role='system', content='policy'), dict(role='user', content='snapshot'),
        dict(type='reasoning', id='rs_1', status=None, summary=[],
             content=None, encrypted_content='opaque-encrypted-bytes'),
        dict(type='function_call', id='fc_1', status='completed', call_id='call_1',
             name='click', arguments='{"x": 1}'),
        dict(type='function_call_output', call_id='call_1', output=[
            dict(type='input_text', text='actual feedback'),
            dict(type='input_image', image_url='data:image/png;base64,AA==', detail='high')])]
    saved = deepcopy(items)
    result = actor_transport_input(items)
    assert result[2] == dict(type='reasoning', id='rs_1', summary=[],
                             encrypted_content='opaque-encrypted-bytes')
    assert result[3] == {k: v for k, v in items[3].items() if k != 'status'}
    assert result[4] == items[4]
    assert items == saved
    assert actor_transport_input(result) == result
    result[4]['output'][0]['text'] = 'mutated request'
    assert items == saved


def test_preserves_reasoning_summary_content_and_assistant_phase():
    items = [dict(type='reasoning', id='rs_2', status='completed',
        summary=[dict(type='summary_text', text='summary')],
        content=[dict(type='reasoning_text', text='provided text')],
        encrypted_content='opaque'), dict(type='message', role='assistant', id='msg_1',
        status='completed', phase='commentary',
        content=[dict(type='output_text', text='text', annotations=[])])]
    result = actor_transport_input(items)
    assert result[0] == {k: v for k, v in items[0].items() if k != 'status'}
    assert result[1] == items[1]


def test_unknown_reasoning_fields_stop_locally():
    with pytest.raises(ValueError, match='Unknown native reasoning fields'):
        actor_transport_input([dict(type='reasoning', future_context='must not drop')])
