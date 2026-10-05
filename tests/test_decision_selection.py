import asyncio
import json
from pathlib import Path

import httpx
import pytest

from openwebrl.decision_selection import DecisionSelector, selection_payload, selected_index


def observation():
    return dict(active_tab_url='https://example.com', screenshot=b'fixture-png',
                selection_page=dict(url='https://example.com', text='Search'))


def response(model='kev-latest', choice='4'):
    return dict(model=model, answers=dict(selection=dict(type='choice', choice=choice,
        probabilities={str(i): float(str(i)==choice) for i in range(1,6)})))


def test_selection_uses_five_full_candidates_recent_history_and_no_hidden_reference():
    candidates=[dict(thought=f'reason{i}',action=f'action{i}') for i in range(5)]
    payload=selection_payload('kev-latest','find',observation(),[str(i) for i in range(9)],candidates)
    assert payload['state']['candidates']['5']==candidates[-1]
    assert len(payload['state']['recent_history'])==5
    assert selected_index(payload,response())==3
    with pytest.raises(ValueError):selection_payload('kev-latest','find',observation(),[],candidates[:4])
    with pytest.raises(ValueError):selection_payload('kev-latest','find',dict(observation(),active_tab_url='changed'),[],candidates)


@pytest.mark.parametrize('bad',[
    dict(response(),model='wrong'),dict(response(),truncated=True),
    dict(response(),answers={}),
    dict(response(),answers=dict(selection=dict(type='choice',choice='1',probabilities={'1':float('nan')}))),
    dict(response(),answers=dict(selection=dict(type='choice',choice='1',probabilities={str(i):float(i==4) for i in range(1,6)}))),
])
def test_invalid_selection_never_falls_back(bad):
    payload=selection_payload('kev-latest','find',observation(),[],[dict(action='a')]*5)
    with pytest.raises(ValueError):selected_index(payload,bad)


def test_exact_actor_tuple_returned_seeds_differ_and_kev_receives_no_api_key(tmp_path):
    async def run():
        outputs=[];seeds=[];requests=[]
        async def infer(url,text,params,images,timeout_secs):
            seeds.append(params['sampling_seed']);out=(f'reason{len(seeds)}</think>{{"write":"literal"}}',[len(seeds)],[.1],'stop')
            outputs.append(out);return out
        def reply(request):
            requests.append(request);return httpx.Response(200,json=response())
        async with httpx.AsyncClient(transport=httpx.MockTransport(reply)) as client:
            selector=DecisionSelector('kev',tmp_path,client=client,endpoint='http://127.0.0.1:1234/v1/systemone',api_key='SECRET',max_requests=1)
            kwargs=dict(infer=infer,url='actor',input_text='prompt',sampling_params={'temperature':.6},images=[],
                observation=observation(),history=[],task='search',task_id='fixture',turn=0,timeout=30)
            output,meta=await selector(**kwargs)
            assert output is outputs[3] and meta['selected_index']==3
            assert len(set(seeds))==5
            assert 'authorization' not in requests[0].headers
            with pytest.raises(RuntimeError):await selector(**kwargs)
        record=json.loads((tmp_path/'request-00001.json').read_text())
        assert record['actor_outputs']==[o[0] for o in outputs]
        assert 'SECRET' not in (tmp_path/'request-00001.json').read_text()
    asyncio.run(run())


def test_failed_http_is_durable_and_halts_without_retry(tmp_path):
    async def run():
        async def infer(*args,**kwargs):return ('</think>action',[],[],'stop')
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _:httpx.Response(503))) as client:
            selector=DecisionSelector('jev',tmp_path,client=client,api_key='fixture')
            with pytest.raises(httpx.HTTPStatusError):
                await selector(infer=infer,url='actor',input_text='prompt',sampling_params={},images=[],observation=observation(),history=[],task='find',task_id='x',turn=0,timeout=30)
            assert selector.halted
        assert json.loads((tmp_path/'request-00001.json').read_text())['status']=='failed'
        assert (tmp_path/'halt.json').exists()
    asyncio.run(run())


def test_terminal_screenshot_crosses_nested_task_boundary_without_episode_leakage():
    from openwebrl.decision_selection_eval import FINAL_SCREENSHOT, remember_screenshot
    async def episode(image):
        token = FINAL_SCREENSHOT.set({'image': None})
        try:
            async def generation_child():
                await asyncio.sleep(0)
                remember_screenshot(image)
            await asyncio.wait_for(generation_child(), timeout=1)
            return FINAL_SCREENSHOT.get()['image']
        finally:
            FINAL_SCREENSHOT.reset(token)
    async def run():
        assert await asyncio.gather(episode(b'A'), episode(b'B')) == [b'A', b'B']
        assert FINAL_SCREENSHOT.get() is None
    asyncio.run(run())


def test_missing_observation_stops_before_actor_or_selector_and_valid_tuple_is_unchanged(tmp_path):
    from openwebrl.decision_selection import ObservationGuard
    calls = []
    original = ('actor result', [], [], 'stop')
    async def select(**kwargs):
        calls.append(kwargs)
        return original
    async def run():
        guard = ObservationGuard(select, tmp_path)
        for observation in ({}, {'screenshot': None}, {'screenshot': b''}):
            with pytest.raises(RuntimeError, match='no screenshot'):
                await guard(observation=observation, task_id='closed-browser', turn=15)
        assert not calls
        assert await guard(observation={'screenshot': b'valid-image'}, task_id='ok', turn=0) is original
        assert len(calls) == 1
    asyncio.run(run())
    record = json.loads(next(tmp_path.glob('*.json')).read_text())
    assert record['actor_called'] is record['selector_called'] is False
    assert record['turn'] == 15


@pytest.mark.parametrize('model,values,choice,valid', [
    ('jev-1.13.0', [.56, .15, .10, .04, .14], '1', True),
    ('jev-1.13.0', [.56, .15, .10, .06, .14], '1', True),
    ('jev-1.13.0', [.56, .15, .10, .15, .14], '1', False),
    ('kev-latest', [.56, .15, .10, .04, .14], '1', False),
    ('jev-1.13.0', [.561, .15, .10, .04, .14], '1', False),
    ('jev-1.13.0', [.56, .15, .10, .04, .14], '2', False),
])
def test_jev_rounded_probabilities_preserve_argmax_and_reject_other_errors(model, values, choice, valid):
    payload = selection_payload(model, 'find', observation(), [], [dict(action='a')]*5)
    reply = response(model, choice)
    reply['answers']['selection']['probabilities'] = dict(zip('12345', values))
    original = json.dumps(reply)
    if valid:
        assert selected_index(payload, reply) == 0
    else:
        with pytest.raises(ValueError):
            selected_index(payload, reply)
    assert json.dumps(reply) == original
