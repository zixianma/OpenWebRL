import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from types import SimpleNamespace
import importlib.util
from pathlib import Path

import httpx
import pytest

from openwebrl.selection_budget import SelectionBudget
from openwebrl.decision_selection import DecisionSelector
from openwebrl.decision_selection_eval import score_step_limit, actor_sampling, PROTOCOL, FULL300_PROTOCOL


@pytest.mark.parametrize('protocol,temperature',[(PROTOCOL,.6),(FULL300_PROTOCOL,1.0)])
def test_five_actor_requests_use_frozen_sampling(tmp_path,protocol,temperature):
    async def run():
        seen=[]
        async def infer(url,text,params,images,timeout_secs):
            seen.append(params)
            return ('<think>reason</think>{"name":"wait","arguments":{}}',[],[],'stop')
        reply={'model':'kev-latest','answers':{'selection':{'type':'choice','choice':'1',
            'probabilities':{'1':1.,'2':0.,'3':0.,'4':0.,'5':0.}}}}
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _:httpx.Response(200,json=reply))) as client:
            selector=DecisionSelector('kev',tmp_path,client=client,endpoint='http://127.0.0.1:1234/v1/systemone')
            await selector(infer=infer,url='actor',input_text='prompt',sampling_params=actor_sampling(protocol),
                images=[],observation={'screenshot':b'fixture','active_tab_url':'https://example.com',
                'selection_page':{'url':'https://example.com'}},history=[],task='task',task_id='id',turn=0,timeout=30)
        assert len(seen)==5
        for params in seen:
            assert params['temperature']==temperature and params['top_p']==.95
            assert params['max_new_tokens']==4096 and params['top_k']==20
        saved=json.loads((tmp_path/'request-00001.json').read_text())
        assert saved['sampling']['temperature']==temperature
    asyncio.run(run())


def test_sampling_rejects_unrecorded_protocol_changes():
    with pytest.raises(ValueError,match='protocol changed'):
        actor_sampling(dict(FULL300_PROTOCOL,max_steps=60))


def test_shared_budget_cannot_reset_or_overbook(tmp_path):
    limits = {'browser_sessions': 7}
    def reserve(_):
        try:
            SelectionBudget(tmp_path, limits).reserve('browser_sessions')
            return True
        except RuntimeError:
            return False
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert sum(pool.map(reserve, range(20))) == 7
    with pytest.raises(RuntimeError, match='exhausted'):
        SelectionBudget(tmp_path, limits).reserve('browser_sessions')
    with pytest.raises(ValueError, match='cannot reset'):
        SelectionBudget(tmp_path, {'browser_sessions': 8}).reserve('browser_sessions')
    assert json.loads((tmp_path/'usage.json').read_text())['reserved']['browser_sessions'] == 7


def test_actor_reservations_survive_failed_generation(tmp_path):
    async def run():
        calls = []
        async def infer(*args, **kwargs):
            calls.append(1)
            raise RuntimeError('generation failure')
        async with httpx.AsyncClient() as client:
            selector = DecisionSelector('kev', tmp_path/'traces', client=client,
                endpoint='http://127.0.0.1:1234/v1/systemone',
                budget=SelectionBudget(tmp_path/'budget', {'actor_proposals': 5}))
            kwargs = dict(infer=infer, url='actor', input_text='x', sampling_params={},
                images=[], observation={}, history=[], task='task', task_id='id', turn=0, timeout=1)
            with pytest.raises(RuntimeError, match='generation failure'):
                await selector(**kwargs)
            with pytest.raises(RuntimeError, match='budget exhausted'):
                await selector(**kwargs)
        assert len(calls) == 5
    asyncio.run(run())


@pytest.mark.parametrize('succeeds,cap,expected_calls', [(True, 3, 2), (False, 3, 3), (True, 1, 1)])
def test_jev_contradictory_response_retry_preserves_candidates_and_charges_calls(tmp_path, succeeds, cap, expected_calls):
    async def run():
        actor_calls, requests = [], []
        outputs = [(f'<think>reason {i}</think>action {i}', [], [], 'stop') for i in range(5)]
        async def infer(*args, **kwargs):
            output = outputs[len(actor_calls)]; actor_calls.append(output); return output
        bad = {'model':'jev-1.13.0','answers':{'selection':{'type':'choice','choice':'1',
            'probabilities':{'3':.14,'4':.07,'1':.35,'2':.08,'5':.36}}}}
        good = json.loads(json.dumps(bad)); good['answers']['selection']['choice'] = '5'
        def handle(request):
            requests.append(request.content)
            return httpx.Response(200, json=good if succeeds and len(requests)>1 else bad)
        budget = SelectionBudget(tmp_path/'budget', {'actor_proposals':5,'jev_requests':cap})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            selector = DecisionSelector('jev', tmp_path/'traces', client=client, api_key='fixture',
                                        validation_retries=2, budget=budget)
            kwargs = dict(infer=infer,url='actor',input_text='prompt',sampling_params={},images=[],
                observation={'screenshot':b'fixture','active_tab_url':'https://example.com',
                             'selection_page':{'url':'https://example.com'}},
                history=[],task='task',task_id='id',turn=0,timeout=30)
            if succeeds and cap>1:
                result, meta = await selector(**kwargs)
                assert result is outputs[4] and meta['fallback'] is None and not selector.halted
            else:
                with pytest.raises((ValueError, RuntimeError)):
                    await selector(**kwargs)
                assert selector.halted
        assert len(actor_calls)==5 and len(requests)==expected_calls and len(set(requests))==1
        usage=json.loads((tmp_path/'budget/usage.json').read_text())['reserved']
        assert usage=={'actor_proposals':5,'jev_requests':expected_calls}
        saved=json.loads((tmp_path/'traces/request-00001.json').read_text())
        assert saved['http_attempts'][0]['response']==bad
        assert saved['http_attempts'][0]['validation_error']=='Decision choice disagrees with probability argmax'
        assert len(saved['http_attempts'])==expected_calls
        assert saved['actor_outputs']==[o[0] for o in outputs]
    asyncio.run(run())


def test_step_limit_uses_same_judge_and_preserves_termination():
    failed = SimpleNamespace(name='FAILED')
    sample = SimpleNamespace(status=failed, metadata={'terminate_reason': 'max_steps_exhausted',
        'messages': ['exact history'], 'full_image_list': ['exact final image']}, remove_sample=False)
    calls = []
    async def judge(args, value, parser):
        assert value is sample and value.status is failed
        calls.append(value.metadata.copy())
        return 0., 'Status: failure', False
    async def skipped(*unused):
        raise AssertionError('Shared status-skipping wrapper must not run')
    reward = SimpleNamespace(_judge=judge, reward_func=skipped, _TOOLS_INFO=[],
        resolve_parser_type=lambda _: 'parser', ToolParser=lambda *args, **kwargs: 'tool-parser')
    result = asyncio.run(score_step_limit(SimpleNamespace(hf_checkpoint='actor'), [sample], reward))
    assert result == [0.]
    assert sample.status is failed and sample.metadata['terminate_reason']=='max_steps_exhausted'
    assert calls[0]['messages']==['exact history'] and calls[0]['full_image_list']==['exact final image']
    assert sample.metadata['reward']['judge_text']=='Status: failure'


def test_aborted_episode_does_not_get_step_limit_judging():
    async def normal(args, sample):return 'unchanged invalid handling'
    sample=SimpleNamespace(status=SimpleNamespace(name='ABORTED'),
        metadata={'terminate_reason':'generation_error'})
    assert asyncio.run(score_step_limit(None,sample,SimpleNamespace(reward_func=normal)))=='unchanged invalid handling'


def test_supervisor_queues_once_until_ack_and_exits_on_verified_completion(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('selection_supervisor',
        Path(__file__).resolve().parents[1]/'scripts/supervise_sft_selection_full300.py')
    supervisor=importlib.util.module_from_spec(spec);spec.loader.exec_module(supervisor)
    # Use the production path length so the queue's1000-byte ceiling is tested.
    root=tmp_path/'evaluations'/'sft-kev27b-full300-20261004';root.mkdir(parents=True)
    pointer=tmp_path/'pointer.json'
    supervisor.write(root/'approval.json',{'approved':True})
    supervisor.write(root/'plan.json',{'modes':['kev-27b']})
    supervisor.write(pointer,{'job_id':'12345','verified_complete':False})
    calls=[]
    monkeypatch.setattr(supervisor.subprocess,'check_output',lambda *a,**k:'12345|PENDING|0|0:0\n')
    def queue(command,**kwargs):
        calls.append(command)
        return SimpleNamespace(stdout='queued',returncode=0)
    monkeypatch.setattr(supervisor.subprocess,'run',queue)
    assert supervisor.poll(root,pointer,'thread')
    assert supervisor.poll(root,pointer,'thread')
    assert len(calls)==1 and calls[0][:2]==['codex','queue']
    assert len(calls[0][-1].encode())<=1000
    supervisor.write(pointer,{'job_id':'12345','verified_complete':True})
    assert supervisor.poll(root,pointer,'thread') is False
    assert len(calls)==1
    assert supervisor.read(root/'supervisor-finished.json')['reason']=='verified completion'
