import asyncio
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_reasoning_actors_full300 as controller
from openwebrl.luna_qwen_metrics import digest
from openwebrl.luna_qwen_policy import Budget
from openwebrl.reasoning_actor_policy import ActorPolicy, MeteredActorAPI, metric_prices


class Data:
    def __init__(self, value): self.value = value
    def model_dump(self): return self.value


class Response:
    def __init__(self, model='gpt-6.1-sol', status='completed'):
        self.model, self.status = model, status
        self.usage = Data(dict(input_tokens=10000, output_tokens=300,
            input_tokens_details=dict(cached_tokens=2000, cache_write_tokens=7000),
            output_tokens_details=dict(reasoning_tokens=250)))
        self.output = [dict(type='function_call', id='fc_test', call_id='call_test',
                            name='click', arguments='{"x":640,"y":360}')]
    def model_dump(self):
        return dict(model=self.model, status=self.status, usage=self.usage.model_dump(), output=deepcopy(self.output),
            incomplete_details={'reason': 'max_output_tokens'} if self.status != 'completed' else None)


def request(model='gpt-6.1-sol'):
    return dict(model=model, reasoning={'effort':'high'}, max_output_tokens=4096,
        service_tier='default', store=False, tool_choice='required', parallel_tool_calls=False,
        input=[dict(role='user',content=[dict(type='input_text',text='task'),
            dict(type='input_image',image_url='data:image/png;base64,placeholder')])],
        tools=[dict(type='function',name='click',parameters={'type':'object'})])


def test_sol_cost_uses_sol_prices_and_preserves_reasoning_as_output_subset(tmp_path):
    async def create(**kwargs): return Response()
    budget=Budget(tmp_path/'budget',100,9900)
    api=MeteredActorAPI(SimpleNamespace(responses=SimpleNamespace(create=create)),budget,tmp_path/'episode',
        'gpt-6.1-sol',metric_prices('gpt-6.1-sol'))
    asyncio.run(api.create('actor',**request()))
    ledger=json.loads((budget.root/'ledger.json').read_text())
    # 1K normal +2K cache reads +7K cache writes +300 total output.
    assert ledger['charged_or_reserved_usd']==pytest.approx((1000*2+2000*.1+7000*2.5+300*10)/1e6)
    row=json.loads(next((tmp_path/'episode/requests').glob('*.json')).read_text())
    assert row['provider']=='gpt-6.1-sol' and row['usage']['output_tokens']==300
    assert row['request']['reasoning']=={'effort':'high'}


@pytest.mark.parametrize('changes',[{'model':'gpt-6-luna'}, {'reasoning':{'effort':'medium'}},
    {'max_output_tokens':8192},{'temperature':1.},{'service_tier':'priority'}])
def test_changed_protocol_refused_before_spending(tmp_path,changes):
    budget=Budget(tmp_path,100,9900)
    api=MeteredActorAPI(None,budget,tmp_path/'episode','gpt-6.1-sol',metric_prices('gpt-6.1-sol'))
    with pytest.raises(ValueError,match='protocol'):
        asyncio.run(api.create('actor',**dict(request(),**changes)))
    assert not (tmp_path/'ledger.json').exists()


@pytest.mark.parametrize('response',[Response(status='incomplete'),Response(model='gpt-6-luna')])
def test_incomplete_or_substituted_response_preserves_usage_and_halts(tmp_path,response):
    async def create(**kwargs):return response
    budget=Budget(tmp_path/'budget',100,9900)
    api=MeteredActorAPI(SimpleNamespace(responses=SimpleNamespace(create=create)),budget,tmp_path/'episode',
        'gpt-6.1-sol',metric_prices('gpt-6.1-sol'))
    with pytest.raises(ValueError):asyncio.run(api.create('actor',**request()))
    assert (budget.root/'halt.json').exists()
    row=json.loads(next((tmp_path/'episode/requests').glob('*.json')).read_text())
    assert row['status']=='received' and row['usage']['output_tokens']==300
    assert row['response']['model']==response.model


def test_unknown_interrupted_cost_is_reserved_and_prevents_overspending(tmp_path):
    async def create(**kwargs):raise TimeoutError('test')
    budget=Budget(tmp_path/'budget',.16,9900)
    api=MeteredActorAPI(SimpleNamespace(responses=SimpleNamespace(create=create)),budget,tmp_path/'episode',
        'gpt-6.1-sol',metric_prices('gpt-6.1-sol'))
    with pytest.raises(TimeoutError):asyncio.run(api.create('actor',**request()))
    first=json.loads((budget.root/'ledger.json').read_text())
    assert first['charged_or_reserved_usd']>0 and first['calls']==1
    with pytest.raises(RuntimeError,match='budget exhausted'):asyncio.run(api.create('actor',**request()))
    assert json.loads((budget.root/'ledger.json').read_text())==first


def test_pixel_action_roundtrip_and_exact_high_model_request(tmp_path):
    captured=[]
    async def create(role,**kwargs):captured.append((role,kwargs));return Response()
    api=SimpleNamespace(model='gpt-6.1-sol',create=create)
    policy=ActorPolicy('sol61_high',tmp_path,api,None,{},{});
    policy.set_context([dict(role='user',content='goal')],
        [dict(function=dict(name='click',parameters={'type':'object'}))],
        SimpleNamespace(encode=lambda text,**kwargs:[1,2]),42)
    generated,meta=asyncio.run(policy(observation={'screenshot':b'test'},turn=0))
    assert '"x": 640, "y": 360' in generated[0]
    assert meta['browser_coordinate_space']=='viewport_pixels' and meta['mode']=='sol61_high'
    assert captured[0][1]['model']=='gpt-6.1-sol' and captured[0][1]['reasoning']=={'effort':'high'}


def test_separate_resource_and_api_approvals_and_retry_accounting():
    proposal={'test':'immutable'}
    approval=dict(approved=True,plan_sha256=digest(proposal),
        pools={m:dict(controller.RESOURCES) for m in controller.MODELS},
        actor_limits=deepcopy(controller.ACTOR_LIMITS),judge_limits=dict(controller.JUDGE_LIMITS),
        attempts=[dict(pool='sol61_high',job_id='2',time_limit_seconds=20000)])
    controller.validate_approval(approval,proposal)
    controller.check_attempt(approval,'sol61_high','2',1600)
    with pytest.raises(ValueError,match='six-hour'):controller.check_attempt(approval,'sol61_high','2',1601)
    with pytest.raises(ValueError,match='registered'):controller.check_attempt(approval,'luna_high','2',0)
    changed=deepcopy(approval);changed['actor_limits']['sol61_high']['max_usd']=101
    with pytest.raises(ValueError,match='Exact approved'):controller.validate_approval(changed,proposal)
    with pytest.raises(ValueError,match='Exact approved'):controller.validate_approval(dict(approval,approved=False),proposal)


def test_cpu_workers_do_not_request_gpus_or_overlap():
    cmd=controller.worker_command('2',Path('/cfg'))
    assert '--gres=none' in cmd and '--cpus-per-task=4' in cmd and '--mem=8G' in cmd
    assert '--overlap' not in cmd and not any(x.startswith('--gpus') for x in cmd)
