import concurrent.futures
import asyncio
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
from prepare_controlled_jev_actor import schedule
from run_controlled_jev_actor import validate_allocation, check_approval
from openwebrl.controlled_sft_state import digest
from openwebrl.controlled_jev_state import JevState
from openwebrl.controlled_jev_eval import (
    Transport,JevBudget,canonical_messages,private_exception_details,native_judge_tools,native_judge_reward_module,
)
from openwebrl.controlled_sft_worker import AttemptJournal


@pytest.fixture
def root(tmp_path):
    tasks = [dict(task_id=str(i), intent='fixture', start_url='about:blank') for i in range(300)]
    path = tmp_path/'tasks.jsonl'
    path.write_text(''.join(json.dumps(dict(metadata=t))+'\n' for t in tasks))
    plan = dict(root=str(tmp_path),task_file=str(path), tasks=tasks, schedule=schedule(tasks), conditions={'L06':{}},
        resources=dict(collectors=4, gpus=0, cpus=8, memory_gib=32, total_seconds=14400),
        limits=dict(browser_sessions=330, concurrent_browsers=4, jev_requests=19800,
            text_http_attempts=19800, text_usd=15., om2w_judge_http_attempts=1320, judge_usd=8.))
    (tmp_path/'plan.json').write_text(json.dumps(plan))
    return tmp_path


def state(root): return JevState(root, allow_unapproved_for_tests=True)


def finish(s, c, valid=True, judge=True):
    result = dict(task_id=c['task_id'], condition='L06', attempt_id=c['attempt_id'], valid=valid,
                  reward=0, browser_closed=True, terminal='done' if judge else 'blocked')
    path = Path(c['artifact_directory'])/'result.json'
    path.write_text(json.dumps(result))
    journal = AttemptJournal(path.parent/'events')
    journal('jev_received', dict(response={'model':'jev-1.13.0'}))
    if judge: journal('judge_received', dict(response={'model':'o4-mini-2025-04-16'}))
    s.finish(c['attempt_id'], dict(result, result_path=str(path), result_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))


def test_schedule_full300_and_fresh_smoke(root):
    plan = json.loads((root/'plan.json').read_text())
    rows = plan['schedule']
    assert len(rows) == 303 and len({r['item_id'] for r in rows}) == 303
    assert sum(r['category']=='smoke' for r in rows) == 3
    assert len({r['task_id'] for r in rows if r['category']=='primary'}) == 300


def test_unapproved_no_claim_or_spend(root):
    s = JevState(root)
    with pytest.raises(FileNotFoundError): s.claim('w')
    with pytest.raises(FileNotFoundError): s.reserve('jev_requests', attempt_id='x', condition='L06')


def test_separate_browser_and_call_budgets(root):
    s=state(root); c=s.claim('w'); b=JevBudget(s,c)
    r=b.reserve('browser_sessions')
    b.reserve('jev_requests', request={'model':'jev-1.13.0'})
    assert s.snapshot()['counters']['active_browsers']==1
    with pytest.raises(ValueError): b.settle(r,{'closed':False})
    b.settle(r,{'closed':True}); finish(s,c)
    assert s.snapshot()['counters']['active_browsers']==0


def test_helper_price_and_usage_settlement(root):
    s=state(root); c=s.claim('w'); b=JevBudget(s,c)
    r=b.reserve('text_http_attempts',request={'messages':[],'max_tokens':1024})
    before=s.snapshot()['counters']['text_usd']
    assert before>0
    b.settle(r,{'prompt_tokens':100,'completion_tokens':10})
    assert s.snapshot()['counters']['text_usd']==pytest.approx((100*.4+10*1.6)/1e6)
    assert s.snapshot()['counters']['text_http_attempts']==1


def test_unknown_helper_usage_retains_charge_and_halts(root):
    s=state(root); c=s.claim('w'); b=JevBudget(s,c)
    r=b.reserve('text_http_attempts',request={'max_tokens':1024})
    before=s.snapshot()['counters']['text_usd']
    with pytest.raises(RuntimeError): b.settle(r,{})
    p=Path(c['artifact_directory'])/'error.json';p.write_text('{}')
    b.finalize_unknown(r,p)
    assert s.snapshot()['counters']['text_usd']==before
    assert s.snapshot()['halt'] and s.snapshot()['unsettled_reservations']==0


def test_caps_halt_instead_of_spending_or_marking_remaining_tasks_failed(root):
    s=state(root); c=s.claim('w'); b=JevBudget(s,c)
    with s.connect() as db:db.execute("INSERT INTO counters VALUES('jev_requests',19800)")
    with pytest.raises(RuntimeError):b.reserve('jev_requests',request={})
    assert s.snapshot()['halt']['reason']=='reservation_rejected'
    assert s.snapshot()['counters']['jev_requests']==19800
    assert s.snapshot()['queue']['pending']==302


def test_judge_full_failed_charge_preserved(root):
    s=state(root);c=s.claim('w');b=JevBudget(s,c)
    r=b.reserve('om2w_judge_http_attempts',request={'max_completion_tokens':4096})
    before=s.snapshot()['counters']['judge_usd']
    p=Path(c['artifact_directory'])/'failed.json';p.write_text('{}')
    b.finalize_unknown(r,p)
    assert s.snapshot()['counters']['judge_usd']==before
    assert s.snapshot()['counters']['om2w_judge_http_attempts']==1


def test_valid_non_done_smokes_do_not_require_a_judge_call(root):
    s=state(root)
    while c:=s.claim('w'):finish(s,c,judge=False)
    s.enable_primary()
    assert s.snapshot()['phase']=='primary'
    assert s.snapshot()['judge_transport_status']=='pending_until_first_done'
    c=s.claim('w');finish(s,c,judge=True)
    assert s.snapshot()['judge_transport_status']=='verified'


def test_smoke_handoff_uses_separate_primary_attempts(root):
    s=state(root);ids=[]
    while c:=s.claim('w'):ids.append(c['attempt_id']);finish(s,c)
    s.enable_primary();c=s.claim('w')
    assert c['category']=='primary' and c['attempt_id'] not in ids


def test_parallel_global_cap_after_smoke(root):
    s=state(root)
    while c:=s.claim('w'):finish(s,c)
    s.enable_primary()
    def claim(i):
        try:return state(root).claim(str(i))
        except RuntimeError:return None
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:claims=list(pool.map(claim,range(5)))
    assert sum(c is not None for c in claims)==4


class Response:
    status_code=200
    text='{}'
    def __init__(self,raw):self.raw=raw
    def json(self):return self.raw


class Client:
    def __init__(self,response):self.response=response;self.calls=[]
    def post(self,*args,**kwargs):self.calls.append((args,kwargs));return self.response


def test_text_helper_preserves_native_recipe_and_metering(root):
    s=state(root);c=s.claim('w');j=AttemptJournal(Path(c['artifact_directory'])/'events')
    client=Client(Response({'model':'gpt-4.1-mini-2025-04-14','usage':{'prompt_tokens':1,'completion_tokens':1}}))
    transport=Transport(JevBudget(s,c),j,client=client)
    transport.post('https://api.openai.com/v1/chat/completions','test-key',{'model':'gpt-4.1-mini-2025-04-14','reasoning':{'enabled':False},'max_tokens':999})
    body=client.calls[0][1]['json']
    assert body['temperature']==.6 and body['top_p']==.95 and body['max_tokens']==1024 and 'reasoning' not in body
    assert s.snapshot()['counters']['text_http_attempts']==1


def test_jev_request_is_one_call_for_many_questions_and_no_images(root):
    s=state(root);c=s.claim('w');j=AttemptJournal(Path(c['artifact_directory'])/'events')
    client=Client(Response({'model':'jev-1.13.0'}));transport=Transport(JevBudget(s,c),j,client=client)
    body={'model':'jev-1.13.0','state':{'page':{'text':'fixture'}},'questions':{'operation':{},'click_target':{},'fill_target':{}}}
    transport.post('https://api.typesafe.ai/v1/systemone','test-key',body)
    assert len(client.calls)==1 and s.snapshot()['counters']['jev_requests']==1


def test_transport_failure_does_not_retry(root):
    class Failure(Client):
        def post(self,*args,**kwargs):self.calls.append(1);raise TimeoutError('fixture')
    s=state(root);c=s.claim('w');j=AttemptJournal(Path(c['artifact_directory'])/'events');client=Failure(None)
    transport=Transport(JevBudget(s,c),j,client=client)
    with pytest.raises(TimeoutError):transport.post('https://api.typesafe.ai/v1/systemone','key',{'model':'jev-1.13.0'})
    assert len(client.calls)==1 and s.snapshot()['halt']


def test_judge_mapping_empty_thoughts_actual_actions_and_native_done():
    rows=[dict(action={'kind':'fill','id':'n4','node':4,'label':'Search'},text='query',status='returned')]
    messages=canonical_messages(rows,'done')
    assert len(messages)==2 and all('<think>' not in m['content'] for m in messages)
    assert 'query' in messages[0]['content'] and '"name":"done"' in messages[1]['content']
    assert len(canonical_messages(rows,'action_limit'))==1
    assert 'Agent stopped' not in str(messages)


def test_native_judge_registry_preserves_every_action_and_exact_receipt_with_real_parser():
    from openwebrl.base.utils import ToolParser
    parser=ToolParser(native_judge_tools(),parser_type='qwen25')
    rows=[dict(number=i+1,action=dict(kind=kind,id=f'e{i}',node=i,label='Native target',
        value='original value',rect={'x':1,'y':2,'w':3,'h':4}),text='typed value' if kind=='fill' else None,
        status='returned',returned={'executed':f'e{i}'})
        for i,kind in enumerate(('click','fill','select','scroll','wait'))]
    frozen=json.dumps(rows,sort_keys=True)
    messages=canonical_messages(rows,'done')
    for message,row in zip(messages,rows):
        parsed=parser.parse(message['content'])
        assert parser._sglang_parser is not None, 'Exercise actual installed canonical parser, not regex fallback'
        assert parsed.success and len(parsed.calls)==1
        call=parsed.calls[0]
        assert call.name==row['action']['kind']
        params=json.loads(call.parameters) if isinstance(call.parameters,str) else call.parameters
        assert params==row
        assert '<think>' not in message['content']
    assert parser.parse(messages[-1]['content']).calls[0].name=='done'
    assert json.dumps(rows,sort_keys=True)==frozen
    with pytest.raises(ValueError,match='Unknown native'):
        canonical_messages([dict(action={'kind':'invented'})],'done')


def test_native_registry_never_mutates_canonical_module_even_on_exception():
    original_tools=[{'type':'function','function':{'name':'original'}}]
    original=SimpleNamespace(_TOOLS_INFO=original_tools,marker=object())
    with pytest.raises(RuntimeError):
        scoped=native_judge_reward_module(original)
        assert scoped is not original and scoped.marker is original.marker
        assert {x['function']['name'] for x in scoped._TOOLS_INFO}=={'click','fill','select','scroll','wait','done'}
        scoped._TOOLS_INFO[0]['function']['name']='local-only mutation'
        raise RuntimeError('Synthetic judge failure')
    assert original._TOOLS_INFO is original_tools
    assert original_tools==[{'type':'function','function':{'name':'original'}}]


def test_canonical_score_formats_native_actions_without_global_reward_changes(monkeypatch):
    from openwebrl import controlled_jev_eval as module
    from openwebrl.eval import reward_online_mind2web as reward
    original=reward._TOOLS_INFO
    monkeypatch.setattr(reward,'resolve_parser_type',lambda checkpoint:'qwen25')
    rows=[dict(number=1,action=dict(kind='select',id='e5',node=5,label='Sort By → Price',value='price asc'),
        text=None,status='returned',returned={'executed':'e5'})]
    captured=[]
    class Budget:
        def reserve(self,*args,**kwargs):return {'reservation_id':'offline'}
        def settle(self,*args,**kwargs):pass
        def finalize_unknown(self,*args,**kwargs):pass
    class Abort(BaseException):pass
    class Transport:
        async def judge(self,**request):
            captured.append(request)
            assert reward._TOOLS_INFO is original
            raise Abort('Stop at fake transport; no HTTP request')
    config=dict(parser_checkpoint='fixture',canonical_reward_sha256=hashlib.sha256(Path(reward.__file__).read_bytes()).hexdigest())
    with pytest.raises(Abort):
        asyncio.run(module.canonical_score({'task_id':'fixture','intent':'Sort results'},rows,'done',b'fixture-image',
            Transport(),Budget(),lambda *args:'offline-evidence',config))
    text=captured[0]['messages'][1]['content'][0]['text']
    assert 'select(' in text and 'price asc' in text and 'returned' in text and 'done(' in text
    assert '(no parseable action)' not in text and 'Thought 1: \nAction 1:' in text
    assert captured[0]['model']=='o4-mini-2025-04-16' and captured[0]['seed']==42
    assert reward._TOOLS_INFO is original


def test_exact_cpu_allocation_and_all_attempt_time(root):
    info='JobId=123 JobState=RUNNING UserId=u(42) NumNodes=1 NumCPUs=8 AllocTRES=cpu=8,mem=32G TimeLimit=04:00:00 EndTime=2030-01-01T04:00:00'
    assert validate_allocation(info,'123',14400,uid=42,now=0)==14220
    with pytest.raises(ValueError):validate_allocation(info.replace('mem=32G','mem=32G,gres/gpu=1'),'123',14400,uid=42,now=0)
    with pytest.raises(ValueError):validate_allocation(info.replace('mem=32G','mem=64G'),'123',14400,uid=42,now=0)
    p=json.loads((root/'plan.json').read_text())
    a=dict(approved=True,plan_sha256=digest(p),resources=p['resources'],limits=p['limits'],attempts=[dict(job_id='123',time_limit_seconds=10000)])
    assert check_approval(p,a,'123',4400)==10000
    with pytest.raises(ValueError):check_approval(p,a,'123',4401)


def test_interrupted_attempts_keep_all_browser_charges(root):
    s=state(root);c=s.claim('w');b=JevBudget(s,c);b.reserve('browser_sessions')
    s.mark_interrupted_after_teardown(teardown_verified=True,reason='test')
    assert s.snapshot()['queue']['interrupted']==1
    assert s.snapshot()['counters']['browser_sessions']==1


@pytest.mark.parametrize('message,terminal',[
    ('Text helper returned no valid field value; nothing typed.','typing_value_unavailable'),
    ('Invalid TypeSafe response; no action executed.','native_choice_invalid')])
def test_native_actor_refusal_is_valid_zero_not_infrastructure_exclusion(root,monkeypatch,message,terminal):
    from contextlib import contextmanager
    from openwebrl import controlled_jev_eval as m
    class Session:
        def __init__(self,*args):pass
        def open(self):return self
        def close(self):return []
    class Agent:
        def __init__(self,*args,**kwargs):
            self.browser=SimpleNamespace(page=SimpleNamespace(screenshot=lambda **kwargs:b'fresh-fixture'))
        def snapshot(self):return {'page':{},'history':[],'decisions':[]}
        def command(self,*args):raise ValueError(message)
    @contextmanager
    def native(*args):yield Agent
    class MockTransport:
        def __init__(self,*args):self.counts={'jev':1,'text':1,'judge':0};self.client=SimpleNamespace(close=lambda:None)
    async def score(task,operations,actual_terminal,*args):
        assert actual_terminal==terminal
        return 0.,{'judge':0.,'judge_text':'Judge not run'},'Status.ABORTED'
    monkeypatch.setattr(m,'LocalSession',Session);monkeypatch.setattr(m,'native_agent',native)
    monkeypatch.setattr(m,'Transport',MockTransport);monkeypatch.setattr(m,'canonical_score',score)
    s=state(root);c=s.claim('w')
    plan=dict(s.plan,worker_config={'browser_manifest':{},'canonical_reward_sha256':'fixture'})
    result=m.run_episode(c,plan,s)
    assert result['valid'] and result['score']==0 and result['api_attempts']['judge']==0
    assert s.snapshot()['halt'] is None and result['terminal']==terminal
    assert result['browser_closed'] and result['final_screenshot']['sha256']==hashlib.sha256(b'fresh-fixture').hexdigest()
    s.finish(c['attempt_id'],result)


def test_done_cannot_commit_without_pinned_judge_response(root):
    s=state(root);c=s.claim('w');p=Path(c['artifact_directory'])/'result.json'
    result=dict(task_id=c['task_id'],condition='L06',attempt_id=c['attempt_id'],valid=True,reward=0,terminal='done')
    p.write_text(json.dumps(result))
    with pytest.raises(ValueError,match='actual pinned'):
        s.finish(c['attempt_id'],dict(result,result_path=str(p),result_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    assert s.snapshot()['queue']['active']==1


@pytest.mark.parametrize('wandb_error',[False,True])
def test_wandb_finishes_before_orphan_group_sweep(monkeypatch,wandb_error):
    import run_controlled_jev_actor as controller
    order=[]
    monkeypatch.setattr(controller,'stop_owned',lambda children,job:order.append('workers') or [])
    monkeypatch.setattr(controller,'stop_browser_descendants',lambda job:order.append('orphans') or [])
    def finish(**kwargs):
        order.append('wandb')
        if wandb_error:raise RuntimeError('fixture')
    receipt=controller.teardown_owned([], '123',SimpleNamespace(finish=finish),1)
    assert order==['workers','wandb','orphans']
    assert receipt['verified'] and receipt['wandb']['finished'] is not wandb_error


def test_shutdown_signal_during_wandb_still_drains_orphans(monkeypatch):
    import run_controlled_jev_actor as controller
    order=[]
    monkeypatch.setattr(controller,'stop_owned',lambda children,job:[])
    monkeypatch.setattr(controller,'stop_browser_descendants',lambda job:order.append('drained') or [])
    def finish(**kwargs):raise SystemExit(143)
    with pytest.raises(SystemExit):controller.teardown_owned([], '123',SimpleNamespace(finish=finish),1)
    assert order==['drained']


def test_private_diagnostics_redact_embedded_credentials_and_url_userinfo():
    key='fixture-private-provider-key-19876'
    try:
        raise RuntimeError('net::ERR_CONNECTION_CLOSED key='+key+' https://user:password@example.test/path '
                           'Authorization: Bearer unknown-header-token api_key=unknown-query-token&next=ok')
    except RuntimeError as exc:
        result=private_exception_details(exc,environ={'TYPESAFE_API_KEY':key,'WANDB_API_KEY':'unused'})
    raw=json.dumps(result)
    assert key not in raw and 'user:password' not in raw
    assert 'unknown-header-token' not in raw and 'unknown-query-token' not in raw
    assert 'net::ERR_CONNECTION_CLOSED' in result['error_message']
    assert 'RuntimeError' in result['traceback'] and '[REDACTED]' in raw


def test_private_diagnostics_redact_before_truncation():
    key='fixture-secret-that-crosses-output-boundary'
    try:raise RuntimeError('x'*250+key+'y'*10000)
    except RuntimeError as exc:result=private_exception_details(exc,environ={'TEXT_MODEL_API_KEY':key},max_chars=256)
    assert 'fixture' not in result['error_message']
    assert len(result['error_message'])<=256 and len(result['traceback'])<=256
    assert result['error_message_truncated'] and result['traceback_truncated']


def test_private_diagnostics_redact_common_unbound_openai_key_pattern():
    result=private_exception_details(ValueError('bad sk-fixture0123456789abcdef'),environ={})
    assert 'sk-fixture0123456789abcdef' not in json.dumps(result)
