import asyncio
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from openwebrl.arm_controlled_inference import MODES, MeteredSelector, schedule, summarize, verify_record
from openwebrl.arm_terminal_budget import CappedJudge


def test_schedule_is_frozen_independent_of_input_order():
    rows=schedule(['a','b','c'])
    assert rows==schedule(['c','a','b'])
    for r in rows:
        assert {a['mode'] for a in r['attempts']}==set(MODES)
        assert len({a['seed'] for a in r['attempts']})==6
    with pytest.raises(ValueError):schedule(['a','a'])


def test_paired_passk_keeps_invalids_and_rejects_missing_slots():
    records=[]
    for task in ['a','b']:
        for mode in MODES:
            r=dict(task_id=task,mode=mode,valid=not(task=='a' and mode=='actor4'),
                   reward=int((task=='a' and mode=='arm') or (task=='b' and mode=='actor0')))
            for k in ('actor_requests','actor_output_tokens','actor_input_tokens','actor_client_request_seconds',
                      'browser_steps','browser_step_seconds','selector_calls','selector_input_tokens',
                      'selector_output_tokens','selector_service_seconds','elapsed_seconds'):r[k]=1
            records.append(r)
    result=summarize(records,['a','b'],True)
    assert result['pass_at_k'][0]['pass_at_k']==pytest.approx(.1)
    assert result['pass_at_k'][-1]['pass_at_k']==.5
    assert result['paired_arm_actor5']['arm_only']==1
    assert result['paired_arm_actor5']['actor5_only']==1
    assert result['paired_arm_actor5']['all_six_valid_tasks']==1
    with pytest.raises(ValueError,match='Incomplete'):summarize(records[:-1],['a','b'],True)


def test_resume_requires_unchanged_durable_result(tmp_path):
    f=tmp_path/'result.json';f.write_text('{}')
    item={'task_id':'a'};attempt={'mode':'arm','seed':1,'key':'key'}
    record=dict(task_id='a',mode='arm',seed=1,key='key',protocol_sha256='pin',valid=False,reward=None,
                artifact=str(f),artifact_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),trace_files=[])
    verify_record(record,item,attempt,tmp_path,'pin')
    f.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='hash mismatch'):verify_record(record,item,attempt,tmp_path,'pin')


def test_rate_override_preserves_native_budget_and_cancelled_reservations(tmp_path):
    native=CappedJudge(tmp_path/'native',client=object(),max_calls=2,max_usd=1.)
    ident=native.reserve(.1);native.settle(ident,.1,dict(prompt_tokens=1000,completion_tokens=100))
    assert json.loads((tmp_path/'native/ledger.json').read_text())['charged_or_reserved_usd']==pytest.approx(.0028)
    class OtherRate(CappedJudge):
        INPUT_USD_PER_MILLION=1.1
        OUTPUT_USD_PER_MILLION=4.4
    other=OtherRate(tmp_path/'other',client=object(),max_calls=2,max_usd=1.)
    ident=other.reserve(.1);other.settle(ident,.1,dict(prompt_tokens=1000,completion_tokens=100))
    other.reserve(.1)  # Interrupted requests keep their reservation.
    assert json.loads((tmp_path/'other/ledger.json').read_text())['charged_or_reserved_usd']==pytest.approx(.10154)
    with pytest.raises(ValueError,match='budget cap'):other.reserve(.1)


def test_actor_meter_counts_backend_tokens_before_delimiter_repair(tmp_path,monkeypatch):
    async def post(url,payload):
        return dict(text='reply',meta_info=dict(prompt_tokens=20,completion_tokens=2,
            output_token_logprobs=[[-.1,1],[-.2,2]],finish_reason={'type':'stop'}))
    def repair(text,tokens,logps):tokens.append(99);logps.append(0);return text
    fake=SimpleNamespace(post=post,_ensure_im_end_w_new_line=repair)
    import openwebrl
    monkeypatch.setattr(openwebrl,'generate_browser',fake,raising=False)
    class Selector:
        async def __call__(self,*,infer,**kwargs):
            return await infer('url','text',{'sampling_seed':42},[],timeout_secs=1)
    meter=MeteredSelector(Selector(),tmp_path)
    result=asyncio.run(meter(infer=None))
    assert len(result[1])==3 and meter.output_tokens==2 and meter.input_tokens==20
    assert json.loads((tmp_path/'actor-request-00000.json').read_text())['status']=='received'


def test_allocation_retries_cannot_reset_eight_hour_cap():
    scripts=Path(__file__).resolve().parents[1]/'scripts';sys.path.insert(0,str(scripts))
    spec=importlib.util.spec_from_file_location('controlled_controller',scripts/'run_arm_controlled_inference.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    approval=dict(approved=True,resources=module.RESOURCES,judge_max_usd=50.,judge_max_calls=7200,
                  attempts=[dict(job_id='123',shard=0,time_limit_seconds=3600)])
    module.verify_approval(approval,'123',0,7*3600)
    with pytest.raises(ValueError,match='eight-hour'):module.verify_approval(approval,'123',0,7*3600+1)
    approval['approved']=False
    with pytest.raises(ValueError,match='approval'):module.verify_approval(approval,'123',0,0)


def test_actor_receipts_do_not_block_peers_and_survive_cancellation(tmp_path,monkeypatch):
    import threading
    import openwebrl
    from openwebrl import arm_controlled_inference as controlled
    started=threading.Event();release=threading.Event();finished=threading.Event()
    original=controlled.write_json
    def slow_write(path,receipt):
        started.set()
        assert release.wait(3), 'Event loop blocked by receipt persistence'
        original(path,receipt);finished.set()
    async def post(url,payload):
        return dict(text='reply',meta_info=dict(prompt_tokens=20,completion_tokens=1,
            output_token_logprobs=[[-.1,1]],finish_reason={'type':'stop'}))
    monkeypatch.setattr(openwebrl,'generate_browser',SimpleNamespace(post=post,
        _ensure_im_end_w_new_line=lambda text,*unused:text),raising=False)
    monkeypatch.setattr(controlled,'write_json',slow_write)
    class Selector:
        async def __call__(self,*,infer,**kwargs):
            return await infer('url','text',{'sampling_seed':42},[],timeout_secs=1)
    async def exercise():
        task=asyncio.create_task(MeteredSelector(Selector(),tmp_path)(infer=None))
        try:
            for _ in range(200):
                if started.is_set():break
                await asyncio.sleep(.005)
            assert started.is_set()
            assert not finished.is_set()
            task.cancel()
            await asyncio.sleep(.03)
            assert not task.done(), 'Cancellation must await durable receipt'
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):await task
        assert finished.is_set()
    asyncio.run(exercise())
    receipt=json.loads((tmp_path/'actor-request-00000.json').read_text())
    assert receipt['status']=='received' and receipt['sampling_params']=={'sampling_seed':42}


def test_threaded_judge_keeps_attempt_identity_and_shared_cap(tmp_path,monkeypatch):
    import openai
    from openwebrl.arm_controlled_inference import ATTEMPT,make_judge
    calls=[]
    async def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(model='o4-mini-2025-04-16',
            usage=SimpleNamespace(model_dump=lambda:dict(prompt_tokens=100,completion_tokens=10)),
            choices=[SimpleNamespace(message=SimpleNamespace(content='success'),finish_reason='stop')])
    monkeypatch.setenv('JUDGE_API_KEY','test-only')
    monkeypatch.setattr(openai,'AsyncOpenAI',lambda **unused:SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    judge=make_judge(tmp_path,max_calls=2,max_usd=1.)
    async def one(key):
        token=ATTEMPT.set(key)
        try:return await judge.create(model='o4-mini',messages=[{'content':'test'}])
        finally:ATTEMPT.reset(token)
    async def exercise():
        await asyncio.gather(one('a'),one('b'))
        with pytest.raises(ValueError,match='budget cap'):await one('c')
    asyncio.run(exercise())
    assert len(calls)==2 and all(c['max_completion_tokens']==4096 for c in calls)
    assert {json.loads(p.read_text())['attempt'] for p in tmp_path.glob('*-usage.json')}=={'a','b'}
    ledger=json.loads((tmp_path/'ledger.json').read_text())
    assert ledger['calls']==2 and ledger['charged_or_reserved_usd']==pytest.approx(.000308)
