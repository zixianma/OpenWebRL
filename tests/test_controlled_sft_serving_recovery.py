import asyncio
import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3

import pytest

from openwebrl.controlled_sft_serving_recovery import (
    DRAIN_PHASE, PURPOSE, SAMPLING, REQUEST_COUNT, immutable_json, sha,
    request_drain, assert_drained, prepare_acceptance_manifest, recovery_config,
    server_max_running_requests, reuse_jev_diagnostic, run_saved_context_acceptance,
)
from openwebrl.controlled_sft_state import SuiteState, digest


def write(path, value):
    Path(path).write_text(json.dumps(value))


@pytest.fixture
def state_root(tmp_path, monkeypatch):
    monkeypatch.setenv('SLURM_JOB_ID', '123')
    plan = dict(root=str(tmp_path), worker_config={'actor_checkpoint':'/pinned/SFT'},
        scientific_config_sha256='unchanged-science', resources={'collectors':8},
        conditions={x:{} for x in ['L01','L08','L10','L11']},
        schedule=[dict(item_id=f'i{i}',task_id=f't{i}',condition='L01',category='primary') for i in range(12)],
        task_file=str(tmp_path/'tasks.jsonl'), limits=dict(local_sft_generations=200,
            browser_sessions=20,browser_sessions_per_condition=20,concurrent_browsers=8,
            luna_selector_requests=10,jev_selector_requests=10,local_kev_requests=10,
            canonical_judge_http_attempts=10,luna_usd=1.,canonical_judge_usd=1.))
    (tmp_path/'tasks.jsonl').write_text(''.join(json.dumps({'metadata':{'task_id':f't{i}'}})+'\n' for i in range(12)))
    write(tmp_path/'plan.json',plan)
    write(tmp_path/'approval.json',dict(approved=True,plan_sha256=digest(plan),attempts=[{'job_id':'123'}]))
    write(tmp_path/'current.json',dict(job_id='123',stage='primary'))
    state = SuiteState(tmp_path,allow_unapproved_for_tests=True)
    with state.connect() as db:
        db.execute("UPDATE metadata SET value='primary' WHERE key='phase'")
    return tmp_path, plan, state


def finish(state, claim):
    result = dict(task_id=claim['task_id'],condition=claim['condition'],attempt_id=claim['attempt_id'],
                  valid=True,reward=0,browser_closed=True)
    path=Path(claim['artifact_directory'])/'result.json'
    write(path,result)
    state.finish(claim['attempt_id'],dict(result,result_path=str(path),result_sha256=sha(path)))


def test_drain_preserves_active_completion_pending_order_and_all_counters(state_root):
    root,plan,state=state_root
    claim=state.claim('collector')
    browser=state.reserve('browser_sessions',attempt_id=claim['attempt_id'],condition='L01')
    state.reserve('local_sft_generations',5,attempt_id=claim['attempt_id'])
    receipt=request_drain(root,expected_job_id='123',receipt_path=root/'drain.json')
    assert state.claim('another') is None
    with pytest.raises(ValueError,match='not finished'):assert_drained(root,receipt)
    # Already-active work may still consume and settle its existing budgets.
    state.reserve('local_sft_generations',1,attempt_id=claim['attempt_id'])
    state.settle(browser,{'closed':True});finish(state,claim)
    result=assert_drained(root,receipt)
    assert result['pending_count']==11
    assert result['counters']['local_sft_generations']==6
    assert result['counters']['browser_sessions']==1
    assert json.loads((root/'plan.json').read_text())==plan
    with state.connect() as db:
        assert db.execute('SELECT COUNT(*) FROM attempts').fetchone()[0]==1
        assert db.execute("SELECT status FROM attempts").fetchone()[0]=='finished'


def test_drain_refuses_wrong_job_or_pending_order_mutation(state_root):
    root,_,state=state_root
    with pytest.raises(ValueError,match='current approved'):
        request_drain(root,expected_job_id='999',receipt_path=root/'wrong.json')
    receipt=request_drain(root,expected_job_id='123',receipt_path=root/'drain.json')
    with state.connect() as db:db.execute("UPDATE queue SET status='finished' WHERE item_id='i0'")
    with pytest.raises(ValueError,match='Pending task identities'):assert_drained(root,receipt)


def acceptance_plan(tmp_path):
    output=tmp_path/'fixture';output.mkdir()
    image=base64.b64encode(b'saved-current-image').decode()
    requests=[]
    for i in range(10):
        prompt=f'saved-{i}<|image_pad|>'
        batch=dict(prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),prompt_tokens=17000+i,
                   screenshot_sha256=hashlib.sha256(b'saved-current-image').hexdigest(),candidate_seeds=[42+i])
        request=dict(text=prompt,image_data=['data:image/png;base64,'+image],return_logprob=True,
                     sampling_params=dict(SAMPLING,sampling_seed=42+i))
        requests.append(dict(attempt_id=str(i),prompt_tokens=17000+i,
            request=immutable_json(output/f'request-{i}.json',request),
            source_batch=immutable_json(output/f'batch-{i}.json',dict(record=batch)),
            source_rollout=immutable_json(output/f'rollout-{i}.json',{}),
            source_result=immutable_json(output/f'result-{i}.json',{})))
    plan=dict(root=str(tmp_path),scientific_config_sha256='unchanged-science',worker_config={'actor_checkpoint':'/pinned/SFT'})
    prior=immutable_json(output/'prior-plan.json',plan)
    manifest=dict(source_plan_sha256=digest(plan),root=str(tmp_path),actor_checkpoint='/pinned/SFT',
                  sampling=SAMPLING,requests=requests)
    plan['serving_recovery']=dict(previous_max_running_requests=5,max_running_requests=10,
        prior_plan_sha256=manifest['source_plan_sha256'],prior_plan=prior,
        scientific_config_sha256='unchanged-science',actor_checkpoint='/pinned/SFT',
        sampling=SAMPLING,acceptance_manifest=immutable_json(output/'manifest.json',manifest))
    return plan


class FakeResponse:
    def __init__(self, value, status=200):self.value,self.status_code=value,status
    def json(self):return self.value
    def raise_for_status(self):
        if self.status_code!=200:raise RuntimeError('HTTP error')


class LocalClient:
    def __init__(self, state, *, server_override=None, broken_index=None):
        self.state,self.override,self.broken=state,server_override or {},broken_index
        self.calls=[];self.active=0;self.peak=0
    async def get(self,url,timeout):
        assert url=='http://127.0.0.1:20000/get_server_info'
        info=dict(model_path='/pinned/SFT',max_running_requests=10,context_length=32768,
            mem_fraction_static=.60,disable_cuda_graph=True,dtype='bfloat16',tp_size=1)
        info.update(self.override)
        return FakeResponse(info)
    async def post(self,url,json,timeout):
        assert url=='http://127.0.0.1:20000/generate' and timeout==180
        assert self.state.snapshot()['counters']['local_sft_generations']==10
        self.calls.append(deepcopy(json));self.active+=1;self.peak=max(self.peak,self.active)
        await asyncio.sleep(.01);self.active-=1
        i=int(json['text'].split('<')[0].split('-')[1])
        return FakeResponse(dict(text='saved diagnostic response',meta_info=dict(
            prompt_tokens=17000+i,completion_tokens=20,finish_reason={'type':'abort' if i==self.broken else 'stop'},total_retractions=0)))


def test_ten_parallel_saved_calls_are_charged_before_dispatch_without_browser_or_api(state_root):
    root,_,state=state_root;plan=acceptance_plan(root);client=LocalClient(state)
    receipt=asyncio.run(run_saved_context_acceptance(plan,state,'http://127.0.0.1:20000',root/'journal',client))
    assert receipt['completed'] and client.peak==10 and len(client.calls)==10
    assert all(c['sampling_params']==dict(SAMPLING,sampling_seed=42+i) for i,c in enumerate(client.calls))
    assert state.snapshot()['counters']=={'local_sft_generations':10,'active_browsers':0}
    assert receipt['browser_sessions']==receipt['external_api_calls']==receipt['executed_actions']==0
    with state.connect() as db:
        row=db.execute('SELECT attempt_id,context_json,count FROM reservations').fetchone()
        assert row['attempt_id'] is None and row['count']==10
        assert json.loads(row['context_json'])['purpose']==PURPOSE
        assert db.execute('SELECT COUNT(*) FROM attempts').fetchone()[0]==0


def test_acceptance_failure_awaits_all_siblings_and_preserves_full_charge(state_root):
    root,_,state=state_root;plan=acceptance_plan(root);client=LocalClient(state,broken_index=3)
    with pytest.raises(RuntimeError,match='primary remains stopped'):
        asyncio.run(run_saved_context_acceptance(plan,state,'http://127.0.0.1:20000',root/'journal',client))
    assert len(client.calls)==10 and client.active==0
    assert state.snapshot()['counters']['local_sft_generations']==10
    assert len(list((root/'journal/serving-acceptance').glob('response-*.json')))==10


@pytest.mark.parametrize('field,value',[('max_running_requests',8),('actor_checkpoint','different'),('sampling',{'temperature':0}),('scientific_config_sha256','changed')])
def test_recovery_rejects_any_unapproved_serving_or_scientific_drift(tmp_path,field,value):
    plan=acceptance_plan(tmp_path);plan['serving_recovery'][field]=value
    with pytest.raises(ValueError):recovery_config(plan)


def test_legacy_controller_cap_remains_five_and_recovery_cap_is_ten(tmp_path):
    assert server_max_running_requests({})==5
    assert server_max_running_requests(acceptance_plan(tmp_path))==10


def test_reused_jev_diagnostic_keeps_old_proof_immutable_and_makes_no_call(tmp_path):
    plan=acceptance_plan(tmp_path)
    proof=dict(completed=True,plan_sha256=plan['serving_recovery']['prior_plan_sha256'],job_id='old-job',expected_model='jev-1.13.0',
        outcome='known_capacity_rejection',http_status=400,error_type='max_tokens_exceeded')
    prior=immutable_json(tmp_path/'old-jev.json',proof);plan['serving_recovery']['prior_jev_diagnostic']=prior
    adopted=reuse_jev_diagnostic(plan,tmp_path/'journal','new-job')
    assert sha(prior['path'])==prior['sha256']
    result=json.loads(Path(adopted['path']).read_text())
    assert result['external_calls_made']==0 and result['reused_from']==prior
    assert result['plan_sha256']==digest(plan) and result['job_id']=='new-job'


def test_reused_accepted_jev_proof_uses_actual_model_identity_schema(tmp_path):
    plan=acceptance_plan(tmp_path)
    prior=immutable_json(tmp_path/'old-jev.json',dict(completed=True,
        plan_sha256=plan['serving_recovery']['prior_plan_sha256'],job_id='old-job',
        expected_model='jev-1.13.0',outcome='accepted',model_identity='jev-1.13.0'))
    plan['serving_recovery']['prior_jev_diagnostic']=prior
    assert reuse_jev_diagnostic(plan,tmp_path/'journal','new-job')


def test_external_acceptance_url_rejected_before_any_reservation(state_root):
    root,_,state=state_root
    with pytest.raises(ValueError,match='only its local actor'):
        asyncio.run(run_saved_context_acceptance(acceptance_plan(root),state,'https://example.com',root/'journal',None))
    assert state.snapshot()['counters']=={'active_browsers':0}


def test_server_cap_mismatch_prevents_paid_reservation(state_root):
    root,_,state=state_root;client=LocalClient(state,server_override={'max_running_requests':5})
    with pytest.raises(ValueError,match='Effective local actor'):
        asyncio.run(run_saved_context_acceptance(acceptance_plan(root),state,'http://127.0.0.1:20000',root/'journal',client))
    assert not client.calls and state.snapshot()['counters']=={'active_browsers':0}


def test_acceptance_charges_cumulative_remaining_cap_without_reset(state_root):
    root,_,state=state_root;state.reserve('local_sft_generations',195,purpose='all prior attempts')
    client=LocalClient(state)
    with pytest.raises(RuntimeError,match='All-attempt call cap'):
        asyncio.run(run_saved_context_acceptance(acceptance_plan(root),state,'http://127.0.0.1:20000',root/'journal',client))
    assert not client.calls and state.snapshot()['counters']['local_sft_generations']==195


def test_saved_request_tampering_fails_before_any_reservation(state_root):
    root,_,state=state_root;plan=acceptance_plan(root);client=LocalClient(state)
    (root/'fixture/request-0.json').write_text('{"changed":true}')
    with pytest.raises(ValueError,match='artifact changed'):
        asyncio.run(run_saved_context_acceptance(plan,state,'http://127.0.0.1:20000',root/'journal',client))
    assert not client.calls and state.snapshot()['counters']=={'active_browsers':0}


@pytest.mark.parametrize('field,value',[('resources',{'total_seconds':99999}),('limits',{'local_sft_generations':99999}),('schedule',[]),('worker_config',{'actor_checkpoint':'other'}),('shard',{'index':1})])
def test_lineage_rejects_resource_budget_checkpoint_or_cohort_changes(tmp_path,field,value):
    plan=acceptance_plan(tmp_path);plan[field]=value
    with pytest.raises(ValueError):recovery_config(plan)


def test_total_request_deadline_cancels_stalled_siblings_and_keeps_charge(state_root,monkeypatch):
    import openwebrl.controlled_sft_serving_recovery as recovery
    root,_,state=state_root;plan=acceptance_plan(root)
    class NeverCompletes(LocalClient):
        async def post(self,*args,**kwargs):
            self.active+=1
            try:await asyncio.Event().wait()
            finally:self.active-=1
    client=NeverCompletes(state);monkeypatch.setattr(recovery,'INFERENCE_TIMEOUT',.02)
    with pytest.raises(RuntimeError,match='primary remains stopped'):
        asyncio.run(run_saved_context_acceptance(plan,state,'http://127.0.0.1:20000',root/'journal',client))
    assert client.active==0 and state.snapshot()['counters']['local_sft_generations']==10
    assert len(list((root/'journal/serving-acceptance').glob('error-*.json')))==10


def test_drain_proves_originally_active_result_still_exists_and_matches_hash(state_root):
    root,_,state=state_root;claim=state.claim('collector')
    receipt=request_drain(root,expected_job_id='123',receipt_path=root/'drain.json')
    finish(state,claim)
    (Path(claim['artifact_directory'])/'result.json').write_text('{"changed":true}')
    with pytest.raises(ValueError,match='result or its browser-closure proof changed'):
        assert_drained(root,receipt)
