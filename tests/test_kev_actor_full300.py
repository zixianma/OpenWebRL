"""Offline safety checks for the paid standalone Kev full300 controller."""
import json
import sys
from pathlib import Path
import httpx
import pytest
from openwebrl import jev_eval as io
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import evaluate_kev_actor_full300 as run


def cfg(tmp_path, cap=1):
    return dict(decision_provider='kev',decision_endpoint='http://127.0.0.1:18763/v1/systemone',
        jev_model='kev-latest',text_provider='openai',max_decisions=60,
        budget=dict(root=str(tmp_path/'budget'),limits=dict(local_kev_requests=cap)))


def test_separate_workers_and_restarts_share_hard_request_cap(tmp_path):
    requests=[]
    def respond(request):
        requests.append(request);return httpx.Response(200,json=dict(model='kev-latest'))
    config=cfg(tmp_path)
    for directory in ('a','b'):(tmp_path/directory).mkdir()
    a=io.ModelTransport(tmp_path/'a',config,httpx.Client(transport=httpx.MockTransport(respond)))
    b=io.ModelTransport(tmp_path/'b',config,httpx.Client(transport=httpx.MockTransport(respond)))
    a.post('https://api.typesafe.ai/v1/systemone','secret',dict(model='kev-latest'))
    with pytest.raises(io.ProviderError,match='budget.*exhausted'):
        b.post('https://api.typesafe.ai/v1/systemone','secret',dict(model='kev-latest'))
    assert len(requests)==1
    assert run.read(tmp_path/'budget/usage.json')['reserved']['local_kev_requests']==1


def test_transport_failure_remains_charged(tmp_path):
    def respond(request):raise httpx.ReadTimeout('ambiguous')
    directory=tmp_path/'task';directory.mkdir()
    transport=io.ModelTransport(directory,cfg(tmp_path),httpx.Client(transport=httpx.MockTransport(respond)))
    with pytest.raises(io.ProviderError,match='transport'):
        transport.post('https://api.typesafe.ai/v1/systemone','key',dict(model='kev-latest'))
    assert run.read(tmp_path/'budget/usage.json')['reserved']['local_kev_requests']==1


def test_retry_reserves_each_http_attempt(tmp_path,monkeypatch):
    requests=[]
    def respond(request):
        requests.append(request);return httpx.Response(503)
    directory=tmp_path/'task';directory.mkdir()
    transport=io.ModelTransport(directory,cfg(tmp_path,2),httpx.Client(transport=httpx.MockTransport(respond)))
    monkeypatch.setattr(io.time,'sleep',lambda _:None)
    with pytest.raises(io.ProviderError,match='budget.*exhausted'):
        transport.post('https://api.typesafe.ai/v1/systemone','key',dict(model='kev-latest'))
    assert len(requests)==2
    assert len((directory/'api-attempts.jsonl').read_text().splitlines())==2


def test_unapproved_submission_cannot_reach_scheduler(tmp_path,monkeypatch):
    plan=dict(resources=run.RESOURCES,limits=run.LIMITS)
    io.write_json(tmp_path/'plan.json',plan)
    io.write_json(tmp_path/'approval.json',dict(approved=False,resources=run.RESOURCES,limits=run.LIMITS))
    monkeypatch.setattr(run,'verify',lambda _:plan)
    monkeypatch.setattr(run.subprocess,'check_output',lambda *a,**k:pytest.fail('Must not call scheduler'))
    with pytest.raises(ValueError,match='Exact resource/API approval'):
        run.submit(tmp_path)


def test_all_failed_scheduler_time_reduces_replacement_limit(tmp_path,monkeypatch):
    value=dict(attempts=[dict(job_id='100'),dict(job_id='101')])
    states={'100':dict(job_id='100',state='FAILED',scheduler_seconds=1003),
            '101':dict(job_id='101',state='TIMEOUT',scheduler_seconds=2000)}
    monkeypatch.setattr(run,'scheduler',lambda job:states[job])
    used,rows=run.previous_usage(value)
    assert used==3003 and (run.RESOURCES['total_seconds']-used)//60==189
    states['101']['state']='RUNNING'
    with pytest.raises(ValueError,match='still active'):run.previous_usage(value)


def test_valid_failed_task_is_never_rerolled(tmp_path):
    tasks=[dict(task_id='failed'),dict(task_id='missing')]
    io.write_json(tmp_path/'actor/tasks'/io.digest('failed')/'result.json',
                  dict(task_id='failed',valid=True,score=0,provider_blocked=False))
    assert run.pending_tasks(tmp_path,dict(tasks=tasks))==[tasks[1]]


@pytest.mark.parametrize('name,value',[('task.json',dict(task_id='x')),
    ('result.json',dict(task_id='x',valid=False,provider_blocked=True))])
def test_interrupted_or_invalid_attempt_requires_preserved_diagnosis(tmp_path,name,value):
    io.write_json(tmp_path/'actor/tasks'/io.digest('x')/name,value)
    with pytest.raises(ValueError,match='diagnosis'):
        run.pending_tasks(tmp_path,dict(tasks=[dict(task_id='x')]))


def test_browser_reservation_cannot_be_reset_with_new_caps(tmp_path):
    config=dict(budget=dict(root=str(tmp_path/'budget'),limits=dict(browser_sessions=1)))
    io.reserve_budget(config,'browser_sessions',task_id='x')
    with pytest.raises(io.ProviderError):io.reserve_budget(config,'browser_sessions',task_id='y')
    config['budget']['limits']['browser_sessions']=2
    with pytest.raises(ValueError,match='limits differ'):io.reserve_budget(config,'browser_sessions')
