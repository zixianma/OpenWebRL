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


def diagnosed_invalid(tmp_path):
    task=dict(task_id='timeout',intent='Preserved task',start_url='https://example.invalid/')
    directory=tmp_path/'actor/tasks'/io.digest(task['task_id'])
    result=dict(task_id=task['task_id'],completed=True,valid=False,score=None,provider_blocked=True,
                actor_error='EpisodeTimeout',judge_error='missing_browser_evidence',
                cleanup_errors=['PlaywrightDispatcherUnavailable'])
    io.write_json(directory/'result.json',result)
    io.write_json(directory/'task.json',task)
    io.write_json(directory/'trajectory.json',dict(task=task,error='EpisodeTimeout',history=[],decisions=[]))
    io.write_json(directory/'browser-session.json',dict(id='owned-session',stopped=True))
    (directory/'worker.log').write_text('Preserved timeout evidence\n')
    diagnosis=dict(task_id=task['task_id'],reason='Owner inspected dispatcher timeout and verified provider stop.',
        result_sha256=run.kev.file_hash(directory/'result.json'),expected_actor_error=result['actor_error'],
        expected_judge_error=result['judge_error'],evidence_sha256={
            str((directory/name).relative_to(tmp_path)):run.kev.file_hash(directory/name)
            for name in ('result.json','trajectory.json','worker.log')})
    io.write_json(tmp_path/'diagnosed-invalids.json',{task['task_id']:diagnosis})
    return task,directory,result,diagnosis


def test_explicit_hash_bound_invalid_is_skipped_without_retry(tmp_path):
    from scripts.kev_actor_final_audit import invalid_diagnosis
    task,directory,result,diagnosis=diagnosed_invalid(tmp_path)
    # The controller consumes the same owner-authored contract as the auditor.
    invalid_diagnosis(tmp_path,directory,result,diagnosis)
    missing=dict(task_id='untouched')
    assert run.pending_tasks(tmp_path,dict(tasks=[task,missing]))==[missing]


@pytest.mark.parametrize('field,value',[
    ('reason',''),('reason','   '),('result_sha256','wrong'),('task_id','another'),
    ('expected_actor_error',None),('expected_judge_error','unparseable_verdict'),
    ('evidence_sha256',{}),
])
def test_invalid_diagnosis_must_match_exact_record(tmp_path,field,value):
    task,_,_,diagnosis=diagnosed_invalid(tmp_path)
    diagnosis[field]=value
    io.write_json(tmp_path/'diagnosed-invalids.json',{task['task_id']:diagnosis})
    with pytest.raises(ValueError,match='diagnosis'):
        run.pending_tasks(tmp_path,dict(tasks=[task]))


def test_missing_invalid_diagnosis_never_skips_or_retries(tmp_path):
    task,*_=diagnosed_invalid(tmp_path)
    io.write_json(tmp_path/'diagnosed-invalids.json',{})
    with pytest.raises(ValueError,match='diagnosis'):
        run.pending_tasks(tmp_path,dict(tasks=[task]))


def test_changed_invalid_evidence_and_live_browser_block_resume(tmp_path):
    task,directory,_,_=diagnosed_invalid(tmp_path)
    io.write_json(directory/'browser-session.json',dict(id='owned-session',stopped=False))
    with pytest.raises(ValueError,match='stopped owned browser'):
        run.pending_tasks(tmp_path,dict(tasks=[task]))
    io.write_json(directory/'browser-session.json',dict(id='owned-session',stopped=True))
    (directory/'worker.log').write_text('Changed log')
    with pytest.raises(ValueError,match='evidence changed'):
        run.pending_tasks(tmp_path,dict(tasks=[task]))


def test_invalid_diagnosis_cannot_reference_other_task_evidence(tmp_path):
    task,_,_,diagnosis=diagnosed_invalid(tmp_path)
    other=tmp_path/'other-task-proof.json';other.write_text('{}')
    diagnosis['evidence_sha256'][str(other.relative_to(tmp_path))]=run.kev.file_hash(other)
    io.write_json(tmp_path/'diagnosed-invalids.json',{task['task_id']:diagnosis})
    with pytest.raises(ValueError,match='escapes its task directory'):
        run.pending_tasks(tmp_path,dict(tasks=[task]))


def test_invalid_diagnosis_cannot_accept_a_canonical_score(tmp_path):
    task,directory,result,diagnosis=diagnosed_invalid(tmp_path)
    result['score']=0
    io.write_json(directory/'result.json',result)
    diagnosis['result_sha256']=run.kev.file_hash(directory/'result.json')
    diagnosis['evidence_sha256'][str((directory/'result.json').relative_to(tmp_path))]=diagnosis['result_sha256']
    io.write_json(tmp_path/'diagnosed-invalids.json',{task['task_id']:diagnosis})
    with pytest.raises(ValueError,match='diagnosis'):
        run.pending_tasks(tmp_path,dict(tasks=[task]))
