"""Offline continuation checks for the standalone Kev full300 supervisor."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def setup(tmp_path, monkeypatch):
    spec=importlib.util.spec_from_file_location('kev_actor_supervisor',
        Path(__file__).resolve().parents[1]/'scripts/supervise_kev_actor_full300.py')
    supervisor=importlib.util.module_from_spec(spec);spec.loader.exec_module(supervisor)
    root=tmp_path/'run';root.mkdir();pointer=tmp_path/'pointer.json'
    supervisor.write(root/'approval.json',{'approved':True})
    supervisor.write(root/'plan.json',{})
    supervisor.write(pointer,{'job_id':'12345','verified_complete':False})
    supervisor.write(root/'heartbeat.json',{'stage':'startup','updated_unix':1000})
    now=[1000.];state=['RUNNING'];calls=[]
    monkeypatch.setattr(supervisor.time,'time',lambda:now[0])
    monkeypatch.setattr(supervisor.subprocess,'check_output',
        lambda *a,**k:f'12345|{state[0]}|100|0:0\n')
    def queue(command,**kwargs):
        calls.append(command)
        return SimpleNamespace(stdout=f'queued-{len(calls)}',returncode=0)
    monkeypatch.setattr(supervisor.subprocess,'run',queue)
    return supervisor,root,pointer,now,state,calls


def test_prompt_fits_production_path_and_suppresses_stale_completion():
    spec=importlib.util.spec_from_file_location('kev_actor_supervisor_prompt',
        Path(__file__).resolve().parents[1]/'scripts/supervise_kev_actor_full300.py')
    supervisor=importlib.util.module_from_spec(spec);spec.loader.exec_module(supervisor)
    runtime=Path('/gpfs/scrubbed/zixianma/openwebrl-runtime')
    pointer=runtime/'current_kev27b_actor_full300.json'
    root=runtime/'evaluations/kev27b-actor-full300-20261004'
    prompt=supervisor.continuation_prompt(root,pointer,'345123')
    assert len(prompt.encode())<=1000
    assert prompt.index(str(pointer))<prompt.index(str(root))
    assert 'silently ack' in prompt and 'no user-facing repeat' in prompt
    assert 'first verified completion' in prompt
    assert 'No added budget.' in prompt


@pytest.mark.parametrize('event',[
    'FAILED','TIMEOUT','CANCELLED by 123','OUT_OF_MEMORY','NODE_FAIL',
    'BOOT_FAIL','PREEMPTED','DEADLINE','COMPLETED','halted','stale',
    'invalid','collection_complete',
])
def test_new_urgent_event_bypasses_pending_routine_once(setup,event):
    supervisor,root,pointer,now,state,calls=setup
    assert supervisor.poll(root,pointer,'thread') and len(calls)==1
    assert supervisor.read(root/'supervisor-state.json')['notification_kind']=='routine'
    now[0]=1001.
    if event=='halted':
        supervisor.write(root/'failure.json',{'job_id':'12345','reason':'provider failure'})
    elif event=='stale':
        now[0]=1300.
    elif event=='invalid':
        task=root/'actor/tasks/failed';task.mkdir(parents=True)
        supervisor.write(task/'result.json',{'valid':False})
    elif event=='collection_complete':
        (root/'actor').mkdir()
        supervisor.write(root/'actor/summary.json',{'collection_complete':True})
    else:
        state[0]=event
    assert supervisor.poll(root,pointer,'thread') and len(calls)==2
    receipt=supervisor.read(root/'supervisor-state.json')
    assert receipt['notification_kind']=='urgent' and receipt['pending_callback_bypassed']
    assert supervisor.poll(root,pointer,'thread') and len(calls)==2
    assert not supervisor.read(root/'supervisor-latest.json')['verified_complete']


def test_ordinary_stage_change_waits_for_ack(setup):
    supervisor,root,pointer,now,state,calls=setup
    supervisor.poll(root,pointer,'thread')
    supervisor.write(root/'heartbeat.json',{'stage':'actor','updated_unix':1000})
    supervisor.poll(root,pointer,'thread')
    assert len(calls)==1
    supervisor.write(root/'supervisor-ack.json',{'epoch':1000})
    supervisor.poll(root,pointer,'thread')
    assert len(calls)==2
    assert not supervisor.read(root/'supervisor-state.json')['pending_callback_bypassed']


def test_old_attempt_failure_does_not_trigger_current_attempt(setup):
    supervisor,root,pointer,now,state,calls=setup
    supervisor.poll(root,pointer,'thread')
    supervisor.write(root/'failure.json',{'job_id':'12344','reason':'old failure'})
    supervisor.poll(root,pointer,'thread')
    assert len(calls)==1
    assert not supervisor.read(root/'supervisor-latest.json')['selector_halted']


def test_verified_completion_exits_before_scheduler_or_new_callback(setup,monkeypatch):
    supervisor,root,pointer,now,state,calls=setup
    supervisor.poll(root,pointer,'thread')
    supervisor.write(pointer,{'job_id':'12345','verified_complete':True})
    monkeypatch.setattr(supervisor.subprocess,'check_output',
        lambda *a,**k:pytest.fail('Completed pointer must avoid scheduler polling'))
    assert supervisor.poll(root,pointer,'thread') is False
    assert len(calls)==1
    assert supervisor.read(root/'supervisor-finished.json')['reason']=='verified completion'


def test_failed_urgent_dispatch_remains_retryable(setup,monkeypatch):
    supervisor,root,pointer,now,state,calls=setup
    supervisor.poll(root,pointer,'thread')
    prior=supervisor.read(root/'supervisor-state.json')
    state[0]='FAILED';now[0]=1001.
    original=supervisor.subprocess.run
    monkeypatch.setattr(supervisor.subprocess,'run',lambda *a,**k:
        SimpleNamespace(stdout='',stderr='unavailable',returncode=1))
    with pytest.raises(RuntimeError,match='Continuation queue failed'):
        supervisor.poll(root,pointer,'thread')
    assert supervisor.read(root/'supervisor-state.json')==prior
    monkeypatch.setattr(supervisor.subprocess,'run',original)
    assert supervisor.poll(root,pointer,'thread') and len(calls)==2
