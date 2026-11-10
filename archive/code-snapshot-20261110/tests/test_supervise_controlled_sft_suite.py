import json
from pathlib import Path
import sqlite3
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import supervise_controlled_sft_suite as supervisor


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


@pytest.fixture
def campaign(tmp_path):
    pointer=tmp_path/'pointer.json';root=tmp_path/'campaign';root.mkdir()
    shards=[]
    for index in (0,1):
        directory=root/f'shard-{index}';directory.mkdir()
        plan=dict(root=str(directory),resources={'total_seconds':28800},limits={'browser_sessions':660})
        job=str(123+index)
        write(directory/'plan.json',plan)
        write(directory/'approval.json',dict(approved=True,plan_sha256=supervisor.digest(plan),
            resources=plan['resources'],limits=plan['limits'],attempts=[{'job_id':job,'time_limit_seconds':28800}]))
        write(directory/'current.json',dict(job_id=job,stage='smoke'))
        shards.append(dict(shard=index,root=str(directory),job_id=job))
    write(pointer,dict(root=str(root),shards=shards,campaign_id='fixture'))
    return root,pointer


def fake_runner(calls,state='RUNNING',exit_code=0):
    def run(command,**kwargs):
        calls.append(command)
        if command[0]=='/usr/bin/sacct':
            jobs=command[command.index('-j')+1].split(',')
            return SimpleNamespace(returncode=0,stdout=''.join(f'{job}|{state}|60|0:0|node|\n' for job in jobs),stderr='')
        return SimpleNamespace(returncode=exit_code,stdout='queued-fixture',stderr='queue unavailable' if exit_code else '')
    return run


def test_initial_poll_queues_and_preserves_snapshot(campaign):
    root,pointer=campaign;calls=[]
    assert supervisor.poll(root,pointer,'root-thread',run=fake_runner(calls),now=1000)
    assert len(calls)==2 and calls[-1][1:4]==['queue','--thread','root-thread']
    assert len(supervisor.read(root/'supervisor-latest.json')['shards'])==2
    assert supervisor.read(root/'supervisor-state.json')['queue_receipt']=='queued-fixture'


def test_no_duplicate_while_callback_pending(campaign):
    root,pointer=campaign;calls=[];run=fake_runner(calls)
    supervisor.poll(root,pointer,'thread',run=run,now=1000)
    supervisor.poll(root,pointer,'thread',run=run,now=2000)
    assert sum(c[0]!=' /usr/bin/sacct'.strip() for c in calls)==1


def test_failure_bypasses_pending_routine_once(campaign):
    root,pointer=campaign;calls=[]
    supervisor.poll(root,pointer,'thread',run=fake_runner(calls),now=1000)
    supervisor.poll(root,pointer,'thread',run=fake_runner(calls,'FAILED'),now=1060)
    supervisor.poll(root,pointer,'thread',run=fake_runner(calls,'FAILED'),now=1120)
    assert sum(c[0]!='/usr/bin/sacct' for c in calls)==2
    assert supervisor.read(root/'supervisor-state.json')['pending_callback_bypassed']


def test_ack_then_fifteen_minute_callback(campaign):
    root,pointer=campaign;calls=[];run=fake_runner(calls)
    supervisor.poll(root,pointer,'thread',run=run,now=1000)
    write(root/'supervisor-ack.json',{'epoch':1020})
    supervisor.poll(root,pointer,'thread',run=run,now=1919)
    assert sum(c[0]!='/usr/bin/sacct' for c in calls)==1
    supervisor.poll(root,pointer,'thread',run=run,now=1920)
    assert sum(c[0]!='/usr/bin/sacct' for c in calls)==2


def test_verified_complete_stops_silently_without_scheduler(campaign):
    root,pointer=campaign;value=supervisor.read(pointer);value['verified_complete']=True;write(pointer,value)
    assert supervisor.poll(root,pointer,'thread',run=lambda *a,**k:pytest.fail('must not call tools'),now=1000) is False
    assert (root/'supervisor-finished.json').exists() and (root/'supervisor-ack.json').exists()


def test_approval_mismatch_prevents_dispatch(campaign):
    root,pointer=campaign;path=root/'shard-0/approval.json';value=supervisor.read(path);value['approved']=False;write(path,value)
    with pytest.raises(ValueError,match='approval'):supervisor.poll(root,pointer,'thread',run=lambda *a,**k:pytest.fail('no query'))


def test_queue_failure_is_not_recorded_as_success(campaign):
    root,pointer=campaign;calls=[]
    with pytest.raises(RuntimeError,match='queue failed'):
        supervisor.poll(root,pointer,'thread',run=fake_runner(calls,exit_code=1),now=1000)
    assert not (root/'supervisor-state.json').exists()


def test_replacement_during_poll_suppresses_old_failure(campaign):
    root,pointer=campaign;calls=[];base=fake_runner(calls,'FAILED')
    def run(command,**kwargs):
        result=base(command,**kwargs)
        value=supervisor.read(pointer);value['shards'][0]['job_id']='999';write(pointer,value)
        return result
    supervisor.poll(root,pointer,'thread',run=run,now=1000)
    assert len(calls)==1 and not (root/'supervisor-state.json').exists()


def test_database_read_only_missing_does_not_create(tmp_path):
    assert supervisor.database_snapshot(tmp_path)=={'present':False}
    assert not (tmp_path/'state.sqlite3').exists()


def test_scheduler_charges_all_attempts(campaign):
    root,pointer=campaign;path=root/'shard-0/approval.json';value=supervisor.read(path)
    value['attempts'].insert(0,{'job_id':'122','time_limit_seconds':100});write(path,value)
    supervisor.poll(root,pointer,'thread',run=fake_runner([]),now=1000)
    assert supervisor.read(root/'supervisor-latest.json')['shards'][0]['all_attempt_scheduler_seconds']==120


def test_prompt_fits_deployed_queue_limit():
    root=Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations/local-sft-selector-controlled-20261006/prepared-parallel-v3')
    assert len(supervisor.continuation_prompt(root,supervisor.DEFAULT_POINTER).encode())<=1000
