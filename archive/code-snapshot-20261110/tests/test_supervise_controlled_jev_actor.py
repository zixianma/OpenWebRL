import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import supervise_controlled_jev_actor as supervisor
from test_supervise_controlled_sft_suite import fake_runner,write


@pytest.fixture
def cohort(tmp_path):
    root=tmp_path/'jev';root.mkdir();pointer=tmp_path/'pointer.json'
    resources=dict(gpus=0,cpus=8,memory_gib=32,total_seconds=14400,collectors=4)
    limits=dict(browser_sessions=330,concurrent_browsers=4,jev_requests=19800,
        text_http_attempts=19800,text_usd=15,om2w_judge_http_attempts=1320,judge_usd=8)
    plan=dict(root=str(root),resources=resources,limits=limits)
    write(root/'plan.json',plan)
    write(root/'approval.json',dict(approved=True,plan_sha256=supervisor.digest(plan),resources=resources,
        limits=limits,attempts=[dict(job_id='123',time_limit_seconds=14400)]))
    write(pointer,dict(root=str(root),job_id='123',shards=[dict(shard=0,root=str(root),job_id='123')]))
    write(root/'current.json',dict(job_id='123',stage='smoke'))
    return root,pointer


def test_single_jev_cohort_and_correct_callback(cohort):
    root,pointer=cohort;calls=[]
    assert supervisor.poll(root,pointer,'root-thread',run=fake_runner(calls),now=1000)
    message=calls[-1][-1]
    assert 'Jev actor' in message and '300 primary+3smokes' in message and '1200' not in message
    assert '0GPU/8CPU/32GiB/14400s' in message
    snapshot=supervisor.read(root/'supervisor-latest.json')
    assert snapshot['primary_episodes']==300 and len(snapshot['shards'])==1


def test_completion_is_silent(cohort):
    root,pointer=cohort;value=supervisor.read(pointer);value['verified_complete']=True;write(pointer,value)
    assert not supervisor.poll(root,pointer,'thread',run=lambda *a,**kw:pytest.fail('No repeat callback'))
    assert (root/'supervisor-ack.json').exists()


def test_failure_overrides_pending_routine_then_deduplicates(cohort):
    root,pointer=cohort;calls=[]
    supervisor.poll(root,pointer,'thread',run=fake_runner(calls),now=1000)
    supervisor.poll(root,pointer,'thread',run=fake_runner(calls,'FAILED'),now=1060)
    supervisor.poll(root,pointer,'thread',run=fake_runner(calls,'FAILED'),now=1120)
    assert len([c for c in calls if c[0]!='/usr/bin/sacct'])==2


def test_wrong_resource_approval_rejected(cohort):
    root,pointer=cohort;approval=supervisor.read(root/'approval.json');approval['resources']['gpus']=2
    write(root/'approval.json',approval)
    with pytest.raises(ValueError,match='approval'):supervisor.poll(root,pointer,'thread')


def test_multiple_shards_rejected(cohort):
    root,pointer=cohort;value=supervisor.read(pointer);value['shards'].append(dict(value['shards'][0]));write(pointer,value)
    with pytest.raises(ValueError,match='one independent'):supervisor.poll(root,pointer,'thread')


def test_replacement_during_read_suppresses_stale_failure(cohort):
    root,pointer=cohort;calls=[];base=fake_runner(calls,'FAILED')
    def run(command,**kwargs):
        result=base(command,**kwargs);value=supervisor.read(pointer)
        value['job_id']='124';value['shards'][0]['job_id']='124';write(pointer,value)
        return result
    supervisor.poll(root,pointer,'thread',run=run,now=1000)
    assert len(calls)==1


def test_all_prior_elapsed_time_charged(cohort):
    root,pointer=cohort;approval=supervisor.read(root/'approval.json')
    approval['attempts'].insert(0,dict(job_id='122',time_limit_seconds=60));write(root/'approval.json',approval)
    supervisor.poll(root,pointer,'thread',run=fake_runner([]),now=1000)
    row=supervisor.read(root/'supervisor-latest.json')['shards'][0]
    assert row['all_attempt_scheduler_seconds']==120 and row['remaining_scheduler_seconds']==14280


def test_live_prompt_fits_queue_limit():
    root=Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations/jev-actor-local-full300-20261006/prepared-v2')
    assert len(supervisor.continuation_prompt(root,supervisor.POINTER).encode())<=1000
