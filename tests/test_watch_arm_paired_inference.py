import copy
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import watch_arm_paired_inference as watcher


def fixtures():
    plan=dict(resources=watcher.RESOURCES,limits=watcher.LIMITS)
    approval=dict(approved=True,plan_sha256=watcher.digest(plan),resources=watcher.RESOURCES,
        limits=watcher.LIMITS,attempts=[dict(job_id='123',time_limit_seconds=14400)])
    return plan,approval


def test_approval_refuses_cross_study_or_expanded_budgets():
    plan,a=fixtures();assert watcher.validate_approval(plan,a)==['123']
    for key,new in [('resources',dict(watcher.RESOURCES,total_seconds=28800)),
                    ('limits',dict(watcher.LIMITS,judge_usd=20.))]:
        bad=copy.deepcopy(plan);bad[key]=new
        b=copy.deepcopy(a);b[key]=new;b['plan_sha256']=watcher.digest(bad)
        with pytest.raises(ValueError):watcher.validate_approval(bad,b)


def test_retry_charges_prior_failed_time(tmp_path,monkeypatch):
    plan,a=fixtures();a['attempts'].insert(0,dict(job_id='122',time_limit_seconds=14400))
    watcher.write(tmp_path/'plan.json',plan);watcher.write(tmp_path/'approval.json',a)
    watcher.write(tmp_path/'current.json',dict(job_id='123',stage='submitted'))
    monkeypatch.setattr(watcher,'scheduler_records',lambda jobs:{
        '122':dict(state='FAILED',elapsed_seconds=600),
        '123':dict(state='PENDING',elapsed_seconds=0)})
    with pytest.raises(ValueError,match='Retry reservation'):watcher.shard_snapshot(tmp_path,1)
    a['attempts'][1]['time_limit_seconds']=13800;watcher.write(tmp_path/'approval.json',a)
    assert watcher.shard_snapshot(tmp_path,1)['allocation_seconds']==600


def test_error_dispatch_is_coalesced_and_does_not_claim_completion(tmp_path,monkeypatch):
    monkeypatch.setattr(watcher,'shard_snapshot',lambda root,now:(_ for _ in ()).throw(ValueError('bad cap')))
    calls=[]
    class Result:
        returncode=0;stdout='queued';stderr=''
    def run(command,**kwargs):calls.append(command);return Result()
    monkeypatch.setattr(watcher.subprocess,'run',run)
    assert watcher.poll(tmp_path,'thread','codex')
    assert watcher.poll(tmp_path,'thread','codex')
    assert len(calls)==1
    assert len(calls[0][-1].encode())<=1000
    assert not watcher.read(tmp_path/'supervisor-latest.json')['verified_complete']


def test_corrected_cohort_excludes_old_records_but_charges_shared_judge(tmp_path,monkeypatch):
    plan,a=fixtures();plan['collection_directory']='collection-policy-restored-v1'
    a['plan_sha256']=watcher.digest(plan)
    watcher.write(tmp_path/'plan.json',plan);watcher.write(tmp_path/'approval.json',a)
    watcher.write(tmp_path/'current.json',dict(job_id='123',stage='collecting'))
    watcher.write(tmp_path/'collection/records/old.json',dict(success=True))
    watcher.write(tmp_path/'collection-policy-restored-v1/records/new.json',dict(success=False))
    watcher.write(tmp_path/'collection/judge-budget/ledger.json',dict(calls=17,charged_or_reserved_usd=1.2))
    monkeypatch.setattr(watcher,'scheduler_records',lambda jobs:{'123':dict(state='RUNNING',elapsed_seconds=100)})
    result=watcher.shard_snapshot(tmp_path,1)
    assert result['records']==1 and result['judge']['calls']==17
    assert result['collection_directory'].endswith('collection-policy-restored-v1')
