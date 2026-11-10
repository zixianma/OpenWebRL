import pytest
from openwebrl.arm_paired_inference import MODES,PROTOCOL,REVISION,schedule,summarize
from openwebrl.arm_controlled_inference import PROTOCOL as OLD_PROTOCOL


def test_original_protocol_and_seed_parity_without_extra_episodes():
    rows=schedule(['b','a'])
    assert rows==schedule(['a','b'])
    assert len(rows)==2 and sum(len(r['attempts']) for r in rows)==6
    for row in rows:
        assert {a['mode'] for a in row['attempts']}==set(MODES)
        assert {a['seed'] for a in row['attempts']}=={42}
    assert PROTOCOL['sampling_request']==dict(temperature=.7,top_p=.9,max_new_tokens=1024)
    assert PROTOCOL['candidates']==5 and PROTOCOL['max_steps']==30
    assert PROTOCOL['selector_revision']==REVISION
    assert OLD_PROTOCOL['selector_revision']!='a1f8d265855ddf1074eae91516c6e86f6f836177'
    with pytest.raises(ValueError):schedule(['a','a'])


def row(t,m,valid,reward):
    r=dict(task_id=t,mode=m,valid=valid,reward=reward)
    r.update({k:1 for k in ('actor_requests','actor_output_tokens','actor_input_tokens',
        'actor_client_request_seconds','browser_steps','browser_step_seconds','selector_calls',
        'selector_input_tokens','selector_output_tokens','selector_service_seconds','elapsed_seconds')})
    return r


def test_paired_analysis_counts_invalids_as_zero_and_keeps_common_valid_separate():
    records=[row('a','arm',True,1),row('a','actor0',True,0),row('b','arm',False,1),row('b','actor0',True,1)]
    records += [row('a','piotr',True,0),row('b','piotr',True,1)]
    result=summarize(records,['a','b'],True)
    assert result['by_mode']['arm']['overall']==.5
    assert result['by_mode']['arm']['valid_only']==1
    pair=result['paired']['arm_minus_actor0']
    assert pair['difference']==0
    assert pair['common_valid_tasks']==1
    assert pair['common_valid_first_successes']==1
    assert pair['first_only']==pair['second_only']==1
    assert set(result['paired'])=={'arm_minus_actor0','piotr_minus_actor0','arm_minus_piotr'}
    assert 'pass_at_k' not in result


def test_partial_duplicate_and_unknown_slots_cannot_look_complete():
    records=[row('a','arm',True,1)]
    assert summarize(records,['a'])['by_mode']['arm']['overall'] is None
    with pytest.raises(ValueError,match='Incomplete'):summarize(records,['a'],True)
    with pytest.raises(ValueError,match='duplicate'):summarize(records*2,['a'])
    with pytest.raises(ValueError):summarize([row('a','actor1',True,1)],['a'])


def test_full300_shards_are_disjoint_and_each_contains_all_three_arms():
    rows=schedule([str(i) for i in range(300)])
    first={r['task_id'] for r in rows if r['shard']==0};second={r['task_id'] for r in rows if r['shard']==1}
    assert len(first)==len(second)==150 and not first&second
    assert len({a['key'] for r in rows for a in r['attempts']})==900


def test_repair_phase_sensitivity_keeps_all_primary_outcomes():
    records=[row(t,m,True,1) for t in ['a','b'] for m in MODES]
    for r in records:
        if r['task_id']=='b' or r['mode']!='actor0':r['artifact_io_policy']='awaited_thread_v1'
    result=summarize(records,['a','b'],True)
    assert all(result['by_mode'][m]['completed']==2 for m in MODES)
    phases=result['execution_phases']
    assert phases['synchronous_receipts_v0']['common_phase_tasks']==0
    assert phases['awaited_thread_v1']['common_phase_tasks']==1
    assert phases['awaited_thread_v1']['by_mode']['arm']['episodes']==2
    assert phases['awaited_thread_v1']['common_phase_by_mode']['arm']['episodes']==1


def test_prepared_scope_is_not_resource_approval_and_retries_share_cap(monkeypatch):
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    import arm_inference_runtime as runtime
    from run_arm_paired_inference import RESOURCES,LIMITS
    plan=dict(resources=RESOURCES,limits=LIMITS)
    approval=dict(approved=False,plan_sha256=runtime.digest(plan),resources=RESOURCES,limits=LIMITS,
        attempts=[dict(job_id='2',time_limit_seconds=14000)])
    with pytest.raises(ValueError,match='approved'):runtime.check_approval(plan,approval,'2',0)
    approval['approved']=True
    assert runtime.check_approval(plan,approval,'2',400)==14000
    with pytest.raises(ValueError,match='cap'):runtime.check_approval(plan,approval,'2',401)
    approval['attempts'].insert(0,dict(job_id='1',time_limit_seconds=400))
    monkeypatch.setattr(runtime.subprocess,'check_output',lambda *a,**k:'1|FAILED|400|cpu=16,gres/gpu:h200=2,mem=240G|\n')
    assert runtime.scheduler_accounting(approval,'2')[0]==400
