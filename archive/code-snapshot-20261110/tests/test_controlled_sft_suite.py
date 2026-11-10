import concurrent.futures
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from openwebrl.controlled_sft_state import SuiteState,digest,validate_shard_cohort
from prepare_controlled_sft_suite import schedule
from run_controlled_sft_suite import (
    check_approval,validate_allocation,process_group_members,validate_topology,
    validate_device_assignments,worker_actor_urls,smoke_performance_receipt,kev_environment,initialize_wandb,
)


SHARD_RESOURCES = dict(gpus=2,gpu_type='H200',cpus=16,memory_gib=240,total_seconds=28800,
                       sft_replicas=1,kev_replicas=1,collectors=8)


def test_kev_cache_overrides_all_inherited_hf_aliases():
    inherited={key:'/unrelated/cache' for key in
               ('HF_HOME','HF_HUB_CACHE','HUGGINGFACE_HUB_CACHE','TRANSFORMERS_CACHE')}
    inherited.update(kev_environment('GPU-kev'))
    expected='/gpfs/scrubbed/zixianma/openwebrl-runtime/hf-cache'
    assert inherited['HF_HOME']==expected
    assert all(inherited[key]==expected+'/hub' for key in
               ('HF_HUB_CACHE','HUGGINGFACE_HUB_CACHE','TRANSFORMERS_CACHE'))
    assert inherited['CUDA_VISIBLE_DEVICES']=='GPU-kev'
    assert inherited['HF_HUB_OFFLINE']==inherited['TRANSFORMERS_OFFLINE']=='1'
    assert inherited['KEV_TRUNCATE_STATES']==inherited['KEV_DATE_FACTS']=='0'


def test_wandb_controller_uses_loaded_credentials_without_logging_them(monkeypatch,tmp_path):
    monkeypatch.delenv('WANDB_API_KEY',raising=False)
    monkeypatch.setitem(sys.modules,'wandb',SimpleNamespace(Settings=lambda **kw:kw,init=lambda **kw:kw))
    plan=dict(experiment_id='fixed-shard',protocol_name='canonical',resources={'gpus':2},limits={'luna_usd':25})
    call=initialize_wandb(plan,tmp_path,{'WANDB_API_KEY':'fixture-credential'})
    assert call['settings']['api_key']=='fixture-credential'
    assert call['project']=='openwebrl-evals' and call['id']=='fixed-shard' and call['resume']=='allow'
    assert 'fixture-credential' not in json.dumps(call['config'])
    assert os.environ.get('WANDB_API_KEY') is None


def shard_plan(tmp_path, index=0):
    ids = [f'task-{i:03d}' for i in range(300)]
    partitions = [ids[:150], ids[150:]]
    task_file = tmp_path/'tasks.jsonl'
    task_file.write_text(''.join(json.dumps({'metadata':{'task_id':i,'intent':'fixture','start_url':'about:blank'}})+'\n' for i in ids))
    return dict(root=str(tmp_path),task_file=str(task_file), campaign_id='fixture-campaign', experiment_id=f'fixture-campaign-shard-{index}',
        campaign_task_shards=partitions, shard=dict(index=index,count=2,task_ids=partitions[index]),
        conditions={x:{} for x in ['L01','L08','L10','L11']}, schedule=schedule(partitions[index]),
        primary_episodes=600,smoke_episodes=12,resources=dict(SHARD_RESOURCES),
        limits=dict(local_sft_generations=158400,browser_sessions=660,browser_sessions_per_condition=165,
                    concurrent_browsers=8,luna_selector_requests=9905,jev_selector_requests=9905,
                    local_kev_requests=9905,luna_usd=25.,canonical_judge_http_attempts=2640,canonical_judge_usd=12.5))


@pytest.fixture
def root(tmp_path):
    ids=['task-a','task-b']
    task_file=tmp_path/'tasks.jsonl'
    task_file.write_text(''.join(json.dumps({'metadata':{'task_id':i,'intent':'fixture','start_url':'about:blank'}})+'\n' for i in ids))
    plan={'root':str(tmp_path),'task_file':str(task_file),'schedule':schedule(ids),'conditions':{x:{} for x in ['L01','L08','L10','L11']},
          'resources':{'collectors':7,'total_seconds':57600,'gpus':8},
          'limits':{'local_sft_generations':100,'browser_sessions':20,'browser_sessions_per_condition':6,
                    'concurrent_browsers':7,'luna_selector_requests':10,'jev_selector_requests':10,
                    'local_kev_requests':10,'canonical_judge_http_attempts':10,'luna_usd':1.,'canonical_judge_usd':1.}}
    (tmp_path/'plan.json').write_text(json.dumps(plan))
    return tmp_path


def state(root):return SuiteState(root,allow_unapproved_for_tests=True)


def finish(s,claim,valid=True):
    result={'task_id':claim['task_id'],'condition':claim['condition'],'attempt_id':claim['attempt_id'],
            'valid':valid,'reward':0,'browser_closed':True}
    p=Path(claim['artifact_directory'])/'result.json';p.write_text(json.dumps(result))
    events=p.parent/'events';events.mkdir(exist_ok=True)
    (events/'0001.json').write_text(json.dumps({'event':'proposal_received','record':{}}))
    models={'L08':'gpt-6-luna','L10':'jev-1.13.0','L11':'kev-latest'}
    if claim['condition'] in models:
        (events/'0002.json').write_text(json.dumps({'event':'selector_received','record':{'response':{'model':models[claim['condition']]}}}))
    (events/'0003.json').write_text(json.dumps({'event':'judge_received','record':{'response':{'model':'o4-mini-2025-04-16'}}}))
    import hashlib
    s.finish(claim['attempt_id'],dict(result,result_path=str(p),result_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))


def test_schedule_contains_all_primary_and_separate_smokes():
    rows=schedule([f't{i}' for i in range(300)])
    assert len(rows)==1212 and len({x['item_id'] for x in rows})==1212
    assert len([x for x in rows if x['category']=='smoke'])==12
    assert rows==schedule([f't{i}' for i in range(300)])


def test_unapproved_cannot_claim_or_reserve(root):
    s=SuiteState(root)
    with pytest.raises((ValueError,FileNotFoundError)):s.claim('w')
    with pytest.raises((ValueError,FileNotFoundError)):s.reserve('local_sft_generations')
    assert s.snapshot()['counters'].get('local_sft_generations',0)==0


def test_copying_plan_cannot_reset_ledger_caps(root):
    other=root/'another-ledger';other.mkdir()
    (other/'plan.json').write_bytes((root/'plan.json').read_bytes())
    with pytest.raises(ValueError,match='durable ledger root'):state(other)


def test_smoke_barrier_and_fresh_primary_ids(root):
    s=state(root);claims=[]
    while c:=s.claim('w'):
        assert c['category']=='smoke';claims.append(c);finish(s,c)
    assert len(claims)==8
    s.enable_primary()
    c=s.claim('w');assert c['category']=='primary'
    assert c['attempt_id'] not in {x['attempt_id'] for x in claims}


def test_failed_smoke_does_not_enable_primary(root):
    s=state(root)
    while c:=s.claim('w'):finish(s,c,False)
    with pytest.raises(RuntimeError):s.enable_primary()


def test_known_jev_capacity_rejection_does_not_block_valid_smokes(root,monkeypatch):
    p=json.loads((root/'plan.json').read_text());p['requires_jev_context_diagnostic']=True
    (root/'plan.json').write_text(json.dumps(p));monkeypatch.setenv('SLURM_JOB_ID','123')
    s=state(root)
    while c:=s.claim('w'):finish(s,c)
    with pytest.raises(FileNotFoundError):s.enable_primary()
    receipt=root/'scheduler-attempts/123/jev-context-preflight.json';receipt.parent.mkdir(parents=True)
    receipt.write_text(json.dumps(dict(completed=True,plan_sha256=digest(p),job_id='123',
        expected_model='jev-1.13.0',state_bytes=262144,outcome='known_capacity_rejection',
        http_status=400,error_type='max_tokens_exceeded')))
    s.enable_primary();assert s.snapshot()['phase']=='primary'


def test_smoke_without_any_actual_judge_response_blocks_primary(root):
    s=state(root)
    while c:=s.claim('w'):
        finish(s,c)
        p=Path(c['artifact_directory'])/'events/0003.json'
        p.write_text(json.dumps({'event':'canonical_reward_applied','record':{'score':0}}))
    with pytest.raises(RuntimeError,match='canonical_judge'):s.enable_primary()


def test_failed_judge_attempt_retains_full_charge_with_evidence(root):
    s=state(root);c=s.claim('w')
    r=s.reserve('om2w_judge_http_attempts',request={'max_completion_tokens':4096},attempt_id=c['attempt_id'])
    error=Path(c['artifact_directory'])/'error.json';error.write_text('{"error":"timeout"}')
    before=s.snapshot()['counters']['canonical_judge_usd']
    s.finalize_unknown(r,error)
    assert s.snapshot()['counters']['canonical_judge_usd']==before
    assert s.snapshot()['unsettled_reservations']==0


def test_parallel_claims_unique_and_capped_at_seven(root):
    s=state(root)
    def claim(i):
        try:return state(root).claim(str(i))
        except RuntimeError:return None
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        rows=list(pool.map(claim,range(8)))
    active=[r for r in rows if r]
    assert len(active)==7 and len({r['attempt_id'] for r in active})==7
    assert s.snapshot()['queue']['active']==7


def test_browser_cap_and_verified_closure(root):
    s=state(root);c=s.claim('w')
    r=s.reserve('browser_sessions',attempt_id=c['attempt_id'],condition=c['condition'])
    with pytest.raises(ValueError):s.settle(r,{'closed':False})
    with pytest.raises(RuntimeError):finish(s,c)
    s.settle(r,{'closed':True})
    # Existing failed fixture result is retained; update its hash-bound descriptor.
    p=Path(c['artifact_directory'])/'result.json';d=json.loads(p.read_text())
    import hashlib
    s.finish(c['attempt_id'],dict(d,result_path=str(p),result_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    assert s.snapshot()['counters']['active_browsers']==0


def test_count_only_reservation_consumed_without_completion_deadlock(root):
    s=state(root);c=s.claim('w')
    s.reserve('local_sft_generations',5,attempt_id=c['attempt_id'],condition=c['condition'])
    finish(s,c)
    assert s.snapshot()['unsettled_reservations']==0
    assert state(root).snapshot()['counters']['local_sft_generations']==5


def test_priced_reservation_settles_and_unknown_usage_halts(root):
    s=state(root);c=s.claim('w')
    r=s.reserve('luna_selector_requests',request={'model':'gpt-6-luna','input':'x','max_output_tokens':4096},attempt_id=c['attempt_id'])
    before=s.snapshot()['counters']['luna_usd'];assert before>0
    s.settle(r,{'input_tokens':10,'output_tokens':10})
    assert s.snapshot()['counters']['luna_usd']<before
    r=s.reserve('om2w_judge_http_attempts',request={'model':'o4-mini-2025-04-16','messages':[],'max_completion_tokens':4096},attempt_id=c['attempt_id'])
    reserved=s.snapshot()['counters']['canonical_judge_usd']
    with pytest.raises(RuntimeError):s.settle(r,{})
    assert s.snapshot()['counters']['canonical_judge_usd']==reserved
    with pytest.raises(RuntimeError):s.reserve('local_sft_generations')


def test_halt_keeps_first_cause(root):
    s=state(root);s.halt('first');s.halt('second')
    assert s.snapshot()['halt']['reason']=='first'


def test_interrupted_attempts_preserved_without_automatic_replay(root):
    s=state(root);c=s.claim('w');s.reserve('browser_sessions',attempt_id=c['attempt_id'],condition=c['condition'])
    with pytest.raises(ValueError):s.mark_interrupted_after_teardown(teardown_verified=False,reason='test')
    s.mark_interrupted_after_teardown(teardown_verified=True,reason='verified fixture teardown')
    snap=s.snapshot();assert snap['queue']['interrupted']==1 and snap['counters']['active_browsers']==0
    assert snap['counters']['browser_sessions']==1


def test_luna_long_context_uses_conservative_multiplier(root):
    s=state(root)
    r=s.reserve('luna_selector_requests',request={'input':'x'*280000,'max_output_tokens':4096})
    expected=(r['input_token_bound']*.25+4096*.75)/1e6
    assert r['reserved_usd']==pytest.approx(expected)


def test_scheduler_budget_charges_previous_attempts(root):
    p=json.loads((root/'plan.json').read_text())
    approval={'approved':True,'plan_sha256':digest(p),'resources':p['resources'],'limits':p['limits'],
              'attempts':[{'job_id':'123','time_limit_seconds':30000}]}
    assert check_approval(p,approval,'123',20000)==30000
    with pytest.raises(ValueError):check_approval(p,approval,'123',30000)
    with pytest.raises(ValueError):check_approval(p,dict(approval,approved=False),'123',0)


def test_exact_slurm_resources_and_time():
    info='JobId=123 JobState=RUNNING UserId=u(42) NumNodes=1 NumCPUs=64 AllocTRES=cpu=64,mem=960G,gres/gpu=8,gres/gpu:h200=8 TimeLimit=16:00:00 EndTime=2030-01-01T16:00:00'
    assert validate_allocation(info,'123',57600,now=0,uid=42)==57420
    with pytest.raises(ValueError):validate_allocation(info.replace('NumCPUs=64','NumCPUs=32'),'123',57600,now=0,uid=42)
    with pytest.raises(ValueError):validate_allocation(info,'123',50000,now=0,uid=42)


def test_owned_descendants_detected_when_group_leader_has_exited(tmp_path):
    child=tmp_path/'1235';child.mkdir()
    (child/'stat').write_text('1235 (chromium worker) S 1 1234 0 0')
    (child/'cgroup').write_text('0::/slurm/job_456/step_batch')
    assert process_group_members(1234,'456',tmp_path)==[1235]
    with pytest.raises(ValueError):process_group_members(1234,'789',tmp_path)
    (child/'stat').write_text('1235 (chromium worker) Z 1 1234 0 0')
    assert process_group_members(1234,'456',tmp_path)==[]


def test_attempt_records_scheduler_lineage(root,monkeypatch):
    monkeypatch.setenv('SLURM_JOB_ID','12345')
    s=state(root);claim=s.claim('worker')
    with s.connect() as db:
        assert db.execute('SELECT job_id FROM attempts WHERE attempt_id=?',(claim['attempt_id'],)).fetchone()['job_id']=='12345'


def test_owned_model_descendants_stopped_after_leader_exit(monkeypatch):
    import run_controlled_sft_suite as controller
    class ExitedLeader:
        pid=1200
        def poll(self):return 1
        def wait(self,timeout):return 1
    live=[1201];signals=[]
    monkeypatch.setattr(controller,'process_group_members',lambda pgid,job:list(live))
    def terminate(pgid,signal):signals.append((pgid,signal));live.clear()
    monkeypatch.setattr(controller.os,'killpg',terminate)
    records=controller.stop_owned([ExitedLeader()],'456',grace=0)
    assert signals and signals[0][0]==1200
    assert records[0]['original_live_group_members']==[1201]
    assert records[0]['process_gone']


def browser_cleanup_fixture(tmp_path, *, pid=1200, ticks=100):
    registry=tmp_path/'suite/attempts/fixture/browser/server-process.json'
    registry.parent.mkdir(parents=True)
    registry.write_text(json.dumps(dict(job_id='456',pid=pid,pgid=pid,proc_start_ticks=ticks)))
    proc=tmp_path/'proc';proc.mkdir()
    def process(number,group,start,job='456'):
        directory=proc/str(number);directory.mkdir()
        parts=['S','1',str(group)]+['0']*16+[str(start)]
        (directory/'stat').write_text(f'{number} (fixture worker) '+ ' '.join(parts))
        (directory/'cgroup').write_text(f'0::/slurm/job_{job}/step_batch')
        return directory
    return tmp_path/'suite',proc,process


def test_browser_cleanup_recycled_nonleader_pid_with_empty_old_group_is_not_signaled(tmp_path,monkeypatch):
    import run_controlled_sft_suite as controller
    root,proc,process=browser_cleanup_fixture(tmp_path)
    process(1200,9900,200,'789')
    signals=[];monkeypatch.setattr(controller.os,'killpg',lambda *args:signals.append(args))
    rows=controller.cleanup_browser_groups(root,'456',proc)
    assert signals==[] and rows[0]['process_gone'] and rows[0]['identity_changed']
    assert rows[0]['observed_identity']['pgid']==9900
    assert rows[0]['remaining_live_group_members']==[]


def test_browser_cleanup_recycled_group_id_still_refuses_unrelated_signal(tmp_path,monkeypatch):
    import run_controlled_sft_suite as controller
    root,proc,process=browser_cleanup_fixture(tmp_path)
    process(1200,1200,200,'789')
    signals=[];monkeypatch.setattr(controller.os,'killpg',lambda *args:signals.append(args))
    with pytest.raises(ValueError,match='different-allocation'):
        controller.cleanup_browser_groups(root,'456',proc)
    assert signals==[]


def test_browser_cleanup_preserves_owned_descendant_cleanup_when_leader_exited(tmp_path,monkeypatch):
    import run_controlled_sft_suite as controller
    root,proc,process=browser_cleanup_fixture(tmp_path)
    child=process(1201,1200,200)
    signals=[]
    def terminate(group,signum):
        signals.append((group,signum))
        text=(child/'stat').read_text();(child/'stat').write_text(text.replace(') S ',') Z '))
    monkeypatch.setattr(controller.os,'killpg',terminate)
    rows=controller.cleanup_browser_groups(root,'456',proc)
    assert signals==[(1200,controller.signal.SIGTERM)]
    assert rows[0]['original_live_group_members']==[1201] and rows[0]['process_gone']


def test_two_shards_are_disjoint_and_cover_all_four_conditions(tmp_path):
    plans=[]
    for index in range(2):
        directory=tmp_path/str(index);directory.mkdir()
        plan=shard_plan(directory,index);plans.append(plan)
        validate_shard_cohort(plan,[f'task-{i:03d}' for i in range(300)])
        assert len(plan['schedule'])==612
    left={(r['task_id'],r['condition']) for r in plans[0]['schedule'] if r['category']=='primary'}
    right={(r['task_id'],r['condition']) for r in plans[1]['schedule'] if r['category']=='primary'}
    assert not left&right and len(left|right)==1200


@pytest.mark.parametrize('mutation',['overlap','missing_condition','outside_task','duplicate_primary','wrong_shard','missing_smoke_arm'])
def test_malformed_shard_cohort_is_rejected_before_queue_creation(tmp_path,mutation):
    plan=shard_plan(tmp_path)
    primary=[r for r in plan['schedule'] if r['category']=='primary']
    smoke=[r for r in plan['schedule'] if r['category']=='smoke']
    if mutation=='overlap':plan['campaign_task_shards'][1][0]=plan['campaign_task_shards'][0][0]
    elif mutation=='missing_condition':primary[0]['condition']='L99'
    elif mutation=='outside_task':primary[0]['task_id']=plan['campaign_task_shards'][1][0]
    elif mutation=='duplicate_primary':
        primary[0]['task_id']=primary[1]['task_id'];primary[0]['condition']=primary[1]['condition']
    elif mutation=='wrong_shard':plan['shard']['index']=1
    elif mutation=='missing_smoke_arm':smoke[0]['condition']=smoke[1]['condition']
    (tmp_path/'plan.json').write_text(json.dumps(plan))
    with pytest.raises(ValueError):state(tmp_path)
    assert not (tmp_path/'state.sqlite3').exists()


@pytest.mark.parametrize('changed',[dict(gpus=1),dict(cpus=32),dict(memory_gib=480),dict(total_seconds=57600),
                                    dict(sft_replicas=2),dict(kev_replicas=0),dict(collectors=7)])
def test_two_gpu_topology_cannot_silently_drift(changed):
    with pytest.raises(ValueError):validate_topology(dict(SHARD_RESOURCES,**changed),sharded=True)


def test_eight_collectors_share_one_verified_actor_and_kev_gets_separate_gpu():
    url='http://127.0.0.1:21000'
    assert worker_actor_urls(SHARD_RESOURCES,[url])==[url]*8
    assert validate_device_assignments(SHARD_RESOURCES,['GPU-actor','GPU-kev'])==(['GPU-actor'],'GPU-kev')
    with pytest.raises(ValueError):worker_actor_urls(SHARD_RESOURCES,[url,'http://127.0.0.1:21016'])
    with pytest.raises(ValueError):validate_device_assignments(SHARD_RESOURCES,['GPU-actor','GPU-actor'])
    with pytest.raises(ValueError):validate_device_assignments(SHARD_RESOURCES,['GPU-actor'])


def test_exact_two_gpu_scheduler_resources_and_cumulative_budget(tmp_path):
    info='JobId=123 JobState=RUNNING UserId=u(42) NumNodes=1 NumCPUs=16 AllocTRES=cpu=16,mem=240G,gres/gpu=2,gres/gpu:h200=2 TimeLimit=08:00:00 EndTime=2030-01-01T08:00:00'
    assert validate_allocation(info,'123',28800,now=0,uid=42,resources=SHARD_RESOURCES)==28620
    for bad in (info.replace('NumCPUs=16','NumCPUs=32'),info.replace('mem=240G','mem=960G'),
                info.replace('gres/gpu=2','gres/gpu=8'),info.replace('TimeLimit=08','TimeLimit=16')):
        with pytest.raises(ValueError):validate_allocation(bad,'123',28800,now=0,uid=42,resources=SHARD_RESOURCES)
    plan=shard_plan(tmp_path)
    approval=dict(approved=True,plan_sha256=digest(plan),resources=plan['resources'],limits=plan['limits'],
                  attempts=[dict(job_id='123',time_limit_seconds=900)])
    assert check_approval(plan,approval,'123',27900)==900
    with pytest.raises(ValueError):check_approval(plan,approval,'123',27901)


def test_shards_keep_independent_nonresetting_caps(tmp_path):
    roots=[]
    for index in range(2):
        directory=tmp_path/str(index);directory.mkdir();roots.append(directory)
        (directory/'plan.json').write_text(json.dumps(shard_plan(directory,index)))
    first,second=state(roots[0]),state(roots[1])
    first.reserve('jev_selector_requests',9905,purpose='offline-counter-test')
    with pytest.raises(RuntimeError):state(roots[0]).reserve('jev_selector_requests',purpose='offline-counter-test')
    second.reserve('jev_selector_requests',purpose='offline-counter-test')
    assert first.snapshot()['counters']['jev_selector_requests']==9905
    assert second.snapshot()['counters']['jev_selector_requests']==1


def test_smoke_performance_reports_observed_evidence_without_claiming_long_context(root):
    s=state(root)
    while c:=s.claim('w'):
        finish(s,c)
        p=Path(c['artifact_directory'])/'events/0010.json'
        p.write_text(json.dumps({'event':'http_received','record':{'provider':'sft','elapsed_seconds':3.5,
            'response':{'meta_info':{'prompt_tokens':4000,'completion_tokens':128}}}}))
    journal=root/'scheduler';journal.mkdir()
    (journal/'sft-0-identity.json').write_text(json.dumps({'model_path':'/fixture/sft'}))
    (journal/'gpu.log').write_text('fixture GPU telemetry; no GPU accessed\n')
    plan=dict(s.plan,smoke_episodes=8,resources=dict(SHARD_RESOURCES),worker_config={'actor_checkpoint':'/fixture/sft'})
    result=smoke_performance_receipt(plan,s,journal)
    assert result['collectors']==8 and result['sft_replicas']==1
    assert result['actor_http_responses']==8 and result['median_actor_http_seconds']==3.5
    assert result['max_observed_prompt_tokens']==4000 and not result['long_context_observed']
    assert result['maximum_concurrent_long_context_acceptance_verified'] is False
