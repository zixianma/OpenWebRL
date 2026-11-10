import asyncio
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from openwebrl.luna_qwen_metrics import (MODES, PRICING, api_cost, episode_usage,
    schedule, summarize, tokens, write_json)
from openwebrl.luna_qwen_policy import Budget, MeteredAPI, Policy, actor_tools, response_messages


def test_reasoning_and_cache_are_subsets_and_cache_writes_are_priced():
    usage = dict(input_tokens=1000, output_tokens=200,
                 input_tokens_details=dict(cached_tokens=400, cache_write_tokens=100),
                 output_tokens_details=dict(reasoning_tokens=150))
    t = tokens(usage, 'luna')
    assert (t['input'], t['output'], t['reasoning_output']) == (1000,200,150)
    # 500 ordinary +400 cached +100 writes, and all200 output tokens once.
    expected = (500*.1 +400*.01 +100*.125 +200*.5)/1e6
    assert api_cost(usage, 'selector')['upper_usd'] == pytest.approx(expected)
    assert api_cost(usage, 'selector')['lower_usd'] == pytest.approx(expected)
    usage['input_tokens_details'].pop('cache_write_tokens')
    bounds = api_cost(usage, 'actor')
    assert bounds['upper_usd'] > bounds['lower_usd']
    usage['output_tokens_details']['reasoning_tokens'] = 201
    with pytest.raises(ValueError):
        tokens(usage, 'luna')


def test_all_candidates_and_failed_responses_count_judge_is_separate(tmp_path):
    for i in range(10):
        write_json(tmp_path/f'requests/qwen-{i}.json', dict(provider='qwen', role='actor',
            usage=dict(prompt_tokens=100, completion_tokens=20, cached_tokens=90), seconds=1.,
            flops=dict(lower=1000, upper=2000), error_type='discarded' if i else None))
    write_json(tmp_path/'requests/luna.json', dict(provider='luna', role='selector',
        usage=dict(input_tokens=500, output_tokens=100, input_tokens_details=dict(cached_tokens=0,cache_write_tokens=0),
                   output_tokens_details=dict(reasoning_tokens=80))))
    write_json(tmp_path/'requests/judge.json', dict(provider='api', role='judge',
        usage=dict(prompt_tokens=9999, completion_tokens=1000)))
    result = episode_usage(tmp_path, 'qwen_luna10', 10.)
    assert result['model_input_tokens'] == 1500
    assert result['model_output_tokens'] == 300
    assert result['qwen_flops_lower'] == 10000
    assert result['local_gpu_reserved_seconds'] == 10
    assert result['by_role']['luna_selector']['reasoning_output'] == 80
    assert episode_usage(tmp_path, 'luna', 10.)['local_gpu_reserved_seconds'] == 0
    write_json(tmp_path/'requests/interrupted.json', dict(provider='luna', role='selector', status='reserved'))
    missing = episode_usage(tmp_path, 'qwen_luna10', 10.)
    assert missing['model_cost_upper_usd'] is None
    assert missing['model_usage_missing_requests'] == 1


def test_budget_reservations_survive_restart_and_failure(tmp_path):
    b = Budget(tmp_path, max_usd=1., max_calls=3)
    b.update(reserve=.6)
    restored = Budget(tmp_path, max_usd=1., max_calls=3)
    with pytest.raises(RuntimeError):
        restored.update(reserve=.5)
    assert json.loads((tmp_path/'ledger.json').read_text())['charged_or_reserved_usd'] == .6


def test_missing_compute_does_not_erase_received_tokens(tmp_path):
    write_json(tmp_path/'requests/qwen-failed.json', dict(provider='qwen', role='actor',
        usage=dict(prompt_tokens=50, completion_tokens=0), seconds=3.))
    usage = episode_usage(tmp_path, 'qwen', 5.)
    assert usage['model_input_tokens'] == 50
    assert usage['qwen_compute_missing_requests'] == 1
    assert usage['qwen_flops_upper'] is None
    assert usage['qwen_request_seconds_sum_overlapping'] == 3.


def test_judge_evidence_excludes_actor_thoughts_without_mutation():
    from openwebrl.luna_qwen_eval import action_only_messages
    messages = [dict(role='user',content='goal'),
                dict(role='assistant',content='<think>private plan</think><tool_call>click</tool_call>')]
    stripped = action_only_messages(messages)
    assert 'private plan' not in str(stripped)
    assert 'click' in stripped[-1]['content']
    assert 'private plan' in messages[-1]['content']


def test_schedule_pairs_every_task_and_order_is_input_independent():
    assert schedule(['a','b']) == schedule(['b','a'])
    assert all(set(row['modes']) == set(MODES) for row in schedule(['a','b']))
    with pytest.raises(ValueError):
        schedule(['a','a'])


def records():
    result = []
    for t in ('a','b'):
        for mode in MODES:
            result.append(dict(task_id=t, mode=mode, valid=t=='a', reward=1 if t=='a' else None,
                episode_seconds=10 if t=='a' else 30, browser_steps=2, browser_step_seconds=1.,
                model_input_tokens=100, model_output_tokens=20, local_gpu_reserved_seconds=10 if mode!='luna' else 0,
                qwen_flops_lower=1000, qwen_flops_upper=2000, model_cost_lower_usd=.01,
                model_cost_upper_usd=.011, model_usage_missing_requests=0, by_role={}))
    return result


def test_primary_denominator_keeps_invalids_and_latency_is_not_success_only():
    result = summarize(records(), ['a','b'])
    assert result['by_mode']['qwen']['overall'] == .5
    assert result['by_mode']['qwen']['metrics']['episode_seconds']['p50'] == 20
    assert result['paired_differences']['luna_minus_qwen']['task_bootstrap_95'] == [0,0]
    with pytest.raises(ValueError, match='Incomplete'):
        summarize(records()[:-1], ['a','b'])


def test_luna_messages_preserve_goal_image_history_and_browser_action_space():
    messages = [{'role':'system','content':'Task policy'}, {'role':'user','content':[
        {'type':'text','text':'goal: find item\nscreenshot:\n<|vision_start|><|image_pad|><|vision_end|>'},
        {'type':'image_url','image_url':{'url':'data:image/png;base64,IMAGE'}}]},
        {'role':'assistant','content':'previous action'}]
    converted = response_messages(messages)
    assert 'find item' in converted[1]['content'][0]['text']
    assert '<|image_pad|>' not in str(converted)
    assert converted[1]['content'][1]['type'] == 'input_image'
    assert converted[2]['content'] == 'previous action'
    tools = actor_tools([dict(type='function',function=dict(name='click',parameters={'type':'object'}))])
    assert tools[0]['name'] == 'click' and not tools[0]['strict']


def test_selector_can_select_tenth_candidate_without_rewriting_or_fallback(tmp_path):
    async def run():
        api = SimpleNamespace(create=AsyncMock(return_value=SimpleNamespace(output_text='{"selection":10}')))
        def builder(task, history, url, clusters, screenshot):
            assert len(clusters)==10
            return [dict(role='system',content=task),dict(role='user',content=[dict(type='text',text=url)])]
        policy = Policy('qwen_luna10',tmp_path,api,builder,{}, {})
        policy.set_context([{'role':'user','content':'goal'}], [{'name':'click'}],None,32000)
        counter = 0
        async def qwen(*args):
            nonlocal counter
            i=counter;counter+=1
            assert args[2]['max_new_tokens']==767
            return (f'thought{i}</think>ACTION{i}', [i],[-.1], 'stop')
        policy.qwen=qwen
        output,meta=await policy(infer=None,url='local',input_text='prompt',
            sampling_params=dict(temperature=1.,top_p=.9,max_new_tokens=4096),images=[],
            observation=dict(screenshot=b'image',active_tab_url='url'), history=[],task='goal',task_id='a',turn=0,timeout=10)
        saved=json.loads((tmp_path/'decisions/000.json').read_text())
        expected=saved['display_order'][9]
        assert output[0] == f'thought{expected}</think>ACTION{expected}'
        assert meta['selected_index']==expected
        kwargs=api.create.call_args.kwargs
        assert 'temperature' not in kwargs and 'top_p' not in kwargs
        assert kwargs['reasoning']=={'effort':'medium'}
        api.create.return_value.output_text='{"selection":11}'
        with pytest.raises(ValueError):
            await policy(infer=None,url='local',input_text='prompt',
                sampling_params=dict(max_new_tokens=4096),images=[],observation=dict(screenshot=b'image'),
                history=[],task='goal',task_id='a',turn=1,timeout=10)
    asyncio.run(run())


def test_api_timeout_preserves_unsettled_charge_and_request(tmp_path):
    async def run():
        client=SimpleNamespace(responses=SimpleNamespace(create=AsyncMock(side_effect=TimeoutError)))
        api=MeteredAPI(client,Budget(tmp_path/'budget',1.,10),tmp_path/'episode')
        with pytest.raises(TimeoutError):
            await api.create('selector',model='gpt-6-luna',input=[dict(role='user',content='goal')],max_output_tokens=4096)
        row=json.loads(next((tmp_path/'episode/requests').glob('*.json')).read_text())
        assert row['status']=='usage_unknown'
        assert json.loads((tmp_path/'budget/ledger.json').read_text())['charged_or_reserved_usd']>0
    asyncio.run(run())


def test_luna_actor_executes_one_native_tool_and_never_samples_qwen(tmp_path):
    async def run():
        call = dict(type='function_call', call_id='call_test', name='click',arguments='{"point_2d":[200,300]}')
        api = SimpleNamespace(create=AsyncMock(return_value=SimpleNamespace(output=[call])))
        policy = Policy('luna',tmp_path,api,None,{}, {})
        tokenizer = SimpleNamespace(encode=lambda text, **kwargs: list(range(len(text))))
        policy.set_context([dict(role='user',content='goal')], [dict(name='click')],tokenizer,100)
        policy.qwen = AsyncMock(side_effect=AssertionError('Luna arm must not invoke Qwen'))
        output,meta = await policy(infer=None,url='local',input_text='prompt',sampling_params={},images=[],
            observation=dict(screenshot=b'image'),history=[],task='goal',task_id='a',turn=0,timeout=10)
        assert '"point_2d": [200, 300]' in output[0]
        assert output[0].startswith('</think>') and meta['mode']=='luna'
        assert api.create.call_args.kwargs['tool_choice']=='required'
        api.create.return_value.output=[call,call]
        policy.set_context([dict(role='user',content='goal'), dict(role='assistant', content=output[0]),
            dict(role='user', content='<tool_response>\nClick succeeded. Current page.\n</tool_response>')],
            [dict(name='click')], tokenizer, 200)
        with pytest.raises(ValueError, match='exactly one'):
            await policy(infer=None,url='local',input_text='prompt',sampling_params={},images=[],
                observation=dict(screenshot=b'image'),history=[],task='goal',task_id='a',turn=1,timeout=10)
    asyncio.run(run())


def test_plotter_writes_three_real_metric_figures(tmp_path):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from report_luna_qwen_inference import plot
    plot(summarize(records(), ['a','b']),tmp_path)
    assert {p.name for p in tmp_path.glob('*.png')} == {'cost.png','latency.png','tokens.png'}
    assert all(p.stat().st_size>1000 for p in tmp_path.glob('*.png'))


def test_retry_never_resets_compute_budget():
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from prepare_luna_qwen_inference import check_approval, RESOURCES, LIMITS
    from openwebrl.luna_qwen_metrics import digest
    plan={'example':'plan'}
    approval=dict(approved=True,resources=RESOURCES,limits=LIMITS,plan_sha256=digest(plan),
                  attempts=[dict(job_id='1',time_limit_seconds=3600)])
    check_approval(approval,plan,'1',1800)
    with pytest.raises(ValueError):
        check_approval(approval,plan,'1',1801)


def test_parallel_workers_claim_whole_tasks_without_duplicates_and_resume(tmp_path):
    from openwebrl.luna_qwen_metrics import claimed_tasks, digest
    items = schedule(['a','b','c','d','e'])
    workers = [claimed_tasks(tmp_path,items,i) for i in range(4)]
    held = [next(w) for w in workers]
    assert len({v['task_id'] for v in held}) == 4
    for item in held:
        for mode in MODES:
            write_json(tmp_path/'records'/(digest([item['task_id'],mode])[:24]+'.json'), {})
    assert next(workers[0]) == items[-1]
    assert list(workers[1]) == []
    # Process exit releases an incomplete task; another worker can resume it.
    for w in workers:
        w.close()
    assert list(claimed_tasks(tmp_path,items,5)) == [items[-1]]


def test_api_cap_is_global_across_parallel_workers(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    def reserve(i):
        try:
            Budget(tmp_path,10.,100).update(reserve=1.)
            return True
        except RuntimeError:
            return False
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert sum(pool.map(reserve,range(30))) == 10
    ledger = json.loads((tmp_path/'ledger.json').read_text())
    assert ledger['calls'] == 10 and ledger['charged_or_reserved_usd'] == 10.


def test_worker_overrides_inherited_four_gpu_step_request():
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from prepare_luna_qwen_inference import worker_command
    command=worker_command('344476',Path('/tmp/worker.json'))
    assert '--gpus=h200:1' in command and '--gres=gpu:h200:1' in command
    assert '--cpus-per-task=8' in command and '--mem=120G' in command


def test_official_sft_fifth_arm_has_five_candidates_and_matched_protocol():
    from openwebrl.luna_qwen_eval import PROTOCOL, SFT_PROTOCOL
    assert SFT_PROTOCOL['actor'] == 'OpenWebRL/OpenWebRL-4B-SFT'
    assert {key for key in PROTOCOL if PROTOCOL[key] != SFT_PROTOCOL[key]} == {
        'actor', 'actor_revision', 'candidates'}
    assert Policy('sft_luna5',Path('/unused'),None,None,{},{}).count == 5


def test_fifth_arm_claims_and_plots_require_its_complete_paired_cohort(tmp_path):
    from openwebrl.luna_qwen_metrics import claimed_tasks, digest
    items = [dict(task_id='a',modes=['sft_luna5'])]
    assert list(claimed_tasks(tmp_path,items,0)) == items
    write_json(tmp_path/'records'/(digest(['a','sft_luna5'])[:24]+'.json'),{})
    assert list(claimed_tasks(tmp_path,items,0)) == []
    all_records = records()
    extra = [dict(row,mode='sft_luna5') for row in all_records if row['mode']=='qwen_luna5']
    modes=(*MODES,'sft_luna5')
    with pytest.raises(ValueError,match='Incomplete'):
        summarize(all_records+extra[:1],['a','b'],modes=modes)
    result=summarize(all_records+extra,['a','b'],modes=modes)
    assert result['recorded_episodes']==10 and result['by_mode']['sft_luna5']['overall']==.5
    from report_luna_qwen_inference import plot
    plot(result,tmp_path/'five-arm')
    assert len(list((tmp_path/'five-arm').glob('*.png')))==3


def test_sft_extension_cannot_reset_api_caps_or_change_tasks(tmp_path,monkeypatch):
    import run_luna_qwen_sft_extension as extension
    from openwebrl.luna_qwen_metrics import digest
    monkeypatch.setattr(extension,'ROOT',tmp_path)
    monkeypatch.setattr(extension,'CONTROL',tmp_path/'sft-extension')
    for role in ('luna','judge'):
        budget=tmp_path/f'pilot/{role}-budget'
        budget.mkdir(parents=True)
        link=extension.CONTROL/f'pilot/{role}-budget'
        link.parent.mkdir(parents=True,exist_ok=True)
        link.symlink_to(budget,target_is_directory=True)
    base=dict(schedule=[dict(task_id='a',modes=list(MODES))],resources={'total_seconds':5400},
        limits=dict(luna_usd=15,luna_calls=902,judge_usd=5,judge_calls=160))
    plan=dict(protocol=extension.SFT_PROTOCOL,actor_path=str(extension.ACTOR),
        schedule=[dict(task_id='a',modes=['sft_luna5'])],resources=base['resources'],limits=dict(base['limits']))
    approval=lambda:dict(approved=True,plan_sha256=digest(plan))
    extension.validate_extension(plan,approval(),base)
    plan['limits']['luna_calls']=1202
    with pytest.raises(ValueError,match='caps'):
        extension.validate_extension(plan,approval(),base)
    plan['limits']=base['limits']
    plan['schedule'][0]['task_id']='b'
    with pytest.raises(ValueError,match='ten tasks'):
        extension.validate_extension(plan,approval(),base)


def test_context_repair_selects_transport_failure_not_unsuccessful_tasks(tmp_path):
    from run_luna_qwen_sft_extension import boundary_failure
    record=dict(valid=False,mode='qwen_luna5',artifact=str(tmp_path/'result.json'))
    assert not boundary_failure(record)  # A browser failure is not this bug.
    write_json(tmp_path/'requests/qwen-1.json',dict(provider='qwen',error_type='HTTPStatusError',
        sampling=dict(max_new_tokens=3599)))
    assert boundary_failure(record)
    assert not boundary_failure(dict(record,valid=True,reward=0))
    assert not boundary_failure(dict(record,mode='luna'))


def test_parallel_dispatch_starts_sft_without_waiting_and_reassigns_free_slots():
    from run_luna_qwen_parallel import choose_family
    pending={'qwen':6,'sft':10};active={'qwen':0,'sft':0};families=[]
    for slot in range(4):
        family=choose_family(pending,active,'sft' if slot<2 else 'qwen')
        families.append(family);active[family]+=1
    assert families==['sft','sft','qwen','qwen']
    # A released Qwen slot helps SFT immediately, independent of Qwen completion.
    active['qwen']-=1
    assert choose_family(pending,active)=='sft'
    # No duplicate worker for a task block already held by another worker.
    assert choose_family({'qwen':1,'sft':0},{'qwen':1,'sft':0}) is None
    assert choose_family({'qwen':3,'sft':0},{'qwen':1,'sft':0})=='qwen'


def test_parallel_pending_blocks_preserve_completed_episodes(tmp_path):
    from run_luna_qwen_parallel import pending_blocks
    from openwebrl.luna_qwen_metrics import digest
    items=[dict(task_id='a',modes=['qwen','qwen_luna5']),dict(task_id='b',modes=['qwen'])]
    assert pending_blocks(tmp_path,items)==2
    write_json(tmp_path/'pilot/records'/(digest(['a','qwen'])[:24]+'.json'),{})
    assert pending_blocks(tmp_path,items)==2
    write_json(tmp_path/'pilot/records'/(digest(['a','qwen_luna5'])[:24]+'.json'),{})
    assert pending_blocks(tmp_path,items)==1
