"""Offline regression tests for scientific/completion gates in the SFT auditor."""
import base64
import copy
import json
from pathlib import Path
import struct
import sys
import zlib
import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import audit_controlled_sft_suite as m


def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,sort_keys=True))
def png():
    def part(k,v):return struct.pack('>I',len(v))+k+v+struct.pack('>I',zlib.crc32(k+v)&0xffffffff)
    return b'\x89PNG\r\n\x1a\n'+part(b'IHDR',struct.pack('>IIBBBBB',1280,720,8,2,0,0,0))+part(b'IDAT',zlib.compress((b'\0'+b'\0'*3840)*720))+part(b'IEND',b'')


def test_refresh_baseline_releases_old_trajectory_payloads(tmp_path,monkeypatch):
    p=tmp_path/'prior.json';proofs={'private/result.json':'abc'}
    write(p,dict(immutable_artifacts=proofs,rows=[dict(assistant_history=['large saved history'])],summary={'old':'counts'}))
    monkeypatch.setattr(Path,'read_text',lambda *_args,**_kwargs:pytest.fail('baseline must stream; never load the complete text'))
    assert m.read_baseline(p)=={'immutable_artifacts':proofs}


def test_streamed_baseline_rejects_truncated_report(tmp_path):
    from ijson.common import IncompleteJSONError
    p=tmp_path/'broken.json';p.write_bytes(b'{"immutable_artifacts":{"saved":"abc"},"rows":[')
    with pytest.raises(IncompleteJSONError):m.read_baseline(p)


def test_proof_maps_cache_hashes_and_still_detect_changed_baseline(tmp_path,monkeypatch):
    p=tmp_path/'artifact.json';write(p,{'value':1});expected=m.sha(p);calls=[];original=m.sha
    monkeypatch.setattr(m,'sha',lambda path:(calls.append(str(path)),original(path))[1])
    c=m.Checks();c.preserve(p,expected,immutable=False);c.preserve(p,expected)
    assert len(calls)==1 and c.artifacts[str(p.resolve())]==expected and not c.issues
    p.write_text('{"value":2}')
    fresh=m.Checks();fresh.preserve(p,expected)
    assert fresh.issues[0]['code']=='artifact_hash_mismatch'


def test_event_hashes_share_proof_map_without_rehash(tmp_path,monkeypatch):
    p=tmp_path/'0001.json';write(p,dict(event='saved',record={'x':1}));expected=m.sha(p)
    c=m.Checks();events=c.events(tmp_path)
    monkeypatch.setattr(m,'sha',lambda path:pytest.fail('event was unnecessarily rehashed'))
    assert c.preserve(p,expected)==expected and events[0][1]['record']=={'x':1} and not c.issues


def coverage():
    plans=[];queues=[]
    for i in range(2):
        tasks=[str(t) for t in range(i*150,(i+1)*150)];schedule=[]
        for category,subset in [('primary',tasks),('smoke',tasks[:3])]:
            for task in subset:
                for arm in m.ARMS:schedule.append(dict(item_id=f'{i}/{category}/{task}/{arm}',task_id=task,condition=arm,category=category))
        plans.append(dict(shard=dict(index=i,task_ids=tasks),schedule=schedule));queues.append(copy.deepcopy(schedule))
    return plans,queues


def test_matched_coverage_rejects_missing_arm_and_cross_shard_duplicate():
    plans,queues=coverage();c=m.Checks();m.check_coverage(plans,queues,c);assert not c.issues
    plans[0]['schedule'][0]['condition']='L08';queues[0][0]['condition']='L08'
    c=m.Checks();m.check_coverage(plans,queues,c);assert any(i['code']=='all_conditions_per_task' for i in c.issues)
    plans,queues=coverage();plans[1]['shard']['task_ids'][0]=plans[0]['shard']['task_ids'][0]
    c=m.Checks();m.check_coverage(plans,queues,c);assert any(i['code']=='disjoint_complete300' for i in c.issues)


def budget_fixture():
    limits=dict(browser_sessions=660,browser_sessions_per_condition=165,concurrent_browsers=8,local_sft_generations=158400,luna_selector_requests=9905,jev_selector_requests=9905,local_kev_requests=9905,canonical_judge_http_attempts=2640,luna_usd=25,canonical_judge_usd=12.5)
    r=dict(reservation_id='browser',kind='browser_sessions',condition='L01',attempt_id='a',count=1,status='settled',charged_usd=0)
    db=dict(reservations=[r],events=[dict(id=1,event='reserve',details_json=json.dumps(dict(kind='browser_sessions',reservation_id='browser'))),dict(id=2,event='settle',details_json=json.dumps(dict(reservation_id='browser')))],counters=[dict(key='browser_sessions',value=1),dict(key='browsers:L01',value=1),dict(key='active_browsers',value=0)])
    return db,dict(shard=dict(index=0),limits=limits)


def test_per_shard_budget_cannot_borrow_spare_peer_budget():
    db,p=budget_fixture();c=m.Checks();m.check_budget(db,p,c);assert not c.issues
    p['limits']['browser_sessions']=0
    c=m.Checks();m.check_budget(db,p,c);assert any(i['code']=='budget_exceeded' for i in c.issues)


def test_unknown_usage_keeps_full_reservation_and_browser_peak_audited():
    db,p=budget_fixture();db['reservations'].append(dict(reservation_id='luna',kind='luna_selector_requests',condition='L08',attempt_id='a',count=1,status='charged_unknown',reserved_usd=.05,charged_usd=.05,context_json=json.dumps(dict(input_token_bound=1000,output_token_bound=4096))))
    db['counters'] += [dict(key='luna_selector_requests',value=1),dict(key='luna_usd',value=.05)]
    c=m.Checks();out=m.check_budget(db,p,c);assert not c.issues and out['peak_browsers']==1 and out['unknown_charges']==1
    db['reservations'][-1]['charged_usd']=.001
    c=m.Checks();m.check_budget(db,p,c);assert any(x['code']=='unknown_charge_not_preserved' for x in c.issues)


def test_archived_recovery_binds_source_and_preserves_all_reservations(tmp_path,monkeypatch):
    archive=tmp_path/'archive';manifest={'recipe_files_sha256':{}}
    write(archive/'source/reference_manifest.json',manifest)
    plan={k:{} for k in ('schedule','task_file_sha256','conditions','limits','resources','actor_revision','actor_files')}
    plan.update(shard={'index':0},source_manifest_sha256=m.sha(archive/'source/reference_manifest.json'))
    write(archive/'plan.json',plan);write(archive/'approval.json',{});(archive/'state.sqlite3').write_bytes(b'private archived snapshot')
    receipt=tmp_path/'preparation.json';write(receipt,dict(archive=str(archive),parent_plan_sha256=m.digest(plan)))
    approval=dict(attempts=[dict(job_id='replacement',administrative_recovery_receipt=str(receipt))])
    r=dict(reservation_id='held',attempt_id='prior',kind='luna_selector_requests',condition='L08',count=1,reserved_usd=.1,context_json='{}')
    before=dict(attempts=[dict(attempt_id='prior')],reservations=[r]);monkeypatch.setattr(m,'snapshot',lambda _:copy.deepcopy(before))
    c=m.Checks();m.check_recovery_lineage(approval,plan,copy.deepcopy(before),c);assert not c.issues
    current=copy.deepcopy(before);current['attempts']=[];current['reservations'][0]['reserved_usd']=0
    c=m.Checks();m.check_recovery_lineage(approval,plan,current,c)
    assert {'recovery_attempt_dropped','recovery_reservation_changed'}<={x['code'] for x in c.issues}


@pytest.fixture
def episode(tmp_path):
    root=tmp_path/'a';events=root/'events';events.mkdir(parents=True);browser=root/'browser';browser.mkdir();image=events/'final.png';image.write_bytes(png());proof=dict(path=str(image),sha256=m.sha(image));prompt=tmp_path/'prompt.md';prompt.write_text('Restored policy')
    text='A real thought.\n</think>\n<tool_call>\n{"name":"wait","arguments":{}}\n</tool_call>'
    capture=dict(sequence=1,image=proof,page=dict(url='https://example.test/',text='visible'),action_attempts=0)
    write(browser/'00000001.json',dict(event='capture_verified',record=capture))
    write(events/'00000001.json',dict(event='episode_started',record=dict(sampling=m.SAMPLING,reward_source_sha256=m.JUDGE_SHA)))
    batch=dict(turn=0,condition='L01',sampling=m.SAMPLING,candidate_seeds=[m.seed('task',0,0)],prompt_tokens=100,screenshot_sha256=proof['sha256'],page_sha256=m.state_digest(capture['page']))
    write(events/'00000002.json',dict(event='proposal_batch_reserved',record=batch));write(events/'00000003.json',dict(event='proposal_received',record=dict(turn=0,candidate=0,output_text=text+'<|im_end|>\n')))
    result=dict(attempt_id='a',task_id='task',condition='L01',status='Status.ABORTED',reward=0.,valid=True,invalid_reason=None,judge_http_attempts=0,actor_decisions=1,browser_operations=0,browser_closed=True,final_screenshot=proof,
        metadata=dict(intent='Goal',messages=[dict(role='system',content='Restored policy'),dict(role='assistant',content='<think>\n'+text)],full_image_list=[base64.b64encode(png()).decode()],
        controlled_terminal=dict(screenshot_sha256=proof['sha256'],original_status='Status.ABORTED',reward_source_sha256=m.JUDGE_SHA,policy_boundary_reason='episode_deadline',terminal_reason=None),reward=dict(protocol='online_mind2web',judge_text='Judge not run')))
    write(root/'result.json',result)
    a=dict(attempt_id='a',status='finished',artifact_directory=str(root));q=dict(condition='L01',task_id='task',category='primary',task_json=json.dumps(dict(metadata=dict(intent='Goal'))))
    plan=dict(shard=dict(index=0),worker_config=dict(prompt_path=str(prompt)));reservations=[dict(kind='local_sft_generations',count=1)]
    def refresh():a['result_json']=json.dumps(dict((k,v) for k,v in result.items() if k not in ('metadata','response'))|dict(result_path=str(root/'result.json'),result_sha256=m.sha(root/'result.json')))
    refresh()
    return root,a,q,plan,reservations,result,refresh


def run_episode_fixture(fixture):
    root,a,q,p,r,value,refresh=fixture;write(root/'result.json',value);refresh();c=m.Checks();row=m.audit_attempt(a,q,p,r,c,None,None);return row,c


def test_valid_aborted_is_zero_without_judge_and_requires_known_boundary(episode):
    row,c=run_episode_fixture(episode);assert not c.issues and row['valid'] and row['canonical_reward']==0 and not row['success']
    episode[-2]['metadata']['controlled_terminal']['policy_boundary_reason']=None
    row,c=run_episode_fixture(episode);assert any(x['code']=='valid_aborted_boundary_missing' for x in c.issues)


def test_finished_attempt_requires_saved_result_but_interrupted_attempt_preserves_evidence(episode):
    root,a,q,p,reservations,_,_=episode
    a['result_json']=None;c=m.Checks()
    assert m.audit_attempt(a,q,p,reservations,c,None,None) is None
    assert [x['code'] for x in c.issues]==['finished_result_missing']
    assert str((root/'events/00000001.json').resolve()) in c.artifacts
    a['status']='interrupted';c=m.Checks()
    assert m.audit_attempt(a,q,p,reservations,c,None,None) is None
    assert not c.issues


def test_invalid_aborted_zero_is_preserved_separately(episode):
    result=episode[-2];result.update(valid=False,invalid_reason='missing_fresh_terminal_evidence');result.pop('final_screenshot');result['metadata'].pop('controlled_terminal')
    row,c=run_episode_fixture(episode);assert not c.issues and not row['valid'] and row['canonical_reward']==0 and not row['success']


def test_history_tamper_and_missing_policy_fail(episode):
    episode[-2]['metadata']['messages'][1]['content']+="\nInvented later observation."
    episode[-2]['metadata']['messages'][0]['content']=''
    row,c=run_episode_fixture(episode)
    assert {'selected_proposal_history_binding','restored_policy_history'}<={x['code'] for x in c.issues}


def test_different_final_image_and_reservation_drift_fail(episode):
    episode[-2]['final_screenshot']['sha256']='wrong';episode[4][0]['count']=5
    row,c=run_episode_fixture(episode)
    assert {'artifact_hash_mismatch','actor_batch_reservations','terminal_evidence_binding'}<={x['code'] for x in c.issues}


def test_selector_argmax_guard_and_luna_single_json():
    native=dict(answers=dict(selection=dict(choice='1',probabilities={'1':.1,'2':.6,'3':.1,'4':.1,'5':.1})))
    with pytest.raises(ValueError,match='rejected'):m.selector_index(native,'jev')
    luna=dict(status='completed',output=[dict(type='message',content=[dict(type='output_text',text='{"selected_index":3}')])])
    assert m.selector_index(luna,'luna')==3
    luna['output'][0]['content'].append(dict(type='output_text',text='{"selected_index":1}'))
    with pytest.raises(ValueError):m.selector_index(luna,'luna')


def test_private_review_copies_only_intended_image_and_escapes_task(episode,tmp_path):
    row,c=run_episode_fixture(episode);row['task']='<script>bad()</script>'
    summary={arm:dict(label=label,finished=0,successes=0,valid=0,invalid=0,final_success_rate=None) for arm,label in m.LABELS.items()}
    report=dict(rows=[row],summary=summary,created_utc='fixture',status='partial_unverified',issues=[],pending=['still running'],shards=[],limitations=[])
    dest=tmp_path/'review';m.write_review(report,dest);page=(dest/'index.html').read_text()
    assert '&lt;script&gt;bad()&lt;/script&gt;' in page and '<script>bad()</script>' not in page
    evidence=json.loads((dest/'episodes/a/evidence.json').read_text())
    assert 'full_image_list' not in evidence and evidence['final_screenshot']['path']=='final.png'
    assert (dest/'episodes/a/final.png').read_bytes()==png()


def test_shared_selector_view_uses_same_page_five_candidates_and_five_executed_actions():
    page=dict(url='https://example.test/',title='Page',text='p'*17000,text_characters=17000,interactive_elements=[dict(id=7)],tabs=[],screen_size=[1280,720])
    capture=dict(page=page,action_attempts=7);ops=[dict(attempt=i,action={'name':'wait'},success=True) for i in range(1,10)]
    outputs={i:'Original '+str(i) for i in range(5)};order=[4,2,0,1,3]
    state=m.expected_selector_state(capture,'Goal',outputs,order,ops)
    assert len(state['page']['text'])==16000 and state['page']['text_truncated']
    assert [a['attempt'] for a in state['recent_executed_actions']]==[3,4,5,6,7]
    assert state['candidates']['1']['text']=='Original 4' and set(state)=={'task','page','recent_executed_actions','candidates'}


def test_identical_static_images_do_not_overwrite_earlier_execution_context():
    page=dict(url='https://example.test/',title='Static',text='same page',interactive_elements=[],tabs=[],screen_size=[1280,720])
    outputs={i:str(i) for i in range(5)};ops=[dict(attempt=i,action={'name':'wait'}) for i in range(1,11)];order=list(range(5))
    captures=[dict(page=page,action_attempts=3),dict(page=page,action_attempts=10)]
    supplied=m.expected_selector_state(captures[0],'Goal',outputs,order,ops)
    assert supplied!=m.expected_selector_state(captures[-1],'Goal',outputs,order,ops)
    assert any(supplied==m.expected_selector_state(c,'Goal',outputs,order,ops) for c in captures)


@pytest.fixture
def startup_work(tmp_path):
    def proof(path,value):
        write(path,value);return dict(path=str(path),sha256=m.sha(path))
    root=tmp_path;old=root/'scheduler-attempts/old';new=root/'scheduler-attempts/new'
    plan=dict(root=str(root),shard=dict(index=0),worker_config=dict(actor_checkpoint='/pinned/SFT'))
    prior=proof(root/'prior-plan.json',plan);prior_hash=m.digest(plan)
    request=dict(model='jev-1.13.0');event=proof(old/'jev-request.json',dict(record=dict(request=request)))
    original=dict(completed=True,expected_model='jev-1.13.0',state_bytes=262144,outcome='known_capacity_rejection',http_status=400,error_type='max_tokens_exceeded',plan_sha256=prior_hash,job_id='old',request_evidence_path=event['path'],request_sha256=m.digest(request),browser_episodes=0,sft_proposals=0,reservation=dict(reservation_id='jev'))
    original_proof=proof(old/'jev-context-preflight.json',original)
    entries=[];image=base64.b64encode(b'current image').decode()
    for i in range(10):
        text=f'original saved prompt {i}';attempt=f'saved-{i}'
        batch=dict(prompt_sha256=m.raw_sha(text.encode()),screenshot_sha256=m.raw_sha(b'current image'),candidate_seeds=[100+i],prompt_tokens=20000+i)
        payload=dict(text=text,image_data=[image],return_logprob=True,sampling_params=dict(m.SAMPLING,sampling_seed=100+i))
        entries.append(dict(attempt_id=attempt,prompt_tokens=20000+i,request=proof(root/f'input/request-{i}.json',payload),source_batch=proof(root/f'input/batch-{i}.json',dict(record=batch)),source_rollout=proof(root/f'input/rollout-{i}.json',dict(llm_input_texts=text)),source_result=proof(root/f'input/result-{i}.json',dict(attempt_id=attempt))))
    manifest=proof(root/'manifest.json',dict(source_plan_sha256=prior_hash,root=str(root),actor_checkpoint='/pinned/SFT',sampling=m.SAMPLING,requests=entries))
    plan['serving_recovery']=dict(previous_max_running_requests=5,max_running_requests=10,sampling=m.SAMPLING,prior_plan=prior,prior_plan_sha256=prior_hash,acceptance_manifest=manifest,prior_jev_diagnostic=original_proof)
    proof(new/'allocation-validated.json',dict(plan_sha256=m.digest(plan)))
    proof(new/'jev-context-preflight.json',dict(original,plan_sha256=m.digest(plan),job_id='new',external_calls_made=0,reused_from=original_proof))
    context=dict(purpose=m.ACCEPTANCE_PURPOSE,request_sha256=m.digest(dict(acceptance_manifest_sha256=manifest['sha256'])))
    reservation=dict(reservation_id='accept',kind='local_sft_generations',count=10,reserved_usd=0,**context)
    proof(new/'serving-acceptance/reservation.json',reservation)
    proof(new/'serving-acceptance-server-info.json',dict(model_path='/pinned/SFT',max_running_requests=10,context_length=32768,mem_fraction_static=.60,disable_cuda_graph=True,dtype='bfloat16',tp_size=1))
    receipt=dict(completed=True,purpose=m.ACCEPTANCE_PURPOSE,plan_sha256=m.digest(plan),reservation=reservation,requests=10,charged_local_sft_generations=10,browser_sessions=0,external_api_calls=0,executed_actions=0,acceptance_manifest=manifest,actor_checkpoint='/pinned/SFT',max_running_requests=10,prompt_tokens=[e['prompt_tokens'] for e in entries],failures=[])
    proof(new/'serving-acceptance.json',receipt)
    for i,e in enumerate(entries):
        proof(new/f'serving-acceptance/response-{i:02d}.json',dict(index=i,request_sha256=e['request']['sha256'],http_status=200,elapsed_seconds=.1,response=dict(text='No action executed',meta_info=dict(prompt_tokens=e['prompt_tokens'],completion_tokens=100,finish_reason=dict(type='stop'),total_retractions=0))))
    proof(new/'worker-runtime.json',{})
    db=dict(reservations=[dict(reservation_id='jev',attempt_id=None,kind='jev_selector_requests',count=1,context_json=json.dumps(dict(request_sha256=m.digest(request)))),dict(reservation_id='accept',attempt_id=None,condition=None,kind='local_sft_generations',count=10,status='consumed',reserved_usd=0,charged_usd=0,context_json=json.dumps(context))])
    return root,plan,db


def test_adopted_jev_proof_not_double_counted_and_ten_local_calls_separate(startup_work):
    root,p,db=startup_work;c=m.Checks();out=m.check_startup_work(root,p,db,c)
    assert not c.issues and not c.pending
    assert out['diagnostic_adoptions']==[dict(job_id='new',reservation_id='jev',new_external_calls=0)]
    assert out['saved_context_acceptances'][0]['generations_charged']==10
    assert out['saved_context_acceptances'][0]['successful_responses']==10
    assert 'rows' not in out


def test_adopted_jev_receipt_cannot_disguise_additional_external_call(startup_work):
    root,p,db=startup_work;path=root/'scheduler-attempts/new/jev-context-preflight.json';value=m.read(path);value['external_calls_made']=1;write(path,value)
    c=m.Checks();m.check_startup_work(root,p,db,c)
    assert any(x['code']=='adopted_diagnostic_identity' for x in c.issues)


def test_acceptance_wrong_seed_or_image_is_detected_even_with_rehashed_manifest(startup_work):
    root,p,db=startup_work;manifest_path=Path(p['serving_recovery']['acceptance_manifest']['path']);manifest=m.read(manifest_path)
    entry=manifest['requests'][0];request_path=Path(entry['request']['path']);value=m.read(request_path);value['sampling_params']['sampling_seed']=999;value['image_data']=[base64.b64encode(b'other image').decode()];write(request_path,value)
    entry['request']['sha256']=m.sha(request_path);write(manifest_path,manifest);p['serving_recovery']['acceptance_manifest']['sha256']=m.sha(manifest_path)
    c=m.Checks();m.check_startup_work(root,p,db,c)
    assert any(x['code']=='acceptance_request_science' for x in c.issues)


def test_acceptance_counter_loss_and_retracted_response_fail(startup_work):
    root,p,db=startup_work;db['reservations'][1]['count']=9
    path=root/'scheduler-attempts/new/serving-acceptance/response-02.json';value=m.read(path);value['response']['meta_info']['total_retractions']=1;write(path,value)
    c=m.Checks();m.check_startup_work(root,p,db,c)
    assert {'acceptance_reservation_binding','acceptance_response_validation'}<={x['code'] for x in c.issues}


def test_unaccounted_startup_request_or_reservation_double_count_fails(startup_work):
    root,p,db=startup_work;extra=copy.deepcopy(db['reservations'][0]);extra['reservation_id']='extra';db['reservations'].append(extra)
    c=m.Checks();m.check_startup_work(root,p,db,c)
    assert any(x['code']=='startup_reservation_coverage' for x in c.issues)


def test_failed_acceptance_retains_full_charge_and_requires_no_workers(startup_work):
    root,p,db=startup_work;directory=root/'scheduler-attempts/new';path=directory/'serving-acceptance.json';value=m.read(path);value.update(completed=False,failures=['ValueError']);write(path,value)
    write(directory/'serving-acceptance/error-00.json',dict(error_type='ValueError'))
    c=m.Checks();m.check_startup_work(root,p,db,c)
    assert any(x['code']=='serving_workers_after_failed_acceptance' for x in c.issues)
    assert db['reservations'][1]['count']==10


def test_eleventh_acceptance_response_is_not_silently_ignored(startup_work):
    root,p,db=startup_work;directory=root/'scheduler-attempts/new/serving-acceptance'
    write(directory/'response-10.json',m.read(directory/'response-00.json'))
    c=m.Checks();m.check_startup_work(root,p,db,c)
    assert any(x['code']=='acceptance_unaccounted_outcome' for x in c.issues)
