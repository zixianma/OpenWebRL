"""Offline failures that the final read-only Jev audit must detect."""
import base64
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import struct
import time
import zlib

import pytest

SPEC=importlib.util.spec_from_file_location('audit_jev',Path(__file__).parents[1]/'scripts/audit_controlled_jev_actor.py')
m=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(m)


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,sort_keys=True))


def png():
    def chunk(kind,body):return struct.pack('>I',len(body))+kind+body+struct.pack('>I',zlib.crc32(kind+body)&0xffffffff)
    return b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',1280,720,8,2,0,0,0))+chunk(b'IDAT',zlib.compress((b'\0'+b'\0'*3840)*720))+chunk(b'IEND',b'')


@pytest.fixture
def cohort(tmp_path,monkeypatch):
    root=tmp_path/'cohort';source=root/'source';source.mkdir(parents=True)
    reward=source/'openwebrl/eval/reward_online_mind2web.py';reward.parent.mkdir(parents=True)
    shutil.copyfile(Path(__file__).parents[1]/'openwebrl/eval/reward_online_mind2web.py',reward)
    manifest={'recipe_files_sha256':{'openwebrl/eval/reward_online_mind2web.py':m.sha(reward)}}
    write(source/'reference_manifest.json',manifest);write(root/'pricing-verification.json',{})
    tasks=[dict(task_id=str(i),intent='Goal '+str(i),start_url='https://example.test/') for i in range(300)]
    (root/'tasks.jsonl').write_text('\n'.join(json.dumps(t) for t in tasks))
    limits=dict(browser_sessions=330,concurrent_browsers=4,jev_requests=19800,text_http_attempts=19800,text_usd=15,om2w_judge_http_attempts=1320,judge_usd=8)
    schedule=[dict(item_id=str(i),task_id=str(i%300),category='primary' if i<300 else 'smoke',condition='L06') for i in range(303)]
    plan=dict(root=str(root),source=str(source),source_manifest_sha256=m.sha(source/'reference_manifest.json'),
        task_file=str(root/'tasks.jsonl'),task_file_sha256=m.sha(root/'tasks.jsonl'),pricing_sha256=m.sha(root/'pricing-verification.json'),
        schedule=schedule,tasks=tasks,limits=limits,resources=dict(total_seconds=14400),wandb_project='openwebrl-evals',experiment_id='test')
    write(root/'plan.json',plan);write(root/'approval.json',dict(approved=True,plan_sha256=m.digest(plan),limits=limits,resources=plan['resources'],attempts=[dict(job_id='1')]))
    tables=dict(queue=[],attempts=[],reservations=[],counters=[],metadata=[dict(key='plan_sha256',value=m.digest(plan))])
    for item in schedule:
        aid=item['item_id'];directory=root/'attempts'/aid;events=directory/'events';events.mkdir(parents=True)
        (events/'final.png').write_bytes(png());image=dict(path=str(events/'final.png'),sha256=m.sha(events/'final.png'))
        write(events/'00000001.json',dict(event='episode_started',record=dict(actor=m.MODELS['jev'],helper=m.MODELS['text'],harness_revision='actionable-v2',actor_images=False,canonical_thoughts='empty: native Jev does not generate reasoning text')))
        write(events/'00000002.json',dict(event='native_final_state',record=dict(terminal='episode_deadline',operations=[],native_snapshot={},final_screenshot=image)))
        result=dict(attempt_id=aid,task_id=item['task_id'],category=item['category'],condition='L06',terminal='episode_deadline',valid=True,
            actor_model=m.MODELS['jev'],typing_model=m.MODELS['text'],judge_source_sha256=m.JUDGE_SHA,actor_error=None,actions=0,decisions=0,
            browser_closed=True,cleanup_errors=[],final_screenshot=image,api_attempts=dict(jev=0,text=0,judge=0),score=0,judge=dict(judge_text='Judge not run'))
        write(directory/'result.json',result)
        tables['reservations'].append(dict(reservation_id='browser-'+aid,attempt_id=aid,kind='browser_sessions',count=1,status='settled',charged_usd=0))
        tables['queue'].append(dict(item,position=int(aid),status='finished',task_json=json.dumps(dict(task_name='Goal '+item['task_id'],metadata=dict(intent='Goal '+item['task_id'])))))
        descriptor=dict(result,result_path=str(directory/'result.json'),result_sha256=m.sha(directory/'result.json'))
        tables['attempts'].append(dict(attempt_id=aid,item_id=aid,status='finished',artifact_directory=str(directory),result_json=json.dumps(descriptor)))
    tables['counters']=[dict(key='browser_sessions',value=303),dict(key='active_browsers',value=0)]
    write(root/'scheduler-attempts/1/owned-process-teardown.json',dict(verified=True))
    write(root/'collection-complete.json',dict(collection_complete=True))
    write(root/'wandb-current-readonly-verification.json',dict(state='finished',project='openwebrl-evals',id='test',checked_unix=time.time()+2))
    records=[dict(job_id='1',state='COMPLETED',elapsed_seconds=10,alloc_tres='cpu=8,mem=32G')]
    monkeypatch.setattr(m,'database',lambda root:copy.deepcopy(tables))
    return root,tables,records


def edit_result(root,tables,aid,**changes):
    a=next(a for a in tables['attempts'] if a['attempt_id']==str(aid));p=Path(a['artifact_directory'])/'result.json';r=m.read(p);r.update(changes);write(p,r)
    a['result_json']=json.dumps(dict(r,result_path=str(p),result_sha256=m.sha(p)))
    event=p.parent/'events/00000002.json';e=m.read(event)
    for key in ('terminal','final_screenshot'):
        if key in changes:e['record'][key]=changes[key]
    write(event,e)


def test_complete_zero_done_cohort_can_pass_without_judge_calls(cohort):
    root,_,records=cohort;r=m.audit(root,scheduler=records)
    assert not r['issues'] and not r['pending']
    assert r['audit_passed'] and not r['verified_complete']
    assert r['counts']['primary']==300 and r['counts']['smoke']==3
    assert r['counts']['overall_success_rate']==0


def test_original_done_score_is_suppressed_even_if_success(cohort):
    root,tables,records=cohort;edit_result(root,tables,0,terminal='done',actions=1,score=1)
    r=m.audit(root,scheduler=records)
    assert not r['audit_passed'] and 'saved_rejudge_required:0' in r['pending']
    assert r['rows'][0]['score'] is None and r['rows'][0]['original_superseded_score']==1
    assert r['counts']['overall_success_rate'] is None


def test_invalid_is_unscored_and_missing_image_is_not_false_failure(cohort):
    root,tables,records=cohort;edit_result(root,tables,0,valid=False,terminal='error',actor_error='Error',score=None,final_screenshot=None)
    r=m.audit(root,scheduler=records)
    assert not r['issues'] and r['counts']['invalid_primary']==1
    assert r['rows'][0]['score'] is None
    edit_result(root,tables,0,score=0)
    assert any(x['code']=='invalid_scored_as_failure' for x in m.audit(root,scheduler=records)['issues'])


def test_counter_drift_and_overspend_are_blockers(cohort):
    root,tables,records=cohort
    tables['reservations'].append(dict(reservation_id='b',attempt_id='0',kind='browser_sessions',count=331,status='settled',charged_usd=0))
    r=m.audit(root,scheduler=records)
    assert {'counter_reconciliation','budget_exceeded'}<={x['code'] for x in r['issues']}


def test_changed_prior_artifact_and_running_scheduler_block(cohort):
    root,_,records=cohort;r=m.audit(root,scheduler=records)
    previous={'immutable_artifacts':{next(iter(r['immutable_artifacts'])):'wrong'}}
    records[0]['state']='RUNNING';r=m.audit(root,scheduler=records,previous=previous)
    assert any(x['code']=='prior_audit_artifact_changed' for x in r['issues'])
    assert 'scheduler_not_final:1' in r['pending']


def test_bad_png_is_not_valid_evidence(cohort):
    root,tables,records=cohort;image=m.read(root/'attempts/0/result.json')['final_screenshot'];Path(image['path']).write_bytes(b'bad')
    r=m.audit(root,scheduler=records)
    assert {'artifact_hash_mismatch','invalid_png','valid_evidence'}<={x['code'] for x in r['issues']}


def test_repaired_request_binds_exact_ops_empty_thoughts_and_image():
    source=Path(__file__).parents[1]/'openwebrl/eval/reward_online_mind2web.py'
    ops=[dict(action=dict(kind='fill',node=8),number=1,text='target value',status='returned',returned={'executed':'e7'})]
    system,user=m.canonical_text('Goal',ops,source);image=png()
    request=dict(model=m.MODELS['judge'],seed=42,max_completion_tokens=4096,messages=[dict(role='system',content=system),dict(role='user',content=[dict(type='text',text=user),dict(type='image_url',image_url=dict(url='data:image/png;base64,'+base64.b64encode(image).decode(),detail='high'))])])
    assert m.check_repaired_request(request,'Goal',ops,image,source) is None
    bad=copy.deepcopy(request);bad['messages'][1]['content'][0]['text']=user.replace('fill(','click(')
    assert m.check_repaired_request(bad,'Goal',ops,image,source)=='repaired_judge_evidence_differs'
    bad=copy.deepcopy(request);bad['messages'][1]['content'][0]['text']=user.replace('Thought 1: ','Thought 1: invented thought')
    assert m.check_repaired_request(bad,'Goal',ops,image,source)=='repaired_judge_evidence_differs'
    bad=copy.deepcopy(request);bad['messages'][1]['content'][1]['image_url']['url']='data:image/png;base64,bad'
    assert m.check_repaired_request(bad,'Goal',ops,image,source)=='repaired_judge_evidence_differs'
    bad=copy.deepcopy(request);bad['seed']=43
    assert m.check_repaired_request(bad,'Goal',ops,image,source)=='repaired_judge_settings'


def test_review_is_self_contained_and_escapes_untrusted_tasks(cohort,tmp_path):
    root,_,records=cohort;r=m.audit(root,scheduler=records);r['rows']=r['rows'][:1];r['rows'][0]['task']='<script>bad()</script>'
    dest=tmp_path/'review';r=m.review_assets(r,dest);page=m.render(r,dest/'index.html')
    assert '&lt;script&gt;bad()&lt;/script&gt;' in page and '<script>bad()</script>' not in page
    assert '../attempts/' not in page and 'src="assets/0/final.png"' in page
    assert (dest/'assets/0/result.json').is_file()
    (dest/'assets/0/final.png').write_bytes(b'tampered')
    with pytest.raises(ValueError,match='changed|differs'):m.review_assets(r,dest)
