import hashlib
import io
import json
from pathlib import Path
import sqlite3
import sys
from types import SimpleNamespace

import pytest
from PIL import Image

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import rejudge_controlled_jev_saved as m
from openwebrl.controlled_sft_worker import AttemptJournal
from openwebrl.controlled_jev_state import JevState
from openwebrl.controlled_sft_state import digest


@pytest.fixture
def cohort(tmp_path):
    tasks=[dict(task_id=str(i),intent='Fixture task',start_url='about:blank') for i in range(4)]
    path=tmp_path/'tasks.jsonl';path.write_text(''.join(json.dumps(dict(metadata=t))+'\n' for t in tasks))
    plan=dict(root=str(tmp_path),task_file=str(path),tasks=tasks,conditions={'L06':{}},
        schedule=[dict(item_id=str(i),task_id=str(i),condition='L06',category='primary') for i in range(4)],
        resources=dict(collectors=4,gpus=0,cpus=8,memory_gib=32,total_seconds=14400),
        limits=dict(browser_sessions=330,concurrent_browsers=4,jev_requests=19800,text_http_attempts=19800,
                    text_usd=15.,om2w_judge_http_attempts=1320,judge_usd=8.),
        worker_config=dict(canonical_reward_sha256='frozen-reward'))
    (tmp_path/'plan.json').write_text(json.dumps(plan))
    state=JevState(tmp_path,allow_unapproved_for_tests=True)
    with state.connect() as db:db.execute("UPDATE metadata SET value='primary' WHERE key='phase'")
    output=io.BytesIO();Image.new('RGB',(1280,720),'white').save(output,format='PNG')
    for i in range(4):
        claim=state.claim('worker'+str(i));directory=Path(claim['artifact_directory'])
        journal=AttemptJournal(directory/'events');screenshot=journal.blob('final.png',output.getvalue())
        terminal='blocked' if i==2 else 'done';valid=i!=3
        operations=[dict(action=dict(kind='fill',id='n4',label='Query'),text='fixture',status='returned')]
        journal('native_final_state',dict(terminal=terminal,operations=operations,native_snapshot={},
            final_screenshot=screenshot,cleanup_errors=[]))
        if terminal=='done' and valid:journal('judge_received',dict(response={'model':'o4-mini-2025-04-16'}))
        result=dict(task_id=claim['task_id'],condition='L06',category='primary',attempt_id=claim['attempt_id'],
            valid=valid,terminal=terminal,browser_closed=True,actor_error=None,cleanup_errors=[],
            final_screenshot=screenshot,actions=len(operations)+(terminal=='done'),decisions=2,
            actor_model='jev-1.13.0',typing_model='gpt-4.1-mini-2025-04-14',score=float(i%2),reward=float(i%2),
            judge={'judge_text':'OLD VERDICT MUST NOT LEAK'},api_attempts={'jev':2,'text':1,'judge':1},elapsed_seconds=17.)
        proof=m.immutable_json(directory/'result.json',result)
        state.finish(claim['attempt_id'],dict(result,result_path=proof['path'],result_sha256=proof['sha256']))
    return tmp_path,state


def prepared(cohort):
    root,state=cohort;proof=m.prepare_manifest(root,root/'rejudge-inputs.json');manifest=m.read(proof['path'])
    plan=state.plan;plan['saved_rejudge']=dict(input_manifest=proof['path'],input_manifest_sha256=proof['sha256'],controller_sha256=m.sha(m.__file__))
    (root/'plan.json').write_text(json.dumps(plan))
    with state.connect() as db:
        db.execute('UPDATE metadata SET value=? WHERE key=?',(digest(plan),'plan_sha256'))
        for entry in manifest['entries']:
            db.execute("UPDATE attempts SET status='superseded_judge_input' WHERE attempt_id=?",(entry['source_attempt_id'],))
            db.execute("UPDATE queue SET status='pending' WHERE item_id=?",(entry['item_id'],))
    return root,JevState(root,allow_unapproved_for_tests=True),manifest


def test_manifest_selects_all_valid_done_both_old_scores_without_mutation(cohort):
    root,state=cohort;before=m.sha(root/'state.sqlite3')
    proof=m.prepare_manifest(root,root/'inputs.json');manifest=m.read(proof['path'])
    assert m.sha(root/'state.sqlite3')==before
    assert manifest['selected_count']==2 and {e['task_id'] for e in manifest['entries']}=={'0','1'}
    assert manifest['new_browser_episodes']==0 and manifest['external_calls']=='canonical_judge_only'
    with pytest.raises(FileExistsError):m.prepare_manifest(root,root/'inputs.json')


def test_incomplete_collection_cannot_prepare(cohort):
    root,state=cohort
    with state.connect() as db:db.execute("UPDATE queue SET status='active' WHERE item_id='0'")
    with pytest.raises(ValueError,match='Finish browser'):m.prepare_manifest(root,root/'inputs.json')


@pytest.mark.parametrize('field',['result','native_final_state','final_screenshot'])
def test_tampered_saved_evidence_fails_before_judge(cohort,field):
    root,state,manifest=prepared(cohort);entry=manifest['entries'][0]
    path=Path(entry[field]['path']);path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError,match='changed'):m.load_evidence(root,entry,state.plan)


def test_manifest_controller_and_task_pins(cohort):
    root,state,manifest=prepared(cohort)
    assert m.validate_manifest(root,state.plan)['selected_count']==2
    state.plan['saved_rejudge']['controller_sha256']='bad'
    with pytest.raises(ValueError,match='controller changed'):m.validate_manifest(root,state.plan)
    state.plan['tasks'][0]['intent']='Changed goal'
    with pytest.raises(ValueError,match='task identity'):m.load_evidence(root,manifest['entries'][0],state.plan)


def test_only_judge_calls_allowed(cohort):
    root,state,manifest=prepared(cohort);claim=state.claim('judge-worker')
    budget=m.JudgeOnlyBudget(state,claim);transport=m.JudgeOnlyTransport(budget,AttemptJournal(Path(claim['artifact_directory'])/'events'))
    before=state.snapshot()['counters']
    for kind in ('browser_sessions','jev_requests','text_http_attempts'):
        with pytest.raises(ValueError,match='forbids'):budget.reserve(kind)
    with pytest.raises(ValueError,match='no actor/helper'):transport.post('url','key',{})
    with pytest.raises(ValueError,match='no actor/helper'):transport.request('jev','url','key',{})
    assert state.snapshot()['counters']==before


def test_source_supersession_required_and_nonmanifest_task_cannot_run(cohort):
    root,state,manifest=prepared(cohort);m.validate_recovery_queue(state,manifest)
    with state.connect() as db:db.execute("UPDATE queue SET status='pending' WHERE item_id='2'")
    with pytest.raises(ValueError,match='new browser'):m.validate_recovery_queue(state,manifest)
    with state.connect() as db:
        db.execute("UPDATE queue SET status='finished' WHERE item_id='2'")
        db.execute("UPDATE attempts SET status='finished' WHERE attempt_id=?",(manifest['entries'][0]['source_attempt_id'],))
    with pytest.raises(ValueError,match='superseded'):m.validate_recovery_queue(state,manifest)


def test_rejudge_preserves_actor_evidence_and_charges_only_new_judge(cohort,monkeypatch):
    root,state,manifest=prepared(cohort);claim=state.claim('judge-worker');entry=manifest['entries'][0]
    original_sha=m.sha(entry['result']['path']);original_png=Path(entry['final_screenshot']['path']).read_bytes()
    async def score(task,operations,terminal,image,transport,budget,journal,config):
        assert task['intent']=='Fixture task' and terminal=='done' and image==original_png
        assert operations[0]['action']['kind']=='fill'
        assert 'OLD VERDICT' not in json.dumps([task,operations,config])
        request={'model':'o4-mini-2025-04-16','max_completion_tokens':4096,'messages':[]}
        reservation=budget.reserve('om2w_judge_http_attempts',request=request)
        budget.settle(reservation,{'prompt_tokens':100,'completion_tokens':10})
        journal('judge_received',dict(response={'model':'o4-mini-2025-04-16'}))
        transport.counts['judge']+=1
        return 1.,{'judge':1.,'judge_text':'Fresh corrected evidence'},'Status.COMPLETED'
    monkeypatch.setattr(m,'canonical_score',score)
    result=m.rejudge_claim(claim,state.plan,state,entry);state.finish(claim['attempt_id'],result)
    assert result['score']==1 and result['actions']==2 and result['decisions']==2
    assert result['api_attempts']=={'jev':0,'text':0,'judge':1}
    assert result['saved_rejudge']['source_actor_api_attempts']=={'jev':2,'text':1}
    assert result['saved_rejudge']['source_episode_elapsed_seconds']==17.
    assert Path(result['final_screenshot']['path']).read_bytes()==original_png
    assert m.sha(entry['result']['path'])==original_sha
    counters=state.snapshot()['counters']
    assert counters['om2w_judge_http_attempts']==1 and counters.get('browser_sessions',0)==0
    assert counters['judge_usd']==pytest.approx(.000154)
    assert counters.get('jev_requests',0)==0 and counters.get('text_http_attempts',0)==0
    m.validate_recovery_queue(state,manifest)


def test_judge_error_preserves_source_and_active_attempt(cohort,monkeypatch):
    root,state,manifest=prepared(cohort);claim=state.claim('judge-worker');entry=manifest['entries'][0]
    async def score(*args):raise RuntimeError('mock judge transport failure')
    monkeypatch.setattr(m,'canonical_score',score)
    with pytest.raises(RuntimeError):m.rejudge_claim(claim,state.plan,state,entry)
    assert state.snapshot()['queue']['active']==1 and m.sha(entry['result']['path'])==entry['result']['sha256']
    assert (Path(claim['artifact_directory'])/'events/final.png').exists()


def test_omitting_a_valid_done_is_rejected(cohort):
    root,state,manifest=prepared(cohort);omitted=manifest['entries'].pop()
    with state.connect() as db:
        db.execute("UPDATE attempts SET status='finished' WHERE attempt_id=?",(omitted['source_attempt_id'],))
        db.execute("UPDATE queue SET status='finished' WHERE item_id=?",(omitted['item_id'],))
    with pytest.raises(ValueError,match='omitted'):m.validate_recovery_queue(state,manifest)


def test_worker_failure_halts_and_preserves_owned_claim(cohort,monkeypatch):
    root,state,manifest=prepared(cohort)
    monkeypatch.setattr(m,'validate_frozen',lambda root:state.plan)
    monkeypatch.setattr(m,'JevState',lambda root:state)
    async def score(*args):raise RuntimeError('offline fixture judge failure')
    monkeypatch.setattr(m,'canonical_score',score)
    with pytest.raises(RuntimeError,match='fixture'):m.worker(root,'rejudge-worker')
    assert state.snapshot()['queue']['active']==1
    assert state.snapshot()['halt']['reason']=='Saved-only judge repair failed'
    with state.connect() as db:
        row=db.execute("SELECT artifact_directory FROM attempts WHERE status='active'").fetchone()
    events=[m.read(p) for p in (Path(row['artifact_directory'])/'events').glob('*.json')]
    assert any(e['event']=='saved_rejudge_failed' for e in events)
