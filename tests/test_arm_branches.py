import asyncio
import base64
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from PIL import Image

from openwebrl.arm_branch_browser import digest
from openwebrl.arm_branch_protocol import (
    action_key, choose_distinct, compare_observations, estimate_selected_success,
    observation_evidence, ReconstructionMismatch, SAMPLING,
)
from openwebrl.arm_branch_teacher import request, schema, parse
from openwebrl.arm_branch_worker import BranchPolicy, Captured, ReplayRejected, AttemptJournal, save_observation, sha, write, branch_environment
from openwebrl.arm_branch_state import BranchState








def observation():
    image=io.BytesIO(); Image.new('RGB',(1280,720),'white').save(image,format='PNG')
    image=image.getvalue()
    page=dict(url='http://fixture/',title='Fixture',text='Ready',screen_size=[1280,720],
              interactive_elements=[],tabs=[],text_characters=5)
    snapshot=dict(storage={},pages=[dict(forms=[],text='Ready')],active_tab=0)
    return dict(screenshot=image,selection_page=page,active_tab_url=page['url'],screen_size=[1280,720],
        branch_snapshot=snapshot,branch_snapshot_sha256=digest(snapshot),
        controlled=dict(sequence=1,screenshot_sha256=hashlib.sha256(image).hexdigest(),page_sha256=digest(page),
            capture_started_monotonic=0.,capture_finished_monotonic=1.,action_attempts=0,executed_actions=[],
            terminal_reason=None,infrastructure_invalid=False))


def output(i=0):
    return ('<think>Consider the task.</think><tool_call>'+json.dumps(dict(name='click',arguments={'point_2d':[i,20]}))+'</tool_call>',[i+1],[-.3],'stop')


class Budget:
    def __init__(self): self.reservations=[]
    def reserve(self,kind,count,**context): self.reservations.append((kind,count)); return {}


def policy(tmp_path, **kwargs):
    tmp_path.mkdir(parents=True,exist_ok=True)
    prompt=tmp_path/'prompt.txt'; prompt.write_text('Browser policy')
    journal=AttemptJournal(tmp_path/'events'); budget=Budget()
    p=BranchPolicy(directory=tmp_path,task_id='fixture',budget=budget,journal=journal,selector_call=None,
        prompt_path=prompt,prompt_sha256=sha(prompt),**kwargs)
    p.set_training_context(dict(prompt_tokens=[1,2]))
    return p,budget


def kwargs(infer,obs=None,turn=0):
    obs=obs or observation()
    return dict(infer=infer,url='http://localhost:1/generate',input_text='Browser policy',
        sampling_params=dict(SAMPLING),images=[base64.b64encode(obs['screenshot']).decode()],
        observation=obs,history=[],task='Find the paper',task_id='fixture',turn=turn,timeout=180)


def anchor(tmp_path, count=3):
    obs=observation(); journal=AttemptJournal(tmp_path/'anchor-events')
    before=save_observation(journal,obs,'before')
    return dict(depth=0,task='Find the paper',task_id='fixture',history=[],prefix=[],
        candidates=choose_distinct([output(i) for i in range(count)], count),before_observation=before,
        evidence=observation_evidence(obs),artifact_sha256='a'*64)


def test_distinctness_ignores_reasoning_and_rejects_truncation():
    same=list(output(0)); same[0]=same[0].replace('Consider the task.','Different thought.')
    assert action_key(same)==action_key(output(0))
    bad=list(output(1)); bad[3]='length'
    assert [x['draw'] for x in choose_distinct([output(0),same,bad,output(1),output(2)])]==[0,3,4]
    with pytest.raises(ValueError): choose_distinct([output(0)]*9)


def test_snapshot_mismatch_rejects_even_when_pixels_match():
    obs=observation(); before=observation_evidence(obs); changed=deepcopy(before)
    changed['snapshot']['pages'][0]['forms']=['changed hidden input']
    with pytest.raises(ReconstructionMismatch):
        compare_observations(before,changed,obs['screenshot'],obs['screenshot'])


def test_visible_replay_records_storage_difference_but_rejects_page_and_form_changes():
    obs=observation(); obs['branch_replay_protocol']='visible-replay-v2'
    before=observation_evidence(obs); changed=deepcopy(before)
    changed['snapshot']['storage']={'cookies':[{'value':'another-session'}]}
    changed['snapshot']['pages'][0]['session']={'anonymous_id':'new'}
    result=compare_observations(before,changed,obs['screenshot'],obs['screenshot'])
    assert result['passed'] and result['storage_difference_recorded'] and not result['exact_snapshot']
    for field in ('forms','text'):
        bad=deepcopy(changed); bad['snapshot']['pages'][0][field]='DIFFERENT'
        with pytest.raises(ReconstructionMismatch): compare_observations(before,bad,obs['screenshot'],obs['screenshot'])
    bad=deepcopy(changed); bad['page']['text']='DIFFERENT'
    with pytest.raises(ReconstructionMismatch): compare_observations(before,bad,obs['screenshot'],obs['screenshot'])
    bad=deepcopy(changed); bad['protocol']='observable-replay-v1'
    with pytest.raises(ReconstructionMismatch): compare_observations(before,bad,obs['screenshot'],obs['screenshot'])


def test_fixed_panel_keeps_five_identical_draws_without_replacement(tmp_path):
    async def run():
        p,b=policy(tmp_path,mode='discover',anchor_depth=0,candidate_count=5,candidate_draws=5,
                   retain_duplicate_candidates=True)
        infer=AsyncMock(return_value=output(0))
        with pytest.raises(Captured): await p(**kwargs(infer))
        panel=json.loads((tmp_path/'anchor.json').read_text())['candidates']
        assert len(panel)==5 and [x['draw'] for x in panel]==list(range(5))
        assert len({x['action_key'] for x in panel})==1
        assert infer.await_count==5 and sum(n for k,n in b.reservations if k=='local_sft_generations')==5
    asyncio.run(run())


def test_fixed_panel_rejects_malformed_draw_without_replacement():
    from openwebrl.arm_branch_protocol import fixed_candidates
    bad=list(output(0)); bad[3]='length'
    with pytest.raises(ValueError): fixed_candidates([output(0)]*4+[bad],5)


@pytest.mark.parametrize('candidates,draws', [(3,9),(5,15)])
def test_discovery_reserves_all_proposals_and_executes_none(tmp_path,candidates,draws):
    async def run():
        p,b=policy(tmp_path,mode='discover',anchor_depth=0,candidate_count=candidates,candidate_draws=draws)
        count=0
        async def infer(*args,**kw):
            nonlocal count
            value=output(count); count+=1; return value
        with pytest.raises(Captured): await p(**kwargs(infer))
        assert count==draws and sum(n for k,n in b.reservations if k=='local_sft_generations')==draws
        assert len(json.loads((tmp_path/'anchor.json').read_text())['candidates'])==candidates
        assert p.captured and not p.released
    asyncio.run(run())


def test_reconstruction_mismatch_calls_no_actor_and_creates_no_ready(tmp_path):
    async def run():
        a=anchor(tmp_path); a['evidence']['page']['text']='different'
        directory=tmp_path/'attempt'
        p,b=policy(directory,mode='branch',anchor=a,candidate=0,repeat=0,state_directory=tmp_path/'state')
        infer=AsyncMock()
        with pytest.raises(ReplayRejected): await p(**kwargs(infer))
        infer.assert_not_called(); assert not (directory/'ready.json').exists() and not b.reservations
    asyncio.run(run())


@pytest.mark.parametrize('count', [3,5])
def test_forced_candidate_waits_for_group_release_without_actor_resampling(tmp_path,count):
    async def run():
        a=anchor(tmp_path,count); directory=tmp_path/'attempt'; state=tmp_path/'state'; state.mkdir()
        p,b=policy(directory,mode='branch',anchor=a,candidate=count-1,repeat=1,state_directory=state)
        infer=AsyncMock(); task=asyncio.create_task(p(**kwargs(infer)))
        while not (directory/'ready.json').exists(): await asyncio.sleep(.01)
        assert not task.done() and not p.released; infer.assert_not_called()
        write(state/'release.json',dict(anchor_sha256=a['artifact_sha256'],attempt_directories=[str(directory)]))
        chosen,_=await task
        assert chosen==output(count-1) and p.released and p.awaiting_candidate and not b.reservations
        infer.assert_not_called()
    asyncio.run(run())


@pytest.mark.parametrize('count', [3,5])
def test_teacher_uses_only_agreed_output_and_no_continuation_evidence(tmp_path,count):
    a=anchor(tmp_path,count)
    a.update(final_reward=1, suffix='SECRET_SUFFIX', final_observation='SECRET_FINAL')
    immediate=[dict(observation=a['before_observation'],tool_feedback={'tool_responses':[]},
                    final_reward=1,suffix='SECRET_SUFFIX') for _ in range(count)]
    order=list(range(count))
    before=request(a,None,'index',order); after=request(a,immediate,'index',order[-1:]+order[:-1])
    assert set(schema('index')['properties'])=={'selection'}
    with pytest.raises(ValueError): schema('factored')
    for value in (before,after):
        assert 'SECRET' not in json.dumps(value) and 'final_reward' not in json.dumps(value)
        assert value['reasoning']=={'effort':'high'}
        assert value['text']['format']['schema']['properties']['selection']['maximum']==count
    assert sum(x['type']=='input_image' for x in before['input'][0]['content'])==1
    assert sum(x['type']=='input_image' for x in after['input'][0]['content'])==count+1
    payload=json.loads(after['input'][0]['content'][0]['text'])
    assert payload['candidates'][0]['response']==output(count-1)[0]
    assert [x['candidate'] for x in payload['immediate_results']]==list(range(1,count+1))
    paired_after=request(a,immediate,'index',order)
    before_payload=json.loads(before['input'][0]['content'][0]['text'])
    after_payload=json.loads(paired_after['input'][0]['content'][0]['text'])
    after_payload.pop('immediate_results')
    assert before_payload==after_payload
    assert {k:v for k,v in before.items() if k!='input'}=={k:v for k,v in paired_after.items() if k!='input'}
    assert before['input'][0]['content'][1:]==paired_after['input'][0]['content'][1:3]
    with pytest.raises(ValueError): request(a,immediate[:-1],'index',order)
    with pytest.raises(ValueError): request(a,immediate,'index',[0]*count)


def test_selected_success_does_not_take_best_of_repetitions():
    records=[dict(state='s',candidate=0,repeat=i,valid=i!=2,reward=int(i==0)) for i in range(3)]
    selected=estimate_selected_success(records,[dict(state='s',candidate=0,repeat=i) for i in range(3)])
    assert selected['overall']==1/3 and selected['valid_only']==.5
    with pytest.raises(ValueError): estimate_selected_success(records+records,[])


def make_state(tmp_path, candidates=3, repetitions=3):
    tmp_path.mkdir(parents=True,exist_ok=True)
    tasks=tmp_path/'tasks.jsonl'; tasks.write_text(json.dumps(dict(metadata={'task_id':'t'}))+'\n')
    plan=dict(root=str(tmp_path),task_file=str(tasks),schedule=[dict(item_id='x',task_id='t',condition='L01',mode='branch')],
        candidate_count=candidates,continuation_repetitions=repetitions,
        resources={'collectors':20},conditions={'L01':{}},limits=dict(local_sft_generations=2))
    write(tmp_path/'plan.json',plan)
    return BranchState(tmp_path,allow_unapproved_for_tests=True)


def test_restart_never_resets_call_budget_or_reclaims_active_attempt(tmp_path):
    state=make_state(tmp_path); claim=state.claim_item('x','w')
    state.reserve('local_sft_generations',2,attempt_id=claim['attempt_id'],condition='L01')
    restored=BranchState(tmp_path,allow_unapproved_for_tests=True)
    with pytest.raises(ValueError): restored.claim_item('x','w2')
    with pytest.raises(RuntimeError): restored.reserve('local_sft_generations',1)
    assert restored.snapshot()['counters']['local_sft_generations']==2


@pytest.mark.parametrize('candidates,repetitions', [(3,3),(3,5),(5,3)])
def test_release_requires_all_unique_candidate_repetition_pairs(tmp_path,candidates,repetitions):
    state=make_state(tmp_path/'ledger',candidates,repetitions); dirs=[]
    for c in range(candidates):
        for r in range(repetitions):
            p=tmp_path/f'{c}-{r}'; p.mkdir(); dirs.append(str(p))
            write(p/'ready.json',dict(candidate=c,repeat=r,anchor_sha256='a'*64,
                attempt_directory=str(p),checks=[dict(passed=True)]))
    with pytest.raises(ValueError): state.release('s','a'*64,dirs[:-1])
    state.release('s','a'*64,dirs)
    assert (state.root/'states/s/release.json').exists()


def test_immediate_evidence_is_saved_for_a_terminal_candidate(tmp_path,monkeypatch):
    async def run():
        a=anchor(tmp_path); obs=observation()
        monkeypatch.setattr('openwebrl.controlled_browser_server.fresh_observation',AsyncMock(return_value=obs))
        raw=SimpleNamespace(step=AsyncMock(return_value=(obs,0.,True,False,{'tool_responses':['done']})),exit=AsyncMock())
        generation=SimpleNamespace(_create_env=AsyncMock(return_value=(raw,{})))
        budget=SimpleNamespace(reserve=lambda *args,**kw:{},settle=lambda *args,**kw:None)
        p=SimpleNamespace(awaiting_candidate=True,anchor=a,journal=AttemptJournal(tmp_path/'events'),
            directory=tmp_path,candidate=0,repeat=1)
        config=dict(mode='local_process',width=1280,height=720,dpr=1,resize_output_coords=True,
            resize_scale=1000,use_screenshot=True,use_a11ytree=False,
            local_process=dict(server_module='openwebrl.controlled_browser_server'))
        final={}
        with branch_environment(generation,final,p,budget):
            client,_=await generation._create_env('fixture',config)
            response=await client.step([dict(name='done',args={})])
            assert response[2] is True and not p.awaiting_candidate
            record=json.loads((tmp_path/'immediate.json').read_text())
            assert record['terminated'] and record['candidate']==0 and record['repeat']==1
            assert record['tool_feedback']['tool_responses']==['done']
            await client.exit()
        assert final['browser_closed']
    asyncio.run(run())


@pytest.mark.parametrize('always_changes', [False,True])
def test_branch_snapshot_is_checked_inside_each_bounded_capture_attempt(monkeypatch,always_changes):
    from openwebrl.arm_branch_browser import branch_environment_class
    from openwebrl.controlled_browser_server import ObservationMismatch
    class Parent:
        def __init__(self):
            self.attempts=0;self.version=0;self.events=[]
            self.controlled_journal=lambda name,record:self.events.append((name,record))
        async def _capture_once(self):
            self.attempts+=1
            if self.attempts==1 or always_changes:self.version+=1
            return dict(controlled={'screenshot_sha256':'image'})
        async def controlled_observation(self):
            for attempt in range(3):
                try:return await self._capture_once()
                except ObservationMismatch:
                    if attempt==2:raise
    async def state(env):return dict(version=env.version),{}
    monkeypatch.setattr('openwebrl.arm_branch_browser.browser_state',state)
    async def run():
        env=branch_environment_class(Parent,ObservationMismatch)()
        if always_changes:
            with pytest.raises(ObservationMismatch):await env.controlled_observation()
            assert env.attempts==3
            assert all(name=='branch_capture_rejected' for name,_ in env.events)
        else:
            observation=await env.controlled_observation()
            assert env.attempts==2 and observation['branch_snapshot']=={'version':1}
            assert [name for name,_ in env.events]==['branch_capture_rejected','branch_state']
    asyncio.run(run())


def test_failed_branch_initialization_closes_and_settles_owned_browser(tmp_path,monkeypatch):
    from openwebrl.env import local_process_env
    async def run():
        raw=SimpleNamespace(initialize=AsyncMock(side_effect=RuntimeError('unstable reset')),exit=AsyncMock())
        monkeypatch.setattr(local_process_env,'create_local_process_env',AsyncMock(return_value=raw))
        monkeypatch.setattr('openwebrl.controlled_browser_server.fresh_observation',AsyncMock(side_effect=RuntimeError('uninitialized')))
        async def native_factory(task_id,config):
            client=await local_process_env.create_local_process_env(config['local_process'])
            try:await client.initialize()
            except BaseException:
                await client.exit();raise
        generation=SimpleNamespace(_create_env=native_factory)
        from unittest.mock import Mock
        budget=SimpleNamespace(reserve=Mock(return_value={'reservation_id':'one'}),settle=Mock())
        policy=SimpleNamespace(awaiting_candidate=False)
        final={}
        config=dict(mode='local_process',width=1280,height=720,dpr=1,resize_output_coords=True,
            resize_scale=1000,use_screenshot=True,use_a11ytree=False,
            local_process=dict(server_module='openwebrl.controlled_browser_server'))
        with branch_environment(generation,final,policy,budget):
            with pytest.raises(RuntimeError,match='unstable reset'):await generation._create_env('fixture',config)
            await final['client'].exit()
        assert final['browser_closed']
        raw.exit.assert_awaited_once()
        budget.settle.assert_called_once_with({'reservation_id':'one'},{'closed':True})
    asyncio.run(run())


@pytest.mark.parametrize('change_during', ['page_read', 'storage_read'])
def test_popup_race_retries_only_observation_under_existing_capture_bound(change_during):
    from openwebrl.arm_branch_browser import branch_environment_class
    from openwebrl.controlled_browser_server import ObservationMismatch, controlled_environment

    class Parent:
        # Exercise the actual shared retry/deadline implementation.
        controlled_observation = controlled_environment(object, {}).controlled_observation

        def __init__(self):
            self.events = []
            self.controlled_capture_attempts = 0
            self.controlled_journal = lambda name, record: self.events.append((name, record))
            self.changed = False
            self.page = SimpleNamespace(evaluate=AsyncMock(side_effect=self.page_state))
            self.original_page = self.page
            self.popup = SimpleNamespace(evaluate=AsyncMock(return_value={'url': 'popup'}))
            self.context = SimpleNamespace(pages=[self.page], storage_state=AsyncMock(side_effect=self.storage))

        def popup_arrives(self):
            if not self.changed:
                self.context.pages.append(self.popup)
                self.page = self.popup
                self.changed = True

        async def page_state(self, script):
            if change_during == 'page_read':
                self.popup_arrives()
            return {'url': 'original'}

        async def storage(self, *, indexed_db):
            assert indexed_db is True
            if change_during == 'storage_read':
                self.popup_arrives()
            return {'cookies': [], 'origins': []}

        async def _capture_once(self):
            self.controlled_capture_attempts += 1
            return {'controlled': {'screenshot_sha256': 'popup-image'}}

    async def run():
        env = branch_environment_class(Parent, ObservationMismatch)()
        observation = await env.controlled_observation()
        assert observation['branch_snapshot']['active_tab'] == 1
        assert observation['branch_snapshot']['pages'] == [{'url': 'original'}, {'url': 'popup'}]
        assert env.page is env.popup  # The browser event chose it, not a fallback.
        assert [name for name, _ in env.events] == ['branch_capture_rejected', 'branch_state']
        assert env.controlled_capture_attempts == 1
    asyncio.run(run())


def test_persistently_missing_active_tab_is_rejected_without_substitution():
    from openwebrl.arm_branch_browser import branch_environment_class
    from openwebrl.controlled_browser_server import ObservationMismatch, controlled_environment

    class Parent:
        controlled_observation = controlled_environment(object, {}).controlled_observation

        def __init__(self):
            self.events = []
            self.page = object()
            self.context = SimpleNamespace(pages=[], storage_state=AsyncMock())
            self.controlled_journal = lambda name, record: self.events.append((name, record))

        async def _capture_once(self):
            raise AssertionError('An inconsistent active tab must not produce an image')

    async def run():
        env = branch_environment_class(Parent, ObservationMismatch)()
        original_page = env.page
        with pytest.raises(ObservationMismatch, match='absent'):
            await env.controlled_observation()
        assert env.page is original_page
        assert len(env.events) == 3
        env.context.storage_state.assert_not_called()
    asyncio.run(run())


def test_release_json_is_published_whole_and_cannot_replace_a_receipt(tmp_path, monkeypatch):
    import openwebrl.arm_branch_worker as worker
    path = tmp_path/'release.json'
    original_dump = worker.json.dump
    def interrupted_dump(value, handle, **kwargs):
        handle.write(' ')
        handle.flush()
        assert not path.exists(), 'A polling worker must not see partial JSON'
        original_dump(value, handle, **kwargs)
    monkeypatch.setattr(worker.json, 'dump', interrupted_dump)
    worker.write(path, {'cohort_committed_before_candidate_dispatch': True})
    saved = path.read_bytes()
    assert json.loads(saved)['cohort_committed_before_candidate_dispatch'] is True
    monkeypatch.setattr(worker.json, 'dump', original_dump)
    with pytest.raises(FileExistsError):
        worker.write(path, {'different': 'receipt'})
    assert path.read_bytes() == saved
