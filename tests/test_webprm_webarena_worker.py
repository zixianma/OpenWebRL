"""End-to-end synthetic worker contracts with all network connections blocked."""
import base64
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from openwebrl.webprm_webarena import EvaluationNotReady, digest, load_native_evaluator
from openwebrl.webprm_webarena_episode import TaskEnvelope
from openwebrl.webprm_webarena_worker import (ACTOR_SNAPSHOTS, SELECTORS, ViewerActor, ViewerSelector,
    _run_episode, broker_request, cell_name, claim_lease, episode_evaluator, load_viewer_reward, require_supported, run_loop,
    protocol_definition, single_action_system_prompt, validate_execution_config)

SOURCE = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-baselines-20261009/sources/arbiter')
REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('webarena_collector', REPO/'scripts/run_webprm_webarena_pilot.py')
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)
PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/l9sAAAAASUVORK5CYII=')


def wrapped(kind='string_match', answer='42'):
    evaluation = {'eval_types': [kind]}
    if kind == 'string_match':
        evaluation['reference_answers'] = {'exact_match': answer}
    elif kind == 'url_match':
        evaluation['reference_url'] = 'http://127.0.0.1/goal0'
    else:
        evaluation['program_html'] = [{'url': 'last', 'locator': 'document.body.textContent',
                                      'required_contents': {'exact_match': answer}}]
    return TaskEnvelope.from_release(dict(task_id=5, intent='Synthetic visible task', start_url='__REDDIT__',
        sites=['reddit'], storage_state='private', eval=evaluation, private_reference='NEVER_SEND_THIS'),
        {'reddit': 'http://127.0.0.1'})


class Scripted:
    def __init__(self, terminal_after=1, count=5):
        self.requests, self.terminals = [], 0
        self.terminal_after, self.count = terminal_after, count

    def __call__(self, config, payload, journal, **kwargs):
        self.requests.append(dict(payload=deepcopy(payload), **kwargs))
        purpose = kwargs.get('purpose')
        if purpose == 'terminal':
            stop = self.terminals >= self.terminal_after
            self.terminals += 1
            text = json.dumps(dict(thought='Synthetic stopping decision',
                                  action='send_msg_to_user [42]' if stop else None, confidence=.9))
        elif purpose == 'state_diff':
            text = 'The synthetic page changed.'
        elif purpose == 'policy':
            text = json.dumps({'BLOCKS': [dict(thought=f'Candidate {index}',
                action=f'goto [http://127.0.0.1/goal{index}]', confidence=.5) for index in range(self.count)]})
        elif purpose == 'selector_pair':
            text = '<Answer>Response 1</Answer>'
        elif purpose == 'selector_checklist':
            text = '<answer>Reach the stated page.</answer>'
        elif purpose == 'selector_pointwise':
            text = '<answer>Yes</answer>'
        elif purpose == 'selector_piotr':
            text = '{"selection":4}'
        else:
            raise AssertionError('Unexpected request purpose: ' + str(purpose))
        choice = dict(message={'content': text}, finish_reason='stop')
        if purpose == 'selector_pointwise':
            choice['logprobs'] = {'content': [dict(token='<answer>', top_logprobs=[]),
                dict(token='Yes', top_logprobs=[dict(token='Yes', logprob=-.1),
                    dict(token='No', logprob=-3.), dict(token='In', logprob=-4.)])]}
        return dict(model=payload['model'], choices=[choice])


class FakeEnv:
    def __init__(self, output, *, fail_close=False, terminal_error=False):
        self.output, self.fail_close, self.terminal_error = output, fail_close, terminal_error
        self.events, self.actions, self.messages = [], [], []
        self.page = SimpleNamespace(url='http://127.0.0.1', screenshot=lambda **kw: PNG,
            evaluate=lambda code: '42', content=lambda: '42', is_closed=lambda: False)

    def observation(self, error=None):
        import numpy as np
        return dict(url=self.page.url, chat_messages=deepcopy(self.messages), last_action_error=error,
                    screenshot=np.zeros((2, 2, 3), dtype=np.uint8))

    def reset(self, seed):
        self.events.append('reset')
        return self.observation(), {}

    def step(self, action):
        self.events.append('step'); self.actions.append(action)
        if action.startswith('send_msg_to_user'):
            if not self.terminal_error:
                self.messages.append(dict(role='assistant', message='42'))
            return self.observation('controlled terminal error' if self.terminal_error else None), 0, True, False, {}
        self.page.url = action.split('[', 1)[1][:-1]
        return self.observation(), 0, False, False, {}

    def close(self):
        self.events.append('close')
        saved = json.loads((self.output/'result.json').read_text())
        assert saved['cleanup_status'] == 'pending'
        assert (self.output/'native-evaluation/terminal.png').exists()
        if self.fail_close:
            raise TimeoutError('Controlled cleanup failure')


@unittest.skipUnless(SOURCE.exists(), 'Pinned source unavailable')
class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.network = patch('socket.socket.connect', side_effect=AssertionError('Network forbidden'))
        self.network.start(); self.addCleanup(self.network.stop)
        self.temporary = tempfile.TemporaryDirectory(prefix='wa-worker-', dir='/tmp')
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)

    def config(self, selector='first5', turns=2):
        return dict(arbiter_source=str(SOURCE), shepherd_source=str(SOURCE.parent/'shepherd'),
            actor_model=ACTOR_SNAPSHOTS[0], selector=selector, seed=42, max_turns=turns, deadline=time.time()+60,
            reset_id='synthetic-reset', identity_sha256='synthetic-identity',
            cell_name=cell_name(ACTOR_SNAPSHOTS[0], selector),
            arbiter_model='WebArbiter7B', arbiter_endpoint='http://critic.invalid/v1',
            shepherd_model='WebShepherd8B', shepherd_endpoint='http://critic.invalid/v1')

    def run_case(self, name='first5', kind='string_match', *, turns=2, fail_close=False):
        folder=self.folder/(name+'-'+kind); folder.mkdir()
        config=self.config(name,turns); request=Scripted(count=1 if name=='no_search' else 5)
        actor=ViewerActor(config,folder/'requests.jsonl',request=request)
        selector=ViewerSelector(name,config,folder/'requests.jsonl',request=request)
        env=FakeEnv(folder,fail_close=fail_close)
        native=load_native_evaluator(SOURCE)
        def evaluate(*args,**kwargs):
            env.events.append('evaluate')
            return native.evaluate(*args,**kwargs)
        result=run_loop(wrapped(kind),name,config,env,actor,selector,
            lambda environment,obs:dict(text='Actual synthetic observation',url=environment.page.url,
                last_action_error=obs.get('last_action_error')),evaluate,folder)
        return result,env,request,folder

    def test_run_episode_constructs_with_no_task_and_resets_once_using_pinned_viewport(self):
        # Use the real BrowserEnv constructor: task/page are None until reset,
        # and its task class has no class-level viewport. Only browser I/O,
        # credentials/site and model transport are replaced with local doubles.
        from browsergym.core.env import BrowserEnv
        config=dict(self.config('no_search'),output=str(self.folder),endpoints={'reddit':'http://127.0.0.1'},
                    authorization_status='approved',api_usd_cap=1.,code_sha256={})
        task=wrapped();folder=self.folder/task.execution_id/config['cell_name']
        request=Scripted(count=1);instances=[];events=[];test=self
        class DeferredTaskEnv(BrowserEnv):
            def __init__(self,**kwargs):
                super().__init__(**kwargs)
                test.assertIsNone(self.task);test.assertIsNone(self.page)
                test.assertFalse(hasattr(self.task_entrypoint,'viewport'))
                test.assertEqual(self.viewport,{'width':1280,'height':720})
                self.output=folder;self.fail_close=False;self.terminal_error=False
                self.events=events;self.actions=[];self.messages=[]
                instances.append(self);events.append('construct')
            observation=FakeEnv.observation
            step=FakeEnv.step
            close=FakeEnv.close
            def reset(self,seed):
                test.assertIsNone(self.task)
                self.task=self.task_entrypoint(seed=seed,**self.task_kwargs)
                test.assertEqual(self.task.viewport,self.viewport)
                self.page=SimpleNamespace(url='http://127.0.0.1',viewport_size=deepcopy(self.viewport),
                    screenshot=lambda **kw:PNG,evaluate=lambda code:'42',content=lambda:'42',is_closed=lambda:False)
                return FakeEnv.reset(self,seed)
        def processor_factory(root,viewport_size):
            test.assertIsNone(instances[0].task)
            test.assertEqual(viewport_size,{'width':1280,'height':720})
            events.append('processor')
            def observe(page):
                test.assertEqual(page.viewport_size,viewport_size)
                events.append('observe')
                return 'Actual synthetic observation',{}
            return SimpleNamespace(observe=observe,manifest={'synthetic':True})
        module='openwebrl.webprm_webarena_worker.'
        with patch(module+'broker_request',return_value=request), \
                patch(module+'claim_lease',return_value=({'scope':'synthetic'},'synthetic-reset')), \
                patch(module+'prepare_browser'), \
                patch(module+'authenticated_state',return_value={'cookies':[],'origins':[]}), \
                patch('browsergym.core.env.BrowserEnv',DeferredTaskEnv), \
                patch(module+'load_native_observation_processor',side_effect=processor_factory):
            result=_run_episode({'released':task.native_config()},config['cell_name'],config)
        self.assertTrue(result['valid'],result)
        self.assertEqual(result['success'],1)
        self.assertEqual(events[:4],['construct','processor','reset','observe'])
        self.assertEqual(events.count('reset'),1)
        self.assertEqual(events.count('close'),1)
        self.assertEqual([row['purpose'] for row in request.requests],['terminal','policy','state_diff','terminal'])
        self.assertTrue((folder/'native-evaluation/native-verdict.json').exists())

    def test_five_search_selectors_share_identical_actor_candidates_and_only_execute_selected(self):
        hashes=[]
        for name in ['first5','random5','shepherd8b','arbiter7b','gpt4omini']:
            result,env,request,folder=self.run_case(name)
            self.assertTrue(result['valid']); self.assertEqual(result['success'],1)
            self.assertEqual(env.events[-2:],['evaluate','close'])
            rollout=json.loads((folder/'rollout.json').read_text())
            hashes.append(rollout[0]['proposal']['candidates_sha256'])
            self.assertEqual(len(env.actions),2)
            self.assertEqual(env.actions[0],rollout[0]['proposal']['candidates'][rollout[0]['selected_index']]['action'])
            self.assertNotIn('NEVER_SEND_THIS',json.dumps(request.requests))
            self.assertNotIn('reference_answers',json.dumps(request.requests))
            actor_requests=[r for r in request.requests if r['purpose'] in {'terminal','state_diff','policy'}]
            self.assertEqual([r['purpose'] for r in actor_requests],['terminal','policy','state_diff','terminal'])
            self.assertTrue(all(r['payload']['model']==ACTOR_SNAPSHOTS[0] for r in actor_requests))
            trajectory=json.loads((folder/'native-evaluation/trajectory.json').read_text())
            self.assertEqual(trajectory[-1]['answer'],'42')
        self.assertEqual(len(set(hashes)),1)

    def test_no_search_requests_one_action_without_hidden_search_or_selector_calls(self):
        result,env,request,folder=self.run_case('no_search')
        self.assertTrue(result['valid']);self.assertEqual(result['success'],1)
        self.assertEqual([row['purpose'] for row in request.requests],['terminal','policy','state_diff','terminal'])
        policy=next(row['payload'] for row in request.requests if row['purpose']=='policy')
        self.assertEqual(policy['n'],1)
        system=policy['messages'][0]['content']
        self.assertIn('Generate exactly ONE action',system)
        self.assertNotIn('5 items',system);self.assertNotIn('Generate 5',system)
        self.assertNotIn('DIVERSITY REQUIREMENT',system)
        original=ViewerActor(self.config(),'unused').harness.policy_system_prompt
        self.assertEqual(system,single_action_system_prompt(original))
        self.assertEqual(system.split('# ACTION RULES',1)[1],original.split('# ACTION RULES',1)[1])
        rollout=json.loads((folder/'rollout.json').read_text())
        self.assertEqual(rollout[0]['proposal']['requested_raw_slots'],1)
        self.assertEqual(rollout[0]['proposal']['raw_slot_count'],1)
        self.assertTrue(rollout[0]['proposal']['exact_requested_raw_slots'])
        self.assertEqual(len(env.actions),2)

    def test_clarified_no_search_prompt_is_opt_in_and_native_parseable(self):
        from openwebrl.webprm_webarena_worker import ENVELOPE_NO_SEARCH_PROMPT
        config = self.config('no_search')
        old_actor = ViewerActor(config, 'unused')
        actor = ViewerActor(dict(config, no_search_prompt_version=ENVELOPE_NO_SEARCH_PROMPT), 'unused')
        original = actor.harness.policy_system_prompt
        prompt = actor.policy_system_prompt
        good = prompt.split('# EXAMPLE (GOOD - with clear justification)\n')[1].split('# ANTI-EXAMPLES')[0]
        example = next(line for line in good.splitlines() if line.startswith('{'))
        block = json.loads(example)['BLOCKS']
        self.assertEqual(len(block), 1)
        parsed = actor.harness.parse_policy_response(example).candidates
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0].action, block[0]['action'])
        bare = json.dumps(block[0])
        self.assertEqual(len(actor.harness.parse_policy_response(bare).candidates), 0)
        self.assertNotEqual(old_actor.policy_system_prompt, prompt)
        self.assertEqual(prompt.split('# ACTION RULES\n')[1].split('# EXAMPLE')[0],
                         original.split('# ACTION RULES\n')[1].split('# EXAMPLE')[0])
        self.assertEqual(prompt.split('# ANTI-EXAMPLES')[1], original.split('# ANTI-EXAMPLES')[1])
        guided = ViewerActor(dict(self.config('arbiter7b'), no_search_prompt_version=ENVELOPE_NO_SEARCH_PROMPT), 'unused')
        self.assertEqual(guided.policy_system_prompt, original)

    def test_clarified_no_search_version_is_bound_to_protocol_and_request(self):
        from openwebrl.webprm_webarena_worker import ENVELOPE_NO_SEARCH_PROMPT, NA_TURN_LIMIT, validate_protocol_contract
        contract = protocol_definition(SOURCE, turn_limit=NA_TURN_LIMIT, no_search_prompt=ENVELOPE_NO_SEARCH_PROMPT)
        config = dict(self.config('no_search'), turn_limit_protocol=NA_TURN_LIMIT,
                      no_search_prompt_version=ENVELOPE_NO_SEARCH_PROMPT,
                      protocol=contract['identity'], protocol_contract=contract)
        self.assertEqual(validate_protocol_contract(config)['identity'], 'webarena_table5_viewer_reconstruction_v4')
        request = Scripted(count=1)
        actor = ViewerActor(config, 'unused', request=request)
        actor.propose(wrapped().actor_view, dict(text='state', url='http://127.0.0.1'), [])
        payload = next(row['payload'] for row in request.requests if row['purpose'] == 'policy')
        self.assertEqual(hashlib.sha256(payload['messages'][0]['content'].encode()).hexdigest(),
                         contract['single_action_prompt_sha256'])
        changed = dict(config); changed.pop('no_search_prompt_version')
        with self.assertRaises(EvaluationNotReady): validate_protocol_contract(changed)
        with self.assertRaises(EvaluationNotReady):
            protocol_definition(SOURCE, no_search_prompt=ENVELOPE_NO_SEARCH_PROMPT)
        with self.assertRaises(EvaluationNotReady):
            ViewerActor(dict(config, no_search_prompt_version='unknown'), 'unused')
        with self.assertRaises(EvaluationNotReady):
            single_action_system_prompt(actor.harness.policy_system_prompt.replace(
                '# EXAMPLE (GOOD - with clear justification)', '# CHANGED'), version=ENVELOPE_NO_SEARCH_PROMPT)

    def test_gpt_selector_matches_arbiter_messages_queue_and_uses_fixed_paid_mini(self):
        config=self.config();actor_request=Scripted()
        proposal=ViewerActor(config,'unused',request=actor_request).propose(wrapped().actor_view,
            dict(text='state',url='http://127.0.0.1'),[])
        requests=[];outcomes=[]
        for name in ['gpt4omini','arbiter7b']:
            request=Scripted()
            actor4o=dict(config,actor_model=ACTOR_SNAPSHOTS[1],selector=name)
            outcomes.append(ViewerSelector(name,actor4o,'unused',request=request).select(proposal,0))
            requests.append(request.requests)
        self.assertEqual(outcomes[0],outcomes[1]);self.assertEqual(len(requests[0]),4)
        for paid,local in zip(*requests):
            self.assertFalse(paid['local']);self.assertTrue(local['local'])
            self.assertEqual(paid['payload']['model'],ACTOR_SNAPSHOTS[0])
            self.assertEqual(paid['payload']['messages'],local['payload']['messages'])
            self.assertEqual(paid['payload']['temperature'],0.)
            self.assertEqual(paid['payload']['top_p'],1.)
            self.assertEqual(paid['purpose'],'selector_pair')

    def test_malformed_gpt_comparison_never_synthesizes_a_winner_or_retries(self):
        config=self.config('gpt4omini');request=Scripted()
        proposal=ViewerActor(config,'unused',request=request).propose(wrapped().actor_view,
            dict(text='state',url='http://127.0.0.1'),[])
        def malformed(*args,**kwargs):
            response=request(*args,**kwargs)
            response['choices'][0]['message']['content']='No preference verdict'
            return response
        with self.assertRaises(ValueError):
            ViewerSelector('gpt4omini',config,'unused',request=malformed).select(proposal,0)
        self.assertEqual(sum(row['purpose']=='selector_pair' for row in request.requests),1)

    def test_no_search_does_not_promote_an_extra_slot_after_bad_single_action(self):
        config=self.config('no_search');request=Scripted(count=5)
        def overreturned(*args,**kwargs):
            response=request(*args,**kwargs)
            if kwargs.get('purpose')=='policy':
                blocks=json.loads(response['choices'][0]['message']['content'])
                blocks['BLOCKS'][0]=None
                response['choices'][0]['message']['content']=json.dumps(blocks)
            return response
        proposal=ViewerActor(config,'unused',request=overreturned).propose(wrapped().actor_view,
            dict(text='state',url='http://127.0.0.1'),[])
        self.assertEqual(proposal['requested_raw_slots'],1)
        self.assertEqual(proposal['candidates'],[])
        self.assertEqual([row['raw_slot'] for row in proposal['rejected_slots']],[0,1,2,3,4])
        self.assertEqual(len(request.requests),2)

    def test_piotr_receives_snapshot_before_actor_and_frozen_prior_action_display(self):
        config=self.config('piotr_selectionarm',turns=3);request=Scripted(terminal_after=2)
        env=FakeEnv(self.folder);actor=ViewerActor(config,'unused',request=request)
        evidence_seen=[];actor_propose=actor.propose
        changed_png=PNG+b'changed-after-actor'
        def propose(*args):
            proposal=actor_propose(*args)
            env.page.screenshot=lambda **kw:changed_png
            return proposal
        actor.propose=propose
        class CausalSelector:
            def historical_action(self,raw,**kw):raise AssertionError('No unselected action in this case')
            def select(self,proposal,turn,*,evidence=None):
                if proposal['terminal']:
                    return 0,[]
                evidence_seen.append(deepcopy(evidence))
                self_case.assertEqual(len(env.actions),turn)
                return 3,{'selected_grounded_action':f'grounded action saved at turn {turn}'}
        self_case=self
        result=run_loop(wrapped(),'piotr_selectionarm',config,env,actor,CausalSelector(),
            lambda e,o:dict(text='state',url=e.page.url,viewport={'width':1280,'height':720},
                element_bounds={'1':[1,2,10,20]}),load_native_evaluator(SOURCE).evaluate,self.folder)
        self.assertTrue(result['valid']);self.assertEqual(result['success'],1)
        self.assertEqual(len(evidence_seen),2)
        self.assertEqual(evidence_seen[0]['screenshot'],PNG)
        self.assertEqual(evidence_seen[1]['screenshot'],changed_png)
        self.assertEqual(evidence_seen[1]['history'][0]['selectionarm_action'],'grounded action saved at turn 0')
        self.assertEqual(env.actions[:2],['goto [http://127.0.0.1/goal3]']*2)
        rollout=json.loads((self.folder/'rollout.json').read_text())
        self.assertEqual(rollout[0]['selection_evidence']['screenshot_sha256'],hashlib.sha256(PNG).hexdigest())
        self.assertNotIn('selectionarm_action',json.dumps(request.requests))

    def test_actual_piotr_adapter_selects_original_native_action_and_hides_evaluator(self):
        import io
        from PIL import Image
        from openwebrl.webprm_webarena_piotr import CHECKPOINT, MODEL, REVISION, SOURCE as PIOTR_SOURCE
        config=dict(self.config('piotr_selectionarm'),piotr_model=MODEL,piotr_revision=REVISION,
            piotr_checkpoint=str(CHECKPOINT),piotr_source=str(PIOTR_SOURCE),piotr_endpoint='http://critic.invalid/v1')
        request=Scripted();env=FakeEnv(self.folder)
        png=io.BytesIO();Image.new('RGB',(1280,720)).save(png,format='PNG')
        env.page.screenshot=lambda **kw:png.getvalue()
        result=run_loop(wrapped(),'piotr_selectionarm',config,env,
            ViewerActor(config,'unused',request=request),ViewerSelector('piotr_selectionarm',config,'unused',request=request),
            lambda e,o:dict(text='actual synthetic state',url=e.page.url,viewport={'width':1280,'height':720},
                element_bounds={}),load_native_evaluator(SOURCE).evaluate,self.folder)
        self.assertTrue(result['valid']);self.assertEqual(result['success'],1)
        self.assertEqual(env.actions[0],'goto [http://127.0.0.1/goal3]')
        calls=[row for row in request.requests if row['purpose']=='selector_piotr']
        self.assertEqual(len(calls),1);self.assertTrue(calls[0]['local'])
        self.assertEqual(calls[0]['payload']['model'],MODEL)
        self.assertNotIn('NEVER_SEND_THIS',json.dumps(calls))
        self.assertNotIn('reference_answers',json.dumps(calls))
        rollout=json.loads((self.folder/'rollout.json').read_text())
        self.assertEqual(rollout[0]['selected_index'],3)
        self.assertEqual(rollout[0]['selection_trace']['selected_raw_slot'],3)
        self.assertEqual(rollout[0]['proposal']['requested_raw_slots'],5)

    def test_piotr_history_grounds_detector_fallback_action_without_selection(self):
        # 356527 task 97: a detector-override click bypassed selection, then the next
        # Piotr prompt rejected that history row for lacking grounded display.
        import io
        from PIL import Image
        from openwebrl.webprm_webarena_piotr import MODEL,REVISION,CHECKPOINT,SOURCE as PIOTR_SOURCE
        config=dict(self.config('piotr_selectionarm',turns=4),piotr_model=MODEL,piotr_revision=REVISION,
            piotr_checkpoint=str(CHECKPOINT),piotr_source=str(PIOTR_SOURCE),piotr_endpoint='http://critic.invalid/v1')
        request=Scripted(terminal_after=2);calls=[]
        def responding(*a,**kw):
            response=request(*a,**kw)
            if kw.get('purpose')=='terminal' and request.terminals==1:
                response['choices'][0]['message']['content']=json.dumps(dict(thought='Detector navigates',
                    action='goto [http://127.0.0.1/goal9]',confidence=.9))
            if kw.get('purpose')=='selector_piotr':calls.append(deepcopy(a[1]))
            return response
        env=FakeEnv(self.folder)
        png=io.BytesIO();Image.new('RGB',(1280,720)).save(png,format='PNG');env.page.screenshot=lambda **kw:png.getvalue()
        result=run_loop(wrapped(),'piotr_selectionarm',config,env,ViewerActor(config,'unused',request=responding),
            ViewerSelector('piotr_selectionarm',config,'unused',request=responding),
            lambda e,o:dict(text='synthetic',url=e.page.url,viewport={'width':1280,'height':720},element_bounds={}),
            load_native_evaluator(SOURCE).evaluate,self.folder)
        self.assertTrue(result['valid']);self.assertEqual(result['success'],1)
        self.assertEqual(env.actions,['goto [http://127.0.0.1/goal9]','goto [http://127.0.0.1/goal3]','send_msg_to_user [42]'])
        self.assertEqual(len(calls),1)
        self.assertIn('goal9',json.dumps(calls[0]['messages']))

    def test_piotr_v2_executes_original_unavailable_candidate_and_records_error(self):
        import io
        from PIL import Image
        from openwebrl.webprm_webarena_piotr import MODEL,REVISION,CHECKPOINT,SOURCE as PIOTR_SOURCE,REPRESENTATION_V2
        config=dict(self.config('piotr_selectionarm'),piotr_model=MODEL,piotr_revision=REVISION,
            piotr_checkpoint=str(CHECKPOINT),piotr_source=str(PIOTR_SOURCE),piotr_representation=REPRESENTATION_V2)
        request=Scripted()
        actions=['click [404]','scroll [down]','new_tab','go_forward','close_tab']
        def responding(*a,**kw):
            response=request(*a,**kw)
            if kw.get('purpose')=='policy':
                response['choices'][0]['message']['content']=json.dumps({'BLOCKS':[
                    dict(thought='Original thought '+str(i),action=action,confidence=.5) for i,action in enumerate(actions)]})
            if kw.get('purpose')=='selector_piotr':response['choices'][0]['message']['content']='{"selection":1}'
            return response
        env=FakeEnv(self.folder);original_step=env.step
        def step(action):
            if action=='click [404]':
                env.actions.append(action);env.events.append('step')
                return env.observation('ActionParsingError: Native coordinate target lacks captured viewport bounds'),0,False,False,{}
            return original_step(action)
        env.step=step
        png=io.BytesIO();Image.new('RGB',(1280,720)).save(png,format='PNG');env.page.screenshot=lambda **kw:png.getvalue()
        result=run_loop(wrapped(),'piotr_selectionarm',config,env,ViewerActor(config,'unused',request=responding),
            ViewerSelector('piotr_selectionarm',config,'unused',request=responding),
            lambda e,o:dict(text='synthetic',url=e.page.url,viewport={'width':1280,'height':720},element_bounds={}),
            load_native_evaluator(SOURCE).evaluate,self.folder)
        self.assertTrue(result['valid']);self.assertEqual(env.actions,['click [404]','send_msg_to_user [42]'])
        rollout=json.loads((self.folder/'rollout.json').read_text())
        self.assertEqual([row['action'] for row in rollout[0]['proposal']['candidates']],actions)
        self.assertEqual(rollout[0]['selected_index'],0)
        self.assertIn('captured viewport bounds',rollout[0]['execution_error'])
        display=json.loads(rollout[0]['selection_trace']['selected_grounded_action'])
        self.assertEqual(display['native_action'],'click [404]');self.assertEqual(display['target_status'],'unavailable')
        self.assertEqual(sum(row['purpose']=='policy' for row in request.requests),1)
        self.assertEqual(sum(row['purpose']=='selector_piotr' for row in request.requests),1)

    def test_actual_native_fuzzy_hook_uses_marked_broker_and_explicit_model_gate(self):
        from openwebrl.webprm_broker import payload_sha256
        from openwebrl.webprm_webarena_fuzzy import MODEL, NATIVE_PROTOCOL, REPLACEMENT_MODEL, REPLACEMENT_PROTOCOL
        raw=wrapped().native_config();raw['eval']['reference_answers']={'fuzzy_match':['42']}
        task=TaskEnvelope.from_release(raw,{})
        config=dict(self.config(),fuzzy_enabled=True,fuzzy_model=MODEL,fuzzy_protocol=NATIVE_PROTOCOL,
            fuzzy_max_input_tokens=8192,fuzzy_max_calls=1,broker_socket=str(self.folder/'broker.sock'),
            identity_sha256='a'*64)
        calls=[]
        def client(call_config,payload,journal,**kwargs):
            calls.append(deepcopy(payload))
            self.assertEqual(kwargs['purpose'],'native_fuzzy')
            usage={'prompt_tokens':120,'completion_tokens':1}
            response=dict(model=payload['model'],usage=usage,
                choices=[dict(finish_reason='stop',message={'content':'correct'})])
            response['_webprm_broker']=dict(schema_version=1,budget_owner='webprm_worker.completion',
                model=payload['model'],purpose='native_fuzzy',local=False,reservation_id=0,reservation_state='settled',
                usage=usage,identity_sha256='a'*64,journal_sha256='b'*64,budget_ledger_sha256='c'*64,
                policy_sha256='d'*64,payload_sha256=payload_sha256(payload))
            return response
        client._webprm_broker_client=True;client._webprm_budget_owner='webprm_worker.completion'
        with patch('openwebrl.webprm_broker.make_broker_client',return_value=client):
            request=broker_request(config,task,'first5')
        native=episode_evaluator(task,config,request,self.folder)
        require_supported(task,'first5',config)
        result=native.evaluate(task.native_config(),[{'observation':{'text':'state'},'info':{}},
            {'action_type':'send_msg_to_user','answer':'42'}])
        self.assertTrue(result['valid']);self.assertTrue(result['success'])
        self.assertEqual(result['model_provider_calls'],1)
        self.assertEqual(calls[0]['model'],MODEL);self.assertNotIn('context_length',calls[0])
        for changes in [dict(fuzzy_enabled=False),dict(fuzzy_model=REPLACEMENT_MODEL,fuzzy_protocol=REPLACEMENT_PROTOCOL)]:
            with self.assertRaises(EvaluationNotReady):require_supported(task,'first5',dict(config,**changes))
        self.assertEqual(len(calls),1)

    def test_piotr_malformed_pool_retains_raw_proposal_and_never_calls_selector(self):
        config=self.config('piotr_selectionarm',turns=1);request=Scripted(count=0)
        env=FakeEnv(self.folder)
        with self.assertRaisesRegex(EvaluationNotReady,'five original admissible'):
            run_loop(wrapped('url_match'),'piotr_selectionarm',config,env,
                ViewerActor(config,'unused',request=request),
                SimpleNamespace(select=lambda *a,**kw:self.fail('No Piotr call on malformed pool')),
                lambda e,o:dict(text='state',url=e.page.url),lambda *a,**kw:self.fail('No score'),self.folder)
        self.assertEqual(env.actions,[])
        proposal=json.loads((self.folder/'proposal-00.json').read_text())
        self.assertEqual(proposal['raw_blocks'],[])
        self.assertEqual(len(proposal['rejected_slots']),5)
        self.assertEqual((self.folder/'preaction-00.png').read_bytes(),PNG)

    def test_native_url_and_html_score_real_final_state_at_turn_limit(self):
        for kind in ['url_match','program_html']:
            result,env,request,folder=self.run_case(kind=kind,turns=1)
            self.assertTrue(result['valid']); self.assertEqual(result['success'],1)
            self.assertEqual(result['terminal']['termination'],'turn_limit')
            trajectory=json.loads((folder/'native-evaluation/trajectory.json').read_text())
            self.assertEqual(len(trajectory),3)
            self.assertEqual(trajectory[-1]['observation']['url'],env.page.url)

    def test_close_failure_preserves_native_result_in_watchdog_cell_location(self):
        result,_,_,folder=self.run_case(fail_close=True)
        self.assertTrue(result['valid']); self.assertEqual(result['success'],1)
        self.assertEqual(result['cleanup_status'],'error')
        self.assertTrue(json.loads((folder/'result.json').read_text())['valid'])

    def test_terminal_dispatch_failure_does_not_fabricate_answer(self):
        config=self.config();request=Scripted(terminal_after=0)
        env=FakeEnv(self.folder,terminal_error=True)
        with self.assertRaisesRegex(EvaluationNotReady,'did not execute'):
            run_loop(wrapped(),'first5',config,env,ViewerActor(config,'unused',request=request),
                ViewerSelector('first5',config,'unused',request=request),
                lambda e,o:dict(text='state',url=e.page.url),lambda *a,**k:self.fail('No score'),self.folder)
        self.assertFalse((self.folder/'native-evaluation').exists())

    def test_fixed_draw_shortfall_is_not_resampled(self):
        request=Scripted(count=4);config=self.config()
        actor=ViewerActor(config,'unused',request=request)
        proposal=actor.propose(wrapped().actor_view,dict(text='state',url='http://127.0.0.1'),[])
        self.assertEqual(len(proposal['candidates']),4)
        self.assertEqual(proposal['requested_raw_slots'],5)
        self.assertEqual(proposal['raw_slot_count'],4)
        self.assertFalse(proposal['exact_five_raw_slots'])
        self.assertEqual(proposal['rejected_slots'][0]['raw_slot'],4)
        self.assertEqual([r['purpose'] for r in request.requests],['terminal','policy'])

    def test_malformed_slot_preserves_other_original_indices(self):
        request=Scripted();config=self.config()
        def one_bad(*args,**kwargs):
            response=request(*args,**kwargs)
            if kwargs.get('purpose')=='policy':
                blocks=json.loads(response['choices'][0]['message']['content'])
                blocks['BLOCKS'][0]['action']='unsupported [1]'
                response['choices'][0]['message']['content']=json.dumps(blocks)
            return response
        actor=ViewerActor(config,'unused',request=one_bad)
        proposal=actor.propose(wrapped().actor_view,dict(text='state',url='http://127.0.0.1'),[])
        self.assertEqual([row['raw_slot'] for row in proposal['candidates']],[1,2,3,4])
        self.assertEqual(proposal['rejected_slots'][0]['raw_slot'],0)
        selected,_=ViewerSelector('first5',config,'unused').select(proposal,0)
        self.assertEqual(proposal['candidates'][selected]['action'],'goto [http://127.0.0.1/goal1]')

    def test_native_parser_drops_do_not_redefine_raw_slot_count(self):
        request=Scripted();config=self.config()
        def malformed(*args,**kwargs):
            response=request(*args,**kwargs)
            if kwargs.get('purpose')=='policy':
                blocks=json.loads(response['choices'][0]['message']['content'])
                del blocks['BLOCKS'][0]['thought']
                blocks['BLOCKS'][1]=None
                blocks['BLOCKS'][2]['action']=17
                response['choices'][0]['message']['content']=json.dumps(blocks)
            return response
        proposal=ViewerActor(config,'unused',request=malformed).propose(
            wrapped().actor_view,dict(text='state',url='http://127.0.0.1'),[])
        self.assertEqual(proposal['raw_slot_count'],5)
        self.assertTrue(proposal['exact_five_raw_slots'])
        self.assertEqual([row['raw_slot'] for row in proposal['candidates']],[3,4])
        self.assertEqual([row['raw_slot'] for row in proposal['rejected_slots']],[0,1,2])
        self.assertIsNone(proposal['raw_blocks'][1])
        self.assertEqual(len(proposal['raw_blocks']),5)

    def test_empty_pool_consumes_turn_without_invented_native_action(self):
        config=self.config(turns=1);request=Scripted(count=0)
        env=FakeEnv(self.folder)
        actor=ViewerActor(config,'unused',request=request)
        result=run_loop(wrapped('url_match'),'first5',config,env,actor,
            ViewerSelector('first5',config,'unused'),lambda e,o:dict(text='state',url=e.page.url),
            load_native_evaluator(SOURCE).evaluate,self.folder)
        self.assertEqual(env.actions,[])
        self.assertEqual(result['turns'],1)
        self.assertTrue(result['valid']);self.assertEqual(result['success'],0)
        self.assertEqual(len(json.loads((self.folder/'native-evaluation/trajectory.json').read_text())),1)

    def test_second_actor_snapshot_is_used_by_every_auxiliary_and_policy_call(self):
        request=Scripted();config=self.config();config['actor_model']=ACTOR_SNAPSHOTS[1]
        actor=ViewerActor(config,'unused',request=request)
        actor.propose(wrapped().actor_view,dict(text='state',url='http://127.0.0.1'),[])
        actor.propose(wrapped().actor_view,dict(text='new state',url='http://127.0.0.1/goal'),
                      [dict(thought='reason',action='goto [http://127.0.0.1/goal]')])
        self.assertTrue(all(row['payload']['model']==ACTOR_SNAPSHOTS[1] for row in request.requests))

    def test_unsupported_conditions_and_fuzzy_tasks_remain_blocked(self):
        for selector in ['our_prm','our_arm','unregistered_selector']:
            with self.assertRaises(EvaluationNotReady):require_supported(wrapped(),selector)
        raw=wrapped().native_config();raw['eval']['reference_answers']={'fuzzy_match':['42']}
        with self.assertRaisesRegex(EvaluationNotReady,'GPT-4-1106-preview'):
            require_supported(TaskEnvelope.from_release(raw,{}),'first5')

    def test_online_reward_uses_released_builder_and_state_diff_not_static_template(self):
        reward=load_viewer_reward(SOURCE)
        system,user=reward.build(intent='x',observation='tree',current_url='http://now',
            trajectory='History '*20,state_diff='Actual changed field',domain_knowledge=None,
            candidate_a_thought='a',candidate_a_action='click [1]',candidate_b_thought='b',candidate_b_action='click [2]')
        self.assertIn('Learning from Failures',system)
        self.assertIn('Actual changed field',user)
        self.assertIn('viewer/src/prompts/reward_p.py',reward.manifest['files'])

    def test_actual_state_claim_blocks_copying_a_lease_to_reuse_reset(self):
        state=self.folder/'state';state.mkdir()
        instance=dict(scope='original',state_path=str(state),endpoint='http://127.0.0.1',state_provenance_sha256='a'*64)
        instance_path=self.folder/'instance.json';instance_path.write_text(json.dumps(instance))
        lease=dict(task_id=wrapped().execution_id,actor_model=ACTOR_SNAPSHOTS[0],selector='first5',
            instance_receipt=str(instance_path),instance_receipt_sha256=hashlib.sha256(instance_path.read_bytes()).hexdigest())
        config=self.config();config['endpoints']={'reddit':'http://127.0.0.1'}
        with patch('scripts.check_webprm_reddit_reset.validate_instance'):
            for index in range(2):
                path=self.folder/f'lease{index}.json';path.write_text(json.dumps(lease))
                config.update(lease_path=str(path),lease_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                if index==0:claim_lease(wrapped(),'first5',config)
                else:
                    with self.assertRaises(FileExistsError):claim_lease(wrapped(),'first5',config)

    def test_dynamic_accounting_uses_both_actor_blocks_and_missing_is_not_failure(self):
        plan=dict(task_ids=[5,14],actors=list(ACTOR_SNAPSHOTS),selectors=['no_search','arbiter7b','piotr_selectionarm'])
        rows=[dict(task_id=task_id,arm=arm,valid=True,success=0,attempted=True)
              for task_id,arm in sorted(collector.expected_cells(plan))]
        self.assertEqual(collector.cohort_status(rows,plan)['planned_cells'],12)
        self.assertEqual(collector.cohort_status(rows,plan)['status'],'completed')
        status=collector.cohort_status(rows[:-1],plan)
        self.assertEqual(status['status'],'partial');self.assertEqual(len(status['missing_cells']),1)
        with self.assertRaises(ValueError):collector.cohort_status(rows+[rows[0]],plan)
        with self.assertRaises(ValueError):collector.cohort_status([dict(rows[0],attempted=False)],plan)

    def test_execution_rejects_nonfinite_budgets_and_protected_symlink_before_worker(self):
        config=dict(self.config(),max_turns=30,authorization_status='approved',job_id='synthetic-job',
            api_usd_cap=1.,prices_usd_per_token={ACTOR_SNAPSHOTS[0]:[1e-6,2e-6]},
            output=str(self.folder),budget_path=str(self.folder/'budget.json'),
            lease_path=str(self.folder/'lease.json'),broker_socket=str(self.folder/'broker.sock'))
        with patch.dict('os.environ',{'SLURM_JOB_ID':'synthetic-job'}):
            validate_execution_config(config)
            critic_config=dict(config,selector='arbiter7b',arbiter_endpoint=None)
            with self.assertRaisesRegex(EvaluationNotReady,'critic model'):
                validate_execution_config(critic_config)
            for field,value in [('api_usd_cap',float('nan')),('deadline',float('inf')),
                                ('api_usd_cap',0),('max_turns',True),('authorization_status','pending')]:
                with self.subTest(field=field,value=value),self.assertRaises(EvaluationNotReady):
                    validate_execution_config(dict(config,**{field:value}))
            unsafe=self.folder/'protected';unsafe.symlink_to('/gpfs/home/zixianma')
            for field in ['output','budget_path','lease_path','broker_socket']:
                with self.subTest(field=field),self.assertRaises(ValueError):
                    validate_execution_config(dict(config,**{field:str(unsafe/'never-created')}))

    def test_broker_wrapper_forwards_only_the_assigned_cell_and_marks_local_phase(self):
        config=dict(self.config(),broker_socket=str(self.folder/'broker.sock'))
        calls=[]
        with patch('openwebrl.webprm_broker.make_broker_client',return_value=lambda *a,**kw:calls.append((a,kw))) as factory:
            request=broker_request(config,wrapped(),'first5')
            request(config,{'model':ACTOR_SNAPSHOTS[0]},self.folder/'requests.jsonl',purpose='policy')
            self.assertEqual(factory.call_args.kwargs['cell_id'],wrapped().execution_id+'/'+config['cell_name'])
            self.assertEqual(len(calls),1)
        with self.assertRaisesRegex(EvaluationNotReady,'broker'):
            broker_request(self.config(),wrapped(),'first5')


if __name__ == '__main__': unittest.main()
