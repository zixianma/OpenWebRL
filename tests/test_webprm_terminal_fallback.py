"""Offline native parity and saved-response regressions; no provider/browser I/O."""
import ast
import asyncio
from copy import deepcopy
import hashlib
import importlib.util
import json
import logging
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from openwebrl.webprm_arbiter_harness import load_arbiter_harness, native_execution_code
from openwebrl.webprm_webarena import load_native_evaluator
from openwebrl.webprm_webarena_worker import ViewerActor, ViewerSelector, run_loop
from openwebrl.webprm_webarena_piotr import (PiotrSelectionARMSelector, MODEL as PIOTR_MODEL, REVISION as PIOTR_REVISION,
    CHECKPOINT as PIOTR_CHECKPOINT, SOURCE as PIOTR_SOURCE, REPRESENTATION_V2)
from openwebrl.webprm_worker import arbiter_auxiliary
from scripts.audit_webprm_table5_cell import audit_rollout

SOURCE = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-baselines-20261009/sources/arbiter')
CELLS = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/'
             'reddit-table5-190-v3/campaign/attempt-356022/cells')
SAVED = {
    'click': ('ca64358ae423b3f6fce5e62b085403e95b2473d3f78f6ffd1d58ca9fa64bac5d',
              'gpt-4o-2024-05-13__piotr_selectionarm',
              'e6e12311dead233c51547284b290ea875fc8f8c9eb6efabfaad0c5219c67477b'),
    'malformed': ('09dba46e58df890ef6634e7835201f5f6dce2269e723b4160eb8c576c062f2fb',
                  'gpt-4o-2024-05-13__gpt4omini',
                  '6aad873d715c469d1942856d2186a8b021adccd6e045702c6397c8099bd5789b'),
}


def saved(kind):
    cell, name, checksum = SAVED[kind]
    path = CELLS / cell / 'episodes/webarena_lite_005' / name / 'api.jsonl'
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != checksum:
        raise AssertionError('Preserved primary journal changed')
    rows = [json.loads(line) for line in raw.splitlines()]
    assert rows[-2]['purpose'] == 'terminal'
    return rows[-2], rows[-1]['response']


def response(text):
    return dict(model='gpt-4o-2024-05-13', choices=[dict(finish_reason='stop',
                                                     message=dict(content=text))])


@unittest.skipUnless(SOURCE.is_dir(), 'Pinned native checkout unavailable')
class TerminalFallbackTests(unittest.TestCase):
    def setUp(self):
        self.net = patch('socket.socket.connect', side_effect=AssertionError('Network forbidden'))
        self.net.start(); self.addCleanup(self.net.stop)
        self.harness = load_arbiter_harness(SOURCE)
        self.config = dict(arbiter_source=str(SOURCE), actor_model='gpt-4o-2024-05-13',
                           selector='gpt4omini', seed=42, max_turns=2, deadline=time.time()+60,
                           reset_id='fixture', identity_sha256='fixture',
                           cell_name='gpt-4o-2024-05-13__gpt4omini')
        self.view = dict(intent='Synthetic visible task', start_url='http://127.0.0.1')
        self.context = dict(self.view, current_url='http://127.0.0.1', text_observation='fixture',
                            trajectory='')

    def auxiliary(self, reply):
        calls = []
        def request(config, payload, journal, **kwargs):
            calls.append(dict(payload=payload, **kwargs)); return deepcopy(reply)
        value = arbiter_auxiliary(self.harness, self.config, None, self.context, None, None, None,
                                  actor_model=self.config['actor_model'], request=request)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]['purpose'], 'terminal')
        self.assertEqual({k: calls[0]['payload'][k] for k in ('model','temperature','top_p','max_tokens')},
                         dict(model=self.config['actor_model'], temperature=1., top_p=.9, max_tokens=2048))
        return value[1]

    @unittest.skipUnless(CELLS.is_dir(), 'Preserved primary journals unavailable')
    def test_saved_malformed_response_preserves_native_none(self):
        _, reply = saved('malformed')
        self.assertIsNone(self.harness.parse_terminal_response(reply['choices'][0]['message']['content']).terminal_action)
        self.assertIsNone(self.auxiliary(reply))

    @unittest.skipUnless(CELLS.is_dir(), 'Preserved primary journals unavailable')
    def test_saved_click_response_is_an_override_not_an_error(self):
        _, reply = saved('click')
        block = self.harness.parse_terminal_response(reply['choices'][0]['message']['content']).terminal_action
        self.assertEqual(self.auxiliary(reply), dict(thought=block.thought, action='click [2871]', confidence=1.))
        self.assertEqual(self.harness.parse_action(block.action)['kind'], 'click')

    def test_good_terminal_and_null_controls_preserve_thought_and_confidence(self):
        for action in ('send_msg_to_user [42]', None):
            with self.subTest(action=action):
                value = dict(thought='Exact native thought', action=action, confidence=.6)
                self.assertEqual(self.auxiliary(response(json.dumps(value))), value)

    def test_incomplete_or_empty_provider_response_remains_invalid(self):
        for reply in (response(''), response('malformed')):
            if reply['choices'][0]['message']['content']:
                reply['choices'][0]['finish_reason'] = 'length'
            with self.subTest(reply=reply), self.assertRaises(ValueError):
                self.auxiliary(reply)

    def test_released_step_loop_selects_click_and_only_send_message_terminates(self):
        # Execute the unchanged native orchestration method with service doubles.
        # This is independent of ViewerActor's detector-bypass implementation.
        path = SOURCE / 'viewer/src/server/service/executors/step_loop_executor.py'
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'StepLoopExecutor')
        method = next(n for n in cls.body if isinstance(n, ast.AsyncFunctionDef) and n.name == '_execute_single_step')
        pure = ast.ClassDef(name='NativeStep', bases=[], keywords=[], body=[method], decorator_list=[])
        async def emit(event): pass
        ns = dict(asyncio=asyncio, time=time, logger=logging.getLogger(__name__),
                  SamplingCommand=SimpleNamespace, RerankAPIRequest=SimpleNamespace, Event=lambda *a, **kw: kw,
                  event_hub=SimpleNamespace(emit=emit))
        future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[future,pure], type_ignores=[])),str(path),'exec'),ns)
        for action in (None, 'click [2871]', 'send_msg_to_user [42]'):
            with self.subTest(action=action):
                calls = []; obj = ns['NativeStep']()
                async def detect(**kwargs):
                    return None if action is None else dict(action=action,thought='native',confidence=1.)
                async def sample(**kwargs):
                    return SimpleNamespace(candidates=[dict(action='click [1]')])
                async def rerank(**kwargs):
                    calls.append('rerank'); return SimpleNamespace(success=True,candidates=[dict(action='click [1]')])
                async def finish(*args): calls.append('terminal')
                def execute(**kwargs):
                    calls.append(kwargs['action'])
                    return SimpleNamespace(is_terminal=self.harness.parse_action(kwargs['action'])['kind']=='send_message_to_user',
                                           new_step=1,url='http://127.0.0.1')
                obj._detect_terminal_if_needed=detect; obj._handle_terminal_action=finish
                obj.run_context=SimpleNamespace(intent='task',domain_knowledge=None,policy_model=None)
                obj.state_preparator=SimpleNamespace(prepare_agent_inputs=lambda **kw: dict(
                    observation_text='tree',trajectory_text='',current_url='http://127.0.0.1',state_diff=''))
                obj.service=SimpleNamespace(sampling=sample,rerank=rerank,execute_action_and_append_state=execute)
                obj.terminal_handler=SimpleNamespace(extract_action=lambda value, **kw:value['selected_action'],
                                                     extract_action_string=lambda value:value['action'])
                result=asyncio.run(obj._execute_single_step('fixture',0,0,object(),None,time.time()))
                self.assertEqual(result['is_terminal'],action=='send_msg_to_user [42]')
                self.assertIn(action or 'click [1]',calls)
                self.assertEqual('rerank' in calls,action is None)

    def run_saved_episode(self, kind):
        # Reuse only the existing fake page/environment; actual worker stages,
        # native action conversion, native evaluator and rollout audit execute.
        spec=importlib.util.spec_from_file_location('worker_fixture',Path(__file__).with_name('test_webprm_webarena_worker.py'))
        fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
        tmp=tempfile.TemporaryDirectory(prefix='native-terminal-repair-',dir='/tmp')
        self.addCleanup(tmp.cleanup); folder=Path(tmp.name)
        first=saved(kind)[1]; calls=[]; terminals=0
        def request(config,payload,journal,**kwargs):
            nonlocal terminals
            calls.append(dict(payload=deepcopy(payload),**kwargs))
            if kwargs['purpose']=='terminal':
                terminals+=1
                return deepcopy(first) if terminals==1 else response(json.dumps(dict(thought='done',action='send_msg_to_user [42]',confidence=1.)))
            if kwargs['purpose']=='state_diff':return response('No changes')
            if kwargs['purpose']=='policy':
                return response(json.dumps({'BLOCKS':[dict(thought='same',action='click [1]',confidence=1.) for _ in range(5)]}))
            self.fail('Unexpected model/critic call')
        dispatch={}; harness=self.harness
        bounds={'2871':[10,20,30,40],'1':[50,60,30,40]}
        class Env(fixture.FakeEnv):
            def step(env,action):
                parsed=harness.parse_action(action)
                code=native_execution_code(SOURCE,parsed,bounds,key_mapping=harness.key_mapping)
                dispatch.update(action=action,element_bounds=deepcopy(bounds),code=code,
                                code_sha256=hashlib.sha256(code.encode()).hexdigest())
                if parsed['kind']=='click':
                    env.events.append('step');env.actions.append(action)
                    return env.observation(),0,False,False,{}
                return super().step(action)
        selector_name='piotr_selectionarm' if kind=='click' else 'gpt4omini'
        config=dict(self.config,selector=selector_name,cell_name=self.config['actor_model']+'__'+selector_name)
        env=Env(folder);actor=ViewerActor(config,None,request=request)
        selector=object.__new__(ViewerSelector);selector.name=selector_name
        # Real offline Piotr adapter: grounds the fallback click for later history (no model call).
        selector.piotr=None
        if selector_name=='piotr_selectionarm':
            selector.piotr=PiotrSelectionARMSelector(dict(config,piotr_model=PIOTR_MODEL,piotr_revision=PIOTR_REVISION,
                piotr_checkpoint=str(PIOTR_CHECKPOINT),piotr_source=str(PIOTR_SOURCE),piotr_representation=REPRESENTATION_V2),None)
        # Exercise the real bypass before uninitialized critic state can be read.
        # The malformed control uses normal original-candidate selection.
        if kind=='malformed':selector=SimpleNamespace(select=lambda proposal,turn:(0,[]))
        result=run_loop(fixture.wrapped(),selector_name,config,env,actor,selector,
            lambda e,o:dict(text='tree',url=e.page.url,element_bounds=deepcopy(bounds),
                           viewport=dict(width=1280,height=720)),
            load_native_evaluator(SOURCE).evaluate,folder,dispatch_evidence=dispatch)
        rollout=json.loads((folder/'rollout.json').read_text())
        trajectory=json.loads((folder/'native-evaluation/trajectory.json').read_text())
        audit_rollout(config,trajectory,rollout,result)
        return result,rollout,trajectory,env,calls

    @unittest.skipUnless(CELLS.is_dir(), 'Preserved primary journals unavailable')
    def test_saved_click_executes_then_real_terminal_and_full_rollout_audit_passes(self):
        result,rows,trajectory,env,calls=self.run_saved_episode('click')
        self.assertEqual(env.actions,['click [2871]','send_msg_to_user [42]'])
        self.assertEqual(result['turns'],2);self.assertTrue(result['valid']);self.assertEqual(result['success'],1)
        self.assertEqual(trajectory[1]['action_type'],'click')
        self.assertEqual(trajectory[-1]['action_type'],'send_msg_to_user')
        self.assertEqual([r['purpose'] for r in calls],['terminal','state_diff','terminal'])
        self.assertEqual(rows[0]['selection_trace'],[])
        self.assertIsNone(rows[0]['selection_evidence'])
        self.assertEqual(rows[0]['proposal']['requested_raw_slots'],0)

    @unittest.skipUnless(CELLS.is_dir(), 'Preserved primary journals unavailable')
    def test_saved_malformed_response_continues_five_fixed_policy_slots_and_audit_passes(self):
        result,rows,trajectory,env,calls=self.run_saved_episode('malformed')
        self.assertEqual(env.actions,['click [1]','send_msg_to_user [42]'])
        self.assertTrue(result['valid']);self.assertEqual(result['turns'],2)
        self.assertIsNone(rows[0]['proposal']['native_terminal'])
        self.assertFalse(rows[0]['proposal']['terminal'])
        self.assertEqual(rows[0]['proposal']['requested_raw_slots'],5)
        self.assertEqual([r['purpose'] for r in calls],['terminal','policy','state_diff','terminal'])
        self.assertEqual(len(rows[0]['proposal']['candidates']),5)


if __name__=='__main__':unittest.main()
