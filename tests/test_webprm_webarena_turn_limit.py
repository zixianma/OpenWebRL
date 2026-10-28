"""Isolated proposal tests; all network connections forbidden."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from copy import deepcopy
from unittest.mock import patch

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
import openwebrl.webprm_webarena_worker as w
spec=importlib.util.spec_from_file_location('original_worker_tests',REPO/'tests/test_webprm_webarena_worker.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
from openwebrl.webprm_broker import payload_sha256
from openwebrl.webprm_webarena_fuzzy import MODEL,NATIVE_PROTOCOL

class Env(base.FakeEnv):
    def __init__(self,*args,mode='normal',**kwargs):
        super().__init__(*args,**kwargs);self.mode=mode;self.nonterminal=0
    def step(self,action):
        if action.startswith('send_msg_to_user'):
            self.events.append('step');self.actions.append(action)
            if self.mode=='raise_terminal':raise RuntimeError('fatal dispatch failure')
            if self.mode!='missing_chat':
                self.messages.append(dict(role='assistant',message=action.split('[',1)[1][:-1]))
            return self.observation(),0,True,False,{}
        self.nonterminal+=1
        if self.mode=='fatal_action':raise RuntimeError('fatal action failure')
        obs,*rest=super().step(action)
        if self.mode=='recoverable_error' and self.nonterminal==1:
            obs['last_action_error']='Unknown key: 8714'
        if self.mode=='early_truncation':rest[-2]=True
        if self.mode=='early_stop':rest[-3]=True
        return (obs,*rest)

class FallbackTests(unittest.TestCase):
    config=base.WorkerTests.config
    def setUp(self):
        self.network=patch('socket.socket.connect',side_effect=AssertionError('Network forbidden'))
        self.network.start();self.addCleanup(self.network.stop)
        self.temp=tempfile.TemporaryDirectory(prefix='wa-fallback-',dir='/tmp')
        self.addCleanup(self.temp.cleanup);self.folder=Path(self.temp.name)
    def run_case(self,*,mode='normal',actual_terminal=False,strict=False,expired=False,actor_failure=False):
        config=dict(self.config(turns=30),fuzzy_enabled=True,fuzzy_model=MODEL,
            fuzzy_protocol=NATIVE_PROTOCOL,fuzzy_max_input_tokens=8192,fuzzy_max_calls=1,
            identity_sha256='a'*64)
        if not strict:config['turn_limit_protocol']=w.NA_TURN_LIMIT
        raw=base.wrapped().native_config();raw['eval']['reference_answers']={'fuzzy_match':['42']}
        task=base.TaskEnvelope.from_release(raw,{})
        env=Env(self.folder,mode=mode)
        request=base.Scripted(terminal_after=29 if actual_terminal else 100)
        calls=[]
        def client(call_config,payload,journal,**kwargs):
            calls.append(deepcopy(payload))
            usage=dict(prompt_tokens=120,completion_tokens=1)
            return dict(model=payload['model'],usage=usage,
                choices=[dict(finish_reason='stop',message={'content':'incorrect'})],
                _webprm_broker=dict(schema_version=1,budget_owner='webprm_worker.completion',
                    model=payload['model'],purpose='native_fuzzy',local=False,reservation_id=0,
                    reservation_state='settled',usage=usage,identity_sha256='a'*64,
                    journal_sha256='b'*64,budget_ledger_sha256='c'*64,policy_sha256='d'*64,
                    payload_sha256=payload_sha256(payload)))
        client._webprm_broker_client=True;client._webprm_budget_owner='webprm_worker.completion'
        native=w.episode_evaluator(task,config,client,self.folder)
        if expired:config['deadline']=time.time()-1
        actor=w.ViewerActor(config,'unused',request=request)
        if actor_failure:actor.propose=lambda *a: (_ for _ in ()).throw(ConnectionError('actor transport failed'))
        selector=w.ViewerSelector('first5',config,'unused',request=request)
        dispatch={};original_step=env.step
        def step(action):
            code=w.native_execution_code(base.SOURCE,actor.harness.parse_action(action),{},key_mapping=actor.harness.key_mapping)
            dispatch.update(action=action,code=code,code_sha256=hashlib.sha256(code.encode()).hexdigest(),element_bounds={})
            return original_step(action)
        env.step=step
        self.last=(env,request,calls,config)
        return w.run_loop(task,'first5',config,env,actor,selector,
            lambda e,o:dict(text='synthetic state',url=e.page.url),native.evaluate,self.folder,dispatch_evidence=dispatch)
    def test_normal_thirty_turns_executes_protocol_message_and_calls_native_judge_once(self):
        result=self.run_case();env,request,calls,_=self.last
        self.assertTrue(result['valid'],result);self.assertEqual(result['success'],0)
        self.assertEqual(result['turns'],30);self.assertEqual(len(env.actions),31)
        self.assertEqual(env.actions[-1],'send_msg_to_user [N/A]');self.assertEqual(len(calls),1)
        rollout=json.loads((self.folder/'rollout.json').read_text())
        self.assertEqual(len(rollout),30);self.assertEqual(len(list(self.folder.glob('proposal-*.json'))),30)
        self.assertEqual(sum(x['purpose']=='policy' for x in request.requests),30)
        fallback=json.loads((self.folder/'protocol-fallback.json').read_text())
        self.assertFalse(fallback['actor_selected']);self.assertEqual(fallback['status'],'executed')
        trajectory=json.loads((self.folder/'native-evaluation/trajectory.json').read_text())
        self.assertEqual(w.digest(trajectory[:-1]),fallback['pre_fallback_trajectory_sha256'])
        self.assertEqual(trajectory[-1]['origin'],'released_viewer_normal_turn_limit')
        self.assertEqual(trajectory[-1]['answer'],'N/A')
        self.assertIn('n/a',json.dumps(calls[0]['messages']).lower())
        self.assertEqual(env.events[-1],'close')
        from scripts.analyze_webprm_webarena_pilot import audit_terminal_action
        plan=deepcopy(self.last[3]);plan['protocol_contract']=w.protocol_definition(base.SOURCE,turn_limit=w.NA_TURN_LIMIT)
        plan['protocol']=plan['protocol_contract']['identity']
        audit_terminal_action(self.folder,result,plan,trajectory,rollout)
    def test_default_strict_contract_is_unchanged(self):
        result=self.run_case(strict=True)
        self.assertFalse(result['valid']);self.assertEqual(len(self.last[2]),0)
        self.assertFalse((self.folder/'protocol-fallback.json').exists())
        self.assertEqual(len(self.last[0].actions),30)
    def test_actual_answer_on_last_turn_preserved(self):
        result=self.run_case(actual_terminal=True)
        self.assertTrue(result['valid']);self.assertEqual(len(self.last[0].actions),30)
        self.assertEqual(self.last[0].actions[-1],'send_msg_to_user [42]')
        self.assertFalse((self.folder/'protocol-fallback.json').exists());self.assertEqual(len(self.last[2]),1)
    def test_prior_recoverable_unknown_key_error_still_reaches_normal_fallback(self):
        result=self.run_case(mode='recoverable_error')
        self.assertTrue(result['valid']);self.assertEqual(len(self.last[2]),1)
        rollout=json.loads((self.folder/'rollout.json').read_text())
        self.assertEqual(rollout[0]['execution_error'],'Unknown key: 8714')
        self.assertEqual(rollout[-1]['execution_error'],None)
        self.assertIn('Unknown key: 8714',json.dumps(self.last[1].requests))
    def test_failed_fallback_dispatch_has_no_verdict_and_durable_failure(self):
        with self.assertRaisesRegex(RuntimeError,'fatal dispatch'):self.run_case(mode='raise_terminal')
        self.assertEqual(len(self.last[2]),0)
        self.assertFalse((self.folder/'native-evaluation/native-verdict.json').exists())
        fallback=json.loads((self.folder/'protocol-fallback.json').read_text())
        self.assertEqual(fallback['status'],'failed')
        trajectory=json.loads((self.folder/'trajectory-progress.json').read_text())
        self.assertNotIn('answer',trajectory[-1])
    def test_missing_fresh_assistant_chat_cannot_claim_execution(self):
        with self.assertRaisesRegex(w.EvaluationNotReady,'actual literal answer'):self.run_case(mode='missing_chat')
        self.assertEqual(len(self.last[2]),0)
        self.assertEqual(json.loads((self.folder/'protocol-fallback.json').read_text())['status'],'failed')
    def test_fatal_action_exception_never_falls_back(self):
        with self.assertRaisesRegex(RuntimeError,'fatal action'):self.run_case(mode='fatal_action')
        self.assertFalse((self.folder/'protocol-fallback.json').exists());self.assertEqual(len(self.last[2]),0)
    def test_actor_transport_exception_never_falls_back(self):
        with self.assertRaisesRegex(ConnectionError,'actor transport'):self.run_case(actor_failure=True)
        self.assertFalse((self.folder/'protocol-fallback.json').exists());self.assertEqual(self.last[0].actions,[])
    def test_expired_deadline_never_falls_back(self):
        with self.assertRaises(TimeoutError):self.run_case(expired=True)
        self.assertFalse((self.folder/'protocol-fallback.json').exists());self.assertEqual(self.last[0].actions,[])
    def test_early_env_truncation_never_falls_back(self):
        result=self.run_case(mode='early_truncation')
        self.assertFalse(result['valid']);self.assertEqual(len(self.last[0].actions),1)
        self.assertFalse((self.folder/'protocol-fallback.json').exists());self.assertEqual(len(self.last[2]),0)
    def test_early_env_stop_never_falls_back(self):
        result=self.run_case(mode='early_stop')
        self.assertFalse(result['valid']);self.assertEqual(len(self.last[0].actions),1)
        self.assertFalse((self.folder/'protocol-fallback.json').exists());self.assertEqual(len(self.last[2]),0)
    def test_outer_episode_records_failed_protocol_dispatch_invalid_and_closes(self):
        task=base.wrapped();config=dict(self.config(turns=30),output=str(self.folder),
            authorization_status='approved',api_usd_cap=1.,code_sha256={},
            endpoints={'reddit':'http://127.0.0.1'},turn_limit_protocol=w.NA_TURN_LIMIT)
        folder=self.folder/task.execution_id/config['cell_name']
        env=Env(folder,mode='raise_terminal')
        def close():
            env.events.append('close')
            self.assertFalse(json.loads((folder/'result.json').read_text())['valid'])
        env.close=close
        module='openwebrl.webprm_webarena_worker.'
        from types import SimpleNamespace
        with patch(module+'broker_request',return_value=base.Scripted(terminal_after=100)), \
                patch(module+'claim_lease',return_value=({'scope':'synthetic'},'synthetic-reset')), \
                patch(module+'prepare_browser'), \
                patch(module+'authenticated_state',return_value={'cookies':[],'origins':[]}), \
                patch(module+'open_authenticated_browser',return_value=env), \
                patch(module+'load_native_observation_processor',return_value=SimpleNamespace(
                    observe=lambda page:('actual synthetic state',{}),manifest={'synthetic':True})):
            env.page.viewport_size={'width':1280,'height':720}
            result=w._run_episode({'released':task.native_config()},config['cell_name'],config)
        self.assertTrue(result['attempted']);self.assertFalse(result['valid'])
        self.assertEqual(result['error_type'],'RuntimeError')
        self.assertIn('fatal dispatch failure',result['error'])
        self.assertEqual(env.events.count('close'),1)
        self.assertEqual(json.loads((folder/'protocol-fallback.json').read_text())['status'],'failed')
        self.assertFalse((folder/'native-evaluation/native-verdict.json').exists())
    def test_collector_and_worker_validate_exact_v2_and_v3_contracts(self):
        plan_path=Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/reddit-pilot-campaign-v1/attempt-355187/worker-plan.json')
        plan=json.loads(plan_path.read_text())
        plan.update(authorization_status='proposal_only',api_usd_cap=0,deadline=time.time()+60,
            code_sha256={str(REPO/name):hashlib.sha256((REPO/name).read_bytes()).hexdigest()
                         for name in base.collector.REQUIRED_CODE})
        from openwebrl.webprm_webarena_piotr import REPRESENTATION_V2,validate_piotr_selectionarm
        for mode,descriptor in [(w.STRICT_TURN_LIMIT,None),(w.NA_TURN_LIMIT,None),(w.NA_TURN_LIMIT,REPRESENTATION_V2)]:
            proposed=deepcopy(plan)
            if mode==w.NA_TURN_LIMIT:proposed['turn_limit_protocol']=mode
            if descriptor is not None:
                proposed['piotr_representation']=descriptor
                proposed['piotr_manifest']=validate_piotr_selectionarm(proposed)
            contract=w.protocol_definition(base.SOURCE,turn_limit=mode,piotr_representation=descriptor)
            proposed.update(protocol=contract['identity'],protocol_contract=contract)
            self.assertEqual(len(base.collector.validate_plan(proposed)),1)
            execution=dict(proposed,authorization_status='approved',api_usd_cap=15,job_id='synthetic',
                actor_model=base.ACTOR_SNAPSHOTS[0],selector='first5',output=str(self.folder),
                budget_path=str(self.folder/'budget'),lease_path=str(self.folder/'lease'),
                broker_socket=str(self.folder/'broker'))
            with patch.dict('os.environ',{'SLURM_JOB_ID':'synthetic'}):
                w.validate_execution_config(execution)
                bad=dict(execution,protocol='incorrect-v2-v3-contract')
                with self.assertRaises(w.EvaluationNotReady):w.validate_execution_config(bad)
            with self.assertRaises(w.EvaluationNotReady):
                base.collector.validate_plan(dict(proposed,protocol='incorrect-v2-v3-contract'))

    def test_unknown_protocol_rejected_and_manifest_explicit(self):
        with self.assertRaises(w.EvaluationNotReady):w.turn_limit_protocol({'turn_limit_protocol':True})
        self.assertEqual(w.protocol_definition(base.SOURCE)['identity'],w.PROTOCOL)
        manifest=w.protocol_definition(base.SOURCE,turn_limit=w.NA_TURN_LIMIT)
        self.assertEqual(manifest['identity'],'webarena_table5_viewer_reconstruction_v3')
        self.assertEqual(manifest['turn_limit_protocol'],w.NA_TURN_LIMIT)

if __name__=='__main__':
    unittest.main()
