"""CPU-only adverse transport/provenance tests; zero browser/GPU/provider calls."""
import asyncio
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from openwebrl.action_credit import digest
import openwebrl.action_credit_eval_executor as executor
from openwebrl.action_credit_evaluation import _turn_actor_provenance, evaluation_protocol
from openwebrl.action_credit_judge import JudgeHTTPProtocol


RAW = dict(text='x', meta_info=dict(output_token_logprobs=[[-.2,7]], finish_reason=dict(type='stop')))


class FakeStore:
    """Transport crash-fixture only; persistent real-ledger tests are separate."""
    def __init__(self):
        self.state = dict(online_actor_requests={}, subrequests={})
    def snapshot(self): return deepcopy(self.state)
    def reserve_evaluation_actor_request(self,key,**identity):
        if key in self.state['online_actor_requests']: return dict(created=False)
        self.state['online_actor_requests'][key] = dict(identity,status='reserved')
        return dict(created=True)
    def settle_evaluation_actor_request(self,key,**completion):
        self.state['online_actor_requests'][key].update(completion)


class FakeHTTP:
    def __init__(self,result=None,error=None):
        self.result,self.error,self.calls,self.options=result,error,[],None
    def factory(self,**kwargs): self.options=kwargs; return self
    async def __aenter__(self): return self
    async def __aexit__(self,*args): pass
    async def post(self,url,json):
        self.calls.append((url,json))
        if self.error: raise self.error
        return SimpleNamespace(raise_for_status=lambda:None,json=lambda:deepcopy(self.result))


class ExecutorTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Retain fixture artifacts under /tmp; never delete experiment artifacts.
        self.root=Path(tempfile.mkdtemp(prefix='action-credit-eval-executor-',dir='/tmp'))
        self.store=FakeStore()
        self.journal=executor.TransportJournal(self.root/'transport')
        self.actor=executor.EvaluationActor(store=self.store,request=dict(task_id='task',policy_sha256='a'*64),
            attempt_id='attempt',endpoint='http://127.0.0.1:8000/generate',version='1',journal=self.journal,
            deadline_unix=time.time()+120)
        self.kwargs=dict(url=self.actor.endpoint,input_text='prompt',sampling_params=dict(temperature=0.,
            top_p=1.,top_k=1,max_new_tokens=4096),images=['image-base64'],task_id='task',turn=0,timeout=30.)

    async def call(self,http,versions=None):
        with patch('httpx.AsyncClient',side_effect=http.factory), patch.object(executor,'served_weight_version',
                side_effect=AsyncMock(side_effect=versions or ['1','1'])):
            return await self.actor(**self.kwargs)

    async def test_received_is_persisted_and_settled_before_normalization(self):
        http=FakeHTTP(result=RAW)
        result,metadata=await self.call(http)
        self.assertEqual(len(http.calls),1)
        self.assertFalse(http.options['follow_redirects']); self.assertFalse(http.options['trust_env'])
        self.assertEqual(result[1],[7,151645,198])
        key=metadata['evaluation_actor_request_id']
        row=self.journal.actor[key]
        self.assertEqual(executor._json(row['request']),http.calls[0][1])
        self.assertEqual(executor._json(row['response']),RAW)
        self.assertEqual(self.store.snapshot()['online_actor_requests'][key]['status'],'received')

    async def test_actor_timeout_retains_unknown_charge_no_retry(self):
        http=FakeHTTP(error=TimeoutError('CPU test timeout'))
        with self.assertRaises(TimeoutError):await self.call(http)
        self.assertEqual(len(http.calls),1)
        row=next(iter(self.store.snapshot()['online_actor_requests'].values()))
        self.assertEqual(row['status'],'unknown')
        self.assertEqual(len(self.journal.actor),1)
        self.assertIn('error',next(iter(self.journal.actor.values())))
        with self.assertRaisesRegex(ValueError,'never redispatch'):await self.call(http)
        self.assertEqual(len(http.calls),1)

    async def test_cancelled_http_retains_unknown_and_no_implicit_retry(self):
        http=FakeHTTP(error=asyncio.CancelledError())
        with self.assertRaises(asyncio.CancelledError):await self.call(http)
        call=next(iter(self.store.snapshot()['online_actor_requests'].values()))
        self.assertEqual(call['status'],'unknown')
        self.assertEqual(call['reason'],'CancelledError')

    async def test_malformed_received_payload_is_retained_not_refunded(self):
        http=FakeHTTP(result=dict(text='x'))
        with self.assertRaises(ValueError):await self.call(http)
        row=next(iter(self.store.snapshot()['online_actor_requests'].values()))
        self.assertEqual(row['status'],'received')
        self.assertIn('response',next(iter(self.journal.actor.values())))

    async def test_policy_changes_after_http_fail_closed_preserving_response(self):
        http=FakeHTTP(result=RAW)
        with self.assertRaisesRegex(ValueError,'during dispatch'):await self.call(http,['1','2'])
        self.assertEqual(len(http.calls),1)
        self.assertEqual(next(iter(self.store.snapshot()['online_actor_requests'].values()))['status'],'received')

    async def test_wrong_sampling_deadline_and_policy_prevent_dispatch(self):
        http=FakeHTTP(result=RAW)
        self.kwargs['sampling_params']['temperature']=.6
        with self.assertRaisesRegex(ValueError,'sampling'):await self.call(http)
        self.kwargs['sampling_params']['temperature']=0.
        self.actor.deadline=time.time()-1
        with self.assertRaisesRegex(ValueError,'deadline'):await self.call(http)
        self.actor.deadline=time.time()+120
        with self.assertRaisesRegex(ValueError,'before dispatch'):await self.call(http,['2'])
        self.assertEqual(http.calls,[]);self.assertEqual(self.store.snapshot()['online_actor_requests'],{})

    async def test_request_artifact_failure_cannot_dispatch_or_refund(self):
        http=FakeHTTP(result=RAW)
        with patch.object(executor,'saved_json',side_effect=OSError('CPU full disk')):
            with self.assertRaises(OSError):await self.call(http)
        self.assertEqual(http.calls,[])
        self.assertEqual(next(iter(self.store.snapshot()['online_actor_requests'].values()))['status'],'reserved')
        with self.assertRaisesRegex(ValueError,'never redispatch'):await self.call(http)
        self.assertEqual(http.calls,[])

    async def test_response_artifact_failure_preserves_unknown_liability(self):
        http=FakeHTTP(result=RAW);original=executor.saved_json
        def save(path,value):
            if path.name=='response.json':raise OSError('CPU full disk')
            return original(path,value)
        with patch.object(executor,'saved_json',side_effect=save):
            with self.assertRaises(OSError):await self.call(http)
        self.assertEqual(len(http.calls),1)
        self.assertEqual(next(iter(self.store.snapshot()['online_actor_requests'].values()))['status'],'unknown')

    def test_endpoints_reject_redirect_auth_and_remote_protocols(self):
        for url in ['https://host:12/generate','http://x:1/generate?foo=1','http://u:p@x:1/generate','http://x/generate','http://x:1/other']:
            with self.subTest(url=url), self.assertRaises(ValueError):executor._endpoint(url)

    def test_sampler_native_suffix_and_malformed_output(self):
        result=executor.native_actor_output(RAW);self.assertEqual(result,('x<|im_end|>\n',[7,151645,198],[-.2,0.,0.],'stop'))
        end=deepcopy(RAW);end['meta_info']['output_token_logprobs']=[[-.1,151645]]
        self.assertEqual(executor.native_actor_output(end)[1],[151645,198])
        for values in [[],[[float('nan'),7]],[[.5,7]],[[-1.,True]],[[-1.,7]]*4097]:
            raw=deepcopy(RAW);raw['meta_info']['output_token_logprobs']=values
            with self.assertRaises(ValueError):executor.native_actor_output(raw)
        raw=deepcopy(RAW);raw['meta_info']['finish_reason']['type']='abort'
        with self.assertRaises(ValueError):executor.native_actor_output(raw)

    def test_every_inherited_baseline_context_control_is_pinned(self):
        controls=executor.native_evaluation_controls()
        executor.validate_native_evaluation_controls(controls)
        self.assertEqual(controls['context_num_screenshots'],1)
        self.assertEqual(controls['browser_history_env_feedback_max_chars'],300)
        self.assertEqual(controls['browser_tool_response_max_chars'],512)
        self.assertEqual(controls['rollout_max_context_len'],32768)
        self.assertEqual(controls['max_consecutive_parse_failures'],3)
        self.assertIsNone(controls['serial_action_variant'])
        self.assertEqual(evaluation_protocol()['native_generation_controls'],controls)
        for key,value in controls.items():
            changed=dict(controls);changed[key]='mutated'
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'history/context'):
                executor.validate_native_evaluation_controls(changed)
            omitted=dict(controls);omitted.pop(key)
            with self.subTest(missing=key),self.assertRaises(ValueError):
                executor.validate_native_evaluation_controls(omitted)

    def test_transport_journal_preserves_failed_and_succeeded_http_attempts(self):
        for index in range(2):
            key='judge-'+str(index)
            self.journal('judge_http_reserved',dict(request_id=key,payload=dict(model='gpt-4.1',max_completion_tokens=1024)))
        self.journal('judge_http_error',dict(request_id='judge-0',error_type='TimeoutError'))
        self.journal('judge_http_response',dict(request_id='judge-1',response=dict(usage=dict(prompt_tokens=5,completion_tokens=2))))
        rows=self.journal.inventory()['http_attempts']
        self.assertEqual(len(rows),2);self.assertIn('error',rows[0]);self.assertIn('response',rows[1])
        self.assertEqual(executor._json(rows[1]['request'])['max_completion_tokens'],1024)

    def test_retained_json_is_immutable(self):
        ref=executor.saved_json(self.root/'once.json',dict(a=1))
        self.assertEqual(ref,executor.saved_json(self.root/'once.json',dict(a=1)))
        with self.assertRaises(ValueError):executor.saved_json(self.root/'once.json',dict(a=2))

    def placeholder(self):
        task_row=dict(prompt='CPU task',metadata=dict(task_id='task'))
        task=self.root/'tasks.jsonl';task.write_text(json.dumps(task_row)+'\n')
        source=executor.saved_json(self.root/'source.json',dict(files={'openwebrl/generate_browser.py':'b'*64}))
        request=executor.saved_json(self.root/'request.json',dict(task_manifest=executor.reference(task),source_inventory=source))
        row=dict(task_id='task',attempt_id='attempt',rollout=dict(path=str(self.root/'rollout.pt'),sha256='c'*64))
        turn=dict(response='',response_length=0,tokens=[],status='aborted',remove_sample=True,reward=None,
                  metadata=dict(terminate_reason='generation_error: CPU failure'))
        row['no_response']=executor.saved_json(self.root/'no-response.json',dict(task_id='task',attempt_id='attempt',
            rollout_sha256='c'*64,request_sha256=request['sha256'],reason=turn['metadata']['terminate_reason'],
            generation_source_sha256='b'*64,sample_sha256=digest(task_row)))
        return row,turn,request

    def test_native_no_response_placeholder_can_only_be_invalid_and_empty(self):
        row,turn,request=self.placeholder()
        _turn_actor_provenance(row,[turn],set(),request,{},dict(actor_attempts=[]))
        for key,value in [('tokens',[7]),('response','forged'),('response_length',1),('status','completed')]:
            changed=deepcopy(turn);changed[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                _turn_actor_provenance(row,[changed],set(),request,{},dict(actor_attempts=[]))

    def test_no_response_cannot_hide_received_response_or_another_task(self):
        row,turn,request=self.placeholder()
        with self.assertRaises(ValueError):_turn_actor_provenance(row,[turn],{('attempt',0)},request,{}, {})
        row['task_id']='forged'
        with self.assertRaises(ValueError):_turn_actor_provenance(row,[turn],set(),request,{}, {})

    def test_generated_turn_must_match_received_tokens_and_request_id(self):
        raw=executor.saved_json(self.root/'received.json',RAW)
        output=executor.native_actor_output(RAW)
        row=dict(attempt_id='attempt')
        turn=dict(response=output[0],response_length=len(output[1]),tokens=[123]+output[1],
            metadata=dict(turn_index=0,arm=dict(evaluation_actor_request_id='actor')))
        state=dict(online_actor_requests=dict(actor=dict(attempt_id='attempt',turn=0,status='received')))
        result=dict(actor_attempts=[dict(request_id='actor',response=raw)])
        _turn_actor_provenance(row,[turn],{('attempt',0)},None,state,result)
        for key,value in [('response','forged'),('tokens',[123,9,151645,198]),('response_length',1)]:
            changed=deepcopy(turn);changed[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                _turn_actor_provenance(row,[changed],{('attempt',0)},None,state,result)
        turn['metadata']['arm']['evaluation_actor_request_id']='other'
        with self.assertRaises(ValueError):_turn_actor_provenance(row,[turn],{('attempt',0)},None,state,result)


    async def test_native_format_failure_keeps_combined_penalty_and_binary_failure(self):
        from slime.utils.types import Sample
        from openwebrl import reward_browser
        turn=Sample(response='bad',status=Sample.Status.FAILED,metadata=dict(turn_index=0,
            is_last_turn=True,terminate_reason='format_error_failed'))
        args=SimpleNamespace(browser_response_format_mode='browser_env',judge_max_attached_imgs=0,
            context_num_screenshots=0,judge_prompt_variant='action_history',judge_api_model='gpt-4.1')
        client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock())))
        await executor._native_reward(args,[turn],client,self.journal,executor._hash(Path(reward_browser.__file__)))
        self.assertEqual(turn.reward,-1.)
        self.assertEqual(executor._native_status([turn.to_dict()]),(True,0.))
        client.chat.completions.create.assert_not_awaited()

    async def test_paid_dispatch_blocked_escapes_canonical_retry_without_invalid_result(self):
        from slime.utils.types import Sample
        from openwebrl import reward_browser
        from openwebrl.action_credit_judge import PaidDispatchBlocked
        turn=Sample(response='<think>done</think><tool_call>{"name":"done","arguments":{"response":"finished"}}</tool_call>',
            status=Sample.Status.COMPLETED,metadata=dict(turn_index=0,is_last_turn=True,intent='CPU task',
            terminate_reason='task_completed'))
        args=SimpleNamespace(browser_response_format_mode='browser_env',judge_max_attached_imgs=3,
            context_num_screenshots=1,judge_prompt_variant='action_history',judge_api_model='gpt-4.1',
            judge_timeout_secs=120.,browser_history_env_feedback_max_chars=300)
        create=AsyncMock(side_effect=PaidDispatchBlocked('CPU-only exact API authority refused'))
        client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with self.assertRaises(PaidDispatchBlocked):
            await executor._native_reward(args,[turn],client,self.journal,executor._hash(Path(reward_browser.__file__)))
        self.assertEqual(create.await_count,1)
        self.assertIsNone(turn.reward)
        self.assertNotIn('reward',turn.metadata)

    async def test_all8_restore_and_transfer_are_awaited_before_endpoints(self):
        import openwebrl.action_credit_driver as driver
        @dataclass
        class Contract:
            update_id:int=0
            policy_sha256:str='a'*64
        launch=SimpleNamespace(validate=lambda:None,deadline_unix=time.time()+120,initial_contract={},
            manifest=executor.saved_json(self.root/'pilot.json',{}),store_root=str(self.root/'pilot'),native_acceptance={})
        request=dict(iteration=0,policy_sha256='a'*64,checkpoint_inventory={'sha256':'a'*64},source_inventory={})
        trace=[]
        async def calls(handles,method,*args):
            self.assertEqual(len(handles),8);trace.append(method)
            if method=='action_credit_operation':return [{'sha256':'init'}]*8
            if method=='action_credit_driver_operation':return [dict(policy_sha256='a'*64,update_id=0)]*8
            return [None]*8
        async def url(index):
            self.assertEqual(trace, ['action_credit_operation','action_credit_driver_operation','update_weights'])
            return 'http://127.0.0.1:'+str(18000+index)
        engines=[SimpleNamespace(get_url=SimpleNamespace(remote=lambda index=i:url(index))) for i in range(4)]
        async def engine_refs():return engines,None,None,None,None
        context=dict(launch=launch,actor_model=object(),rollout_manager=SimpleNamespace(
            get_updatable_engines_and_lock=SimpleNamespace(remote=engine_refs)))
        pilot=SimpleNamespace(snapshot=lambda:dict(updates={}))
        with patch.object(driver,'_contract',return_value=Contract()), patch.object(driver,'_real_handlers',return_value=list(range(8))), \
             patch.object(driver,'_call_all',side_effect=calls), patch('openwebrl.action_credit_state.CampaignManifest.from_dict',return_value=None), \
             patch('openwebrl.action_credit_state.CampaignStore',return_value=pilot), \
             patch.object(executor,'served_weight_version',new=AsyncMock(return_value='1')):
            urls,version,proof=await executor._restore_and_serve(request,context,self.root)
        self.assertEqual(len(urls),4);self.assertEqual(version,'1')
        self.assertEqual(executor._json(proof)['transfer_rank_count'],8)
        # All proof data above is synthetic CPU RPC output, never production acceptance.

    async def test_initial_cannot_rewind_consumed_pilot_even_with_shared_billing(self):
        import openwebrl.action_credit_driver as driver
        @dataclass
        class Contract:
            update_id:int=0
            policy_sha256:str='a'*64
        launch=SimpleNamespace(validate=lambda:None,deadline_unix=time.time()+120,initial_contract={},
            manifest=executor.saved_json(self.root/'pilot.json',{}),store_root=str(self.root/'pilot'),native_acceptance={})
        request=dict(iteration=0,policy_sha256='a'*64)
        pilot=SimpleNamespace(snapshot=lambda:dict(updates={'0':dict(phase='consumed')}))
        rpc=AsyncMock()
        with patch.object(driver,'_contract',return_value=Contract()), patch.object(driver,'_real_handlers',return_value=list(range(8))), \
             patch.object(driver,'_call_all',new=rpc), patch('openwebrl.action_credit_state.CampaignManifest.from_dict',return_value=None), \
             patch('openwebrl.action_credit_state.CampaignStore',return_value=pilot):
            with self.assertRaisesRegex(ValueError,'rewind'):
                await executor._restore_and_serve(request,dict(launch=launch,actor_model=object()),self.root)
        rpc.assert_not_awaited()

    def test_wandb_receipt_only_after_owned_successful_finish(self):
        events=[]
        request=executor.saved_json(self.root/'request.json',dict(cohort_id='c'*64,protocol=evaluation_protocol(),
            policy_sha256='a'*64,iteration=0))
        run=SimpleNamespace(project='openwebrl-evals',id='ac-eval-'+('c'*20),
            config=dict(request_sha256=request['sha256']),summary={})
        run.log=lambda metrics:events.append('log')
        def finish(exit_code):
            self.assertFalse((self.root/'wandb.json').exists());events.append(('finish',exit_code))
        run.finish=finish
        fake=SimpleNamespace(init=lambda **kwargs:run)
        packet=dict(request=request,entity=None,metrics=dict(tasks=300),output=str(self.root/'wandb.json'),deadline_unix=time.time()+30)
        with patch.dict('sys.modules',{'wandb':fake}),patch.object(executor,'verify_worker_environment'):
            ref=executor.finish_wandb(packet)
        self.assertEqual(events,['log',('finish',0)])
        self.assertEqual(executor._json(ref)['state'],'finished')

    def test_wandb_tracking_error_never_writes_completion(self):
        request=executor.saved_json(self.root/'request.json',dict(cohort_id='c'*64,protocol=evaluation_protocol(),
            policy_sha256='a'*64,iteration=0))
        codes=[]
        run=SimpleNamespace(project='wrong-project',id='ac-eval-'+('c'*20),config={},finish=lambda exit_code:codes.append(exit_code))
        with patch.dict('sys.modules',{'wandb':SimpleNamespace(init=lambda **kwargs:run)}),patch.object(executor,'verify_worker_environment'):
            with self.assertRaises(ValueError):executor.finish_wandb(dict(request=request,entity=None,metrics={},
                output=str(self.root/'wandb.json'),deadline_unix=time.time()+30))
        self.assertEqual(codes,[1]);self.assertFalse((self.root/'wandb.json').exists())


if __name__=='__main__': unittest.main()
