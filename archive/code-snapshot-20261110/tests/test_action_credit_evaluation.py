"""CPU synthetic monitor archives only; no actor/browser/judge/W&B calls."""
from copy import deepcopy
import asyncio
import fcntl
import json
from pathlib import Path
import tempfile
import time
from dataclasses import asdict
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import openwebrl.action_credit_evaluation as evaluation

import torch

from openwebrl.action_credit import digest
from openwebrl.action_credit_evaluation import Full300EvaluationOwner, evaluation_protocol, verify_full300,_ids,_actor_inventory,_native_status
from openwebrl.action_credit_native_controller import _hash,_json,_write
from openwebrl.action_credit_judge import JudgeHTTPProtocol
from openwebrl.action_credit_state import CampaignManifest,CampaignStore
from test_action_credit_state import manifest_data,sha,reserve


class EvaluationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory(prefix='action-credit-eval-',dir='/tmp')
        self.addCleanup(temp.cleanup);self.root=Path(temp.name)
        self.ids=[f'eval-{i}' for i in range(300)]
        self.tasks=_write(self.root/'tasks.json',dict(task_ids=self.ids))
        model=self.root/'model.bin';model.write_bytes(b'CPU fixture, no model')
        self.original=_write(self.root/'original.json',dict(files=[dict(path=str(model),sha256=_hash(model))]))
        data=manifest_data();data['initialization']['original_actor_sha256']=self.original['sha256']
        self.protocol=JudgeHTTPProtocol('gpt-4.1',sha('CPU approval only'),sha('CPU bound'),32,50,1_000_000,1_000_000)
        data['science']['judge']['http_protocol_sha256']=self.protocol.sha256
        data['science']['algorithm']['local_actor_requests_cap']=1000
        data['budget_caps']['browser_dispatches']=1000
        self.store=CampaignStore.create(self.root/'store',CampaignManifest.from_dict(data))
        repo=Path(__file__).resolve().parents[1]
        names=['openwebrl/'+name+'.py' for name in ('eval_monitor','reward_browser','generate_browser','rl_recipe','response_format','action_credit_evaluation')]
        self.source=_write(self.root/'sources.json',dict(source_root=str(repo),files={name:_hash(repo/name) for name in names}))
        self.calls=0
        self.paid_fixture=False
        self.prior_fixture=False
        outer=self
        class Executor:
            async def run_or_resume(self,store,request_ref,root,context):
                outer.calls+=1
                return outer.materialize(request_ref,root)
        self.owner=Full300EvaluationOwner(task_manifest_ref=self.tasks,original_actor_inventory_ref=self.original,
            source_inventory_ref=self.source,shared_initial_root=self.root/'shared',arm_output_root=self.root/'arm',
            verification_store=self.store,executor=Executor(),diagnostic_only=True)
        self.contract=SimpleNamespace(update_id=0,policy_sha256=self.original['sha256'],train_task_ids=('train-only',))

    def materialize(self,request_ref,root):
        request=_json(request_ref); rows=[];http_rows=[];prior_rows=[]
        for index,task in enumerate(self.ids):
            attempt='attempt-'+task
            if self.prior_fixture and index==0:
                prior_rows=self.materialize_prior(request_ref,root,task)
            self.store.start_evaluation_attempt(request['cohort_id'],task,attempt)
            valid=index<250
            reward=dict(format=0.,judge=0.,combined=-1.,valid=valid,judge_timeout=False,
                judge_text='Judge not run for terminate_reason=format_error_failed',judge_prompt_variant='action_history')
            request_ids=[];responses=[]
            if self.paid_fixture and index==0:
                reward.update(format=1.,judge=1.,combined=1.,judge_text='SUCCESS')
                payload=dict(model='gpt-4.1',messages=[dict(role='user',content='CPU fixture')],max_completion_tokens=32)
                payload_ref=_write(root/'http-request.json',payload)
                for suffix in ('unknown','succeeded'):
                    key='http-'+suffix;request_ids.append(key)
                    reserved=dict(api_requests=1,api_microdollars=50,gpu_seconds=0,browser_dispatches=0)
                    self.store.reserve_subrequest(key,parent_attempt_id=attempt,payload_sha256=digest(payload),reserved_cost=reserved)
                    if suffix=='unknown':
                        self.store.settle_subrequest(key,status='unknown',actual_cost=dict(reserved,api_microdollars=0),reason='CPU timeout')
                        http_rows.append(dict(request_id=key,request=payload_ref,error=_write(root/'http-error.json',dict(error='CPU timeout'))))
                    else:
                        response=dict(usage=dict(prompt_tokens=4,completion_tokens=2),choices=[dict(message=dict(content='SUCCESS'))])
                        ref=_write(root/'http-response.json',response);responses.append(ref)
                        self.store.settle_subrequest(key,status='succeeded',actual_cost=dict(reserved,api_microdollars=6),response_sha256=digest(response))
                        http_rows.append(dict(request_id=key,request=payload_ref,response=ref))
            success=self.paid_fixture and index==0
            text='<think>CPU fixture</think><tool_call>{"name":"done","arguments":{"response":"finished"}}</tool_call>' if success else 'CPU fixture'
            turns=[dict(metadata=dict(task_id=task,turn_index=0,terminate_reason='task_completed' if success else 'format_error_failed',reward=reward),
                status='completed' if success else 'failed',response=text,tokens=[1,2],response_length=1,remove_sample=not valid,reward=reward['combined'])]
            archive=root/(task+'.pt');torch.save(dict(task_id=task,turns=turns,error_type=None),archive)
            rollout=dict(path=str(archive),sha256=_hash(archive))
            record=_write(root/(task+'-record.json'),dict(task_id=task,rollout_file=archive.name,turns=1,error_type=None,
                judge_model='gpt-4.1',judge_prompt_variant='action_history',reward_metadata=reward,
                metrics=dict(trajectories=1,valid_trajectories=int(valid),invalid_trajectories=int(not valid),successes=int(success))))
            verdict=_write(root/(task+'-verdict.json'),dict(task_id=task,request_sha256=request_ref['sha256'],
                rollout_sha256=rollout['sha256'],judge_model='gpt-4.1',judge_prompt_variant='action_history',
                status='valid' if valid else 'invalid',reward=float(reward['judge']==1.) if valid else None,
                invalid_reason=None if valid else 'CPU invalid observation',http_responses=responses,http_request_ids=request_ids))
            outcome=dict(artifact_sha256=verdict['sha256'])
            if valid:outcome['reward']=float(reward['judge']==1.)
            self.store.finish_evaluation_attempt(attempt,status='valid' if valid else 'invalid',outcome=outcome,
                reason=None if valid else 'CPU invalid observation')
            rows.append(dict(task_id=task,attempt_id=attempt,record=record,rollout=rollout,verdict=verdict))
        log=root/'worker.log'; log.write_text('CPU mock restore; never physical acceptance\n')
        proof=_write(root/'restore.json',dict(verification_kind='cpu_mock',policy_sha256=request['policy_sha256'],
            checkpoint_inventory=request['checkpoint_inventory'],restored_on_gpu=True,source_inventory=self.source,
            worker_log=dict(path=str(log),sha256=_hash(log)),restore_line='CPU mock restore; never physical acceptance'))
        runtime=_write(root/'runtime.json',dict(verification_kind='cpu_mock',request_sha256=request_ref['sha256'],
            checkpoint_inventory=request['checkpoint_inventory'],policy_sha256=request['policy_sha256'],protocol=evaluation_protocol(),
            worker_returncode=0,launcher_returncode=0,optimizer_updates=0,physical_restore_review=proof))
        successes=int(self.paid_fixture)
        metrics=dict(tasks=300,successes=successes,valid=250,invalid=50,overall_rate=successes/300,valid_only_rate=successes/250)
        wandb=_write(root/'wandb.json',dict(project='openwebrl-evals',state='finished',run_id='CPU-mock',
            request_sha256=request_ref['sha256'],metrics=metrics))
        result=dict(verification_kind='cpu_mock',request_sha256=request_ref['sha256'],protocol=evaluation_protocol(),
            policy_sha256=request['policy_sha256'],checkpoint_inventory=request['checkpoint_inventory'],source_inventory=self.source,
            runtime=runtime,tasks=rows,prior_attempts=prior_rows,metrics=metrics,wandb=wandb,http_attempts=http_rows,judge_http_protocol=asdict(self.protocol))
        return _write(root/'result.json',result)

    def materialize_prior(self,request_ref,root,task):
        request=_json(request_ref);missing='prior-missing';invalid='prior-invalid';reason='CPU browser failed'
        self.store.start_evaluation_attempt(request['cohort_id'],task,missing)
        self.store.finish_evaluation_attempt(missing,status='missing',reason=reason)
        identity=dict(request_sha256=request_ref['sha256'],task_id=task,attempt_id=missing)
        log=root/'prior.log';log.write_text('CPU failure and cleanup evidence, no live worker\n')
        logref=dict(path=str(log),sha256=_hash(log))
        missingrow=dict(task_id=task,attempt_id=missing,partial_artifacts=[],
            failure=_write(root/'prior-failure.json',dict(identity,reason=reason,log=logref)),
            cleanup=_write(root/'prior-cleanup.json',dict(identity,status='complete',processes_alive=[],log=logref)))
        self.store.start_evaluation_attempt(request['cohort_id'],task,invalid)
        reward=dict(format=0.,judge=0.,combined=-1.,valid=False,judge_timeout=False,
            judge_text='Judge not run for terminate_reason=format_error_failed',judge_prompt_variant='action_history')
        turns=[dict(metadata=dict(task_id=task,turn_index=0,terminate_reason='format_error_failed',reward=reward),
            status='failed',response='CPU invalid',tokens=[1,2],response_length=1,remove_sample=True,reward=-1.)]
        archive=root/'prior.pt';torch.save(dict(task_id=task,turns=turns,error_type=None),archive)
        rollout=dict(path=str(archive),sha256=_hash(archive))
        record=_write(root/'prior-record.json',dict(task_id=task,rollout_file=archive.name,turns=1,error_type=None,
            judge_model='gpt-4.1',judge_prompt_variant='action_history',reward_metadata=reward,
            metrics=dict(trajectories=1,valid_trajectories=0,invalid_trajectories=1,successes=0)))
        verdict=_write(root/'prior-verdict.json',dict(task_id=task,request_sha256=request_ref['sha256'],
            rollout_sha256=rollout['sha256'],judge_model='gpt-4.1',judge_prompt_variant='action_history',
            status='invalid',reward=None,invalid_reason='CPU removed observation',http_responses=[],http_request_ids=[]))
        self.store.finish_evaluation_attempt(invalid,status='invalid',outcome=dict(artifact_sha256=verdict['sha256']),
            reason='CPU removed observation')
        return [missingrow,dict(task_id=task,attempt_id=invalid,record=record,rollout=rollout,verdict=verdict)]

    async def cohort(self):
        review=_json(await self.owner.initial(self.store,self.contract,{}))
        return review,review['request'],review['result']

    async def test_full300_recomputed_and_shared_initial_is_idempotent(self):
        a,b=await asyncio.gather(self.owner.initial(self.store,self.contract,{}),self.owner.initial(self.store,self.contract,{}))
        review=_json(a);self.assertEqual(a,b)
        again=_json(await self.owner.initial(self.store,self.contract,{}))
        self.assertEqual(self.calls,1);self.assertEqual(review,again)
        self.assertFalse(review['verified_complete']);self.assertEqual(review['verification_kind'],'cpu_mock')
        self.assertEqual(review['metrics']['valid'],250);self.assertEqual(review['metrics']['invalid'],50)
        self.assertEqual(len(review['task_outcomes']),300)
        self.assertEqual(review['charged_resources']['browser_dispatches'],300)
        draw=reserve(self.store,'unrelated-validation',policy=self.original['sha256'])['draw_id']
        cost=dict(gpu_seconds=1,api_requests=0,api_microdollars=0,browser_dispatches=1)
        self.store.start_attempt(draw,'unrelated-validation',reserved_cost=cost)
        self.store.finish_attempt('unrelated-validation',status='missing',actual_cost=cost,reason='CPU diagnostic')
        later=_json(await self.owner.initial(self.store,self.contract,{}))
        self.assertEqual(self.calls,1);self.assertEqual(later['metrics'],review['metrics'])
        self.assertGreater(later['ledger_revision'],review['ledger_revision'])
        with (Path(review['request']['path']).parent/'owner.lock').open('a+b') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            started=time.monotonic()
            with self.assertRaisesRegex(TimeoutError,'allocation deadline'):
                await self.owner.initial(self.store,self.contract,dict(launch=SimpleNamespace(deadline_unix=time.time()+.05)))
            self.assertLess(time.monotonic()-started,1.)
        with self.assertRaisesRegex(TimeoutError,'allocation deadline'):
            await self.owner.initial(self.store,self.contract,dict(launch=SimpleNamespace(deadline_unix=time.time()-1)))
        self.assertEqual(self.calls,1)

    async def test_missing_or_duplicate_task_and_changed_metrics_rejected(self):
        _,request,result=await self.cohort();base=_json(result)
        for i,change in enumerate(('missing','duplicate','metrics')):
            value=deepcopy(base)
            if change=='missing':value['tasks'].pop()
            if change=='duplicate':value['tasks'][1]=value['tasks'][0]
            if change=='metrics':value['metrics']['successes']=1
            altered=_write(self.root/f'changed-{i}.json',value)
            with self.subTest(change=change),self.assertRaises(ValueError):
                verify_full300(self.store,request,altered,diagnostic_only=True)

    async def test_corrupt_actual_rollout_and_production_claim_fail(self):
        _,request,result=await self.cohort()
        with self.assertRaises(ValueError):
            verify_full300(self.store,request,result)
        rollout=_json(result)['tasks'][0]['rollout'];Path(rollout['path']).write_bytes(b'corrupt')
        with self.assertRaises(ValueError):verify_full300(self.store,request,result,diagnostic_only=True)

    async def test_verdict_binding_and_wandb_project_are_required(self):
        _,request,result=await self.cohort();value=_json(result)
        with self.store._transaction() as state:state['evaluation_attempts']['attempt-eval-0']['completion']['outcome']['reward']=1.
        with self.assertRaisesRegex(ValueError,'settled evaluation'):
            verify_full300(self.store,request,result,diagnostic_only=True)
        with self.store._transaction() as state:state['evaluation_attempts']['attempt-eval-0']['completion']['outcome']['reward']=0.
        relabel=deepcopy(value);verdict=_json(relabel['tasks'][0]['verdict'])
        verdict.update(status='invalid',reward=None,invalid_reason='Cannot relabel native valid failure')
        relabel['tasks'][0]['verdict']=_write(self.root/'relabel-valid.json',verdict)
        with self.assertRaisesRegex(ValueError,'validity differs'):
            verify_full300(self.store,request,_write(self.root/'relabel-result.json',relabel),diagnostic_only=True)
        wandb=_json(value['wandb']);wandb['project']='openwebrl'
        value['wandb']=_write(self.root/'wrong-project.json',wandb)
        with self.assertRaisesRegex(ValueError,'W&B'):
            verify_full300(self.store,request,_write(self.root/'wrong-result.json',value),diagnostic_only=True)

    async def test_training_overlap_and_uncommitted_final_checkpoint_rejected(self):
        changed=SimpleNamespace(**dict(self.contract.__dict__,train_task_ids=('eval-0',)))
        with self.assertRaisesRegex(ValueError,'overlap'):await self.owner.initial(self.store,changed,{})
        self.assertEqual(self.calls,0)
        with self.store._transaction() as state:
            state.update(updates={'9':dict(phase='consumed',training_receipt=dict(new_checkpoint_sha256=sha('wrong')))},
                native_completed_updates={'9':dict(checkpoint_inventory=self.original)})
        with self.assertRaisesRegex(ValueError,'committed native checkpoint'):
            await self.owner.after_update(self.store,10,dict(committed=True),{})
        self.assertIsNone(await self.owner.after_update(self.store,9,{},{}))

    async def test_unknown_http_liability_and_success_remain_in_same_cap(self):
        self.paid_fixture=True
        review,request,result=await self.cohort()
        self.assertEqual(review['charged_resources']['api_microdollars'],56)
        self.assertEqual(review['charged_resources']['api_requests'],2)
        self.assertEqual(review['metrics']['successes'],1)
        with self.assertRaisesRegex(ValueError,'cannot be retried'):
            self.store.start_evaluation_attempt(review['cohort_id'],'eval-1','retry-valid-failure')
        value=_json(result);value['http_attempts']=value['http_attempts'][1:]
        with self.assertRaisesRegex(ValueError,'omits an evaluation retry'):
            verify_full300(self.store,request,_write(self.root/'omitted-liability.json',value),diagnostic_only=True)

    async def test_prior_browser_artifacts_and_concurrent_unrelated_spending(self):
        self.prior_fixture=True
        review,request,result=await self.cohort();value=_json(result)
        self.assertEqual(review['charged_resources']['browser_dispatches'],302)
        self.assertEqual(review['metrics']['valid'],250)
        missing=deepcopy(value);missing['prior_attempts'].pop()
        with self.assertRaisesRegex(ValueError,'Prior browser attempt inventory'):
            verify_full300(self.store,request,_write(self.root/'lost-prior.json',missing),diagnostic_only=True)
        changed=deepcopy(value);cleanup=_json(changed['prior_attempts'][0]['cleanup']);cleanup['processes_alive']=[999]
        changed['prior_attempts'][0]['cleanup']=_write(self.root/'unresolved-cleanup.json',cleanup)
        with self.assertRaisesRegex(ValueError,'cleanup remains unresolved'):
            verify_full300(self.store,request,_write(self.root/'unresolved-prior.json',changed),diagnostic_only=True)
        original_archive=evaluation._archive;spent=False
        def spend_during_scan(*args):
            nonlocal spent
            if not spent:
                spent=True
                draw=reserve(self.store,'concurrent-validation',policy=self.original['sha256'])['draw_id']
                cost=dict(gpu_seconds=2,api_requests=0,api_microdollars=0,browser_dispatches=1)
                self.store.start_attempt(draw,'concurrent-validation',reserved_cost=cost)
                self.store.finish_attempt('concurrent-validation',status='missing',actual_cost=cost,reason='CPU diagnostic')
            return original_archive(*args)
        with patch.object(evaluation,'_archive',side_effect=spend_during_scan):
            fresh=verify_full300(self.store,request,result,diagnostic_only=True)
        self.assertEqual(fresh['charged_resources']['gpu_seconds'],2)
        self.assertEqual(fresh['charged_resources']['browser_dispatches'],303)
        self.assertGreater(fresh['ledger_revision'],review['ledger_revision'])
        def change_cohort(*args):
            with self.store._transaction() as state:
                state['evaluation_attempts']['prior-missing']['completion']['reason']='changed during scan'
            return original_archive(*args)
        with patch.object(evaluation,'_archive',side_effect=change_cohort),self.assertRaisesRegex(ValueError,'cohort changed'):
            verify_full300(self.store,request,result,diagnostic_only=True)
        with self.store._transaction() as state:
            state['evaluation_attempts']['prior-missing']['completion']['reason']='CPU browser failed'
        changed=deepcopy(value);prior=changed['prior_attempts'][1]
        archive=original_archive(prior['rollout'],'eval-0');archive['turns'][0]['remove_sample']=False
        archive['turns'][0]['metadata']['reward']['valid']=True
        path=self.root/'wrong-valid-prior.pt';torch.save(archive,path);prior['rollout']=dict(path=str(path),sha256=_hash(path))
        verdict=_json(prior['verdict']);verdict['rollout_sha256']=prior['rollout']['sha256']
        prior['verdict']=_write(self.root/'wrong-valid-prior-verdict.json',verdict)
        record=_json(prior['record']);record['rollout_file']=path.name;record['reward_metadata']['valid']=True
        prior['record']=_write(self.root/'wrong-valid-prior-record.json',record)
        with self.assertRaisesRegex(ValueError,'validity differs'):
            verify_full300(self.store,request,_write(self.root/'wrong-valid-prior-result.json',changed),diagnostic_only=True)

    def test_native_prompt_metadata_cohort_schema_and_conflicting_ids(self):
        path=self.root/'native.jsonl'
        path.write_text(''.join(json.dumps(dict(prompt='CPU task',metadata=dict(task_id=task)))+'\n' for task in self.ids))
        self.assertEqual(_ids(dict(path=str(path),sha256=_hash(path))),tuple(self.ids))
        path.write_text(path.read_text().replace('"prompt": "CPU task"','"task_id": "wrong", "prompt": "CPU task"',1))
        with self.assertRaisesRegex(ValueError,'Conflicting'):_ids(dict(path=str(path),sha256=_hash(path)))

    def test_actor_inventory_binds_all_received_and_unknown_shared_requests(self):
        request=dict(schema_version=1,task_ids=self.ids,task_manifest=self.tasks,protocol=evaluation_protocol(),
            protocol_sha256=digest(evaluation_protocol()),checkpoint_inventory=self.original,policy_sha256=self.original['sha256'],
            iteration=0,source_inventory=self.source,accounting_manifest_sha256=self.store.manifest.sha256,
            artifact_schema='native_monitor_full300_v1',diagnostic_only=True)
        request['cohort_id']=digest(request)
        ref=_write(self.root/'actor-cohort.json',request)
        self.store.register_evaluation_cohort(request['cohort_id'],ref)
        self.store.start_evaluation_attempt(request['cohort_id'],'eval-0','browser-attempt')
        rows=[]
        for turn,status in enumerate(('unknown','received')):
            key='actor-'+status;payload=dict(input_ids=[1],turn=turn,
                sampling_params=dict(temperature=0.,top_p=1.,top_k=1,max_new_tokens=4096))
            payload_ref=_write(self.root/(key+'-request.json'),payload)
            self.store.reserve_evaluation_actor_request(key,parent_attempt_id='browser-attempt',task_id='eval-0',
                policy_sha256=self.original['sha256'],payload_sha256=digest(payload),turn=turn)
            row=dict(request_id=key,request=payload_ref)
            if status=='received':
                response=dict(response_ids=[2],text='CPU fake');row['response']=_write(self.root/'actor-response.json',response)
                self.store.settle_evaluation_actor_request(key,status=status,response_sha256=digest(response))
            else:
                row['error']=_write(self.root/'actor-error.json',dict(error='CPU timeout'))
                self.store.settle_evaluation_actor_request(key,status=status,reason='CPU timeout')
            rows.append(row)
        state=self.store.snapshot();attempts=state['evaluation_attempts']
        self.assertEqual(_actor_inventory(self.store,state,attempts,dict(actor_attempts=rows),self.original['sha256']),{('browser-attempt',1)})
        with self.assertRaisesRegex(ValueError,'omits an evaluation dispatch'):
            _actor_inventory(self.store,state,attempts,dict(actor_attempts=rows[1:]),self.original['sha256'])
        changed=deepcopy(rows);changed[1]['response']=_write(self.root/'changed-response.json',dict(response_ids=[666],text='changed'))
        with self.assertRaisesRegex(ValueError,'transport digest'):
            _actor_inventory(self.store,state,attempts,dict(actor_attempts=changed),self.original['sha256'])
        altered=deepcopy(rows);modified_state=deepcopy(state)
        payload=_json(rows[1]['request']);payload['sampling_params']['temperature']=.8
        altered[1]['request']=_write(self.root/'nongreedy-request.json',payload)
        modified_state['online_actor_requests']['actor-received']['payload_sha256']=digest(payload)
        with self.assertRaisesRegex(ValueError,'T0 evaluation sampling'):
            _actor_inventory(self.store,modified_state,attempts,dict(actor_attempts=altered),self.original['sha256'])

    def test_native_gating_handles_valid_failures_and_rejects_malformed_judge(self):
        good='<think>CPU</think><tool_call>{"name":"done","arguments":{"response":"finished"}}</tool_call>'
        base=dict(status='completed',response=good,reward=1.,remove_sample=False,
            metadata=dict(turn_index=0,terminate_reason='task_completed',reward=dict(format=1.,judge=1.,combined=1.,valid=True,judge_timeout=False)))
        self.assertEqual(_native_status([base]),(True,1.))
        failed=deepcopy(base);failed['status']='failed'
        with self.assertRaisesRegex(ValueError,'cannot receive judge success'):_native_status([failed])
        malformed=deepcopy(base);malformed['metadata']['reward']['judge']=None
        with self.assertRaisesRegex(ValueError,'finite binary judge'):_native_status([malformed])
        malformed['remove_sample']=True
        self.assertEqual(_native_status([malformed]),(False,None))
        invalid_format=deepcopy(base);invalid_format['response']='bad';invalid_format['reward']=0.
        invalid_format['metadata']['reward'].update(format=0.,combined=0.)
        self.assertEqual(_native_status([invalid_format]),(True,0.))
        parse_failure=deepcopy(invalid_format);parse_failure['status']='failed';parse_failure['reward']=-1.
        parse_failure['metadata']['terminate_reason']='format_error_failed'
        parse_failure['metadata']['reward'].update(judge=0.,combined=-1.)
        self.assertEqual(_native_status([parse_failure]),(True,0.))


if __name__=='__main__':unittest.main()
