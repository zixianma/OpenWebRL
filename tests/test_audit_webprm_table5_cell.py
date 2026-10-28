"""Offline audit tamper tests; no browser, model, scheduler or network calls."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import audit_webprm_table5_cell as audit
from openwebrl.webprm_webarena import digest, load_native_evaluator
from openwebrl.webprm_webarena_episode import TaskEnvelope
from openwebrl.webprm_evaluator_transcript import EvaluationRecorder

SOURCE = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-baselines-20261009/sources/arbiter')


class CellAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='webprm-cell-audit-',dir='/tmp')
        self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        network=patch('socket.socket.connect',side_effect=AssertionError('Offline audit must not use network'))
        network.start();self.addCleanup(network.stop)

    def put(self,name,value):
        path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value));return path

    def ref(self,path):return dict(path=str(path),sha256=audit.sha(path))

    def test_evidence_reference_rejects_changes_aliases_and_campaign_escape(self):
        path=self.put('evidence.json',{'initial':True});ref=self.ref(path)
        self.assertEqual(audit.reference(ref,root=self.root),path)
        path.write_text('changed')
        with self.assertRaisesRegex(ValueError,'changed'):audit.reference(ref)
        alias=self.root/'alias';alias.symlink_to(path)
        with self.assertRaises(ValueError):audit.reference(self.ref(alias))
        with self.assertRaisesRegex(ValueError,'escaped'):audit.reference(self.ref(path),root=self.root/'other')

    def closure_fixture(self):
        watchdog=dict(remaining_owned_processes={},child_exitcode=0)
        site=dict(remaining_owned_processes={},child_exitcode=-15)
        wp=self.put('watchdog.json',watchdog);sp=self.put('site.json',site)
        status=self.put('broker-status.json',dict(active_cell_ids=['another-cell'],active_cells=1,policy_sha256='5'*64,
            budget=dict(budget_prefix_bytes=0,budget_ledger_sha256='4'*64)))
        closure=dict(cell_id='ours',worker_remaining_processes={},service_remaining_processes={},broker_cell_inflight=0,
            worker_watchdog_sha256=audit.sha(wp),site_closure_sha256=audit.sha(sp),broker_status_reference=self.ref(status))
        cp=self.put('closure.json',closure)
        return dict(cell={'cell_id':'ours'},broker={'cell_id':'ours'},budget=dict(prefix_bytes=0,prefix_sha256='4'*64)),dict(watchdog=wp,site_closure=sp,closure=cp),dict(watchdog=watchdog)

    def test_closure_allows_other_cell_calls_but_rejects_own_inflight(self):
        inputs,paths,result=self.closure_fixture();audit.audit_closure(inputs,paths,result)
        status=self.put('broker-status.json',dict(active_cell_ids=['ours'],active_cells=1))
        c=audit.read(paths['closure']);c['broker_status_reference']=self.ref(status);self.put('closure.json',c)
        with self.assertRaisesRegex(ValueError,'inactive'):audit.audit_closure(inputs,paths,result)

    def test_closure_rejects_live_processes_and_changed_proofs(self):
        inputs,paths,result=self.closure_fixture()
        c=audit.read(paths['closure']);c['service_remaining_processes']={'123':{}};self.put('closure.json',c)
        with self.assertRaisesRegex(ValueError,'closure'):audit.audit_closure(inputs,paths,result)

    def test_closure_cannot_substitute_an_earlier_budget_snapshot(self):
        inputs,paths,result=self.closure_fixture();inputs['budget']['prefix_sha256']='6'*64
        with self.assertRaisesRegex(ValueError,'financial prefix'):audit.audit_closure(inputs,paths,result)
        inputs,paths,result=self.closure_fixture();self.put('site.json',{'remaining_owned_processes':{},'changed':True})
        with self.assertRaisesRegex(ValueError,'closure'):audit.audit_closure(inputs,paths,result)

    def fuzzy_fixture(self):
        from openwebrl.webprm_webarena_fuzzy import NativeFuzzyJudge,REPLACEMENT_MODEL,REPLACEMENT_PROTOCOL
        plan=dict(arbiter_source=str(SOURCE),fuzzy_enabled=True,fuzzy_model=REPLACEMENT_MODEL,
            fuzzy_protocol=REPLACEMENT_PROTOCOL,fuzzy_model_change_approved=True,
            fuzzy_max_input_tokens=8192,fuzzy_max_calls=1,deadline=1800000000)
        native=NativeFuzzyJudge(SOURCE,enabled=False,model=REPLACEMENT_MODEL,protocol=REPLACEMENT_PROTOCOL,
            model_change_approved=True,max_input_tokens=8192,max_calls=1,deadline=1800000000)
        captured=[];usage=dict(prompt_tokens=123,completion_tokens=2)
        def reply(**request):captured.append(request);return 'correct',usage
        native._native['generate_from_openai_chat_completion']=reply
        self.assertEqual(native._native['llm_fuzzy_match']('0','0','Fixture count?'),1.)
        payload=captured[0];payload.pop('context_length');payload['model']=REPLACEMENT_MODEL
        broker=dict(request_id='1'*32,reservation_id=7,reservation_state='settled')
        response=dict(model=REPLACEMENT_MODEL,choices=[dict(finish_reason='stop',message={'content':'correct'})],usage=usage)
        pair=dict(request=dict(purpose='native_fuzzy',payload=payload,reservation=7),
            response=dict(response=response,reservation=7),broker=broker)
        events=[dict(event='response',response=dict(response,_webprm_broker=broker))]
        return plan,[pair],events

    def test_saved_fuzzy_native_prompt_replays_without_provider(self):
        plan,pairs,events=self.fuzzy_fixture();proxy,count=audit.saved_fuzzy_proxy(plan,pairs,events)
        self.assertEqual(count,1);self.assertEqual(proxy.fuzzy_match('0','0','Fixture count?'),1.)
        self.assertEqual(proxy.provider_calls,1)
        with self.assertRaisesRegex(ValueError,'Unrecorded'):proxy.fuzzy_match('0','0','Fixture count?')

    def test_fuzzy_replay_rejects_changed_question_metered_response_or_receipt(self):
        plan,pairs,events=self.fuzzy_fixture()
        proxy,_=audit.saved_fuzzy_proxy(plan,pairs,events)
        with self.assertRaisesRegex(ValueError,'changed'):proxy.fuzzy_match('0','0','Different task?')
        for change in ('content','reservation'):
            changed=deepcopy(events)
            if change=='content':changed[0]['response']['choices'][0]['message']['content']='incorrect'
            else:changed[0]['response']['_webprm_broker']['reservation_id']=8
            proxy,_=audit.saved_fuzzy_proxy(plan,pairs,changed)
            with self.assertRaisesRegex(ValueError,'changed'):proxy.fuzzy_match('0','0','Fixture count?')

    def native_fixture(self):
        from PIL import Image
        from openwebrl.webprm_arbiter_harness import load_arbiter_harness,native_execution_code
        raw=dict(task_id=999,sites=['reddit'],intent='Synthetic terminal audit fixture',start_url='__REDDIT__',
                 storage_state='required.json',eval={'eval_types':['string_match'],'reference_answers':{'exact_match':'done'}})
        task=TaskEnvelope.from_release(raw,{'reddit':'http://127.0.0.1:9999'})
        plan=dict(arbiter_source=str(SOURCE),max_turns=30)
        trajectory=[dict(observation={'text':'Synthetic state'},info={}),dict(action_type='send_msg_to_user',answer='done',raw_action='send_msg_to_user [done]')]
        transcript=self.root/'native-evaluator-transcript.json'
        recorder=EvaluationRecorder(transcript)
        verdict=load_native_evaluator(SOURCE,observation_recorder=recorder).evaluate(task.native_config(),trajectory,termination='stop')
        screenshot=self.root/'terminal.png';Image.new('RGB',(2,2)).save(screenshot)
        terminal=dict(url='http://127.0.0.1:9999',screenshot=str(screenshot),screenshot_sha256=audit.sha(screenshot),termination='stop')
        result=dict(task_id=task.execution_id,released_task_id=999,raw_config_sha256=task.raw_config_sha256,
            configured_config_sha256=task.configured_config_sha256,valid=True,success=1,score=1.,turns=1,reset_id='reset',terminal=terminal,
            observation_transcript=verdict['observation_transcript'])
        instance=dict(synthetic=True)
        harness=load_arbiter_harness(SOURCE);action='send_msg_to_user [done]'
        code=native_execution_code(SOURCE,harness.parse_action(action),{},key_mapping=harness.key_mapping)
        dispatch=dict(action=action,element_bounds={},code=code,code_sha256=hashlib.sha256(code.encode()).hexdigest())
        values=dict(trajectory=trajectory,verdict=verdict,terminal=terminal,outcome=dict(result),
            rollout=[dict(turn=0,selected_index=0,proposal={'candidates':[{'action':action}]},execution_error=None,native_dispatch=dispatch)],
            provenance=dict(identity_sha256=digest(plan),original_instance=instance,reset_id='reset'),
            authentication=dict(authenticated=True,state_provenance_sha256='reset'),result=result)
        paths={k:self.put(k+'.json',v) for k,v in values.items()}
        paths.update(evaluator_inputs=transcript,terminal_screenshot=screenshot)
        return task,plan,paths,result,{'pairs':[]},instance

    def test_native_terminal_verdict_replays_from_complete_saved_evidence(self):
        args=self.native_fixture();outcome,proof=audit.audit_native(*args)
        self.assertTrue(proof['verified']);self.assertEqual(outcome['success'],1)
        self.assertFalse(proof['website_truth_independently_verified'])

    def test_native_audit_rejects_changed_screenshot_trajectory_and_before_close_score(self):
        args=self.native_fixture();paths=args[2]
        original=paths['terminal_screenshot'].read_bytes();paths['terminal_screenshot'].write_bytes(b'changed')
        with self.assertRaises(ValueError):audit.audit_native(*args)
        paths['terminal_screenshot'].write_bytes(original)
        before=audit.read(paths['outcome']);self.put('outcome.json',dict(before,score=0))
        with self.assertRaisesRegex(ValueError,'provenance'):audit.audit_native(*args)
        self.put('outcome.json',before)
        trajectory=audit.read(paths['trajectory']);trajectory[-1]['answer']='changed';self.put('trajectory.json',trajectory)
        with self.assertRaises(ValueError):audit.audit_native(*args)

    def test_legacy_transcript_cannot_supply_guarded_protocol_result(self):
        from openwebrl.webprm_webarena import ABSTENTION_EVALUATOR_PROTOCOL
        args=list(self.native_fixture());args[1]=dict(args[1],evaluator_protocol=ABSTENTION_EVALUATOR_PROTOCOL)
        with self.assertRaisesRegex(ValueError,'protocol/termination'):audit.audit_native(*args)

    def test_native_termination_must_match_actual_saved_terminal(self):
        args=self.native_fixture();path=args[2]['terminal'];terminal=audit.read(path)
        terminal['termination']='different';self.put('terminal.json',terminal)
        with self.assertRaisesRegex(ValueError,'protocol/termination'):audit.audit_native(*args)

    def test_earlier_malformed_action_cannot_hide_behind_valid_terminal(self):
        args=self.native_fixture();paths=args[2];rollout=audit.read(paths['rollout'])
        rollout[0]['turn']=1;rollout.insert(0,dict(turn=0,selected_index=999,proposal={'candidates':[]}))
        self.put('rollout.json',rollout);args[3]['turns']=2
        outcome=audit.read(paths['outcome']);outcome['turns']=2;self.put('outcome.json',outcome)
        with self.assertRaisesRegex(ValueError,'candidate index'):audit.audit_native(*args)

    def test_normal_terminal_cannot_exceed_frozen_turn_limit(self):
        args=self.native_fixture();plan=dict(args[1],max_turns=0)
        with self.assertRaisesRegex(ValueError,'turn limit'):
            audit.audit_rollout(plan,audit.read(args[2]['trajectory']),audit.read(args[2]['rollout']),args[3])

    def rejection_fixture(self):
        args=self.native_fixture();plan=args[1]
        trajectory=audit.read(args[2]['trajectory']);rollout=audit.read(args[2]['rollout'])
        error='ActionParsingError: Native coordinate target lacks captured viewport bounds'
        action='click [404]'
        from openwebrl.webprm_arbiter_harness import load_arbiter_harness
        parsed=load_arbiter_harness(SOURCE).parse_action(action)
        observation=dict(text='Synthetic state',element_bounds={'7':[1,2,3,4]})
        trajectory[:0]=[dict(observation=observation,info={}),
            dict(action_type=parsed['kind'],raw_action=action,args=list(parsed['args']),kwargs=parsed['kwargs'])]
        trajectory[2]['info']=dict(execution_error=error,native_dispatch={})
        rollout[0]['turn']=1
        rollout.insert(0,dict(turn=0,selected_index=0,proposal={'candidates':[{'action':action}]},
                             execution_error=error,native_dispatch={}))
        return plan,trajectory,rollout,dict(turns=2)

    def test_empty_dispatch_requires_exact_replayed_pre_dispatch_rejection(self):
        audit.audit_rollout(*self.rejection_fixture())

    def test_empty_dispatch_cannot_hide_successful_conversion_or_unrelated_errors(self):
        fixture=self.rejection_fixture()
        for change in ('available_target','missing_bounds','missing_error','unrelated_error','different_error'):
            with self.subTest(change=change):
                plan,trajectory,rollout,result=deepcopy(fixture)
                if change=='available_target':trajectory[0]['observation']['element_bounds']['404']=[1,2,3,4]
                elif change=='missing_bounds':trajectory[0]['observation'].pop('element_bounds')
                elif change=='missing_error':rollout[0]['execution_error']=None
                elif change=='unrelated_error':rollout[0]['execution_error']='RuntimeError: browser closed'
                else:rollout[0]['execution_error']='ActionParsingError: another parsing failure'
                with self.assertRaises(ValueError):audit.audit_rollout(plan,trajectory,rollout,result)

    def test_replayed_rejection_still_requires_exact_saved_transition(self):
        plan,trajectory,rollout,result=self.rejection_fixture()
        trajectory[2]['info']['native_dispatch']=None
        with self.assertRaisesRegex(ValueError,'action/state'):
            audit.audit_rollout(plan,trajectory,rollout,result)

    def test_nonempty_but_truncated_implementation_freeze_is_rejected(self):
        path=self.put('unrelated.json',{'unrelated':True});code={str(path):audit.sha(path)}
        with self.assertRaisesRegex(ValueError,'incomplete'):
            audit.audit_code_freeze({'source_sha256':code},{'code_sha256':code})

    def test_other_original_domains_remain_gated(self):
        with self.assertRaisesRegex(ValueError,'no completed'):
            audit.audit_reddit_reset({'cell':{'domain':'map'}},{},{},{})

    def test_changed_cohort_task_cannot_keep_the_declared_release_hash(self):
        from scripts.prepare_webprm_table5_campaign import TASKS,cohort
        candidate=cohort(audit.read(TASKS));changed=deepcopy(candidate)
        target=changed['cells'][0]['task_id']
        for row in changed['cells']:
            if row['task_id']==target:row['task_config_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'members/order'):
            audit.audit_identity({'cell':changed['cells'][0]}, {}, changed)

    def test_interrupted_cell_journal_is_not_silently_truncated(self):
        path=self.root/'api.jsonl';path.write_bytes(b'{"event":"request"}\n{"partial":')
        with self.assertRaisesRegex(ValueError,'Incomplete'):audit.read_lines(path)

    def test_unicode_release_uses_cohort_hash_and_keeps_worker_utf8_hash(self):
        from scripts.prepare_webprm_table5_campaign import TASKS, cohort, digest as cohort_digest
        from openwebrl import webprm_webarena_worker as worker
        released=audit.read(TASKS);candidate=cohort(released)
        protocol={};candidate['protocol_sha256']=digest(protocol)
        raw=next(row for row in released if row['task_id']==89)
        cell=next(row for row in candidate['cells'] if row['task_id']==89 and row['selector']=='no_search')
        self.assertNotEqual(cohort_digest(raw),digest(raw))
        plan=dict(task_ids=[89],tasks_sha256=digest([raw]),actor_model=cell['actor'],selector=cell['selector'],
            actors=[cell['actor']],selectors=[cell['selector']],max_turns=30,paper_exact_protocol=False,endpoints={})
        with patch.object(worker,'validate_protocol_contract',return_value=protocol),\
                patch.object(worker,'validate_frozen_sources'),patch.object(audit,'audit_code_freeze'),\
                patch.object(TaskEnvelope,'from_release',return_value='bound') as envelope:
            self.assertEqual(audit.audit_identity({'cell':cell},plan,candidate),'bound')
            envelope.assert_called_once_with(raw,{})
            changed=dict(plan,tasks_sha256=cohort_digest([raw]))
            with self.assertRaisesRegex(ValueError,'Worker plan'):
                audit.audit_identity({'cell':cell},changed,candidate)

    def test_verified_invalid_attempt_has_no_success_and_retains_unreplayable_state(self):
        inputs,paths,result=self.closure_fixture()
        cell=dict(cell_id='ours',task_id=999,actor='actor',selector='no_search',domain='reddit',task_config_sha256='1'*64)
        plan={};identity=digest(plan)
        result.update(identity_sha256=identity,actor_model='actor',selector='no_search',task_id='webarena_lite_999',
            attempted=True,valid=False,error_type='SyntheticBrowserStartupFailure')
        paths['result']=self.put('episode/result.json',result);paths['api']=self.root/'episode/api.jsonl';paths['api'].write_bytes(b'')
        candidate=dict(protocol_sha256='2'*64,source_sha256='3'*64)
        cp=self.put('cohort.json',{'cohort':candidate});pp=self.put('worker-plan.json',plan)
        # Keep closure's actual broker ID consistent with the full worker identity.
        inputs.update(schema_version=1,campaign_root=str(self.root),cell=cell,cohort=self.ref(cp),worker_plan=self.ref(pp),
            artifacts={k:self.ref(p) for k,p in paths.items()},
            broker=dict(cell_id='webarena_lite_999/actor__no_search',audit_path='unused',prefix_bytes=0,prefix_sha256='4'*64),
            budget=dict(financial_contract=dict(cohort_sha256=digest(candidate),protocol_sha256='2'*64),
                        event_path='unused',prefix_bytes=0,prefix_sha256='4'*64))
        task=SimpleNamespace(execution_id='webarena_lite_999')
        with patch.object(audit,'audit_identity',return_value=task),\
                patch('openwebrl.webprm_journal_audit.audit_cell_calls',return_value={'verified':True,'pairs':[],
                    'broker_policy_sha256s':['5'*64]}),\
                patch.object(audit,'audit_reddit_reset',side_effect=AssertionError('Invalid result must not assert reset truth')):
            report=audit.verify(inputs)
        self.assertFalse(report['record']['valid']);self.assertIsNone(report['record']['success'])
        self.assertEqual(report['record']['effective_semantic_status'],'not_applicable')
        self.assertFalse(report['complete_for_disposal'])
        result['attempted']=False;self.put('episode/result.json',result)
        inputs['artifacts']['result']=self.ref(paths['result'])
        with patch.object(audit,'audit_identity',return_value=task),\
                patch('openwebrl.webprm_journal_audit.audit_cell_calls',return_value=dict(verified=True,pairs=[],
                    broker_policy_sha256s=['5'*64],api_request_count=1,paid_calls=1)):
            with self.assertRaisesRegex(ValueError,'Unattempted'):
                audit.verify(inputs)

    def test_failed_audit_is_durable_and_cannot_create_verified_record(self):
        path=self.put('input.json',dict(schema_version=0,campaign_root=str(self.root),artifacts={}))
        output=self.root/'audit.json';result=audit.audit_cell(path,output)
        self.assertFalse(result['audit_verified']);self.assertFalse(result['complete_for_disposal'])
        self.assertIsNone(result['normalized_record']);self.assertFalse(output.with_suffix('.record.json').exists())
        self.assertEqual(audit.read(output)['input_sha256'],audit.sha(path))
        with self.assertRaises(FileExistsError):audit.audit_cell(path,output)

    def test_normalized_record_uses_actual_audit_file_hash_without_cycle(self):
        path=self.put('input.json',dict(schema_version=1,campaign_root=str(self.root),artifacts={}))
        output=self.root/'audit.json'
        with patch.object(audit,'verify',return_value=dict(record={'cell_id':'synthetic'},complete_for_disposal=False)):
            result=audit.audit_cell(path,output)
        saved=audit.read(output);record=audit.read(output.with_suffix('.record.json'))
        self.assertNotIn('audit_sha256',saved['record'])
        self.assertEqual(record['audit_sha256'],audit.sha(output))
        self.assertEqual(result['normalized_record'],record)


if __name__=='__main__':unittest.main()
