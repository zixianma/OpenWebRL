import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts.analyze_webprm_webarena_pilot import (
    summarize,audit_ledger,audit,digest,proposal_digest,ROWS,ACTOR,JUDGE,PRICES,
    nonanswer_numeric_reference_positive,attach_outcome_quality,
)


class PilotAuditTests(unittest.TestCase):
    def test_nonanswer_audit_is_narrow_and_preserves_native_scores(self):
        task = dict(eval=dict(eval_types=['string_match'], reference_answers=dict(fuzzy_match=['0'])))
        verdict = dict(success=True, score=1)
        for answer in ('N/A', 'n/a', ' "N/A" '):
            self.assertTrue(nonanswer_numeric_reference_positive(task, [dict(answer=answer)], verdict))
        for answer in ('0', 'zero', 'There are no qualifying comments.', ''):
            self.assertFalse(nonanswer_numeric_reference_positive(task, [dict(answer=answer)], verdict))
        for reference in (['N/A'], ['0', 'N/A'], ['unknown'], [], ['zero']):
            alternate = dict(eval=dict(eval_types=['string_match'], reference_answers=dict(fuzzy_match=reference)))
            self.assertFalse(nonanswer_numeric_reference_positive(alternate, [dict(answer='N/A')], verdict))
        self.assertFalse(nonanswer_numeric_reference_positive(task, [dict(answer='N/A')], dict(success=False, score=0)))
        self.assertFalse(nonanswer_numeric_reference_positive(dict(eval=dict(eval_types=['url_match'])), [dict(answer='N/A')], verdict))
        records = [dict(selector=name, attempted=True, valid=True, success=1) for name in ROWS]
        original = json.dumps(records, sort_keys=True)
        summary = summarize(records)
        attach_outcome_quality(summary, {name: True for name in ROWS})
        self.assertEqual(summary['totals']['successes'], 5)
        self.assertEqual(summary['native_reported_rates']['overall_success_pct'], 100)
        self.assertIsNone(summary['overall_success_pct'])
        self.assertIsNone(summary['valid_only_success_pct'])
        self.assertEqual(summary['outcome_quality']['flagged_cells'], 5)
        self.assertTrue(all(row['overall_success_pct'] is None for row in summary['rows']))
        self.assertEqual(json.dumps(records, sort_keys=True), original)

    def test_missing_cells_have_no_complete_cohort_rate(self):
        summary=summarize([dict(selector='no_search',attempted=True,valid=True,success=0)])
        self.assertEqual({k:summary['totals'][k] for k in ('planned','recorded','attempted','valid','invalid','missing','successes')},dict(planned=5,recorded=1,attempted=1,valid=1,invalid=0,missing=4,successes=0))
        self.assertIsNone(summary['overall_success_pct'])
        self.assertEqual(summary['valid_only_success_pct'],0)
        self.assertFalse(summary['all_mechanics_valid'])

    def test_invalid_outcomes_stay_in_fixed_denominator(self):
        records=[dict(selector=row,attempted=True,valid=i<3,success=int(i==0)) for i,row in enumerate(ROWS)]
        summary=summarize(records)
        self.assertEqual(summary['overall_success_pct'],20)
        self.assertAlmostEqual(summary['valid_only_success_pct'],100/3)
        self.assertEqual(summary['totals']['invalid'],2)
        self.assertFalse(summary['all_mechanics_valid'])

    def test_success_requires_valid_attempt_and_no_duplicate_cell(self):
        record=dict(selector='no_search',attempted=True,valid=True,success=1)
        for records in [[record,record],[dict(record,valid=False)],[dict(record,attempted=False)]]:
            with self.assertRaises(ValueError):summarize(records)

    def test_unknown_paid_call_keeps_full_reservation(self):
        usage=dict(prompt_tokens=100,completion_tokens=10)
        cost=100*.15/1e6+10*.60/1e6
        ledger=dict(cap=15,charged=.1+cost,calls=[
            dict(id=0,model=ACTOR,state='reserved',reserved=.1),
            dict(id=1,model=ACTOR,state='settled',reserved=.2,charged=cost,usage=usage)])
        audit=audit_ledger(ledger,{0:ACTOR,1:ACTOR},{1:usage})
        self.assertEqual(audit['retained_reservations_usd'],.1)
        self.assertAlmostEqual(audit['settled_usd'],cost)
        with self.assertRaisesRegex(ValueError,'matching request/response'):audit_ledger(ledger,{1:ACTOR},{})
        ledger['charged']=0
        with self.assertRaisesRegex(ValueError,'total differs'):audit_ledger(ledger,{1:ACTOR},{1:usage})

    def fixture(self, campaign, *, fallback=False, guarded=False, cpu_only=False):
        if cpu_only: guarded=True
        if guarded: fallback=True
        from PIL import Image
        from types import SimpleNamespace
        from openwebrl.webprm_webarena import verify_source,load_native_evaluator,TASK_FILE
        from openwebrl.webprm_webarena_episode import TaskEnvelope
        from openwebrl.webprm_webarena_fuzzy import NativeFuzzyJudge,REPLACEMENT_PROTOCOL
        from openwebrl.webprm_broker import payload_sha256
        from openwebrl.webprm_reddit_provenance import ORIGINAL_MANIFEST,ORIGINAL_CONFIG,original_layers
        source=Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-baselines-20261009/sources/arbiter')
        raw=next(t for t in json.loads((source/TASK_FILE).read_text()) if t['task_id']==5)
        plan=dict(arbiter_source=str(source),source_manifest=verify_source(source),task_ids=[5],
            actors=[ACTOR],selectors=list(ROWS),tasks_sha256=digest([raw]),endpoints=dict(reddit='http://127.0.0.1'),
            arbiter_model='synthetic-arbiter',shepherd_model='synthetic-shepherd',piotr_model='synthetic-piotr',
            fuzzy_enabled=True,fuzzy_model=JUDGE,fuzzy_protocol=REPLACEMENT_PROTOCOL,
            fuzzy_model_change_approved=True,fuzzy_max_input_tokens=8192,fuzzy_max_calls=1,deadline=1800000000)
        if fallback:
            from openwebrl.webprm_webarena_worker import protocol_definition,NA_TURN_LIMIT
            plan.update(max_turns=30,turn_limit_protocol=NA_TURN_LIMIT)
            plan['protocol_contract']=protocol_definition(source,turn_limit=NA_TURN_LIMIT)
            plan['protocol']=plan['protocol_contract']['identity']
        if guarded:
            from openwebrl.webprm_webarena import ABSTENTION_EVALUATOR_PROTOCOL
            from openwebrl.webprm_webarena_worker import ENVELOPE_NO_SEARCH_PROMPT
            plan.update(no_search_prompt_version=ENVELOPE_NO_SEARCH_PROMPT,
                evaluator_protocol=ABSTENTION_EVALUATOR_PROTOCOL)
            plan['protocol_contract']=protocol_definition(source,turn_limit=NA_TURN_LIMIT,
                no_search_prompt=ENVELOPE_NO_SEARCH_PROMPT,scoring_protocol=ABSTENTION_EVALUATOR_PROTOCOL)
            plan['protocol']=plan['protocol_contract']['identity']
        if cpu_only:
            plan.update(diagnostic_protocol='reddit_no_search_guarded_browser_v1',selectors=['no_search'])
        task=TaskEnvelope.from_release(raw,plan['endpoints'])
        attempt=campaign/'attempt-synthetic';proposal={'description':'Synthetic Unicode identity: π'}
        approval={}
        if fallback:
            from scripts.analyze_webprm_webarena_pilot import V3_PROFILE,V3_RESOURCES,V3_LIMITS
            proposal.update(resource_profile=V3_PROFILE,resources=V3_RESOURCES,limits=V3_LIMITS)
            approval.update(status='approved',resource_profile=V3_PROFILE,resources=V3_RESOURCES,limits=V3_LIMITS,
                turn_limit_protocol=NA_TURN_LIMIT,turn_limit_protocol_approved=True)
        if guarded:
            from scripts.analyze_webprm_webarena_pilot import V4_PROFILE,V4_RESOURCES,V4_LIMITS
            for record in (proposal,approval):
                record.update(resource_profile=V4_PROFILE,resources=V4_RESOURCES,limits=V4_LIMITS,
                    no_search_prompt_version=ENVELOPE_NO_SEARCH_PROMPT,evaluator_protocol=ABSTENTION_EVALUATOR_PROTOCOL)
            approval.update(no_search_prompt_version_approved=True,evaluator_protocol_approved=True)
        if cpu_only:
            from scripts.analyze_webprm_webarena_pilot import CPU_PROFILE,CPU_RESOURCES,CPU_LIMITS
            for record in (proposal,approval):
                record.update(resource_profile=CPU_PROFILE,resources=CPU_RESOURCES,limits=CPU_LIMITS)
            proposal.update(diagnostic_protocol=plan['diagnostic_protocol'],rows=['no_search'],maximum_allocation_attempts=1)
        approval['proposal_sha256']=proposal_digest(proposal)
        self.put(attempt/'execution-context.json',dict(proposal=proposal,proposal_sha256=proposal_digest(proposal),
            approval=approval))
        self.put(attempt/'controller-closure.json',dict(remaining_owned_processes={}))
        self.put(attempt/'worker-plan.json',plan)
        verification=campaign/'original-verification.json'
        self.put(verification,dict(status='verified',all_original_layers_verified=True,
            source_manifest_digest='sha256:'+ORIGINAL_MANIFEST,config_sha256=ORIGINAL_CONFIG,
            layers=[dict(row,verified=True) for row in original_layers()]))
        download=campaign/'download.json'
        self.put(download,dict(status='complete',source_manifest_digest='sha256:'+ORIGINAL_MANIFEST,
            config_digest='sha256:'+ORIGINAL_CONFIG,oci_reference='oci:/tmp/synthetic-oci'))
        build=campaign/'build.json';baseline='/tmp/synthetic-immutable-baseline'
        self.put(build,dict(kind='original_reddit_sandbox_build',status='completed',returncode=0,
            source_manifest_sha256=ORIGINAL_MANIFEST,config_sha256=ORIGINAL_CONFIG,sandbox_path=baseline,
            download_receipt_path=str(download),download_receipt_sha256=self.sha(download),
            argv=['apptainer','build','--disable-cache','--sandbox',baseline,'oci:/tmp/synthetic-oci']))
        usage=dict(prompt_tokens=100,completion_tokens=10);charge=100*5/1e6+10*15/1e6
        calls=[];dispatches=[];folders=[]
        for index,name in enumerate(plan['selectors']):
            self.put(attempt/('worker-launch-'+name+'.json'),dict(selector=name))
            folder=attempt/'episodes/webarena_lite_005'/(ACTOR+'__'+name);folders.append(folder)
            native=folder/'native-evaluation';native.mkdir(parents=True)
            Image.new('RGB',(1280,720)).save(native/'terminal.png')
            terminal=dict(url='http://127.0.0.1',screenshot='terminal.png',termination='stop',
                screenshot_sha256=self.sha(native/'terminal.png'))
            state_path='/tmp/synthetic-state-'+name;state=campaign/('copy-'+name+'.json')
            self.put(state,dict(kind='original_reddit_state_copy',status='completed',returncode=0,
                state_path=state_path,source_sandbox_path=baseline,build_receipt_path=str(build),
                build_receipt_sha256=self.sha(build),original_verification_path=str(verification),
                original_verification_sha256=self.sha(verification),argv=['cp','-a','--reflink=auto','--',baseline,state_path]))
            reset=self.sha(state)
            instance=dict(scope='original',image_sha256=ORIGINAL_MANIFEST,config_sha256=ORIGINAL_CONFIG,
                verified_original_layers=29,endpoint='http://127.0.0.1',state_path=state_path,
                state_provenance_path=str(state),state_provenance_sha256=reset,build_receipt_path=str(build),
                build_receipt_sha256=self.sha(build),original_verification_sha256=self.sha(verification))
            instance_path=campaign/('instance-'+name+'.json');self.put(instance_path,instance)
            self.put(attempt/('lease-'+name+'.json'),dict(task_id='webarena_lite_005',actor_model=ACTOR,
                selector=name,instance_receipt=str(instance_path),instance_receipt_sha256=self.sha(instance_path)))
            self.put(folder/'authentication.json',dict(authenticated=True,state_provenance_sha256=reset))
            self.put(folder/'provenance.json',dict(identity_sha256=digest(plan),reset_id=reset,original_instance=instance))
            answer='N/A' if fallback else 'Synthetic saved answer π';action='send_msg_to_user ['+answer+']'
            trajectory=[dict(observation=dict(text='Synthetic state'),info=dict(reset_id=reset)),
                dict(action_type='send_msg_to_user',answer=answer,raw_action=action)]
            fallback_record=None
            if fallback:
                from openwebrl.webprm_arbiter_harness import load_arbiter_harness,native_execution_code
                from openwebrl.webprm_source import ARBITER_REVISION
                harness=load_arbiter_harness(source);actor_action='goto [http://127.0.0.1]'
                parsed=harness.parse_action(actor_action)
                trajectory=trajectory[:1];rollout=[]
                for turn in range(30):
                    trajectory.extend([dict(action_type=parsed['kind'],raw_action=actor_action,
                        args=list(parsed['args']),kwargs=parsed['kwargs']),
                        dict(observation=dict(text='Synthetic state'),info=dict(execution_error=''))])
                    rollout.append(dict(turn=turn,selected_index=0,execution_error='',
                        proposal=dict(candidates=[dict(action=actor_action,thought='Saved nonterminal action')])))
                code=native_execution_code(source,harness.parse_action(action),{},key_mapping=harness.key_mapping)
                origin='released_viewer_normal_turn_limit'
                fallback_record=dict(schema_version=1,protocol=NA_TURN_LIMIT,origin=origin,
                    termination_reason='turn_limit',action=action,answer='N/A',actor_selected=False,
                    actor_turns=30,pre_fallback_trajectory_sha256=digest(trajectory),source_revision=ARBITER_REVISION,
                    source_file='viewer/src/server/service/executors/step_loop_executor.py',status='executed',
                    chat_messages_before=[],chat_messages_after=[dict(role='assistant',message='N/A')],
                    appended_messages=[dict(role='assistant',message='N/A')],native_dispatch=dict(action=action,
                    code=code,code_sha256=hashlib.sha256(code.encode()).hexdigest(),element_bounds={}))
                trajectory.append(dict(action_type='send_msg_to_user',answer='N/A',raw_action=action,
                    origin=origin,protocol=NA_TURN_LIMIT))
                self.put(folder/'protocol-fallback.json',fallback_record)
            native_helper=NativeFuzzyJudge(source,enabled=False,model=JUDGE,protocol=REPLACEMENT_PROTOCOL,
                model_change_approved=True,max_input_tokens=8192,max_calls=1,deadline=plan['deadline'])
            judge=SimpleNamespace(enabled=True,provider_calls=0,manifest=dict(native_helper.manifest,
                enabled=True,budget_owner='webprm_worker.completion'))
            payload={};text='correct' if index==0 else 'incorrect'
            def callback(**request):
                request.pop('context_length');request['model']=JUDGE;payload.update(request)
                judge.provider_calls+=1
                return text,usage
            native_helper._native['generate_from_openai_chat_completion']=callback
            judge.fuzzy_match=native_helper._native['llm_fuzzy_match'];judge.ua_match=native_helper._native['llm_ua_match']
            from openwebrl.webprm_webarena import NATIVE_EVALUATOR_PROTOCOL
            verdict=load_native_evaluator(source,fuzzy_judge=judge,
                scoring_protocol=plan.get('evaluator_protocol',NATIVE_EVALUATOR_PROTOCOL)).evaluate(task.native_config(),trajectory,termination='stop')
            success=int(verdict['success']);watchdog=dict(remaining_owned_processes={},reason=None,child_exitcode=0)
            outcome=dict(task_id='webarena_lite_005',released_task_id=5,raw_config_sha256=task.raw_config_sha256,
                configured_config_sha256=task.configured_config_sha256,valid=True,success=bool(success),score=success,
                terminal=terminal,cleanup_status='pending')
            if guarded:
                outcome.update({key:verdict[key] for key in ('evaluator_protocol','native_score','native_success',
                    'native_evaluator_details','scoring_amendments')})
            self.put(native/'outcome-before-close.json',outcome)
            result=dict(outcome,arm=ACTOR+'__'+name,identity_sha256=digest(plan),attempted=True,success=success,
                turns=30 if fallback else 1,watchdog=watchdog,reset_id=reset,cleanup_status='completed')
            if fallback:result.update(turn_limit_protocol=NA_TURN_LIMIT,protocol_fallback=fallback_record)
            self.put(folder/'result.json',result);self.put(folder/'watchdog.json',watchdog)
            self.put(folder/'rollout.json',[dict(turn=0,selected_index=0,execution_error='',
                proposal=dict(candidates=[dict(action=action,thought='Saved terminal action')]))] if not fallback else rollout)
            self.put(native/'terminal.json',terminal);self.put(native/'trajectory.json',trajectory)
            self.put(native/'native-verdict.json',verdict)
            response=dict(model=JUDGE,choices=[dict(finish_reason='stop',message=dict(role='assistant',content=text))],usage=usage)
            meta=dict(request_id='synthetic-'+name,identity_sha256=digest(plan),cell_id='webarena_lite_005/'+ACTOR+'__'+name,
                model=JUDGE,purpose='native_fuzzy',local=False,payload_sha256=payload_sha256(payload))
            receipt=dict(meta,reservation_id=index,reservation_state='settled')
            self.journal(folder/'api.jsonl',[
                dict(event='request',local=False,reservation=index,purpose='native_fuzzy',payload=payload),
                dict(event='response',reservation=index,response=dict(response,id='synthetic-completion',
                    object='chat.completion',created=1))])
            self.journal(folder/'native-fuzzy.jsonl',[
                dict(event='response',response=dict(response,_webprm_broker=receipt)),dict(event='native_verdict',score=verdict.get('native_score',success))])
            calls.append(dict(id=index,model=JUDGE,state='settled',reserved=.1,charged=charge,usage=usage))
            dispatches.extend([dict(meta,event='dispatch_started'),dict(receipt,event='dispatch_completed')])
        self.put(campaign/'shared-budget.json',dict(cap=2 if cpu_only else 15,charged=len(plan['selectors'])*charge,calls=calls))
        self.journal(attempt/'broker.jsonl',dispatches)
        for folder in folders:self.add_call(campaign,attempt,folder,'terminal',ACTOR,False)
        if fallback:
            from tests.test_webprm_reddit_fuzzy_audit import add_saved_fuzzy_preflight
            add_saved_fuzzy_preflight(campaign,attempt,plan,guarded=guarded)
        return attempt,folders

    def test_v4_replays_both_scores_and_eight_fixed_controls_without_mutation(self):
        with TemporaryDirectory(prefix='webprm-v4-audit-',dir='/tmp') as tmp, \
                patch('socket.socket.connect',side_effect=AssertionError('Network forbidden')):
            campaign=Path(tmp);attempt,folders=self.fixture(campaign,guarded=True)
            before={str(p):self.sha(p) for p in campaign.rglob('*') if p.is_file()}
            result=audit(campaign)
            self.assertEqual(result['native_positive_count'],1)
            self.assertEqual(result['effective_positive_count'],0)
            self.assertEqual(result['scoring_amendment_count'],5)
            self.assertEqual(result['native_fuzzy_preflights'][attempt.name]['paid_dispatches'],8)
            self.assertEqual(result['requests']['paid_cap'],600)
            self.assertEqual(result['totals']['valid'],5)
            self.assertEqual(before,{str(p):self.sha(p) for p in campaign.rglob('*') if p.is_file()})
            corrupted=json.loads((folders[0]/'result.json').read_text())
            corrupted['native_score']=0
            self.put(folders[0]/'result.json',corrupted)
            with self.assertRaisesRegex(ValueError,'native/effective outcome provenance'):
                audit(campaign)

    def test_cpu_diagnostic_is_one_episode_with_separate_cap_and_no_performance_rate(self):
        with TemporaryDirectory(prefix='webprm-cpu-audit-',dir='/tmp') as tmp, \
                patch('socket.socket.connect',side_effect=AssertionError('Network forbidden')):
            campaign=Path(tmp);attempt,folders=self.fixture(campaign,cpu_only=True)
            result=audit(campaign)
            self.assertEqual(result['totals']['planned'],1)
            self.assertEqual(result['totals']['valid'],1)
            self.assertEqual(result['native_positive_count'],1)
            self.assertEqual(result['effective_positive_count'],0)
            self.assertEqual(result['requests']['local_cap'],0)
            self.assertEqual(result['requests']['paid_cap'],98)
            self.assertEqual(result['api']['cap_usd'],2)
            self.assertIsNone(result['overall_success_pct'])
            self.assertTrue(result['diagnostic_only'])


    def test_v3_normal_limit_cohort_replays_native_NA_without_provider_calls(self):
        with TemporaryDirectory(prefix='webprm-v3-audit-',dir='/tmp') as tmp, \
                patch('socket.socket.connect',side_effect=AssertionError('Network forbidden')):
            campaign=Path(tmp);self.fixture(campaign,fallback=True)
            before={str(p):self.sha(p) for p in campaign.rglob('*') if p.is_file()}
            result=audit(campaign)
            self.assertTrue(result['all_native_verdicts_valid']);self.assertEqual(result['totals']['valid'],5)
            self.assertEqual(result['requests']['local_cap'],400)
            self.assertFalse(result['all_guided_selectors_exercised'])
            self.assertEqual(before,{str(p):self.sha(p) for p in campaign.rglob('*') if p.is_file()})

    def test_corrected_resource_caps_require_exact_approved_profile(self):
        from copy import deepcopy
        from scripts.analyze_webprm_webarena_pilot import approved_campaign_profile,V3_PROFILE,V3_RESOURCES,V3_LIMITS
        good=dict(proposal=dict(resource_profile=V3_PROFILE,resources=V3_RESOURCES,limits=V3_LIMITS),
            approval=dict(status='approved',resource_profile=V3_PROFILE,resources=V3_RESOURCES,limits=V3_LIMITS,
                turn_limit_protocol='released_viewer_na_fallback_v1',turn_limit_protocol_approved=True))
        self.assertEqual(approved_campaign_profile(good),(V3_PROFILE,575,400))
        self.assertEqual(approved_campaign_profile(dict(proposal={},approval={})),('legacy_v2',575,320))
        for changed in ['pending','no_protocol','more_local','more_time','wrong_profile']:
            bad=deepcopy(good)
            if changed=='pending':bad['approval']['status']='pending'
            if changed=='no_protocol':bad['approval'].pop('turn_limit_protocol_approved')
            if changed=='more_local':bad['proposal']['limits']['local_requests']=401
            if changed=='more_time':bad['proposal']['resources']['total_seconds']=10800
            if changed=='wrong_profile':bad['proposal']['resource_profile']='arbitrary'
            with self.subTest(changed=changed),self.assertRaises(ValueError):approved_campaign_profile(bad)

    def test_descriptor_v2_requires_explicit_proposal_and_approval(self):
        from copy import deepcopy
        from scripts.analyze_webprm_webarena_pilot import approved_campaign_profile,V3_PROFILE,V3_RESOURCES,V3_LIMITS
        from openwebrl.webprm_webarena_piotr import REPRESENTATION_V2
        context=dict(proposal=dict(resource_profile=V3_PROFILE,resources=V3_RESOURCES,limits=V3_LIMITS,
            piotr_representation=REPRESENTATION_V2),approval=dict(status='approved',resource_profile=V3_PROFILE,
            resources=V3_RESOURCES,limits=V3_LIMITS,turn_limit_protocol='released_viewer_na_fallback_v1',
            turn_limit_protocol_approved=True,piotr_representation=REPRESENTATION_V2,piotr_representation_approved=True))
        self.assertEqual(approved_campaign_profile(context),(V3_PROFILE,575,400))
        for key in ['piotr_representation','piotr_representation_approved']:
            bad=deepcopy(context);bad['approval'].pop(key)
            with self.assertRaises(ValueError):approved_campaign_profile(bad)

    def test_fallback_fabrication_hash_chat_dispatch_and_answer_overwrite_rejected(self):
        from copy import deepcopy
        from scripts.analyze_webprm_webarena_pilot import audit_terminal_action
        with TemporaryDirectory(prefix='webprm-v3-tamper-',dir='/tmp') as tmp, \
                patch('socket.socket.connect',side_effect=AssertionError('Network forbidden')):
            campaign=Path(tmp);attempt,folders=self.fixture(campaign,fallback=True)
            folder=folders[0];plan=json.loads((attempt/'worker-plan.json').read_text())
            result=json.loads((folder/'result.json').read_text())
            trajectory=json.loads((folder/'native-evaluation/trajectory.json').read_text())
            rollout=json.loads((folder/'rollout.json').read_text())
            audit_terminal_action(folder,result,plan,trajectory,rollout)
            for corruption in ['hash','chat','dispatch','answer','overwrite','artifact','missing','strict']:
                r,p,t,rows=map(deepcopy,(result,plan,trajectory,rollout))
                if corruption=='hash':r['protocol_fallback']['pre_fallback_trajectory_sha256']='0'*64
                if corruption=='chat':r['protocol_fallback']['chat_messages_before']=[dict(role='assistant',message='N/A')]
                if corruption=='dispatch':r['protocol_fallback']['native_dispatch']['code']='pass'
                if corruption=='answer':t[-1]['answer']='Fabricated answer'
                if corruption=='overwrite':
                    rows[-1]['proposal']['candidates'][0]['action']='send_msg_to_user [Actual answer]'
                    t[-3]=dict(action_type='send_msg_to_user',answer='Actual answer',raw_action='send_msg_to_user [Actual answer]')
                    r['protocol_fallback']['pre_fallback_trajectory_sha256']=digest(t[:-1])
                if corruption=='missing':r['protocol_fallback']=None
                if corruption=='strict':p.pop('turn_limit_protocol')
                disk=deepcopy(r['protocol_fallback']) if r['protocol_fallback'] is not None else result['protocol_fallback']
                if corruption=='artifact':disk['status']='dispatch_pending'
                self.put(folder/'protocol-fallback.json',disk)
                with self.subTest(corruption=corruption),self.assertRaises(ValueError):
                    audit_terminal_action(folder,r,p,t,rows)
            self.put(folder/'protocol-fallback.json',result['protocol_fallback'])

    def add_call(self,campaign,attempt,folder,purpose,model,local,*,finish_reason='stop',completed=True):
        """Append realistic host-journal/broker evidence without executing a model."""
        from openwebrl.webprm_broker import payload_sha256
        plan=json.loads((attempt/'worker-plan.json').read_text())
        rows=[json.loads(line) for line in (folder/'api.jsonl').read_text().splitlines()]
        broker=[json.loads(line) for line in (attempt/'broker.jsonl').read_text().splitlines()]
        ledger=json.loads((campaign/'shared-budget.json').read_text());reservation=None
        usage=dict(prompt_tokens=100,completion_tokens=10)
        if not local:
            reservation=len(ledger['calls']);rates=PRICES[model]
            cost=usage['prompt_tokens']*rates[0]+usage['completion_tokens']*rates[1]
            ledger['calls'].append(dict(id=reservation,model=model,state='settled',reserved=.1,charged=cost,usage=usage))
            ledger['charged']+=cost;self.put(campaign/'shared-budget.json',ledger)
        payload=dict(model=model,messages=[dict(role='user',content='Synthetic '+purpose)],max_tokens=128)
        response=dict(id='synthetic-extra',object='chat.completion',created=1,model=model,usage=usage,
            choices=[dict(finish_reason=finish_reason,message=dict(role='assistant',content='Synthetic complete response'))])
        rows.extend([dict(event='request',local=local,reservation=reservation,purpose=purpose,payload=payload),
                     dict(event='response',reservation=reservation,response=response)])
        self.journal(folder/'api.jsonl',rows)
        meta=dict(request_id='extra-'+str(len(broker)),identity_sha256=digest(plan),
            cell_id='webarena_lite_005/'+folder.name,model=model,purpose=purpose,local=local,
            payload_sha256=payload_sha256(payload))
        broker.append(dict(meta,event='dispatch_started'))
        if completed:broker.append(dict(meta,event='dispatch_completed',reservation_id=reservation,
            reservation_state='local' if local else 'settled'))
        self.journal(attempt/'broker.jsonl',broker)

    @staticmethod
    def put(path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))

    @staticmethod
    def journal(path,rows):
        path.write_text('\n'.join(map(json.dumps,rows)))

    @staticmethod
    def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

    def test_saved_cohort_native_replay_is_read_only_and_accepts_raw_provider_metadata(self):
        with TemporaryDirectory(prefix='webprm-result-audit-',dir='/tmp') as tmp, \
                patch('socket.socket.connect',side_effect=AssertionError('Network forbidden')):
            campaign=Path(tmp);self.fixture(campaign)
            before={str(p):self.sha(p) for p in campaign.rglob('*') if p.is_file()}
            result=audit(campaign)
            self.assertTrue(result['all_native_verdicts_valid']);self.assertFalse(result['all_mechanics_valid'])
            self.assertFalse(result['all_guided_selectors_exercised'])
            self.assertTrue(result['all_registered_controllers_closed'])
            for row in result['rows']:
                self.assertEqual(row['call_evidence']['by_group']['selector']['matched_complete'],0)
                self.assertEqual(row['call_evidence']['by_group']['actor_aux']['matched_complete'],1)
                self.assertEqual(row['call_evidence']['by_group']['native_judge']['matched_complete'],1)
            self.assertEqual(result['overall_success_pct'],20);self.assertEqual(result['totals']['successes'],1)
            self.assertNotIn(tmp,json.dumps(result))
            self.assertEqual(before,{str(p):self.sha(p) for p in campaign.rglob('*') if p.is_file()})

    def test_guided_call_coverage_is_separate_from_native_outcome_rates(self):
        with TemporaryDirectory(prefix='webprm-result-audit-',dir='/tmp') as tmp, \
                patch('socket.socket.connect',side_effect=AssertionError('Network forbidden')):
            campaign=Path(tmp);attempt,folders=self.fixture(campaign)
            before=audit(campaign)
            for _ in range(4):self.add_call(campaign,attempt,folders[1],'selector_pair',ACTOR,False)
            self.add_call(campaign,attempt,folders[2],'selector_checklist','synthetic-shepherd',True)
            for _ in range(5):self.add_call(campaign,attempt,folders[2],'selector_pointwise','synthetic-shepherd',True)
            for _ in range(4):self.add_call(campaign,attempt,folders[3],'selector_pair','synthetic-arbiter',True)
            self.add_call(campaign,attempt,folders[4],'selector_piotr','synthetic-piotr',True)
            result=audit(campaign)
            self.assertTrue(result['all_native_verdicts_valid']);self.assertTrue(result['all_guided_selectors_exercised'])
            self.assertTrue(result['all_mechanics_valid'])
            self.assertIsNone(result['rows'][0]['guided_selector_exercised'])
            self.assertEqual([row['call_evidence']['by_group']['selector']['matched_complete']
                              for row in result['rows']],[0,4,6,4,1])
            for key in ('totals','overall_success_pct','valid_only_success_pct'):
                self.assertEqual(result[key],before[key])
            self.put(attempt/'controller-closure.json',dict(remaining_owned_processes={'123':{}}))
            self.assertFalse(audit(campaign)['all_mechanics_valid'])

    def test_checklist_only_truncated_or_unmatched_responses_do_not_prove_selector_exercise(self):
        with TemporaryDirectory(prefix='webprm-result-audit-',dir='/tmp') as tmp:
            campaign=Path(tmp);attempt,folders=self.fixture(campaign)
            self.add_call(campaign,attempt,folders[1],'selector_pair',ACTOR,False,completed=False)
            self.add_call(campaign,attempt,folders[2],'selector_checklist','synthetic-shepherd',True)
            self.add_call(campaign,attempt,folders[3],'selector_pair','synthetic-arbiter',True,finish_reason='length')
            self.add_call(campaign,attempt,folders[4],'selector_piotr','wrong-model',True)
            result=audit(campaign)
            self.assertTrue(result['all_native_verdicts_valid']);self.assertFalse(result['all_mechanics_valid'])
            self.assertFalse(any(row['guided_selector_exercised'] for row in result['rows']))
            gpt=result['rows'][1]['call_evidence']['by_group']['selector']
            self.assertEqual(gpt,dict(requests=1,complete_responses=1,broker_started=1,broker_completed=0,matched_complete=0))
            arbiter=result['rows'][3]['call_evidence']['by_group']['selector']
            self.assertEqual(arbiter['complete_responses'],0);self.assertEqual(arbiter['matched_complete'],0)

    def test_missing_and_truncated_ledgers_cannot_erase_paid_requests(self):
        with TemporaryDirectory(prefix='webprm-result-audit-',dir='/tmp') as tmp:
            campaign=Path(tmp);self.fixture(campaign);ledger=campaign/'shared-budget.json'
            ledger.rename(campaign/'preserved-ledger.json')
            with self.assertRaisesRegex(ValueError,'lost.*ledger'):audit(campaign)
            self.put(ledger,dict(cap=15,charged=0,calls=[]))
            with self.assertRaisesRegex(ValueError,'reservation missing'):audit(campaign)

    def test_saved_answer_prompt_verdict_and_reset_tampering_rejected(self):
        with TemporaryDirectory(prefix='webprm-result-audit-',dir='/tmp') as tmp, \
                patch('socket.socket.connect',side_effect=AssertionError('Network forbidden')):
            campaign=Path(tmp);attempt,folders=self.fixture(campaign);folder=folders[0]
            mutations=[
                (folder/'native-evaluation/trajectory.json',lambda x:x[-1].update(answer='Different answer')),
                (folder/'native-evaluation/trajectory.json',lambda x:x.pop()),
                (folder/'api.jsonl',lambda x:x[0]['payload']['messages'][1].update(content='Different prompt')),
                (folder/'native-evaluation/native-verdict.json',lambda x:x.update(source={})),
                (folder/'native-fuzzy.jsonl',lambda x:x[0]['response']['choices'][0]['message'].update(content='incorrect')),
                (folder/'provenance.json',lambda x:x.update(reset_id='different')),
                (folder/'authentication.json',lambda x:x.update(authenticated=False)),
                (attempt/'lease-no_search.json',lambda x:x.update(instance_receipt_sha256='0'*64)),
            ]
            for path,change in mutations:
                original=path.read_text()
                try:
                    data=[json.loads(line) for line in original.splitlines()] if path.suffix=='.jsonl' else json.loads(original)
                    change(data)
                    if path.suffix=='.jsonl':self.journal(path,data)
                    else:self.put(path,data)
                    with self.subTest(path=str(path)),self.assertRaises((ValueError,RuntimeError)):audit(campaign)
                finally:path.write_text(original)

    def test_launched_missing_and_recorded_unattempted_are_distinct(self):
        summary=summarize([dict(selector='no_search',attempted=False,valid=False,success=0,
            error=dict(stage='native_evaluation',type='EvaluationNotReady',message='private'))],{'no_search','gpt4omini'})
        self.assertEqual(summary['totals']['recorded_unattempted'],1)
        self.assertEqual(summary['totals']['attempted_unknown'],1)
        self.assertEqual(summary['totals']['not_launched'],3)
        self.assertEqual(summary['totals']['attempted'],0)
        self.assertEqual(summary['rows'][0]['error_type'],'EvaluationNotReady')
        self.assertEqual(summary['rows'][0]['error_stage'],'native_evaluation')
        self.assertEqual(summary['status'],'partial');self.assertIsNone(summary['overall_success_pct'])

    def test_lost_result_remains_unknown_and_does_not_become_invalid(self):
        with TemporaryDirectory(prefix='webprm-result-audit-',dir='/tmp') as tmp:
            campaign=Path(tmp);_,folders=self.fixture(campaign)
            (folders[0]/'result.json').rename(folders[0]/'preserved-result.json')
            summary=audit(campaign)
            self.assertEqual(summary['totals']['attempted_unknown'],1)
            self.assertEqual(summary['totals']['attempted'],4)
            self.assertEqual(summary['totals']['invalid'],0)
            self.assertIsNone(summary['overall_success_pct'])

    def test_coordinated_success_flip_cannot_override_saved_judge_text(self):
        with TemporaryDirectory(prefix='webprm-result-audit-',dir='/tmp') as tmp, \
                patch('socket.socket.connect',side_effect=AssertionError('Network forbidden')):
            campaign=Path(tmp);_,folders=self.fixture(campaign);folder=folders[1]
            # Change every saved score claim but retain the actual 'incorrect'
            # provider response. Only native text replay detects this mismatch.
            for path in [folder/'result.json',folder/'native-evaluation/native-verdict.json',
                         folder/'native-evaluation/outcome-before-close.json']:
                value=json.loads(path.read_text());value.update(score=1,success=1 if path==folder/'result.json' else True)
                if 'evaluator_details' in value:value['evaluator_details'][0]['score']=1.
                self.put(path,value)
            fuzzy=folder/'native-fuzzy.jsonl';values=[json.loads(line) for line in fuzzy.read_text().splitlines()]
            values[-1]['score']=1;self.journal(fuzzy,values)
            with self.assertRaisesRegex(ValueError,'offline replay'):audit(campaign)

    def test_saved_prompt_is_compared_to_pinned_native_rendering(self):
        from scripts.analyze_webprm_webarena_pilot import replay_native_verdict
        with TemporaryDirectory(prefix='webprm-result-audit-',dir='/tmp') as tmp, \
                patch('socket.socket.connect',side_effect=AssertionError('Network forbidden')):
            campaign=Path(tmp);attempt,folders=self.fixture(campaign);folder=folders[0]
            plan=json.loads((attempt/'worker-plan.json').read_text())
            terminal=json.loads((folder/'native-evaluation/terminal.json').read_text())
            trajectory=json.loads((folder/'native-evaluation/trajectory.json').read_text())
            api=[json.loads(line) for line in (folder/'api.jsonl').read_text().splitlines()]
            payload=api[0]['payload'];payload['messages'][1]['content']+=' Different question'
            with self.assertRaisesRegex(ValueError,'prompt does not match'):
                replay_native_verdict(plan,trajectory,terminal,api[1]['response'],payload)


if __name__=='__main__':unittest.main()
