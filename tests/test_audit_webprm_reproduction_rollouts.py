"""Financial/payload tampering checks for the standalone offline auditor."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from scripts import audit_webprm_reproduction_rollouts as audit
from openwebrl.webprm_arbiter_harness import load_arbiter_harness
from openwebrl.webprm_reproduction_rollout import contract, CONDITIONS

SOURCE='/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-baselines-20261009/sources/arbiter'
MINI='gpt-4o-mini-2024-07-18'
LARGE='gpt-4o-2024-05-13'


def fixture(count=1, reserved=False, condition=CONDITIONS[0]):
    plan=dict(arbiter_source=SOURCE, no_search_prompt_version='single_blocks_envelope_v2',
        reproduction_diagnostic=contract(condition), prices_usd_per_token={MINI:[.15e-6,.60e-6], LARGE:[5e-6,15e-6]})
    harness=load_arbiter_harness(SOURCE)
    system,user=harness.build_terminal_prompt(intent='Save the change',observation='Pending form',
        current_url='http://127.0.0.1',trajectory='',state_diff='',previous_terminal_action=None)
    payload=dict(model=MINI,messages=[dict(role='system',content=system+plan['reproduction_diagnostic']['terminal_system_suffix']),
        dict(role='user',content=user)],temperature=1.,top_p=.9,max_tokens=2048)
    cell='webarena_lite_097/'+LARGE+'__no_search'
    journal=[];broker=[];budget=dict(cap=15.,charged=0.,calls=[])
    for i in range(count):
        meta=dict(cell_id=cell,identity_sha256=audit.digest(plan),local=False,model=MINI,purpose='terminal',
            request_id=str(i),policy_sha256='policy',payload_sha256=audit.digest(payload))
        broker.append(dict(meta,event='dispatch_started'))
        journal.append(dict(event='request',reservation=i,payload=deepcopy(payload),purpose='terminal',local=False))
        financial=dict(id=i,model=MINI,reserved=.1,state='reserved' if reserved else 'settled')
        if reserved:
            broker.append(dict(meta,event='dispatch_failed',error_type='TimeoutError'))
            budget['charged']+=.1
        else:
            usage=dict(prompt_tokens=100,completion_tokens=10)
            response=dict(model=MINI,usage=usage,choices=[dict(finish_reason='stop',message=dict(content='{"action":null}'))])
            journal.append(dict(event='response',reservation=i,response=response))
            broker.append(dict(meta,event='dispatch_completed',reservation_id=i,reservation_state='settled'))
            financial.update(usage=deepcopy(usage),charged=21e-6);budget['charged']+=21e-6
        budget['calls'].append(financial)
    return plan,cell,journal,broker,budget


class AuditRolloutTests(unittest.TestCase):
    def setUp(self):
        guard=patch('socket.socket.connect',side_effect=AssertionError('No network in offline audit'))
        guard.start();self.addCleanup(guard.stop)

    def test_settled_call_and_identical_repeated_payloads_join_distinct_reservations(self):
        for n in (1,2):
            args=fixture(n);used=set();out=audit.audit_calls(*args,used)
            self.assertEqual(out['dispatches'],n);self.assertEqual(used,set(range(n)))
            self.assertEqual([p['broker']['reservation_id'] for p in out['pairs']],list(range(n)))

    def test_changed_prompt_condition_is_checked_on_actual_request(self):
        args=fixture(condition=CONDITIONS[1]);audit.audit_calls(*args,set())
        args[2][0]['payload']['messages'][0]['content']+='Unapproved extra rule'
        with self.assertRaisesRegex(ValueError,'prompt'):audit.audit_calls(*args,set())

    def test_wrong_helper_model_and_sampling_are_rejected(self):
        for key,value in [('model',LARGE),('temperature',0),('max_tokens',4096)]:
            args=fixture();args[2][0]['payload'][key]=value
            with self.assertRaises(ValueError):audit.audit_calls(*args,set())

    def test_missing_or_forged_broker_completion_is_rejected(self):
        args=fixture();args[3].pop()
        with self.assertRaisesRegex(ValueError,'unresolved'):audit.audit_calls(*args,set())
        args=fixture();args[3][-1]['reservation_id']=7
        with self.assertRaisesRegex(ValueError,'settlement'):audit.audit_calls(*args,set())

    def test_changed_paid_charge_or_usage_is_rejected(self):
        args=fixture();args[4]['calls'][0]['charged']=0
        with self.assertRaisesRegex(ValueError,'charge'):audit.audit_calls(*args,set())
        args=fixture();args[2][-1]['response']['usage']['prompt_tokens']=101
        with self.assertRaisesRegex(ValueError,'usage'):audit.audit_calls(*args,set())

    def test_duplicate_reservations_cannot_hide_an_additional_attempt(self):
        args=fixture(2);args[2][2]['reservation']=0
        with self.assertRaisesRegex(ValueError,'duplicate'):audit.audit_calls(*args,set())

    def test_unknown_provider_outcome_keeps_full_reservation(self):
        args=fixture(reserved=True);out=audit.audit_calls(*args,set())
        self.assertEqual(out['incomplete_reservations'],1)
        self.assertEqual(out['failures'],1)
        self.assertEqual(args[4]['charged'],.1)
        self.assertIsNone(out['pairs'][0]['response'])

    def test_admission_failure_still_counts_dispatch(self):
        plan,cell,_,broker,_=fixture(reserved=True)
        out=audit.audit_calls(plan,cell,[],broker,dict(calls=[]),set())
        self.assertEqual(out['dispatches'],1)
        self.assertEqual(out['journal_requests'],0)

    def test_native_replay_uses_real_saved_artifact_layout_and_rejects_verdict_tampering(self):
        # Entirely synthetic page and actor. Exercise the actual worker save
        # path, recorder and evaluator replay without a browser or API request.
        from test_webprm_webarena_worker import FakeEnv, Scripted, wrapped
        from openwebrl.webprm_webarena_worker import ViewerActor, ViewerSelector, run_loop
        from openwebrl.webprm_arbiter_harness import native_execution_code
        from openwebrl.webprm_webarena import load_native_evaluator
        from openwebrl.webprm_evaluator_transcript import EvaluationRecorder
        from scripts.audit_webprm_table5_cell import audit_native
        root=Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/reproduction-rollouts-20-v4')
        plan=json.loads((root/'plan-released_helper_defaults.json').read_text())
        with tempfile.TemporaryDirectory(dir='/tmp') as temporary:
            folder=Path(temporary);task=wrapped();request=Scripted(count=1)
            config=dict(plan,actor_model=MINI,selector='no_search',reset_id='synthetic-reset',
                identity_sha256=audit.digest(plan),cell_name=MINI+'__no_search',deadline=time.time()+60)
            # Routing is separately covered above; this fixture isolates native
            # persistence/replay using the original mini helper behavior.
            config.pop('reproduction_diagnostic')
            actor=ViewerActor(config,folder/'unused',request=request)
            selector=ViewerSelector('no_search',config,folder/'unused',request=request)
            env=FakeEnv(folder);original_step=env.step;dispatch={}
            import io
            from PIL import Image
            png=io.BytesIO();Image.new('RGB',(2,2)).save(png,format='PNG')
            env.page.screenshot=lambda **kwargs:png.getvalue()
            def step(action):
                code=native_execution_code(SOURCE,actor.harness.parse_action(action),{},key_mapping=actor.harness.key_mapping)
                dispatch.clear();dispatch.update(action=action,code=code,code_sha256=__import__('hashlib').sha256(code.encode()).hexdigest(),element_bounds={})
                return original_step(action)
            env.step=step
            recorder=EvaluationRecorder(folder/'native-evaluator-transcript.json',plan['evaluator_observations'])
            native=load_native_evaluator(SOURCE,scoring_protocol=plan['evaluator_protocol'],observation_recorder=recorder)
            result=run_loop(task,'no_search',config,env,actor,selector,
                lambda environment,obs:dict(text='Synthetic page',url=environment.page.url,last_action_error=obs.get('last_action_error')),
                native.evaluate,folder,dispatch_evidence=dispatch)
            self.assertTrue(result['valid']);self.assertEqual(result['success'],1)
            instance=dict(synthetic=True)
            provenance=dict(identity_sha256=audit.digest(plan),original_instance=instance,reset_id='synthetic-reset',
                actor=actor.harness.manifest,observation=plan['source_manifests']['arbiter_observation'],critic=None)
            (folder/'provenance.json').write_text(json.dumps(provenance))
            (folder/'authentication.json').write_text(json.dumps(dict(authenticated=True,state_provenance_sha256='synthetic-reset')))
            paths=audit.native_paths(folder)
            for key,path in paths.items():
                if key!='native_fuzzy':self.assertTrue(path.exists(),key)
            outcome,proof=audit_native(task,plan,paths,result,dict(pairs=[]),instance)
            self.assertEqual(outcome['success'],1);self.assertTrue(proof['verified'])
            verdict=json.loads(paths['verdict'].read_text());verdict['success']=False
            paths['verdict'].write_text(json.dumps(verdict))
            with self.assertRaises(ValueError):audit_native(task,plan,paths,result,dict(pairs=[]),instance)


class CumulativeTimeTests(unittest.TestCase):
    def test_replacement_charges_failed_time_inside_original_cap(self):
        result=audit.audit_time(dict(elapsed_seconds=100),dict(resources=dict(total_seconds=14340)),
            dict(prior_scheduler_seconds=5,original_total_cap_seconds=14400))
        self.assertEqual(result['cumulative_elapsed_seconds'],105)

    def test_replacement_cannot_reset_four_hour_budget(self):
        with self.assertRaises(ValueError):
            audit.audit_time(dict(elapsed_seconds=100),dict(resources=dict(total_seconds=14400)),
                dict(prior_scheduler_seconds=5,original_total_cap_seconds=14400))

    def test_current_attempt_cannot_overrun_its_smaller_limit(self):
        with self.assertRaises(ValueError):
            audit.audit_time(dict(elapsed_seconds=14341),dict(resources=dict(total_seconds=14340)),
                dict(prior_scheduler_seconds=5,original_total_cap_seconds=14400))


if __name__=='__main__':unittest.main()
