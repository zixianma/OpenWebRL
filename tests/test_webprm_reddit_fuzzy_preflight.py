"""Synthetic broker replies only; no website, model, provider or allocation."""
from contextlib import ExitStack
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

from openwebrl.webprm_broker import payload_sha256
from openwebrl.webprm_reddit_pilot_preflight import (
    NATIVE_FUZZY_PREFLIGHT, native_fuzzy_preflight_cell, validate_native_fuzzy)
from openwebrl.webprm_webarena import EvaluationNotReady
from scripts import launch_webprm_reddit_pilot as pilot

SOURCE = '/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-baselines-20261009/sources/arbiter'


class FuzzyPreflightTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix='wfuzzy-preflight-', dir='/tmp'))
        self.config = dict(authorization_status='approved', arbiter_source=SOURCE,
            fuzzy_enabled=True, fuzzy_model='gpt-4o-2024-05-13',
            fuzzy_protocol='webarena_fuzzy_gpt4o_20240513_v1', fuzzy_model_change_approved=True,
            fuzzy_max_input_tokens=8192, fuzzy_max_calls=1, deadline=time.time()+60,
            identity_sha256='a'*64, api_usd_cap=15,
            budget_path=str(self.root/'campaign-shared-budget.json'))
        self.calls = []
        network = patch('socket.socket.connect', side_effect=AssertionError('Network forbidden'))
        network.start(); self.addCleanup(network.stop)

    def client(self, text='correct', error=None, receipt_change=None):
        def request(config, payload, journal, *, purpose):
            self.calls.append((deepcopy(config), deepcopy(payload), str(journal), purpose))
            self.assertTrue((self.root/'native-fuzzy-preflight/input.json').is_file())
            if error:
                raise error
            response = dict(model=config['fuzzy_model'],
                choices=[dict(finish_reason='stop', message=dict(role='assistant', content=text))],
                usage=dict(prompt_tokens=120, completion_tokens=2))
            response['_webprm_broker'] = dict(schema_version=1,
                budget_owner='webprm_worker.completion', model=response['model'], purpose=purpose,
                local=False, reservation_id='test-reservation', reservation_state='settled',
                usage=response['usage'], cell_id='native_fuzzy_preflight', request_id='test-request',
                identity_sha256='a'*64, journal_sha256='b'*64, budget_ledger_sha256='c'*64,
                policy_sha256='d'*64, payload_sha256=payload_sha256(payload))
            if receipt_change:
                response['_webprm_broker'].update(receipt_change)
            return response
        request._webprm_broker_client = True
        request._webprm_budget_owner = 'webprm_worker.completion'
        return request

    def test_native_request_parser_and_single_budget_owner(self):
        original = deepcopy(self.config)
        proof = validate_native_fuzzy(self.config, self.client(), self.root)
        self.assertEqual(self.config, original)
        self.assertEqual((proof['status'], proof['score'], proof['provider_calls']), ('passed',1,1))
        self.assertFalse(proof['primary_admitted'])
        self.assertEqual(len(self.calls), 1)
        config, payload, _, purpose = self.calls[0]
        self.assertEqual(config['budget_path'], self.config['budget_path'])
        self.assertEqual(purpose, 'native_fuzzy')
        self.assertEqual(payload, dict(model='gpt-4o-2024-05-13', temperature=0, top_p=1.0,
            max_tokens=768, messages=[dict(role='system',content='You are a helpful assistant'),
            dict(role='user',content='Help a teacher to grade the answer of a student given a question. Keep in mind that the student may use different phrasing or wording to answer the question. The goal is to evaluate whether the answer is semantically equivalent to the reference answer.\nquestion: Which synthetic shape and color are displayed?\nreference answer: blue triangle\nall the string \'N/A\' that you see is a special sequence that means \'not achievable\'\nstudent answer: blue triangle\nConclude the judgement by \'correct\', \'incorrect\', or \'partially correct\'. Only output one of these options, and nothing else.')]))
        events = [json.loads(x)['event'] for x in
            (self.root/'native-fuzzy-preflight/native-fuzzy.jsonl').read_text().splitlines()]
        self.assertEqual(events, ['broker_request','dispatch','response','broker_settled','native_verdict'])
        self.assertFalse((self.root/'campaign-shared-budget.json').exists())
        self.assertEqual(json.loads((self.root/'native-fuzzy-preflight/passed.json').read_text()),proof)

    def test_disabled_pending_or_unapproved_model_never_dispatches(self):
        for change in (dict(authorization_status='pending'), dict(fuzzy_enabled=False),
                       dict(fuzzy_model_change_approved=False),dict(fuzzy_model='gpt-4o')):
            with self.subTest(change=change), self.assertRaises(EvaluationNotReady):
                validate_native_fuzzy(dict(self.config, **change), self.client(), self.root)
        self.assertEqual(self.calls, [])
        self.assertFalse((self.root/'native-fuzzy-preflight').exists())

    def test_wrong_expected_verdict_is_preserved_failure_without_retry(self):
        with self.assertRaisesRegex(EvaluationNotReady, 'expected verdict'):
            validate_native_fuzzy(self.config, self.client('incorrect'), self.root)
        self.assertEqual(len(self.calls), 1)
        failed=json.loads((self.root/'native-fuzzy-preflight/failed.json').read_text())
        self.assertEqual(failed['provider_calls'],1)
        self.assertFalse(failed['primary_admitted'])
        self.assertFalse((self.root/'native-fuzzy-preflight/passed.json').exists())
        with self.assertRaises(FileExistsError):
            validate_native_fuzzy(self.config, self.client(), self.root)
        self.assertEqual(len(self.calls), 1)

    def test_parser_error_is_preserved_without_retry(self):
        with self.assertRaises(EvaluationNotReady):
            validate_native_fuzzy(self.config, self.client('undetermined'), self.root)
        self.assertEqual(len(self.calls), 1)
        self.assertTrue((self.root/'native-fuzzy-preflight/failed.json').is_file())

    def test_provider_error_is_preserved_without_retry(self):
        with self.assertRaises(EvaluationNotReady):
            validate_native_fuzzy(self.config, self.client(error=OSError('synthetic error')), self.root)
        self.assertEqual(len(self.calls), 1)
        self.assertNotIn('synthetic error',(self.root/'native-fuzzy-preflight/failed.json').read_text())
        self.assertTrue((self.root/'native-fuzzy-preflight/failed.json').is_file())

    def test_unsettled_budget_receipt_cannot_pass(self):
        with self.assertRaisesRegex(EvaluationNotReady, 'budget receipt'):
            validate_native_fuzzy(self.config,self.client(receipt_change=dict(reservation_state='reserved')),self.root)
        self.assertEqual(len(self.calls),1)
        self.assertFalse((self.root/'native-fuzzy-preflight/passed.json').exists())

    def test_host_registration_allows_only_the_exact_paid_fuzzy_route(self):
        cell=native_fuzzy_preflight_cell(self.config,'/not/read/credentials',self.root)
        self.assertEqual(cell['allowed_routes'],[['gpt-4o-2024-05-13','native_fuzzy']])
        self.assertEqual(cell['config']['env_file'],'/not/read/credentials')
        self.assertEqual(cell['config']['budget_path'],self.config['budget_path'])
        self.assertEqual(cell['journal'],str(self.root/'native-fuzzy-preflight-api.jsonl'))
        self.assertNotIn('env_file',self.config)
        self.assertEqual(self.calls,[])

    def test_actual_runner_aborts_before_primary_claim_and_owns_cleanup(self):
        self.runner_abort_case()

    def test_cpu_runner_starts_zero_models_and_stops_before_primary_on_failed_controls(self):
        self.runner_abort_case(cpu_only=True)

    def runner_abort_case(self,cpu_only=False):
        # Fake startup dependencies, but execute the real runner, broker-cell
        # registration and NativeFuzzyJudge gate. No child process is launched.
        scratch=self.root/'scratch'; scratch.mkdir()
        (self.root/'browser-preflight').mkdir()
        (self.root/'browser-preflight/receipt.json').write_text('{"status":"passed"}')
        if cpu_only:self.config['evaluator_protocol']=pilot.GUARDED_EVALUATOR
        plan=self.root/'pending-plan.json'; plan.write_text(json.dumps(self.config))
        models={} if cpu_only else {name:dict(served_model=name) for name in ('shepherd8b','arbiter7b','piotr_selectionarm')}
        proposal=dict(resource_profile=pilot.CPU_VALIDATION_PROFILE if cpu_only else pilot.CORRECTED_PROFILE,
            resources=pilot.CPU_VALIDATION_RESOURCES if cpu_only else pilot.CORRECTED_RESOURCES,
            limits=pilot.CPU_VALIDATION_LIMITS if cpu_only else pilot.CORRECTED_LIMITS,
            rows=['no_search'] if cpu_only else list(pilot.ROWS),
            worker_plan=str(plan),models=models,campaign_root=str(self.root/'campaign'),
            activation_changes=dict(worker_scientific_fields={k:v for k,v in self.config.items() if k.startswith('fuzzy_')}),
            env_file='/not/read/credentials',cell_routes={'no_search':[['gpt-4o-2024-05-13','native_fuzzy']]},
            broker_routes=[dict(model='gpt-4o-2024-05-13',purpose='native_fuzzy',local=False,
                max_input_tokens=8192,max_output_tokens=768)])
        context=dict(deadline=time.time()+60,job_id='42',prior_retained_scratch_bytes=0,
            reuse_build_receipt=str(self.root/'existing-build/receipt.json'),remaining_rows=['no_search'],
            remaining_paid_requests=98 if cpu_only else 575,remaining_local_requests=0 if cpu_only else 400)
        child=Mock(); child.poll.return_value=None
        preparation=Mock(); lease=Mock(); broker=Mock(); broker.close.return_value=dict(closed=True)
        registration={}
        check_output=pilot.subprocess.check_output
        def broker_factory(**kwargs):
            registration.update(kwargs); result=Mock();result.start.return_value=broker;return result
        with ExitStack() as stack:
            for target, replacement in [('RUNTIME',self.root),('SCRATCH',self.root)]:
                stack.enter_context(patch.object(pilot,target,replacement))
            stack.enter_context(patch.object(pilot.os,'setsid'))
            stack.enter_context(patch.object(pilot,'environment',return_value={}))
            stack.enter_context(patch.object(pilot,'reference',return_value=preparation))
            stack.enter_context(patch.object(pilot,'_command',return_value=child))
            stack.enter_context(patch.object(pilot,'_wait'))
            stop=stack.enter_context(patch.object(pilot,'_stop'))
            model_command=stack.enter_context(patch.object(pilot,'model_command',return_value=['never-executed']))
            stack.enter_context(patch.object(pilot,'piotr_compatible_config',return_value={}))
            stack.enter_context(patch.object(pilot.subprocess,'check_output',
                side_effect=lambda args,**kwargs:'0 scratch' if args[0]=='du' else check_output(args,**kwargs)))
            stack.enter_context(patch.object(pilot,'urlopen',side_effect=lambda *a,**k:io.StringIO(json.dumps(dict(data=[dict(id=x) for x in models])))))
            stack.enter_context(patch('scripts.runtime_ports.lease_ports',return_value=(lease,20000)))
            stack.enter_context(patch('openwebrl.webprm_reddit_pilot_preflight.validate_local_critics',return_value=dict(status='passed')))
            stack.enter_context(patch.object(pilot,'BrokerProcess',side_effect=broker_factory))
            stack.enter_context(patch.object(pilot,'make_broker_client',return_value=self.client('incorrect')))
            with self.assertRaisesRegex(EvaluationNotReady,'positive/negative' if cpu_only else 'expected verdict'):
                pilot._runner(proposal,context,self.root,scratch)
        self.assertEqual(len(self.calls),8 if cpu_only else 1)
        self.assertEqual(registration['cells']['native_fuzzy_preflight']['allowed_routes'],
                         [['gpt-4o-2024-05-13','native_fuzzy']])
        self.assertEqual(registration['max_paid_dispatches'],98 if cpu_only else 575)
        self.assertEqual(registration['max_local_dispatches'],0 if cpu_only else 400)
        self.assertEqual(registration['cells']['native_fuzzy_preflight']['config']['env_file'],'/not/read/credentials')
        preparation.prepare.assert_not_called()
        self.assertEqual(list(self.root.glob('worker-launch-*.json')),[])
        broker.close.assert_called_once();lease.close.assert_called_once()
        self.assertEqual(sum(call.args==(child,) for call in stop.call_args_list),1 if cpu_only else 4)
        if cpu_only:model_command.assert_not_called()


if __name__ == '__main__':
    unittest.main()
