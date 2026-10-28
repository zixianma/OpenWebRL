"""Controlled factor changes and fail-closed, no-duplicate paid dispatch."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import URLError

from scripts.probe_webprm_reproduction import (BATCHES, FOUR_O, MINI, SOURCE,
    change_payload, digest, execute, parse_result, read, upstream_defaults,
    validate_approval, write_once)


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir='/tmp', prefix='webprm-probe-test-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.network = patch('socket.socket.connect', side_effect=AssertionError('No network in tests'))
        self.network.start()
        self.addCleanup(self.network.stop)
        self.payload = dict(model=FOUR_O, max_tokens=2048, temperature=1.0, top_p=.9,
                            messages=[dict(role='system', content='Original rule'),
                                      dict(role='user', content='Saved observation')])

    def test_each_treatment_changes_only_its_prespecified_variable(self):
        original = deepcopy(self.payload)
        for batch in BATCHES:
            self.assertEqual(change_payload(self.payload, batch, 'control'), original)
            result = change_payload(self.payload, batch, 'treatment')
            changed_keys = {key for key in original if result[key] != original[key]}
            expected = 'model' if batch.endswith('_model') else 'temperature' if batch.endswith('_temperature') else 'messages'
            self.assertEqual(changed_keys, {expected})
            if expected == 'messages':
                self.assertEqual(result['messages'][1:], original['messages'][1:])
                self.assertTrue(result['messages'][0]['content'].startswith(original['messages'][0]['content']))
        self.assertEqual(self.payload, original)

    def test_pending_or_changed_plan_cannot_reach_credentials(self):
        with self.assertRaisesRegex(ValueError, 'Explicit approval'):
            validate_approval({'resources': {}}, {'status': 'pending'})
        with self.assertRaisesRegex(ValueError, 'Explicit approval'):
            validate_approval({'resources': {}}, dict(status='approved', plan_sha256='other', exact_user_approval='yes'))

    @unittest.skipUnless(SOURCE.exists(), 'Pinned upstream source missing')
    def test_real_released_defaults_are_mini_temperature_one_top_p_point_nine(self):
        default = upstream_defaults()['defaults']
        self.assertEqual(default['model'], 'gpt-4o-mini')
        self.assertEqual(default['gen_config']['temperature'], 1.0)
        self.assertEqual(default['gen_config']['top_p'], .9)

    @unittest.skipUnless(SOURCE.exists(), 'Pinned upstream source missing')
    def test_malformed_terminal_is_invalid_instead_of_a_correct_negative(self):
        def response(text):
            return dict(choices=[dict(finish_reason='stop', message=dict(content=text))])
        invalid = parse_result(self.payload, response('not json'), purpose='terminal')
        self.assertFalse(invalid['parse_valid'])
        negative = parse_result(self.payload, response(json.dumps(dict(thought='Pending form', action=None, confidence=1))), purpose='terminal')
        self.assertTrue(negative['parse_valid'])
        self.assertFalse(negative['completion_claim'])

    def execution_plan(self):
        call = dict(call_id='one', batch='terminal_model', case_id='fixture', variant='control',
                    purpose='terminal', payload=self.payload, expected_complete=False)
        plan = dict(batches=['terminal_model'], calls=[call], env_file='/tmp/no-real-credentials',
                    api_base='https://api.openai.com/v1',
                    prices_usd_per_token={FOUR_O: [5e-6, 15e-6], MINI: [.15e-6, .6e-6]},
                    resources=dict(api_usd_cap=5.0, cumulative_wall_seconds=1800))
        write_once(self.root / 'plan.json', plan)
        write_once(self.root / 'approval.json', dict(status='pending'))
        return plan

    def test_unknown_claim_cannot_dispatch_again(self):
        self.execution_plan()
        write_once(self.root / 'calls/one/claim.json', dict(previous_dispatch='unknown'))
        with patch('scripts.probe_webprm_reproduction.validate_approval'), \
             patch('openwebrl.webprm_runtime.load_api_credentials', return_value='unused'), \
             patch('openwebrl.webprm_worker.completion', side_effect=AssertionError('No duplicate call')):
            with self.assertRaisesRegex(ValueError, 'must not be repeated'):
                execute(self.root, self.root / 'approval.json', 'terminal_model')

    def test_timeout_retains_reservation_and_rerun_never_repeats_attempt(self):
        self.execution_plan()
        with patch('scripts.probe_webprm_reproduction.validate_approval'), \
             patch('openwebrl.webprm_runtime.load_api_credentials', return_value='unused'), \
             patch('openwebrl.webprm_worker.load_api_credentials', return_value='unused'), \
             patch('openwebrl.webprm_worker.urlopen', side_effect=URLError('synthetic timeout')) as provider:
            execute(self.root, self.root / 'approval.json', 'terminal_model')
            budget = read(self.root / 'budget.json')
            self.assertGreater(budget['charged'], 0)
            self.assertEqual(budget['calls'][0]['state'], 'reserved')
            execute(self.root, self.root / 'approval.json', 'terminal_model')
            self.assertEqual(provider.call_count, 1)
            self.assertEqual(read(self.root / 'budget.json'), budget)
        self.assertFalse(read(self.root / 'terminal_model-summary.json')['all_responses_recorded'])


if __name__ == '__main__':
    unittest.main()
