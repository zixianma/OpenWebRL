"""Offline guards for the original-site deployment fixture; no website started."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/check_webprm_reddit_reset.py'
spec = importlib.util.spec_from_file_location('reddit_reset_fixture', SCRIPT)
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


class RedditResetGuards(unittest.TestCase):
    def receipt(self):
        return {'image_sha256': 'a' * 64, 'instance_id': 'fixture-baseline',
                'state_path': '/tmp/webprm-reset-fixture-baseline',
                'endpoint': 'http://127.0.0.1:29999', 'job_id': '123',
                'service_pid': 200, 'service_start_ticks': 400, 'hostname': 'fixture-host',
                'scope': 'diagnostic', 'build_receipt_sha256': 'b' * 64,
                'state_provenance_sha256': 'c' * 64}

    def test_reset_requires_same_image_and_distinct_state_and_identity(self):
        baseline = self.receipt()
        current = deepcopy(baseline)
        current.update(instance_id='fixture-reset', state_path='/tmp/webprm-reset-fixture-fresh',
                       state_provenance_sha256='d' * 64)
        fixture.validate_reset_identity(baseline, current)
        for field in ['instance_id', 'state_path']:
            wrong = deepcopy(current)
            wrong[field] = baseline[field]
            with self.assertRaises(ValueError):
                fixture.validate_reset_identity(baseline, wrong)
        current['image_sha256'] = 'b' * 64
        with self.assertRaises(ValueError):
            fixture.validate_reset_identity(baseline, current)

    def test_owner_job_host_and_pid_must_match(self):
        instance = self.receipt()
        with patch.dict('os.environ', {'SLURM_JOB_ID': '123'}), \
             patch.object(fixture.socket, 'gethostname', return_value='fixture-host'), \
             patch.object(fixture.os, 'readlink', return_value='net:[100]'), \
             patch.object(fixture, '_process', return_value={'start_ticks': 400, 'state': 'S'}):
            fixture.validate_instance(instance)
            for field, value in [('job_id', '456'), ('hostname', 'other-host'), ('service_start_ticks', 401)]:
                wrong = deepcopy(instance)
                wrong[field] = value
                with self.assertRaises(ValueError):
                    fixture.validate_instance(wrong)

    def test_matching_host_and_job_do_not_hide_a_network_namespace_mismatch(self):
        with patch.dict('os.environ', {'SLURM_JOB_ID': '123'}), \
             patch.object(fixture.socket, 'gethostname', return_value='fixture-host'), \
             patch.object(fixture.os, 'readlink', side_effect=['net:[100]', 'net:[200]']), \
             patch.object(fixture, '_process', return_value={'start_ticks': 400, 'state': 'S'}):
            with self.assertRaisesRegex(ValueError, 'same network namespace'):
                fixture.validate_instance(self.receipt())

    def test_only_literal_loopback_endpoints_are_accepted(self):
        for endpoint in ['http://127.0.0.1:29999', 'http://[::1]:29999']:
            instance = self.receipt()
            instance['endpoint'] = endpoint
            fixture.validate_instance(instance, enforce_owner=False)
        for endpoint in ['http://10.0.0.1:29999', 'http://198.51.100.1:29999',
                         'http://localhost:29999', 'http://fixture-host:29999']:
            instance = self.receipt()
            instance['endpoint'] = endpoint
            with self.assertRaisesRegex(ValueError, 'literal loopback'):
                fixture.validate_instance(instance, enforce_owner=False)

    def test_partial_image_diagnostic_baseline_cannot_authorize_mutation_or_reset(self):
        diagnostic = self.receipt()
        diagnostic['scope'] = 'diagnostic'
        status = fixture.validate_fixture_scope(diagnostic, 'baseline')
        self.assertFalse(status['eligible_as_reset_baseline'])
        for phase in ['mutate', 'after-reset']:
            with self.assertRaisesRegex(ValueError, 'Partial-image'):
                fixture.validate_fixture_scope(diagnostic, phase)
            with self.assertRaisesRegex(ValueError, 'Partial-image'):
                fixture.validate_fixture_scope(dict(self.receipt(), scope='original'), phase, [status])

    def test_missing_or_unknown_scope_is_rejected(self):
        for value in [None, '', 'production', 'original-ish']:
            instance = self.receipt()
            instance['scope'] = value
            with self.assertRaisesRegex(ValueError, 'explicitly'):
                fixture.validate_fixture_scope(instance, 'baseline')
        instance.pop('scope')
        with self.assertRaisesRegex(ValueError, 'explicitly'):
            fixture.validate_fixture_scope(instance, 'baseline')

    def test_predecessor_eligibility_cannot_be_inferred_from_missing_metadata(self):
        instance = dict(self.receipt(), scope='original')
        eligible = {'eligible_as_reset_baseline': True, 'instance': instance}
        fixture.validate_fixture_scope(instance, 'mutate', [eligible])
        for predecessor in [{}, {'instance': instance}, {'eligible_as_reset_baseline': True},
                            dict(eligible, instance=self.receipt())]:
            with self.assertRaisesRegex(ValueError, 'explicitly establish'):
                fixture.validate_fixture_scope(instance, 'mutate', [predecessor])

    def test_original_provenance_dispatch_propagates_unverified_failure(self):
        instance = dict(self.receipt(), scope='original')
        with patch('openwebrl.webprm_reddit_provenance.validate_original_provenance',
                   side_effect=ValueError('Controlled original provenance rejection')) as verify:
            with self.assertRaisesRegex(ValueError, 'provenance rejection'):
                fixture.validate_instance(instance, enforce_owner=False)
            verify.assert_called_once_with(instance)
        summary = {'verified_original_layers': 29, 'build_receipt_sha256': 'b' * 64}
        with patch('openwebrl.webprm_reddit_provenance.validate_original_provenance',
                   return_value=summary) as verify:
            observed = fixture.validate_instance(instance, enforce_owner=False)
            self.assertEqual(observed['original_provenance'], summary)
            verify.assert_called_once_with(instance)

    def test_resource_diagnostics_count_failures_but_cap_paths_and_exclude_other_origins(self):
        evidence = {'http_failures_by_type_status': {}, 'http_failure_paths': [],
                    'request_failures_by_type': {}, 'request_failure_paths': []}
        origin = ('http', '127.0.0.1:29999')
        for index in range(25):
            request = SimpleNamespace(resource_type='image', failure='net::ERR_ABORTED',
                                      url=f'http://127.0.0.1:29999/image/{index}?private=excluded')
            response = SimpleNamespace(request=request, url=request.url, status=404)
            fixture.record_response_failure(evidence, origin, response)
            fixture.record_request_failure(evidence, origin, request)
        for url in ['https://outside.test/image', 'http://127.0.0.1:30000/image']:
            request = SimpleNamespace(resource_type='image', failure='net::ERR_ABORTED', url=url)
            fixture.record_response_failure(evidence, origin, SimpleNamespace(request=request, url=url, status=404))
            fixture.record_request_failure(evidence, origin, request)
        request = SimpleNamespace(resource_type='document', url='http://127.0.0.1:29999/')
        fixture.record_response_failure(evidence, origin, SimpleNamespace(request=request, url=request.url, status=200))
        self.assertEqual(evidence['http_failures_by_type_status'], {'image:404': 25})
        self.assertEqual(evidence['request_failures_by_type'], {'image': 25})
        self.assertEqual(len(evidence['http_failure_paths']), 20)
        self.assertEqual(len(evidence['request_failure_paths']), 20)
        self.assertNotIn('private', json.dumps(evidence))
        self.assertNotIn('outside.test', json.dumps(evidence))

    def test_reset_requires_common_build_and_distinct_state_provenance(self):
        baseline = self.receipt()
        current = deepcopy(baseline)
        current.update(instance_id='fresh', state_path='/tmp/fresh-provenance',
                       state_provenance_sha256='d' * 64)
        fixture.validate_reset_identity(baseline, current)
        for field, value in [('build_receipt_sha256', 'e' * 64),
                             ('state_provenance_sha256', baseline['state_provenance_sha256']),
                             ('build_receipt_sha256', None)]:
            wrong = deepcopy(current)
            wrong[field] = value
            with self.assertRaises(ValueError):
                fixture.validate_reset_identity(baseline, wrong)

    def test_mutation_identity_is_durable_before_cleanup_failure(self):
        output = Path(tempfile.mkdtemp(prefix='webprm-reset-guard-', dir='/tmp'))
        report = {'phase': 'mutate', 'status': 'passed', 'post_path': '/f/books/123/fixture',
                  'run_id': 'test-run', 'instance': self.receipt()}
        def stalled_cleanup():
            saved = json.loads((output / 'mutation-outcome.json').read_text())
            self.assertEqual(saved['post_path'], report['post_path'])
            self.assertTrue(saved['cleanup_pending'])
            self.assertEqual(saved['status'], 'passed')
            raise TimeoutError('Controlled browser cleanup failure')
        browser = SimpleNamespace(close=lambda: self.fail('Browser close must follow context close'))
        with self.assertRaises(TimeoutError):
            fixture.close_browser_phase(report, output, SimpleNamespace(close=stalled_cleanup), browser)
        self.assertEqual(json.loads((output / 'mutation-outcome.json').read_text())['post_path'], report['post_path'])

    def test_atomic_mutation_claim_blocks_later_or_concurrent_duplicate_submission(self):
        output = Path(tempfile.mkdtemp(prefix='webprm-reset-claim-', dir='/tmp'))
        baseline = {'baseline_directory': str(output), 'run_id': 'test-run', 'synthetic_title': 'fixture'}
        receipt = output / 'report.json'
        fixture.write_new(receipt, baseline)
        fixture.reject_previous_mutation(receipt)
        fixture.reserve_mutation(receipt, output / 'mutation-output', baseline, self.receipt())
        with self.assertRaisesRegex(ValueError, 'mutation attempt already exists'):
            fixture.reject_previous_mutation(receipt)
        with self.assertRaises(FileExistsError):
            fixture.reserve_mutation(receipt, output / 'second-output', baseline, self.receipt())
        copied = Path(tempfile.mkdtemp(prefix='webprm-reset-copy-', dir='/tmp')) / 'report.json'
        fixture.write_new(copied, baseline)
        with self.assertRaisesRegex(ValueError, 'original baseline receipt'):
            fixture.mutation_claim_path(copied)

    def test_protected_state_path_and_credentials_rejected(self):
        for field, value in [('state_path', '/gpfs/home/zixianma/unsafe'),
                             ('endpoint', 'http://user:password@localhost:29999'),
                             ('endpoint', 'http://localhost:29999/?key=value'),
                             ('image_sha256', 'unverified')]:
            wrong = self.receipt()
            wrong[field] = value
            with self.assertRaises(ValueError):
                fixture.validate_instance(wrong, enforce_owner=False)

    @unittest.skipUnless(Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-baselines-20261009/sources/arbiter').exists(),
                         'Pinned release is unavailable')
    def test_native_auth_contract_loads_without_logging_or_importing_settings(self):
        source = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-baselines-20261009/sources/arbiter')
        account, hashes = fixture.load_auth(source)
        self.assertEqual(set(account), {'username', 'password'})
        self.assertTrue(all(account.values()))
        self.assertEqual(len(hashes), 2)


if __name__ == '__main__':
    unittest.main()
