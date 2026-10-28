"""Offline site-interface fixtures; no actual sites, browsers, APIs or allocations."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import socket
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from openwebrl.webprm_lifecycle import _process
from openwebrl.webprm_webarena import EvaluationNotReady, digest
from openwebrl.webprm_webarena_episode import EpisodeHooks, TaskEnvelope, open_authenticated_browser
from openwebrl.webprm_webarena_sites import (INSTANCE_SCHEMA, SITE_RUNTIME_CONFIGURATION,
    SITE_ADAPTERS, NO_LOGIN, SiteAdapter, site_adapter, registry_status, site_authentication,
    validate_original_site_receipt)
from openwebrl.webprm_webarena_worker import (claim_lease, protocol_definition,
    validate_protocol_contract, broker_request)

SOURCE = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-baselines-20261009/sources/arbiter')
ENDPOINTS = {'map': 'http://127.0.0.1:31991', 'reddit': 'http://127.0.0.1:31992'}


def task(site='map', storage=False):
    return TaskEnvelope.from_release(dict(task_id=999, intent='Synthetic interface fixture',
        sites=[site], start_url='__' + site.upper() + '__',
        storage_state='./.auth/synthetic.json' if storage else None,
        eval={'eval_types': ['string_match'], 'reference_answers': {'exact_match': '0'}}), ENDPOINTS)


class SiteInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.network = patch('socket.socket.connect', side_effect=AssertionError('Network forbidden'))
        self.network.start(); self.addCleanup(self.network.stop)
        self.tmp = TemporaryDirectory(prefix='webprm-site-interface-', dir='/tmp')
        self.addCleanup(self.tmp.cleanup); self.root = Path(self.tmp.name)
        self.env = patch.dict(os.environ, {'SLURM_JOB_ID': 'offline-fixture-not-an-allocation'})
        self.env.start(); self.addCleanup(self.env.stop)
        self.config = dict(site_runtime=deepcopy(SITE_RUNTIME_CONFIGURATION), endpoints=ENDPOINTS,
            actor_model='gpt-4o-mini-2024-07-18', arbiter_source=str(SOURCE))
        self.validator = Mock(return_value={'synthetic_only': True})

    def fixture_adapter(self):
        # This in-memory patch exists only in tests. Production receipt/plan JSON
        # cannot supply executable validators or enable the unavailable Map adapter.
        return patch('openwebrl.webprm_webarena_sites.SITE_ADAPTERS', dict(SITE_ADAPTERS,
            map=SiteAdapter('map', NO_LOGIN, self.validator)))

    def receipt(self):
        state = self.root / 'private-map-state'; state.mkdir()
        proof = self.root / 'synthetic-control.json'; proof.write_text('{"fixture_only":true}')
        ref = dict(path=str(proof), sha256=hashlib.sha256(proof.read_bytes()).hexdigest())
        base = dict(fixture_only=True, no_original_website=True)
        provenance = dict(domain='map', base_identity_sha256=digest(base), instance_id='fixture-instance',
            state_path=str(state), reset_validation=ref)
        pid, ticks = os.getpid(), _process(os.getpid())['start_ticks']
        return dict(schema=INSTANCE_SCHEMA, domain='map', scope='original', instance_id='fixture-instance',
            endpoint=ENDPOINTS['map'], state_path=str(state), state_provenance=provenance,
            state_provenance_sha256=digest(provenance), base_identity=base, base_identity_sha256=digest(base),
            job_id=os.environ['SLURM_JOB_ID'], hostname=socket.gethostname(),
            controller_pid=pid, controller_start_ticks=ticks, service_pid=pid, service_start_ticks=ticks,
            network_namespace=os.readlink('/proc/self/ns/net'), service_validation=ref, reset_validation=ref)

    def lease(self, instance):
        path = self.root / 'instance.json'; path.write_text(json.dumps(instance))
        lease = dict(task_id=task().execution_id, actor_model=self.config['actor_model'], selector='no_search',
            instance_receipt=str(path), instance_receipt_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        path = self.root / 'lease.json'; path.write_text(json.dumps(lease))
        self.config.update(lease_path=str(path), lease_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        return path

    def test_registry_only_advertises_proven_reddit_adapter(self):
        status = registry_status()
        self.assertEqual([key for key, value in status.items() if value['live_adapter_ready']], ['reddit', 'shopping', 'shopping_admin'])
        self.assertEqual(status['map']['authentication'], NO_LOGIN)
        self.assertTrue(site_adapter(task('reddit', storage=True), {}).legacy_reddit)
        with self.assertRaisesRegex(EvaluationNotReady, 'explicit original-site'):
            site_adapter(task(), {})
        for extra in ({}, {'ready': True}, {'validator': 'arbitrary_python_path'}):
            with self.subTest(extra=extra), self.assertRaisesRegex(EvaluationNotReady, 'unverified'):
                site_adapter(task(), dict(self.config, **extra))
        unsupported = SimpleNamespace(actor_view={'sites': ('new_domain',)}, requires_storage_state=False)
        with self.assertRaisesRegex(EvaluationNotReady, 'supported single'):
            site_adapter(unsupported, self.config)
        with self.assertRaisesRegex(EvaluationNotReady, 'supported single'):
            site_adapter(SimpleNamespace(actor_view={'sites': ('map', 'reddit')}), self.config)

    def test_no_claim_file_is_read_for_unverified_domain(self):
        with self.assertRaisesRegex(EvaluationNotReady, 'unverified'):
            claim_lease(task(), 'no_search', dict(self.config, lease_path='/does/not/exist'))

    def test_map_cannot_inherit_an_authenticated_task_or_unclaimed_receipt(self):
        with self.fixture_adapter():
            with self.assertRaisesRegex(EvaluationNotReady, 'unexpectedly requires storage'):
                site_adapter(task(storage=True), self.config)
            with self.assertRaisesRegex(EvaluationNotReady, 'verified fresh'):
                site_authentication(task(), self.config, self.receipt(), None, self.root,
                    reddit_authenticate=Mock(side_effect=AssertionError('Must not login')))

    def test_fresh_map_claim_produces_null_auth_and_no_browser_storage_keyword(self):
        receipt = self.receipt(); self.lease(receipt)
        with self.fixture_adapter():
            instance, reset = claim_lease(task(), 'no_search', self.config)
            self.config['reset_id'] = reset
            self.validator.assert_called_once()
            state, proof = site_authentication(task(), self.config, instance, None, self.root,
                reddit_authenticate=Mock(side_effect=AssertionError('Must not login')))
            self.assertIsNone(state); self.assertFalse(proof['authenticated'])
            seen = []
            def factory(**kwargs):
                seen.append(kwargs); return 'fixture-browser'
            result = open_authenticated_browser(task(), hooks=EpisodeHooks(dependency=lambda _: proof),
                browser_factory=factory, task_entrypoint=object, action_mapping=lambda x: x, storage_state=state)
            self.assertEqual(result, 'fixture-browser')
            self.assertNotIn('storage_state', seen[0].get('pw_context_kwargs', {}))
            self.assertEqual(seen[0]['task_kwargs']['start_url'], ENDPOINTS['map'])
            with self.assertRaises(FileExistsError):
                claim_lease(task(), 'no_search', self.config)

    def test_map_claim_binds_configured_task_lease_reset_and_cell(self):
        receipt = self.receipt(); self.lease(receipt)
        with self.fixture_adapter():
            instance, reset = claim_lease(task(), 'no_search', self.config)
            self.config['reset_id'] = reset
            for key in ('configured_task_sha256', 'lease_sha256', 'reset_id'):
                changed = deepcopy(instance); changed['episode_claim'][key] = 'changed'
                with self.subTest(key=key), self.assertRaisesRegex(EvaluationNotReady, 'verified fresh'):
                    site_authentication(task(), self.config, changed, None, self.root, reddit_authenticate=Mock())
            with self.assertRaisesRegex(EvaluationNotReady, 'different cell'):
                claim_lease(task(), 'random5', self.config)

    def test_common_receipt_rejects_source_reset_process_or_namespace_drift(self):
        receipt = self.receipt()
        validate_original_site_receipt(receipt, 'map')
        for field, value in [('base_identity_sha256', '0'*64), ('state_provenance_sha256', '0'*64),
                             ('service_start_ticks', 1), ('network_namespace', 'net:[invalid]'),
                             ('job_id', 'not-owned'), ('scope', 'synthetic'),
                             ('endpoint', 'http://example.com:80')]:
            with self.subTest(field=field), self.assertRaises(EvaluationNotReady):
                validate_original_site_receipt(dict(receipt, **{field: value}), 'map')
        Path(receipt['service_validation']['path']).write_text('{"changed":true}')
        with self.assertRaisesRegex(EvaluationNotReady, 'receipt changed'):
            validate_original_site_receipt(receipt, 'map')

    def test_cms_native_admin_prefix_is_required_only_for_shopping_admin(self):
        receipt=self.receipt()
        receipt['domain']='shopping_admin'
        receipt['state_provenance']['domain']='shopping_admin'
        receipt['state_provenance_sha256']=digest(receipt['state_provenance'])
        for suffix in ('/admin','/admin/'):
            receipt['endpoint']=ENDPOINTS['map']+suffix
            validate_original_site_receipt(receipt,'shopping_admin')
        for suffix in ('','/','/admin/dashboard','/other'):
            with self.subTest(suffix=suffix),self.assertRaises(EvaluationNotReady):
                validate_original_site_receipt(dict(receipt,endpoint=ENDPOINTS['map']+suffix),'shopping_admin')
        receipt['domain']='map';receipt['state_provenance']['domain']='map'
        receipt['state_provenance_sha256']=digest(receipt['state_provenance'])
        with self.assertRaises(EvaluationNotReady):validate_original_site_receipt(receipt,'map')

    def test_domain_validator_must_accept_before_state_claim(self):
        receipt = self.receipt(); self.lease(receipt)
        self.validator.side_effect = EvaluationNotReady('Native reset control failed')
        with self.fixture_adapter(), self.assertRaisesRegex(EvaluationNotReady, 'reset control failed'):
            claim_lease(task(), 'no_search', self.config)
        self.assertFalse((Path(receipt['state_path']) / '.webprm-episode-claim.json').exists())

    def test_runtime_protocol_is_explicitly_bound_without_changing_legacy_default(self):
        previous = protocol_definition(SOURCE)
        configured = protocol_definition(SOURCE, site_configuration=SITE_RUNTIME_CONFIGURATION)
        self.assertNotIn('site_runtime', previous)
        self.assertNotEqual(digest(previous), digest(configured))
        plan = dict(self.config, protocol=configured['identity'], protocol_contract=configured)
        self.assertEqual(validate_protocol_contract(plan), configured)
        with self.assertRaises(EvaluationNotReady):
            validate_protocol_contract(dict(plan, site_runtime=None))

    def test_generic_broker_cell_identity_needs_no_reddit_rewrite(self):
        client = Mock()
        with patch('openwebrl.webprm_broker.make_broker_client', return_value=client) as factory:
            broker_request(dict(self.config, broker_socket='/tmp/offline.sock', identity_sha256='a'*64,
                deadline=1), task(), 'no_search')
        self.assertEqual(factory.call_args.kwargs['cell_id'],
            task().execution_id + '/' + self.config['actor_model'] + '__no_search')


if __name__ == '__main__':
    unittest.main()
