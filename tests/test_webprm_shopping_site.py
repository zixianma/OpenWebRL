"""Offline Shopping site-adapter contracts; no service, browser or network use."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import socket
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock, patch

from openwebrl import webprm_shopping_site as shop
from openwebrl import webprm_cms_site as cms
from openwebrl.webprm_lifecycle import _process
from openwebrl.webprm_webarena import EvaluationNotReady, TASK_FILE, digest
from openwebrl.webprm_webarena_episode import TaskEnvelope
from openwebrl.webprm_webarena_sites import (INSTANCE_SCHEMA, SITE_RUNTIME_CONFIGURATION, registry_status,
    site_adapter, validate_site_instance)

SOURCE = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-baselines-20261009/sources/arbiter')


def ref(path):
    return dict(path=str(path), sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())


def task():
    return TaskEnvelope.from_release(dict(task_id=997, intent='Synthetic Shopping interface fixture',
        sites=['shopping'], start_url='__SHOPPING__', storage_state='./.auth/shopping_state.json',
        eval={'eval_types': ['string_match'], 'reference_answers': {'exact_match': '0'}}),
        {'shopping': shop.ENDPOINT})


class ShoppingSiteTests(unittest.TestCase):
    def setUp(self):
        block = patch('socket.socket.connect', side_effect=AssertionError('Network forbidden'))
        block.start(); self.addCleanup(block.stop)
        self.tmp = TemporaryDirectory(prefix='webprm-shopping-site-', dir='/tmp')
        self.addCleanup(self.tmp.cleanup); self.root = Path(self.tmp.name)
        env = patch.dict(os.environ, {'SLURM_JOB_ID': 'offline-fixture-not-an-allocation'})
        env.start(); self.addCleanup(env.stop)

    def receipt(self):
        base = self.root/'scratch/baseline'; state = self.root/'scratch/states/cell'
        base.mkdir(parents=True); state.mkdir(parents=True)
        build = self.root/'output/original-build/receipt.json'; build.parent.mkdir(parents=True)
        build.write_text(json.dumps(dict(kind=shop.BUILD_KIND, status='completed', baseline_path=str(base),
            materialized_content_sha256=shop.BASE_CONTENT_SHA256)))
        copy = self.root/'output/cells/cell/copy/receipt.json'; copy.parent.mkdir(parents=True)
        copy.write_text(json.dumps(dict(kind=shop.COPY_KIND, status='completed', state_path=str(state),
            source_path=str(base), build_receipt=ref(build))))
        identity = shop.base_identity()
        provenance = dict(domain='shopping', base_identity_sha256=digest(identity), instance_id='fixture',
            state_path=str(state), reset_validation=shop.HOSTING_VALIDATION, build_receipt=ref(build), copy_receipt=ref(copy))
        pid, ticks = os.getpid(), _process(os.getpid())['start_ticks']
        return dict(schema=INSTANCE_SCHEMA, domain='shopping', scope='original', instance_id='fixture',
            endpoint=shop.ENDPOINT, state_path=str(state), base_identity=identity, base_identity_sha256=digest(identity),
            state_provenance=provenance, state_provenance_sha256=digest(provenance), job_id=os.environ['SLURM_JOB_ID'],
            hostname=socket.gethostname(), controller_pid=pid, controller_start_ticks=ticks, service_pid=pid,
            service_start_ticks=ticks, network_namespace=os.readlink('/proc/self/ns/net'),
            service_validation=shop.HOSTING_VALIDATION, reset_validation=shop.HOSTING_VALIDATION,
            **shop.instance_extras())

    def test_pinned_hosting_validation_and_deterministic_base_identity(self):
        proposal, validation = shop.hosting()
        self.assertEqual(validation['status'], 'passed')
        self.assertEqual(validation['materialization']['materialized_content_sha256'], shop.BASE_CONTENT_SHA256)
        identity = shop.base_identity()
        self.assertEqual(identity['domain'], 'shopping'); self.assertFalse(identity['original_state_equivalence_verified'])
        self.assertEqual(identity['oci_manifest_sha256'], proposal['manifest']['sha256'])
        self.assertEqual(proposal['protocol'], shop.hosting_profile().protocol)

    def test_shared_stage_hash_api_matches_the_original_shopping_validator(self):
        from scripts.validate_webprm_shopping_hosting import content_sha256 as validated_hash
        manifest=dict(destination='/tmp/a',entries={'é':dict(bytes=1)},created_fifos=[],skipped_device_members=[])
        self.assertEqual(shop.content_sha256(manifest),validated_hash(manifest))
        self.assertNotEqual(shop.content_sha256(manifest),cms.content_sha256(manifest))

    def test_registry_marks_shopping_ready_and_full_site_validation_passes(self):
        self.assertTrue(registry_status()['shopping']['live_adapter_ready'])
        config = dict(site_runtime=deepcopy(SITE_RUNTIME_CONFIGURATION), endpoints={'shopping': shop.ENDPOINT})
        self.assertEqual(site_adapter(task(), config).domain, 'shopping')
        validate_site_instance(task(), config, self.receipt())

    def test_instance_drift_and_missing_helper_receipt_are_rejected(self):
        receipt = self.receipt()
        self.assertIn('shopping_base_content_sha256', shop.validate_instance(receipt))
        changes = dict(base=dict(receipt, base_identity=dict(receipt['base_identity'], materialized_content_sha256='0'*64)),
            endpoint=dict(receipt, endpoint='http://127.0.0.1:7771'),
            validation=dict(receipt, service_validation=dict(shop.HOSTING_VALIDATION, sha256='0'*64)),
            helper=dict(receipt, native_helper_accounts=dict(receipt['native_helper_accounts'], sha256='0'*64)),
            no_helper={k: v for k, v in receipt.items() if k != 'native_helper_accounts'})
        for name, value in changes.items():
            with self.subTest(name=name), self.assertRaises(EvaluationNotReady):shop.validate_instance(value)

    def test_released_order_helper_binds_to_the_owned_instance(self):
        from openwebrl.webprm_webarena_helper_runtime import HELPER_CONFIGURATION, owned_native_helpers
        released = next(t for t in json.loads((SOURCE/TASK_FILE).read_text())
                        if t['sites'] == ['shopping'] and 'shopping_get_latest_order_url' in json.dumps(t['eval']))
        envelope = TaskEnvelope.from_release(released, {'shopping': shop.ENDPOINT})
        receipt = self.receipt()
        config = dict(native_helpers=HELPER_CONFIGURATION, arbiter_source=str(SOURCE), reset_id=receipt['state_provenance_sha256'],
                      endpoints={'shopping': shop.ENDPOINT})
        helpers = owned_native_helpers(envelope, config, receipt)
        self.assertEqual(helpers.manifest['shopping_origin'], shop.ORIGIN)
        self.assertNotIn('native_helper_accounts', json.dumps(helpers.manifest))

    def test_login_uses_released_locators_saves_private_state_and_redacts(self):
        page = MagicMock(); page.url = shop.ORIGIN + '/wishlist/'
        page.goto.return_value = MagicMock(status=200)
        context = MagicMock(); context.new_page.return_value = page; context.storage_state.return_value = {'cookies': []}
        browser = MagicMock(); browser.new_context.return_value = context
        playwright = MagicMock(); playwright.__enter__.return_value.chromium.launch.return_value = browser
        accounts = (dict(username='fixture@example.com', password='fixture-secret-value'),
                    dict(shopping_site_admin=dict(username='admin', password='admin-secret-value')))
        out = self.root/'episode'; out.mkdir()
        with patch.object(shop, 'released_accounts', return_value=accounts), \
                patch('playwright.sync_api.sync_playwright', return_value=playwright), \
                patch.object(shop, 'hosting', return_value=({'account_binding': {'sha256': 'b'*64}}, {})):
            state = shop.authenticate(str(SOURCE), dict(endpoint=shop.ENDPOINT, state_provenance_sha256='c'*64), '/chrome', out)
        self.assertEqual(state, {'cookies': []})
        visited = [call.args[0] for call in page.goto.call_args_list]
        self.assertEqual(visited, [shop.ORIGIN + '/customer/account/login/', shop.ORIGIN + '/wishlist/'])
        page.get_by_label.assert_any_call('Email', exact=True); page.get_by_label.assert_any_call('Password', exact=True)
        page.get_by_role.assert_called_with('button', name='Sign In')
        self.assertEqual((out/'storage-state-private.json').stat().st_mode & 0o077, 0)
        self.assertNotIn('fixture-secret-value', (out/'authentication.json').read_text())
        page.goto.side_effect = RuntimeError('boom fixture-secret-value admin-secret-value')
        with patch.object(shop, 'released_accounts', return_value=accounts), \
                patch('playwright.sync_api.sync_playwright', return_value=playwright), \
                self.assertRaises(EvaluationNotReady) as caught:
            shop.authenticate(str(SOURCE), dict(endpoint=shop.ENDPOINT, state_provenance_sha256='c'*64), '/chrome', self.root/'x')
        self.assertNotIn('secret-value', str(caught.exception))
        self.assertIn('[REDACTED]', str(caught.exception))


if __name__ == '__main__':
    unittest.main()
