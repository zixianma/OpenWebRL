"""Offline CMS site-adapter contracts; no service, browser, credential or network use."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import socket
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock, patch

from openwebrl import webprm_cms_site as cms
from openwebrl.webprm_lifecycle import _process
from openwebrl.webprm_webarena import EvaluationNotReady, digest
from openwebrl.webprm_webarena_episode import TaskEnvelope
from openwebrl.webprm_webarena_sites import (INSTANCE_SCHEMA, SITE_RUNTIME_CONFIGURATION, registry_status,
    site_adapter, validate_site_instance)

SOURCE = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-baselines-20261009/sources/arbiter')


def ref(path):
    return dict(path=str(path), sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())


def task():
    return TaskEnvelope.from_release(dict(task_id=998, intent='Synthetic CMS interface fixture',
        sites=['shopping_admin'], start_url='__SHOPPING_ADMIN__', storage_state='./.auth/shopping_admin_state.json',
        eval={'eval_types': ['string_match'], 'reference_answers': {'exact_match': '0'}}),
        {'shopping_admin': cms.ENDPOINT})


class CMSSiteTests(unittest.TestCase):
    def setUp(self):
        block = patch('socket.socket.connect', side_effect=AssertionError('Network forbidden'))
        block.start(); self.addCleanup(block.stop)
        self.tmp = TemporaryDirectory(prefix='webprm-cms-site-', dir='/tmp')
        self.addCleanup(self.tmp.cleanup); self.root = Path(self.tmp.name)
        env = patch.dict(os.environ, {'SLURM_JOB_ID': 'offline-fixture-not-an-allocation'})
        env.start(); self.addCleanup(env.stop)

    def receipt(self):
        base = self.root/'scratch/baseline'; state = self.root/'scratch/states/cell'
        base.mkdir(parents=True); state.mkdir(parents=True)
        build = self.root/'output/original-build/receipt.json'; build.parent.mkdir(parents=True)
        build.write_text(json.dumps(dict(kind=cms.BUILD_KIND, status='completed', baseline_path=str(base),
            materialized_content_sha256=cms.BASE_CONTENT_SHA256)))
        copy = self.root/'output/cells/cell/copy/receipt.json'; copy.parent.mkdir(parents=True)
        copy.write_text(json.dumps(dict(kind=cms.COPY_KIND, status='completed', state_path=str(state),
            source_path=str(base), build_receipt=ref(build))))
        identity = cms.base_identity()
        provenance = dict(domain='shopping_admin', base_identity_sha256=digest(identity), instance_id='fixture',
            state_path=str(state), reset_validation=cms.HOSTING_VALIDATION, build_receipt=ref(build), copy_receipt=ref(copy))
        pid, ticks = os.getpid(), _process(os.getpid())['start_ticks']
        return dict(schema=INSTANCE_SCHEMA, domain='shopping_admin', scope='original', instance_id='fixture',
            endpoint=cms.ENDPOINT, state_path=str(state), base_identity=identity, base_identity_sha256=digest(identity),
            state_provenance=provenance, state_provenance_sha256=digest(provenance), job_id=os.environ['SLURM_JOB_ID'],
            hostname=socket.gethostname(), controller_pid=pid, controller_start_ticks=ticks, service_pid=pid,
            service_start_ticks=ticks, network_namespace=os.readlink('/proc/self/ns/net'),
            service_validation=cms.HOSTING_VALIDATION, reset_validation=cms.HOSTING_VALIDATION)

    def test_pinned_hosting_validation_and_deterministic_base_identity(self):
        proposal, validation = cms.hosting()
        self.assertEqual(validation['status'], 'passed')
        identity = cms.base_identity()
        self.assertFalse(identity['original_state_equivalence_verified'])
        self.assertEqual(identity['source'], 'third_party_digest_pinned_oci_candidate')
        self.assertEqual(identity['oci_manifest_sha256'], proposal['manifest']['sha256'])
        manifest = dict(destination='/tmp/a', entries={'x': 1})
        self.assertEqual(cms.content_sha256(manifest), cms.content_sha256(dict(manifest, destination='/tmp/b')))
        self.assertNotEqual(cms.content_sha256(manifest), cms.content_sha256(dict(manifest, entries={})))

    def test_registry_marks_cms_ready_and_full_site_validation_passes(self):
        self.assertTrue(registry_status()['shopping_admin']['live_adapter_ready'])
        config = dict(site_runtime=deepcopy(SITE_RUNTIME_CONFIGURATION), endpoints={'shopping_admin': cms.ENDPOINT})
        self.assertEqual(site_adapter(task(), config).domain, 'shopping_admin')
        validate_site_instance(task(), config, self.receipt())

    def test_real_validated_and_new_manifest_have_identical_file_inventory(self):
        root=Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation')
        old=json.loads((root/'cms-hosting-validation-v3/campaign/attempt-357831/materialization/materialization.json').read_text())
        new=json.loads((root/'cms-table5-350-v3/campaign/attempt-364917/original-build/materialization/materialization.json').read_text())
        self.assertEqual(old['entries'],new['entries'])
        self.assertEqual(old['regular_file_sha256'],new['regular_file_sha256'])
        self.assertEqual(cms.content_sha256(old),cms.BASE_CONTENT_SHA256)
        self.assertEqual(cms.content_sha256(new),cms.BASE_CONTENT_SHA256)

    def test_identity_never_ignores_nonempty_device_fifo_or_file_changes(self):
        manifest=dict(destination='/tmp/a',entries={'é':dict(kind='file',bytes=1)},regular_file_sha256={'é':'a'*64})
        unchanged=cms.content_sha256(manifest)
        self.assertEqual(unchanged,cms.content_sha256(dict(manifest,created_fifos=[],skipped_device_members=[])))
        for key in ['created_fifos','skipped_device_members']:
            with self.assertRaises(EvaluationNotReady):cms.content_sha256(dict(manifest,**{key:['dev/unreviewed']}))
        changed=deepcopy(manifest);changed['regular_file_sha256']['é']='b'*64
        self.assertNotEqual(unchanged,cms.content_sha256(changed))

    def test_login_has_separate_bounded_navigation_and_never_reclicks(self):
        page=MagicMock()
        cms.submit_admin_login(page)
        page.expect_navigation.assert_called_once_with(wait_until='domcontentloaded',timeout=60000)
        page.get_by_role.assert_called_once_with('button',name='Sign in',exact=True)
        page.get_by_role.return_value.click.assert_called_once_with(no_wait_after=True)
        page=MagicMock();page.expect_navigation.return_value.__exit__.side_effect=TimeoutError('Cold dashboard')
        with self.assertRaises(TimeoutError):cms.submit_admin_login(page)
        self.assertEqual(page.get_by_role.return_value.click.call_count,1)

    def test_instance_drift_is_rejected(self):
        receipt = self.receipt()
        cms.validate_instance(receipt)
        changes = dict(base=dict(receipt, base_identity=dict(receipt['base_identity'], materialized_content_sha256='0'*64)),
            endpoint=dict(receipt, endpoint='http://127.0.0.1:7781/admin'),
            validation=dict(receipt, service_validation=dict(cms.HOSTING_VALIDATION, sha256='0'*64)))
        for name, value in changes.items():
            with self.subTest(name=name), self.assertRaises(EvaluationNotReady):cms.validate_instance(value)
        copy = Path(receipt['state_provenance']['copy_receipt']['path'])
        copy.write_text(copy.read_text().replace('"completed"', '"failed"'))
        with self.assertRaisesRegex(EvaluationNotReady, 'changed'):cms.validate_instance(receipt)

    def test_copy_must_target_this_instance_state_from_validated_build(self):
        receipt = self.receipt(); provenance = deepcopy(receipt['state_provenance'])
        copy = Path(provenance['copy_receipt']['path']); value = json.loads(copy.read_text())
        for field, wrong in (('state_path', str(self.root/'other')), ('source_path', str(self.root/'other'))):
            changed = self.root/(field+'.json'); changed.write_text(json.dumps(dict(value, **{field: wrong})))
            provenance['copy_receipt'] = ref(changed)
            with self.subTest(field=field), self.assertRaisesRegex(EvaluationNotReady, 'provenance'):
                cms.validate_instance(dict(receipt, state_provenance=provenance))

    def test_login_uses_released_locators_saves_private_state_and_redacts(self):
        page = MagicMock(); page.url = cms.ORIGIN + '/admin/admin/dashboard/'
        page.goto.return_value = MagicMock(status=200); page.content.return_value = '<title>Dashboard</title>'
        page.get_by_placeholder.return_value.count.return_value = 0
        context = MagicMock(); context.new_page.return_value = page; context.storage_state.return_value = {'cookies': []}
        browser = MagicMock(); browser.new_context.return_value = context
        playwright = MagicMock(); playwright.__enter__.return_value.chromium.launch.return_value = browser
        account = dict(username='fixture-user', password='fixture-secret-value')
        out = self.root/'episode'; out.mkdir()
        with patch.object(cms, 'released_account', return_value=account), \
                patch('playwright.sync_api.sync_playwright', return_value=playwright), \
                patch.object(cms, 'hosting', return_value=({'account_binding': {'sha256': 'b'*64}}, {})):
            state = cms.authenticate(str(SOURCE), dict(endpoint=cms.ENDPOINT, state_provenance_sha256='c'*64), '/chrome', out)
        self.assertEqual(state, {'cookies': []})
        visited = [call.args[0] for call in page.goto.call_args_list]
        self.assertEqual(visited, [cms.ORIGIN + '/admin', cms.ORIGIN + '/admin/admin/dashboard/'])
        page.get_by_role.assert_called_with('button', name='Sign in', exact=True)
        self.assertEqual((out/'storage-state-private.json').stat().st_mode & 0o077, 0)
        self.assertNotIn('fixture-secret-value', (out/'authentication.json').read_text())
        page.goto.side_effect = RuntimeError('boom fixture-secret-value')
        with patch.object(cms, 'released_account', return_value=account), \
                patch('playwright.sync_api.sync_playwright', return_value=playwright), \
                self.assertRaises(EvaluationNotReady) as caught:
            cms.authenticate(str(SOURCE), dict(endpoint=cms.ENDPOINT, state_provenance_sha256='c'*64), '/chrome', self.root/'x')
        self.assertNotIn('fixture-secret-value', str(caught.exception))
        self.assertIn('[REDACTED]', str(caught.exception))


if __name__ == '__main__':
    unittest.main()
