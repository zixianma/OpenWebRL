"""Offline CMS stage contracts; no materialization, namespace, service or allocation."""
from copy import deepcopy
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts import run_webprm_table5_cms as stage
from openwebrl import webprm_cms_site as cms
from openwebrl.webprm_ephemeral_state import PROTOCOL
from openwebrl.webprm_lifecycle import _process
from openwebrl.webprm_webarena import digest
from openwebrl.webprm_webarena_sites import validate_original_site_receipt


class CMSStageTests(unittest.TestCase):
    def setUp(self):
        block = patch('socket.socket.connect', side_effect=AssertionError('Network forbidden'))
        block.start(); self.addCleanup(block.stop)
        self.tmp = TemporaryDirectory(prefix='webprm-cms-stage-', dir='/tmp')
        self.addCleanup(self.tmp.cleanup); self.root = Path(self.tmp.name)
        env = patch.dict(os.environ, {'SLURM_JOB_ID': '424242'}); env.start(); self.addCleanup(env.stop)
        self.scratch = self.root/'scratch'; self.output = self.root/'campaign/attempt-424242'
        self.scratch.mkdir(); (self.scratch/'states').mkdir(); self.output.mkdir(parents=True)
        cells = [dict(cell_id='cms-cell', domain='shopping_admin'), dict(cell_id='reddit-cell', domain='reddit')]
        (self.root/'cohort.json').write_text(json.dumps({'cohort': {'cells': cells}}))
        policy = {'protocol': PROTOCOL}
        self.owned = dict(scratch=str(self.scratch), output=str(self.output), job_id='424242', deadline=4e9,
            owner_pid=os.getpid(), owner_start_ticks=_process(os.getpid())['start_ticks'],
            proposal=dict(cohort_path=str(self.root/'cohort.json'), ephemeral_policy=policy,
                campaign_root=str(self.root/'campaign'), domains={'shopping_admin': dict(endpoint=cms.ENDPOINT,
                    state_apparent_bytes=15*2**30, hosting_proposal=cms.HOSTING_PROPOSAL,
                    hosting_validation=cms.HOSTING_VALIDATION, base_content_sha256=cms.BASE_CONTENT_SHA256)}))

    def claim(self, cell_id):
        key = stage.cell_key(dict(cell_id=cell_id)); folder = self.output/'cells'/key; folder.mkdir(parents=True)
        state = self.scratch/'states'/key
        (folder/'state-claim.json').write_text(json.dumps(dict(cell_id=cell_id, protocol=PROTOCOL,
            policy_sha256=digest({'protocol': PROTOCOL}), campaign_root=str(self.root/'campaign'),
            scratch_root=str(self.scratch), base_path=str(self.scratch/'baseline'), state_path=str(state),
            destination_absent_at_claim=True, preserve_historical_states=True)))
        return state, folder

    def build_receipt(self):
        path = self.output/'original-build/receipt.json'; path.parent.mkdir()
        path.write_text(json.dumps(dict(kind=cms.BUILD_KIND, status='completed', baseline_path=str(self.scratch/'baseline'),
            materialized_content_sha256=cms.BASE_CONTENT_SHA256)))
        (self.scratch/'baseline').mkdir()
        return path

    def test_stage_paths_bind_cms_cells_and_fixed_build(self):
        stage.stage_paths(self.owned, 'build', self.scratch/'baseline', self.output/'original-build')
        with self.assertRaisesRegex(ValueError, 'fixed private immutable'):
            stage.stage_paths(self.owned, 'build', self.scratch/'other', self.output/'original-build')
        state, folder = self.claim('cms-cell')
        self.assertEqual(stage.stage_paths(self.owned, 'copy', state, folder/'copy'), (state, folder/'copy'))
        reddit, other = self.claim('reddit-cell')
        with self.assertRaisesRegex(ValueError, 'fresh private state claim'):
            stage.stage_paths(self.owned, 'copy', reddit, other/'copy')
        with self.assertRaisesRegex(ValueError, 'fresh private state claim'):
            stage.stage_paths(self.owned, 'serve', state, folder/'site')  # state not copied yet

    def test_domain_entry_must_match_validated_recipe(self):
        stage.hosting_plan(self.owned)
        changed = deepcopy(self.owned); changed['proposal']['domains']['shopping_admin']['endpoint'] = 'http://127.0.0.1'
        with self.assertRaisesRegex(ValueError, 'validated hosting recipe'):stage.hosting_plan(changed)

    def test_build_rejects_a_base_that_differs_from_the_validated_content(self):
        def fake(layers, destination, evidence, deadline, limits=None):
            self.assertEqual(limits, cms.materializer_limits())  # CMS keeps the materializer defaults
            destination.mkdir(); evidence.mkdir()
            path = evidence/'materialization.json'; path.write_text(json.dumps(dict(destination=str(destination), entries={})))
            return dict(manifest=dict(path=str(path), sha256=stage.sha(path)), final_regular_bytes=0)
        with patch.object(stage.materializer, 'materialize', side_effect=fake):
            with self.assertRaisesRegex(ValueError, 'differs from the validated base'):
                stage.build(self.owned, self.scratch/'baseline', self.output/'original-build')
        self.assertFalse((self.output/'original-build/receipt.json').exists())
        expected = cms.content_sha256(dict(entries={}))
        owned = deepcopy(self.owned); owned['proposal']['domains']['shopping_admin']['base_content_sha256'] = expected
        with patch.object(stage.materializer, 'materialize', side_effect=fake), \
                patch.object(stage.site, 'BASE_CONTENT_SHA256', expected):
            stage.build(owned, self.scratch/'baseline2', self.output/'original-build2')
        receipt = json.loads((self.output/'original-build2/receipt.json').read_text())
        self.assertEqual(receipt['kind'], cms.BUILD_KIND); self.assertEqual(receipt['materialized_content_sha256'], expected)

    def test_copy_runs_as_mapped_root_and_keeps_db_controls_off_durable_output(self):
        self.build_receipt(); state, folder = self.claim('cms-cell'); seen = []
        def command(argv, output, deadline, **kw):
            seen.append(argv); Path(argv[-1]).mkdir()
        def controls(state_path, folder_path, plan):
            folder_path.mkdir(); (folder_path/'runtime-adaptations.json').write_text('{}')
        with patch.object(stage.owner, 'command', side_effect=command), \
                patch.object(stage.owner, 'prepare_controls', side_effect=controls):
            stage.copy(self.owned, state, folder/'copy')
        self.assertEqual(seen, [['unshare', '--user', '--map-root-user', 'cp', '-a', '--reflink=auto', '--',
                                 str(self.scratch/'baseline'), str(state)]])
        receipt = json.loads((folder/'copy/receipt.json').read_text())
        self.assertEqual(receipt['kind'], cms.COPY_KIND)
        self.assertTrue(Path(receipt['controls_path']).is_relative_to(self.scratch))
        self.assertFalse(Path(receipt['controls_path']).is_relative_to(self.output))

    def test_instance_receipt_passes_generic_and_cms_validation(self):
        self.build_receipt(); state, folder = self.claim('cms-cell'); state.mkdir()
        (folder/'copy').mkdir()
        (folder/'copy/receipt.json').write_text(json.dumps(dict(kind=cms.COPY_KIND, status='completed', state_path=str(state),
            source_path=str(self.scratch/'baseline'), build_receipt=stage.ref(self.output/'original-build/receipt.json'))))
        (folder/'site').mkdir()
        receipt = stage.instance_receipt(self.owned, state, folder/'site')
        validate_original_site_receipt(receipt, 'shopping_admin')
        cms.validate_instance(receipt)
        self.assertFalse(receipt['original_state_equivalence_verified'])
        self.assertEqual(receipt['service_pid'], os.getpid())

    def test_inside_rejects_a_foreign_launch_before_any_command(self):
        state, folder = self.claim('cms-cell'); out = folder/'site'; out.mkdir()
        (out/'launch.json').write_text(json.dumps(dict(launch_token='a', state=str(state), output=str(out),
            controls=str(stage.controls_path(self.owned, state)))))
        with patch.object(stage.owner, 'command', side_effect=AssertionError('No command')):
            with self.assertRaisesRegex(ValueError, 'launch identity'):
                stage.inside(self.owned, state, out, 'b')


if __name__ == '__main__':
    unittest.main()
