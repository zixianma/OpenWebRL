"""Offline Shopping stage contracts through the shared Magento stage script."""
from copy import deepcopy
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts import run_webprm_table5_cms as stage
from openwebrl import webprm_shopping_site as shop
from openwebrl.webprm_ephemeral_state import PROTOCOL
from openwebrl.webprm_lifecycle import _process
from openwebrl.webprm_webarena import digest
from openwebrl.webprm_webarena_sites import validate_original_site_receipt


class ShoppingStageTests(unittest.TestCase):
    def setUp(self):
        block = patch('socket.socket.connect', side_effect=AssertionError('Network forbidden'))
        block.start(); self.addCleanup(block.stop)
        self.tmp = TemporaryDirectory(prefix='webprm-shopping-stage-', dir='/tmp')
        self.addCleanup(self.tmp.cleanup); self.root = Path(self.tmp.name)
        env = patch.dict(os.environ, {'SLURM_JOB_ID': '424242'}); env.start(); self.addCleanup(env.stop)
        self.scratch = self.root/'scratch'; self.output = self.root/'campaign/attempt-424242'
        self.scratch.mkdir(); (self.scratch/'states').mkdir(); self.output.mkdir(parents=True)
        cells = [dict(cell_id='shop-cell', domain='shopping'), dict(cell_id='cms-cell', domain='shopping_admin')]
        (self.root/'cohort.json').write_text(json.dumps({'cohort': {'cells': cells}}))
        self.owned = dict(scratch=str(self.scratch), output=str(self.output), job_id='424242', deadline=4e9,
            owner_pid=os.getpid(), owner_start_ticks=_process(os.getpid())['start_ticks'],
            proposal=dict(cohort_path=str(self.root/'cohort.json'), ephemeral_policy={'protocol': PROTOCOL},
                campaign_root=str(self.root/'campaign'), domains={'shopping': dict(endpoint=shop.ENDPOINT,
                    state_apparent_bytes=65*2**30, hosting_proposal=shop.HOSTING_PROPOSAL,
                    hosting_validation=shop.HOSTING_VALIDATION, base_content_sha256=shop.BASE_CONTENT_SHA256)}))

    def claim(self, cell_id):
        key = stage.cell_key(dict(cell_id=cell_id)); folder = self.output/'cells'/key; folder.mkdir(parents=True)
        state = self.scratch/'states'/key
        (folder/'state-claim.json').write_text(json.dumps(dict(cell_id=cell_id, protocol=PROTOCOL,
            policy_sha256=digest({'protocol': PROTOCOL}), campaign_root=str(self.root/'campaign'),
            scratch_root=str(self.scratch), base_path=str(self.scratch/'baseline'), state_path=str(state),
            destination_absent_at_claim=True, preserve_historical_states=True)))
        return state, folder

    def test_site_selection_plan_and_origin(self):
        self.assertIs(stage.site_of(self.owned), shop)
        proposal, plan = stage.hosting_plan(self.owned)
        self.assertEqual(proposal['protocol'], shop.hosting_profile().protocol)
        self.assertIn('/etc/nginx/conf.d/default.conf', plan['container_file_install_map'])
        self.assertEqual(stage.origin_commands(shop)[-1], ['setup:store-config:set', '--base-url=http://127.0.0.1:7770'])
        self.assertIn(b"'http://127.0.0.1:7770/'", stage.secure_sql(shop))
        mixed = deepcopy(self.owned); mixed['proposal']['domains']['reddit'] = {}
        with self.assertRaisesRegex(ValueError, 'exactly one'):stage.site_of(mixed)
        state, folder = self.claim('shop-cell')
        self.assertEqual(stage.stage_paths(self.owned, 'copy', state, folder/'copy'), (state, folder/'copy'))
        other, other_folder = self.claim('cms-cell')
        with self.assertRaisesRegex(ValueError, 'fresh private state claim'):
            stage.stage_paths(self.owned, 'copy', other, other_folder/'copy')

    def test_build_uses_shopping_materializer_limits_and_clock(self):
        seen = {}
        def fake(layers, destination, evidence, deadline, limits=None):
            seen.update(limits=limits, seconds=deadline)
            destination.mkdir(); evidence.mkdir()
            path = evidence/'materialization.json'; path.write_text(json.dumps(dict(destination=str(destination), entries={})))
            return dict(manifest=dict(path=str(path), sha256=stage.sha(path)), final_regular_bytes=0)
        with patch.object(stage.materializer, 'materialize', side_effect=fake):
            with self.assertRaisesRegex(ValueError, 'Shopping base differs'):
                stage.build(self.owned, self.scratch/'baseline', self.output/'original-build')
        self.assertEqual(seen['limits'].members, 30000000); self.assertEqual(seen['limits'].raw_bytes, 80*2**30)

    def test_instance_receipt_carries_helper_accounts_and_passes_validation(self):
        build = self.output/'original-build/receipt.json'; build.parent.mkdir()
        build.write_text(json.dumps(dict(kind=shop.BUILD_KIND, status='completed', baseline_path=str(self.scratch/'baseline'),
            materialized_content_sha256=shop.BASE_CONTENT_SHA256)))
        (self.scratch/'baseline').mkdir()
        state, folder = self.claim('shop-cell'); state.mkdir(); (folder/'copy').mkdir()
        (folder/'copy/receipt.json').write_text(json.dumps(dict(kind=shop.COPY_KIND, status='completed', state_path=str(state),
            source_path=str(self.scratch/'baseline'), build_receipt=stage.ref(build))))
        (folder/'site').mkdir()
        receipt = stage.instance_receipt(self.owned, state, folder/'site')
        self.assertEqual(receipt['native_helper_accounts'], shop.helper_accounts_receipt())
        self.assertEqual(receipt['endpoint'], 'http://127.0.0.1:7770')
        validate_original_site_receipt(receipt, 'shopping')
        shop.validate_instance(receipt)


if __name__ == '__main__':
    unittest.main()
