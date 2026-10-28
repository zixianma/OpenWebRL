import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from openwebrl import webprm_ephemeral_state as state


class EphemeralStateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='webprm-state-test-', dir='/tmp')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.scratch = self.root / 'scratch'
        self.campaign = self.root / 'campaign'
        self.base = self.scratch / 'base'
        for directory in (self.scratch, self.campaign, self.base, self.scratch / 'states'):
            directory.mkdir(mode=0o700)
        (self.base / 'original').write_text('original preserved')
        self.cell = 'fixed-cell'
        self.destination = self.scratch / 'states' / hashlib.sha256(self.cell.encode()).hexdigest()
        self.claim = self.campaign / 'claim.json'
        self.policy = '1' * 64

    def claim_args(self):
        return dict(scratch_root=self.scratch, campaign_root=self.campaign, cell_id=self.cell,
                    base_path=self.base, policy_sha256=self.policy)

    def prepare(self):
        state.claim_state(self.claim, self.destination, **self.claim_args())
        self.destination.mkdir(mode=0o700)
        (self.destination / 'database').write_text('fresh state')
        artifacts = {}
        for name in state.REQUIRED_ARTIFACTS:
            path = self.campaign / (name + '.json')
            path.write_text(json.dumps({'fixture': name}))
            artifacts[name] = path
        manifest = state.durable_artifacts(self.campaign, artifacts)
        return dict(approved_policy_sha256=self.policy, artifact_manifest=manifest,
            audit=dict(verified=True, cell_id=self.cell, artifact_manifest_sha256=state.digest(manifest),
                       audit_artifact_sha256=manifest['audit']['sha256']),
            closure=dict(worker_remaining_processes={}, service_remaining_processes={}, broker_cell_inflight=0),
            intent_path=self.campaign / 'disposal-intent.json', receipt_path=self.campaign / 'disposal.json')

    def test_claim_leaves_copy_destination_absent_and_is_exclusive(self):
        state.claim_state(self.claim, self.destination, **self.claim_args())
        self.assertFalse(self.destination.exists())
        with self.assertRaises(FileExistsError):
            state.claim_state(self.claim, self.destination, **self.claim_args())

    def test_existing_historical_state_is_ineligible(self):
        self.destination.mkdir()
        with self.assertRaisesRegex(ValueError, 'absent'):
            state.claim_state(self.claim, self.destination, **self.claim_args())
        self.assertFalse(self.claim.exists())

    def test_wrong_destination_and_public_owner_root_rejected(self):
        with self.assertRaises(ValueError):
            state.claim_state(self.claim, self.base, **self.claim_args())
        self.scratch.chmod(0o755)
        with self.assertRaisesRegex(ValueError, 'private'):
            state.claim_state(self.claim, self.destination, **self.claim_args())

    def test_durable_campaign_cannot_overlap_ephemeral_scratch(self):
        for campaign in (self.scratch, self.root):
            with self.subTest(campaign=campaign), self.assertRaises(ValueError):
                state.claim_state(campaign / 'claim.json', self.destination,
                                  **dict(self.claim_args(), campaign_root=campaign))

    def test_audited_state_disposal_preserves_base_evidence_and_external_symlink_target(self):
        kwargs = self.prepare()
        external = self.root / 'unrelated'
        external.write_text('preserve')
        (self.destination / 'symlink').symlink_to(external)
        receipt = state.dispose_state(self.claim, **kwargs)
        self.assertEqual(receipt['status'], 'fresh_state_disposed_after_durable_audit')
        self.assertFalse(self.destination.exists())
        self.assertEqual(external.read_text(), 'preserve')
        self.assertEqual((self.base / 'original').read_text(), 'original preserved')
        self.assertTrue(all(Path(v['path']).is_file() for v in kwargs['artifact_manifest'].values()))
        intent = json.loads(kwargs['intent_path'].read_text())
        self.assertEqual(intent['audited_artifacts'], kwargs['artifact_manifest'])
        self.assertEqual(state.dispose_state(self.claim, **kwargs), receipt)

    def test_policy_audit_and_actual_closure_are_required(self):
        kwargs = self.prepare()
        changes = [dict(approved_policy_sha256='2'*64),
                   dict(audit=dict(kwargs['audit'], verified=False)),
                   dict(audit=dict(kwargs['audit'], audit_artifact_sha256='2'*64)),
                   dict(closure=dict(kwargs['closure'], broker_cell_inflight=1)),
                   dict(closure=dict(kwargs['closure'], worker_remaining_processes={'123': {}})),
                   dict(closure=dict(kwargs['closure'], service_remaining_processes={'456': {}}))]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                state.dispose_state(self.claim, **dict(kwargs, **change))
            self.assertTrue(self.destination.exists())
            self.assertFalse(kwargs['intent_path'].exists())

    def test_changed_evidence_retains_state(self):
        kwargs = self.prepare()
        Path(kwargs['artifact_manifest']['verdict']['path']).write_text('changed')
        with self.assertRaisesRegex(ValueError, 'changed'):
            state.dispose_state(self.claim, **kwargs)
        self.assertTrue(self.destination.exists())
        self.assertFalse(kwargs['intent_path'].exists())

    def test_changed_disposal_receipt_is_not_accepted_as_completed(self):
        kwargs = self.prepare()
        result = state.dispose_state(self.claim, **kwargs)
        result['historical_states_untouched'] = False
        kwargs['receipt_path'].write_text(json.dumps(result))
        with self.assertRaisesRegex(ValueError, 'changed'):
            state.dispose_state(self.claim, **kwargs)

    def test_missing_evaluator_evidence_is_not_disposable(self):
        kwargs = self.prepare()
        kwargs['artifact_manifest'].pop('evaluator_inputs')
        kwargs['audit']['artifact_manifest_sha256'] = state.digest(kwargs['artifact_manifest'])
        with self.assertRaises(ValueError):
            state.dispose_state(self.claim, **kwargs)
        self.assertTrue(self.destination.exists())

    def test_symlink_evidence_and_descendant_mount_are_rejected(self):
        kwargs = self.prepare()
        path = self.campaign / 'linked-evidence.json'
        path.symlink_to(Path(kwargs['artifact_manifest']['api']['path']))
        with self.assertRaisesRegex(ValueError, 'nonsymlink'):
            state.durable_artifacts(self.campaign, dict(
                {k: v['path'] for k, v in kwargs['artifact_manifest'].items()}, extra=str(path)))
        with patch.object(state, '_has_mount', return_value=True), self.assertRaises(ValueError):
            state.dispose_state(self.claim, **kwargs)
        self.assertTrue(self.destination.exists())

    def test_interrupted_disposal_resumes_only_with_same_intent_and_inode(self):
        kwargs = self.prepare()
        def interrupted(path):
            (path / 'database').unlink()
            raise OSError('simulated interruption')
        interrupted.avoids_symlink_attacks = True
        with patch.object(shutil, 'rmtree', interrupted), self.assertRaises(OSError):
            state.dispose_state(self.claim, **kwargs)
        self.assertTrue(kwargs['intent_path'].exists())
        self.assertFalse(kwargs['receipt_path'].exists())
        receipt = state.dispose_state(self.claim, **kwargs)
        self.assertTrue(receipt['state_absent'])
        self.destination.mkdir()
        with self.assertRaisesRegex(ValueError, 'changed'):
            state.dispose_state(self.claim, **kwargs)
        self.assertTrue(self.destination.exists())


if __name__ == '__main__':
    unittest.main()
