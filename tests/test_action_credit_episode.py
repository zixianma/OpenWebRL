"""Trusted episode save/reload crash boundaries; CPU transports only."""
from dataclasses import replace
from pathlib import Path
import unittest

from openwebrl.action_credit_collect import physical_verifier_sha256, ReconcileRequired
from openwebrl.action_credit_episode import load_pending_episode, save_pending_episode
from openwebrl.action_credit_state import CampaignStore
import test_action_credit_collect as fixtures


class EpisodePersistenceTests(unittest.IsolatedAsyncioTestCase):
    setUp = fixtures.CollectionTests.setUp
    request = fixtures.CollectionTests.request
    actor_call = fixtures.CollectionTests.actor_call
    version = fixtures.CollectionTests.version
    run_episode = fixtures.CollectionTests.run_episode

    def load(self, store=None, actor=None, contract=None):
        return load_pending_episode(store or self.store, contract or self.contract, actor or self.actor,
                                    "attempt1", verifier_sha256=physical_verifier_sha256())

    async def test_restart_loads_exact_episode_without_dispatch_or_extra_charge(self):
        fake = fixtures.NativeFake()
        original = await self.run_episode(fake)
        before = self.store.snapshot()
        reopened = CampaignStore(self.store.root, self.store.manifest)
        restored = self.load(reopened)
        self.assertEqual(restored.request, original.request)
        self.assertEqual(restored.turns[0].row, original.turns[0].row)
        self.assertEqual(restored.native_samples[0].tokens, original.native_samples[0].tokens)
        self.assertEqual(reopened.snapshot()["charged_resources"], before["charged_resources"])
        self.assertEqual((self.calls, fake.created, fake.exited), (1, 1, 1))
        self.assertEqual(before["attempts"]["attempt1"]["status"], "active")
        self.assertEqual(save_pending_episode(self.store, self.contract, self.actor, restored,
            verifier_sha256=physical_verifier_sha256()), before["pending_episodes"]["attempt1"])

    async def test_corrupted_payload_is_rejected_before_deserialization(self):
        await self.run_episode(fixtures.NativeFake())
        pointer = self.store.snapshot()["pending_episodes"]["attempt1"]
        Path(pointer["path"]).write_bytes(b"partial pickle after a crash")
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            self.load()

    async def test_changed_retained_tensor_or_rollout_evidence_is_rejected(self):
        original = await self.run_episode(fixtures.NativeFake())
        evidence = Path(original.artifact_directory)/"0000-processor.pt"
        evidence.write_bytes(b"wrong processor evidence")
        with self.assertRaisesRegex(ValueError, "artifact missing or changed"):
            self.load()

    async def test_uncommitted_file_never_authorizes_redispatch_or_automatic_adoption(self):
        fake = fixtures.NativeFake()
        episode = await self.run_episode(fake)
        with self.store._transaction() as state:
            state["pending_episodes"].pop("attempt1")
        with self.assertRaisesRegex(ValueError, "no committed complete episode"):
            self.load()
        with self.assertRaisesRegex(ValueError, "crash reconciliation"):
            save_pending_episode(self.store, self.contract, self.actor, episode,
                                 verifier_sha256=physical_verifier_sha256())
        with self.assertRaises(ReconcileRequired):
            await self.run_episode(fake, plan=False)
        self.assertEqual((self.calls, fake.created), (1, 1))

    async def test_changed_policy_context_and_reconstructed_tensor_are_rejected(self):
        await self.run_episode(fixtures.NativeFake())
        with self.assertRaises(ValueError):
            self.load(actor=replace(self.actor, weight_version="8"))
        with self.assertRaises(ValueError):
            self.load(contract=replace(self.contract, update_id=1))
        restored = self.load()
        restored.turns[0].tensors["pixel_values"].add_(1)
        with self.assertRaises(ValueError):
            save_pending_episode(self.store, self.contract, self.actor, restored,
                                 verifier_sha256=physical_verifier_sha256())


if __name__ == "__main__":
    unittest.main()
