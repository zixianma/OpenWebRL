"""Crash commit/admission checks only; no real browser, GPU or paid request."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

from openwebrl.action_credit import digest
from openwebrl.action_credit_judge import JudgeHTTPProtocol, PaidDispatchBlocked
from openwebrl.action_credit_state import CampaignManifest, CampaignStore
from openwebrl.action_credit_worker import (
    _approval, _settle_terminal, reconcile_saved_verdict, verify_worker_environment,
    record_clean_pre_actor_failure, settle_clean_pre_actor_failure,
    judge_existing_pending_episode,
)
from openwebrl.controlled_sft_worker import AttemptJournal
import test_action_credit_collect as collection_fixture


class WorkerTests(IsolatedAsyncioTestCase):
    def setUp(self):
        self.fixture = collection_fixture.CollectionTests("test_native_episode_exact_context_masks_and_pending_scores")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = Path(self.fixture.temp.name)
        data = self.fixture.store.manifest.to_dict()
        self.source_sha = digest("CPU fake canonical judge, not physical acceptance")
        data["science"]["judge"]["canonical_reward_source_sha256"] = self.source_sha
        self.fixture.store = self.store = CampaignStore.create(self.root/"worker-ledger", CampaignManifest.from_dict(data))
        self.result = dict(status="valid", reward=1., native_reward={"judge":1.},
                           native_values=[1.], source_sha256=self.source_sha)

    async def episode(self):
        return await self.fixture.run_episode(collection_fixture.NativeFake())

    def journal(self, episode):
        return AttemptJournal(Path(episode.artifact_directory)/"judge")

    async def test_committed_verdict_can_reconcile_repeatedly_without_charging_again(self):
        ep = await self.episode()
        _settle_terminal(self.store, ep, self.result, journal=self.journal(ep), tree_terminal=False)
        before = self.store.snapshot()
        for _ in range(2):
            self.assertIs(reconcile_saved_verdict(self.store, ep), ep)
        after = self.store.snapshot()
        self.assertEqual(before, after)
        self.assertEqual(after["draws"][ep.draw_id]["status"], "valid")
        self.assertEqual(after["attempts"][ep.request.attempt_id]["completion"]["outcome"]["reward"], 1.)

    async def test_crash_after_verdict_pointer_before_attempt_settlement_recovers(self):
        ep = await self.episode()
        with patch.object(self.store, "finish_attempt", side_effect=RuntimeError("simulated process death")):
            with self.assertRaisesRegex(RuntimeError, "simulated process death"):
                _settle_terminal(self.store, ep, self.result, journal=self.journal(ep), tree_terminal=False)
        state = self.store.snapshot()
        self.assertEqual(state["attempts"][ep.request.attempt_id]["status"], "active")
        self.assertIn(ep.request.attempt_id, state["terminal_verdicts"])
        charges = state["charged_resources"]
        reconcile_saved_verdict(self.store, ep)
        self.assertEqual(self.store.snapshot()["charged_resources"], charges)
        self.assertEqual(self.store.snapshot()["draws"][ep.draw_id]["status"], "valid")

    async def test_invalid_verdict_has_no_reward_and_cannot_change_after_commit(self):
        ep = await self.episode()
        invalid = dict(self.result, status="invalid", reward=None)
        _settle_terminal(self.store, ep, invalid, journal=self.journal(ep), tree_terminal=False)
        state = self.store.snapshot()
        self.assertNotIn("reward", state["attempts"][ep.request.attempt_id]["completion"]["outcome"])
        path = Path(ep.artifact_directory)/"judge/terminal-verdict.json"
        value = json.loads(path.read_text())
        value["verdict"] = self.result
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "cannot be replaced"):
            reconcile_saved_verdict(self.store, ep)

    async def test_source_and_episode_identity_are_required_for_crash_reconciliation(self):
        ep = await self.episode()
        _settle_terminal(self.store, ep, self.result, journal=self.journal(ep), tree_terminal=False)
        with self.assertRaisesRegex(ValueError, "exact episode"):
            reconcile_saved_verdict(self.store, replace(ep, request_sha256=digest("wrong request")))
        with self.assertRaisesRegex(ValueError, "tree identity"):
            reconcile_saved_verdict(self.store, ep, tree_terminal=True)

    def test_api_approval_is_campaign_and_cap_specific(self):
        m = self.store.manifest.to_dict()
        approved = dict(status="approved", campaign_id=m["campaign_id"], model="gpt-4.1",
            api_microdollars_cap=m["budget_caps"]["api_microdollars"],
            api_requests_cap=m["budget_caps"]["api_requests"],
            user_authorization_text="CPU fixture, no actual paid authority",
            user_authorization_reference="CPU fixture, no actual paid authority")
        path = self.root/"approval.json"
        path.write_text(json.dumps(approved))
        reference = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        protocol = JudgeHTTPProtocol("gpt-4.1",digest(approved),digest("input bound"),1024,100,1,1)
        self.assertEqual(_approval(self.store, reference, protocol), approved)
        for key, value in (("status", "proposed"), ("campaign_id", "another-arm"),
                           ("api_requests_cap", 99), ("user_authorization_reference", "")):
            changed = dict(approved, **{key:value})
            path.write_text(json.dumps(changed))
            ref = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            with self.assertRaises(PaidDispatchBlocked):
                _approval(self.store, ref, replace(protocol, approval_receipt_sha256=digest(changed)))

    def test_no_implicit_home_cache_allowed(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(ValueError, "worker environment"):
                verify_worker_environment()

    async def test_clean_pre_actor_failure_is_missing_only_after_parent_cleanup(self):
        class StartupFailure(collection_fixture.NativeFake):
            async def generate_turn_sample(self, args, sample, sampling):
                client, _ = await self._create_env('task-a', self._apply_local_process_env_overrides({}))
                await client.exit()
                return [SimpleNamespace(metadata={'terminate_reason':'generation_error: fixture navigation timeout'})]
        fake=StartupFailure(); request=self.fixture.request(fake)
        reason='native generator returned infrastructure failure: generation_error: fixture navigation timeout'
        with self.assertRaisesRegex(ValueError, 'native generator returned infrastructure failure'):
            await self.fixture.run_episode(fake, request)
        record_clean_pre_actor_failure(self.store, request, reason)
        self.assertEqual(self.store.snapshot()['attempts'][request.attempt_id]['status'],'active')
        path=self.root/'cleanup.json'
        path.write_text(json.dumps(dict(attempt_id=request.attempt_id,owned_processes_closed=True)))
        ref=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        settle_clean_pre_actor_failure(self.store,request,process_cleanup_reference=ref)
        state=self.store.snapshot()
        self.assertEqual(state['attempts'][request.attempt_id]['status'],'missing')
        self.assertEqual(state['attempts'][request.attempt_id]['actual']['browser_dispatches'],1)
        self.assertIsNone(state['attempts'][request.attempt_id]['completion']['outcome'])

    async def test_received_actor_response_never_abstains_as_pre_actor_failure(self):
        ep=await self.episode()
        with self.assertRaisesRegex(ValueError,'uncompleted collection attempt'):
            record_clean_pre_actor_failure(self.store,ep.request,
                'native generator returned infrastructure failure: synthetic later failure')
        self.assertEqual(self.store.snapshot()['attempts'][ep.request.attempt_id]['status'],'active')

    async def test_judge_recovery_rejects_unpinned_transport_before_any_dispatch(self):
        protocol=JudgeHTTPProtocol('gpt-4.1',digest('authority'),digest('input'),1024,100,1,1)
        with patch('openwebrl.action_credit_worker.verify_worker_environment'), \
             patch('openwebrl.action_credit_worker._approval') as approval, \
             patch('openwebrl.action_credit_worker.load_pending_episode') as load:
            with self.assertRaisesRegex(ValueError,'Recovered judge transport'):
                await judge_existing_pending_episode(store=self.store,contract=self.fixture.contract,
                    actor=self.fixture.actor,attempt_id='not-dispatched',args=self.fixture.args,
                    approval_reference={},judge_protocol=protocol,canonical_reward_source_sha256=self.source_sha)
            approval.assert_not_called()
            load.assert_not_called()
