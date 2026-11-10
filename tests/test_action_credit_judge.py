"""Canonical judge with in-memory provider responses only; no network/API use."""
from dataclasses import replace
import hashlib
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from openwebrl.action_credit import digest
from openwebrl.action_credit_judge import (JudgeHTTPProtocol, MeteredJudgeClient,
    PaidDispatchBlocked, canonical_action_history_judge)
from openwebrl.action_credit_state import CampaignManifest, CampaignStore
from test_action_credit_state import manifest_data, sha, reserve


class Reply:
    def __init__(self, text="SUCCESS"):
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=text))]
        self.text = text
    def model_dump(self):
        return dict(choices=[dict(message=dict(content=self.text))],
                    usage=dict(prompt_tokens=4, completion_tokens=2))


class JudgeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="action-credit-judge-", dir="/tmp")
        self.addCleanup(temp.cleanup)
        data = manifest_data()
        self.approval = dict(status="approved", campaign_id=data["campaign_id"], model="fixture-judge",
            api_microdollars_cap=100, api_requests_cap=4,
            evidence="synthetic CPU fixture only; no external authorization")
        self.protocol = JudgeHTTPProtocol("fixture-judge", digest(self.approval), sha("CPU text/image bound"),
                                          32, 50, 1_000_000, 1_000_000)
        data["science"]["judge"]["http_protocol_sha256"] = self.protocol.sha256
        self.store = CampaignStore.create(Path(temp.name)/"ledger", CampaignManifest.from_dict(data))
        draw = reserve(self.store)["draw_id"]
        self.store.start_attempt(draw, "episode", reserved_cost=dict(gpu_seconds=1, browser_dispatches=1,
                                                                      api_requests=0, api_microdollars=0))
        self.events, self.calls = [], []

    def journal(self, name, record):
        self.events.append((name, record))

    async def provider(self, **payload):
        self.calls.append(payload)
        entries = self.store.snapshot()["subrequests"]
        self.assertEqual(sum(x["status"] == "reserved" for x in entries.values()), 1)
        return Reply()

    def client(self, **changes):
        args = dict(store=self.store, parent_attempt_id="episode", protocol=self.protocol,
            approval_receipt=self.approval, provider_create=self.provider, provider_max_retries=0, journal=self.journal)
        args.update(input_token_upper_bound=lambda payload: 6,
                    input_bound_protocol_sha256=self.protocol.input_bound_protocol_sha256)
        args.update(changes)
        return MeteredJudgeClient(**args)

    async def test_http_admission_persists_before_transport_and_retains_full_payload(self):
        client = self.client()
        response = await client.create(model="fixture-judge", messages=[{"role":"user","content":"fixture"}])
        self.assertEqual(response.text, "SUCCESS")
        self.assertEqual(self.calls[0]["max_completion_tokens"], 32)
        self.assertEqual(self.store.snapshot()["charged_resources"]["api_microdollars"], 6)
        self.assertEqual(self.store.snapshot()["charged_resources"]["api_requests"], 1)
        self.assertEqual([e[0] for e in self.events], ["judge_http_reserved", "judge_http_response"])

    async def test_timeout_then_retry_counts_two_http_attempts(self):
        async def failed(**kwargs): raise TimeoutError("CPU fixture")
        client = self.client(provider_create=failed)
        with self.assertRaises(TimeoutError):
            await client.create(model="fixture-judge", messages=[])
        client.provider_create = self.provider
        await client.create(model="fixture-judge", messages=[])
        state = self.store.snapshot()
        self.assertEqual(state["charged_resources"]["api_requests"], 2)
        self.assertEqual(state["charged_resources"]["api_microdollars"], 56)
        self.assertEqual(sorted(x["status"] for x in state["subrequests"].values()), ["succeeded", "unknown"])

    async def test_reconstructed_client_cannot_repeat_persisted_request(self):
        await self.client().create(model="fixture-judge", messages=[])
        with self.assertRaises(PaidDispatchBlocked):
            await self.client().create(model="fixture-judge", messages=[])
        self.assertEqual(len(self.calls), 1)

    def test_authority_and_hidden_sdk_retries_fail_closed(self):
        with self.assertRaises(PaidDispatchBlocked):
            self.client(approval_receipt=dict(self.approval, status="proposed"))
        with self.assertRaises(ValueError):
            self.client(provider_max_retries=2)

    async def test_model_and_output_limit_changes_do_not_dispatch(self):
        client = self.client()
        for kwargs in (dict(model="other"), dict(model="fixture-judge", max_completion_tokens=64)):
            with self.assertRaises(PaidDispatchBlocked):
                await client.create(messages=[], **kwargs)
        self.assertFalse(self.calls)

    async def test_cost_bound_that_exceeds_reservation_blocks_before_transport(self):
        with self.assertRaises(PaidDispatchBlocked):
            await self.client(input_token_upper_bound=lambda payload: 100).create(model="fixture-judge", messages=[])
        self.assertFalse(self.calls)
        self.assertFalse(self.store.snapshot().get("subrequests"))

    async def test_actual_canonical_actionhistory_function_with_fake_provider(self):
        from openwebrl import reward_browser
        from slime.utils.types import Sample
        sample = Sample(index=0, prompt="fixture", response='<tool_call>{"name":"done","arguments":{"response":"finished"}}</tool_call>',
            status=Sample.Status.COMPLETED, remove_sample=True,
            metadata=dict(intent="fixture", turn_index=0, is_last_turn=True, terminate_reason="task_completed",
                action_credit_training_admission="pending_frozen_scores_and_sealed_panel"))
        args = SimpleNamespace(judge_prompt_variant="action_history", judge_api_model="fixture-judge",
            judge_api_mode="served", judge_timeout_secs=1., judge_max_attached_imgs=0,
            context_num_screenshots=0, browser_response_format_mode="browser_env")
        original = reward_browser._get_openai_client
        result = await canonical_action_history_judge(args=args, samples=[sample], client=self.client(),
            journal=self.journal, source_sha256=hashlib.sha256(Path(reward_browser.__file__).read_bytes()).hexdigest())
        self.assertEqual((result["status"], result["reward"]), ("valid", 1.))
        self.assertTrue(sample.remove_sample)
        self.assertIs(reward_browser._get_openai_client, original)
        self.assertEqual(len(self.calls), 1)

    async def test_native_model_failure_dispositions_are_valid_zero_without_judge(self):
        from openwebrl import reward_browser
        from slime.utils.types import Sample
        args = SimpleNamespace(judge_prompt_variant="action_history", judge_api_model="fixture-judge",
            judge_api_mode="served", judge_timeout_secs=1., judge_max_attached_imgs=0,
            context_num_screenshots=0, browser_response_format_mode="browser_env")
        cases = ((Sample.Status.FAILED, "format_error_failed", "malformed response"),
                 (Sample.Status.TRUNCATED, "length", "<tool_call>{"),
                 (Sample.Status.COMPLETED, "task_completed", "no final answer"))
        for status, reason, response in cases:
            with self.subTest(status=status, reason=reason):
                sample = Sample(index=0, prompt="fixture", response=response, status=status, remove_sample=True,
                    metadata=dict(intent="fixture", turn_index=0, is_last_turn=True, terminate_reason=reason,
                        action_credit_training_admission="pending_frozen_scores_and_sealed_panel"))
                result = await canonical_action_history_judge(args=args, samples=[sample], client=self.client(),
                    journal=self.journal, source_sha256=hashlib.sha256(Path(reward_browser.__file__).read_bytes()).hexdigest())
                self.assertEqual((result["status"], result["reward"]), ("valid", 0.))
                self.assertTrue(result["native_reward"]["judge_text"])
        self.assertFalse(self.calls)
        self.assertEqual(self.store.snapshot()["charged_resources"]["api_requests"], 0)


if __name__ == "__main__":
    unittest.main()
