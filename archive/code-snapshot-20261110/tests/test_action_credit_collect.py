"""Native-API CPU fakes only: no browser, actor, judge, network or GPU calls."""
import asyncio
import base64
from copy import deepcopy
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from PIL import Image

from openwebrl.action_credit import CollectionContract, digest
from openwebrl.action_credit_binding import token_mask_contract
from openwebrl.action_credit_collect import (ActorOutput, CollectionRejected, EpisodeRequest,
    FrozenActor, ReconcileRequired, SGLangActorTransport, prefix_sha256, generator_protocol, physical_verifier_sha256, run_native_episode)
from openwebrl.action_credit_state import CampaignManifest, CampaignStore
from openwebrl.arm_branch_browser import digest as browser_digest
from test_action_credit_state import manifest_data, sha, cost


def observation(*, changed=False):
    stream = io.BytesIO()
    Image.new("RGB", (4, 4), "black").save(stream, format="PNG")
    image = stream.getvalue()
    page = {"url": "https://fixture.invalid", "text": "changed" if changed else "same"}
    snapshot = dict(pages=[dict(page)], active_tab=0, storage={})
    return dict(screenshot=image, selection_page=page, active_tab_url=page["url"],
        screen_size=[1280, 720], branch_snapshot=snapshot, branch_snapshot_sha256=browser_digest(snapshot),
        branch_replay_protocol="observable-replay-v1", controlled=dict(
            screenshot_sha256=hashlib.sha256(image).hexdigest(), page_sha256=browser_digest(page),
            action_attempts=0, executed_actions=[], sequence=1,
            capture_started_monotonic=1., capture_finished_monotonic=2.))


class NativeFake:
    """Exercises the real admission/selector/client wrappers via native API names."""
    def __init__(self, *, changed=False, swallow=False, dispatch=True):
        self.changed, self.swallow, self.dispatch = changed, swallow, dispatch
        self.dispatches, self.created, self.exited = 0, 0, 0
        self.task_data, self.tools, self.env_policy = {"intent": "fixture"}, [], "fixture-policy"
        self._apply_local_process_env_overrides = lambda x: x
        self._load_local_resources = lambda *a, **k: (self.task_data, self.tools, self.env_policy)
        self._create_env = self.create

    async def create(self, task, config):
        self.created += 1
        data = self._load_local_resources(task, config)
        outer = self
        class Client:
            async def step(self, actions):
                outer.dispatches += 1
                return observation(), 0., False, False, {}
            async def exit(self):
                outer.exited += 1
        return Client(), data[0]

    async def capture(self, client):
        return observation(changed=self.changed)

    async def generate_turn_sample(self, args, sample, sampling):
        client, data = await self._create_env("task-a", self._apply_local_process_env_overrides({}))
        try:
            obs = observation()
            ids, pixels = [10, 11], torch.tensor([[1., 2.]])
            args.browser_action_selector.set_training_context(dict(prompt_tokens=ids,
                multimodal_train_inputs={"pixel_values": pixels}))
            output, metadata = await args.browser_action_selector(infer=None, url="fixture://actor", input_text="fixture",
                sampling_params=sampling, images=[base64.b64encode(obs["screenshot"]).decode()], observation=obs,
                history=[], task=data["intent"], task_id="task-a", turn=0, timeout=1.)
            # Upstream mutates input IDs when it appends the response. The saved
            # processor row must remain unchanged.
            native_pixels = pixels.clone()
            ids.extend(output[1])
            pixels.add_(100)
            if self.dispatch:
                await client.step([dict(name="click")])
            return [SimpleNamespace(metadata={"arm": metadata, "terminate_reason":"task_completed" if self.dispatch else "format_error_failed",
                "turn_index":0, "is_last_turn":True}, loss_mask=[1]*len(output[1]), tokens=ids,
                response_length=len(output[1]), response=output[0],
                status="COMPLETED" if self.dispatch else "FAILED", rollout_log_probs=output[2], remove_sample=False,
                multimodal_train_inputs={"pixel_values": native_pixels})]
        except Exception:
            if self.swallow:
                return [SimpleNamespace(metadata={"terminate_reason": "aborted"})]
            raise
        finally:
            await client.exit()


class CollectionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="action-credit-collect-", dir="/tmp")
        self.addCleanup(self.temp.cleanup)
        self.sampling = dict(temperature=.8, top_p=1., top_k=-1, max_new_tokens=1024)
        self.contract = CollectionContract(0, sha("actor"), sha("tokenizer"), sha("processor"),
            sha("tasks"), sha("judge"), digest(self.sampling), .8, ("task-a",), ("task-b",))
        data = manifest_data()
        self.args = SimpleNamespace(max_steps=2, context_num_screenshots=1, turn_history_reasoning_mode="full",
            browser_history_reasoning_max_turns=0, browser_include_tool_response=1,
            browser_history_env_feedback_max_chars=4000, rollout_max_context_len=32768,
            max_consecutive_parse_failures=3, serial_action_variant=None, browser_response_format_mode="browser_env")
        data["science"]["generation"]["max_turns"] = 2
        data["science"]["generation"]["native_protocol_sha256"] = digest(generator_protocol(self.args))
        data["science"]["algorithm"]["local_actor_requests_cap"] = 20
        data["science"]["algorithm"].update({k: getattr(self.contract, k) for k in
            ("tokenizer_sha256", "processor_sha256", "task_manifest_sha256", "judge_protocol_sha256",
             "generation_sha256", "likelihood_temperature", "likelihood_convention")})
        data["science"]["token_mask"] = dict(contract=token_mask_contract(), protocol_sha256=digest(token_mask_contract()))
        data["budget_caps"] = {k: 100 for k in cost()}
        self.store = CampaignStore.create(Path(self.temp.name)/"ledger", CampaignManifest.from_dict(data))
        self.actor = FrozenActor(sha("actor"), "7", sha("actual-restore-needed"), sha("actual-freeze-needed"), digest("fixture://actor"))
        from openwebrl import arm_branch_browser
        self.browser = dict(mode="local_process", width=1280, height=720, dpr=1,
            use_screenshot=True, use_a11ytree=False, resize_output_coords=True, resize_scale=1000,
            local_process=dict(server_module="openwebrl.arm_branch_browser",
                server_module_sha256=hashlib.sha256(Path(arm_branch_browser.__file__).read_bytes()).hexdigest(),
                log_dir=str(Path(self.temp.name)/"logs"), port_lock_dir=str(Path(self.temp.name)/"locks")))
        binary = Path(self.temp.name)/"fixture-binary-not-executable"
        binary.write_bytes(b"CPU fixture only, never execute")
        self.browser_manifest = dict(viewport=[1280,720], device_scale_factor=1, locale="en-US",
            timezone="UTC", headless=True, proxy=False, stealth=False, launch_extra_args=[],
            binary_path=str(binary), binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
            branch_state_version="observable-replay-v1")
        self.calls = 0

    def request(self, fake, *, attempt="attempt1", leaf="leaf1", slot=0, prefix=()):
        return EpisodeRequest("request-"+leaf, attempt, leaf, "task-a", "state", "action", slot,
            "train", "group", "tree", digest(fake.task_data), digest(fake.tools), digest(fake.env_policy),
            digest(self.browser), digest(self.browser_manifest), digest(generator_protocol(self.args)), prefix_sha256(prefix))

    async def actor_call(self, **kwargs):
        self.calls += 1
        return ActorOutput("<tool_call>{}</tool_call>", (20, 21), (-1., -2.), "stop", tuple(kwargs["input_ids"]))

    async def version(self):
        return "7"

    async def run_episode(self, fake, request=None, *, plan=True, **overrides):
        request = request or self.request(fake)
        if plan:
            self.store.plan_update(0, policy_sha256=self.contract.policy_sha256, draws=[request.spec()])
        kwargs = dict(store=self.store, contract=self.contract, request=request, actor=self.actor,
            args=self.args, sample=SimpleNamespace(metadata={"task_id": "task-a"}),
            sampling=self.sampling, browser_config=self.browser, browser_manifest=self.browser_manifest,
            reserved_cost=cost(2), local_transport=SimpleNamespace(_server_module=lambda x: "old"),
            actor_call=self.actor_call, weight_version=self.version, verifier_sha256=physical_verifier_sha256(),
            generation=fake, capture=fake.capture)
        kwargs.update(overrides)
        return await run_native_episode(**kwargs)

    async def test_native_episode_exact_context_masks_and_pending_scores(self):
        fake = NativeFake()
        episode = await self.run_episode(fake)
        captured = episode.turns[0]
        self.assertEqual(captured.edge.input_ids, (10, 11))
        self.assertEqual(captured.edge.response_ids, (20, 21, 151645, 198))
        self.assertEqual(captured.edge.sampled_mask, (True, True, False, False))
        self.assertTrue(torch.equal(captured.tensors["pixel_values"], torch.tensor([[1., 2.]])))
        self.assertEqual(captured.edge.old_log_prob_source, "pending_frozen_actor_score")
        self.assertTrue(episode.native_samples[0].remove_sample)
        self.assertEqual(episode.native_samples[0].loss_mask, [1, 1, 0, 0])
        self.assertEqual((fake.created, fake.dispatches, fake.exited), (1, 1, 1))
        state = self.store.snapshot()
        self.assertEqual(state["attempts"]["attempt1"]["status"], "active")
        self.assertEqual(state["online_requests"]["request-leaf1"]["actor"]["weight_version"], "7")
        files = list(Path(episode.artifact_directory).glob("*.json"))
        self.assertTrue(any("physical_dispatch_accepted" in p.read_text() for p in files))

    async def test_generation_seed_is_request_nonce_without_changing_science(self):
        fake = NativeFake()
        request = replace(self.request(fake), generation_seed=1234)
        observed=[]
        async def record(**kwargs):
            observed.append(dict(kwargs["sampling"]))
            return await self.actor_call(**kwargs)
        await self.run_episode(fake, request, actor_call=record)
        self.assertEqual({k:v for k,v in observed[0].items() if k != "sampling_seed"}, self.sampling)
        self.assertIs(type(observed[0]["sampling_seed"]),int)
        self.assertEqual(self.store.snapshot()["online_requests"][request.request_id]["request"]["generation_seed"],1234)

    async def test_candidate_id_is_fixed_before_exact_prefix_dispatch(self):
        from openwebrl.action_credit_binding import q_inventory_contract
        data=self.store.manifest.to_dict()
        data["arm_id"]="continuation_q"
        data["science"]["algorithm"]["q_inventory_protocol"]=q_inventory_contract()
        self.store=CampaignStore.create(Path(self.temp.name)/"q-ledger",CampaignManifest.from_dict(data))
        fake=NativeFake()
        request=replace(self.request(fake),action_id="fixed-candidate",inventory_role="q_fit",
            planned_candidate_edge_id="fixed-candidate",candidate_prefix_length=0)
        episode=await self.run_episode(fake,request)
        self.assertEqual(episode.turns[0].edge.edge_id,"fixed-candidate")
        with self.assertRaisesRegex(CollectionRejected,"prefix boundary"):
            await self.run_episode(fake,replace(request,candidate_prefix_length=1),plan=False)
        self.assertEqual(fake.created,1)

    async def test_generation_seed_rejects_boolean_before_dispatch(self):
        fake=NativeFake()
        with self.assertRaisesRegex(CollectionRejected,"integer generation seed"):
            await self.run_episode(fake,replace(self.request(fake),generation_seed=True))
        self.assertEqual(fake.created,0)

    async def test_crash_or_repeat_never_redispatches_and_preserves_reservation(self):
        fake = NativeFake()
        await self.run_episode(fake)
        with self.assertRaises(ReconcileRequired):
            await self.run_episode(fake, plan=False)
        self.assertEqual(fake.created, 1)
        self.assertEqual(self.store.snapshot()["charged_resources"], cost(2))

    async def test_physical_mismatch_cannot_be_hidden_by_native_swallowed_exception(self):
        fake = NativeFake(changed=True, swallow=True)
        with self.assertRaises(CollectionRejected):
            await self.run_episode(fake)
        self.assertEqual((fake.dispatches, fake.exited), (0, 1))
        self.assertEqual(self.store.snapshot()["attempts"]["attempt1"]["status"], "active")

    async def test_actor_version_change_fails_before_browser_start(self):
        fake = NativeFake()
        async def changed(): return "8"
        with self.assertRaises(CollectionRejected):
            await self.run_episode(fake, weight_version=changed)
        self.assertEqual(fake.created, 0)

    async def test_actor_context_lie_rejected_before_dispatch(self):
        fake = NativeFake(swallow=True)
        async def bad(**kwargs):
            return ActorOutput("text", (20,), (-1.,), "stop", (666,))
        with self.assertRaises(CollectionRejected):
            await self.run_episode(fake, actor_call=bad)
        self.assertEqual(fake.dispatches, 0)

    async def test_task_payload_change_rejected_before_sampling(self):
        fake = NativeFake()
        request = self.request(fake)
        fake.task_data["intent"] = "changed"
        with self.assertRaises(CollectionRejected):
            await self.run_episode(fake, request)
        self.assertEqual(self.calls, 0)

    async def test_changed_request_cannot_reuse_slot(self):
        fake = NativeFake()
        request = self.request(fake)
        await self.run_episode(fake, request)
        with self.assertRaises(CollectionRejected):
            await self.run_episode(fake, replace(request, tools_sha256=sha("changed")), plan=False)
        self.assertEqual(fake.created, 1)

    async def test_explicit_failed_attempt_retry_charges_same_slot_without_reset(self):
        bad = NativeFake(changed=True, swallow=True)
        with self.assertRaises(CollectionRejected):
            await self.run_episode(bad)
        self.store.finish_attempt("attempt1", status="invalid", actual_cost=cost(2), reason="physical mismatch")
        good = NativeFake()
        request = replace(self.request(good), attempt_id="attempt2")
        await self.run_episode(good, request, plan=False)
        self.assertEqual(self.store.snapshot()["charged_resources"], cost(4))
        self.assertEqual(len(self.store.snapshot()["draws"]), 1)

    async def test_two_draw_prefix_replays_original_edge_without_actor_call(self):
        fake = NativeFake()
        first = self.request(fake)
        second_spec = dict(first.spec(), draw_slot=1)
        self.store.plan_update(0, policy_sha256=self.contract.policy_sha256, draws=[first.spec(), second_spec])
        first_result = await self.run_episode(fake, first, plan=False)
        second = self.request(fake, attempt="attempt2", leaf="leaf2", slot=1, prefix=first_result.turns)
        second_result = await self.run_episode(fake, second, plan=False, prefix=first_result.turns)
        self.assertEqual(self.calls, 1)
        self.assertEqual(first_result.turns[0].edge, second_result.turns[0].edge)
        self.assertEqual(fake.dispatches, 2)

    async def test_explicit_browser_budget_and_frozen_sampling_required(self):
        for override in ({"reserved_cost": cost(0)}, {"sampling": dict(self.sampling, temperature=1.)}):
            fake = NativeFake()
            with self.assertRaises(CollectionRejected):
                await self.run_episode(fake, **override)
            self.assertEqual(fake.created, 0)

    async def test_actor_transport_is_reserved_before_call_and_failure_is_counted(self):
        fake = NativeFake(swallow=True)
        async def failed(**kwargs):
            calls = self.store.snapshot()["online_actor_requests"]
            self.assertEqual(len(calls), 1)
            self.assertEqual(next(iter(calls.values()))["status"], "reserved")
            raise TimeoutError("CPU injected failure")
        with self.assertRaises(CollectionRejected):
            await self.run_episode(fake, actor_call=failed)
        self.assertEqual(next(iter(self.store.snapshot()["online_actor_requests"].values()))["status"], "failed")
        self.assertEqual(fake.dispatches, 0)

    async def test_model_failure_without_tool_dispatch_retains_sampled_edge(self):
        fake = NativeFake(dispatch=False)
        episode = await self.run_episode(fake)
        self.assertEqual(len(episode.turns), 1)
        self.assertEqual(fake.dispatches, 0)
        self.assertEqual(episode.turns[0].edge.sampled_mask, (True, True, False, False))
        self.assertEqual(episode.native_samples[0].metadata["terminate_reason"], "format_error_failed")
        events = [p.read_text() for p in Path(episode.artifact_directory).glob("*.json")]
        self.assertTrue(any("unexecuted_response_retained" in event for event in events))

    async def test_sglang_transport_checks_raw_input_ids(self):
        calls = []
        async def post(url, payload):
            calls.append(payload)
            return dict(text="text", meta_info=dict(input_token_logprobs=[(None, 10), (-1., 11)],
                output_token_logprobs=[(-.5, 20)], finish_reason={"type": "stop"}))
        with self.assertRaisesRegex(ValueError, "one HTTP attempt"):
            SGLangActorTransport(post)
        transport = SGLangActorTransport(post, transport_attempts=1)
        kwargs = dict(url="fake://actor", text="text", sampling=self.sampling, images=[], input_ids=(10, 11), timeout=1.)
        result = await transport(**kwargs)
        self.assertEqual(result.token_ids, (20,))
        self.assertEqual(calls[0]["logprob_start_len"], 0)
        with self.assertRaises(CollectionRejected):
            await transport(**dict(kwargs, input_ids=(11, 10)))

    async def test_default_actor_transport_has_no_hidden_retry_or_fallback(self):
        calls, options = [], []
        class Client:
            def __init__(self, **kwargs): options.append(kwargs)
            async def __aenter__(self): return self
            async def __aexit__(self, *args): pass
            async def post(self, url, **kwargs):
                calls.append((url, kwargs))
                raise TimeoutError("CPU fake transport; no request sent")
        with patch("httpx.AsyncClient", Client), patch("httpx.AsyncHTTPTransport") as transport:
            with self.assertRaises(TimeoutError):
                await SGLangActorTransport()(url="fake://actor", text="text", sampling=self.sampling,
                                            images=[], input_ids=(10,11), timeout=1.)
            transport.assert_called_once_with(retries=0)
        self.assertEqual(len(calls), 1)
        self.assertFalse(options[0]["follow_redirects"])
        self.assertFalse(options[0]["trust_env"])

    def test_sampler_stop_is_sampled_but_transport_newline_is_not(self):
        output, mask = ActorOutput("text<|im_end|>", (20, 151645), (-1., -.1), "stop", (10,)).normalized()
        self.assertEqual(output[1], [20, 151645, 198])
        self.assertEqual(mask, (True, True, False))


if __name__ == "__main__":
    unittest.main()
