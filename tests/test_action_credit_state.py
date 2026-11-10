"""Offline persistence/invariant tests; these do not establish GPU readiness."""
from concurrent.futures import ProcessPoolExecutor
import copy
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from openwebrl.action_credit_state import (
    ARM_IDS, COST_DIMENSIONS, READINESS_GATES, BudgetExceeded,
    CampaignManifest, CampaignStore, StateError, digest,
)


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def manifest_data():
    splits = {"task-a": "train", "task-b": "holdout"}
    return dict(schema_version=1, campaign_id="offline-test", arm_id="chain_outcome_grpo",
        initialization=dict(original_actor_sha256=sha("actor"), iteration=0,
                            fresh_optimizer=True, fresh_scheduler=True, fresh_cursor=True),
        science=dict(checkpoint=dict(protocol_sha256=sha("checkpoint")),
            dataset=dict(source_sha256=sha("dataset")),
            judge=dict(model="fixture-judge", prompt_sha256=sha("judge")),
            generation=dict(temperature=.8, top_p=1., top_k=-1,
                            max_response_tokens=1024, max_turns=15),
            token_mask=dict(protocol_sha256=sha("mask")),
            shared_prefix_normalization=dict(protocol_sha256=sha("prefix")),
            algorithm=dict(protocol_sha256=sha("algorithm"))),
        task_splits=splits, split_sha256=digest(splits), code_revision_sha256=sha("code"),
        budget_caps=dict(gpu_seconds=20, api_microdollars=100, api_requests=4, browser_dispatches=4),
        max_attempts_per_draw=3)


def cost(value=1):
    return {key: value for key in COST_DIMENSIONS}


def outcome():
    return dict(artifact_sha256=sha("saved result"), reward=1.)


def spec(slot=0, task="task-a", role="train", action="action"):
    return dict(task_id=task, state_id="state", action_id=action, draw_slot=slot, draw_role=role,
                group_id="group", tree_id="tree")


def reserve(store, request="request", *, task="task-a", split="train", update=0,
            policy=None, slot=0):
    policy = policy or sha("actor")
    if str(update) not in store.snapshot()["updates"]:
        store.plan_update(update, policy_sha256=policy, draws=[spec(slot, task, split)])
    return store.reserve_draw(request, update_id=update, policy_sha256=policy,
                              task_id=task, state_id="state", action_id="action",
                              draw_slot=slot, draw_role=split, group_id="group", tree_id="tree")


def checkpoint_receipt(store, update=0, new_policy="next policy"):
    """Synthetic bound receipt for offline transaction tests, not GPU evidence."""
    state = store.snapshot()["updates"][str(update)]
    return dict(manifest_sha256=store.manifest.sha256,
        input_export_sha256=state["export_sha256"], policy_sha256=state["policy_sha256"],
        new_policy_sha256=sha(new_policy), new_checkpoint_sha256=sha("checkpoint " + str(update)),
        optimizer_state_sha256=sha("optimizer " + str(update)),
        scheduler_state_sha256=sha("scheduler " + str(update)),
        cursor_state_sha256=sha("cursor " + str(update)), validator_evidence_sha256=sha("validator fixture"),
        validator_code_revision_sha256=sha("code"), verification_kind="production_runtime")


def fixture_commit(store, update, new_policy):
    store.seal_update(update)
    store.begin_update(update, attempt_id="train-" + str(update))
    return store.complete_update(update, attempt_id="train-" + str(update),
                                 receipt=checkpoint_receipt(store, update, new_policy))


def process_attempt(root, manifest, attempt):
    store = CampaignStore(root, CampaignManifest.from_dict(manifest))
    draw = reserve(store)["draw_id"]
    return store.start_attempt(draw, attempt, reserved_cost=cost())


def unique_process_attempt(root, manifest, i):
    store = CampaignStore(root, CampaignManifest.from_dict(manifest))
    draw = reserve(store, "request-" + str(i), slot=i)["draw_id"]
    try:
        store.start_attempt(draw, "attempt-" + str(i), reserved_cost=cost())
        return True
    except BudgetExceeded:
        return False


class ActionCreditStateTests(unittest.TestCase):
    def test_q_inventory_roles_are_explicit_immutable_and_disjoint(self):
        data = manifest_data()
        data["arm_id"] = "continuation_q"
        with tempfile.TemporaryDirectory(prefix="action-credit-q-roles-", dir="/tmp") as temp:
            store = CampaignStore.create(Path(temp)/"ledger", CampaignManifest.from_dict(data))
            with self.assertRaisesRegex(StateError, "explicitly prescribe"):
                store.plan_update(0, policy_sha256=sha("actor"), draws=[spec()])
            for inventory_role, draw_role in (("unknown", "train"), ("q_fit", "holdout"), ("q_holdout", "train")):
                with self.assertRaises(StateError):
                    store.plan_update(0, policy_sha256=sha("actor"), draws=[
                        dict(spec(), draw_role=draw_role, inventory_role=inventory_role)])
            specs = [dict(spec(0), inventory_role="base"), dict(spec(1), inventory_role="q_fit"),
                     dict(spec(2, role="holdout"), inventory_role="q_holdout")]
            store.plan_update(0, policy_sha256=sha("actor"), draws=specs)
            changed = [dict(x) for x in specs]
            changed[1]["inventory_role"] = "base"
            with self.assertRaisesRegex(StateError, "immutable"):
                store.plan_update(0, policy_sha256=sha("actor"), draws=changed)
            with self.assertRaisesRegex(StateError, "Unplanned"):
                store.reserve_draw("relabelled", update_id=0, policy_sha256=sha("actor"), **changed[1])
            for i, planned in enumerate(specs):
                draw = store.reserve_draw(str(i), update_id=0, policy_sha256=sha("actor"), **planned)["draw_id"]
                store.mark_missing_draw(draw, reason="CPU fixture admission unavailable")
            store.seal_update(0)
            self.assertEqual({row["inventory_role"] for row in store.training_draws(0)}, {"base", "q_fit"})

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="action-credit-", dir="/tmp")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "campaign"
        self.data = manifest_data()
        self.manifest = CampaignManifest.from_dict(self.data)
        self.store = CampaignStore.create(self.root, self.manifest)

    def restart(self):
        return CampaignStore(self.root, self.manifest)

    def test_all_five_arm_ids_are_explicit_and_manifest_is_deeply_immutable(self):
        self.assertEqual(len(ARM_IDS), 5)
        for arm in ARM_IDS:
            data = manifest_data(); data["arm_id"] = arm
            self.assertEqual(CampaignManifest.from_dict(data).to_dict()["arm_id"], arm)
        self.data["science"]["generation"]["temperature"] = .1
        exported = self.manifest.to_dict(); exported["budget_caps"]["gpu_seconds"] = 10**9
        self.assertEqual(self.manifest.to_dict()["science"]["generation"]["temperature"], .8)
        self.assertEqual(self.manifest.to_dict()["budget_caps"]["gpu_seconds"], 20)

    def test_new_experiment_initialization_caps_and_protocol_pins_fail_closed(self):
        changes = [
            lambda m: m.update(arm_id="new-arm"),
            lambda m: m["initialization"].update(iteration=20),
            lambda m: m["initialization"].update(fresh_optimizer=False),
            lambda m: m["initialization"].update(fresh_scheduler=False),
            lambda m: m["initialization"].update(fresh_cursor=False),
            lambda m: m["budget_caps"].pop("gpu_seconds"),
            lambda m: m["budget_caps"].update(api_microdollars=float("inf")),
            lambda m: m["budget_caps"].update(api_requests=True),
            lambda m: m["budget_caps"].update(gpu_seconds=-1),
            lambda m: m.update(max_attempts_per_draw=0),
            lambda m: m["science"].pop("shared_prefix_normalization"),
            lambda m: m["science"]["generation"].pop("temperature"),
            lambda m: m["science"]["generation"].update(temperature=float("nan")),
            lambda m: m["science"]["generation"].update(top_p=0),
            lambda m: m["science"]["token_mask"].update(protocol_sha256="unpinned"),
            lambda m: m["task_splits"].update({"task-b": "train"}),
        ]
        for change in changes:
            with self.subTest(change=change):
                m = manifest_data(); change(m)
                with self.assertRaises(StateError): CampaignManifest.from_dict(m)

    def test_reopen_refuses_every_changed_science_or_budget_pin(self):
        changes = [lambda m: m["science"]["generation"].update(temperature=.6),
                   lambda m: m["budget_caps"].update(gpu_seconds=21),
                   lambda m: m.update(code_revision_sha256=sha("changed code")),
                   lambda m: m.update(arm_id="gigpo")]
        for change in changes:
            m = manifest_data(); change(m)
            with self.assertRaises(StateError): CampaignStore(self.root, CampaignManifest.from_dict(m))

    def test_draw_and_attempt_ids_are_idempotent_across_restart(self):
        first = reserve(self.store); self.assertTrue(first["created"])
        again = reserve(self.restart()); self.assertEqual(again["draw_id"], first["draw_id"])
        self.assertFalse(again["created"])
        started = self.store.start_attempt(first["draw_id"], "attempt", reserved_cost=cost())
        self.assertTrue(started["created"])
        for _ in range(3):
            self.assertFalse(self.restart().start_attempt(first["draw_id"], "attempt", reserved_cost=cost())["created"])
        self.assertEqual(self.restart().snapshot()["charged_resources"], cost())
        with self.assertRaises(StateError): reserve(self.store, slot=1)
        with self.assertRaises(StateError): reserve(self.store, "different-request")
        with self.assertRaises(StateError): self.store.start_attempt(first["draw_id"], "attempt", reserved_cost=cost(2))

    def test_policy_digest_frozen_per_update_and_original_actor_required_at_zero(self):
        with self.assertRaises(StateError): reserve(self.store, policy=sha("other"))
        with self.assertRaises(StateError): reserve(self.store, update=1, policy=sha("updated policy"))
        initial = reserve(self.store)["draw_id"]
        self.store.mark_missing_draw(initial, reason="explicit fixture omission")
        with self.assertRaises(StateError): reserve(self.store, "unconsumed", update=1, policy=sha("updated policy"))
        fixture_commit(self.store, 0, "updated policy")
        second = reserve(self.store, "second", update=1, policy=sha("updated policy"))["draw_id"]
        with self.assertRaises(StateError): reserve(self.store, "next", update=1, slot=1, policy=sha("other"))
        with self.assertRaises(StateError): reserve(self.store, "skipped", update=3, policy=sha("other"))
        self.store.mark_missing_draw(second, reason="explicit fixture omission")
        fixture_commit(self.store, 1, "other")
        reserve(self.restart(), "next", update=2, policy=sha("other"))

    def test_holdout_persisted_but_never_exported_for_training(self):
        with self.assertRaises(StateError): reserve(self.store, task="task-b", split="train")
        held = reserve(self.store, task="task-b", split="holdout")["draw_id"]
        self.store.start_attempt(held, "held", reserved_cost=cost())
        self.store.finish_attempt("held", status="valid", actual_cost=cost(), outcome=outcome())
        self.store.seal_update(0)
        self.assertEqual(self.restart().training_draws(0), [])
        with self.assertRaises(StateError): self.store.training_draws(0, draw_ids=[held])

    def test_invalid_missing_and_pending_cannot_be_silently_treated_as_reward_zero(self):
        draw = reserve(self.store)["draw_id"]
        with self.assertRaises(StateError): self.store.training_draws(0)
        self.store.start_attempt(draw, "a", reserved_cost=cost())
        with self.assertRaises(StateError): self.store.finish_attempt("a", status="invalid", actual_cost=cost(), outcome=outcome(), reason="browser error")
        with self.assertRaises(StateError): self.store.finish_attempt("a", status="missing", actual_cost=cost())
        self.store.finish_attempt("a", status="invalid", actual_cost=cost(), reason="browser error")
        self.assertEqual(self.store.snapshot()["draws"][draw]["status"], "invalid")
        with self.assertRaises(StateError): self.store.training_draws(0)
        self.store.start_attempt(draw, "b", reserved_cost=cost())
        self.store.finish_attempt("b", status="missing", actual_cost=cost(0), reason="worker crashed")
        self.store.seal_update(0)
        self.assertEqual(self.store.training_draws(0)[0]["status"], "missing")
        self.assertEqual(self.restart().snapshot()["charged_resources"], cost(2))

    def test_retry_costs_persist_no_refunds_and_success_cannot_be_retried(self):
        draw = reserve(self.store)["draw_id"]
        self.store.start_attempt(draw, "a", reserved_cost=cost())
        with self.assertRaises(StateError): self.restart().start_attempt(draw, "b", reserved_cost=cost())
        self.store.finish_attempt("a", status="missing", actual_cost=cost(0), reason="interrupted")
        self.restart().start_attempt(draw, "b", reserved_cost=cost())
        completed = self.store.finish_attempt("b", status="valid", actual_cost=cost(), outcome=outcome())
        self.assertTrue(completed["recorded"])
        self.assertFalse(self.restart().finish_attempt("b", status="valid", actual_cost=cost(), outcome=outcome())["recorded"])
        with self.assertRaises(StateError): self.store.finish_attempt("b", status="valid", actual_cost=cost(0), outcome=outcome())
        with self.assertRaises(StateError): self.store.start_attempt(draw, "c", reserved_cost=cost())
        self.assertEqual(self.store.snapshot()["charged_resources"], cost(2))
        self.store.seal_update(0)
        rows = self.store.training_draws(0)
        self.assertEqual(len(rows), 1); self.assertEqual(rows[0]["outcome"]["reward"], 1.)
        with self.assertRaises(StateError): self.store.training_draws(0, draw_ids=[draw, draw])

    def test_attempt_limit_and_resource_caps_count_failed_attempts(self):
        self.store.plan_update(0, policy_sha256=sha("actor"), draws=[spec(0), spec(1)])
        draw = reserve(self.store)["draw_id"]
        for i in range(3):
            self.store.start_attempt(draw, str(i), reserved_cost=cost())
            self.store.finish_attempt(str(i), status="invalid", actual_cost=cost(0), reason="failure")
        with self.assertRaises(StateError): self.store.start_attempt(draw, "fourth", reserved_cost=cost())
        other = reserve(self.store, "other", slot=1)["draw_id"]
        with self.assertRaises(BudgetExceeded): self.store.start_attempt(other, "big", reserved_cost=cost(2))
        self.assertEqual(self.restart().snapshot()["charged_resources"], cost(3))
        self.assertNotIn("big", self.store.snapshot()["attempts"])

    def test_actual_overrun_is_durable_before_error_and_halts_future_work(self):
        draw = reserve(self.store)["draw_id"]
        self.store.start_attempt(draw, "a", reserved_cost=cost())
        with self.assertRaises(BudgetExceeded):
            self.store.finish_attempt("a", status="invalid", actual_cost=cost(5), reason="provider overrun")
        saved = self.restart().snapshot()
        self.assertEqual(saved["charged_resources"], cost(5))
        self.assertEqual(saved["attempts"]["a"]["actual"], cost(5))
        self.assertEqual(saved["halt"]["reason"], "reservation_exceeded")
        with self.assertRaises(StateError): reserve(self.store, "new", slot=1)

    def test_crash_before_atomic_replace_preserves_old_state_and_does_not_authorize_dispatch(self):
        draw = reserve(self.store)["draw_id"]
        before = (self.root / "state.json").read_bytes()
        with patch("openwebrl.action_credit_state.os.replace", side_effect=OSError("crash before commit")):
            with self.assertRaises(OSError): self.store.start_attempt(draw, "a", reserved_cost=cost())
        self.assertEqual((self.root / "state.json").read_bytes(), before)
        self.assertEqual(self.restart().snapshot()["charged_resources"], cost(0))
        self.assertTrue(list(self.root.glob("state.*.tmp")))
        self.assertTrue(self.restart().start_attempt(draw, "a", reserved_cost=cost())["created"])

    def test_crash_after_replace_recovers_existing_attempt_without_double_spend(self):
        draw = reserve(self.store)["draw_id"]
        real_fsync = os.fsync
        def fail_directory(fd):
            import stat
            if stat.S_ISDIR(os.fstat(fd).st_mode): raise OSError("crash after replace")
            return real_fsync(fd)
        with patch("openwebrl.action_credit_state.os.fsync", side_effect=fail_directory):
            with self.assertRaises(OSError): self.store.start_attempt(draw, "a", reserved_cost=cost())
        resumed = self.restart()
        self.assertFalse(resumed.start_attempt(draw, "a", reserved_cost=cost())["created"])
        self.assertEqual(resumed.snapshot()["charged_resources"], cost())

    def test_lost_corrupt_or_partial_initial_state_never_resets_budget(self):
        original = (self.root / "state.json").read_bytes()
        (self.root / "state.json").rename(self.root / "state.saved")
        with self.assertRaises(StateError): self.restart()
        with self.assertRaises(FileExistsError): CampaignStore.create(self.root, self.manifest)
        (self.root / "state.json").write_text('{"invalid":')
        with self.assertRaises(StateError): self.restart()
        (self.root / "state.json").write_bytes(original)
        envelope = json.loads(original); envelope["state"]["revision"] += 1
        (self.root / "state.json").write_text(json.dumps(envelope))
        with self.assertRaises(StateError): self.restart()
        fresh = Path(self.temp.name) / "interrupted-create"
        with patch("openwebrl.action_credit_state.os.replace", side_effect=OSError("crash")):
            with self.assertRaises(OSError): CampaignStore.create(fresh, self.manifest)
        with self.assertRaises(StateError): CampaignStore(fresh, self.manifest)

    def test_symlinks_and_forbidden_home_are_rejected_before_any_write(self):
        target = Path(self.temp.name) / "target"; target.mkdir()
        link = Path(self.temp.name) / "link"; link.symlink_to(target, target_is_directory=True)
        with self.assertRaises(StateError): CampaignStore.create(link / "child", self.manifest)
        self.assertEqual(list(target.iterdir()), [])
        home_link = Path(self.temp.name) / "home-link"; home_link.symlink_to("/gpfs/home/zixianma", target_is_directory=True)
        with self.assertRaises(StateError): CampaignStore.create(home_link / "never-write", self.manifest)
        with self.assertRaises(StateError): CampaignStore.create("/gpfs/home/zixianma/never-write", self.manifest)
        state = self.root / "state.json"; state.rename(self.root / "state.original")
        state.symlink_to(self.root / "state.original")
        with self.assertRaises(StateError): self.restart()

    def test_process_races_share_one_idempotent_reservation(self):
        reserve(self.store)
        with ProcessPoolExecutor(max_workers=3, mp_context=multiprocessing.get_context("spawn")) as pool:
            results = list(pool.map(process_attempt, [str(self.root)] * 6,
                                    [manifest_data()] * 6, ["same-attempt"] * 6))
        self.assertEqual(sum(x["created"] for x in results), 1)
        self.assertEqual(self.restart().snapshot()["charged_resources"], cost())

    def test_process_races_cannot_oversubscribe_resource_caps(self):
        self.store.plan_update(0, policy_sha256=sha("actor"), draws=[spec(i) for i in range(9)])
        with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("spawn")) as pool:
            results = list(pool.map(unique_process_attempt, [str(self.root)] * 9,
                                    [manifest_data()] * 9, range(9)))
        self.assertEqual(sum(results), 4)
        self.assertEqual(self.restart().snapshot()["charged_resources"], cost(4))

    def test_plan_required_and_slots_candidates_roles_frozen_before_collection(self):
        args = dict(update_id=0, policy_sha256=sha("actor"), **spec())
        with self.assertRaises(StateError): self.store.reserve_draw("r", **args)
        planned = self.store.plan_update(0, policy_sha256=sha("actor"), draws=[spec(0), spec(1)])
        self.assertFalse(self.restart().plan_update(0, policy_sha256=sha("actor"), draws=[spec(1), spec(0)])["created"])
        self.assertEqual(len(planned["draw_ids"]), 2)
        for changed in ([spec(0)], [spec(0), spec(1, role="holdout")], [spec(0), spec(1, action="changed")]):
            with self.assertRaises(StateError): self.store.plan_update(0, policy_sha256=sha("actor"), draws=changed)
        with self.assertRaises(StateError): reserve(self.store, slot=2)
        with self.assertRaises(StateError): self.store.plan_update(1, policy_sha256=sha("next"), draws=[spec(0), spec(0, role="holdout")])

    def test_group_and_tree_are_mandatory_prespecified_and_immutable(self):
        for field in ("group_id", "tree_id"):
            incomplete = spec()
            del incomplete[field]
            with self.assertRaises(StateError):
                self.store.plan_update(0, policy_sha256=sha("actor"), draws=[incomplete])
        self.store.plan_update(0, policy_sha256=sha("actor"), draws=[spec()])
        for field in ("group_id", "tree_id"):
            changed = dict(spec(), **{field: "post-hoc-regrouping"})
            with self.assertRaises(StateError):
                self.store.plan_update(0, policy_sha256=sha("actor"), draws=[changed])
            with self.assertRaises(StateError):
                self.store.reserve_draw("changed", update_id=0, policy_sha256=sha("actor"), **changed)

    def test_terminal_sampling_metadata_requires_artifact_and_cannot_be_rewritten(self):
        draw = reserve(self.store)["draw_id"]
        self.store.start_attempt(draw, "attempt", reserved_cost=cost())
        for status in ("missing", "invalid"):
            with self.assertRaises(StateError):
                self.store.finish_attempt("attempt", status=status, actual_cost=cost(),
                    reason="failed fixture", sampling_terminal_node_id="terminal")
        artifact = {"artifact_sha256": sha("invalid terminal artifact")}
        self.store.finish_attempt("attempt", status="invalid", actual_cost=cost(),
            reason="failed fixture", outcome=artifact, sampling_terminal_node_id="terminal")
        with self.assertRaises(StateError):
            self.store.finish_attempt("attempt", status="invalid", actual_cost=cost(),
                reason="failed fixture", outcome=artifact, sampling_terminal_node_id="other-terminal")
        self.store.seal_update(0)
        self.assertEqual(self.store.training_draws(0)[0]["sampling_terminal_node_id"], "terminal")

    def test_same_task_holdout_continuations_stay_separate_and_full_panel_must_close(self):
        planned = [spec(i, role="train" if i < 6 else "holdout") for i in range(8)]
        self.store.plan_update(0, policy_sha256=sha("actor"), draws=planned)
        ids = []
        for i in range(8):
            ids.append(reserve(self.store, "r" + str(i), slot=i, split="train" if i < 6 else "holdout")["draw_id"])
        self.store.start_attempt(ids[0], "a", reserved_cost=cost())
        self.store.finish_attempt("a", status="valid", actual_cost=cost(), outcome=outcome())
        for i in range(1, 7): self.store.mark_missing_draw(ids[i], reason="prespecified collection stopped")
        with self.assertRaises(StateError): self.store.seal_update(0)
        with self.assertRaises(StateError): self.store.training_draws(0, draw_ids=[ids[0]])
        self.store.mark_missing_draw(ids[7], reason="prespecified collection stopped")
        sealed = self.store.seal_update(0)
        rows = self.store.training_draws(0)
        self.assertEqual(len(rows), 6)
        self.assertEqual(sum(row["valid"] for row in rows), 1)
        self.assertEqual(sum(row["status"] == "missing" for row in rows), 5)
        for bad_ids in ([ids[0]], ids, ids[:5], ids[:6] + [ids[0]]):
            with self.assertRaises(StateError): self.store.training_draws(0, draw_ids=bad_ids)
        self.assertEqual(len(self.store.training_draws(0, draw_ids=ids[:6])), 6)
        with self.assertRaises(StateError): self.store.start_attempt(ids[1], "late-retry", reserved_cost=cost())
        with self.assertRaises(StateError): reserve(self.store, "late-draw", slot=9)
        pending = self.store.consume_update(0, attempt_id="train-0", draw_ids=ids[:6])
        self.assertTrue(pending["created"])
        self.assertEqual(pending["phase"], "training_pending")
        self.assertEqual(pending["input_export_sha256"], sealed["export_sha256"])
        replay = self.restart().begin_update(0, attempt_id="train-0")
        self.assertFalse(replay["created"]); self.assertTrue(replay["reconcile_required"])
        with self.assertRaises(StateError): self.store.begin_update(0, attempt_id="blind-retry")
        with self.assertRaises(StateError): self.store.plan_update(1, policy_sha256=sha("next policy"), draws=[spec()])
        self.store.complete_update(0, attempt_id="train-0", receipt=checkpoint_receipt(self.store))
        done = self.restart().begin_update(0, attempt_id="train-0")
        self.assertEqual(done["phase"], "consumed"); self.assertFalse(done["reconcile_required"])
        self.assertFalse(self.restart().seal_update(0)["created"])
        with self.assertRaises(StateError): self.store.plan_update(1, policy_sha256=sha("wrong output"), draws=[spec()])

    def test_invalid_artifact_preserved_with_zero_training_mask(self):
        draw = reserve(self.store)["draw_id"]
        self.store.start_attempt(draw, "a", reserved_cost=cost())
        invalid = dict(artifact_sha256=sha("invalid saved outcome"))
        self.store.finish_attempt("a", status="invalid", actual_cost=cost(), outcome=invalid, reason="browser error")
        self.store.seal_update(0)
        row = self.store.training_draws(0)[0]
        self.assertFalse(row["valid"])
        self.assertEqual(row["outcome"], invalid)
        self.assertNotIn("reward", row["outcome"])

    def test_inflight_attempt_can_settle_after_another_attempt_halts_campaign(self):
        self.store.plan_update(0, policy_sha256=sha("actor"), draws=[spec(0), spec(1)])
        for i in range(2):
            draw = reserve(self.store, "r" + str(i), slot=i)["draw_id"]
            self.store.start_attempt(draw, "a" + str(i), reserved_cost=cost())
        with self.assertRaises(BudgetExceeded):
            self.store.finish_attempt("a0", status="missing", actual_cost=cost(2), reason="overrun")
        self.restart().finish_attempt("a1", status="valid", actual_cost=cost(), outcome=outcome())
        self.assertEqual(self.store.snapshot()["charged_resources"], cost(3))
        self.assertIsNotNone(self.store.snapshot()["halt"])

    def test_optimizer_intent_crash_requires_reconciliation_and_complete_receipt(self):
        draw = reserve(self.store)["draw_id"]
        self.store.mark_missing_draw(draw, reason="fixture unattempted slot")
        self.store.seal_update(0)
        real_replace = os.replace
        def committed_then_crashed(src, dst):
            real_replace(src, dst)
            raise OSError("process lost after intent rename")
        with patch("openwebrl.action_credit_state.os.replace", side_effect=committed_then_crashed):
            with self.assertRaises(OSError): self.store.begin_update(0, attempt_id="optimizer-a")
        replay = self.restart().begin_update(0, attempt_id="optimizer-a")
        self.assertFalse(replay["created"]); self.assertTrue(replay["reconcile_required"])
        self.assertEqual(self.store.snapshot()["updates"]["0"]["phase"], "training_pending")
        complete = checkpoint_receipt(self.store)
        for field in complete:
            invalid = copy.deepcopy(complete); invalid.pop(field)
            with self.assertRaises(StateError): self.store.complete_update(0, attempt_id="optimizer-a", receipt=invalid)
        for field in ("manifest_sha256", "input_export_sha256", "policy_sha256", "validator_code_revision_sha256"):
            invalid = dict(complete); invalid[field] = sha("other")
            with self.assertRaises(StateError): self.store.complete_update(0, attempt_id="optimizer-a", receipt=invalid)
        with self.assertRaises(StateError):
            self.store.complete_update(0, attempt_id="optimizer-a", receipt=dict(complete, verification_kind="cpu"))
        with self.assertRaises(StateError): self.store.complete_update(0, attempt_id="wrong", receipt=complete)
        self.assertEqual(self.store.snapshot()["updates"]["0"]["phase"], "training_pending")

    def test_checkpoint_receipt_commit_crashes_reconcile_without_automatic_skip_or_replay(self):
        draw = reserve(self.store)["draw_id"]
        self.store.mark_missing_draw(draw, reason="fixture unattempted slot")
        self.store.seal_update(0); self.store.begin_update(0, attempt_id="optimizer-a")
        receipt = checkpoint_receipt(self.store)
        with patch("openwebrl.action_credit_state.os.replace", side_effect=OSError("before ledger commit")):
            with self.assertRaises(OSError): self.store.complete_update(0, attempt_id="optimizer-a", receipt=receipt)
        self.assertTrue(self.restart().begin_update(0, attempt_id="optimizer-a")["reconcile_required"])
        real_replace = os.replace
        def committed_then_crashed(src, dst):
            real_replace(src, dst)
            raise OSError("after durable snapshot rename")
        with patch("openwebrl.action_credit_state.os.replace", side_effect=committed_then_crashed):
            with self.assertRaises(OSError): self.store.complete_update(0, attempt_id="optimizer-a", receipt=receipt)
        resumed = self.restart()
        self.assertEqual(resumed.snapshot()["updates"]["0"]["training_receipt"], receipt)
        self.assertFalse(resumed.complete_update(0, attempt_id="optimizer-a", receipt=receipt)["recorded"])
        self.assertEqual(resumed.begin_update(0, attempt_id="optimizer-a")["phase"], "consumed")
        with self.assertRaises(StateError):
            resumed.complete_update(0, attempt_id="optimizer-a", receipt=dict(receipt, new_checkpoint_sha256=sha("changed")))

    def test_readiness_binds_manifest_code_protocol_and_policy_without_authorizing_launch(self):
        reserve(self.store)
        evidence = dict(manifest_sha256=self.manifest.sha256,
            code_revision_sha256=sha("code"), protocol_sha256=self.manifest.protocol_sha256,
            policy_sha256=sha("actor"), plan_sha256=self.store.snapshot()["updates"]["0"]["plan_sha256"],
            evidence_sha256=sha("cpu proof"), verification_kind="cpu")
        for key in ("manifest_sha256", "code_revision_sha256", "protocol_sha256", "policy_sha256", "plan_sha256"):
            changed = dict(evidence); changed[key] = sha("mismatched")
            with self.assertRaises(StateError):
                self.store.record_readiness_gate("token_masks", update_id=0, passed=True, evidence=changed)
        for gate in READINESS_GATES:
            self.store.record_readiness_gate(gate, update_id=0, passed=True, evidence=evidence)
        self.assertFalse(self.restart().readiness(0)["evidence_complete"])
        for gate in READINESS_GATES:
            production = dict(evidence, evidence_sha256=sha("runtime " + gate), verification_kind="production_runtime")
            self.store.record_readiness_gate(gate, update_id=0, passed=True, evidence=production)
            self.store.record_readiness_gate(gate, update_id=0, passed=True, evidence=production)
            with self.assertRaises(StateError):
                self.store.record_readiness_gate(gate, update_id=0, passed=False, evidence=production)
        status = self.restart().readiness(0)
        self.assertTrue(status["evidence_complete"])
        self.assertFalse(status["launch_ready"])
        self.assertFalse(status["authorizes_execution"])
        self.assertFalse(self.store.readiness(1)["evidence_complete"])
        failed = dict(evidence, evidence_sha256=sha("later failure"), verification_kind="production_runtime")
        self.store.record_readiness_gate("browser_replay", update_id=0, passed=False, evidence=failed)
        self.assertFalse(self.restart().readiness(0)["evidence_complete"])
        later = dict(failed, evidence_sha256=sha("subsequent pass"))
        self.store.record_readiness_gate("browser_replay", update_id=0, passed=True, evidence=later)
        self.assertFalse(self.restart().readiness(0)["evidence_complete"])


if __name__ == "__main__":
    unittest.main()
