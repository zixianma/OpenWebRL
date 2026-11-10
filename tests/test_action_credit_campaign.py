"""Complete fixed-task ledger -> scored saved batch, CPU fixtures only."""
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch

from openwebrl.action_credit import CollectionContract, ResponseEdge, digest
from openwebrl.action_credit_binding import token_mask_contract, shared_prefix_contract
from openwebrl.action_credit_campaign import (plan_independent_update, draft_independent_update,
                                            save_scored_pending_update, validate_optimizer_rng_artifact)
from openwebrl.action_credit_collect import CapturedResponse, EpisodeRequest, PendingEpisode, prefix_sha256
from openwebrl.action_credit_megatron import BackendRow, FrozenScoreReceipt, tensor_payload_sha256
from openwebrl.action_credit_receipts import ReplayReceipt
from openwebrl.action_credit_state import CampaignManifest, CampaignStore
from openwebrl.action_credit_windows import WindowConfig
from test_action_credit_state import manifest_data, sha, checkpoint_receipt
from test_action_credit_binding import config


class CampaignTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="action-credit-campaign-", dir="/tmp")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.config = config("chain_outcome_grpo")
        self.window_config = WindowConfig(2, 256, 42)
        self.contract = CollectionContract(0, sha("actor"), sha("tokenizer"), sha("processor"),
            sha("tasks"), sha("judge"), sha("generation"), .8, ("task-a",), ("task-b",))
        order = dict(source_sha256=sha("dataset"), groups_per_iteration=1, outcome_dependent_replacement=False,
            original_manifest_task_order=["task-a"], iterations=[dict(iteration=1,
                groups=[dict(group_id="iter01-group00", task_id="task-a", source_index=0)])])
        self.order = self.root/"task-order.json"
        self.order.write_text(json.dumps(order))
        data = manifest_data()
        data["science"]["dataset"]["task_order_sha256"] = hashlib.sha256(self.order.read_bytes()).hexdigest()
        algorithm = data["science"]["algorithm"]
        algorithm.update(groups_per_iteration=1, chains_per_group=2,
            objective_config_sha256=digest(asdict(self.config)), leaf_selection={"mode":"all_fit_inventory"},
            ppo_window_config=asdict(self.window_config), training_world_size=8)
        data["science"]["checkpoint"].update(native_initial_state_root=str(self.root/"native-initial-state"),
            native_checkpoint_validator_sha256=sha("synthetic CPU validator; no physical proof"))
        algorithm.update({k:getattr(self.contract,k) for k in ("tokenizer_sha256", "processor_sha256",
            "task_manifest_sha256", "judge_protocol_sha256", "generation_sha256", "likelihood_temperature", "likelihood_convention")})
        for section, value in (("token_mask", token_mask_contract()),
                               ("shared_prefix_normalization", shared_prefix_contract(self.config.reduction))):
            data["science"][section] = dict(contract=value, protocol_sha256=digest(value))
        self.store = CampaignStore.create(self.root/"ledger", CampaignManifest.from_dict(data))
        self.plan = plan_independent_update(self.store, self.contract, self.order)
        self.rng_artifact = self.rng_fixture()

    def write_artifact(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    def rng_fixture(self):
        """Hashable CPU placeholders only; not GPU restoration evidence."""
        initial = self.root/"native-initial-state"
        # Native shards can contain several components; hash each file once.
        combined = self.write_artifact(initial/"synthetic-shard.json", dict(evidence="CPU fixture, not native checkpoint"))
        previous = self.write_artifact(initial/"inventory.json", dict(schema_version=1,
            policy_sha256=self.contract.policy_sha256, completed_updates=0, checkpoint_kind="native_initialization",
            fresh_optimizer=True, fresh_scheduler=True, fresh_cursor=True,
            files=[dict(combined, components=["model", "optimizer", "scheduler", "cursor", "rng"])]))
        ranks = [dict(self.write_artifact(self.root/f"rng/rank-{i}.json", dict(rank=i, evidence="CPU fixture")), rank=i)
                 for i in range(8)]
        return self.write_artifact(self.root/"rng/sidecar.json", dict(schema_version=1,
            manifest_sha256=self.store.manifest.sha256, policy_sha256=self.contract.policy_sha256,
            graph_contract_sha256=self.contract.sha256, update_id=0, world_size=8,
            rank_rng_files=ranks, previous_checkpoint_inventory=previous,
            validator_source_sha256=sha("synthetic CPU validator; no physical proof")))

    def complete(self, item, reward=1.):
        spec, leaf = item["spec"], item["leaf_id"]
        request_id, attempt = "request-"+leaf, "attempt-"+leaf
        draw = self.store.reserve_draw(request_id, update_id=0, policy_sha256=self.contract.policy_sha256, **spec)["draw_id"]
        cost = dict(gpu_seconds=1, browser_dispatches=1, api_requests=0, api_microdollars=0)
        self.store.start_attempt(draw, attempt, reserved_cost=cost)
        self.store.finish_attempt(attempt, status="valid", actual_cost=cost,
                                  outcome=dict(artifact_sha256=sha("CPU artifact "+leaf), reward=reward))
        edge_id = "edge-"+leaf
        receipt = ReplayReceipt(edge_id, self.contract.sha256, sha("state"),
            digest(dict(input_ids=(10,11), media_sha256=())), 15, sha("CPU verifier"), sha("CPU evidence"), True)
        edge = ResponseEdge(edge_id, spec["task_id"], spec["tree_id"], None, self.contract.sha256,
            (10,11), (), (20,21), (True,True), (0.,0.), self.contract.policy_sha256, sha("state"), receipt.sha256,
            15, "pending_frozen_actor_score")
        row = BackendRow(edge_id, edge.input_ids, edge.response_ids, edge.sampled_mask, (),
                          self.contract.processor_sha256, tensor_payload_sha256({}))
        captured = CapturedResponse(edge, row, receipt, ("fixture", [20,21], [-1.,-1.], "stop"), "{}", b"", {}, sha("CPU evidence"))
        request = EpisodeRequest(request_id=request_id, attempt_id=attempt, leaf_id=leaf, **spec,
            task_data_sha256=sha("task"), tools_sha256=sha("tools"), environment_policy_sha256=sha("environment"),
            browser_config_sha256=sha("browser"), browser_manifest_sha256=sha("browsermanifest"),
            generator_protocol_sha256=sha("native protocol"), prefix_sha256=prefix_sha256(()))
        return PendingEpisode(draw, sha("request"), request, (captured,), {}, (), str(self.root))

    def scores(self, draft):
        graph = draft["graph"]
        return {edge.edge_id: FrozenScoreReceipt(edge.edge_id, graph.contract.sha256, graph.contract.policy_sha256,
            edge.context_sha256, digest(dict(response_ids=edge.response_ids, sampled_mask=edge.sampled_mask)),
            draft["rows"][edge.edge_id].tensor_payload_sha256, sha("CPU scorer"), .8, (-1., -2.)) for edge in graph.edges}

    def test_full_plan_scored_and_saved_before_optimizer_intent(self):
        episodes = [self.complete(item, float(i==0)) for i,item in enumerate(self.plan["draws"])]
        draft = draft_independent_update(self.store, self.contract, self.plan, episodes)
        with self.assertRaises(ValueError): draft["graph"].validate()
        pointer, prepared = save_scored_pending_update(self.store, self.config, draft, self.scores(draft),
            scorer_source_sha256=sha("CPU scorer"), window_config=self.window_config,
            optimizer_rng_artifact=self.rng_artifact)
        prepared.plan.validate()
        self.assertEqual(sum(t.weight for t in prepared.plan.terms), 1.)
        raw = Path(pointer["path"]).read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), pointer["file_sha256"])
        restored = torch.load(pointer["path"], weights_only=False, map_location="cpu")
        restored["graph"].validate(independent_chains=True)
        restored["ppo_schedule"].validate(prepared)
        self.assertEqual(restored["ppo_schedule"].sha256, pointer["ppo_schedule_sha256"])
        self.assertEqual(len(restored["ppo_schedule"].windows), 2)
        self.assertTrue(all(w.virtual_padding_slots == 254 for w in restored["ppo_schedule"].windows))
        self.assertEqual(restored["optimizer_rng_artifact"], pointer["optimizer_rng_artifact"])
        self.assertFalse(pointer["optimizer_rng_artifact"]["physical_restore_verified_by_this_function"])
        state = self.store.snapshot()
        self.assertEqual(state["updates"]["0"]["phase"], "sealed")
        self.assertIsNone(state["updates"]["0"]["training_intent"])
        self.assertFalse(pointer["optimizer_started"])

    def test_changed_window_rng_or_policy_rejected_before_sealing(self):
        episodes = [self.complete(item, float(i==0)) for i,item in enumerate(self.plan["draws"])]
        draft = draft_independent_update(self.store, self.contract, self.plan, episodes)
        with self.assertRaisesRegex(ValueError, "frozen campaign"):
            save_scored_pending_update(self.store, self.config, draft, self.scores(draft),
                scorer_source_sha256=sha("CPU scorer"), window_config=replace(self.window_config, seed=43),
                optimizer_rng_artifact=self.rng_artifact)
        self.assertEqual(self.store.snapshot()["updates"]["0"]["phase"], "collecting")

    def test_rng_sidecar_covers_all_ranks_and_full_previous_native_initial_state(self):
        validate_optimizer_rng_artifact(self.store, self.contract, self.rng_artifact)
        original = json.loads(Path(self.rng_artifact["path"]).read_text())
        for changes in (dict(rank_rng_files=original["rank_rng_files"][:-1]),
                        dict(world_size=4), dict(validator_source_sha256=sha("wrong validator")),
                        dict(policy_sha256=sha("late checkpoint")), dict(update_id=1)):
            changed = self.write_artifact(self.root/"rng/changed.json", dict(original, **changes))
            with self.assertRaises(ValueError):
                validate_optimizer_rng_artifact(self.store, self.contract, changed)
        rank = Path(original["rank_rng_files"][7]["path"])
        rank.write_text("changed after snapshot")
        with self.assertRaisesRegex(ValueError, "SHA mismatch"):
            validate_optimizer_rng_artifact(self.store, self.contract, self.rng_artifact)

    def test_rng_sidecar_rejects_weights_only_checkpoint_and_home_or_symlink_reference(self):
        saved = json.loads(Path(self.rng_artifact["path"]).read_text())
        previous_ref = saved["previous_checkpoint_inventory"]
        previous = json.loads(Path(previous_ref["path"]).read_text())
        previous["files"][0]["components"] = ["model"]
        saved["previous_checkpoint_inventory"] = self.write_artifact(Path(previous_ref["path"]), previous)
        changed = self.write_artifact(self.root/"rng/weights-only.json", saved)
        with self.assertRaisesRegex(ValueError, "full model/optimizer"):
            validate_optimizer_rng_artifact(self.store, self.contract, changed)
        with self.assertRaises(ValueError):
            validate_optimizer_rng_artifact(self.store, self.contract,
                dict(path="/gpfs/home/zixianma/forbidden-artifact", sha256=sha("not read")))
        with self.assertRaisesRegex(ValueError, "traversal"):
            validate_optimizer_rng_artifact(self.store, self.contract,
                dict(path="/tmp/../etc/hosts", sha256=sha("not read")))
        link = self.root/"sidecar-symlink.json"
        link.symlink_to(self.rng_artifact["path"])
        with self.assertRaises(ValueError):
            validate_optimizer_rng_artifact(self.store, self.contract,
                dict(path=str(link), sha256=self.rng_artifact["sha256"]))

    def test_continuation_cannot_substitute_optimizer_under_same_policy(self):
        for i, item in enumerate(self.plan["draws"]):
            self.complete(item, float(i == 0))
        self.store.seal_update(0)
        self.store.begin_update(0, attempt_id="CPU optimizer intent")
        receipt = checkpoint_receipt(self.store)
        contract = replace(self.contract, update_id=1, policy_sha256=receipt["new_policy_sha256"])
        old_sidecar = json.loads(Path(self.rng_artifact["path"]).read_text())
        files = json.loads(Path(old_sidecar["previous_checkpoint_inventory"]["path"]).read_text())["files"]
        inventory = dict(schema_version=1, policy_sha256=contract.policy_sha256, completed_updates=1,
            checkpoint_kind="native_completed_update", files=files,
            component_state_sha256={name: receipt[name+"_state_sha256"] for name in ("optimizer", "scheduler", "cursor")})
        previous = self.write_artifact(self.root/"completed-update1/inventory.json", inventory)
        receipt["new_checkpoint_sha256"] = previous["sha256"]
        self.store.complete_update(0, attempt_id="CPU optimizer intent", receipt=receipt)
        sidecar = dict(old_sidecar, policy_sha256=contract.policy_sha256, graph_contract_sha256=contract.sha256,
                       update_id=1, previous_checkpoint_inventory=previous)
        reference = self.write_artifact(self.root/"rng/update1.json", sidecar)
        validate_optimizer_rng_artifact(self.store, contract, reference)
        inventory["component_state_sha256"]["optimizer"] = sha("different optimizer, same actor policy")
        sidecar["previous_checkpoint_inventory"] = self.write_artifact(self.root/"substitute/inventory.json", inventory)
        changed = self.write_artifact(self.root/"rng/substitute.json", sidecar)
        with self.assertRaisesRegex(ValueError, "consumed prior training receipt"):
            validate_optimizer_rng_artifact(self.store, contract, changed)

    def test_partial_or_dropped_valid_sibling_rejected(self):
        first = self.complete(self.plan["draws"][0])
        with self.assertRaises(ValueError):
            draft_independent_update(self.store, self.contract, self.plan, [first])
        self.complete(self.plan["draws"][1])
        with self.assertRaises(ValueError):
            draft_independent_update(self.store, self.contract, self.plan, [first])

    def test_explicit_missing_slot_survives_full_inventory(self):
        episode = self.complete(self.plan["draws"][0])
        item = self.plan["draws"][1]
        draw = self.store.reserve_draw("missing-request", update_id=0, policy_sha256=self.contract.policy_sha256,
                                      **item["spec"])["draw_id"]
        self.store.mark_missing_draw(draw, reason="CPU fixture prespecified admission failure")
        draft = draft_independent_update(self.store, self.contract, self.plan, [episode])
        self.assertEqual([x.status for x in draft["graph"].leaves], ["valid", "missing"])
        self.assertEqual(len(draft["graph"].planned_leaf_ids), 2)

    def test_task_order_mutation_and_cursor_change_rejected(self):
        self.order.write_text(self.order.read_text()+" ")
        with self.assertRaises(ValueError):
            plan_independent_update(self.store, self.contract, self.order)
        with self.assertRaises(ValueError):
            plan_independent_update(self.store, replace(self.contract, update_id=1), self.order)


if __name__ == "__main__": unittest.main()
