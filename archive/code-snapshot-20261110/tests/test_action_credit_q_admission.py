"""Sealed real-ledger Q abstentions; all outcomes/episodes are CPU fixtures."""
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path
import tempfile
import unittest

from test_action_credit import graph_fixture
from test_action_credit_objective import config
from test_action_credit_state import manifest_data, cost
from openwebrl.action_credit import Leaf, ResponseGraph, digest
from openwebrl.action_credit_binding import (SealedCollection, q_inventory_contract,
    shared_prefix_contract, token_mask_contract)
from openwebrl.action_credit_objective import ContinuationPanel, build_objective
from openwebrl.action_credit_receipts import ReplayReceipt
from openwebrl.action_credit_state import CampaignManifest, CampaignStore


@contextmanager
def admission_fixture(*, base_count=3, candidate_incomplete=False, mutation=None):
    original = graph_fixture()
    contract = original.contract
    cfg = config("continuation_q")
    panel_id = "group:localq"
    source_id = "group:chain00:terminal"
    edges, leaves, specs = [], [], []
    for index in range(base_count):
        leaf_id = source_id if index == 0 else "base-"+str(index)
        tree_id = "base-tree-"+str(index)
        valid = index != 0 or candidate_incomplete
        edge_id = "base-edge-"+str(index)
        if valid:
            edges.append(replace(original.edges[0], edge_id=edge_id, tree_id=tree_id))
        leaves.append(Leaf(leaf_id, "task", tree_id, (edge_id,) if valid else (),
                           "valid" if valid else "missing", float(index % 2) if valid else None,
                           group_id="group"))
        specs.append(dict(task_id="task", state_id="base-state", action_id="base-action",
            draw_slot=index, draw_role="train", group_id="group", tree_id=tree_id, inventory_role="base"))
    for candidate in range(5):
        action = "candidate-"+str(candidate)
        valid = candidate_incomplete and candidate == 0
        if valid:
            edges.append(replace(original.edges[0], edge_id=action, tree_id="panel-tree"))
        for slot in range(8):
            key = action+":"+str(slot)
            split = "train" if slot < 6 else "holdout"
            leaves.append(Leaf(key, "task", "panel-tree", (action,) if valid else (),
                "valid" if valid else "missing", float(slot % 2) if valid else None,
                split=split, group_id="group"))
            specs.append(dict(task_id="task", state_id=panel_id, action_id=action,
                draw_slot=slot, draw_role=split, group_id="group", tree_id="panel-tree",
                inventory_role="q_fit" if slot < 6 else "q_holdout"))
    receipts = {e.edge_id: ReplayReceipt(e.edge_id, e.contract_sha256, e.state_sha256,
        e.context_sha256, e.remaining_turns, digest("fixture-verifier"), digest("fixture-evidence"), True)
        for e in edges}
    graph = ResponseGraph(contract, tuple(replace(e, replay_evidence_sha256=receipts[e.edge_id].sha256)
        for e in edges), tuple(leaves), tuple(l.leaf_id for l in leaves))
    manifest = manifest_data()
    manifest.update(arm_id=cfg.arm_id, task_splits={"task": "train", "test": "holdout"}, budget_caps=cost(0))
    manifest["split_sha256"] = digest(manifest["task_splits"])
    manifest["initialization"]["original_actor_sha256"] = contract.policy_sha256
    science = manifest["science"]
    science["algorithm"].update(objective_config_sha256=digest(asdict(cfg)),
        leaf_selection={"mode": "all_fit_inventory"}, q_inventory_protocol=q_inventory_contract())
    science["checkpoint"].update(tokenizer_sha256=contract.tokenizer_sha256, processor_sha256=contract.processor_sha256)
    science["dataset"]["task_manifest_sha256"] = contract.task_manifest_sha256
    science["judge"]["judge_protocol_sha256"] = contract.judge_protocol_sha256
    science["generation"].update(generation_sha256=contract.generation_sha256,
        likelihood_temperature=contract.likelihood_temperature, likelihood_convention=contract.likelihood_convention)
    for section, protocol in (("token_mask", token_mask_contract()),
                              ("shared_prefix_normalization", shared_prefix_contract(cfg.reduction))):
        science[section] = dict(contract=protocol, protocol_sha256=digest(protocol))
    with tempfile.TemporaryDirectory(dir="/tmp", prefix="action-credit-q-admission-") as temp:
        store = CampaignStore.create(Path(temp)/"campaign", CampaignManifest.from_dict(manifest))
        store.plan_update(0, policy_sha256=contract.policy_sha256, draws=specs)
        mapping = {}
        for leaf, spec in zip(leaves, specs):
            draw = store.reserve_draw("request-"+leaf.leaf_id, update_id=0,
                policy_sha256=contract.policy_sha256, **spec)["draw_id"]
            mapping[leaf.leaf_id] = draw
            if leaf.status == "missing":
                store.mark_missing_draw(draw, reason="prespecified source/candidate unavailable")
            else:
                attempt = "attempt-"+leaf.leaf_id
                store.start_attempt(draw, attempt, reserved_cost=cost(0))
                store.finish_attempt(attempt, status="valid", actual_cost=cost(0),
                    outcome=dict(artifact_sha256=digest(leaf.leaf_id), reward=leaf.reward))
        with store._transaction() as state:
            protocol = {"source_selection": "fixed-first-chain-first-eight-groups"}
            state["online_collection_plans"] = {"0": dict(graph_contract_sha256=contract.sha256,
                groups=[dict(group_id="group")], protocol=protocol,
                draws=[dict(leaf_id=leaf.leaf_id, spec=spec) for leaf, spec in zip(leaves, specs)])}
            state["online_panel_admissions"] = {"0:"+panel_id: dict(
                status="admitted" if candidate_incomplete else "abstained", source_leaf_id=source_id,
                source_attempt_id="attempt-"+source_id if candidate_incomplete else None,
                depth=0 if candidate_incomplete else None,
                source_context_sha256=digest("context") if candidate_incomplete else None,
                source_evidence_sha256=digest("evidence") if candidate_incomplete else None,
                policy_sha256=contract.policy_sha256, selection_policy_sha256=digest(protocol),
                reason="fixture physical infrastructure unavailable")}
            if candidate_incomplete:
                # The binding checks durable episode identity membership. This
                # synthetic pointer is not a physical-browser acceptance claim.
                state["pending_episodes"] = {"attempt-"+source_id: {"fixture": True}}
                closed = {key: {k: state["draws"][draw][k] for k in ("status", "selected_attempt")}
                          for key, draw in mapping.items() if key.startswith("candidate-")}
                state["online_panel_completions"] = {"0:"+panel_id: dict(status="candidate_incomplete",
                    missing_candidate_edge_ids=["candidate-"+str(i) for i in range(1, 5)],
                    closed_inventory_sha256=digest(closed))}
            if mutation:
                mutation(state, mapping)
        store.seal_update(0)
        yield graph, cfg, receipts, store, mapping


class QAdmission(unittest.TestCase):
    def prepare(self, values, **kwargs):
        graph, cfg, receipts, store, mapping = values
        binding = SealedCollection.from_store(store, graph.contract, mapping)
        return build_objective(graph, cfg, collection_binding=binding, replay_receipts=receipts, **kwargs)

    def test_all_384_base_slots_survive_unavailable_source_panel(self):
        with admission_fixture(base_count=384) as values:
            result = self.prepare(values)
            self.assertEqual(len(result.plan.graph.leaves), 424)
            self.assertEqual(len(result.plan.terms), 383)
            self.assertEqual(result.diagnostics[0].groups[0].planned_count, 384)
            self.assertEqual(result.diagnostics[0].groups[0].missing_count, 1)
            self.assertEqual({term.component for term in result.plan.terms}, {"outcome"})
            self.assertTrue(all(term.source_id.startswith("base-") for term in result.plan.terms))

    def test_candidate_incomplete_retains_real_other_candidate_outcomes_only_as_inventory(self):
        with admission_fixture(candidate_incomplete=True) as values:
            result = self.prepare(values)
            self.assertEqual(len(result.plan.graph.leaves), 43)
            self.assertEqual(len(result.plan.terms), 3)
            self.assertEqual(result.diagnostics[0].groups[0].planned_count, 3)
            self.assertFalse(any(term.edge_id.startswith("candidate-") for term in result.plan.terms))
        def received_partial_candidate(state, mapping):
            draw_id = mapping["candidate-1:0"]
            attempt_id = "failed-after-receiving-candidate"
            completion = dict(status="missing", actual_cost=cost(0), outcome=None,
                              reason="infrastructure failure after response", sampling_terminal_node_id=None)
            state["attempts"][attempt_id] = dict(draw_id=draw_id, reserved=cost(0), charged=cost(0),
                actual=cost(0), status="missing", completion=completion)
            state["draws"][draw_id].update(attempts=[attempt_id], selected_attempt=attempt_id)
            state["online_actor_requests"] = {"request": dict(attempt_id=attempt_id, turn=0, status="received")}
            closed = {key: {k: state["draws"][draw][k] for k in ("status", "selected_attempt")}
                      for key, draw in mapping.items() if key.startswith("candidate-")}
            state["online_panel_completions"]["0:group:localq"]["closed_inventory_sha256"] = digest(closed)
        with admission_fixture(candidate_incomplete=True, mutation=received_partial_candidate) as values:
            with self.assertRaisesRegex(ValueError, "Received candidate tokens"):
                self.prepare(values)

    def test_unsealed_panel_omission_and_changed_source_selection_are_rejected(self):
        for mutate in (
            lambda state, _: state.pop("online_panel_admissions"),
            lambda state, _: state["online_panel_admissions"]["0:group:localq"].update(source_leaf_id="base-1"),
            lambda state, _: state["online_panel_admissions"]["0:group:localq"].update(selection_policy_sha256=digest("changed")),
        ):
            with admission_fixture(mutation=mutate) as values, self.assertRaises(ValueError):
                self.prepare(values)

    def test_abstained_panel_cannot_also_supply_targets_or_fake_sampled_candidate(self):
        with admission_fixture(candidate_incomplete=True) as values:
            graph, cfg, receipts, store, mapping = values
            panel = ContinuationPanel("group:localq", tuple("candidate-"+str(i) for i in range(5)),
                tuple(tuple("candidate-"+str(i)+":"+str(j) for j in range(8)) for i in range(5)))
            with self.assertRaises(ValueError):
                self.prepare(values, panels=(panel,))
            extra = replace(graph.edges[-1], edge_id="candidate-1")
            changed = replace(graph, edges=(*graph.edges, extra))
            # Fail at graph/receipt/admission boundary, never create local loss.
            with self.assertRaises(ValueError):
                self.prepare((changed, cfg, receipts, store, mapping))


if __name__ == "__main__":
    unittest.main()
