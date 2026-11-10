"""Real-ledger boundary tests with synthetic offline outcomes, never GPU proof."""
from contextlib import contextmanager
from dataclasses import FrozenInstanceError, asdict, replace
from pathlib import Path
import tempfile
import unittest

from test_action_credit import graph_fixture
from test_action_credit_state import manifest_data, cost, checkpoint_receipt
from openwebrl.action_credit import Leaf, digest
from openwebrl.action_credit_binding import (SealedCollection, validate_collection_binding,
    token_mask_contract, shared_prefix_contract, q_inventory_contract)
from openwebrl.action_credit_objective import ContinuationPanel, ObjectiveConfig
from openwebrl.action_credit_state import CampaignManifest, CampaignStore


def config(arm):
    return ObjectiveConfig(arm, "response_mean", "mean_std", "zero", .2, .28, None,
        1. if arm == "continuation_q" else 0., 1. if arm == "gigpo" else 0., 1., "trajectory",
        "state_history_horizon")


def panel():
    return ContinuationPanel("s", ("good", "bad"), (("a",), ("b",)))


@contextmanager
def collection_fixture(graph, config, panels=(), *, seal=True, manifest_change=None, selection=None,
                       terminal_metadata=None, role_overrides=None):
    """Test-only ledger generated from a fixture; not an independent collection.

    Only explicitly panel-assigned leaves estimate Q. Unassigned leaves are
    the independent base population. Real collectors prescribe these roles
    before observing any outcome; this helper is only for synthetic fixtures.
    """
    manifest = manifest_data()
    contract = graph.contract
    manifest["arm_id"] = config.arm_id
    manifest["science"]["algorithm"]["objective_config_sha256"] = digest(asdict(config))
    manifest["science"]["algorithm"]["leaf_selection"] = (
        selection.policy if selection is not None else {"mode": "all_fit_inventory"})
    if config.arm_id == "continuation_q":
        manifest["science"]["algorithm"]["q_inventory_protocol"] = q_inventory_contract()
    for section, protocol in (("token_mask", token_mask_contract()),
             ("shared_prefix_normalization", shared_prefix_contract(config.reduction))):
        manifest["science"][section] = dict(contract=protocol, protocol_sha256=digest(protocol))
    manifest["initialization"]["original_actor_sha256"] = contract.policy_sha256
    manifest["task_splits"] = {task: "train" for task in contract.train_task_ids}
    manifest["task_splits"].update({task: "holdout" for task in contract.heldout_task_ids})
    manifest["split_sha256"] = digest(manifest["task_splits"])
    manifest["budget_caps"] = cost(0)
    manifest["science"]["checkpoint"].update(tokenizer_sha256=contract.tokenizer_sha256,
                                               processor_sha256=contract.processor_sha256)
    manifest["science"]["dataset"]["task_manifest_sha256"] = contract.task_manifest_sha256
    manifest["science"]["judge"]["judge_protocol_sha256"] = contract.judge_protocol_sha256
    manifest["science"]["generation"].update(generation_sha256=contract.generation_sha256,
        likelihood_temperature=contract.likelihood_temperature,
        likelihood_convention=contract.likelihood_convention)
    if manifest_change:
        manifest_change(manifest)
    assigned = {leaf: (p.panel_id, action) for p in panels
                for action, leaves in zip(p.candidate_edges, p.draw_leaf_ids) for leaf in leaves}
    terminals = {leaf: terminal for terminal, leaf in selection.terminal_to_leaf} if selection is not None else {}
    terminals.update(terminal_metadata or {})
    from openwebrl.action_credit_slot_selection import SlotSelectionReceipt
    specs = []
    for index, leaf in enumerate(graph.leaves):
        default = ("fixture-state", leaf.edge_ids[-1] if leaf.edge_ids else "fixture-missing-action")
        state_id, action_id = assigned.get(leaf.leaf_id, default)
        slot = index
        if type(selection) is SlotSelectionReceipt and leaf.split == "train":
            state_id, raw_slot = leaf.leaf_id.rsplit(":draw", 1)
            action_id, slot = "tree_continuation", int(raw_slot)
        role = ("q_fit" if leaf.split == "train" else "q_holdout") if leaf.leaf_id in assigned else "base"
        role = (role_overrides or {}).get(leaf.leaf_id, role)
        specs.append(dict(task_id=leaf.task_id, state_id=state_id, action_id=action_id,
                          draw_slot=slot, draw_role=leaf.split, group_id=leaf.group_id,
                          tree_id=leaf.tree_id, inventory_role=role))
    with tempfile.TemporaryDirectory(prefix="action-credit-binding-", dir="/tmp") as temp:
        store = CampaignStore.create(Path(temp) / "campaign", CampaignManifest.from_dict(manifest))
        store.plan_update(contract.update_id, policy_sha256=contract.policy_sha256, draws=specs)
        mapping = {}
        for leaf, spec in zip(graph.leaves, specs):
            draw = store.reserve_draw("request-" + leaf.leaf_id, update_id=contract.update_id,
                policy_sha256=contract.policy_sha256, **spec)["draw_id"]
            mapping[leaf.leaf_id] = draw
            if leaf.status == "missing":
                store.mark_missing_draw(draw, reason="fixture missing slot")
            else:
                attempt = "attempt-" + leaf.leaf_id
                store.start_attempt(draw, attempt, reserved_cost=cost(0))
                store.finish_attempt(attempt, status=leaf.status, actual_cost=cost(0),
                    outcome=dict(artifact_sha256=digest(leaf.leaf_id), reward=leaf.reward)
                            if leaf.status == "valid" else (
                                dict(artifact_sha256=digest(leaf.leaf_id)) if leaf.leaf_id in terminals else None),
                    reason=None if leaf.status == "valid" else "fixture invalid outcome",
                    sampling_terminal_node_id=terminals.get(leaf.leaf_id))
        if seal:
            store.seal_update(contract.update_id)
        yield store, mapping


def make_binding(graph, config, panels=(), *, selection=None, role_overrides=None):
    """Numerical-test helper; all outcomes are synthetic and confer no authority."""
    with collection_fixture(graph, config, panels, selection=selection, role_overrides=role_overrides) as (store, mapping):
        return SealedCollection.from_store(store, graph.contract, mapping)


class SealedBinding(unittest.TestCase):
    def test_real_sealed_export_is_deeply_immutable_and_bound(self):
        graph = graph_fixture()
        with collection_fixture(graph, config("continuation_q"), (panel(),)) as (store, mapping):
            binding = SealedCollection.from_store(store, graph.contract, mapping)
            validate_collection_binding(graph, config("continuation_q"), (panel(),), binding)
            saved_hash = binding.sha256
            copy = binding.to_dict()
            copy["inventory"]["a"]["reward"] = 0
            self.assertEqual(binding.sha256, saved_hash)
            self.assertEqual(binding.to_dict()["export_sha256"], digest(store.training_draws(0)))
            with self.assertRaises(FrozenInstanceError):
                binding.canonical_json = "{}"
            with self.assertRaises(TypeError):
                SealedCollection("{}")

    def test_unsealed_partial_and_nonbijective_mappings_fail_closed(self):
        graph = graph_fixture()
        with collection_fixture(graph, config("continuation_q"), (panel(),), seal=False) as (store, mapping):
            with self.assertRaisesRegex(ValueError, "sealed"):
                SealedCollection.from_store(store, graph.contract, mapping)
            store.seal_update(0)
            for wrong in ({"a": mapping["a"]}, {"a": mapping["a"], "b": mapping["a"]},
                          dict(mapping, extra="unknown-draw")):
                with self.assertRaisesRegex(ValueError, "bijectively"):
                    SealedCollection.from_store(store, graph.contract, wrong)

    def test_contract_science_arm_and_outcome_relabels_rejected(self):
        graph = graph_fixture()
        binding = make_binding(graph, config("continuation_q"), (panel(),))
        for fields in ({"task_id": "test", "split": "holdout"}, {"reward": 0.},
                       {"split": "holdout"}, {"status": "invalid", "reward": None}):
            changed = replace(graph, leaves=(replace(graph.leaves[0], **fields), graph.leaves[1]))
            with self.assertRaises(ValueError):
                validate_collection_binding(changed, config("continuation_q"), (panel(),), binding)
        with self.assertRaisesRegex(ValueError, "arm"):
            validate_collection_binding(graph, config("tree_grpo"), (), binding)
        with self.assertRaises(ValueError):
            validate_collection_binding(graph, config("continuation_q"), (panel(),), binding.to_dict())
        with collection_fixture(graph, config("continuation_q"), (panel(),)) as (store, mapping):
            for field in ("policy_sha256", "generation_sha256", "tokenizer_sha256"):
                with self.assertRaises(ValueError):
                    SealedCollection.from_store(store, replace(graph.contract, **{field: digest("wrong")}), mapping)
            with self.assertRaises(ValueError):
                SealedCollection.from_store(store, replace(graph.contract, update_id=1), mapping)
            with self.assertRaises(ValueError):
                SealedCollection.from_store(store, replace(graph.contract, heldout_task_ids=()), mapping)

    def test_reduced_graph_and_candidate_or_whole_panel_drops_rejected(self):
        graph = graph_fixture()
        panels = (ContinuationPanel("s1", ("good",), (("a",),)),
                  ContinuationPanel("s2", ("bad",), (("b",),)))
        binding = make_binding(graph, config("continuation_q"), panels)
        validate_collection_binding(graph, config("continuation_q"), panels, binding)
        reduced = replace(graph, edges=graph.edges[:2], leaves=graph.leaves[:1], planned_leaf_ids=("a",))
        for changed, changed_panels in ((reduced, panels[:1]), (graph, panels[:1]), (graph, ()),
              (graph, (replace(panels[0], panel_id="renamed"), panels[1])),
              (graph, (replace(panels[0], candidate_edges=("bad",)), panels[1]))):
            with self.assertRaises(ValueError):
                validate_collection_binding(changed, config("continuation_q"), changed_panels, binding)

    def test_posthoc_advantage_regrouping_and_tree_relabels_rejected(self):
        graph = graph_fixture()
        edges = (replace(graph.edges[1], tree_id="first", parent_edge_id=None),
                 replace(graph.edges[2], tree_id="second", parent_edge_id=None))
        leaves = (replace(graph.leaves[0], tree_id="first", edge_ids=("good",)),
                  replace(graph.leaves[1], tree_id="second", edge_ids=("bad",)))
        graph = replace(graph, edges=edges, leaves=leaves)
        cfg = config("chain_outcome_grpo")
        binding = make_binding(graph, cfg)
        validate_collection_binding(graph, cfg, (), binding)
        for changed in (
            replace(graph, leaves=(leaves[0], replace(leaves[1], group_id="post-hoc-singleton"))),
            replace(graph, edges=(edges[0], replace(edges[1], tree_id="post-hoc-tree")),
                    leaves=(leaves[0], replace(leaves[1], tree_id="post-hoc-tree")))):
            changed.validate(independent_chains=True)
            with self.assertRaisesRegex(ValueError, "group, tree"):
                validate_collection_binding(changed, cfg, (), binding)

    def test_all_scientific_pins_are_required_unique_and_exact(self):
        graph = graph_fixture()
        fields = (("checkpoint", "tokenizer_sha256"), ("checkpoint", "processor_sha256"),
            ("dataset", "task_manifest_sha256"), ("judge", "judge_protocol_sha256"),
            ("generation", "generation_sha256"), ("generation", "likelihood_temperature"),
            ("generation", "likelihood_convention"))
        for section, name in fields:
            for duplicate in (False, True):
                def mutate(manifest):
                    original = manifest["science"][section][name]
                    if duplicate:
                        manifest["science"]["algorithm"][name] = original
                    else:
                        del manifest["science"][section][name]
                with collection_fixture(graph, config("continuation_q"), (panel(),),
                                        manifest_change=mutate) as (store, mapping):
                    with self.assertRaisesRegex(ValueError, "scientific pin"):
                        SealedCollection.from_store(store, graph.contract, mapping)

    def test_full_objective_and_mask_prefix_protocols_are_pinned(self):
        graph = graph_fixture()
        original = config("continuation_q")
        binding = make_binding(graph, original, (panel(),))
        for change in ({"clip_low": .1}, {"clip_high": .3}, {"dual_clip": 3.},
                       {"local_coefficient": .5}, {"reduction": "token_mean"},
                       {"normalization": "mean"}, {"singleton": "upstream"},
                       {"gigpo_anchor_mode": "observation"}, {"gigpo_gamma": .5},
                       {"gigpo_episode_weighting": "step"}):
            with self.assertRaisesRegex(ValueError, "objective configuration"):
                validate_collection_binding(graph, replace(original, **change), (panel(),), binding)
        for section in ("token_mask", "shared_prefix_normalization"):
            def mutate(manifest):
                manifest["science"][section]["contract"] = {"silently_changed": True}
                manifest["science"][section]["protocol_sha256"] = digest({"silently_changed": True})
            with collection_fixture(graph, original, (panel(),), manifest_change=mutate) as (store, mapping):
                with self.assertRaises(ValueError):
                    changed = SealedCollection.from_store(store, graph.contract, mapping)
                    validate_collection_binding(graph, original, (panel(),), changed)
        zero = replace(original, local_coefficient=0.)
        validate_collection_binding(graph, zero, (), make_binding(graph, zero))

    def test_truly_heldout_task_draws_are_kept_but_never_exported(self):
        graph = graph_fixture()
        edge = replace(graph.edges[1], edge_id="test-edge", task_id="test", tree_id="heldout-tree",
                       parent_edge_id=None)
        leaf = Leaf("test-leaf", "test", "heldout-tree", ("test-edge",), "valid", 1.,
                    split="holdout", group_id="heldout-group")
        graph = replace(graph, edges=(*graph.edges, edge), leaves=(*graph.leaves, leaf),
                        planned_leaf_ids=(*graph.planned_leaf_ids, leaf.leaf_id))
        cfg = config("tree_outcome_sampling")
        with collection_fixture(graph, cfg) as (store, mapping):
            binding = SealedCollection.from_store(store, graph.contract, mapping)
            validate_collection_binding(graph, cfg, (), binding)
            self.assertEqual(binding.to_dict()["inventory"]["test-leaf"]["task_split"], "holdout")
            self.assertTrue(all(row["task_id"] != "test" for row in store.training_draws(0)))

    def test_selection_policy_and_terminal_mapping_are_independently_sealed(self):
        from test_action_credit_selection import selection_fixture, remake
        graph, selection = selection_fixture()
        cfg = config("tree_outcome_sampling")
        binding = make_binding(graph, cfg, selection=selection)
        validate_collection_binding(graph, cfg, (), binding, selection=selection)
        with self.assertRaisesRegex(ValueError, "selection schedule"):
            validate_collection_binding(graph, cfg, (), binding)
        with self.assertRaises(ValueError):
            validate_collection_binding(graph, cfg, (), binding,
                selection=replace(selection, schedule=replace(selection.schedule, seed=43)))
        mapping = dict(selection.terminal_to_leaf)
        first, second = list(mapping)[:2]
        mapping[first], mapping[second] = mapping[second], mapping[first]
        permuted = remake(graph, selection, terminal_to_leaf=mapping)
        with self.assertRaisesRegex(ValueError, "sealed outcome metadata"):
            validate_collection_binding(graph, cfg, (), binding, selection=permuted)
        with self.assertRaises(ValueError):
            validate_collection_binding(graph, cfg, (), binding,
                selection=replace(selection, terminal_to_leaf=selection.terminal_to_leaf[:-1]))
        # Matching declared schedule alone cannot manufacture terminal provenance.
        def pin_policy(manifest):
            manifest["science"]["algorithm"]["leaf_selection"] = selection.policy
        with collection_fixture(graph, cfg, manifest_change=pin_policy) as (store, mapping):
            absent = SealedCollection.from_store(store, graph.contract, mapping)
            with self.assertRaisesRegex(ValueError, "sealed outcome metadata"):
                validate_collection_binding(graph, cfg, (), absent, selection=selection)

    def test_invalid_terminal_preserved_and_holdout_terminal_is_not_selection_eligible(self):
        from test_action_credit_selection import selection_fixture, remake
        graph, selection = selection_fixture()
        invalid = selection.generated_leaf_ids[0]
        graph = replace(graph, leaves=tuple(replace(leaf, status="invalid", reward=None)
                        if leaf.leaf_id == invalid else leaf for leaf in graph.leaves))
        selection = remake(graph, selection)
        cfg = config("tree_grpo")
        held = next(leaf.leaf_id for leaf in graph.leaves if leaf.split == "holdout")
        with collection_fixture(graph, cfg, selection=selection,
                                terminal_metadata={held: "independent-heldout-terminal"}) as (store, mapping):
            binding = SealedCollection.from_store(store, graph.contract, mapping)
            validate_collection_binding(graph, cfg, (), binding, selection=selection)
            inventory = binding.to_dict()["inventory"]
            self.assertIsNotNone(inventory[invalid]["sampling_terminal_node_id"])
            self.assertIsNone(inventory[invalid]["reward"])
            self.assertEqual(inventory[held]["sampling_terminal_node_id"], "independent-heldout-terminal")
            self.assertNotIn(held, selection.generated_leaf_ids)

    def test_missing_invalid_and_train_task_holdout_are_preserved(self):
        graph = graph_fixture()
        leaves = (*graph.leaves,
            Leaf("missing", "task", "tree", (), "missing", None, group_id="task-group"),
            Leaf("invalid", "task", "tree", (), "invalid", None, group_id="task-group"),
            replace(graph.leaves[0], leaf_id="held", split="holdout", reward=0.))
        graph = replace(graph, leaves=leaves, planned_leaf_ids=tuple(leaf.leaf_id for leaf in leaves))
        panels = (replace(panel(), draw_leaf_ids=(("a", "missing", "held"), ("b", "invalid"))),)
        with collection_fixture(graph, config("continuation_q"), panels) as (store, mapping):
            binding = SealedCollection.from_store(store, graph.contract, mapping)
            validate_collection_binding(graph, config("continuation_q"), panels, binding)
            self.assertEqual(len(binding.to_dict()["inventory"]), 5)
            self.assertEqual(len(store.training_draws(0)), 4)
            self.assertTrue(all(row["draw_role"] == "train" for row in store.training_draws(0)))
            with self.assertRaises(ValueError):
                SealedCollection.from_store(store, graph.contract, {k: v for k, v in mapping.items() if k != "held"})
            for omitted in ("missing", "invalid", "held"):
                partial = replace(panels[0], draw_leaf_ids=tuple(
                    tuple(leaf for leaf in group if leaf != omitted) for group in panels[0].draw_leaf_ids))
                with self.assertRaises(ValueError):
                    validate_collection_binding(graph, config("continuation_q"), (partial,), binding)

    def test_q_panels_cover_estimation_inventory_without_absorbing_independent_base(self):
        graph = graph_fixture()
        edge = replace(graph.edges[1], edge_id="independent-base-edge", tree_id="base-tree", parent_edge_id=None)
        base = replace(graph.leaves[0], leaf_id="base", tree_id="base-tree", edge_ids=(edge.edge_id,))
        held = replace(graph.leaves[0], leaf_id="q-heldout", split="holdout", reward=0.)
        graph = replace(graph, edges=(*graph.edges, edge), leaves=(*graph.leaves, base, held),
                        planned_leaf_ids=(*graph.planned_leaf_ids, "base", "q-heldout"))
        panels = (replace(panel(), draw_leaf_ids=(("a", "q-heldout"), ("b",))),)
        cfg = config("continuation_q")
        binding = make_binding(graph, cfg, panels)
        validate_collection_binding(graph, cfg, panels, binding)
        roles = {key: row["inventory_role"] for key, row in binding.to_dict()["inventory"].items()}
        self.assertEqual(roles, {"base": "base", "a": "q_fit", "b": "q_fit", "q-heldout": "q_holdout"})
        for altered in ((replace(panels[0], draw_leaf_ids=(("a", "q-heldout", "base"), ("b",))),),
                        (replace(panels[0], draw_leaf_ids=(("a",), ("b",))),), ()):
            with self.assertRaisesRegex(ValueError, "ALL draw slots"):
                validate_collection_binding(graph, cfg, altered, binding)
        with collection_fixture(graph, cfg, panels, role_overrides={"a": "base"}) as (store, mapping):
            changed = SealedCollection.from_store(store, graph.contract, mapping)
            with self.assertRaisesRegex(ValueError, "excluding base"):
                validate_collection_binding(graph, cfg, panels, changed)

    def test_q_inventory_science_protocol_is_mandatory(self):
        graph, cfg = graph_fixture(), config("continuation_q")
        for remove in (True, False):
            def mutate(manifest):
                algorithm = manifest["science"]["algorithm"]
                if remove:
                    algorithm.pop("q_inventory_protocol")
                else:
                    algorithm["q_inventory_protocol"]["q_continuations_in_outcome_objective"] = True
            with collection_fixture(graph, cfg, (panel(),), manifest_change=mutate) as (store, mapping):
                with self.assertRaisesRegex(ValueError, "independent-base"):
                    binding = SealedCollection.from_store(store, graph.contract, mapping)
                    validate_collection_binding(graph, cfg, (panel(),), binding)

    def test_pending_and_consumed_snapshots_do_not_authorize_execution(self):
        graph = graph_fixture()
        with collection_fixture(graph, config("tree_outcome_sampling")) as (store, mapping):
            store.begin_update(0, attempt_id="optimizer")
            pending = SealedCollection.from_store(store, graph.contract, mapping)
            self.assertEqual(pending.to_dict()["phase"], "training_pending")
            validate_collection_binding(graph, config("tree_outcome_sampling"), (), pending)
            store.complete_update(0, attempt_id="optimizer", receipt=checkpoint_receipt(store))
            consumed = SealedCollection.from_store(store, graph.contract, mapping)
            self.assertEqual(consumed.to_dict()["phase"], "consumed")
            self.assertEqual(consumed.to_dict()["inventory"], pending.to_dict()["inventory"])
            self.assertNotIn("authorizes_execution", consumed.to_dict())


if __name__ == "__main__":
    unittest.main()
