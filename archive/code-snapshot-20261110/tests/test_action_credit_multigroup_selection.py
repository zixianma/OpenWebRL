"""Offline multi-group inventory proofs, including a real-ledger objective boundary."""
from dataclasses import replace
import unittest

from openwebrl.action_credit import CollectionContract, Leaf, ResponseEdge, ResponseGraph, digest
from openwebrl.action_credit_receipts import ReplayReceipt
from openwebrl.action_credit_sampling import SamplingNode, RoundSeal, RequestCompletion, forest_sha256, plan_round
from openwebrl.action_credit_selection import (
    TreeSchedulePin, TreeUpdateSelectionPolicy, TreeSelectionPolicy,
    make_training_selection, make_multigroup_training_selection,
    selection_partition_graph, selection_partition_graphs, validate_selection,
)
from test_action_credit_selection import graph_receipts


def multigroup_fixture(*, update_id=0, shared_node_ids=False, group_count=2):
    policy = TreeSelectionPolicy(digest("frozen-ten-update-task-order"), tuple(
        TreeUpdateSelectionPolicy(u, tuple(TreeSchedulePin(f"task-{i}", f"group-{i}", 1, 2, 1, 2, 42 + u)
                                          for i in range(group_count))) for u in range(10)))
    c = CollectionContract(update_id, *[digest(s) for s in
        (f"policy-{update_id}", "tokenizer", "processor", "tasks", "judge", "generation")],
        .8, tuple(f"task-{i}" for i in range(group_count)), ("holdout-task",))
    edges, leaves, proofs = [], [], []
    for i, pin in enumerate(policy.updates[update_id].schedules):
        schedule = pin.bind(c)
        tree = schedule.tree_ids[0]
        def node(suffix, parent, terminal, round_index):
            identity = suffix if shared_node_ids else f"{i}-{suffix}"
            receipt = ReplayReceipt(identity, c.sha256, digest("state:" + identity),
                digest("context:" + identity), 3 if parent is None else 2,
                digest("synthetic-verifier"), digest("snapshot:" + identity), True)
            return SamplingNode(identity, pin.task_id, tree, parent, c.policy_sha256,
                receipt.context_sha256, receipt.state_sha256, receipt.evidence_sha256,
                receipt.remaining_turns, terminal, round_index, receipt.sha256), receipt
        root, root_receipt = node("root", None, False, 0)
        terminal, terminal_receipt = node("initial", root.node_id, True, 0)
        nodes = (root, terminal)
        receipts = {root.node_id: root_receipt, terminal.node_id: terminal_receipt}
        requests = plan_round(schedule, nodes, receipts, round_index=1)
        completions = []
        for j, request in enumerate(requests):
            terminal, receipt = node(f"expanded-{j}", root.node_id, True, 1)
            nodes += (terminal,)
            receipts[terminal.node_id] = receipt
            completions.append(RequestCompletion(request.request_id, "valid", terminal.node_id))
        seal = RoundSeal(schedule.sha256, 1, forest_sha256(nodes, before_round=1), tuple(completions))
        mapping = {}
        for j, terminal in enumerate(n for n in nodes if n.terminal):
            edge_id, leaf_id = f"edge-{i}-{j}", f"leaf-{i}-{j}"
            edges.append(ResponseEdge(edge_id, pin.task_id, tree, None, c.sha256, (0, 1), (),
                (2, 3), (True, True), (-2., -2.), c.policy_sha256, digest("edge-state"), digest("placeholder"), 3))
            leaves.append(Leaf(leaf_id, pin.task_id, tree, (edge_id,), "valid", float(j != 0), group_id=pin.group_id))
            mapping[terminal.node_id] = leaf_id
        proofs.append(dict(schedule=schedule, nodes=nodes, replay_receipts=receipts,
                           seals=(seal,), terminal_to_leaf=mapping))
    # A separate held-out task has no training schedule and must remain globally bound.
    edges.append(ResponseEdge("heldout-edge", "holdout-task", "heldout-tree", None, c.sha256,
        (0, 1), (), (2, 3), (True, True), (-2., -2.), c.policy_sha256,
        digest("heldout-state"), digest("placeholder"), 3))
    leaves.append(Leaf("heldout-leaf", "holdout-task", "heldout-tree", ("heldout-edge",), "valid", 1.,
                       split="holdout", group_id="heldout-group"))
    graph = ResponseGraph(c, tuple(edges), tuple(leaves), tuple(l.leaf_id for l in leaves))
    receipts = graph_receipts(graph)
    graph = replace(graph, edges=tuple(replace(e, replay_evidence_sha256=receipts[e.edge_id].sha256)
                                       for e in graph.edges))
    subgraphs = selection_partition_graphs(graph)
    partitions = tuple(make_training_selection(subgraphs[p["schedule"].task_id, p["schedule"].group_id], **p)
                       for p in proofs)
    return graph, policy, partitions


def compose(graph, policy, partitions):
    return make_multigroup_training_selection(graph, policy=policy, partitions=partitions)


def remake_partitions(graph, partitions):
    return tuple(make_training_selection(selection_partition_graph(graph,
        task_id=s.schedule.task_id, group_id=s.schedule.group_id), schedule=s.schedule, nodes=s.nodes,
        replay_receipts={r.edge_id: r for r in s.replay_receipts}, seals=s.seals,
        terminal_to_leaf=dict(s.terminal_to_leaf)) for s in partitions)


class MultiGroupSelectionTests(unittest.TestCase):
    def test_forty_eight_group_update_is_composed_without_inventory_pruning(self):
        graph, policy, parts = multigroup_fixture(group_count=48)
        bundle = compose(graph, policy, parts)
        self.assertEqual(len(bundle.partitions), 48)
        self.assertEqual(len(bundle.generated_leaf_ids), 48 * 3)
        self.assertEqual(len(validate_selection(graph, bundle)), 48 * 2)
        self.assertEqual(len(graph.leaves), 48 * 3 + 1)
        subgraphs = selection_partition_graphs(graph)
        self.assertEqual(len(subgraphs), 48)
        self.assertEqual(subgraphs["task-0", "group-0"],
            selection_partition_graph(graph, task_id="task-0", group_id="group-0"))

    def test_exact_group_union_retains_full_inventory_and_outside_holdout(self):
        graph, policy, partitions = multigroup_fixture()
        bundle = compose(graph, policy, partitions)
        self.assertEqual(validate_selection(graph, bundle), bundle.selected_leaf_ids)
        self.assertEqual(len(bundle.generated_leaf_ids), 6)
        self.assertEqual(len(bundle.selected_leaf_ids), 4)
        self.assertEqual(bundle.outside_partition_holdout_leaf_ids, ("heldout-leaf",))
        self.assertEqual(bundle.terminal_to_leaf, tuple(sorted(p for s in partitions for p in s.terminal_to_leaf)))
        self.assertNotIn("heldout-leaf", bundle.selected_leaf_ids)
        self.assertEqual(bundle.contract_sha256, graph.contract.sha256)
        self.assertEqual(bundle.policy_sha256, digest(bundle.policy))

    def test_campaign_policy_is_stable_across_all_ten_evolving_actor_updates(self):
        first_graph, first_policy, first_parts = multigroup_fixture()
        first = compose(first_graph, first_policy, first_parts)
        for update in range(1, 10):
            graph, policy, parts = multigroup_fixture(update_id=update)
            bundle = compose(graph, policy, parts)
            self.assertEqual(bundle.policy_sha256, first.policy_sha256)
            self.assertNotEqual(bundle.contract_sha256, first.contract_sha256)
            self.assertEqual(validate_selection(graph, bundle), bundle.selected_leaf_ids)
            with self.assertRaises(ValueError):
                validate_selection(graph, first)

    def test_policy_is_immutable_complete_and_has_unique_group_task_identities(self):
        graph, policy, parts = multigroup_fixture()
        first = policy.updates[0]
        for bad in (replace(policy, updates=policy.updates[1:]),
                    replace(policy, updates=(first, first)),
                    replace(policy, updates=list(policy.updates)),
                    replace(policy, task_order_sha256="bad"),
                    replace(policy, updates=(replace(first, schedules=(first.schedules[0],) * 2),))):
            with self.subTest(policy=bad), self.assertRaises(ValueError):
                compose(graph, bad, parts)
        with self.assertRaises(AttributeError):
            policy.updates = ()

    def test_repeated_absent_extra_and_changed_current_schedules_fail_closed(self):
        graph, policy, parts = multigroup_fixture()
        changed = replace(policy, updates=(replace(policy.updates[0], schedules=tuple(
            replace(s, seed=s.seed + 1) for s in policy.updates[0].schedules)),) + policy.updates[1:])
        extra = replace(policy, updates=(replace(policy.updates[0], schedules=policy.updates[0].schedules +
            (TreeSchedulePin("other-task", "other-group", 1, 2, 1, 2, 42),)),) + policy.updates[1:])
        for p, receipts in ((policy, parts[:1]), (policy, parts + parts[:1]), (changed, parts), (extra, parts)):
            with self.assertRaises(ValueError):
                compose(graph, p, receipts)
        only_one = selection_partition_graph(graph, task_id="task-0", group_id="group-0")
        with self.assertRaises(ValueError):
            compose(only_one, policy, parts)

    def test_global_node_identity_overlap_is_rejected(self):
        graph, policy, parts = multigroup_fixture(shared_node_ids=True)
        with self.assertRaisesRegex(ValueError, "overlapping"):
            compose(graph, policy, parts)

    def test_invalid_missing_and_within_group_holdout_are_bound_without_being_dropped(self):
        graph, policy, parts = multigroup_fixture()
        invalid_id = parts[0].selected_leaf_ids[0]
        missing = Leaf("missing", "task-0", parts[0].schedule.tree_ids[0], (), "missing", None, group_id="group-0")
        same_group_holdout = replace(graph.leaves[0], leaf_id="within-group-holdout", split="holdout")
        graph = replace(graph, leaves=tuple(replace(l, status="invalid", reward=None)
            if l.leaf_id == invalid_id else l for l in graph.leaves) + (missing, same_group_holdout),
            planned_leaf_ids=graph.planned_leaf_ids + (missing.leaf_id, same_group_holdout.leaf_id))
        bundle = compose(graph, policy, remake_partitions(graph, parts))
        self.assertIn(invalid_id, validate_selection(graph, bundle))
        self.assertNotIn(missing.leaf_id, bundle.generated_leaf_ids)
        self.assertNotIn(same_group_holdout.leaf_id, bundle.selected_leaf_ids)
        for removed in (invalid_id, missing.leaf_id, same_group_holdout.leaf_id, "heldout-leaf"):
            leaves = tuple(l for l in graph.leaves if l.leaf_id != removed)
            edge_ids = {e for l in leaves for e in l.edge_ids}
            smaller = replace(graph, leaves=leaves, edges=tuple(e for e in graph.edges if e.edge_id in edge_ids),
                              planned_leaf_ids=tuple(l.leaf_id for l in leaves))
            with self.subTest(removed=removed), self.assertRaises(ValueError):
                validate_selection(smaller, bundle)

    def test_all_invalid_group_is_still_required_and_unknown_tree_disposition_rejected(self):
        graph, policy, parts = multigroup_fixture()
        graph = replace(graph, leaves=tuple(replace(l, status="invalid", reward=None)
            if l.group_id == "group-1" else l for l in graph.leaves))
        parts = remake_partitions(graph, parts)
        bundle = compose(graph, policy, parts)
        self.assertEqual(len(validate_selection(graph, bundle)), 4)
        with self.assertRaises(ValueError):
            compose(graph, policy, parts[:1])
        missing = Leaf("bad-tree-missing", "task-1", "unknown-tree", (), "missing", None, group_id="group-1")
        changed = replace(graph, leaves=graph.leaves + (missing,), planned_leaf_ids=graph.planned_leaf_ids + (missing.leaf_id,))
        with self.assertRaisesRegex(ValueError, "unplanned tree"):
            compose(changed, policy, remake_partitions(changed, parts))

    def test_forged_union_contract_partition_and_holdout_fields_are_rejected(self):
        graph, policy, parts = multigroup_fixture()
        bundle = compose(graph, policy, parts)
        for bad in (replace(bundle, update_id=1), replace(bundle, contract_sha256=digest("other")),
                    replace(bundle, selected_leaf_ids=bundle.generated_leaf_ids),
                    replace(bundle, generated_leaf_ids=bundle.selected_leaf_ids),
                    replace(bundle, terminal_to_leaf=bundle.terminal_to_leaf[:-1]),
                    replace(bundle, outside_partition_holdout_leaf_ids=()),
                    replace(bundle, partitions=(replace(parts[0], seals=()), parts[1])),
                    replace(bundle, partitions=list(parts))):
            with self.subTest(bundle=bad), self.assertRaises(ValueError):
                validate_selection(graph, bad)

    def test_canonical_graph_and_partition_order(self):
        graph, policy, parts = multigroup_fixture()
        bundle = compose(graph, policy, parts)
        permuted = replace(graph, edges=tuple(reversed(graph.edges)), leaves=tuple(reversed(graph.leaves)),
                           planned_leaf_ids=tuple(reversed(graph.planned_leaf_ids)))
        self.assertEqual(compose(permuted, policy, tuple(reversed(parts))), bundle)
        self.assertEqual(validate_selection(permuted, bundle), bundle.selected_leaf_ids)

    def test_binding_pins_policy_task_order_and_independent_terminal_mapping(self):
        from test_action_credit_binding import collection_fixture, config
        from openwebrl.action_credit_binding import SealedCollection, validate_collection_binding
        graph, policy, parts = multigroup_fixture()
        bundle = compose(graph, policy, parts)
        cfg = config("tree_outcome_sampling")
        def manifest_pin(manifest):
            manifest["science"]["dataset"]["task_order_sha256"] = policy.task_order_sha256
        with collection_fixture(graph, cfg, selection=bundle, manifest_change=manifest_pin) as (store, mapping):
            binding = SealedCollection.from_store(store, graph.contract, mapping)
            validate_collection_binding(graph, cfg, (), binding, selection=bundle)
            # A changed future schedule must fail even though the current proof is unchanged.
            future = policy.updates[-1]
            wrong = replace(policy, updates=policy.updates[:-1] + (replace(future,
                schedules=tuple(replace(s, seed=s.seed + 1) for s in future.schedules)),))
            with self.assertRaisesRegex(ValueError, "prespecified campaign"):
                validate_collection_binding(graph, cfg, (), binding, selection=replace(bundle, frozen_policy=wrong))
        with collection_fixture(graph, cfg, selection=bundle) as (store, mapping):
            with self.assertRaisesRegex(ValueError, "task order"):
                validate_collection_binding(graph, cfg, (), SealedCollection.from_store(store, graph.contract, mapping),
                                            selection=bundle)
        # A same-group permutation would pass membership checks but fails sealed terminal metadata.
        first = parts[0]
        old_map = dict(first.terminal_to_leaf)
        keys = list(old_map)
        old_map[keys[0]], old_map[keys[1]] = old_map[keys[1]], old_map[keys[0]]
        bad_part = make_training_selection(selection_partition_graph(graph, task_id="task-0", group_id="group-0"),
            schedule=first.schedule, nodes=first.nodes, replay_receipts={r.edge_id: r for r in first.replay_receipts},
            seals=first.seals, terminal_to_leaf=old_map)
        bad_bundle = compose(graph, policy, (bad_part, parts[1]))
        with collection_fixture(graph, cfg, selection=bundle, manifest_change=manifest_pin) as (store, mapping):
            with self.assertRaisesRegex(ValueError, "independently sealed"):
                validate_collection_binding(graph, cfg, (), SealedCollection.from_store(store, graph.contract, mapping),
                                            selection=bad_bundle)

    def test_integrated_objective_uses_one_global_denominator_not_equal_group_means(self):
        from test_action_credit_binding import collection_fixture, config
        from openwebrl.action_credit_binding import SealedCollection
        from openwebrl.action_credit_objective import build_objective
        graph, policy, parts = multigroup_fixture()
        invalid = parts[1].selected_leaf_ids[0]
        graph = replace(graph, leaves=tuple(replace(l, status="invalid", reward=None)
            if l.leaf_id == invalid else l for l in graph.leaves))
        bundle = compose(graph, policy, remake_partitions(graph, parts))
        for arm in ("tree_outcome_sampling", "tree_grpo"):
            cfg = config(arm)
            with collection_fixture(graph, cfg, selection=bundle, manifest_change=lambda m:
                    m["science"]["dataset"].update(task_order_sha256=policy.task_order_sha256)) as (store, mapping):
                result = build_objective(graph, cfg, selection=bundle,
                    collection_binding=SealedCollection.from_store(store, graph.contract, mapping),
                    replay_receipts=graph_receipts(graph))
            self.assertIs(result.plan.graph, graph)
            self.assertEqual(len(result.plan.terms), 3)
            self.assertEqual({t.source_id for t in result.plan.terms}, set(bundle.selected_leaf_ids) - {invalid})
            self.assertTrue(all(abs(t.weight - 1/3) < 1e-12 for t in result.plan.terms))


if __name__ == "__main__":
    unittest.main()
