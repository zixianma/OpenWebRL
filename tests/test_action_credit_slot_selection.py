"""Slot-law tests use synthetic replay records, never physical acceptance."""
from dataclasses import replace
import unittest

from test_action_credit import graph_fixture
from test_action_credit_sampling import forest, node, schedule
from openwebrl.action_credit import Leaf, ResponseGraph, digest
from openwebrl.action_credit_sampling import RoundSeal, RequestCompletion, forest_sha256, plan_round
from openwebrl.action_credit_selection import TreeSchedulePin, TreeUpdateSelectionPolicy, TreeSelectionPolicy
from openwebrl.action_credit_slot_selection import (make_slot_selection, plan_partial_round,
    slot_selection_policy, validate_slot_selection)


def slot_fixture(*, groups=1, missing_tree=True, expanded=False):
    base = graph_fixture()
    tasks = tuple("task" if groups == 1 else "task-"+str(i) for i in range(groups))
    contract = replace(base.contract, train_task_ids=tasks)
    pins = tuple(TreeSchedulePin(task, "group-"+str(i), 2, 2, 2, 4, 42+i)
                 for i, task in enumerate(tasks))
    policy = TreeSelectionPolicy(digest("fixed-task-order"), (TreeUpdateSelectionPolicy(0, pins),))
    all_edges, all_leaves, partitions = [], [], []
    for pin in pins:
        sched = pin.bind(contract)
        nodes, receipts, mapping = (), {}, {}
        statuses = {}
        for index, tree in enumerate(sched.tree_ids):
            for r, slot in ((0, 0), (1, 0), (1, 1), (2, 0), (2, 1)):
                statuses[f"{tree}:round{r}:draw{slot}"] = "missing"
            if missing_tree and index == 1:
                continue
            root, rr = node(sched, tree+":root", tree)
            terminal, tr = node(sched, tree+":initial", tree, root.node_id, remaining=2, terminal=True)
            root, terminal = replace(root, task_id=pin.task_id), replace(terminal, task_id=pin.task_id)
            nodes += (root, terminal)
            receipts.update({root.node_id: rr, terminal.node_id: tr})
            key = f"{tree}:round0:draw0"
            statuses[key] = "valid"
            mapping[terminal.node_id] = key
        seals = ()
        for r in (1, 2):
            requests = plan_partial_round(sched, nodes, receipts, round_index=r, seals=seals)
            completions = []
            for request in requests:
                key = f"{request.tree_id}:round{r}:draw{request.draw_slot}"
                if expanded:
                    terminal, tr = node(sched, key+":actual", request.tree_id,
                        request.source_node_id, remaining=request.remaining_turns-1, terminal=True, round=r)
                    terminal = replace(terminal, task_id=pin.task_id)
                    nodes += (terminal,)
                    receipts[terminal.node_id] = tr
                    mapping[terminal.node_id] = key
                    statuses[key] = "valid"
                    completions.append(RequestCompletion(request.request_id, "valid", terminal.node_id))
                else:
                    completions.append(RequestCompletion(request.request_id, "missing"))
            seals += (RoundSeal(sched.sha256, r, forest_sha256(nodes, before_round=r), tuple(completions)),)
        for index, (key, status) in enumerate(statuses.items()):
            tree = key.split(":round", 1)[0]
            edge_id = "edge:"+key
            if status == "valid":
                all_edges.append(replace(base.edges[0], edge_id=edge_id, task_id=pin.task_id,
                    tree_id=tree, contract_sha256=contract.sha256))
            all_leaves.append(Leaf(key, pin.task_id, tree, (edge_id,) if status == "valid" else (),
                status, float(index % 2) if status == "valid" else None, group_id=pin.group_id))
        partitions.append(dict(schedule=sched, nodes=nodes, replay_receipts=receipts,
                               seals=seals, terminal_to_leaf=mapping))
    graph = ResponseGraph(contract, tuple(all_edges), tuple(all_leaves), tuple(l.leaf_id for l in all_leaves))
    receipt = make_slot_selection(graph, policy=policy, plan_sha256=digest("sealed-plan"), partitions=tuple(partitions))
    return graph, receipt, tuple(partitions)


class SlotSelection(unittest.TestCase):
    def test_source_sampling_matches_existing_law_and_missing_tree_has_no_requests(self):
        s = schedule(m=2, n=2, l=2, k=4)
        nodes, receipts = forest(s)
        full = plan_round(s, nodes, receipts, round_index=1)
        self.assertEqual(full, plan_partial_round(s, nodes, receipts, round_index=1))
        partial = tuple(n for n in nodes if n.tree_id == s.tree_ids[0])
        rs = {n.node_id: receipts[n.node_id] for n in partial}
        self.assertEqual(tuple(r for r in full if r.tree_id == s.tree_ids[0]),
                         plan_partial_round(s, partial, rs, round_index=1))
        self.assertEqual(plan_partial_round(s, (), {}, round_index=1), ())

    def test_missing_initial_and_four_failed_expansions_still_select_exact_k_slots(self):
        graph, selection, _ = slot_fixture()
        self.assertEqual(len(graph.leaves), 10)
        self.assertEqual(len(selection.generated_leaf_ids), 1)
        self.assertEqual(len(selection.selected_leaf_ids), 8)
        self.assertEqual(validate_slot_selection(graph, selection), selection.selected_leaf_ids)
        self.assertTrue(any(l.status == "missing" for l in graph.leaves if l.leaf_id in selection.selected_leaf_ids))
        missing_tree = selection.partitions[0].schedule.tree_ids[1]
        self.assertFalse(any(n.tree_id == missing_tree for n in selection.partitions[0].nodes))
        self.assertEqual(sum(key.startswith(missing_tree) for key in selection.selected_leaf_ids), 4)
        # A charged infrastructure failure remains invalid; only unavailable
        # dependent expansion slots are unattempted/missing.
        changed = replace(graph, leaves=tuple(replace(l, status="invalid")
            if l.leaf_id == f"{missing_tree}:round0:draw0" else l for l in graph.leaves))
        failed = make_slot_selection(changed, policy=selection.frozen_policy,
            plan_sha256=selection.plan_sha256, partitions=selection.partitions)
        self.assertEqual(failed.selected_leaf_ids, selection.selected_leaf_ids)

    def test_complete_and_missing_forests_have_same_outcome_blind_slot_selection(self):
        partial_graph, partial, _ = slot_fixture()
        graph, complete, _ = slot_fixture(missing_tree=False, expanded=True)
        self.assertEqual(complete.selected_leaf_ids, partial.selected_leaf_ids)
        self.assertEqual(len(complete.generated_leaf_ids), 10)
        changed = replace(graph, leaves=tuple(replace(l, reward=1-l.reward) for l in graph.leaves))
        new = make_slot_selection(changed, policy=complete.frozen_policy, plan_sha256=complete.plan_sha256,
                                   partitions=complete.partitions)
        self.assertEqual(new.selected_leaf_ids, complete.selected_leaf_ids)
        self.assertEqual(new.policy_sha256, partial.policy_sha256)
        with self.assertRaises(ValueError):
            validate_slot_selection(changed, complete)

    def test_48_group_composition_retains_all480_slots_and_selects384(self):
        graph, selection, _ = slot_fixture(groups=48)
        self.assertEqual(len(selection.partitions), 48)
        self.assertEqual(len(graph.planned_leaf_ids), 480)
        self.assertEqual(len(selection.selected_leaf_ids), 384)
        self.assertEqual(validate_slot_selection(graph, selection), selection.selected_leaf_ids)

    def test_dispositions_mapping_partitions_and_unsealed_rounds_fail_closed(self):
        graph, selection, parts = slot_fixture(missing_tree=False, expanded=True)
        for corrupted in (replace(selection, selected_leaf_ids=selection.selected_leaf_ids[:-1]),
                          replace(selection, terminal_to_leaf=selection.terminal_to_leaf[:-1]),
                          replace(selection, partitions=selection.partitions*2)):
            with self.assertRaises(ValueError):
                validate_slot_selection(graph, corrupted)
        p = parts[0]
        initial = tuple(n for n in p["nodes"] if n.created_round == 0)
        receipts = {n.node_id: p["replay_receipts"][n.node_id] for n in initial}
        with self.assertRaises(ValueError):
            plan_partial_round(p["schedule"], initial, receipts, round_index=2)
        wrong = dict(p)
        mapping = dict(wrong["terminal_to_leaf"])
        ids = [key for key in mapping if ":round1:" in key]
        mapping[ids[0]], mapping[ids[1]] = mapping[ids[1]], mapping[ids[0]]
        wrong["terminal_to_leaf"] = mapping
        with self.assertRaises(ValueError):
            make_slot_selection(graph, policy=selection.frozen_policy, plan_sha256=selection.plan_sha256,
                                partitions=(wrong,))
        with self.assertRaises(ValueError):
            make_slot_selection(replace(graph, leaves=graph.leaves[:-1], planned_leaf_ids=graph.planned_leaf_ids[:-1]),
                policy=selection.frozen_policy, plan_sha256=selection.plan_sha256, partitions=parts)

    def test_future_nodes_empty_rounds_and_stable_campaign_policy(self):
        s = schedule(m=2, n=2, l=2, k=4)
        empty_seal = RoundSeal(s.sha256, 1, forest_sha256((), before_round=1), ())
        self.assertEqual(plan_partial_round(s, (), {}, round_index=2, seals=(empty_seal,)), ())
        nodes, receipts = forest(s)
        later, proof = node(s, "future", s.tree_ids[0], "root-0", remaining=2, terminal=True, round=1)
        with self.assertRaises(ValueError):
            plan_partial_round(s, nodes+(later,), {**receipts, later.node_id: proof}, round_index=1)
        _, selection, _ = slot_fixture()
        policy = slot_selection_policy(selection.frozen_policy)
        self.assertEqual(policy["mode"], "prescheduled_tree_slot_k_v2")
        self.assertNotIn("selected_leaf_ids", policy)
        self.assertNotIn("policy_sha256", policy)

    def test_both_tree_objectives_keep_selected_missing_slots_out_of_loss(self):
        from test_action_credit_binding import collection_fixture, config
        from test_action_credit_selection import graph_receipts
        from openwebrl.action_credit_binding import SealedCollection
        from openwebrl.action_credit_objective import build_objective
        graph, selection, parts = slot_fixture(expanded=True)
        receipts = graph_receipts(graph)
        graph = replace(graph, edges=tuple(replace(e, replay_evidence_sha256=receipts[e.edge_id].sha256)
                                           for e in graph.edges))
        selection = make_slot_selection(graph, policy=selection.frozen_policy,
                                        plan_sha256=selection.plan_sha256, partitions=parts)
        def pin(manifest):
            manifest["science"]["dataset"]["task_order_sha256"] = selection.frozen_policy.task_order_sha256
        for arm in ("tree_outcome_sampling", "tree_grpo"):
            cfg = config(arm)
            with collection_fixture(graph, cfg, selection=selection, manifest_change=pin) as (store, mapping):
                bound = SealedCollection.from_store(store, graph.contract, mapping)
                exact = replace(selection, plan_sha256=bound.to_dict()["plan_sha256"])
                prepared = build_objective(graph, cfg, collection_binding=bound,
                                           replay_receipts=receipts, selection=exact)
                self.assertEqual(len(prepared.plan.graph.leaves), 10)
                self.assertEqual(len(prepared.plan.selected_leaf_ids), 8)
                self.assertEqual(len(prepared.plan.terms), 4)
                self.assertTrue(all(term.weight == .25 for term in prepared.plan.terms))
                missing = [leaf.leaf_id for leaf in graph.leaves if leaf.status == "missing"]
                swapped = dict(mapping)
                swapped[missing[0]], swapped[missing[1]] = swapped[missing[1]], swapped[missing[0]]
                bad = SealedCollection.from_store(store, graph.contract, swapped)
                with self.assertRaises(ValueError):
                    build_objective(graph, cfg, collection_binding=bad,
                                    replay_receipts=receipts, selection=exact)


if __name__ == "__main__":
    unittest.main()
