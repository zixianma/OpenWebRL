from dataclasses import replace
import unittest
import torch

from test_action_credit import graph_fixture
from openwebrl.action_credit import Leaf, ResponseGraph, digest
from openwebrl.action_credit_objective import ObjectiveConfig, ContinuationPanel, build_objective as _build_objective
from openwebrl.action_credit_receipts import ReplayReceipt
from test_action_credit_binding import make_binding


def build_objective(graph, cfg, **kwargs):
    # Synthetic fixtures only: production receipts must come from a separately
    # validated collector; a CPU test receipt is not browser acceptance evidence.
    if cfg.arm_id == "continuation_q" and kwargs.get("panels"):
        graph = with_q_base(graph)
    role_overrides = kwargs.pop("role_overrides", None)
    receipts = {e.edge_id: ReplayReceipt(e.edge_id, e.contract_sha256, e.state_sha256,
        e.context_sha256, e.remaining_turns, digest("cpu-test-verifier"), digest("cpu-test-artifact"), True)
        for e in graph.edges}
    graph = replace(graph, edges=tuple(replace(e, replay_evidence_sha256=receipts[e.edge_id].sha256) for e in graph.edges))
    binding = make_binding(graph, cfg, panels=kwargs.get("panels", ()), selection=kwargs.get("selection"),
                           **({"role_overrides": role_overrides} if role_overrides else {}))
    return _build_objective(graph, cfg, collection_binding=binding, replay_receipts=receipts, **kwargs)


def config(arm):
    return ObjectiveConfig(arm, "response_mean", "mean_std", "zero", .2, .28, None,
                           1. if arm == "continuation_q" else 0., 1. if arm == "gigpo" else 0.,
                           1., "trajectory", "state_history_horizon")


def chains():
    graph = graph_fixture()
    edges = (replace(graph.edges[1], tree_id="t1", parent_edge_id=None),
             replace(graph.edges[2], tree_id="t2", parent_edge_id=None))
    return replace(graph, edges=edges, leaves=(replace(graph.leaves[0], tree_id="t1", edge_ids=("good",)),
                                               replace(graph.leaves[1], tree_id="t2", edge_ids=("bad",))))


def with_q_base(graph):
    """Separate synthetic ordinary chains; panel draws never supply base loss."""
    base = chains()
    edges = tuple(replace(e, edge_id="base-"+e.edge_id, tree_id="base-"+e.tree_id) for e in base.edges)
    leaves = tuple(replace(l, leaf_id="base-"+l.leaf_id, tree_id="base-"+l.tree_id,
                           edge_ids=tuple("base-"+e for e in l.edge_ids)) for l in base.leaves)
    return replace(graph, edges=(*graph.edges, *edges), leaves=(*graph.leaves, *leaves),
                   planned_leaf_ids=(*graph.planned_leaf_ids, *(l.leaf_id for l in leaves)))


class FiveArms(unittest.TestCase):
    def test_all_five_build_different_expected_credit_with_one_loss_contract(self):
        tree = graph_fixture()
        panel = ContinuationPanel("s", ("good", "bad"), (("a",), ("b",)))
        plans = {arm: build_objective(chains() if arm in ("chain_outcome_grpo", "gigpo") else tree,
                    config(arm), panels=(panel,) if arm == "continuation_q" else ())
                 for arm in ("chain_outcome_grpo", "tree_outcome_sampling", "tree_grpo", "continuation_q", "gigpo")}
        for arm, prepared in plans.items():
            values = {e: torch.tensor([-1.9, -2.1], dtype=torch.float64, requires_grad=True)
                      for e in {t.edge_id for t in prepared.plan.terms}}
            loss = prepared.loss(values)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(all(torch.isfinite(x.grad).all() for x in values.values()))
        outcome_a = plans["tree_outcome_sampling"].plan.terms[0].advantage
        self.assertAlmostEqual(plans["tree_grpo"].plan.terms[0].advantage, 2 * outcome_a)
        self.assertAlmostEqual(plans["gigpo"].plan.terms[0].advantage, 2 * outcome_a)
        local = [t for t in plans["continuation_q"].plan.terms if t.component == "local"]
        self.assertEqual({t.edge_id: t.advantage for t in local}, {"good": .5, "bad": -.5})

    def test_zero_local_and_zero_gigpo_match_outcome_values_and_gradients(self):
        for arm, graph in (("continuation_q", graph_fixture()), ("gigpo", chains())):
            baseline_arm = "tree_outcome_sampling" if arm == "continuation_q" else "chain_outcome_grpo"
            baseline = build_objective(graph, config(baseline_arm))
            cfg = replace(config(arm), local_coefficient=0., gigpo_step_weight=0.)
            actual = build_objective(graph, cfg)
            self.assertEqual(actual.plan.terms, baseline.plan.terms)

    def test_gigpo_history_horizon_and_images_prevent_spurious_state_merge(self):
        graph = chains()
        for change in ({"remaining_turns": 5}, {"input_ids": (4, 5)}, {"media_sha256": (digest("other-image"),)},
                       {"state_sha256": digest("different-browser-state")}):
            changed = replace(graph, edges=(graph.edges[0], replace(graph.edges[1], **change)))
            prepared = build_objective(changed, config("gigpo"))
            for value in prepared.diagnostics[0].values:
                self.assertEqual(dict(value.components)["step"], 0.)

    def test_observation_only_gigpo_is_a_separately_named_approximation(self):
        graph = chains()
        graph = replace(graph, edges=(graph.edges[0], replace(graph.edges[1], input_ids=(4, 5))))
        cfg = replace(config("gigpo"), gigpo_anchor_mode="observation")
        result = build_objective(graph, cfg)
        self.assertTrue(all(dict(v.components)["step"] != 0 for v in result.diagnostics[0].values))

    def test_continuation_holdout_on_train_task_never_enters_q_or_loss(self):
        graph = graph_fixture()
        holdout_edge = replace(graph.edges[1], edge_id="holdout-suffix", parent_edge_id="good", remaining_turns=28)
        holdout = Leaf("h", "task", "tree", ("prefix", "good", "holdout-suffix"),
                       "valid", 0., split="holdout", group_id="task-group")
        graph = replace(graph, edges=(*graph.edges, holdout_edge), leaves=(*graph.leaves, holdout),
                        planned_leaf_ids=("a", "b", "h"))
        prepared = build_objective(graph, config("continuation_q"),
            panels=(ContinuationPanel("s", ("good", "bad"), (("a", "h"), ("b",))),))
        self.assertNotIn("holdout-suffix", {t.edge_id for t in prepared.plan.terms})
        self.assertEqual({t.edge_id: t.advantage for t in prepared.plan.terms if t.component == "local"}, {"good": .5, "bad": -.5})

    def test_root_branch_without_shared_sample_is_valid_tree(self):
        graph = chains()
        graph = replace(graph, edges=tuple(replace(e, tree_id="tree") for e in graph.edges),
                        leaves=tuple(replace(l, tree_id="tree") for l in graph.leaves))
        prepared = build_objective(graph, config("continuation_q"),
            panels=(ContinuationPanel("root", ("good", "bad"), (("a",), ("b",))),))
        self.assertEqual(len(prepared.plan.terms), 4)

    def test_incomplete_manifest_cannot_silently_drop_draws(self):
        graph = graph_fixture()
        with self.assertRaises(ValueError):
            replace(graph, planned_leaf_ids=("a", "b", "lost")).validate()

    def test_reject_candidate_path_mismatch_and_cross_panel_draw_reuse(self):
        graph = graph_fixture()
        for panels in ((ContinuationPanel("s", ("good", "bad"), (("b",), ("a",))),),
                       (ContinuationPanel("s", ("good", "bad"), (("a",), ("b",))),
                        ContinuationPanel("s2", ("good", "bad"), (("a",), ("b",))))):
            with self.assertRaises(ValueError):
                build_objective(graph, config("continuation_q"), panels=panels)

    def test_reject_selectively_omitted_candidate_continuation(self):
        graph = graph_fixture()
        graph = replace(graph, leaves=(*graph.leaves, replace(graph.leaves[0], leaf_id="a2", reward=0.)),
                        planned_leaf_ids=("a", "b", "a2"))
        with self.assertRaises(ValueError):
            build_objective(graph, config("continuation_q"),
                panels=(ContinuationPanel("s", ("good", "bad"), (("a",), ("b",))),))

    def test_empty_missing_slot_cannot_disappear_from_local_q_manifest(self):
        graph = graph_fixture()
        missing = Leaf("missing", "task", "tree", (), "missing", None, group_id="task-group")
        graph = replace(graph, leaves=(*graph.leaves, missing), planned_leaf_ids=("a", "b", "missing"))
        with self.assertRaises(ValueError):
            build_objective(graph, config("continuation_q"),
                panels=(ContinuationPanel("s", ("good", "bad"), (("a",), ("b",))),),
                role_overrides={"missing": "q_fit"})
        result = build_objective(graph, config("continuation_q"),
            panels=(ContinuationPanel("s", ("good", "bad"), (("a", "missing"), ("b",))),))
        q = result.diagnostics[-1]
        self.assertEqual(sum(v.missing_count for v in q.values), 1)

    def test_chain_control_rejects_tree_and_q_admission_abstention_keeps_base(self):
        with self.assertRaises(ValueError):
            build_objective(graph_fixture(), config("chain_outcome_grpo"))
        prepared = build_objective(chains(), config("continuation_q"))
        self.assertTrue(prepared.plan.terms)
        self.assertEqual({t.component for t in prepared.plan.terms}, {"outcome"})

    def test_q_extra_draws_do_not_change_base_normalization_or_outcome_exposure(self):
        graph = graph_fixture()
        panels = (ContinuationPanel("s", ("good", "bad"), (("a",), ("b",))),)
        original = build_objective(graph, config("continuation_q"), panels=panels)
        changed = replace(graph, leaves=tuple(replace(l, reward=1.-l.reward) for l in graph.leaves))
        other = build_objective(changed, config("continuation_q"), panels=panels)
        base_terms = tuple(t for t in original.plan.terms if t.component == "outcome")
        self.assertEqual(base_terms, tuple(t for t in other.plan.terms if t.component == "outcome"))
        self.assertEqual({t.edge_id for t in base_terms}, {"base-good", "base-bad"})
        self.assertEqual({t.source_id for t in base_terms}, {"base-a", "base-b"})
        local = {t.edge_id: t.advantage for t in original.plan.terms if t.component == "local"}
        self.assertEqual({t.edge_id: t.advantage for t in other.plan.terms if t.component == "local"},
                         {key: -value for key, value in local.items()})


if __name__ == "__main__":
    unittest.main()
