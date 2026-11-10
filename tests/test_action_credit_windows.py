"""CPU minibatch math/identity tests; native actual-model execution is separate."""
from collections import Counter
import copy
from dataclasses import replace
import math
import unittest
from unittest.mock import patch

import torch

from test_action_credit import graph_fixture
from test_action_credit_objective import config
from openwebrl.action_credit import append_local_terms, digest, leaf_loss_plan
from openwebrl.action_credit_objective import PreparedObjective
from openwebrl.action_credit_megatron import loss_from_log_probs, shard_edges
from openwebrl.action_credit_windows import (WindowConfig, PPOResumeCursor,
    build_window_objective, occurrence_id, plan_ppo_windows,native_train_collection)


def prepared_fixture(*, local=False, copies=1):
    """Explicit numerical fixture: not a sealed online collection."""
    graph = graph_fixture()
    if copies > 1:
        leaves = tuple(replace(leaf,leaf_id=f"{leaf.leaf_id}-{i}")
                       for i in range(copies) for leaf in graph.leaves)
        graph = replace(graph,leaves=leaves,planned_leaf_ids=tuple(l.leaf_id for l in leaves))
    arm = "continuation_q" if local else "tree_outcome_sampling"
    plan = leaf_loss_plan(graph,{l.leaf_id:1. if l.reward else -1. for l in graph.leaves},
                          arm_id=arm,reduction="response_mean")
    if local:
        plan = append_local_terms(plan,({"good":.5,"bad":-.5},),coefficient=1.)
    return PreparedObjective(config(arm),plan,())


def logps(edges=("prefix","good","bad")):
    return {e:torch.tensor([-2+math.log(1.5),-2+math.log(.7)],dtype=torch.float64,requires_grad=True)
            for e in edges}


class WindowPlanning(unittest.TestCase):
    def test_two_epochs_preserve_every_occurrence_and_fixed_tail(self):
        source = prepared_fixture(copies=65)  # 260 response occurrences.
        schedule = plan_ppo_windows(source,WindowConfig(2,256,17))
        expected = Counter(occurrence_id(t) for t in source.plan.terms)
        self.assertEqual(len(schedule.windows),4)
        for epoch in range(2):
            windows = [w for w in schedule.windows if w.epoch==epoch]
            self.assertEqual([w.effective_base_count for w in windows],[256,4])
            self.assertEqual([w.virtual_padding_slots for w in windows],[0,252])
            self.assertEqual(Counter(k for w in windows for k in w.occurrence_ids),expected)
        self.assertNotEqual(schedule.windows[0].occurrence_ids,schedule.windows[2].occurrence_ids)

    def test_order_invariance_and_reward_blind_shuffle(self):
        source = prepared_fixture()
        cfg = WindowConfig(2,3,123)
        expected = plan_ppo_windows(source,cfg)
        graph = replace(source.plan.graph,edges=source.plan.graph.edges[::-1],leaves=source.plan.graph.leaves[::-1],
                        planned_leaf_ids=source.plan.graph.planned_leaf_ids[::-1])
        permuted = replace(source,plan=replace(source.plan,graph=graph,terms=source.plan.terms[::-1]))
        self.assertEqual(expected,plan_ppo_windows(permuted,cfg))
        changed = replace(source,plan=replace(source.plan,terms=tuple(replace(t,advantage=-2*t.advantage)
                                                                     for t in source.plan.terms)))
        other = plan_ppo_windows(changed,cfg)
        self.assertEqual(expected.windows,other.windows)
        self.assertNotEqual(expected.sha256,other.sha256)
        with self.assertRaises(ValueError): expected.validate(changed)

    def test_tamper_membership_order_padding_epoch_and_old_scores_rejected(self):
        source = prepared_fixture()
        schedule = plan_ppo_windows(source,WindowConfig(2,3,4))
        first = schedule.windows[0]
        for changed in (replace(first,occurrence_ids=first.occurrence_ids[::-1]),
                        replace(first,occurrence_ids=first.occurrence_ids[:-1]),
                        replace(first,virtual_padding_slots=99),replace(first,epoch=1)):
            with self.subTest(changed=changed),self.assertRaises(ValueError):
                replace(schedule,windows=(changed,*schedule.windows[1:])).validate(source)
        graph = source.plan.graph
        altered = replace(graph,edges=(replace(graph.edges[0],old_log_probs=(-2.1,-2.1)),*graph.edges[1:]))
        with self.assertRaises(ValueError): schedule.validate(replace(source,plan=replace(source.plan,graph=altered)))

    def test_invalid_protocol_types_and_nonresponse_reduction_rejected(self):
        source = prepared_fixture()
        for kwargs in (dict(epochs=True),dict(epochs=0),dict(global_batch_size=0),dict(seed=-1),
                       dict(tail_policy="trim"),dict(checkpoint_policy="each_window"),
                       dict(local_policy="sample_one_panel")):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                plan_ppo_windows(source,replace(WindowConfig(2,256,0),**kwargs))
        graph = source.plan.graph
        other = replace(source,plan=leaf_loss_plan(graph,{"a":1.,"b":-1.},
                        arm_id="tree_outcome_sampling",reduction="trajectory_mean"))
        with self.assertRaises(ValueError): plan_ppo_windows(other,WindowConfig(2,256,0))

    def test_cursor_accepts_only_committed_full_update_boundaries(self):
        schedule = plan_ppo_windows(prepared_fixture(),WindowConfig(2,3,0))
        cursor = PPOResumeCursor(schedule.sha256,0,*[digest(x) for x in
            ("checkpoint","optimizer","scheduler","rng","independent-review")])
        cursor.validate(schedule)
        replace(cursor,next_window=len(schedule.windows)).validate(schedule)
        for change in (dict(next_window=1),dict(next_window=True),dict(next_window=99),
                       dict(plan_sha256=digest("different")),dict(rng_state_sha256="")):
            with self.subTest(change=change),self.assertRaises(ValueError): replace(cursor,**change).validate(schedule)


class WindowMath(unittest.TestCase):
    def test_tail_and_q_panels_have_exact_prescribed_weights(self):
        source = prepared_fixture(local=True)
        schedule = plan_ppo_windows(source,WindowConfig(2,3,0))
        original_local = {t.edge_id:t for t in source.plan.terms if t.component=="local"}
        for w in schedule.windows:
            objective = build_window_objective(source,schedule,w.ordinal)
            outcomes = [t for t in objective.plan.terms if t.component=="outcome"]
            local = [t for t in objective.plan.terms if t.component=="local"]
            self.assertEqual(len(outcomes),w.effective_base_count)
            self.assertTrue(all(t.weight==1/3 for t in outcomes))
            self.assertEqual({t.edge_id for t in local},set(original_local))
            for t in local:
                self.assertEqual(t.weight,original_local[t.edge_id].weight*w.effective_base_count/3)
            self.assertIs(objective.plan.graph,source.plan.graph)
        tail = build_window_objective(source,schedule,1)
        self.assertEqual(sum(t.weight for t in tail.plan.terms if t.component=="outcome"),1/3)
        self.assertEqual(sum(t.weight for t in tail.plan.terms if t.component=="local"),1/3)

    def test_epoch_sum_matches_scaled_full_objective_values_and_gradients(self):
        for local in (False,True):
            source = prepared_fixture(local=local)
            schedule = plan_ppo_windows(source,WindowConfig(2,3,9))
            values = logps()
            expected = source.loss(values)*(4/3)
            expected_grad = torch.autograd.grad(expected,tuple(values.values()),retain_graph=True)
            for epoch in range(2):
                actual = sum(build_window_objective(source,schedule,w.ordinal).loss(
                    {e:values[e] for e in {t.edge_id for t in build_window_objective(source,schedule,w.ordinal).plan.terms}})
                    for w in schedule.windows if w.epoch==epoch)
                torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
                actual_grad = torch.autograd.grad(actual,tuple(values.values()),retain_graph=True)
                for a,b in zip(actual_grad,expected_grad):torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12)

    def test_shared_prefix_opposite_signs_remain_separate_through_clipping(self):
        source = prepared_fixture()
        objective = build_window_objective(source,plan_ppo_windows(source,WindowConfig(2,4,0)),0)
        terms = [t for t in objective.plan.terms if t.edge_id=="prefix"]
        self.assertEqual(sorted(t.advantage for t in terms),[-1.,1.])
        values = {"prefix":torch.full((2,),-2+math.log(1.5),dtype=torch.float64,requires_grad=True)}
        actual = objective.loss(values,edge_subset=frozenset(values))
        self.assertAlmostEqual(float(actual.detach()),(.25*(1.5-1.28)),places=12)
        actual.backward()
        self.assertGreater(float(values["prefix"].grad.sum()),0.)
        self.assertEqual(objective.accounting["unique_forwards"],3)
        self.assertEqual(objective.accounting["shared_base_forward_savings"],1)

    def test_native_global_scaling_with_q_extra_edges_and_dp_zero_padding(self):
        source = prepared_fixture(local=True)
        schedule = plan_ppo_windows(source,WindowConfig(2,3,6))
        for ordinal in range(len(schedule.windows)):
            objective = build_window_objective(source,schedule,ordinal)
            edge_ids = tuple(sorted({t.edge_id for t in objective.plan.terms}))
            for dp in (1,2,4,8):
                values = logps(edge_ids)
                expected = objective.loss(values)
                expected_grad = torch.autograd.grad(expected,tuple(values.values()),retain_graph=True)
                ranks = []
                for rank in range(dp):
                    slots = shard_edges(edge_ids,dp_world_size=dp,dp_rank=rank)
                    accum = []
                    for edge in slots:
                        raw = sum(v.sum()*0 for v in values.values()) if edge is None else loss_from_log_probs(
                            objective,values[edge],edge,global_batch_size=3)
                        accum.append(raw*len(slots)/3*dp/len(slots))
                    ranks.append(torch.stack(accum).sum())
                actual = torch.stack(ranks).mean()
                torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
                observed = torch.autograd.grad(actual,tuple(values.values()))
                for a,b in zip(observed,expected_grad):torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12)

    def test_local_extra_forwards_accounted_and_edge_cache_immutable(self):
        source = prepared_fixture(local=True)
        schedule = plan_ppo_windows(source,WindowConfig(2,1,3))
        for w in schedule.windows:
            objective = build_window_objective(source,schedule,w.ordinal)
            self.assertEqual(objective.accounting["base_occurrences"],1)
            self.assertEqual(objective.accounting["local_terms"],2)
            base = next(t.edge_id for t in objective.plan.terms if t.component=="outcome")
            self.assertEqual(objective.accounting["additional_local_forwards"],2 if base=="prefix" else 1)
            with self.assertRaises(TypeError):objective.plan.validate()["injected"] = source.plan.graph.edges[0]
        with self.assertRaises(ValueError):build_window_objective(source,schedule,True)
        with self.assertRaises(ValueError):build_window_objective(source,schedule,-1)

    def test_full_collection_crash_replays_from_saved_model_adam_and_schedule(self):
        # CPU Adam demonstrates the recovery recipe; this is not a native GPU
        # checkpoint or physical artifact durability acceptance test.
        source=prepared_fixture(local=True)
        schedule=plan_ppo_windows(source,WindowConfig(2,3,31))
        parameters={key:torch.nn.Parameter(torch.tensor([-1.6,-2.3],dtype=torch.float64))
                    for key in ("prefix","good","bad")}
        optimizer=torch.optim.Adam(parameters.values(),lr=.02)
        initial_parameters={k:v.detach().clone() for k,v in parameters.items()}
        initial_optimizer=copy.deepcopy(optimizer.state_dict())
        original_old=tuple(e.old_log_probs for e in source.plan.graph.edges)
        def native_stub(actor,window,rows,tensors):
            optimizer.zero_grad()
            window.loss({e:parameters[e] for e in {t.edge_id for t in window.plan.terms}}).backward()
            optimizer.step()
            return dict(window.accounting,durable_checkpoint_committed=False)
        with patch("openwebrl.action_credit_windows.native_train_window",side_effect=native_stub):
            report=native_train_collection(None,source,schedule,{}, {})
            expected_parameters={k:v.detach().clone() for k,v in parameters.items()}
            expected_optimizer=copy.deepcopy(optimizer.state_dict())
            def restore():
                with torch.no_grad():
                    for k,v in parameters.items():v.copy_(initial_parameters[k])
                optimizer.load_state_dict(copy.deepcopy(initial_optimizer))
            restore()
            seen=[]
            def fail_after_partial(window):
                seen.append(window.plan.ordinal)
                if window.plan.ordinal==2:raise RuntimeError("injected before commit")
            with self.assertRaises(RuntimeError):
                native_train_collection(None,source,schedule,{}, {},before_window=fail_after_partial)
            self.assertEqual(seen,[0,1,2])
            restore()
            repeated=native_train_collection(None,source,schedule,{}, {})
        self.assertEqual(report,repeated)
        self.assertFalse(report["durable_checkpoint_committed"])
        self.assertEqual(report["base_occurrence_exposures"],8)
        self.assertEqual(report["optimizer_updates_in_memory"],4)
        self.assertEqual(tuple(e.old_log_probs for e in source.plan.graph.edges),original_old)
        for k,v in parameters.items():torch.testing.assert_close(v,expected_parameters[k],rtol=0,atol=0)
        actual_optimizer=optimizer.state_dict()
        self.assertEqual(actual_optimizer["param_groups"],expected_optimizer["param_groups"])
        for key,state in expected_optimizer["state"].items():
            for name,value in state.items():
                torch.testing.assert_close(actual_optimizer["state"][key][name],value,rtol=0,atol=0)


if __name__ == "__main__":
    unittest.main()
