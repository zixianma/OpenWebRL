"""Gate preparation tests; no browser, CUDA, model download or allocation."""
from dataclasses import replace
from pathlib import Path
import random
import unittest
from unittest.mock import patch
import numpy as np
import torch

from openwebrl.action_credit import ARMS, CollectionContract, digest
from openwebrl.action_credit_native_gate import (validate_fixture, mechanics_graph, mechanics_objective,
    compare_full_gradients, state_digest, writable_output, optimizer_runtime_state, first_adam_moment, repeat_window,
    window_probe_inputs,window_probe_objective,rng_component_sha256)
from openwebrl.action_credit_windows import WindowConfig,plan_ppo_windows,build_window_objective


def saved_fixture():
    return dict(schema_version=1, origin="saved_browser_rows_off_policy_mechanics_only",
        source=dict(path="/tmp/private-original.pt", sha256=digest("original"), saved_iteration=10),
        rows=[dict(tokens=torch.tensor([1, 2, 3, 4+i]), response_length=2,
                   loss_mask=torch.tensor([1, 0]),
                   multimodal_train_inputs={"pixel_values":torch.arange(12).reshape(3,4).float(),
                                            "image_grid_thw":torch.tensor([[1,2,2]])}) for i in range(2)])


def contract():
    return CollectionContract(0, *[digest(x) for x in ("policy","tokenizer","processor","tasks","judge","generation")],
                              .8, ("diagnostic-task",), ())


class GateContracts(unittest.TestCase):
    def test_real_fixture_shapes_and_processor_values(self):
        result = validate_fixture(saved_fixture())
        self.assertEqual(result[0]["input_ids"], (1,2))
        self.assertEqual(result[0]["sampled_mask"], (True,False))
        self.assertNotEqual(result[0]["response_ids"],result[1]["response_ids"])

    def test_all_five_mechanics_graphs_validate_and_exercise_local_credit(self):
        saved = validate_fixture(saved_fixture())
        plans = {}
        for arm in ARMS:
            draft, rows, tensors = mechanics_graph(contract(), saved, arm)
            with self.assertRaises(ValueError):
                draft.validate()
            graph = replace(draft, edges=tuple(replace(e,old_log_probs=(-2.,)*len(e.response_ids),
                old_log_prob_source="frozen_actor_teacher_forced") for e in draft.edges))
            graph.validate(independent_chains=arm in ("gigpo","chain_outcome_grpo"))
            prepared = mechanics_objective(graph,arm)
            prepared.plan.validate()
            plans[arm] = prepared
        local = [t for t in plans["continuation_q"].plan.terms if t.component=="local"]
        self.assertEqual({t.advantage for t in local},{.5,-.5})
        baseline = plans["chain_outcome_grpo"].plan.terms[0].advantage
        self.assertAlmostEqual(plans["gigpo"].plan.terms[0].advantage,2*baseline)

    def test_fixture_rejects_no_media_bad_masks_identical_responses(self):
        for change in (dict(multimodal_train_inputs={}), dict(loss_mask=[0,0]),
                       dict(response_length=0), dict(loss_mask=[2,1]),dict(tokens=[1,-1,3,4])):
            fixture = saved_fixture()
            fixture["rows"][0].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_fixture(fixture)
        fixture=saved_fixture();fixture["rows"][1]=fixture["rows"][0]
        with self.assertRaises(ValueError):validate_fixture(fixture)

    def test_full256_occurrences_preserve_global_weights_and_all_rows(self):
        draft, rows, tensors = mechanics_graph(contract(),validate_fixture(saved_fixture()),"chain_outcome_grpo")
        graph=replace(draft,edges=tuple(replace(e,old_log_prob_source="frozen_actor_teacher_forced") for e in draft.edges))
        expanded, repeated_rows, repeated_tensors=repeat_window(graph,rows,tensors)
        prepared=mechanics_objective(expanded,"chain_outcome_grpo")
        self.assertEqual(len(prepared.plan.terms),256)
        self.assertEqual(len(expanded.edges),256)
        self.assertEqual({t.weight for t in prepared.plan.terms},{1/256})
        self.assertEqual(sum(l.reward for l in expanded.leaves),128)
        self.assertEqual(len(repeated_rows),256)
        self.assertEqual(len({id(v) for v in repeated_tensors.values()}),2)

    def test_full_gradient_comparison_detects_unsampled_tail_error(self):
        parameter=torch.nn.Parameter(torch.zeros(10000))
        parameter.grad=torch.arange(10000,dtype=torch.float32)/10000
        expected=[parameter.grad.clone()]
        good=compare_full_gradients([parameter],expected,atol=1e-8,rtol=1e-8)
        self.assertEqual(good["outside_tolerance"],0)
        self.assertEqual(good["elements"],10000)
        parameter.grad[-1] +=1
        bad=compare_full_gradients([parameter],expected,atol=1e-8,rtol=1e-8)
        self.assertEqual(bad["outside_tolerance"],1)
        parameter.grad[99]=float("nan")
        self.assertEqual(compare_full_gradients([parameter],expected,atol=1e-8,rtol=1e-8)["outside_tolerance"],2)

    def test_native_window_fixture_separates_base_and_extra_local_forwards(self):
        draft,rows,tensors=mechanics_graph(contract(),validate_fixture(saved_fixture()),"continuation_q")
        draft,rows,tensors=window_probe_inputs(draft,rows,tensors)
        graph=replace(draft,edges=tuple(replace(e,old_log_probs=(-2.,)*len(e.response_ids),
            old_log_prob_source="frozen_actor_teacher_forced") for e in draft.edges))
        prepared=window_probe_objective(graph)
        self.assertEqual(len([t for t in prepared.plan.terms if t.component=="outcome"]),260)
        self.assertEqual({t.edge_id for t in prepared.plan.terms if t.component=="outcome"},{"prefix","good","bad"})
        self.assertEqual({t.edge_id for t in prepared.plan.terms if t.component=="local"},{"q-good","q-bad"})
        self.assertEqual(len(graph.leaves),132)
        schedule=plan_ppo_windows(prepared,WindowConfig(2,256,17))
        self.assertEqual([w.effective_base_count for w in schedule.windows],[256,4,256,4])
        for w in schedule.windows:
            objective=build_window_objective(prepared,schedule,w.ordinal)
            self.assertEqual(objective.accounting["additional_local_forwards"],2)
            self.assertEqual(len([t for t in objective.plan.terms if t.component=="local"]),2)
            self.assertAlmostEqual(sum(t.weight for t in objective.plan.terms if t.component=="local"),w.effective_base_count/256)

    def test_state_hash_includes_scalar_tensors_and_actual_optimizer_moments(self):
        parameter=torch.nn.Parameter(torch.tensor([1.,2.]))
        optimizer=torch.optim.Adam([parameter],lr=.01)
        parameter.grad=torch.tensor([.1,.2]);optimizer.step()
        class Wrapper:
            def __init__(self):self.optimizer=optimizer
            def get_parameters(self):return [parameter]
            def state_dict(self):return {"metadata_only":True}
        wrapped=Wrapper()
        original=state_digest(optimizer_runtime_state(wrapped))
        first_adam_moment(wrapped)[0]+=1
        self.assertNotEqual(original,state_digest(optimizer_runtime_state(wrapped)))
        self.assertNotEqual(state_digest(torch.tensor(1.)),state_digest(torch.tensor(2.)))

    def test_output_refuses_home_and_unapproved_roots(self):
        for path in ("/gpfs/home/zixianma/validation", "/gpfs/projects/krishna/other"):
            with self.assertRaises(ValueError):writable_output(path)
        self.assertEqual(str(writable_output("/tmp/action-credit-test")),"/tmp/action-credit-test")
        with patch.object(Path,"resolve",return_value=Path("/tmp/home-alias")) as resolve:
            with self.assertRaises(ValueError):writable_output("/gpfs/home/zixianma/link-to-scrubbed/output")
            resolve.assert_not_called()
        with patch.object(Path,"resolve",return_value=Path("/gpfs/home/zixianma/output")):
            with self.assertRaises(ValueError):writable_output("/tmp/link-to-home/output")

    def test_rng_hashes_bind_python_numpy_torch_and_supplied_gpu_components(self):
        original=(random.getstate(),np.random.get_state(),torch.get_rng_state())
        # CUDA/tracker inputs here are explicit CPU stand-ins; their real GPU
        # state transitions are required separately by the native gate.
        cuda=torch.tensor([1,2,3],dtype=torch.uint8)
        tracker={"model-parallel-rng":torch.tensor([4,5,6],dtype=torch.uint8)}
        try:
            before=rng_component_sha256(cuda,tracker)
            random.random();np.random.random();torch.rand(1)
            changed=rng_component_sha256(cuda,tracker)
            self.assertTrue(all(before[k]!=changed[k] for k in ("python","numpy","torch_cpu")))
            self.assertEqual(before["cuda"],changed["cuda"])
            self.assertEqual(before["megatron_tracker"],changed["megatron_tracker"])
            self.assertNotEqual(before["cuda"],rng_component_sha256(cuda+1,tracker)["cuda"])
            self.assertNotEqual(before["megatron_tracker"],rng_component_sha256(cuda,{"other":tracker["model-parallel-rng"]})["megatron_tracker"])
            random.setstate(original[0]);np.random.set_state(original[1]);torch.set_rng_state(original[2])
            self.assertEqual(before,rng_component_sha256(cuda,tracker))
        finally:
            random.setstate(original[0]);np.random.set_state(original[1]);torch.set_rng_state(original[2])


if __name__ == "__main__":
    unittest.main()
