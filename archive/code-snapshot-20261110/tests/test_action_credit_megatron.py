"""CPU contract tests, not certification of native GPU execution."""
from dataclasses import replace
from types import SimpleNamespace
import math
import unittest
from unittest.mock import patch
import torch

from test_action_credit import graph_fixture
from test_action_credit_objective import build_objective, config
from openwebrl.action_credit import digest
from openwebrl.action_credit_megatron import (BackendRow, FrozenScoreReceipt, EdgeIterator,
    tensor_payload_sha256, finalize_frozen_scores, shard_edges, loss_from_log_probs, check_topology)


def fixture():
    graph = graph_fixture()
    payloads = {e.edge_id: {"pixel_values": torch.arange(12, dtype=torch.float32).reshape(3, 4),
                            "image_grid_thw": torch.tensor([[1, 2, 2]])} for e in graph.edges}
    rows = {e.edge_id: BackendRow(e.edge_id, e.input_ids, e.response_ids, e.sampled_mask,
        e.media_sha256, graph.contract.processor_sha256, tensor_payload_sha256(payloads[e.edge_id])) for e in graph.edges}
    return graph, rows, payloads


def receipts(graph, rows):
    return {e.edge_id: FrozenScoreReceipt(e.edge_id, graph.contract.sha256, graph.contract.policy_sha256,
        e.context_sha256, digest({"response_ids": e.response_ids, "sampled_mask": e.sampled_mask}),
        rows[e.edge_id].tensor_payload_sha256, digest("scorer"), graph.contract.likelihood_temperature,
        (-1., -3.)) for e in graph.edges}


def args():
    return SimpleNamespace(micro_batch_size=1, qkv_format="thd", calculate_per_token_loss=False,
        entropy_coef=0., use_kl_loss=False, loss_type="custom_loss", global_batch_size=256,
        custom_loss_function_path="openwebrl.action_credit_megatron.loss")


class TensorAndScoring(unittest.TestCase):
    def test_tensor_digest_binds_values_shapes_types_and_keys(self):
        original = {"pixels": torch.tensor([1., 2.], dtype=torch.bfloat16), "shape": torch.tensor([1, 2])}
        expected = tensor_payload_sha256(original)
        self.assertEqual(expected, tensor_payload_sha256(dict(reversed(tuple(original.items())))))
        for changed in ({**original, "pixels": original["pixels"].float()},
                        {**original, "pixels": original["pixels"].reshape(1, 2)},
                        {**original, "pixels": original["pixels"] + 1}, {"other": original["pixels"]}):
            self.assertNotEqual(expected, tensor_payload_sha256(changed))
        with self.assertRaises(ValueError):
            tensor_payload_sha256({"pixels": torch.tensor([float("nan")])})

    def test_pending_score_draft_cannot_train_but_finalizes_before_seal(self):
        graph, rows, _ = fixture()
        draft = replace(graph, edges=tuple(replace(e, old_log_prob_source="pending_frozen_actor_score") for e in graph.edges))
        with self.assertRaises(ValueError):
            draft.validate()
        result = finalize_frozen_scores(draft, receipts(draft, rows), rows=rows, scorer_source_sha256=digest("scorer"))
        result.validate()
        self.assertTrue(all(e.old_log_probs == (-1., -3.) for e in result.edges))
        self.assertTrue(all(e.old_log_prob_source == "frozen_actor_teacher_forced" for e in result.edges))

    def test_scorer_receipt_mismatch_rejected(self):
        graph, rows, _ = fixture()
        for changed in (dict(policy_sha256=digest("stale")), dict(context_sha256=digest("changed")),
                        dict(response_sha256=digest("action only")), dict(likelihood_temperature=1.),
                        dict(tensor_payload_sha256=digest("other image")), dict(source="sglang"),
                        dict(log_probs=(float("nan"), -2.)), dict(log_probs=(-2.,)), dict(log_probs=(1., -2.))):
            records = receipts(graph, rows)
            records["prefix"] = replace(records["prefix"], **changed)
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                finalize_frozen_scores(graph, records, rows=rows, scorer_source_sha256=digest("scorer"))
        records = receipts(graph, rows)
        del records["bad"]
        with self.assertRaises(ValueError):
            finalize_frozen_scores(graph, records, rows=rows, scorer_source_sha256=digest("scorer"))

    def test_backend_row_pins_full_response_images_and_mask(self):
        graph, rows, tensors = fixture()
        edge = graph.edges[0]
        for change in (dict(input_ids=(1, 2)), dict(response_ids=(3, 4)), dict(sampled_mask=(False, True)),
                       dict(media_sha256=(digest("other"),)), dict(processor_sha256=digest("other"))):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(rows[edge.edge_id], **change).validate(edge, graph.contract, tensors[edge.edge_id])
        with self.assertRaises(ValueError):
            rows[edge.edge_id].validate(edge, graph.contract, {})


class NativeTransport(unittest.TestCase):
    def test_sharding_complete_order_invariant_and_equal_collectives(self):
        for dp in (1, 2, 4, 8):
            shards = [shard_edges(("z", "a", "m"), dp_world_size=dp, dp_rank=rank) for rank in range(dp)]
            self.assertEqual(sorted(x for s in shards for x in s if x is not None), ["a", "m", "z"])
            self.assertEqual(len({len(s) for s in shards}), 1)
            self.assertEqual(shards[0], shard_edges(("m", "a", "z"), dp_world_size=dp, dp_rank=0))
        for values in ((), ("a", "a")):
            with self.assertRaises(ValueError):
                shard_edges(values, dp_world_size=2, dp_rank=0)

    def test_iterator_preserves_metadata_across_native_requested_keys(self):
        graph, rows, payloads = fixture()
        iterator = EdgeIterator(graph, rows, payloads, ("good", None), device="cpu")
        first = iterator.get_next(("tokens", "loss_masks", "missing_optional"))
        self.assertEqual(first["tokens"][0].tolist(), [0, 1, 2, 3])
        self.assertEqual(first["loss_masks"][0].tolist(), [1, 1])
        self.assertEqual(first["action_credit_edge_id"], "good")
        self.assertIsNone(first["missing_optional"])
        self.assertIsNone(iterator.get_next(("tokens",))["action_credit_edge_id"])
        with self.assertRaises(ValueError):
            iterator.get_next(("tokens",))
        self.assertEqual(iterator.reset().get_next(())["action_credit_edge_id"], "good")

    def test_iterator_rejects_tampered_tensor(self):
        graph, rows, payloads = fixture()
        payloads["good"]["pixel_values"][0, 0] = 99
        with self.assertRaises(ValueError):
            EdgeIterator(graph, rows, payloads, ("good",), device="cpu")

    def test_iterator_hashes_only_unique_local_inputs_and_padding(self):
        graph, rows, payloads = fixture()
        with patch("openwebrl.action_credit_megatron.tensor_payload_sha256", wraps=tensor_payload_sha256) as hashed:
            EdgeIterator(graph, rows, payloads, ("good", "good", None), device="cpu")
        self.assertEqual(hashed.call_count, 2)
        self.assertEqual({id(c.args[0]) for c in hashed.call_args_list},
                         {id(payloads["good"]), id(payloads[min(payloads)])})
        # Nonlocal tensor bytes are not read on this rank. Their owning rank
        # rejects a changed payload before it forwards that edge.
        payloads["prefix"]["pixel_values"][0, 0] = 99
        EdgeIterator(graph, rows, payloads, ("good",), device="cpu")
        with self.assertRaises(ValueError):
            EdgeIterator(graph, rows, payloads, ("prefix",), device="cpu")
        payloads[min(payloads)]["pixel_values"][0, 0] = 99
        with self.assertRaises(ValueError):
            EdgeIterator(graph, rows, payloads, (None,), device="cpu")

    def test_nonlocal_metadata_and_global_coverage_still_fail_closed(self):
        graph, rows, payloads = fixture()
        for changed_rows, changed_payloads in (
                ({**rows, "foreign": rows["prefix"]}, payloads),
                (rows, {k: v for k, v in payloads.items() if k != "prefix"}),
                ({**rows, "prefix": replace(rows["prefix"], sampled_mask=(False, True))}, payloads)):
            with self.assertRaises(ValueError):
                EdgeIterator(graph, changed_rows, changed_payloads, ("good",), device="cpu")

    def test_topology_and_loss_modifiers_fail_closed(self):
        check_topology(args())
        for change in (dict(micro_batch_size=2), dict(qkv_format="bshd"), dict(entropy_coef=.01),
                       dict(calculate_per_token_loss=True), dict(use_kl_loss=True), dict(enable_mtp_training=True),
                       dict(use_dynamic_batch_size=True), dict(loss_type="policy_loss"), dict(recompute_loss_function=True)):
            options = args()
            options.__dict__.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                check_topology(options)
        for cp, pp in ((2, 1), (1, 2)):
            with self.assertRaises(ValueError):
                check_topology(args(), cp=cp, pp=pp)

    def test_native_scaling_matches_loss_and_gradients_with_shared_prefix(self):
        prepared = build_objective(graph_fixture(), config("tree_grpo"))
        for dp in (1, 2, 4, 8):
            for gbs in (7, 256):
                logps = {e: torch.tensor([-2.+math.log(1.5), -2.+math.log(.7)], dtype=torch.float64, requires_grad=True)
                         for e in ("prefix", "good", "bad")}
                expected = prepared.loss(logps)
                expected_grads = torch.autograd.grad(expected, tuple(logps.values()), retain_graph=True)
                ranks = []
                for rank in range(dp):
                    slots = shard_edges(tuple(logps), dp_world_size=dp, dp_rank=rank)
                    accum = []
                    for edge in slots:
                        raw = sum(p.sum()*0 for p in logps.values()) if edge is None else loss_from_log_probs(
                            prepared, logps[edge], edge, global_batch_size=gbs)
                        # Actual Slime wrapper, Megatron accumulation, DDP mean.
                        accum.append(raw * len(slots) / gbs * dp / len(slots))
                    ranks.append(torch.stack(accum).sum())
                actual = torch.stack(ranks).mean()
                self.assertAlmostEqual(float(actual), float(expected), places=12)
                grads = torch.autograd.grad(actual, tuple(logps.values()))
                for observed, reference in zip(grads, expected_grads):
                    torch.testing.assert_close(observed, reference, atol=1e-12, rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
