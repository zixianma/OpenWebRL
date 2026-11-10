"""Opt-in full-response Megatron boundary; never imported by a live controller.

Supported mechanics: TP vocabulary shards, DP gradient averaging, CP1/PP1,
static microbatch1/thd. A unique edge is forwarded once; its leaf occurrences
remain distinct until AFTER clipping. Global LossPlan weights are retained.
Slime multiplies custom loss by n_microbatches/GBS*DP, Megatron divides by
n_microbatches, and DDP averages. Returning GBS*edge_loss cancels these factors.
This scaling contract MUST pass the actual-model acceptance gate for the pinned
backend; CPU tests do not certify a different Megatron revision/topology.

Receipts bind scorer outputs and processor tensors, not physical browser truth
or cryptographically authenticated policy weights. The collector/launcher owns
independent policy, replay, artifact and allocation verification.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
from typing import Mapping

from openwebrl.action_credit import ResponseGraph, _sha, digest


def tensor_payload_sha256(tensors: Mapping[str, object]) -> str:
    """Hash exact CPU/GPU tensor bytes, including dtype/shape/key boundaries."""
    import torch
    if not isinstance(tensors, Mapping) or any(not isinstance(k, str) or not k for k in tensors):
        raise ValueError("named processor tensor mapping required")
    h = hashlib.sha256()
    for key in sorted(tensors):
        value = tensors[key]
        if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
            raise ValueError("dense processor tensors required")
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError("nonfinite processor tensor")
        header = digest((key, str(value.dtype), tuple(value.shape))).encode()
        h.update(header)
        raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
        h.update(memoryview(raw).cast("B"))
    return h.hexdigest()


@dataclass(frozen=True)
class BackendRow:
    edge_id: str
    input_ids: tuple[int, ...]
    response_ids: tuple[int, ...]
    sampled_mask: tuple[bool, ...]
    media_sha256: tuple[str, ...]
    processor_sha256: str
    tensor_payload_sha256: str

    def validate_structure(self, edge, contract, tensors):
        """Check metadata without touching possibly file-backed tensor storage."""
        if (self.edge_id != edge.edge_id or self.input_ids != edge.input_ids
                or self.response_ids != edge.response_ids or self.sampled_mask != edge.sampled_mask
                or self.media_sha256 != edge.media_sha256 or self.processor_sha256 != contract.processor_sha256):
            raise ValueError("backend row differs from exact actor context/response/protocol")
        if not 0 < len(self.input_ids) < len(self.input_ids) + len(self.response_ids) <= 32768:
            raise ValueError("empty or over-limit actor sequence")
        _sha(self.tensor_payload_sha256, "tensor_payload_sha256")
        if not isinstance(tensors, Mapping) or bool(self.media_sha256) != bool(tensors):
            raise ValueError("source media and processor tensors must both be present or both absent")

    def validate(self, edge, contract, tensors):
        self.validate_structure(edge, contract, tensors)
        if tensor_payload_sha256(tensors) != self.tensor_payload_sha256:
            raise ValueError("processor tensor payload changed")


def validate_backend_structure(graph, rows, tensor_payloads):
    """Exact global coverage and metadata; byte validation happens on forwarding.

    This is not a tensor integrity certificate. Immutable private batch files
    must first pass their full file digest. Each DP worker then hashes every
    unique tensor payload it forwards, including a zero-loss padding fallback.
    No content-hash cache is used across windows or optimizer updates.
    """
    edges = structural_graph(graph).validate()
    if set(rows) != set(edges) or set(tensor_payloads) != set(edges):
        raise ValueError("exact backend row/tensor coverage required")
    for key, edge in edges.items():
        if not isinstance(rows[key], BackendRow):
            raise ValueError("typed backend row required")
        rows[key].validate_structure(edge, graph.contract, tensor_payloads[key])
    return edges


@dataclass(frozen=True)
class FrozenScoreReceipt:
    edge_id: str
    contract_sha256: str
    policy_sha256: str
    context_sha256: str
    response_sha256: str
    tensor_payload_sha256: str
    scorer_source_sha256: str
    likelihood_temperature: float
    log_probs: tuple[float, ...]
    source: str = "frozen_actor_teacher_forced"

    @property
    def sha256(self):
        return digest(self.__dict__)


def structural_graph(graph):
    """Permit an explicitly unscored draft only during frozen-policy scoring."""
    if any(e.old_log_prob_source not in ("pending_frozen_actor_score", "frozen_actor_teacher_forced") for e in graph.edges):
        raise ValueError("unrecognized old-policy score source")
    temporary = replace(graph, edges=tuple(replace(e, old_log_prob_source="frozen_actor_teacher_forced") for e in graph.edges))
    temporary.validate()
    return temporary


def finalize_frozen_scores(graph, receipts, *, rows, scorer_source_sha256):
    """Finalize BEFORE sealing collection inventory, never after PPO begins."""
    edges = structural_graph(graph).validate()
    _sha(scorer_source_sha256, "scorer_source_sha256")
    if set(receipts) != set(edges) or set(rows) != set(edges):
        raise ValueError("score receipts and rows must exactly cover graph edges")
    updated = []
    for edge in graph.edges:
        receipt, row = receipts[edge.edge_id], rows[edge.edge_id]
        if not isinstance(receipt, FrozenScoreReceipt) or not isinstance(row, BackendRow):
            raise ValueError("typed scorer receipts/rows required")
        if (row.edge_id != edge.edge_id or row.input_ids != edge.input_ids or row.response_ids != edge.response_ids
                or row.sampled_mask != edge.sampled_mask or row.media_sha256 != edge.media_sha256
                or row.processor_sha256 != graph.contract.processor_sha256):
            raise ValueError("scorer row changed its context, response or processor")
        expected = (edge.edge_id, graph.contract.sha256, graph.contract.policy_sha256,
            edge.context_sha256, digest({"response_ids": edge.response_ids, "sampled_mask": edge.sampled_mask}),
            row.tensor_payload_sha256, scorer_source_sha256, graph.contract.likelihood_temperature)
        actual = (receipt.edge_id, receipt.contract_sha256, receipt.policy_sha256, receipt.context_sha256,
            receipt.response_sha256, receipt.tensor_payload_sha256, receipt.scorer_source_sha256,
            receipt.likelihood_temperature)
        if actual != expected or receipt.source != "frozen_actor_teacher_forced":
            raise ValueError("stale or mismatched frozen-policy receipt")
        if (not isinstance(receipt.log_probs, tuple) or len(receipt.log_probs) != len(edge.response_ids)
                or any(isinstance(p, bool) or not math.isfinite(p) or p > 1e-7 for p in receipt.log_probs)):
            raise ValueError("invalid full-response frozen scores")
        updated.append(replace(edge, old_log_probs=receipt.log_probs, old_log_prob_source=receipt.source))
    result = replace(graph, edges=tuple(updated))
    result.validate()
    return result


def shard_edges(edge_ids, *, dp_world_size, dp_rank):
    """Each unique edge once globally; None pads DP ranks to equal work count."""
    if (type(dp_world_size) is not int or dp_world_size < 1 or type(dp_rank) is not int
            or not 0 <= dp_rank < dp_world_size or not edge_ids or len(set(edge_ids)) != len(edge_ids)):
        raise ValueError("nonempty unique edges and valid DP topology required")
    ordered = tuple(sorted(edge_ids))
    local = ordered[dp_rank::dp_world_size]
    count = (len(ordered) + dp_world_size - 1) // dp_world_size
    return local + (None,) * (count - len(local))


def check_topology(args, *, cp=1, pp=1):
    if cp != 1 or pp != 1 or args.micro_batch_size != 1 or args.qkv_format != "thd":
        raise ValueError("action credit requires CP1/PP1, microbatch1/thd")
    if (args.calculate_per_token_loss or args.entropy_coef != 0 or args.use_kl_loss
            or getattr(args, "use_critic", False) or getattr(args, "enable_mtp_training", False)
            or getattr(args, "use_dynamic_batch_size", False) or getattr(args, "allgather_cp", False)
            or getattr(args, "recompute_loss_function", False)):
        raise ValueError("unsupported loss modifier, dynamic packing, CP or loss recomputation")
    if args.loss_type != "custom_loss" or args.custom_loss_function_path != "openwebrl.action_credit_megatron.loss":
        raise ValueError("native action-credit custom loss must be selected explicitly")
    if type(args.global_batch_size) is not int or args.global_batch_size <= 0:
        raise ValueError("positive native scheduler/global batch size required")


class EdgeIterator:
    """Duck-typed native DataIterator carrying immutable objective metadata."""
    def __init__(self, graph, rows, tensor_payloads, slots, *, device, prepared=None, global_batch_size=1):
        import torch
        edges = validate_backend_structure(graph, rows, tensor_payloads)
        if not slots or any(x is not None and x not in edges for x in slots):
            raise ValueError("invalid edge microbatch slots")
        if prepared is not None and prepared.plan.graph != graph:
            raise ValueError("objective graph differs from native row graph")
        self.graph, self.prepared, self.slots, self.offset = graph, prepared, tuple(slots), 0
        self.rollout_data = {k: [] for k in ("tokens", "total_lengths", "response_lengths", "loss_masks", "multimodal_train_inputs")}
        self.rollout_data["effective_global_batch_size"] = global_batch_size
        fallback = min(edges)
        # Scan only the actual local forwarding set, never the entire archived
        # media corpus on every window/rank. Padding still forwards real inputs.
        for key in {slot if slot is not None else fallback for slot in slots}:
            rows[key].validate(edges[key], graph.contract, tensor_payloads[key])
        for slot in slots:
            edge = edges[slot if slot is not None else fallback]
            values = dict(tokens=torch.tensor(edge.input_ids + edge.response_ids, dtype=torch.long, device=device),
                total_lengths=len(edge.input_ids) + len(edge.response_ids), response_lengths=len(edge.response_ids),
                loss_masks=torch.tensor(edge.sampled_mask, dtype=torch.int, device=device),
                multimodal_train_inputs=tensor_payloads[edge.edge_id] or None)
            for key, value in values.items():
                self.rollout_data[key].append(value)

    def get_next(self, keys):
        if self.offset >= len(self.slots):
            raise ValueError("edge iterator exhausted")
        batch = {key: [self.rollout_data[key][self.offset]] if key in self.rollout_data else None for key in keys}
        batch["action_credit_edge_id"] = self.slots[self.offset]
        batch["action_credit_objective"] = self.prepared
        if getattr(self, "oracle", None) is not None:
            batch["action_credit_oracle"] = self.oracle
        self.offset += 1
        return batch

    def reset(self):
        self.offset = 0
        return self


def loss_from_log_probs(prepared, logp, edge_id, *, global_batch_size):
    """Global contribution with Slime wrapper correction; no local renormalization."""
    if type(global_batch_size) is not int or global_batch_size <= 0:
        raise ValueError("invalid native global batch size")
    return prepared.loss({edge_id: logp}, edge_subset=frozenset((edge_id,))) * global_batch_size


def loss(args, batch, logits, reducer):
    """Slime --custom-loss-function-path entry; reducer deliberately unused."""
    from megatron.core import mpu
    from slime.backends.megatron_utils.loss import get_log_probs_and_entropy
    check_topology(args, cp=mpu.get_context_parallel_world_size(), pp=mpu.get_pipeline_model_parallel_world_size())
    prepared, edge_id = batch["action_credit_objective"], batch["action_credit_edge_id"]
    if prepared is None:
        raise ValueError("missing prepared objective in native iterator")
    if args.rollout_temperature != prepared.plan.graph.contract.likelihood_temperature:
        raise ValueError("native likelihood temperature differs from frozen-policy protocol")
    if edge_id is None:
        value = logits.sum() * 0.
    else:
        edge = next(e for e in prepared.plan.graph.edges if e.edge_id == edge_id)
        actual = tuple(batch["unconcat_tokens"][0].tolist())
        if (len(batch["unconcat_tokens"]) != 1 or actual != edge.input_ids + edge.response_ids
                or batch["response_lengths"] != [len(edge.response_ids)]
                or tuple(bool(x) for x in batch["loss_masks"][0].tolist()) != edge.sampled_mask):
            raise ValueError("packed native response IDs/mask changed")
        _, scored = get_log_probs_and_entropy(logits, args=args, unconcat_tokens=batch["unconcat_tokens"],
            total_lengths=batch["total_lengths"], response_lengths=batch["response_lengths"], with_entropy=False,
            max_seq_lens=batch.get("max_seq_lens"))
        if len(scored["log_probs"]) != 1:
            raise ValueError("native scorer changed response count")
        logp = scored["log_probs"][0]
        if batch.get("action_credit_oracle") is not None:
            batch["action_credit_oracle"](args, batch, logits, logp)
        value = loss_from_log_probs(prepared, logp, edge_id,
            global_batch_size=batch.get("effective_global_batch_size", args.global_batch_size))
    return value, {"action_credit_scaled_loss": value.detach()}


def native_frozen_scores(actor, graph, rows, tensor_payloads, *, scorer_source_sha256):
    """All native ranks participate, before the first optimizer update."""
    import torch
    import torch.distributed as dist
    from megatron.core import mpu
    graph.contract.validate()
    check_topology(actor.args, cp=mpu.get_context_parallel_world_size(), pp=mpu.get_pipeline_model_parallel_world_size())
    if actor.args.rollout_temperature != graph.contract.likelihood_temperature or len(actor.model) != 1:
        raise ValueError("native likelihood/model topology mismatch")
    dp, rank = mpu.get_data_parallel_world_size(False), mpu.get_data_parallel_rank(False)
    slots = shard_edges(tuple(e.edge_id for e in graph.edges), dp_world_size=dp, dp_rank=rank)
    iterator = EdgeIterator(graph, rows, tensor_payloads, slots, device=torch.cuda.current_device())
    with torch.no_grad():
        scores = actor.compute_log_prob([iterator], [len(slots)])["log_probs"]
    if len(scores) != len(slots):
        raise ValueError("native frozen scorer omitted a row")
    local = {key: tuple(float(x) for x in value.detach().cpu().tolist()) for key, value in zip(slots, scores) if key is not None}
    gathered = [None] * dp
    dist.all_gather_object(gathered, local, group=mpu.get_data_parallel_group_gloo(with_context_parallel=False))
    merged = {}
    for shard in gathered:
        if set(merged) & set(shard):
            raise ValueError("duplicated DP frozen scores")
        merged.update(shard)
    result = {}
    for edge in graph.edges:
        result[edge.edge_id] = FrozenScoreReceipt(edge.edge_id, graph.contract.sha256, graph.contract.policy_sha256,
            edge.context_sha256, digest({"response_ids": edge.response_ids, "sampled_mask": edge.sampled_mask}),
            rows[edge.edge_id].tensor_payload_sha256, scorer_source_sha256, graph.contract.likelihood_temperature,
            merged[edge.edge_id])
    finalize_frozen_scores(graph, result, rows=rows, scorer_source_sha256=scorer_source_sha256)
    return result


def native_train_iterator(actor, prepared, rows, tensor_payloads):
    import torch
    from megatron.core import mpu
    check_topology(actor.args, cp=mpu.get_context_parallel_world_size(), pp=mpu.get_pipeline_model_parallel_world_size())
    prepared.plan.validate()
    slots = shard_edges(tuple({t.edge_id for t in prepared.plan.terms}),
        dp_world_size=mpu.get_data_parallel_world_size(False), dp_rank=mpu.get_data_parallel_rank(False))
    return EdgeIterator(prepared.plan.graph, rows, tensor_payloads, slots, device=torch.cuda.current_device(),
        prepared=prepared, global_batch_size=actor.args.global_batch_size)


def configure_native_training(actor):
    """Same gradient synchronization/scale setup as native model.train()."""
    from megatron.core.distributed import finalize_model_grads
    from megatron.core.utils import get_model_config
    from megatron.core import mpu
    args = actor.args
    check_topology(args, cp=mpu.get_context_parallel_world_size(), pp=mpu.get_pipeline_model_parallel_world_size())
    if len(actor.model) != 1 or getattr(args, "overlap_param_gather", False):
        raise ValueError("action-credit native driver does not yet support overlapped parameter gathering")
    for module in actor.model:
        module.train()
    config = get_model_config(actor.model[0])
    config.grad_scale_func = actor.optimizer.scale_loss
    config.timers = None
    config.no_sync_func = actor.model[0].no_sync if args.overlap_grad_reduce else None
    config.grad_sync_func = actor.model[0].start_grad_sync if args.overlap_grad_reduce and args.align_grad_reduce else None
    config.param_sync_func = None
    config.finalize_model_grads_func = finalize_model_grads
