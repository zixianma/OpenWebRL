"""Validated response graph and clipped policy-loss reference for action credit.

This opt-in module is not connected to the live training controller. A graph
stores each sampled response once, but a loss plan retains separate leaf terms:
opposite-sign advantages must never be averaged before PPO clipping. Explicit
response-mean and trajectory-mean reductions are different estimands. The
former matches OpenWebRL's current turn-sample reduction. Neither is silently
changed to unique-edge averaging.

Local continuation targets supervise the complete sampled thought/action
response. They are not evidence of the effect of a tool call stripped of its
reasoning. Input IDs and image digests identify the actor context; a separate
replay evidence digest is required for environment-state comparisons.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Mapping, Sequence


ARMS = frozenset({"chain_outcome_grpo", "tree_outcome_sampling", "tree_grpo",
                  "continuation_q", "gigpo"})


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _sha(value: str, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase SHA256")


def _id(value: str, name: str) -> None:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a nonempty canonical identity")


@dataclass(frozen=True)
class CollectionContract:
    update_id: int
    policy_sha256: str
    tokenizer_sha256: str
    processor_sha256: str
    task_manifest_sha256: str
    judge_protocol_sha256: str
    generation_sha256: str
    likelihood_temperature: float
    train_task_ids: tuple[str, ...]
    heldout_task_ids: tuple[str, ...]
    # Standard tempered-softmax PPO surrogate, with no top-p/top-k correction.
    # The generation hash separately pins sampling. This is not asserted to be
    # the truncated sampler's exact probability distribution.
    likelihood_convention: str = "tempered_full_softmax"

    def validate(self):
        if type(self.update_id) is not int or self.update_id < 0:
            raise ValueError("update_id must be a nonnegative integer")
        for key in ("policy_sha256", "tokenizer_sha256", "processor_sha256",
                    "task_manifest_sha256", "judge_protocol_sha256", "generation_sha256"):
            _sha(getattr(self, key), key)
        if (isinstance(self.likelihood_temperature, bool)
                or not math.isfinite(self.likelihood_temperature) or self.likelihood_temperature <= 0):
            raise ValueError("positive finite likelihood temperature required")
        if self.likelihood_convention != "tempered_full_softmax":
            raise ValueError("unvalidated likelihood convention")
        for tasks in (self.train_task_ids, self.heldout_task_ids):
            if not isinstance(tasks, tuple) or len(set(tasks)) != len(tasks):
                raise ValueError("task lists must be immutable and unique")
            for task in tasks:
                _id(task, "task_id")
        if not self.train_task_ids or set(self.train_task_ids) & set(self.heldout_task_ids):
            raise ValueError("missing training tasks or task split leakage")

    @property
    def sha256(self):
        self.validate()
        return digest(self.__dict__)


@dataclass(frozen=True)
class ResponseEdge:
    edge_id: str
    task_id: str
    tree_id: str
    parent_edge_id: str | None
    contract_sha256: str
    input_ids: tuple[int, ...]
    media_sha256: tuple[str, ...]
    response_ids: tuple[int, ...]
    # A response may contain non-sampled transport padding; those tokens never
    # contribute. Prompt, observation and image tokens belong in input_ids.
    sampled_mask: tuple[bool, ...]
    old_log_probs: tuple[float, ...]
    old_policy_sha256: str
    state_sha256: str
    replay_evidence_sha256: str
    remaining_turns: int
    old_log_prob_source: str = "frozen_actor_teacher_forced"

    @property
    def context_sha256(self):
        return digest({"input_ids": self.input_ids, "media_sha256": self.media_sha256})


@dataclass(frozen=True)
class Leaf:
    leaf_id: str
    task_id: str
    tree_id: str
    edge_ids: tuple[str, ...]
    status: str                    # valid / invalid / missing
    reward: float | None
    split: str = "train"
    group_id: str = ""


@dataclass(frozen=True)
class ResponseGraph:
    contract: CollectionContract
    edges: tuple[ResponseEdge, ...]
    leaves: tuple[Leaf, ...]
    planned_leaf_ids: tuple[str, ...]

    def validate(self, *, independent_chains=False):
        self.contract.validate()
        contract_hash = self.contract.sha256
        if not isinstance(self.edges, tuple) or not isinstance(self.leaves, tuple) or not self.leaves:
            raise ValueError("immutable nonempty collection required")
        if (not isinstance(self.planned_leaf_ids, tuple)
                or len(set(self.planned_leaf_ids)) != len(self.planned_leaf_ids)
                or set(self.planned_leaf_ids) != {leaf.leaf_id for leaf in self.leaves}):
            raise ValueError("planned leaf manifest mismatch; retain explicit missing records")
        edges = {}
        for edge in self.edges:
            for key in ("edge_id", "task_id", "tree_id"):
                _id(getattr(edge, key), key)
            if edge.edge_id in edges:
                raise ValueError("duplicate response edge")
            if edge.contract_sha256 != contract_hash or edge.old_policy_sha256 != self.contract.policy_sha256:
                raise ValueError("stale collection/behavior policy or mismatched protocol")
            if edge.old_log_prob_source != "frozen_actor_teacher_forced":
                raise ValueError("require verified frozen-policy scores, not synthetic generator logps")
            _sha(edge.state_sha256, "state_sha256")
            _sha(edge.replay_evidence_sha256, "replay_evidence_sha256")
            if type(edge.remaining_turns) is not int or edge.remaining_turns <= 0:
                raise ValueError("remaining turn horizon must be a positive integer")
            for name in ("input_ids", "media_sha256", "response_ids", "sampled_mask", "old_log_probs"):
                if not isinstance(getattr(edge, name), tuple):
                    raise ValueError("edge fields must be immutable tuples")
            if not edge.input_ids or not edge.response_ids:
                raise ValueError("empty actor context or response")
            if any(type(i) is not int or i < 0 for i in edge.input_ids + edge.response_ids):
                raise ValueError("token IDs must be nonnegative integers")
            for media in edge.media_sha256:
                _sha(media, "media_sha256")
            if (len(edge.response_ids) != len(edge.sampled_mask)
                    or len(edge.response_ids) != len(edge.old_log_probs)
                    or any(type(m) is not bool for m in edge.sampled_mask) or not any(edge.sampled_mask)):
                raise ValueError("misaligned or empty sampled-response mask")
            if any(isinstance(p, bool) or not math.isfinite(p) or p > 1e-7 for p in edge.old_log_probs):
                raise ValueError("old log probabilities must be finite and nonpositive")
            edges[edge.edge_id] = edge
        leaf_ids, used, trees = set(), set(), {}
        for leaf in self.leaves:
            for name in ("leaf_id", "task_id", "tree_id", "group_id"):
                _id(getattr(leaf, name), name)
            if leaf.leaf_id in leaf_ids:
                raise ValueError("duplicate leaf identity")
            leaf_ids.add(leaf.leaf_id)
            allowed = (self.contract.train_task_ids if leaf.split == "train" else
                       self.contract.train_task_ids + self.contract.heldout_task_ids)
            if leaf.split not in ("train", "holdout") or leaf.task_id not in allowed:
                raise ValueError("task/split not in pinned manifest")
            if leaf.status not in ("valid", "invalid", "missing"):
                raise ValueError("unknown terminal status")
            if leaf.status == "valid":
                if isinstance(leaf.reward, bool) or leaf.reward not in (0., 1.):
                    raise ValueError("valid terminal success must be binary")
            elif leaf.reward is not None:
                raise ValueError("invalid/missing outcomes must not be imputed as failures")
            if not isinstance(leaf.edge_ids, tuple) or len(set(leaf.edge_ids)) != len(leaf.edge_ids):
                raise ValueError("mutable or cyclic path")
            if leaf.status == "valid" and not leaf.edge_ids:
                raise ValueError("valid leaf requires a sampled response")
            parent = None
            for edge_id in leaf.edge_ids:
                if edge_id not in edges:
                    raise ValueError("path references missing response")
                edge = edges[edge_id]
                if (edge.parent_edge_id != parent or edge.task_id != leaf.task_id or edge.tree_id != leaf.tree_id):
                    raise ValueError("broken path, cross-task edge, or cross-tree shared prefix")
                if parent is not None and edge.remaining_turns != edges[parent].remaining_turns - 1:
                    raise ValueError("child resets or changes the one-decision-per-turn horizon")
                if independent_chains and edge_id in used:
                    raise ValueError("independent chains cannot share sampled responses")
                used.add(edge_id)
                parent = edge_id
            root = edges[leaf.edge_ids[0]] if leaf.edge_ids else None
            tree_key = ((leaf.task_id, leaf.group_id, root.context_sha256, root.state_sha256, root.remaining_turns)
                        if root is not None else None)
            if root is not None and leaf.tree_id in trees and trees[leaf.tree_id] != tree_key:
                raise ValueError("tree identity maps to different roots/tasks/splits")
            if root is not None:
                trees[leaf.tree_id] = tree_key
        if set(edges) != used:
            raise ValueError("unreferenced response edge")
        return edges


@dataclass(frozen=True)
class LossTerm:
    edge_id: str
    advantage: float
    weight: float
    source_id: str
    component: str = "outcome"


@dataclass(frozen=True)
class LossPlan:
    graph: ResponseGraph
    arm_id: str
    reduction: str
    terms: tuple[LossTerm, ...]
    # Ordinary outcome population, separate from fit/holdout role. Tree
    # selection and Q's base-only population both retain the complete graph.
    # The objective adapter verifies their immutable collection bindings.
    selected_leaf_ids: tuple[str, ...] | None = None

    def validate(self):
        if self.arm_id not in ARMS or self.reduction not in ("response_mean", "trajectory_mean", "token_mean"):
            raise ValueError("unknown arm or reduction")
        edges = self.graph.validate(independent_chains=self.arm_id in ("chain_outcome_grpo", "gigpo"))
        if not isinstance(self.terms, tuple) or not self.terms:
            raise ValueError("no trainable terms; skip this optimizer update explicitly")
        selected = self.selected_leaf_ids
        fit_ids = {leaf.leaf_id for leaf in self.graph.leaves if leaf.split == "train"}
        if selected is not None and (not isinstance(selected, tuple) or not selected
                or len(set(selected)) != len(selected) or not set(selected) <= fit_ids):
            raise ValueError("selected leaves must be unique retained training slots")
        leaves = {leaf.leaf_id: leaf for leaf in self.graph.leaves if leaf.split == "train"
                  and leaf.status == "valid" and (selected is None or leaf.leaf_id in selected)}
        eligible = {e for leaf in leaves.values() for e in leaf.edge_ids}
        local_eligible = {e for leaf in self.graph.leaves
                          if leaf.split == "train" and leaf.status == "valid"
                          for e in leaf.edge_ids}
        expected_pairs = {(leaf.leaf_id, e) for leaf in leaves.values() for e in leaf.edge_ids}
        actual_pairs, local_groups = [], {}
        token_count = sum(sum(edges[e].sampled_mask) for leaf in leaves.values() for e in leaf.edge_ids)
        for term in self.terms:
            if not isinstance(term, LossTerm):
                raise ValueError("typed immutable loss terms required")
            if term.edge_id not in (local_eligible if term.component == "local" else eligible):
                raise ValueError("loss references an invalid or held-out-only response")
            if (isinstance(term.advantage, bool) or not math.isfinite(term.advantage)
                    or isinstance(term.weight, bool) or not math.isfinite(term.weight) or term.weight <= 0):
                raise ValueError("nonfinite advantage or nonpositive weight")
            _id(term.source_id, "source_id")
            if term.component not in ("outcome", "local"):
                raise ValueError("unknown loss component")
            if term.component == "local" and self.arm_id != "continuation_q":
                raise ValueError("auxiliary terms only belong to continuation_q")
            if term.component == "outcome":
                pair = (term.source_id, term.edge_id)
                if pair not in expected_pairs:
                    raise ValueError("outcome term must belong to its exact valid training leaf")
                actual_pairs.append(pair)
                expected_weight = (1 / len(expected_pairs) if self.reduction == "response_mean" else
                    sum(edges[term.edge_id].sampled_mask) / token_count if self.reduction == "token_mean" else
                    1 / (len(leaves) * len(leaves[term.source_id].edge_ids)))
                if not math.isclose(term.weight, expected_weight, rel_tol=1e-12, abs_tol=1e-15):
                    raise ValueError("outcome term changes the declared global reduction")
            else:
                local_groups.setdefault(term.source_id, []).append(term)
        if len(actual_pairs) != len(expected_pairs) or set(actual_pairs) != expected_pairs:
            raise ValueError("loss must include each eligible leaf/response exactly once")
        seen_local, masses = set(), []
        for group in local_groups.values():
            if len(group) < 2 or not math.isclose(sum(t.advantage for t in group), 0., abs_tol=1e-10):
                raise ValueError("local state must retain all centered candidate terms")
            identities = set()
            for term in group:
                edge = edges[term.edge_id]
                identities.add((edge.task_id, edge.tree_id, edge.parent_edge_id, edge.context_sha256,
                                edge.state_sha256, edge.remaining_turns))
                if term.edge_id in seen_local or not math.isclose(term.weight, group[0].weight, rel_tol=1e-12):
                    raise ValueError("candidate duplicated or weighted by draw count")
                seen_local.add(term.edge_id)
            if len(identities) != 1:
                raise ValueError("local state contains mismatched candidates")
            masses.append(sum(t.weight for t in group))
        if masses and any(not math.isclose(m, masses[0], rel_tol=1e-12) for m in masses):
            raise ValueError("local states must have equal aggregate weight")
        return edges


def leaf_loss_plan(graph: ResponseGraph, advantages: Mapping[str, float], *, arm_id: str,
                   reduction: str, selected_leaf_ids: tuple[str, ...] | None = None) -> LossPlan:
    """Keep each leaf advantage separate through clipping, including prefixes.

    response_mean averages all valid training leaf/response occurrences.
    trajectory_mean averages each trajectory's response means, then leaves.
    Forward outputs can be shared by edge; their separate PPO terms cannot.
    """
    graph.validate(independent_chains=arm_id in ("chain_outcome_grpo", "gigpo"))
    leaves = [x for x in graph.leaves if x.split == "train" and x.status == "valid"
              and (selected_leaf_ids is None or x.leaf_id in selected_leaf_ids)]
    if set(advantages) != {x.leaf_id for x in leaves}:
        raise ValueError("advantages must cover exactly the valid training leaves")
    edges = {e.edge_id: e for e in graph.edges}
    count = sum(len(x.edge_ids) for x in leaves)
    token_count = sum(sum(edges[e].sampled_mask) for x in leaves for e in x.edge_ids)
    terms = []
    for leaf in leaves:
        for edge in leaf.edge_ids:
            weight = (1 / count if reduction == "response_mean" else
                      sum(edges[edge].sampled_mask) / token_count if reduction == "token_mean" else
                      1 / (len(leaves) * len(leaf.edge_ids)))
            terms.append(LossTerm(edge, advantages[leaf.leaf_id], weight, leaf.leaf_id))
    plan = LossPlan(graph, arm_id, reduction, tuple(terms), selected_leaf_ids)
    plan.validate()
    return plan


def append_local_terms(base: LossPlan, panels: Sequence[Mapping[str, float]], *, coefficient: float) -> LossPlan:
    """Add one equally weighted candidate contribution per eligible state.

    Panels map candidate edge IDs to centered continuation-Q advantages. A
    candidate contributes once, regardless of its valid continuation count.
    Zero-advantage candidates/states remain in the denominator. A panel with
    fewer than two valid candidates has no identifiable comparison and must be
    excluded before calling, with coverage recorded by the numerical core.
    """
    edges = base.validate()
    if base.arm_id != "continuation_q":
        raise ValueError("local targets require the continuation_q arm")
    if isinstance(coefficient, bool) or not math.isfinite(coefficient) or coefficient < 0:
        raise ValueError("nonnegative finite local coefficient required")
    if coefficient == 0:
        return base
    if not panels:
        return base
    terms, seen = list(base.terms), set()
    for i, panel in enumerate(panels):
        if len(panel) < 2 or any(e not in edges for e in panel):
            raise ValueError("local panel needs two existing candidate responses")
        siblings = [edges[e] for e in panel]
        keys = {(e.task_id, e.tree_id, e.parent_edge_id, e.context_sha256, e.state_sha256, e.remaining_turns) for e in siblings}
        if len(keys) != 1 or any(e in seen for e in panel):
            raise ValueError("local candidates must be distinct same-state/context siblings")
        if not all(math.isfinite(a) for a in panel.values()) or not math.isclose(sum(panel.values()), 0., abs_tol=1e-10):
            raise ValueError("local candidate advantages must be finite and centered")
        seen.update(panel)
        terms.extend(LossTerm(e, a, coefficient / (len(panels) * len(panel)), f"panel:{i}", "local")
                     for e, a in panel.items())
    plan = LossPlan(base.graph, base.arm_id, base.reduction, tuple(terms), base.selected_leaf_ids)
    plan.validate()
    return plan


def clipped_loss(plan: LossPlan, log_probs: Mapping[str, object], *, clip_low: float,
                 clip_high: float, dual_clip: float | None = None,
                 edge_subset: frozenset[str] | None = None):
    """Differentiable reference; rank shards retain GLOBAL term denominators.

    A DDP implementation that averages gradients must multiply each rank's
    contribution by world_size before backward. This routine does not guess
    the framework's gradient reduction convention or Megatron scaling.
    """
    import torch
    edges = plan.validate()
    if not (0 <= clip_low < 1 and math.isfinite(clip_high) and clip_high >= 0):
        raise ValueError("invalid asymmetric PPO clipping widths")
    if dual_clip is not None and (not math.isfinite(dual_clip) or dual_clip <= 1):
        raise ValueError("dual clipping requires a finite coefficient > 1")
    expected = {x.edge_id for x in plan.terms}
    if edge_subset is not None:
        if not isinstance(edge_subset, frozenset) or not edge_subset <= expected:
            raise ValueError("invalid rank/microbatch edge subset")
        expected = set(edge_subset)
    if set(log_probs) != expected or not expected:
        raise ValueError("exact nonempty trainable edge log probabilities required")
    token_losses = {}
    for edge_id, logp in log_probs.items():
        edge = edges[edge_id]
        if (not isinstance(logp, torch.Tensor) or logp.ndim != 1
                or len(logp) != len(edge.response_ids) or not logp.is_floating_point()
                or not bool(torch.isfinite(logp).all()) or bool((logp > 1e-6).any())):
            raise ValueError("response log probabilities must be aligned, finite and nonpositive")
        # Cast reduced precision before exponentiation; do not clamp log ratios
        # and silently alter negative-advantage PPO. Overflow fails explicitly.
        work = logp if logp.dtype == torch.float64 else logp.float()
        old = torch.tensor(edge.old_log_probs, dtype=work.dtype, device=work.device)
        mask = torch.tensor(edge.sampled_mask, dtype=torch.bool, device=work.device)
        ratio = (work[mask] - old[mask]).exp()
        if not bool(torch.isfinite(ratio).all()):
            raise ValueError("nonfinite importance ratio")
        token_losses[edge_id] = (ratio, ratio.clamp(1 - clip_low, 1 + clip_high))
    terms = []
    for term in plan.terms:
        if term.edge_id in token_losses:
            ratio, clipped = token_losses[term.edge_id]
            objective = torch.minimum(ratio * term.advantage, clipped * term.advantage)
            if dual_clip is not None and term.advantage < 0:
                objective = torch.maximum(objective, torch.full_like(objective, dual_clip * term.advantage))
            terms.append(-objective.mean() * term.weight)
    loss = torch.stack(terms).sum()
    if not bool(torch.isfinite(loss)):
        raise ValueError("nonfinite policy loss")
    return loss


def response_log_probs(logits, response_ids: Sequence[int], *, temperature: float):
    """Already next-token-aligned logits only; prompts/images never enter loss.

    The caller must align logits to each response token before this function;
    the production multimodal/TP adapter has a separate acceptance gate.
    """
    import torch
    if (not isinstance(logits, torch.Tensor) or logits.ndim != 2
            or logits.shape[0] != len(response_ids) or not logits.is_floating_point()
            or not bool(torch.isfinite(logits).all()) or not math.isfinite(temperature) or temperature <= 0):
        raise ValueError("finite aligned [response_tokens, vocabulary] logits required")
    if any(type(i) is not int or i < 0 or i >= logits.shape[1] for i in response_ids):
        raise ValueError("response token outside vocabulary")
    work = logits if logits.dtype == torch.float64 else logits.float()
    ids = torch.tensor(response_ids, dtype=torch.long, device=work.device)
    return (work / temperature).log_softmax(-1).gather(-1, ids[:, None]).squeeze(-1)
