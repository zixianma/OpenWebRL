"""Build all five proposed objectives from a validated response graph.

This is the CPU-testable trainer boundary, not an online browser collector or a
Megatron adapter. Production enablement requires the separately recorded GPU,
multimodal, replay and resume gates. The default comparison must choose ONE
common loss reduction/normalization/clipping contract for all five arms. Scalar
formula parity with a paper is distinct from reproducing its complete training
recipe. In particular, upstream Tree-GRPO token-mean/dual-clip and GiGPO's step-
weighted episode statistics must not silently replace the outcome control.
The caller must obtain the complete immutable collection from CampaignStore.
A required sealed binding prevents changing its inventory or scientific recipe;
this numerical adapter does not itself own collection or optimizer state.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

from openwebrl.action_credit import (ARMS, ResponseGraph, LossPlan, LossTerm,
    append_local_terms, clipped_loss, digest, leaf_loss_plan)
from openwebrl.action_credit_advantages import (Outcome, CandidateDraw, Episode, Step,
    outcome_group_advantages, tree_grpo_advantages, candidate_q_advantages, gigpo_advantages)


@dataclass(frozen=True)
class ObjectiveConfig:
    arm_id: str
    reduction: str
    normalization: str
    singleton: str
    clip_low: float
    clip_high: float
    dual_clip: float | None
    local_coefficient: float
    gigpo_step_weight: float
    gigpo_gamma: float
    gigpo_episode_weighting: str
    gigpo_anchor_mode: str

    def validate(self):
        if self.arm_id not in ARMS or self.reduction not in ("response_mean", "trajectory_mean", "token_mean"):
            raise ValueError("unknown arm/reduction")
        if self.normalization not in ("mean", "mean_std") or self.singleton not in ("zero", "upstream"):
            raise ValueError("unknown normalization/singleton convention")
        if self.arm_id == "tree_grpo" and self.normalization != "mean_std":
            raise ValueError("published Tree-GRPO scalar estimator uses std normalization")
        numbers = (self.clip_low, self.clip_high, self.local_coefficient, self.gigpo_step_weight, self.gigpo_gamma)
        if any(isinstance(x, bool) or not math.isfinite(x) for x in numbers):
            raise ValueError("objective coefficients must be finite numbers")
        if not 0 <= self.clip_low < 1 or self.clip_high < 0:
            raise ValueError("invalid clipping widths")
        if self.dual_clip is not None and (not math.isfinite(self.dual_clip) or self.dual_clip <= 1):
            raise ValueError("invalid dual clipping coefficient")
        if self.local_coefficient < 0 or self.gigpo_step_weight < 0 or not 0 <= self.gigpo_gamma <= 1:
            raise ValueError("invalid local coefficient, step weight or discount")
        if self.arm_id != "continuation_q" and self.local_coefficient != 0:
            raise ValueError("local coefficient would change an unrelated arm")
        if self.arm_id != "gigpo" and self.gigpo_step_weight != 0:
            raise ValueError("GiGPO coefficient would change an unrelated arm")
        if self.gigpo_episode_weighting not in ("trajectory", "step"):
            raise ValueError("unknown GiGPO episode weighting")
        if self.gigpo_anchor_mode not in ("state_history_horizon", "observation"):
            raise ValueError("unknown GiGPO anchor convention")


@dataclass(frozen=True)
class ContinuationPanel:
    panel_id: str
    candidate_edges: tuple[str, ...]
    # Every prespecified fit AND holdout draw is assigned to its candidate.
    # IDs must exactly cover the caller's sealed collection manifest.
    draw_leaf_ids: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class PreparedObjective:
    config: ObjectiveConfig
    plan: LossPlan
    diagnostics: tuple

    def loss(self, log_probs, *, edge_subset=None):
        return clipped_loss(self.plan, log_probs, clip_low=self.config.clip_low,
                            clip_high=self.config.clip_high, dual_clip=self.config.dual_clip,
                            edge_subset=edge_subset)


def _sealed_panel_abstentions(graph, inventory, bound):
    """Return only whole, independently bound Q panels with no local target.

    The binding validator authenticates admission/completion records against the
    sealed draw ledger. These additional numerical-boundary checks ensure that
    abstention cannot remove a base trajectory or invent an absent candidate.
    Candidate-generation infrastructure failures can leave other candidates'
    real outcomes in the inventory; none of those extra draws enter base loss.
    """
    records = bound.get("panel_abstentions", {})
    if not isinstance(records, dict):
        raise ValueError("sealed panel abstentions must be a mapping")
    leaves = {leaf.leaf_id: leaf for leaf in graph.leaves}
    edge_ids = {edge.edge_id for edge in graph.edges}
    excluded = set()
    for panel_id, record in records.items():
        if (not isinstance(panel_id, str) or not panel_id or not isinstance(record, dict)
                or record.get("status") not in ("source_inadmissible", "candidate_incomplete")
                or record.get("policy_sha256") != graph.contract.policy_sha256):
            raise ValueError("invalid sealed panel abstention")
        rows = {leaf_id: row for leaf_id, row in inventory.items()
                if row["inventory_role"] in ("q_fit", "q_holdout")
                and row["state_id"] == panel_id}
        candidates = {row["action_id"] for row in rows.values()}
        if len(rows) != 40 or len(candidates) != 5:
            raise ValueError("panel abstention must preserve all five eight-draw candidates")
        if len({(row["task_id"], row["group_id"], row["tree_id"]) for row in rows.values()}) != 1:
            raise ValueError("panel abstention crosses task, group or tree")
        for candidate in candidates:
            candidate_rows = [row for row in rows.values() if row["action_id"] == candidate]
            if (len(candidate_rows) != 8
                    or sum(row["inventory_role"] == "q_fit" for row in candidate_rows) != 6
                    or sum(row["inventory_role"] == "q_holdout" for row in candidate_rows) != 2):
                raise ValueError("panel abstention changed the six-fit two-holdout inventory")
        if record["status"] == "source_inadmissible":
            if any(leaves[key].status != "missing" or leaves[key].edge_ids or row["reward"] is not None
                   for key, row in rows.items()):
                raise ValueError("source admission abstention requires entirely missing edge-free slots")
        else:
            missing = record.get("missing_candidate_edge_ids")
            if (not isinstance(missing, (list, tuple)) or not missing
                    or not all(isinstance(value, str) for value in missing)
                    or len(set(missing)) != len(missing)
                    or set(missing) != candidates - edge_ids):
                raise ValueError("candidate abstention must identify every physically absent candidate")
            if any(leaves[key].status == "valid" or row["reward"] is not None
                   for key, row in rows.items() if row["action_id"] in missing):
                raise ValueError("absent candidates cannot have valid continuation outcomes")
        excluded.update(rows)
    return excluded


def build_objective(graph: ResponseGraph, config: ObjectiveConfig, *,
                    collection_binding, replay_receipts,
                    panels: tuple[ContinuationPanel, ...] = (), selection=None) -> PreparedObjective:
    from openwebrl.action_credit_receipts import validate_receipts
    from openwebrl.action_credit_binding import validate_collection_binding
    config.validate()
    edges = graph.validate(independent_chains=config.arm_id in ("chain_outcome_grpo", "gigpo"))
    validate_receipts(graph, replay_receipts)
    validate_collection_binding(graph, config, panels, collection_binding, selection=selection)
    if panels and config.arm_id != "continuation_q":
        raise ValueError("continuation targets supplied to another arm")
    selected = None
    if selection is not None:
        from openwebrl.action_credit_selection import validate_selection
        if config.arm_id not in ("tree_outcome_sampling", "tree_grpo"):
            raise ValueError("K-leaf tree selection is distinct from fixed-panel and independent-chain collection")
        selected = validate_selection(graph, selection)
    bound = collection_binding.to_dict()
    inventory = bound["inventory"]
    if config.arm_id == "continuation_q":
        # The extra continuations estimate Q only. Including their outcomes in
        # base normalization/loss would silently change the matched control.
        selected = tuple(sorted(leaf_id for leaf_id, row in inventory.items()
                                if row["inventory_role"] == "base"
                                and row["draw_role"] == "train"))
        if not selected:
            raise ValueError("continuation-Q requires its prespecified ordinary base population")
    fit = tuple(x for x in graph.leaves if x.split == "train"
                and (selected is None or x.leaf_id in selected))
    expected_ids = tuple(x.leaf_id for x in fit)
    outcomes = tuple(Outcome(x.leaf_id, x.task_id, x.group_id, x.reward, x.status, x.tree_id) for x in fit)
    if config.arm_id == "gigpo":
        episodes = []
        for leaf in fit:
            steps = []
            for position, edge_id in enumerate(leaf.edge_ids):
                edge = edges[edge_id]
                # Same visible state alone is insufficient. Matching also pins
                # actor history/images and remaining horizon, within task/group.
                state_key = (digest((edge.state_sha256, edge.context_sha256, edge.remaining_turns))
                             if config.gigpo_anchor_mode == "state_history_horizon" else edge.state_sha256)
                terminal = leaf.reward if leaf.status == "valid" and position == len(leaf.edge_ids) - 1 else 0.
                steps.append(Step(edge_id, state_key, terminal, position))
            episodes.append(Episode(leaf.leaf_id, leaf.task_id, leaf.group_id, leaf.reward, tuple(steps), leaf.status))
        result = gigpo_advantages(episodes, expected_ids=expected_ids, gamma=config.gigpo_gamma,
            step_weight=config.gigpo_step_weight, normalization=config.normalization,
            singleton=config.singleton, episode_weighting=config.gigpo_episode_weighting)
        # Establish the exact same base reduction, then replace ONLY step credit.
        plan = leaf_loss_plan(graph, {x.leaf_id: 0. for x in fit if x.status == "valid"},
                              arm_id=config.arm_id, reduction=config.reduction)
        lookup = {v.key: v.advantage for v in result.values}
        plan = replace(plan, terms=tuple(replace(t, advantage=lookup[(t.source_id, t.edge_id)]) for t in plan.terms))
        plan.validate()
        return PreparedObjective(config, plan, (result,))
    if config.arm_id == "tree_grpo":
        result = tree_grpo_advantages(outcomes, expected_ids=expected_ids, singleton=config.singleton)
    else:
        result = outcome_group_advantages(outcomes, expected_ids=expected_ids,
            normalization=config.normalization, singleton=config.singleton)
    plan = leaf_loss_plan(graph, {v.key[0]: v.advantage for v in result.values if v.advantage is not None},
                          arm_id=config.arm_id, reduction=config.reduction, selected_leaf_ids=selected)
    diagnostics = [result]
    if config.arm_id == "continuation_q":
        leaves = {x.leaf_id: x for x in graph.leaves}
        abstained_leaves = _sealed_panel_abstentions(graph, inventory, bound)
        seen_panels, seen_leaves, eligible = set(), set(), []
        for panel in panels:
            if (not isinstance(panel, ContinuationPanel) or not panel.panel_id or panel.panel_id in seen_panels
                    or not isinstance(panel.candidate_edges, tuple) or not isinstance(panel.draw_leaf_ids, tuple)
                    or len(panel.candidate_edges) < 2 or len(set(panel.candidate_edges)) != len(panel.candidate_edges)
                    or len(panel.candidate_edges) != len(panel.draw_leaf_ids)):
                raise ValueError("duplicate or malformed fixed-candidate panel")
            if panel.panel_id in bound.get("panel_abstentions", {}):
                raise ValueError("an abstained panel cannot also supply local candidate targets")
            seen_panels.add(panel.panel_id)
            draws = []
            sibling_keys = set()
            for candidate, leaf_ids in zip(panel.candidate_edges, panel.draw_leaf_ids):
                if candidate not in edges or not isinstance(leaf_ids, tuple) or not leaf_ids:
                    raise ValueError("candidate or prespecified draw slots absent")
                edge = edges[candidate]
                descendants = {leaf.leaf_id for leaf in graph.leaves if candidate in leaf.edge_ids}
                if not descendants <= set(leaf_ids):
                    raise ValueError("panel selectively omitted executed continuation draws")
                sibling_keys.add((edge.task_id, edge.tree_id, edge.parent_edge_id,
                                  edge.context_sha256, edge.state_sha256, edge.remaining_turns))
                fit_count = 0
                for leaf_id in leaf_ids:
                    if leaf_id not in leaves or leaf_id in seen_leaves:
                        raise ValueError("missing/duplicated draw leaf")
                    seen_leaves.add(leaf_id)
                    leaf = leaves[leaf_id]
                    if (leaf.task_id != edge.task_id or leaf.tree_id != edge.tree_id
                            or (leaf.status == "valid" and candidate not in leaf.edge_ids)):
                        raise ValueError("draw path does not execute its fixed candidate")
                    if leaf.split == "holdout":
                        continue
                    fit_count += 1
                    draws.append(CandidateDraw(leaf_id, edge.task_id, panel.panel_id, candidate, leaf.reward, leaf.status))
                if not fit_count:
                    raise ValueError("each candidate needs explicit fit draw slots")
            if len(sibling_keys) != 1:
                raise ValueError("panel compares mismatched states, contexts or horizons")
            q = candidate_q_advantages(draws, expected_ids=tuple(d.draw_id for d in draws))
            diagnostics.append(q)
            if all(v.advantage is not None for v in q.values):
                eligible.append({v.key[-1]: v.advantage for v in q.values})
        panel_ids = {leaf_id for leaf_id, row in inventory.items()
                     if row["inventory_role"] in ("q_fit", "q_holdout")}
        if seen_leaves & abstained_leaves or seen_leaves | abstained_leaves != panel_ids:
            raise ValueError("fixed-panel local-Q adapter requires every planned leaf, including empty missing slots")
        plan = append_local_terms(plan, eligible, coefficient=config.local_coefficient)
    return PreparedObjective(config, plan, tuple(diagnostics))
