"""Executable fixed-inventory collection orchestration for the five-arm pilot.

The caller supplies an isolated worker dispatching the real native collector and
metered terminal judge. Every returned episode is reloaded from its committed
private pointer; in-memory fake results are insufficient. This module allocates
nothing and never fabricates browser states, native scores, or launch readiness.
A failed/ambiguous attempt stops for explicit recovery rather than being resent.
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import random

from .action_credit import Leaf, ResponseGraph, digest
from .action_credit_binding import _check_contract
from .action_credit_campaign import save_scored_pending_update
from .action_credit_collect import EpisodeRequest, PendingEpisode, ReconcileRequired, FrozenActor, prefix_sha256
from .action_credit_episode import load_pending_episode
from .action_credit_megatron import structural_graph, finalize_frozen_scores
from .action_credit_objective import ContinuationPanel
from .action_credit_receipts import ReplayReceipt
from .action_credit_sampling import (SamplingNode, RoundSeal, RequestCompletion,
    plan_round, forest_sha256)
from .action_credit_selection import (TreeSelectionPolicy, selection_partition_graph,
    make_training_selection, make_multigroup_training_selection)
from .action_credit_slot_selection import slot_selection_policy, plan_partial_round, make_slot_selection


def online_collection_protocol(seed):
    if type(seed) is not int or seed < 0:
        raise ValueError("explicit collection seed required")
    return dict(version="fixed_five_arm_online_v1", seed=seed, groups=48, base_chains=8,
        updates=10, tree_m=2, tree_n=2, tree_l=2, tree_k=4,
        q_source_groups=8, q_source_chain=0, q_candidates=5, q_draws=8, q_fit_draws=6,
        q_depth="keyed_uniform_observed_preaction_context_v1",
        q_candidates_protocol="independent_raw_responses_distinct_sampling_events",
        replacement="none", disposition="retain_every_planned_draw")


def terminal_node_id(request):
    return digest([request.request_id, "terminal_observation"])


def _persist(store, section, key, value):
    # Canonical copy prevents caller mutation and fails if an earlier decision
    # existed, including decisions made before a process crash.
    value = json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    with store._transaction() as state:
        records = state.setdefault(section, {})
        old = records.get(key)
        if old is not None and old != value:
            raise ValueError("immutable online decision changed: " + section)
        records[key] = value
    return value


async def _all(*operations):
    """Own every dispatched worker through closure, including sibling failures."""
    results = await asyncio.gather(*operations, return_exceptions=True)
    for result in results:
        if isinstance(result, BaseException):
            raise result
    return results


def _seed(*identity):
    return int(digest(identity)[:15], 16)


def plan_online_update(store, contract, task_order_path, *, tree_policy=None):
    """Freeze all384 base/480 tree/704 Q slots before the first dispatch."""
    manifest = store.manifest.to_dict()
    _check_contract(manifest, contract)
    algorithm, dataset = manifest["science"]["algorithm"], manifest["science"]["dataset"]
    protocol = algorithm.get("online_collection_protocol", {})
    if protocol != online_collection_protocol(protocol.get("seed")):
        raise ValueError("pin the complete fixed online collection protocol")
    raw = Path(task_order_path).read_bytes()
    order_sha = hashlib.sha256(raw).hexdigest()
    order = json.loads(raw)
    if (order_sha != dataset.get("task_order_sha256") or order["source_sha256"] != dataset["source_sha256"]
            or order["groups_per_iteration"] != 48 or len(order["iterations"]) != 10
            or order.get("outcome_dependent_replacement") is not False):
        raise ValueError("fixed48-group, ten-update task-order artifact mismatch")
    entry = order["iterations"][contract.update_id]
    groups = entry["groups"]
    if (entry["iteration"] != contract.update_id+1 or len(groups) != 48
            or len({g["group_id"] for g in groups}) != 48):
        raise ValueError("complete fixed task-group cursor required")
    for group in groups:
        if (group["task_id"] not in contract.train_task_ids
                or order["original_manifest_task_order"][group["source_index"]] != group["task_id"]):
            raise ValueError("task source membership changed")
    arm = manifest["arm_id"]
    tree_arm = arm in ("tree_outcome_sampling", "tree_grpo")
    schedules = {}
    if tree_arm:
        if (type(tree_policy) is not TreeSelectionPolicy or tree_policy.task_order_sha256 != order_sha
                or slot_selection_policy(tree_policy) != algorithm.get("leaf_selection") or len(tree_policy.updates) != 10):
            raise ValueError("complete predeclared tree selection policy required")
        pins = tree_policy.updates[contract.update_id].schedules
        if [(p.task_id, p.group_id) for p in pins] != [(g["task_id"], g["group_id"]) for g in groups]:
            raise ValueError("tree schedules differ from fixed task order")
        for pin in pins:
            if (pin.m, pin.n, pin.l, pin.k) != (2, 2, 2, 4):
                raise ValueError("tree schedule must preserve M2 N2 L2 K4")
            schedules[pin.group_id] = pin.bind(contract)
    elif tree_policy is not None:
        raise ValueError("tree policy supplied to a non-tree arm")
    draws = []
    def append(group, tree, slot, state, action, role, leaf, **meta):
        spec = dict(task_id=group["task_id"], group_id=group["group_id"], tree_id=tree,
            draw_slot=slot, state_id=state, action_id=action,
            draw_role="holdout" if role == "q_holdout" else "train", inventory_role=role)
        draws.append(dict(leaf_id=leaf, spec=spec, **meta))
    for group in groups:
        gid = group["group_id"]
        if tree_arm:
            for tree in schedules[gid].tree_ids:
                for r, count in ((0, 1), (1, 2), (2, 2)):
                    for slot in range(count):
                        leaf = f"{tree}:round{r}:draw{slot}"
                        append(group, tree, slot, f"{tree}:round{r}", "tree_continuation", "base", leaf, round_index=r)
        else:
            for slot in range(8):
                tree = f"{gid}:chain{slot:02d}"
                append(group, tree, slot, gid+":root", "independent_chain", "base", tree+":terminal")
    if arm == "continuation_q":
        for group in groups[:8]:
            gid, panel = group["group_id"], group["group_id"]+":localq"
            for c in range(5):
                edge = digest([contract.sha256, panel, "candidate", c])
                for slot in range(8):
                    append(group, f"{gid}:chain00", slot, panel, edge,
                           "q_fit" if slot < 6 else "q_holdout", f"{panel}:candidate{c}:draw{slot}", candidate_index=c)
    elif arm not in ("chain_outcome_grpo", "gigpo", "tree_outcome_sampling", "tree_grpo"):
        raise ValueError("unknown online arm")
    result = store.plan_update(contract.update_id, policy_sha256=contract.policy_sha256,
                              draws=[item["spec"] for item in draws])
    plan = dict(update_id=contract.update_id, graph_contract_sha256=contract.sha256,
        policy_sha256=contract.policy_sha256, task_order_sha256=order_sha, groups=groups,
        plan_sha256=result["plan_sha256"], protocol=protocol, draws=draws)
    _persist(store, "online_collection_plans", str(contract.update_id), plan)
    return plan, schedules


def draft_online_update(store, contract, plan, episodes, *, panels=(), selection_builder_data=None):
    """Assemble all terminal slots, sharing only byte-identical sampled edges."""
    state = store.snapshot()
    if state.get("online_collection_plans", {}).get(str(contract.update_id)) != plan:
        raise ValueError("use the durable precollection plan")
    by_leaf = {episode.request.leaf_id: episode for episode in episodes}
    if len(by_leaf) != len(episodes) or set(by_leaf) - {d["leaf_id"] for d in plan["draws"]}:
        raise ValueError("duplicate or unplanned episode")
    edges, rows, tensors, receipts, leaves, leaf_to_draw = {}, {}, {}, {}, [], {}
    for item in plan["draws"]:
        spec, leaf_id = item["spec"], item["leaf_id"]
        draw_id = digest(store._draw_identity(contract.update_id, contract.policy_sha256, spec))
        draw = state["draws"].get(draw_id)
        if not draw or draw["status"] not in ("valid", "invalid", "missing"):
            raise ValueError("every planned draw requires a terminal disposition")
        selected = draw["selected_attempt"]
        completion = state["attempts"][selected]["completion"] if selected is not None else None
        outcome = completion["outcome"] if completion else None
        reward = outcome.get("reward") if outcome else None
        episode = by_leaf.get(leaf_id)
        if episode is None:
            if draw["status"] == "valid" or (completion and completion.get("sampling_terminal_node_id")):
                raise ValueError("terminal artifact required for generated/valid draw")
            leaf = Leaf(leaf_id, spec["task_id"], spec["tree_id"], (), draw["status"], None,
                        spec["draw_role"], spec["group_id"])
        else:
            if (episode.draw_id != draw_id or episode.request.spec() != spec
                    or episode.request.attempt_id != selected):
                raise ValueError("episode differs from the selected planned attempt")
            leaf = episode.leaf(status=draw["status"], reward=reward)
            for turn in episode.turns:
                edge = turn.edge
                turn.row.validate(edge, contract, turn.tensors)
                if edge.edge_id in edges:
                    if (edges[edge.edge_id] != edge or rows[edge.edge_id] != turn.row
                            or receipts[edge.edge_id] != turn.replay_receipt):
                        raise ValueError("shared sampled edge changed between continuations")
                else:
                    edges[edge.edge_id], rows[edge.edge_id] = edge, turn.row
                    tensors[edge.edge_id], receipts[edge.edge_id] = turn.tensors, turn.replay_receipt
        leaves.append(leaf); leaf_to_draw[leaf_id] = draw_id
    graph = ResponseGraph(contract, tuple(edges.values()), tuple(leaves), tuple(d["leaf_id"] for d in plan["draws"]))
    structural_graph(graph).validate()
    return dict(graph=graph, rows=rows, tensor_payloads=tensors, replay_receipts=receipts,
                leaf_to_draw=leaf_to_draw, panels=tuple(panels), selection_builder_data=selection_builder_data)


def materialize_tree_episode(episode, contract, *, round_index, source=None):
    """Convert actual captured contexts to post-action sampler nodes.

    Nonterminal post-action context is the next actual processor input. Terminal
    context is explicitly an evidence/history identity, never alleged actor IDs.
    Replayed ancestors are supplied by the previously materialized source.
    """
    turns = episode.turns
    depth = len(source[2]) if source else 0
    if not turns or len(turns) <= depth:
        raise ValueError("tree continuation produced no new sampled response")
    verifier = turns[0].replay_receipt.verifier_sha256
    nodes, receipts, prefixes = [], {}, {}
    def add(node_id, parent, context, state, snapshot, horizon, terminal, prefix):
        receipt = ReplayReceipt(node_id, contract.sha256, state, context, horizon, verifier, snapshot, True)
        node = SamplingNode(node_id, episode.request.task_id, episode.request.tree_id, parent,
            contract.policy_sha256, context, state, snapshot, horizon, terminal, round_index, receipt.sha256)
        nodes.append(node); receipts[node_id] = receipt; prefixes[node_id] = tuple(prefix)
        return node
    if source is None:
        turn = turns[0]
        root = add(digest([episode.request.tree_id, "root"]), None, turn.edge.context_sha256,
            turn.edge.state_sha256, turn.evidence_sha256, turn.edge.remaining_turns, False, ())
        parent = root.node_id
    else:
        node, receipt, prefix = source
        boundary = turns[depth]
        if (boundary.edge.context_sha256 != node.context_sha256 or boundary.edge.state_sha256 != node.state_sha256
                or boundary.edge.remaining_turns != node.remaining_turns
                or tuple(t.sha256 for t in turns[:depth]) != tuple(t.sha256 for t in prefix)):
            raise ValueError("actual continuation source differs from the frozen sampled source")
        parent = node.node_id
    for index in range(depth, len(turns)):
        terminal = index == len(turns)-1
        if terminal:
            observation = episode.terminal_evidence["observation"]
            context = digest(dict(terminal_observation=observation,
                policy_history=tuple(t.edge.edge_id for t in turns), domain="terminal_evidence_not_actor_input"))
            state, evidence = digest(observation["snapshot"]), digest(episode.terminal_evidence)
            node_id = terminal_node_id(episode.request)
        else:
            following = turns[index+1]
            context, state, evidence = following.edge.context_sha256, following.edge.state_sha256, following.evidence_sha256
            node_id = digest([following.edge.edge_id, "preaction_node"])
        node = add(node_id, parent, context, state, evidence,
                   turns[index].edge.remaining_turns-1, terminal, turns[:index+1])
        parent = node.node_id
    return tuple(nodes), receipts, prefixes


async def collect_online_update(store, contract, actor, task_order_path, *, dispatch, request_pins,
                                attempt_namespace, verifier_sha256, tree_policy=None, resume_pending=None):
    """Dispatch isolated trusted native workers; reload their durable results.

    ``await dispatch(request, prefix)`` must collect, judge, charge and settle the
    exact attempt. It returns PendingEpisode or None for an explicitly closed
    missing/invalid slot. Concurrency limits belong to the caller's worker pool.
    ``request_pins(task_id)`` returns the six exact task/runtime digest fields.
    """
    plan, schedules = plan_online_update(store, contract, task_order_path, tree_policy=tree_policy)
    protocol, episodes = plan["protocol"], {}
    version_identity = None
    draw_by_leaf = {d["leaf_id"]: d for d in plan["draws"]}
    def identity(item):
        return digest(store._draw_identity(contract.update_id, contract.policy_sha256, item["spec"]))
    async def get(item, prefix=(), *, seed=None, candidate_depth=None):
        nonlocal version_identity
        spec, leaf_id = item["spec"], item["leaf_id"]
        request_id = digest([store.manifest.sha256, contract.sha256, leaf_id, "episode"])
        attempt = digest([request_id, attempt_namespace, "attempt"])
        pins = request_pins(spec["task_id"])
        request = EpisodeRequest(request_id=request_id, attempt_id=attempt, leaf_id=leaf_id, **spec,
            **pins, prefix_sha256=prefix_sha256(prefix),
            generation_seed=seed if seed is not None else _seed(protocol["seed"], contract.update_id, leaf_id),
            planned_candidate_edge_id=spec["action_id"] if candidate_depth is not None else None,
            candidate_prefix_length=candidate_depth)
        current_actor = actor(request) if callable(actor) else actor
        if type(current_actor) is not FrozenActor:
            raise ValueError("actor resolver must return a verified FrozenActor for the exact request")
        current_actor.validate(contract)
        version = dict(policy_sha256=contract.policy_sha256, weight_version=current_actor.weight_version)
        if version_identity is None:
            _persist(store, "online_collection_weight_versions", str(contract.update_id), version)
            version_identity = version
        elif version_identity != version:
            raise ValueError("all actor replicas must expose the same frozen policy and weight version")
        def checked_episode(episode):
            actual, expected = asdict(episode.request), asdict(request)
            actual.pop("attempt_id"); expected.pop("attempt_id")
            if actual != expected:
                raise ReconcileRequired("saved episode differs from immutable request/prefix/seed/runtime inputs")
            return episode
        snap = store.snapshot()
        draw = snap["draws"].get(identity(item))
        if draw and draw["status"] in ("valid", "invalid", "missing"):
            selected = draw["selected_attempt"]
            if selected and selected in snap.get("pending_episodes", {}):
                episode = checked_episode(load_pending_episode(store, contract, current_actor, selected,
                                                               verifier_sha256=verifier_sha256))
                episodes[leaf_id] = episode
                return episode
            if draw["status"] == "valid":
                raise ReconcileRequired("valid draw missing committed physical episode")
            return None
        if draw and draw["attempts"]:
            active = [a for a in draw["attempts"] if snap["attempts"][a]["status"] == "active"]
            if len(active) == 1 and active[0] in snap.get("pending_episodes", {}) and resume_pending is not None:
                saved = checked_episode(load_pending_episode(store, contract, current_actor, active[0],
                                                             verifier_sha256=verifier_sha256))
                await resume_pending(saved)
                current = store.snapshot()["draws"][identity(item)]
                if current["status"] not in ("valid", "invalid") or current["selected_attempt"] != active[0]:
                    raise ReconcileRequired("pending judge recovery did not settle its exact attempt")
                episodes[leaf_id] = saved
                return saved
            # Even a different execution namespace cannot redispatch an active
            # reservation or abandon a committed pending episode after a crash.
            raise ReconcileRequired("existing attempt requires recovery/judge reconciliation before collection")
        returned = await dispatch(request, tuple(prefix))
        snap = store.snapshot(); draw = snap["draws"].get(identity(item))
        if not draw or draw["status"] not in ("valid", "invalid", "missing"):
            raise ReconcileRequired("worker did not durably settle the planned episode")
        if returned is None:
            if draw["status"] == "valid":
                raise ValueError("valid outcome without physical episode")
            return None
        if type(returned) is not PendingEpisode or returned.request != request:
            raise ValueError("worker returned a different request/attempt")
        episode = load_pending_episode(store, contract, current_actor, attempt, verifier_sha256=verifier_sha256)
        if episode.request != request or episode.draw_id != identity(item) or draw["selected_attempt"] != attempt:
            raise ValueError("durable episode does not match dispatched identity")
        episodes[leaf_id] = episode
        return episode
    def missing(item, reason):
        rid = digest([store.manifest.sha256, contract.sha256, item["leaf_id"], "episode"])
        draw = store.reserve_draw(rid, update_id=contract.update_id,
                                 policy_sha256=contract.policy_sha256, **item["spec"])
        old = store.snapshot()["draws"][draw["draw_id"]]
        if old["status"] != "missing":
            store.mark_missing_draw(draw["draw_id"], reason=reason)
    if schedules:
        async def collect_tree(group):
            schedule = schedules[group["group_id"]]
            nodes, receipts, prefixes, seals, mapping = [], {}, {}, [], {}
            items = [d for d in plan["draws"] if d["spec"]["group_id"] == group["group_id"]]
            initial = [d for d in items if d["round_index"] == 0]
            first = await _all(*(get(item) for item in initial))
            for item, episode in zip(initial, first):
                if episode is None:
                    for dependent in items:
                        if dependent["spec"]["tree_id"] == item["spec"]["tree_id"] and dependent["round_index"] > 0:
                            missing(dependent, "initial tree source unavailable; no expansion replacement")
                    continue
                added, bound, paths = materialize_tree_episode(episode, contract, round_index=0)
                nodes.extend(added); receipts.update(bound); prefixes.update(paths)
                mapping[added[-1].node_id] = item["leaf_id"]
            for round_index in (1, 2):
                requests = plan_partial_round(schedule, tuple(nodes), receipts, round_index=round_index, seals=tuple(seals))
                _persist(store, "online_tree_rounds", f"{contract.update_id}:{group['group_id']}:{round_index}",
                    dict(requests=[asdict(q) for q in requests], forest_sha256=forest_sha256(nodes, before_round=round_index)))
                sources = {n.node_id: n for n in nodes}
                selected_items = [next(d for d in items if d["spec"]["tree_id"] == q.tree_id
                    and d["round_index"] == round_index and d["spec"]["draw_slot"] == q.draw_slot) for q in requests]
                result = await _all(*(get(item, prefixes[q.source_node_id], seed=q.generation_seed)
                    for item, q in zip(selected_items, requests)))
                completion = []
                for item, q, episode in zip(selected_items, requests, result):
                    draw = store.snapshot()["draws"][identity(item)]
                    node_id = None
                    if episode is not None:
                        source = (sources[q.source_node_id], receipts[q.source_node_id], prefixes[q.source_node_id])
                        added, bound, paths = materialize_tree_episode(episode, contract, round_index=round_index, source=source)
                        nodes.extend(added); receipts.update(bound); prefixes.update(paths)
                        node_id = added[-1].node_id; mapping[node_id] = item["leaf_id"]
                    completion.append(RequestCompletion(q.request_id, draw["status"], node_id))
                seals.append(RoundSeal(schedule.sha256, round_index,
                    forest_sha256(nodes, before_round=round_index), tuple(completion)))
            return dict(schedule=schedule, nodes=tuple(nodes), replay_receipts=receipts,
                        seals=tuple(seals), terminal_to_leaf=mapping)
        partitions = await _all(*(collect_tree(g) for g in plan["groups"]))
        return draft_online_update(store, contract, plan, tuple(episodes.values()),
            selection_builder_data=dict(policy=tree_policy, partitions=tuple(partitions), plan_sha256=plan["plan_sha256"]))
    await _all(*(get(item) for item in plan["draws"] if item["spec"]["inventory_role"] == "base"))
    panels = []
    if store.manifest.to_dict()["arm_id"] == "continuation_q":
        async def collect_q(group):
            gid, panel_id = group["group_id"], group["group_id"]+":localq"
            items = [d for d in plan["draws"] if d["spec"]["state_id"] == panel_id]
            source_id = f"{gid}:chain00:terminal"
            source = episodes.get(source_id)
            depth = random.Random(_seed(protocol["seed"], contract.update_id, panel_id, "depth")).randrange(len(source.turns)) if source else None
            admission = dict(status="admitted" if source else "abstained", source_leaf_id=source_id,
                source_attempt_id=source.request.attempt_id if source else None, depth=depth,
                source_context_sha256=source.turns[depth].edge.context_sha256 if source else None,
                source_evidence_sha256=source.turns[depth].evidence_sha256 if source else None,
                policy_sha256=contract.policy_sha256, reason=None if source else "no_complete_physical_source_episode",
                selection_policy_sha256=digest(protocol))
            _persist(store, "online_panel_admissions", f"{contract.update_id}:{panel_id}", admission)
            if source is None:
                for item in items: missing(item, "prespecified source admission failed; no replacement")
                return None
            prefix = source.turns[:depth]
            async def candidate(index):
                slots = sorted((d for d in items if d["candidate_index"] == index), key=lambda d:d["spec"]["draw_slot"])
                first = await get(slots[0], prefix, candidate_depth=depth)
                if first is None or len(first.turns) <= depth:
                    snapshot = store.snapshot()
                    attempts = set(snapshot["draws"][identity(slots[0])]["attempts"])
                    if any(call["attempt_id"] in attempts and call["turn"] == depth and call["status"] == "received"
                           for call in snapshot.get("online_actor_requests", {}).values()):
                        raise ReconcileRequired("sampled candidate exists in partial artifacts; recover it instead of abstaining")
                    for item in slots[1:]:
                        missing(item, "candidate infrastructure failure before sampled edge; no replacement")
                    return None
                candidate_prefix = first.turns[:depth+1]
                if candidate_prefix[-1].edge.edge_id != slots[0]["spec"]["action_id"]:
                    raise ValueError("sampled candidate differs from planned raw-response identity")
                await _all(*(get(item, candidate_prefix, candidate_depth=depth) for item in slots[1:]))
                return slots[0]["spec"]["action_id"], tuple(item["leaf_id"] for item in slots)
            candidates = await _all(*(candidate(i) for i in range(5)))
            if any(value is None for value in candidates):
                snapshot = store.snapshot()
                closed = {item["leaf_id"]: {k:snapshot["draws"][identity(item)][k]
                    for k in ("status", "selected_attempt")} for item in items}
                _persist(store, "online_panel_completions", f"{contract.update_id}:{panel_id}",
                    dict(status="candidate_incomplete", missing_candidate_edge_ids=[
                        next(d["spec"]["action_id"] for d in items if d["candidate_index"] == i)
                        for i,value in enumerate(candidates) if value is None],
                        closed_inventory_sha256=digest(closed), reason="candidate_infrastructure_failure_before_sampled_edge"))
                return None
            return ContinuationPanel(panel_id, tuple(x[0] for x in candidates), tuple(x[1] for x in candidates))
        panels = [p for p in await _all(*(collect_q(g) for g in plan["groups"][:8])) if p is not None]
    return draft_online_update(store, contract, plan, tuple(episodes.values()), panels=panels)


def save_scored_draft(store, config, draft, scores, *, scorer_source_sha256, window_config, optimizer_rng_artifact):
    """Construct outcome-blind tree selection only after authentic frozen scores."""
    selection = None
    builder = draft.get("selection_builder_data")
    if builder is not None:
        graph = finalize_frozen_scores(draft["graph"], scores, rows=draft["rows"], scorer_source_sha256=scorer_source_sha256)
        selection = make_slot_selection(graph, policy=builder["policy"],
            plan_sha256=builder["plan_sha256"], partitions=builder["partitions"])
    return save_scored_pending_update(store, config, draft, scores, scorer_source_sha256=scorer_source_sha256,
        window_config=window_config, optimizer_rng_artifact=optimizer_rng_artifact,
        panels=draft.get("panels", ()), selection=selection)
