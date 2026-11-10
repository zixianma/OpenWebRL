"""Prescheduled-slot Tree-GRPO browser adaptation, with explicit missingness.

K slots per tree are chosen without observing outcomes, including unexecuted or
invalid slots. Only valid selected outcomes enter the existing objective. This
is not upstream terminal-only selection and does not assume missingness is
random. Actual sampler nodes/replay proofs remain separate from planned slots;
an unavailable initial tree never acquires a fabricated root.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import random

from .action_credit import ResponseGraph, _sha, digest
from .action_credit_receipts import ReplayReceipt
from .action_credit_sampling import (RoundSeal, RequestCompletion, SamplingNode,
    TreeSchedule, _bind, _request, forest_sha256)
from .action_credit_selection import TreeSelectionPolicy, inventory_sha256


def slot_selection_policy(policy: TreeSelectionPolicy):
    policy.validate()
    return dict(mode="prescheduled_tree_slot_k_v2", task_order_sha256=policy.task_order_sha256,
        updates=[dict(update_id=u.update_id, schedules=[asdict(s) for s in u.schedules])
                 for u in policy.updates],
        rng="python_random_keyed_per_tree_planned_slot_v2", selection="without_replacement",
        candidate_pool="all_prescheduled_terminal_draw_slots_including_invalid_and_missing",
        missing_initial="no_nodes_no_expansion_dispatch_all_dependent_slots_explicitly_missing",
        invalid_selected="retain_inventory_exclude_from_valid_statistics_and_loss_without_replacement",
        normalization="single_global_objective_after_selection_union")


def _slots(schedule, tree):
    return (f"{tree}:round0:draw0", *(f"{tree}:round{r}:draw{s}"
            for r in range(1, schedule.l + 1) for s in range(schedule.n)))


def _partial_forest(schedule, nodes, receipts, cutoff):
    schedule.sha256
    if type(nodes) is not tuple or not isinstance(receipts, Mapping):
        raise ValueError("immutable actual nodes and exact replay receipt mapping required")
    if (any(type(n) is not SamplingNode for n in nodes)
            or len({n.node_id for n in nodes}) != len(nodes)
            or {n.node_id for n in nodes} != set(receipts)):
        raise ValueError("duplicate node or replay receipt inventory mismatch")
    indexed = {n.node_id: n for n in nodes}
    roots = {}
    for node in nodes:
        _bind(schedule, node, receipts[node.node_id])
        if node.tree_id not in schedule.tree_ids or not 0 <= node.created_round <= cutoff:
            raise ValueError("unexpected tree or uncommitted future-round node")
        if node.parent_node_id is None:
            if node.tree_id in roots or node.created_round != 0 or node.terminal:
                raise ValueError("actual trees require one nonterminal initial root")
            roots[node.tree_id] = node
        else:
            parent = indexed.get(node.parent_node_id)
            if (parent is None or parent.tree_id != node.tree_id or parent.terminal
                    or parent.created_round > node.created_round
                    or parent.remaining_turns != node.remaining_turns + 1):
                raise ValueError("broken ancestry or inconsistent continuation horizon")
    if {n.tree_id for n in nodes} != set(roots):
        raise ValueError("actual nodes lack a real initial tree root")
    if len({(r.context_sha256, r.state_sha256, r.remaining_turns) for r in roots.values()}) > 1:
        raise ValueError("initial trees differ in their starting task state")
    for tree in roots:
        initial = [n for n in nodes if n.tree_id == tree and n.created_round == 0]
        children = {n.node_id: 0 for n in initial}
        for node in initial:
            if node.parent_node_id is not None:
                children[node.parent_node_id] += 1
        if (sum(n.terminal for n in initial) != 1
                or any(children[n.node_id] != (0 if n.terminal else 1) for n in initial)):
            raise ValueError("each admitted initial tree must be a complete actual chain")
    return roots


def _partial_requests(schedule, nodes, round_index):
    requests = []
    for tree in schedule.tree_ids:
        pool = sorted((n for n in nodes if n.tree_id == tree and not n.terminal
                       and n.created_round < round_index), key=lambda n: n.node_id)
        if not pool:
            continue
        # Exactly the existing source-sampling law for every admitted tree.
        rng = random.Random(int(digest([schedule.sha256, tree, round_index, "expand"]), 16))
        requests.extend(_request(schedule, node, tree, round_index, slot)
                        for slot, node in enumerate(rng.choices(pool, k=schedule.n)))
    return tuple(requests)


def _partial_seals(schedule, nodes, seals, last_round):
    if (type(seals) is not tuple or any(type(s) is not RoundSeal for s in seals)
            or len(seals) != last_round
            or {s.round_index for s in seals} != set(range(1, last_round + 1))):
        raise ValueError("exact completed prior-round seals required, including empty rounds")
    indexed = {n.node_id: n for n in nodes}
    assigned = set()
    for seal in seals:
        round_index = seal.round_index
        expected = {q.request_id: q for q in _partial_requests(schedule, nodes, round_index)}
        if (type(round_index) is not int or seal.schedule_sha256 != schedule.sha256
                or seal.input_forest_sha256 != forest_sha256(nodes, before_round=round_index)
                or type(seal.completions) is not tuple
                or any(type(c) is not RequestCompletion for c in seal.completions)
                or len(seal.completions) != len(expected)
                or {c.request_id for c in seal.completions} != set(expected)):
            raise ValueError("partial round seal differs from exact outcome-blind requests")
        covered = set()
        for completion in seal.completions:
            if completion.status not in ("valid", "invalid", "missing"):
                raise ValueError("round request must have a closed disposition")
            if completion.terminal_node_id is None:
                if completion.status == "valid":
                    raise ValueError("valid request lacks its actual terminal artifact")
                continue
            leaf = indexed.get(completion.terminal_node_id)
            if (completion.status == "missing" or leaf is None or not leaf.terminal
                    or leaf.created_round != round_index or leaf.node_id in assigned):
                raise ValueError("missing, duplicate or wrong-round terminal artifact")
            assigned.add(leaf.node_id)
            source = expected[completion.request_id].source_node_id
            ancestor = leaf
            while ancestor.node_id != source and ancestor.parent_node_id is not None:
                if ancestor.created_round != round_index or ancestor.node_id in covered:
                    raise ValueError("continuation must bind distinct newly sampled nodes")
                covered.add(ancestor.node_id)
                ancestor = indexed[ancestor.parent_node_id]
            if ancestor.node_id != source or leaf.node_id == source:
                raise ValueError("terminal does not descend from its requested source")
        if covered != {n.node_id for n in nodes if n.created_round == round_index}:
            raise ValueError("actual round nodes are not bound by closed requests")


def plan_partial_round(schedule, nodes, receipts, *, round_index, seals=()):
    """Plan only actual initial trees; do not manufacture absent source states."""
    if type(round_index) is not int or not 1 <= round_index <= schedule.l:
        raise ValueError("round outside frozen schedule")
    _partial_forest(schedule, nodes, receipts, round_index - 1)
    _partial_seals(schedule, nodes, seals, round_index - 1)
    return _partial_requests(schedule, nodes, round_index)


@dataclass(frozen=True)
class SlotSelectionPartition:
    schedule: TreeSchedule
    nodes: tuple[SamplingNode, ...]
    replay_receipts: tuple[ReplayReceipt, ...]
    seals: tuple[RoundSeal, ...]
    terminal_to_leaf: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class SlotSelectionReceipt:
    frozen_policy: TreeSelectionPolicy
    update_id: int
    contract_sha256: str
    plan_sha256: str
    partitions: tuple[SlotSelectionPartition, ...]
    inventory_sha256: str
    terminal_to_leaf: tuple[tuple[str, str], ...]
    generated_leaf_ids: tuple[str, ...]
    selected_leaf_ids: tuple[str, ...]

    @property
    def policy(self):
        return slot_selection_policy(self.frozen_policy)

    @property
    def policy_sha256(self):
        return digest(self.policy)

    @property
    def sha256(self):
        return digest(dict(schema_version=2, **asdict(self)))


def _freeze_partition(value):
    if type(value) is SlotSelectionPartition:
        return value
    if not isinstance(value, Mapping) or set(value) != {
            "schedule", "nodes", "replay_receipts", "seals", "terminal_to_leaf"}:
        raise ValueError("exact typed sampler partition fields required")
    receipts, mapping = value["replay_receipts"], value["terminal_to_leaf"]
    if not isinstance(receipts, Mapping) or not isinstance(mapping, Mapping):
        raise ValueError("exact replay and terminal-to-leaf mappings required")
    return SlotSelectionPartition(value["schedule"], tuple(value["nodes"]),
        tuple(receipts[key] for key in sorted(receipts)), tuple(value["seals"]), tuple(sorted(mapping.items())))


def _selection(graph, policy, partitions):
    graph.validate()
    policy.validate()
    update = graph.contract.update_id
    if update >= len(policy.updates):
        raise ValueError("update absent from frozen all-update schedule")
    pins = policy.updates[update].schedules
    if (type(partitions) is not tuple or any(type(p) is not SlotSelectionPartition for p in partitions)
            or any(type(p.schedule) is not TreeSchedule for p in partitions)):
        raise ValueError("immutable typed actual sampler partitions required")
    keys = [(p.schedule.task_id, p.schedule.group_id) for p in partitions]
    expected_keys = {(p.task_id, p.group_id) for p in pins}
    if (len(keys) != len(set(keys)) or set(keys) != expected_keys
            or {(l.task_id, l.group_id) for l in graph.leaves if l.split == "train"} != expected_keys):
        raise ValueError("partitions must exactly cover every planned task/group")
    by_key = dict(zip(keys, partitions))
    picked, mappings, all_nodes = [], [], []
    for pin in pins:
        part = by_key[pin.task_id, pin.group_id]
        schedule = pin.bind(graph.contract)
        if part.schedule != schedule:
            raise ValueError("sampler partition differs from its frozen schedule")
        for value in (part.nodes, part.replay_receipts, part.seals, part.terminal_to_leaf):
            if type(value) is not tuple:
                raise ValueError("mutable sampler proof")
        if (any(type(r) is not ReplayReceipt for r in part.replay_receipts)
                or len({r.edge_id for r in part.replay_receipts}) != len(part.replay_receipts)):
            raise ValueError("duplicate or malformed sampler replay receipt")
        receipts = {r.edge_id: r for r in part.replay_receipts}
        roots = _partial_forest(schedule, part.nodes, receipts, schedule.l)
        _partial_seals(schedule, part.nodes, part.seals, schedule.l)
        all_nodes.extend(n.node_id for n in part.nodes)
        leaves = {l.leaf_id: l for l in graph.leaves
                  if (l.task_id, l.group_id) == (pin.task_id, pin.group_id) and l.split == "train"}
        expected_slots = {slot for tree in schedule.tree_ids for slot in _slots(schedule, tree)}
        if set(leaves) != expected_slots:
            raise ValueError("every prescheduled terminal slot must remain explicit")
        mapping = dict(part.terminal_to_leaf)
        terminals = {n.node_id: n for n in part.nodes if n.terminal}
        if (len(mapping) != len(part.terminal_to_leaf) or set(mapping) != set(terminals)
                or len(set(mapping.values())) != len(mapping)):
            raise ValueError("terminal mapping must exactly cover actual sampled terminals")
        for terminal, leaf_id in mapping.items():
            leaf, node = leaves.get(leaf_id), terminals[terminal]
            if leaf is None or leaf.tree_id != node.tree_id or leaf.status == "missing":
                raise ValueError("actual terminal maps to missing or another tree's slot")
            if node.created_round == 0 and leaf_id != f"{node.tree_id}:round0:draw0":
                raise ValueError("initial terminal relabeled as another prescheduled slot")
        for tree in schedule.tree_ids:
            if any(leaves[slot].tree_id != tree for slot in _slots(schedule, tree)):
                raise ValueError("prescheduled slot changed its tree identity")
            if tree not in roots:
                initial, *dependent = _slots(schedule, tree)
                if (leaves[initial].status not in ("invalid", "missing") or leaves[initial].edge_ids
                        or any(leaves[slot].status != "missing" or leaves[slot].edge_ids for slot in dependent)):
                    raise ValueError("unavailable initial trees require an empty failed initial slot and missing dependents")
            rng = random.Random(int(digest([schedule.sha256, tree, "planned_slot_leaf_v2"]), 16))
            picked.extend(rng.sample(sorted(_slots(schedule, tree)), schedule.k))
        for seal in part.seals:
            requests = {q.request_id: q for q in _partial_requests(schedule, part.nodes, seal.round_index)}
            for completion in seal.completions:
                request = requests[completion.request_id]
                slot = f"{request.tree_id}:round{seal.round_index}:draw{request.draw_slot}"
                if (leaves[slot].status != completion.status
                        or (completion.terminal_node_id is not None
                            and mapping[completion.terminal_node_id] != slot)):
                    raise ValueError("closed expansion differs from its immutable slot disposition")
        if any(l.status == "valid" and l.leaf_id not in mapping.values() for l in leaves.values()):
            raise ValueError("valid slot lacks an actual sampled terminal")
        mappings.extend(mapping.items())
    if (len(set(all_nodes)) != len(all_nodes) or len(set(picked)) != len(picked)
            or len({k for k, _ in mappings}) != len(mappings)
            or len({v for _, v in mappings}) != len(mappings)):
        raise ValueError("sampler identities overlap between task/group partitions")
    return tuple(sorted(picked)), tuple(sorted(mappings))


def make_slot_selection(graph, *, policy, plan_sha256, partitions):
    _sha(plan_sha256, "plan_sha256")
    frozen = tuple(_freeze_partition(p) for p in partitions)
    picked, mapping = _selection(graph, policy, frozen)
    return SlotSelectionReceipt(policy, graph.contract.update_id, graph.contract.sha256,
        plan_sha256, tuple(sorted(frozen, key=lambda p: (p.schedule.task_id, p.schedule.group_id))),
        inventory_sha256(graph), mapping, tuple(sorted(v for _, v in mapping)), picked)


def validate_slot_selection(graph, selection):
    if type(selection) is not SlotSelectionReceipt:
        raise ValueError("typed immutable prescheduled-slot receipt required")
    _sha(selection.plan_sha256, "plan_sha256")
    if (selection.update_id != graph.contract.update_id
            or selection.contract_sha256 != graph.contract.sha256
            or selection.inventory_sha256 != inventory_sha256(graph)):
        raise ValueError("selection receipt does not bind this exact update/inventory")
    picked, mapping = _selection(graph, selection.frozen_policy, selection.partitions)
    if (selection.selected_leaf_ids != picked or selection.terminal_to_leaf != mapping
            or selection.generated_leaf_ids != tuple(sorted(v for _, v in mapping))):
        raise ValueError("selected-slot union or actual terminal mapping changed")
    return picked
