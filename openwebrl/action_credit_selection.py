"""Outcome-blind leaf selection receipts for one or several sampled forests.

The full graph is retained, including generated-but-unselected training leaves,
holdout leaves and failed/missing slots. Selection and holdout are distinct roles.
The fixed schedule/seed policy must be pinned in the campaign BEFORE dispatch.
Full inventory hashes bind labels AFTER collection but never seed the selection.

Terminal-to-leaf mapping is explicit collector provenance, not inferred physical
proof: post-action SamplingNodes and pre-action ResponseEdges represent different
states. This module checks identities/membership but cannot prove browser replay
or authenticate a mapping. Multi-group composition binds every partition and
the full update inventory; it does not implement an online browser collector.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass

from .action_credit import CollectionContract, ResponseGraph, _id, _sha, digest
from .action_credit_receipts import ReplayReceipt
from .action_credit_sampling import (
    RoundSeal, SamplingNode, TreeSchedule, forest_sha256, select_leaves,
)


def inventory_sha256(graph: ResponseGraph) -> str:
    """Canonical complete graph identity, independent of storage order."""
    graph.validate()
    return digest(dict(contract=graph.contract.sha256,
        edges=[asdict(e) for e in sorted(graph.edges, key=lambda e: e.edge_id)],
        leaves=[asdict(l) for l in sorted(graph.leaves, key=lambda l: l.leaf_id)],
        planned_leaf_ids=sorted(graph.planned_leaf_ids)))


@dataclass(frozen=True)
class TrainingSelectionReceipt:
    schedule: TreeSchedule
    nodes: tuple[SamplingNode, ...]
    replay_receipts: tuple[ReplayReceipt, ...]
    seals: tuple[RoundSeal, ...]
    terminal_to_leaf: tuple[tuple[str, str], ...]
    inventory_sha256: str
    generated_leaf_ids: tuple[str, ...]
    selected_leaf_ids: tuple[str, ...]

    @property
    def policy(self):
        s = self.schedule
        return dict(mode="tree_uniform_terminal_k_v1", m=s.m, n=s.n, l=s.l, k=s.k,
            seed=s.seed, rng="python_random_keyed_per_tree_v1",
            candidate_pool="generated_training_terminal_nodes", selection="without_replacement",
            task_id=s.task_id, group_id=s.group_id)

    @property
    def policy_sha256(self):
        return digest(self.policy)

    @property
    def schedule_sha256(self):
        return self.schedule.sha256

    @property
    def forest_sha256(self):
        return forest_sha256(self.nodes, before_round=self.schedule.l + 1)

    @property
    def seals_sha256(self):
        return digest([asdict(s) for s in self.seals])

    @property
    def mapping_sha256(self):
        return digest(self.terminal_to_leaf)

    @property
    def sha256(self):
        return digest(dict(schema_version=1, **asdict(self)))


@dataclass(frozen=True)
class TreeSchedulePin:
    """Prespecified group schedule without an as-yet-unknown future actor hash."""
    task_id: str
    group_id: str
    m: int
    n: int
    l: int
    k: int
    seed: int

    def validate(self):
        _id(self.task_id, "task_id")
        _id(self.group_id, "group_id")
        if any(type(v) is not int or v < minimum for v, minimum in
               ((self.m, 1), (self.n, 1), (self.l, 0), (self.k, 1), (self.seed, 0))):
            raise ValueError("invalid frozen group schedule")
        if self.k > 1 + self.n * self.l:
            raise ValueError("K exceeds the largest possible terminal inventory per tree")

    def bind(self, contract: CollectionContract) -> TreeSchedule:
        self.validate()
        schedule = TreeSchedule(contract, **asdict(self))
        schedule.sha256  # Validate task membership and the exact current contract.
        return schedule


@dataclass(frozen=True)
class TreeUpdateSelectionPolicy:
    update_id: int
    schedules: tuple[TreeSchedulePin, ...]

    def validate(self):
        if type(self.update_id) is not int or self.update_id < 0:
            raise ValueError("invalid tree policy update ID")
        if (type(self.schedules) is not tuple or not self.schedules
                or any(type(s) is not TreeSchedulePin for s in self.schedules)):
            raise ValueError("immutable nonempty group schedules required")
        for schedule in self.schedules:
            schedule.validate()
        if (len({s.group_id for s in self.schedules}) != len(self.schedules)
                or len({s.task_id for s in self.schedules}) != len(self.schedules)):
            raise ValueError("task and group identities must be unique within an update")


@dataclass(frozen=True)
class TreeSelectionPolicy:
    """Stable campaign pin for every update's ordered task/group schedules.

    Updates are contiguous from zero, with the full campaign endpoint fixed by
    their count. Their order and task-order artifact SHA are part of the policy.
    Actor identity belongs to the current CollectionContract, not this policy.
    """
    task_order_sha256: str
    updates: tuple[TreeUpdateSelectionPolicy, ...]

    def validate(self):
        _sha(self.task_order_sha256, "task_order_sha256")
        if (type(self.updates) is not tuple or not self.updates
                or any(type(u) is not TreeUpdateSelectionPolicy for u in self.updates)):
            raise ValueError("immutable complete campaign update schedules required")
        for update in self.updates:
            update.validate()
        if tuple(u.update_id for u in self.updates) != tuple(range(len(self.updates))):
            raise ValueError("policy updates must be ordered, unique and contiguous from zero")

    @property
    def policy(self):
        self.validate()
        return dict(mode="tree_uniform_terminal_k_multigroup_v1",
            task_order_sha256=self.task_order_sha256, update_count=len(self.updates),
            updates=[dict(update_id=u.update_id, schedules=[asdict(s) for s in u.schedules])
                     for u in self.updates],
            rng="python_random_keyed_per_tree_v1",
            candidate_pool="generated_training_terminal_nodes", selection="without_replacement",
            normalization="single_global_objective_after_selection_union")

    @property
    def sha256(self):
        return digest(self.policy)


@dataclass(frozen=True)
class MultiGroupTrainingSelectionReceipt:
    frozen_policy: TreeSelectionPolicy
    update_id: int
    contract_sha256: str
    partitions: tuple[TrainingSelectionReceipt, ...]
    inventory_sha256: str
    terminal_to_leaf: tuple[tuple[str, str], ...]
    generated_leaf_ids: tuple[str, ...]
    selected_leaf_ids: tuple[str, ...]
    outside_partition_holdout_leaf_ids: tuple[str, ...]

    @property
    def policy(self):
        return self.frozen_policy.policy

    @property
    def policy_sha256(self):
        return self.frozen_policy.sha256

    @property
    def sha256(self):
        return digest(dict(schema_version=1, **asdict(self)))


def selection_partition_graph(graph: ResponseGraph, *, task_id: str, group_id: str) -> ResponseGraph:
    """Keep every leaf in a task/group, including invalid, missing and holdout.

    Only unreferenced edges outside this partition are excluded. The global
    bundle separately binds those edges and all outside-partition holdouts.
    """
    graph.validate()
    return _partition_graph(graph, task_id=task_id, group_id=group_id)


def _partition_graph(graph, *, task_id, group_id):
    """Partition an already-validated global graph without rescanning all tokens."""
    leaves = tuple(l for l in graph.leaves if (l.task_id, l.group_id) == (task_id, group_id))
    if not leaves:
        raise ValueError("scheduled task/group partition is absent from the graph")
    edge_ids = {edge_id for leaf in leaves for edge_id in leaf.edge_ids}
    partition = ResponseGraph(graph.contract, tuple(e for e in graph.edges if e.edge_id in edge_ids),
                              leaves, tuple(l.leaf_id for l in leaves))
    partition.validate()
    return partition


def selection_partition_graphs(graph: ResponseGraph) -> dict[tuple[str, str], ResponseGraph]:
    """Construct all training partitions after one global graph validation."""
    graph.validate()
    keys = sorted({(l.task_id, l.group_id) for l in graph.leaves if l.split == "train"})
    return {key: _partition_graph(graph, task_id=key[0], group_id=key[1]) for key in keys}


def _multigroup_selection(graph, policy, partitions):
    if type(graph) is not ResponseGraph or type(policy) is not TreeSelectionPolicy:
        raise ValueError("typed graph and frozen multi-group policy required")
    graph.validate()
    policy.validate()
    if (type(partitions) is not tuple or not partitions
            or any(type(s) is not TrainingSelectionReceipt for s in partitions)):
        raise ValueError("immutable nonempty single-group receipt partitions required")
    update_id = graph.contract.update_id
    if update_id >= len(policy.updates):
        raise ValueError("current update absent from frozen campaign policy")
    scheduled = policy.updates[update_id].schedules
    keys = {(s.task_id, s.group_id) for s in scheduled}
    actual_keys = [(s.schedule.task_id, s.schedule.group_id) for s in partitions]
    train_keys = {(l.task_id, l.group_id) for l in graph.leaves if l.split == "train"}
    if len(set(actual_keys)) != len(actual_keys) or set(actual_keys) != keys or train_keys != keys:
        raise ValueError("partitions must exactly cover scheduled training groups without repeats or omissions")
    by_key = dict(zip(actual_keys, partitions))
    generated, picked, mapping, node_ids = [], [], [], []
    for pin in scheduled:
        receipt = by_key[pin.task_id, pin.group_id]
        if receipt.schedule != pin.bind(graph.contract):
            raise ValueError("group schedule or current collection contract differs from frozen policy")
        partition = _partition_graph(graph, task_id=pin.task_id, group_id=pin.group_id)
        trees = set(receipt.schedule.tree_ids)
        if any(l.split == "train" and l.tree_id not in trees for l in partition.leaves):
            raise ValueError("training disposition belongs to an unplanned tree")
        picked.extend(validate_selection(partition, receipt))
        generated.extend(receipt.generated_leaf_ids)
        mapping.extend(receipt.terminal_to_leaf)
        node_ids.extend(n.node_id for n in receipt.nodes)
    for values in (generated, picked, node_ids, [p[0] for p in mapping], [p[1] for p in mapping]):
        if len(set(values)) != len(values):
            raise ValueError("overlapping partition leaf or sampler node identities")
    outside = tuple(sorted(l.leaf_id for l in graph.leaves if (l.task_id, l.group_id) not in keys))
    # train_keys equality above guarantees every outside leaf is a holdout.
    return tuple(sorted(generated)), tuple(sorted(picked)), tuple(sorted(mapping)), outside


def make_multigroup_training_selection(graph: ResponseGraph, *, policy: TreeSelectionPolicy,
        partitions: tuple[TrainingSelectionReceipt, ...]) -> MultiGroupTrainingSelectionReceipt:
    """Compose proofs only; objective reduction must still occur once globally."""
    generated, picked, mapping, outside = _multigroup_selection(graph, policy, partitions)
    return MultiGroupTrainingSelectionReceipt(policy, graph.contract.update_id, graph.contract.sha256,
        tuple(sorted(partitions, key=lambda s: (s.schedule.task_id, s.schedule.group_id))),
        inventory_sha256(graph), mapping, generated, picked, outside)


def _validate_multigroup_selection(graph, selection):
    s = selection
    for value in (s.partitions, s.terminal_to_leaf, s.generated_leaf_ids, s.selected_leaf_ids,
                  s.outside_partition_holdout_leaf_ids):
        if type(value) is not tuple:
            raise ValueError("multi-group selection proof must be immutable")
    if (type(s.update_id) is not int or s.update_id != graph.contract.update_id
            or s.contract_sha256 != graph.contract.sha256):
        raise ValueError("multi-group receipt is bound to a different update or collection contract")
    if s.inventory_sha256 != inventory_sha256(graph):
        raise ValueError("complete multi-group inventory changed after selection")
    generated, picked, mapping, outside = _multigroup_selection(graph, s.frozen_policy, s.partitions)
    if (s.generated_leaf_ids, s.selected_leaf_ids, s.terminal_to_leaf,
            s.outside_partition_holdout_leaf_ids) != (generated, picked, mapping, outside):
        raise ValueError("multi-group union or outside holdout inventory differs from partition proofs")
    return picked


def _selection(graph, schedule, nodes, replay_receipts, seals, terminal_to_leaf):
    if not isinstance(graph, ResponseGraph) or not isinstance(schedule, TreeSchedule):
        raise ValueError("typed graph and tree schedule required")
    graph.validate()
    if schedule.contract.sha256 != graph.contract.sha256:
        raise ValueError("sampler and training collection contracts differ")
    if not isinstance(terminal_to_leaf, Mapping):
        raise ValueError("explicit terminal-node to graph-leaf mapping required")
    selected = select_leaves(schedule, nodes, replay_receipts, seals=seals)
    terminal_ids = {n.node_id for n in nodes if n.terminal}
    mapping = dict(terminal_to_leaf)
    if (set(mapping) != terminal_ids or any(not isinstance(v, str) for v in mapping.values())
            or len(set(mapping.values())) != len(mapping)):
        raise ValueError("mapping must bijectively cover every generated terminal node")
    leaves, node_by_id = {l.leaf_id: l for l in graph.leaves}, {n.node_id: n for n in nodes}
    for terminal, leaf_id in mapping.items():
        leaf, node = leaves.get(leaf_id), node_by_id[terminal]
        if (leaf is None or leaf.split != "train"
                or (leaf.task_id, leaf.group_id, leaf.tree_id) !=
                   (schedule.task_id, schedule.group_id, node.tree_id)):
            raise ValueError("terminal mapping changes a task/group/tree or leaks holdout")
    # A successful outcome necessarily has a generated terminal artifact. Failed
    # attempts without such an artifact remain in the full inventory, not in K's
    # candidate pool. Failure labels never choose among existing terminal nodes.
    if any(l.split == "train" and l.status == "valid" and l.leaf_id not in mapping.values()
           for l in graph.leaves):
        raise ValueError("valid training leaf omitted from generated terminal inventory")
    picked = tuple(sorted(mapping[n] for s in selected for n in s.selected_leaf_ids))
    generated = tuple(sorted(mapping.values()))
    return generated, picked


def make_training_selection(graph: ResponseGraph, *, schedule: TreeSchedule,
                            nodes: tuple[SamplingNode, ...], replay_receipts,
                            seals: tuple[RoundSeal, ...], terminal_to_leaf) -> TrainingSelectionReceipt:
    """Freeze the complete sampler proof and explicit graph mapping, without pruning."""
    generated, picked = _selection(graph, schedule, nodes, replay_receipts, seals, terminal_to_leaf)
    return TrainingSelectionReceipt(schedule, tuple(sorted(nodes, key=lambda n: n.node_id)),
        tuple(replay_receipts[k] for k in sorted(replay_receipts)),
        tuple(sorted(seals, key=lambda s: s.round_index)), tuple(sorted(terminal_to_leaf.items())),
        inventory_sha256(graph), generated, picked)


def validate_selection(graph: ResponseGraph, selection: TrainingSelectionReceipt |
                       MultiGroupTrainingSelectionReceipt) -> tuple[str, ...]:
    """Recompute outcome-blind selection and return selected train IDs, even invalid ones."""
    from .action_credit_slot_selection import SlotSelectionReceipt, validate_slot_selection
    if type(selection) is SlotSelectionReceipt:
        return validate_slot_selection(graph, selection)
    if type(selection) is MultiGroupTrainingSelectionReceipt:
        return _validate_multigroup_selection(graph, selection)
    if not isinstance(selection, TrainingSelectionReceipt):
        raise ValueError("typed training selection receipt required")
    s = selection
    for value in (s.nodes, s.replay_receipts, s.seals, s.terminal_to_leaf,
                  s.generated_leaf_ids, s.selected_leaf_ids):
        if not isinstance(value, tuple):
            raise ValueError("selection proof must be immutable")
    if (any(not isinstance(r, ReplayReceipt) for r in s.replay_receipts)
            or len({r.edge_id for r in s.replay_receipts}) != len(s.replay_receipts)
            or any(not isinstance(pair, tuple) or len(pair) != 2 for pair in s.terminal_to_leaf)
            or len(dict(s.terminal_to_leaf)) != len(s.terminal_to_leaf)):
        raise ValueError("duplicate or malformed selection proof identities")
    if s.inventory_sha256 != inventory_sha256(graph):
        raise ValueError("complete training/holdout inventory changed after selection")
    generated, picked = _selection(graph, s.schedule, s.nodes,
        {r.edge_id: r for r in s.replay_receipts}, s.seals, dict(s.terminal_to_leaf))
    if s.generated_leaf_ids != generated or s.selected_leaf_ids != picked:
        raise ValueError("selected/generated IDs differ from the pinned outcome-blind sampler")
    return picked
