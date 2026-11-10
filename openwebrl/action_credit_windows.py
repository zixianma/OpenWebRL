"""Explicit PPO minibatches for the new five-arm controlled comparison.

Each epoch shuffles leaf/response OCCURRENCES globally with a pinned seed. Every
occurrence is kept exactly once; final windows retain denominator GBS (256 in
the pilot) with virtual zero-loss slots. A shared edge is forwarded once WITHIN
a window, but each leaf's nonlinear clipped term remains separate. Advantages
and teacher-forced old scores are frozen across the full two-epoch schedule.

Continuation-Q adds every eligible complete local panel to every window, using
its original equal-state/equal-candidate weights times effective_base_count/GBS.
This repeats extra candidate forwards, records their cost, and does not silently
invent panel subsampling or renormalize partial tails. This new protocol differs
from historical owner-balanced trimming. Only response_mean is supported here.

The one-window native executor does not claim durability. Checkpoint commit is
only after the entire collection/PPO2 update. A crash restores the previous
committed optimizer/scheduler/RNG and replays this exact pending schedule from
window zero. Neither an intent nor an in-memory cursor can skip a partial update.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import math
import random
from types import MappingProxyType

from openwebrl.action_credit import LossTerm, _sha, clipped_loss, digest


@dataclass(frozen=True)
class WindowConfig:
    epochs: int
    global_batch_size: int
    seed: int
    shuffle_policy: str = "global_occurrences_v1"
    tail_policy: str = "zero_pad_fixed_denominator"
    local_policy: str = "all_panels_each_window_scaled_by_base_fraction"
    checkpoint_policy: str = "full_collection_ppo_epochs"

    def validate(self):
        if (type(self.epochs) is not int or self.epochs <= 0 or type(self.global_batch_size) is not int
                or self.global_batch_size <= 0 or type(self.seed) is not int or self.seed < 0
                or self.shuffle_policy != "global_occurrences_v1" or self.tail_policy != "zero_pad_fixed_denominator"
                or self.local_policy != "all_panels_each_window_scaled_by_base_fraction"
                or self.checkpoint_policy != "full_collection_ppo_epochs"):
            raise ValueError("explicit global shuffle, fixed denominator, keep-all tail and complete local panels required")


@dataclass(frozen=True)
class TrainWindow:
    ordinal: int
    epoch: int
    index_in_epoch: int
    occurrence_ids: tuple[str, ...]
    virtual_padding_slots: int

    @property
    def effective_base_count(self):
        return len(self.occurrence_ids)

    def validate(self):
        if (any(type(x) is not int or x < 0 for x in
                (self.ordinal,self.epoch,self.index_in_epoch,self.virtual_padding_slots))
                or not isinstance(self.occurrence_ids,tuple) or not self.occurrence_ids
                or len(set(self.occurrence_ids)) != len(self.occurrence_ids)):
            raise ValueError("malformed occurrence window")
        for key in self.occurrence_ids: _sha(key,"occurrence_id")


def occurrence_id(term):
    return digest((term.component, term.source_id, term.edge_id))


def objective_sha256(prepared):
    """Bind exact immutable inputs/targets without building one huge JSON blob."""
    plan = prepared.plan
    return digest(dict(config=asdict(prepared.config), contract=plan.graph.contract.sha256,
        edges=sorted((e.edge_id,digest(asdict(e))) for e in plan.graph.edges),
        leaves=sorted((l.leaf_id,digest(asdict(l))) for l in plan.graph.leaves),
        planned_leaf_ids=sorted(plan.graph.planned_leaf_ids), selected_leaf_ids=plan.selected_leaf_ids,
        terms=sorted((occurrence_id(t),digest(asdict(t))) for t in plan.terms)))


def _windows(prepared, config):
    ids = sorted(occurrence_id(t) for t in prepared.plan.terms if t.component == "outcome")
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("distinct nonempty leaf/response occurrences required")
    windows = []
    for epoch in range(config.epochs):
        order = ids[:]
        seed = int(digest((config.seed, prepared.plan.graph.contract.update_id, epoch,
                           config.shuffle_policy)), 16)
        random.Random(seed).shuffle(order)
        for start in range(0, len(order), config.global_batch_size):
            occurrence_ids = tuple(order[start:start+config.global_batch_size])
            windows.append(TrainWindow(len(windows),epoch,start//config.global_batch_size,occurrence_ids,
                                       config.global_batch_size-len(occurrence_ids)))
    return tuple(windows)


@dataclass(frozen=True)
class PPOWindowPlan:
    config: WindowConfig
    objective_sha256: str
    windows: tuple[TrainWindow, ...]

    @property
    def sha256(self):
        return digest(asdict(self))

    def validate(self, prepared):
        if not isinstance(self.config,WindowConfig): raise ValueError("typed window configuration required")
        self.config.validate()
        prepared.config.validate()
        prepared.plan.validate()
        if prepared.plan.reduction != "response_mean":
            raise ValueError("PPO window protocol requires explicit response_mean reduction")
        if not isinstance(self.windows,tuple) or self.objective_sha256 != objective_sha256(prepared):
            raise ValueError("PPO schedule source or frozen advantages/old scores changed")
        for window in self.windows:
            if not isinstance(window,TrainWindow): raise ValueError("typed occurrence window required")
            window.validate()
        if self.windows != _windows(prepared,self.config):
            raise ValueError("PPO schedule changed membership, order, coverage, padding or epoch")


def plan_ppo_windows(prepared, config):
    if not isinstance(config,WindowConfig): raise ValueError("typed window configuration required")
    config.validate()
    result = PPOWindowPlan(config,objective_sha256(prepared),_windows(prepared,config))
    result.validate(prepared)
    return result


@dataclass(frozen=True)
class WindowLossPlan:
    """Immutable validated term view; validation cached once per native window.

    The parent graph/terms/config are frozen dataclasses of immutable tuples.
    Revalidating millions of context token IDs at EVERY edge forward is avoided.
    dataclasses.replace constructs and validates a new view; no mutable tensor or
    external cache participates in this target/weight cache.
    """
    prepared: object
    schedule: PPOWindowPlan
    ordinal: int
    _terms: tuple = field(init=False,repr=False)
    _edges: object = field(init=False,repr=False,compare=False)

    def __post_init__(self):
        self.schedule.validate(self.prepared)
        if type(self.ordinal) is not int or not 0 <= self.ordinal < len(self.schedule.windows):
            raise ValueError("window ordinal is outside the pinned PPO plan")
        window = self.schedule.windows[self.ordinal]
        outcomes = {occurrence_id(t):t for t in self.prepared.plan.terms if t.component == "outcome"}
        gbs = self.schedule.config.global_batch_size
        terms = tuple(replace(outcomes[key],weight=1/gbs) for key in window.occurrence_ids)
        terms += tuple(replace(t,weight=t.weight*window.effective_base_count/gbs)
                       for t in self.prepared.plan.terms if t.component == "local")
        object.__setattr__(self,"_terms",terms)
        object.__setattr__(self,"_edges",MappingProxyType(self.prepared.plan.validate()))

    @property
    def graph(self): return self.prepared.plan.graph

    @property
    def terms(self): return self._terms

    @property
    def arm_id(self): return self.prepared.plan.arm_id

    @property
    def reduction(self): return "response_mean"

    def validate(self): return self._edges


@dataclass(frozen=True)
class WindowObjective:
    plan: WindowLossPlan

    @property
    def config(self): return self.plan.prepared.config

    def loss(self, log_probs, *, edge_subset=None):
        return clipped_loss(self.plan,log_probs,clip_low=self.config.clip_low,
            clip_high=self.config.clip_high,dual_clip=self.config.dual_clip,edge_subset=edge_subset)

    @property
    def accounting(self):
        window = self.plan.schedule.windows[self.plan.ordinal]
        base_edges = {t.edge_id for t in self.plan.terms if t.component == "outcome"}
        local_edges = {t.edge_id for t in self.plan.terms if t.component == "local"}
        all_edges = base_edges|local_edges
        return dict(plan_sha256=self.plan.schedule.sha256,ordinal=window.ordinal,epoch=window.epoch,
            base_occurrences=window.effective_base_count,virtual_padding_slots=window.virtual_padding_slots,
            global_denominator=self.plan.schedule.config.global_batch_size,
            local_terms=sum(t.component=="local" for t in self.plan.terms),
            unique_forwards=len(all_edges),shared_base_forward_savings=window.effective_base_count-len(base_edges),
            additional_local_forwards=len(local_edges-base_edges),
            forward_input_tokens=sum(len(self.plan._edges[e].input_ids)+len(self.plan._edges[e].response_ids) for e in all_edges))


def build_window_objective(prepared,schedule,ordinal):
    return WindowObjective(WindowLossPlan(prepared,schedule,ordinal))


@dataclass(frozen=True)
class PPOResumeCursor:
    """Whole-update boundary bindings from an inspected checkpoint, not intent."""
    plan_sha256: str
    next_window: int
    checkpoint_sha256: str
    optimizer_state_sha256: str
    scheduler_state_sha256: str
    rng_state_sha256: str
    validator_evidence_sha256: str

    def validate(self,schedule):
        for name in ("plan_sha256","checkpoint_sha256","optimizer_state_sha256",
                     "scheduler_state_sha256","rng_state_sha256","validator_evidence_sha256"):
            _sha(getattr(self,name),name)
        if (self.plan_sha256 != schedule.sha256 or type(self.next_window) is not int
                or self.next_window not in (0,len(schedule.windows))):
            raise ValueError("durable cursor does not match exact pending PPO schedule")


def native_train_window(actor,window,rows,tensor_payloads):
    """Execute ONE pinned window. Caller commits checkpoint/cursor/RNG later.

    A returned result is in-memory progress only; it is not a resume receipt.
    The same fixed old scores are used in both epochs, including repeat forwards.
    """
    from megatron.core import mpu
    from slime.backends.megatron_utils.model import train_one_step
    from openwebrl.action_credit_megatron import native_train_iterator,configure_native_training
    if not isinstance(window,WindowObjective):
        raise ValueError("use a validated occurrence-window objective")
    if (actor.args.global_batch_size != window.plan.schedule.config.global_batch_size
            or actor.args.ppo_epochs != window.plan.schedule.config.epochs):
        raise ValueError("native optimizer configuration differs from pinned PPO schedule")
    configure_native_training(actor)
    iterator = native_train_iterator(actor,window,rows,tensor_payloads)
    before = actor.opt_param_scheduler.num_steps
    metrics,norm = train_one_step(actor.args,window.plan.graph.contract.update_id,window.plan.ordinal,
        [iterator],actor.model,actor.optimizer,actor.opt_param_scheduler,len(iterator.slots))
    if (not math.isfinite(float(norm)) or actor.opt_param_scheduler.num_steps-before != actor.args.global_batch_size):
        raise ValueError("native window failed to complete one optimizer/scheduler update")
    dp = mpu.get_data_parallel_world_size(False)
    return dict(window.accounting,grad_norm=float(norm),native_metrics=metrics,
        native_microbatches_per_rank=len(iterator.slots),dp_padding_forwards=len(iterator.slots)*dp-window.accounting["unique_forwards"],
        optimizer_update_in_memory=True,durable_checkpoint_committed=False,
        scheduler_steps_before=before,scheduler_steps_after=actor.opt_param_scheduler.num_steps,
        next_window_in_memory=window.plan.ordinal+1,resume_window_after_failure=0)


def native_train_collection(actor,prepared,schedule,rows,tensor_payloads,*,before_window=None,after_window=None):
    """Run the entire persisted schedule from its committed starting checkpoint.

    Hooks are for collective validation/heartbeats, never cursor authorization.
    On ANY failure the caller must restore the previous full-update checkpoint
    and replay this same saved batch/schedule from window zero. Saving, artifact
    review and campaign commit occur outside this function, after it succeeds.
    """
    schedule.validate(prepared)
    reports=[]
    for entry in schedule.windows:
        window=build_window_objective(prepared,schedule,entry.ordinal)
        if before_window is not None: before_window(window)
        report=native_train_window(actor,window,rows,tensor_payloads)
        reports.append(report)
        if after_window is not None: after_window(window,report)
    schedule.validate(prepared)
    return dict(plan_sha256=schedule.sha256,epochs=schedule.config.epochs,
        optimizer_updates_in_memory=len(reports),durable_checkpoint_committed=False,
        checkpoint_policy=schedule.config.checkpoint_policy,resume_window_after_failure=0,
        base_occurrence_exposures=sum(r["base_occurrences"] for r in reports),
        virtual_padding_slots=sum(r["virtual_padding_slots"] for r in reports),
        unique_forward_exposures=sum(r["unique_forwards"] for r in reports),
        additional_local_forwards=sum(r["additional_local_forwards"] for r in reports),windows=reports)
