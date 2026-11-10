"""Offline campaign pins and durable all-attempt accounting for action credit.

This module performs no scheduler, browser, model or API execution. Reservations
are upper bounds, never execution authorization. Unsettled and failed attempts
keep their complete reservation; a retry is a new, independently charged attempt.
The store uses one atomically replaced, fsynced snapshot under a process lock.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import tempfile


ARM_IDS = frozenset({"chain_outcome_grpo", "tree_outcome_sampling", "tree_grpo",
                     "continuation_q", "gigpo"})
COST_DIMENSIONS = ("gpu_seconds", "api_microdollars", "api_requests", "browser_dispatches")
READINESS_GATES = ("collected_policy_version", "browser_replay", "token_masks",
                   "tp2dp4_restore", "save_resume", "production_integration")
_FORBIDDEN = Path("/gpfs/home/zixianma")


class StateError(ValueError):
    """An identity, scientific pin or durable-state invariant was violated."""


class BudgetExceeded(StateError):
    """An explicit cap or reservation was exceeded; no new work may start."""


def _canonical(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise StateError("State must contain finite JSON values") from exc


def digest(value):
    """SHA256 of canonical finite JSON, including all scientific settings."""
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _require(condition, message):
    if not condition:
        raise StateError(message)


def _ident(value, name):
    _require(isinstance(value, str) and 0 < len(value) <= 256 and value.strip() == value,
             f"{name} must be an explicit nonempty identity")
    return value


def _sha(value, name):
    _require(isinstance(value, str) and len(value) == 64
             and all(c in "0123456789abcdef" for c in value), f"{name} must be SHA256")
    return value


def _integer(value, name, minimum=0):
    _require(type(value) is int and value >= minimum, f"{name} must be an integer >= {minimum}")
    return value


def _cost(value):
    _require(isinstance(value, dict) and set(value) == set(COST_DIMENSIONS),
             "Every resource dimension must be explicit; no inferred budget defaults")
    return {key: _integer(value[key], key) for key in COST_DIMENSIONS}


def _safe_path(path):
    path = Path(path).absolute()
    resolved = path.resolve(strict=False)
    for candidate in (path, resolved):
        _require(candidate != _FORBIDDEN and _FORBIDDEN not in candidate.parents,
                 "Writes under /gpfs/home/zixianma are forbidden")
    for candidate in (path, *path.parents):
        _require(not candidate.is_symlink(), "Campaign paths must not contain symlinks")
    return path


@dataclass(frozen=True)
class CampaignManifest:
    """An immutable, validated JSON manifest; callers must specify every cap."""

    canonical_json: str

    def __post_init__(self):
        try:
            value = json.loads(self.canonical_json)
        except (TypeError, ValueError) as exc:
            raise StateError("Invalid manifest JSON") from exc
        self._validate(value)
        object.__setattr__(self, "canonical_json", _canonical(value))

    @classmethod
    def from_dict(cls, value):
        return cls(_canonical(value))

    def to_dict(self):
        return json.loads(self.canonical_json)

    @property
    def sha256(self):
        return hashlib.sha256(self.canonical_json.encode()).hexdigest()

    @property
    def protocol_sha256(self):
        return digest(self.to_dict()["science"])

    @staticmethod
    def _validate(m):
        _require(isinstance(m, dict), "Manifest must be an object")
        required = {"schema_version", "campaign_id", "arm_id", "initialization", "science",
                    "task_splits", "split_sha256", "code_revision_sha256", "budget_caps",
                    "max_attempts_per_draw"}
        _require(set(m) == required, "Manifest fields must match the versioned schema")
        _require(type(m["schema_version"]) is int and m["schema_version"] == 1, "Unknown manifest version")
        _ident(m["campaign_id"], "campaign_id")
        _require(m["arm_id"] in ARM_IDS, "Unknown action-credit arm")
        init = m["initialization"]
        _require(isinstance(init, dict) and set(init) == {"original_actor_sha256", "iteration",
                 "fresh_optimizer", "fresh_scheduler", "fresh_cursor"}, "Incomplete initialization pins")
        _sha(init["original_actor_sha256"], "original_actor_sha256")
        _require(type(init["iteration"]) is int and init["iteration"] == 0,
                 "New experiments must start at original actor iteration0")
        _require(all(init[k] is True for k in ("fresh_optimizer", "fresh_scheduler", "fresh_cursor")),
                 "New experiments require fresh optimizer, scheduler and cursor")
        _sha(m["code_revision_sha256"], "code_revision_sha256")
        splits = m["task_splits"]
        _require(isinstance(splits, dict) and bool(splits), "Pin explicit task split membership")
        for task, split in splits.items():
            _ident(task, "task_id")
            _require(split in ("train", "holdout"), "Unknown task split")
        _require(m["split_sha256"] == digest(splits), "Task split digest mismatch")
        science = m["science"]
        _require(isinstance(science, dict) and set(science) == {"checkpoint", "dataset", "judge",
                 "generation", "token_mask", "shared_prefix_normalization", "algorithm"},
                 "All scientific protocols must be pinned")
        for key, value in science.items():
            _require(isinstance(value, dict) and bool(value), f"Missing {key} settings")
        for key in ("checkpoint", "token_mask", "shared_prefix_normalization", "algorithm"):
            _sha(science[key].get("protocol_sha256"), key + ".protocol_sha256")
        _sha(science["dataset"].get("source_sha256"), "dataset.source_sha256")
        _ident(science["judge"].get("model"), "judge.model")
        _sha(science["judge"].get("prompt_sha256"), "judge.prompt_sha256")
        gen = science["generation"]
        for key in ("temperature", "top_p"):
            val = gen.get(key)
            _require(type(val) in (int, float) and math.isfinite(val), f"Explicit finite {key} required")
        _require(gen["temperature"] >= 0 and 0 < gen["top_p"] <= 1, "Invalid generation sampling settings")
        _require(type(gen.get("top_k")) is int and (gen["top_k"] == -1 or gen["top_k"] > 0), "Invalid top_k")
        for key in ("max_response_tokens", "max_turns"):
            _integer(gen.get(key), key, 1)
        _cost(m["budget_caps"])
        _integer(m["max_attempts_per_draw"], "max_attempts_per_draw", 1)


class CampaignStore:
    """A single-arm ledger. Creation is explicit; missing state never resets usage.

    Caller-supplied request/attempt IDs are idempotency keys, not automatic retry
    instructions. A returned existing attempt must never be dispatched again.
    Actual costs above a reservation are persisted before BudgetExceeded is raised.
    """

    def __init__(self, root, manifest):
        _require(isinstance(manifest, CampaignManifest), "Validated CampaignManifest required")
        self.root = _safe_path(root)
        self.manifest = manifest
        _require(self.root.is_dir(), "Campaign does not exist; use create for a new directory")
        with self._lock():
            self._read()

    @classmethod
    def create(cls, root, manifest):
        _require(isinstance(manifest, CampaignManifest), "Validated CampaignManifest required")
        root = _safe_path(root)
        # Existing, partially initialized directories are never silently recycled.
        root.mkdir(exist_ok=False)
        # Persist the directory entry as well as the ledger inside it. Parents
        # must already exist, so creation cannot leave an unflushed directory chain.
        parent = os.open(root.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
        instance = object.__new__(cls)
        instance.root, instance.manifest = root, manifest
        with instance._lock():
            instance._write({"schema_version": 1, "manifest_sha256": manifest.sha256,
                "manifest": manifest.to_dict(), "revision": 0, "updates": {}, "requests": {},
                "draws": {}, "attempts": {}, "readiness": {}, "halt": None})
        return cls(root, manifest)

    @contextmanager
    def _lock(self):
        _safe_path(self.root)
        lock = _safe_path(self.root / "state.lock")
        fd = os.open(lock, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "a") as stream:
            metadata = os.fstat(stream.fileno())
            _require(stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1,
                     "Lock must be an unaliased regular file")
            fcntl.flock(stream, fcntl.LOCK_EX)
            yield

    def _read(self):
        path = _safe_path(self.root / "state.json")
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(fd) as stream:
                metadata = os.fstat(stream.fileno())
                _require(stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1,
                         "State must be an unaliased regular file")
                envelope = json.load(stream)
            state = envelope["state"]
            _require(envelope["state_sha256"] == digest(state), "State checksum mismatch")
            _require(state["schema_version"] == 1, "Unknown durable state version")
            _require(state["manifest_sha256"] == self.manifest.sha256
                     and state["manifest"] == self.manifest.to_dict(), "Campaign scientific pins or caps changed")
            _integer(state["revision"], "revision")
            return state
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise StateError("Missing, corrupt or incompatible campaign state; refusing reset") from exc

    def _write(self, state):
        _safe_path(self.root)
        target = _safe_path(self.root / "state.json")
        payload = _canonical({"state_sha256": digest(state), "state": state}) + "\n"
        fd, name = tempfile.mkstemp(dir=self.root, prefix="state.", suffix=".tmp")
        with os.fdopen(fd, "w") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        # Retain an interrupted temporary snapshot; it never replaces committed state.
        _safe_path(target)
        os.replace(name, target)
        directory = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    @contextmanager
    def _transaction(self):
        with self._lock():
            state = self._read()
            before = digest(state)
            yield state
            if digest(state) != before:
                state["revision"] += 1
                self._write(state)

    @staticmethod
    def _totals(state):
        # HTTP retries are separately reserved before each actual transport
        # call. Their parent episode carries no duplicate API reservation.
        entries = (*state["attempts"].values(), *state.get("subrequests", {}).values(),
                   *state.get("evaluation_attempts", {}).values())
        return {key: sum(a["charged"][key] for a in entries)
                for key in COST_DIMENSIONS}

    def reserve_subrequest(self, request_id, *, parent_attempt_id, payload_sha256, reserved_cost):
        """Reserve exactly one judge HTTP attempt; created=False forbids send.

        This ledger operation grants no API permission. The caller separately
        binds an approved provider/model/pricing/output-limit protocol. A crash
        retains the complete reservation, including unknown provider charges.
        """
        _ident(request_id, "request_id")
        _ident(parent_attempt_id, "parent_attempt_id")
        _sha(payload_sha256, "payload_sha256")
        reserved = _cost(reserved_cost)
        _require(reserved["api_requests"] == 1 and reserved["gpu_seconds"] == 0
                 and reserved["browser_dispatches"] == 0,
                 "Judge subrequest reserves one HTTP attempt and explicit API cost only")
        identity = dict(parent_attempt_id=parent_attempt_id, payload_sha256=payload_sha256,
                        reserved=reserved)
        with self._transaction() as state:
            calls = state.setdefault("subrequests", {})
            if request_id in calls:
                _require(all(calls[request_id][k] == v for k, v in identity.items()),
                         "Subrequest reused with different payload/parent/reservation")
                return dict(created=False, request_id=request_id, status=calls[request_id]["status"])
            _require(not state["halt"], "Campaign halted")
            parent = (state["attempts"].get(parent_attempt_id)
                      or state.get("evaluation_attempts", {}).get(parent_attempt_id))
            _require(parent is not None and parent["status"] == "active", "Active parent attempt required")
            _require(parent["reserved"]["api_requests"] == parent["reserved"]["api_microdollars"] == 0,
                     "Per-HTTP accounting requires zero duplicate episode API reservation")
            caps = self.manifest.to_dict()["budget_caps"]
            total = self._totals(state)
            if any(total[k] + reserved[k] > caps[k] for k in COST_DIMENSIONS):
                raise BudgetExceeded("HTTP reservation exceeds explicit remaining resource caps")
            calls[request_id] = dict(identity, charged=reserved.copy(), actual=None,
                                     status="reserved", completion=None)
        return dict(created=True, request_id=request_id, status="reserved")

    def settle_subrequest(self, request_id, *, status, actual_cost, response_sha256=None, reason=None):
        """Charge metered successes exactly; retain failed/unknown liabilities.

        Releasing an unused successful-response reservation never refunds a
        consumed token or HTTP attempt. Failed or ambiguous requests retain
        their full reservation until an explicit external reconciliation.
        """
        _ident(request_id, "request_id")
        _require(status in ("succeeded", "failed", "unknown"), "Explicit HTTP disposition required")
        actual = _cost(actual_cost)
        _require(actual["api_requests"] == 1 and actual["gpu_seconds"] == actual["browser_dispatches"] == 0,
                 "HTTP actual cost must account for its one transport attempt")
        if status == "succeeded":
            _sha(response_sha256, "response_sha256")
            _require(reason is None, "Successful HTTP response cannot have a failure reason")
        else:
            _ident(reason, "HTTP failure/unknown reason")
            if response_sha256 is not None:
                _sha(response_sha256, "response_sha256")
        completion = dict(status=status, actual_cost=actual,
                          response_sha256=response_sha256, reason=reason)
        exceeded = False
        with self._transaction() as state:
            call = state.get("subrequests", {}).get(request_id)
            _require(call is not None, "Unknown HTTP reservation")
            if call["completion"] is not None:
                _require(call["completion"] == completion, "Settled HTTP attempt cannot be rewritten")
                return dict(recorded=False, request_id=request_id)
            charged = (actual.copy() if status == "succeeded" else
                       {k: max(actual[k], call["reserved"][k]) for k in COST_DIMENSIONS})
            call.update(status=status, actual=actual, completion=completion, charged=charged)
            caps = self.manifest.to_dict()["budget_caps"]
            total = self._totals(state)
            exceeded = (any(total[k] > caps[k] for k in COST_DIMENSIONS)
                        or any(actual[k] > call["reserved"][k] for k in COST_DIMENSIONS))
            if exceeded:
                state["halt"] = "Actual HTTP cost exceeded its reservation or approved resource caps"
        if exceeded:
            raise BudgetExceeded("Actual HTTP overrun recorded; campaign halted")
        return dict(recorded=True, request_id=request_id)

    def snapshot(self):
        with self._lock():
            state = self._read()
            state["charged_resources"] = self._totals(state)
            return state

    def register_evaluation_cohort(self, cohort_id, request_ref):
        from .action_credit_eval_budget import register_cohort
        return register_cohort(self, cohort_id, request_ref)

    def start_evaluation_attempt(self, cohort_id, task_id, attempt_id):
        from .action_credit_eval_budget import start_attempt
        return start_attempt(self, cohort_id, task_id, attempt_id)

    def finish_evaluation_attempt(self, attempt_id, *, status, outcome=None, reason=None):
        from .action_credit_eval_budget import finish_attempt
        return finish_attempt(self, attempt_id, status=status, outcome=outcome, reason=reason)

    def reserve_evaluation_actor_request(self, actor_attempt_id, *, parent_attempt_id, task_id,
                                         policy_sha256, payload_sha256, turn):
        from .action_credit_eval_budget import reserve_actor_request
        return reserve_actor_request(self, actor_attempt_id, parent_attempt_id=parent_attempt_id,
            task_id=task_id, policy_sha256=policy_sha256, payload_sha256=payload_sha256, turn=turn)

    def settle_evaluation_actor_request(self, actor_attempt_id, *, status, response_sha256=None, reason=None):
        from .action_credit_eval_budget import settle_actor_request
        return settle_actor_request(self, actor_attempt_id, status=status,
                                    response_sha256=response_sha256, reason=reason)

    def _draw_identity(self, update_id, policy_sha256, spec):
        required = {"task_id", "state_id", "action_id", "draw_slot", "draw_role", "group_id", "tree_id"}
        _require(isinstance(spec, dict) and set(spec) in (required, required | {"inventory_role"}),
                 "Incomplete planned draw identity")
        spec = dict(spec)
        arm = self.manifest.to_dict()["arm_id"]
        _require(arm != "continuation_q" or "inventory_role" in spec,
                 "Continuation-Q must explicitly prescribe base/q_fit/q_holdout inventory roles")
        # Published non-Q fixtures omitted this field. Their only supported
        # population is base, never an implicitly inferred Q continuation.
        spec.setdefault("inventory_role", "base")
        for key in ("task_id", "state_id", "action_id", "group_id", "tree_id"):
            _ident(spec[key], key)
        _integer(spec["draw_slot"], "draw_slot")
        role = spec["draw_role"]
        _require(role in ("train", "holdout"), "Unknown continuation draw role")
        inventory_role = spec["inventory_role"]
        _require(inventory_role in ("base", "q_fit", "q_holdout"), "Unknown inventory role")
        _require(inventory_role == "base" or arm == "continuation_q",
                 "Only continuation-Q may prescribe Q-estimation draws")
        _require(inventory_role != "q_fit" or role == "train", "Q-fit must have train draw role")
        _require(inventory_role != "q_holdout" or role == "holdout", "Q-holdout must have holdout draw role")
        task_split = self.manifest.to_dict()["task_splits"].get(spec["task_id"])
        _require(task_split in ("train", "holdout"), "Task absent from frozen membership")
        _require(task_split != "holdout" or role == "holdout", "Heldout tasks cannot supply training draws")
        return dict(update_id=update_id, policy_sha256=policy_sha256, task_split=task_split, **spec)

    def plan_update(self, update_id, *, policy_sha256, draws):
        """Freeze every candidate/draw slot, grouping and role before collection.

        A train task may have heldout continuation draws. A task in the heldout
        task partition can only have heldout draws. Roles cannot change on retry.
        """
        _integer(update_id, "update_id")
        _sha(policy_sha256, "policy_sha256")
        _require(isinstance(draws, list) and bool(draws), "An explicit nonempty draw panel is required")
        if update_id == 0:
            _require(policy_sha256 == self.manifest.to_dict()["initialization"]["original_actor_sha256"],
                     "Update0 must use the original actor")
        expected, slots = {}, set()
        for spec in draws:
            identity = self._draw_identity(update_id, policy_sha256, spec)
            slot = (identity["task_id"], identity["state_id"], identity["action_id"], identity["draw_slot"])
            _require(slot not in slots, "One draw slot cannot have duplicate or conflicting roles")
            slots.add(slot)
            expected[digest(identity)] = identity
        plan_sha = digest(expected)
        with self._transaction() as state:
            previous = state["updates"].get(str(update_id))
            if previous is not None:
                _require(previous["policy_sha256"] == policy_sha256 and previous["plan_sha256"] == plan_sha,
                         "An update's policy, candidates, slots, groups, trees and roles are immutable")
                return dict(created=False, plan_sha256=plan_sha, draw_ids=sorted(expected))
            _require(not state["halt"], "Campaign halted")
            last = max((int(key) for key in state["updates"]), default=-1)
            _require(update_id == last + 1, "A new campaign begins at update0 and cannot skip policy updates")
            if last >= 0:
                prior = state["updates"][str(last)]
                _require(prior["phase"] == "consumed",
                         "Reconcile the previous training transaction before advancing")
                _require(prior["training_receipt"]["new_policy_sha256"] == policy_sha256,
                         "Next collection policy must match the committed checkpoint receipt")
            state["updates"][str(update_id)] = dict(policy_sha256=policy_sha256,
                plan_sha256=plan_sha, expected_draws=expected, phase="collecting", export_sha256=None,
                training_intent=None, training_receipt=None)
        return dict(created=True, plan_sha256=plan_sha, draw_ids=sorted(expected))

    def reserve_draw(self, request_id, *, update_id, policy_sha256, task_id, state_id,
                     action_id, draw_slot, draw_role, group_id, tree_id, inventory_role=None):
        """Persist one slot from the frozen panel before any attempt."""
        _ident(request_id, "request_id")
        _integer(update_id, "update_id")
        _sha(policy_sha256, "policy_sha256")
        spec = dict(task_id=task_id,
            state_id=state_id, action_id=action_id, draw_slot=draw_slot, draw_role=draw_role,
            group_id=group_id, tree_id=tree_id)
        if inventory_role is not None:
            spec["inventory_role"] = inventory_role
        identity = self._draw_identity(update_id, policy_sha256, spec)
        draw_id = digest(identity)
        with self._transaction() as state:
            if request_id in state["requests"]:
                _require(state["requests"][request_id] == draw_id, "Request ID reused with different draw")
                return dict(draw_id=draw_id, created=False)
            _require(not state["halt"], "Campaign halted")
            update = state["updates"].get(str(update_id))
            _require(update is not None and update["policy_sha256"] == policy_sha256,
                     "Pin the complete update panel and policy before collecting")
            _require(update["phase"] == "collecting", "Sealed or consumed panels cannot change")
            _require(update["expected_draws"].get(draw_id) == identity, "Unplanned draw or changed role")
            _require(draw_id not in state["draws"], "Draw slot already belongs to another request")
            state["requests"][request_id] = draw_id
            state["draws"][draw_id] = dict(identity, request_id=request_id, status="pending",
                                           attempts=[], selected_attempt=None, missing_reason=None)
        return dict(draw_id=draw_id, created=True)

    def start_attempt(self, draw_id, attempt_id, *, reserved_cost):
        """Durably charge before dispatch; created=False forbids redispatch."""
        _ident(attempt_id, "attempt_id")
        reserved = _cost(reserved_cost)
        with self._transaction() as state:
            _require(attempt_id not in state.get("evaluation_attempts", {}),
                     "Attempt ID already belongs to evaluation")
            if attempt_id in state["attempts"]:
                old = state["attempts"][attempt_id]
                _require(old["draw_id"] == draw_id and old["reserved"] == reserved,
                         "Attempt ID reused with different request or reservation")
                return dict(attempt_id=attempt_id, created=False, status=old["status"])
            _require(not state["halt"], "Campaign halted")
            _require(draw_id in state["draws"], "Unknown draw")
            draw = state["draws"][draw_id]
            _require(state["updates"][str(draw["update_id"])]["phase"] == "collecting",
                     "Sealed or consumed panels cannot be retried")
            _require(draw["status"] != "valid", "Completed valid draw cannot be retried")
            _require(not any(state["attempts"][a]["status"] == "active" for a in draw["attempts"]),
                     "Resolve the existing attempt explicitly before retrying")
            _require(len(draw["attempts"]) < self.manifest.to_dict()["max_attempts_per_draw"],
                     "Draw attempt cap reached")
            totals = self._totals(state)
            caps = self.manifest.to_dict()["budget_caps"]
            if any(totals[k] + reserved[k] > caps[k] for k in COST_DIMENSIONS):
                raise BudgetExceeded("Reservation exceeds explicit remaining resource caps")
            state["attempts"][attempt_id] = dict(draw_id=draw_id, reserved=reserved,
                charged=reserved.copy(), actual=None, status="active", completion=None)
            draw["attempts"].append(attempt_id)
            draw["status"] = "pending"
        return dict(attempt_id=attempt_id, created=True, status="active")

    def finish_attempt(self, attempt_id, *, status, actual_cost, outcome=None, reason=None,
                       sampling_terminal_node_id=None):
        """Record valid/invalid/missing outcomes and all costs, even after a halt."""
        _ident(attempt_id, "attempt_id")
        _require(status in ("valid", "invalid", "missing"), "Explicit outcome status required")
        actual = _cost(actual_cost)
        if status == "valid":
            _require(isinstance(outcome, dict) and set(outcome) == {"artifact_sha256", "reward"},
                     "Valid outcomes require artifact identity and reward")
            _sha(outcome["artifact_sha256"], "outcome.artifact_sha256")
            reward = outcome["reward"]
            _require(type(reward) in (int, float) and math.isfinite(reward), "Reward must be finite")
            _require(reason is None, "Valid result must not carry an invalid reason")
        else:
            _ident(reason, "invalid_or_missing_reason")
            if status == "invalid" and outcome is not None:
                _require(isinstance(outcome, dict) and set(outcome) == {"artifact_sha256"},
                         "Invalid outcomes may preserve artifacts but never a reward")
                _sha(outcome["artifact_sha256"], "invalid.artifact_sha256")
            else:
                _require(outcome is None, "Missing draws must not supply an outcome")
        if sampling_terminal_node_id is not None:
            _ident(sampling_terminal_node_id, "sampling_terminal_node_id")
            _require(status != "missing" and outcome is not None,
                     "A sampled terminal node requires a valid/invalid saved artifact, never missing")
        completion = dict(status=status, actual_cost=actual, outcome=outcome, reason=reason,
                          sampling_terminal_node_id=sampling_terminal_node_id)
        # Freeze arguments before committing; caller mutations cannot alter state.
        completion = json.loads(_canonical(completion))
        exceeded = False
        with self._transaction() as state:
            _require(attempt_id in state["attempts"], "Unknown attempt")
            attempt = state["attempts"][attempt_id]
            children = [x for x in state.get("subrequests", {}).values()
                        if x["parent_attempt_id"] == attempt_id]
            _require(not any(x["status"] == "reserved" for x in children),
                     "Resolve every dispatched HTTP request before closing its episode")
            _require(not children or actual["api_requests"] == actual["api_microdollars"] == 0,
                     "Episode actual cost must exclude separately recorded HTTP costs")
            if attempt["completion"] is not None:
                _require(attempt["completion"] == completion, "Completed attempt cannot be rewritten")
                return dict(attempt_id=attempt_id, recorded=False)
            attempt.update(actual=actual, status=status, completion=completion,
                           charged={k: max(actual[k], attempt["reserved"][k]) for k in COST_DIMENSIONS})
            draw = state["draws"][attempt["draw_id"]]
            draw.update(status=status, selected_attempt=attempt_id)
            exceeded = any(actual[k] > attempt["reserved"][k] for k in COST_DIMENSIONS)
            if exceeded and state["halt"] is None:
                state["halt"] = dict(reason="reservation_exceeded", attempt_id=attempt_id)
        if exceeded:
            raise BudgetExceeded("Observed overrun persisted; campaign halted")
        return dict(attempt_id=attempt_id, recorded=True)

    def mark_missing_draw(self, draw_id, *, reason):
        """Explicitly close an unattempted planned slot without inventing a reward."""
        _ident(reason, "missing_reason")
        with self._transaction() as state:
            _require(draw_id in state["draws"], "Reserve the planned slot before marking it missing")
            draw = state["draws"][draw_id]
            if draw["status"] == "missing" and not draw["attempts"]:
                _require(draw["missing_reason"] == reason, "Missing-slot evidence is immutable")
                return
            _require(state["updates"][str(draw["update_id"])]["phase"] == "collecting",
                     "Sealed or consumed panels cannot change")
            _require(draw["status"] == "pending" and not draw["attempts"],
                     "Attempted slots must settle their charged attempt explicitly")
            draw.update(status="missing", missing_reason=reason)

    @staticmethod
    def _training_rows(state, update_id):
        update = state["updates"].get(str(update_id))
        _require(update is not None, "Unknown update")
        rows = []
        for key, identity in sorted(update["expected_draws"].items()):
            _require(key in state["draws"], "Incomplete planned panel: unreserved slot")
            draw = state["draws"][key]
            _require(draw["status"] in ("valid", "invalid", "missing"),
                     "Incomplete planned panel: unresolved slot")
            if identity["draw_role"] != "train":
                continue
            _require(identity["task_split"] == "train", "Heldout task cannot train")
            selected = draw["selected_attempt"]
            completion = state["attempts"][selected]["completion"] if selected else None
            rows.append(dict(draw_id=key, update_id=update_id, policy_sha256=draw["policy_sha256"],
                task_id=draw["task_id"], state_id=draw["state_id"], action_id=draw["action_id"],
                group_id=draw["group_id"], tree_id=draw["tree_id"],
                draw_slot=draw["draw_slot"], draw_role=draw["draw_role"],
                inventory_role=draw["inventory_role"], status=draw["status"],
                sampling_terminal_node_id=completion.get("sampling_terminal_node_id") if completion else None,
                valid=draw["status"] == "valid", outcome=completion["outcome"] if completion else None,
                reason=completion["reason"] if completion else draw["missing_reason"]))
        return rows

    def seal_update(self, update_id):
        """Seal only a complete panel, including explicit invalid/missing slots."""
        _integer(update_id, "update_id")
        with self._transaction() as state:
            rows = self._training_rows(state, update_id)
            update = state["updates"][str(update_id)]
            export_sha = digest(rows)
            if update["phase"] in ("sealed", "training_pending", "consumed"):
                _require(update["export_sha256"] == export_sha, "Sealed export changed")
                return dict(created=False, export_sha256=export_sha)
            update.update(phase="sealed", export_sha256=export_sha)
        return dict(created=True, export_sha256=export_sha)

    @classmethod
    def _export(cls, state, update_id, draw_ids):
        update = state["updates"].get(str(update_id))
        _require(update is not None and update["phase"] in ("sealed", "training_pending", "consumed"),
                 "Training requires a sealed complete panel")
        rows = cls._training_rows(state, update_id)
        _require(digest(rows) == update["export_sha256"], "Sealed export identity mismatch")
        if draw_ids is not None:
            _require(isinstance(draw_ids, (list, tuple)) and len(set(draw_ids)) == len(draw_ids),
                     "Training draw identities must be a unique explicit list")
            _require(set(draw_ids) == {row["draw_id"] for row in rows},
                     "Training must use the exact full train-role panel, never holdout or partial siblings")
        return rows

    def training_draws(self, update_id, *, draw_ids=None):
        """Read the immutable full train-role export; holdouts never enter it."""
        _integer(update_id, "update_id")
        return self._export(self.snapshot(), update_id, draw_ids)

    def begin_update(self, update_id, *, attempt_id, draw_ids=None):
        """Record an optimizer transaction intent, never claim the model changed.

        Reopening a pending intent requires reconciling the external checkpoint;
        it is neither permission to replay the optimizer nor evidence to skip it.
        """
        _integer(update_id, "update_id")
        _ident(attempt_id, "training_attempt_id")
        with self._transaction() as state:
            self._export(state, update_id, draw_ids)
            update = state["updates"][str(update_id)]
            intent = dict(attempt_id=attempt_id, input_export_sha256=update["export_sha256"],
                          policy_sha256=update["policy_sha256"], manifest_sha256=self.manifest.sha256)
            if update["training_intent"] is not None:
                _require(update["training_intent"] == intent, "Reconcile the existing training attempt; do not replace it")
                created = False
            else:
                _require(update["phase"] == "sealed", "Training intent requires a sealed panel")
                _require(not any(u["training_intent"] and u["training_intent"]["attempt_id"] == attempt_id
                                 for u in state["updates"].values()), "Training attempt ID already belongs to another update")
                update.update(phase="training_pending", training_intent=intent)
                created = True
            result = dict(intent, created=created, phase=update["phase"],
                          reconcile_required=not created and update["phase"] == "training_pending",
                          authorizes_execution=False)
        return result

    def complete_update(self, update_id, *, attempt_id, receipt):
        """Commit only a fully bound external durable-checkpoint review receipt.

        The external production validator owns actual GPU restoration and durable
        checkpoint inspection. This offline ledger validates receipt bindings; it
        cannot independently turn a caller-supplied digest into GPU verification.
        A checkpoint saved before a crash can be reconciled by supplying the same
        validated receipt later, without re-running or silently skipping training.
        """
        _integer(update_id, "update_id")
        _ident(attempt_id, "training_attempt_id")
        hashes = {"manifest_sha256", "input_export_sha256", "policy_sha256", "new_policy_sha256",
                  "new_checkpoint_sha256", "optimizer_state_sha256", "scheduler_state_sha256",
                  "cursor_state_sha256", "validator_evidence_sha256", "validator_code_revision_sha256"}
        _require(isinstance(receipt, dict) and set(receipt) == hashes | {"verification_kind"},
                 "A complete durable-checkpoint validator receipt is required")
        for key in hashes:
            _sha(receipt[key], key)
        _require(receipt["verification_kind"] == "production_runtime",
                 "CPU checks cannot commit an optimizer transaction")
        receipt = json.loads(_canonical(receipt))
        with self._transaction() as state:
            update = state["updates"].get(str(update_id))
            _require(update is not None and update["training_intent"] is not None,
                     "Missing training transaction intent")
            intent = update["training_intent"]
            _require(intent["attempt_id"] == attempt_id, "Wrong training attempt receipt")
            for key in ("manifest_sha256", "input_export_sha256", "policy_sha256"):
                _require(receipt[key] == intent[key], "Checkpoint receipt does not match immutable training inputs")
            _require(receipt["validator_code_revision_sha256"] == self.manifest.to_dict()["code_revision_sha256"],
                     "Validator code must match the pinned campaign revision")
            if update["training_receipt"] is not None:
                _require(update["training_receipt"] == receipt, "Committed checkpoint receipt cannot change")
                return dict(recorded=False, phase="consumed")
            _require(update["phase"] == "training_pending", "No pending optimizer transaction")
            update.update(phase="consumed", training_receipt=receipt)
        return dict(recorded=True, phase="consumed")

    def consume_update(self, update_id, *, attempt_id, draw_ids=None):
        """Deprecated intent-only alias; complete_update is required to consume."""
        return self.begin_update(update_id, attempt_id=attempt_id, draw_ids=draw_ids)

    def record_readiness_gate(self, gate, *, update_id, passed, evidence):
        """Store bound evidence; no assertion here can authorize production use."""
        _require(gate in READINESS_GATES and type(passed) is bool, "Unknown gate or nonboolean status")
        _integer(update_id, "update_id")
        required = {"manifest_sha256", "code_revision_sha256", "protocol_sha256",
                    "policy_sha256", "plan_sha256", "evidence_sha256", "verification_kind"}
        _require(isinstance(evidence, dict) and set(evidence) == required, "Incomplete readiness evidence binding")
        for key in required - {"verification_kind"}:
            _sha(evidence[key], key)
        _require(evidence["verification_kind"] in ("cpu", "production_runtime"), "Unknown verification kind")
        _require(evidence["manifest_sha256"] == self.manifest.sha256
                 and evidence["protocol_sha256"] == self.manifest.protocol_sha256
                 and evidence["code_revision_sha256"] == self.manifest.to_dict()["code_revision_sha256"],
                 "Evidence belongs to a different manifest, code or protocol")
        with self._transaction() as state:
            update = state["updates"].get(str(update_id))
            _require(update is not None and update["policy_sha256"] == evidence["policy_sha256"]
                     and update["plan_sha256"] == evidence["plan_sha256"],
                     "Evidence does not bind the collected policy version and frozen panel")
            key = f"{update_id}:{gate}:{evidence['evidence_sha256']}"
            record = json.loads(_canonical(dict(passed=passed, evidence=evidence)))
            if key in state["readiness"]:
                _require(state["readiness"][key] == record, "Readiness evidence is immutable")
            state["readiness"][key] = record

    def readiness(self, update_id):
        _integer(update_id, "update_id")
        state = self.snapshot()
        gates = {name: [record for key, record in state["readiness"].items()
                        if key.startswith(f"{update_id}:{name}:")]
                 for name in READINESS_GATES}
        complete = all(records and not any(not r["passed"] for r in records)
                       and any(r["passed"] and r["evidence"]["verification_kind"] == "production_runtime"
                               for r in records) for records in gates.values())
        return dict(gates=gates, evidence_complete=complete, launch_ready=False,
                    authorizes_execution=False, reason="Offline evidence ledger; production integration and exact resource authorization are external requirements")
