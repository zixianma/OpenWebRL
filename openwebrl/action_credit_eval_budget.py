"""Held-out evaluation accounting in the existing campaign's all-stage caps.

No model, browser or API is started here. Cohorts never become training draws.
The real evaluation owner verifies archives and verdicts independently; this
ledger preserves dispatch identity, every retry, and unresolved HTTP liability.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .action_credit_state import (
    BudgetExceeded, COST_DIMENSIONS, _canonical, _ident, _require, _safe_path,
    _sha, digest,
)


def _read_ref(reference):
    _require(isinstance(reference, dict) and set(reference) == {"path", "sha256"},
             "Exact artifact path and SHA required")
    _sha(reference["sha256"], "artifact_sha256")
    _require(isinstance(reference["path"], str) and Path(reference["path"]).is_absolute(),
             "Absolute artifact path required")
    path = _safe_path(reference["path"])
    _require(path == path.resolve(), "Canonical artifact path required")
    data = path.read_bytes()
    _require(hashlib.sha256(data).hexdigest() == reference["sha256"], "Artifact SHA changed")
    return data


def accounting_settings(store):
    """Read explicit caps/pins from a real training or verification manifest."""
    m = store.manifest.to_dict()
    if "purpose" in m:
        from .action_credit_verification import VerificationStore, approved_authority
        _require(type(store) is VerificationStore, "A purpose-scoped VerificationStore is required")
        approved_authority(store.manifest)
        return dict(excluded_tasks=m["excluded_training_task_ids"],
            original_actor_sha256=m["original_actor_inventory"]["sha256"],
            max_attempts=m["shared_initial_max_attempts_per_task"],
            actor_requests_cap=m["local_actor_requests_cap"], verification=True)
    return dict(excluded_tasks=list(m["task_splits"]),
        original_actor_sha256=m["initialization"]["original_actor_sha256"],
        max_attempts=m["max_attempts_per_draw"],
        actor_requests_cap=m["science"]["algorithm"].get("local_actor_requests_cap"), verification=False)


def register_cohort(store, cohort_id, request_ref):
    settings = accounting_settings(store)
    request = json.loads(_read_ref(request_ref))
    fields = {"schema_version", "cohort_id", "task_ids", "task_manifest", "protocol",
              "protocol_sha256", "checkpoint_inventory", "policy_sha256", "iteration",
              "source_inventory", "accounting_manifest_sha256", "artifact_schema", "diagnostic_only"}
    _require(set(request) == fields and request["schema_version"] == 1,
             "Exact evaluation request schema required")
    _sha(cohort_id, "cohort_id")
    _require(request["cohort_id"] == cohort_id == digest({k: v for k, v in request.items() if k != "cohort_id"}),
             "Evaluation cohort identity changed")
    ids = request["task_ids"]
    _require(isinstance(ids, list) and len(ids) == 300 and len(set(ids)) == 300,
             "Exactly300 unique evaluation IDs required")
    for task in ids:
        _ident(task, "evaluation_task_id")
    raw = _read_ref(request["task_manifest"])
    if Path(request["task_manifest"]["path"]).suffix == ".jsonl":
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    else:
        rows = json.loads(raw)
        if isinstance(rows, dict):
            rows = rows["task_ids"]
    pinned = [str(row["task_id"] if "task_id" in row else row["metadata"]["task_id"])
              if isinstance(row, dict) else str(row) for row in rows]
    _require(ids == pinned, "Evaluation IDs/order differ from pinned task manifest")
    _require(not set(ids) & set(settings["excluded_tasks"]),
             "Evaluation tasks must remain separate from campaign tasks")
    _require(request["accounting_manifest_sha256"] == store.manifest.sha256,
             "Evaluation cannot acquire a separate spending cap")
    _require(request["protocol_sha256"] == digest(request["protocol"])
             and request["artifact_schema"] == "native_monitor_full300_v1"
             and type(request["diagnostic_only"]) is bool,
             "Evaluation protocol/schema changed")
    _require(type(request["iteration"]) is int and request["iteration"] in (0, 10),
             "Only the approved initial0 and iteration10 cohorts are supported")
    _sha(request["policy_sha256"], "policy_sha256")
    _read_ref(request["checkpoint_inventory"])
    _read_ref(request["source_inventory"])
    if settings["verification"]:
        store.validate_evaluation_request(request)
    record = json.loads(_canonical(dict(request_ref=request_ref, request=request)))
    with store._transaction() as state:
        cohorts = state.setdefault("evaluation_cohorts", {})
        if cohort_id in cohorts:
            _require(all(cohorts[cohort_id][k] == v for k, v in record.items()),
                     "Evaluation request is immutable")
            return dict(created=False, cohort_id=cohort_id)
        _require(not state["halt"], "Campaign halted")
        if request["iteration"] == 0:
            _require(request["policy_sha256"] == settings["original_actor_sha256"]
                     == request["checkpoint_inventory"]["sha256"], "Initial evaluation must use original actor")
        else:
            update = state["updates"].get("9", {})
            receipt = update.get("training_receipt") or {}
            native = state.get("native_completed_updates", {}).get("9", {})
            _require(update.get("phase") == "consumed"
                     and receipt.get("new_policy_sha256") == request["policy_sha256"]
                     and receipt.get("new_checkpoint_sha256") == request["checkpoint_inventory"]["sha256"]
                     and native.get("checkpoint_inventory") == request["checkpoint_inventory"],
                     "Iteration10 evaluation needs the committed native checkpoint")
        _require(not any(c["request"]["iteration"] == request["iteration"] for c in cohorts.values()),
                 "Reconcile the existing cohort rather than starting a duplicate evaluation")
        record["tasks"] = {task: dict(attempts=[], selected_attempt=None, status="pending") for task in ids}
        cohorts[cohort_id] = record
    return dict(created=True, cohort_id=cohort_id)


def start_attempt(store, cohort_id, task_id, attempt_id):
    settings = accounting_settings(store)
    _ident(attempt_id, "evaluation_attempt_id")
    reserved = dict(gpu_seconds=0, api_microdollars=0, api_requests=0, browser_dispatches=1)
    with store._transaction() as state:
        _require(attempt_id not in state["attempts"], "Attempt ID already belongs to training")
        _require(attempt_id not in state.get("verification_attempts", {}), "Attempt ID already belongs to online verification")
        attempts = state.setdefault("evaluation_attempts", {})
        if attempt_id in attempts:
            old = attempts[attempt_id]
            _require(old["cohort_id"] == cohort_id and old["task_id"] == task_id,
                     "Attempt ID reused with different evaluation task")
            return dict(created=False, attempt_id=attempt_id, status=old["status"])
        _require(not state["halt"], "Campaign halted")
        cohort = state.get("evaluation_cohorts", {}).get(cohort_id)
        _require(cohort is not None and task_id in cohort["tasks"], "Unregistered evaluation task")
        slot = cohort["tasks"][task_id]
        _require(slot["status"] != "valid", "Valid evaluation outcomes cannot be retried")
        _require(not any(attempts[a]["status"] == "active" for a in slot["attempts"]),
                 "Resolve the existing evaluation attempt before retrying")
        _require(len(slot["attempts"]) < settings["max_attempts"],
                 "Evaluation retry cap reached")
        total, caps = store._totals(state), store.manifest.to_dict()["budget_caps"]
        if any(total[k] + reserved[k] > caps[k] for k in COST_DIMENSIONS):
            raise BudgetExceeded("Evaluation reservation exceeds the campaign's all-stage cap")
        if settings["verification"]:
            from .action_credit_verification import SHARED
            store._admit(state, SHARED, reserved)
        attempts[attempt_id] = dict(cohort_id=cohort_id, task_id=task_id, reserved=reserved,
            charged=reserved.copy(), actual=None, status="active", completion=None)
        slot["attempts"].append(attempt_id)
        slot["status"] = "pending"
    return dict(created=True, attempt_id=attempt_id, status="active")


def finish_attempt(store, attempt_id, *, status, outcome=None, reason=None):
    _ident(attempt_id, "evaluation_attempt_id")
    _require(status in ("valid", "invalid", "missing"), "Explicit evaluation disposition required")
    if status == "valid":
        _require(isinstance(outcome, dict) and set(outcome) == {"artifact_sha256", "reward"}
                 and type(outcome["reward"]) in (int, float) and outcome["reward"] in (0, 1)
                 and reason is None, "Valid evaluation requires a binary verdict and artifact")
    else:
        _ident(reason, "evaluation_failure_reason")
        _require((status == "missing" and outcome is None)
                 or (status == "invalid" and isinstance(outcome, dict) and set(outcome) == {"artifact_sha256"}),
                 "Invalid evaluations preserve artifacts without a reward; missing has no artifact")
    if outcome is not None:
        _sha(outcome["artifact_sha256"], "evaluation_artifact_sha256")
    completion = json.loads(_canonical(dict(status=status, outcome=outcome, reason=reason)))
    with store._transaction() as state:
        attempt = state.get("evaluation_attempts", {}).get(attempt_id)
        _require(attempt is not None, "Unknown evaluation attempt")
        if attempt["completion"] is not None:
            _require(attempt["completion"] == completion, "Completed evaluation attempt is immutable")
            return dict(recorded=False, attempt_id=attempt_id)
        children = [c for c in state.get("subrequests", {}).values() if c["parent_attempt_id"] == attempt_id]
        _require(not any(c["status"] == "reserved" for c in children),
                 "Settle every evaluation HTTP attempt before closing its browser attempt")
        actor_children = [c for c in state.get("online_actor_requests", {}).values()
                          if c.get("attempt_id") == attempt_id]
        _require(not any(c["status"] == "reserved" for c in actor_children),
                 "Settle every evaluation actor attempt before closing its browser attempt")
        attempt.update(status=status, completion=completion, actual=attempt["reserved"].copy())
        slot = state["evaluation_cohorts"][attempt["cohort_id"]]["tasks"][attempt["task_id"]]
        slot.update(status=status, selected_attempt=attempt_id)
    return dict(recorded=True, attempt_id=attempt_id)


def reserve_actor_request(store, actor_attempt_id, *, parent_attempt_id, task_id,
                          policy_sha256, payload_sha256, turn):
    """Reserve one local actor transport; created=False forbids redispatch.

    This is a transport attempt ID, distinct from its browser parent. Retries
    use a new ID after explicitly settling failed/unknown attempts; every ID
    stays counted against the same map/cap as training NativeCollectionPolicy requests.
    No successful or ambiguous attempt is refunded, including across restarts.
    """
    settings = accounting_settings(store)
    for name,value in (("actor_attempt_id",actor_attempt_id),("parent_attempt_id",parent_attempt_id),
                       ("evaluation_task_id",task_id)):
        _ident(value,name)
    _sha(policy_sha256,"policy_sha256"); _sha(payload_sha256,"payload_sha256")
    _require(type(turn) is int and turn >= 0,"Explicit nonnegative actor turn required")
    with store._transaction() as state:
        parent = state.get("evaluation_attempts",{}).get(parent_attempt_id)
        _require(parent is not None and parent["task_id"] == task_id,"Actor request evaluation parent/task differs")
        cohort = state["evaluation_cohorts"][parent["cohort_id"]]
        maximum = cohort["request"]["protocol"].get("max_turns")
        _require(type(maximum) is int and 0 <= turn < maximum,
                 "Actor turn exceeds the explicit frozen evaluation horizon")
        _require(cohort["request"]["policy_sha256"] == policy_sha256,"Actor request evaluation policy differs")
        identity = dict(kind="evaluation",attempt_id=parent_attempt_id,cohort_id=parent["cohort_id"],
                        task_id=task_id,policy_sha256=policy_sha256,payload_sha256=payload_sha256,turn=turn)
        calls = state.setdefault("online_actor_requests",{})
        if actor_attempt_id in calls:
            old = calls[actor_attempt_id]
            _require(all(old.get(k) == v for k,v in identity.items()),
                     "Actor transport identity reused for another request or training")
            return dict(created=False,actor_attempt_id=actor_attempt_id,status=old["status"])
        _require(not state["halt"],"Campaign halted")
        _require(parent["status"] == "active","Active evaluation parent required")
        _require(not any(c.get("attempt_id") == parent_attempt_id and c["status"] == "reserved"
                         for c in calls.values()),"Reconcile the existing actor request before dispatch")
        cap = settings["actor_requests_cap"]
        _require(type(cap) is int and cap > 0,"Explicit positive local actor request cap required")
        if len(calls) >= cap:
            raise BudgetExceeded("Evaluation actor attempt exceeds shared training/evaluation request cap")
        calls[actor_attempt_id] = dict(identity,status="reserved",completion=None)
    return dict(created=True,actor_attempt_id=actor_attempt_id,status="reserved")


def settle_actor_request(store, actor_attempt_id, *, status, response_sha256=None, reason=None):
    """Persist an immutable transport disposition; retain its spent count."""
    _ident(actor_attempt_id,"actor_attempt_id")
    _require(status in ("received","failed","unknown"),"Explicit actor transport disposition required")
    if status == "received":
        _sha(response_sha256,"actor_response_sha256")
        _require(reason is None,"Received actor response cannot have a failure reason")
    else:
        _ident(reason,"actor_failure_reason")
        if response_sha256 is not None:_sha(response_sha256,"actor_response_sha256")
    completion = dict(status=status,response_sha256=response_sha256,reason=reason)
    with store._transaction() as state:
        call = state.get("online_actor_requests",{}).get(actor_attempt_id)
        _require(call is not None and call.get("kind") == "evaluation","Unknown evaluation actor reservation")
        if call["completion"] is not None:
            _require(call["completion"] == completion,"Actor transport completion is immutable")
            return dict(recorded=False,actor_attempt_id=actor_attempt_id)
        _require(call["status"] == "reserved","Actor request was not reserved")
        call.update(status=status,response_sha256=response_sha256,error=reason,completion=completion)
    return dict(recorded=True,actor_attempt_id=actor_attempt_id)
