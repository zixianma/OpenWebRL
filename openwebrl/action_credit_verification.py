"""Purpose-scoped, all-attempt accounting for shared0 and online verification.

This is deliberately not a training CampaignManifest/Store: it has no arm,
optimizer, update, cursor or training-draw APIs. Only durable ledger primitives
are shared. Preparing a draft does not authorize execution or create a ledger.
Every admission needs the exact separately approved verification receipt.
GPU allocation time belongs to its owning allocation ledger, never this store.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from decimal import Decimal, InvalidOperation

from .action_credit_state import (
    BudgetExceeded, CampaignManifest, CampaignStore, COST_DIMENSIONS, StateError,
    _canonical, _cost, _ident, _integer, _require, _safe_path, _sha, digest,
)
from .action_credit_eval_budget import _read_ref
from .action_credit_judge import JudgeHTTPProtocol

PURPOSE = "shared_initial_and_verification"
SHARED = "shared_initial_full300"
VERIFICATION = "online_verification"


def _ref(value):
    _require(isinstance(value, dict) and set(value) == {"path", "sha256"}, "Exact artifact reference required")
    _sha(value["sha256"], "artifact_sha256")
    _require(isinstance(value["path"], str) and value["path"].startswith("/"), "Absolute artifact path required")
    _safe_path(value["path"])


def _ids(values, name, *, count=None):
    _require(isinstance(values, list) and bool(values), name + " must be nonempty")
    for value in values:
        _ident(value, name)
    _require(len(set(values)) == len(values) and (count is None or len(values) == count), name + " identities differ")


@dataclass(frozen=True)
class VerificationManifest:
    """Explicit immutable pins; null authority is draft-only, never executable."""

    canonical_json: str

    def __post_init__(self):
        try:
            m = json.loads(self.canonical_json)
        except (TypeError, ValueError) as exc:
            raise StateError("Invalid verification manifest JSON") from exc
        self._validate(m)
        object.__setattr__(self, "canonical_json", _canonical(m))

    @classmethod
    def from_dict(cls, value):
        return cls(_canonical(value))

    def to_dict(self):
        return json.loads(self.canonical_json)

    @property
    def sha256(self):
        return hashlib.sha256(self.canonical_json.encode()).hexdigest()

    @staticmethod
    def _validate(m):
        fields = {"schema_version", "purpose", "campaign_id", "proposal_reference",
            "original_actor_inventory", "source_inventory", "evaluation_task_manifest",
            "excluded_training_task_ids", "verification_task_ids", "verification_protocol",
            "verification_protocol_sha256", "budget_caps", "shared_initial_caps",
            "local_actor_requests_cap", "shared_initial_max_attempts_per_task",
            "verification_max_attempts_per_task", "science", "approval_reference", "judge_http_protocol"}
        _require(isinstance(m, dict) and set(m) == fields, "Exact verification manifest schema required")
        _require(type(m["schema_version"]) is int and m["schema_version"] == 1 and m["purpose"] == PURPOSE,
                 "Wrong verification manifest purpose/version")
        _ident(m["campaign_id"], "campaign_id")
        for key in ("proposal_reference", "original_actor_inventory", "source_inventory", "evaluation_task_manifest"):
            _ref(m[key])
        _ids(m["excluded_training_task_ids"], "excluded training tasks")
        _ids(m["verification_task_ids"], "verification tasks")
        _require(set(m["verification_task_ids"]) <= set(m["excluded_training_task_ids"]),
                 "Verification tasks must be explicitly pinned training tasks, never the held-out full300")
        protocol = m["verification_protocol"]
        _require(isinstance(protocol, dict) and protocol.get("optimizer_updates") == 0
                 and type(protocol.get("optimizer_updates")) is int
                 and protocol.get("policy_sha256") == m["original_actor_inventory"]["sha256"]
                 and protocol.get("browser") == "local_process"
                 and protocol.get("judge") == "gpt-4.1/action_history",
                 "Verification must pin the original actor/local browser/judge and no optimizer updates")
        _integer(protocol.get("max_turns"), "verification max_turns", 1)
        _require(protocol["max_turns"] <= 30, "Verification exceeds the approved30-turn horizon")
        _require(m["verification_protocol_sha256"] == digest(protocol), "Verification protocol changed")
        caps, shared = _cost(m["budget_caps"]), _cost(m["shared_initial_caps"])
        _require(caps["gpu_seconds"] == shared["gpu_seconds"] == 0,
                 "GPU time is charged by the owning allocation; this is not a new GPU budget")
        _require(all(shared[k] <= caps[k] for k in COST_DIMENSIONS), "Shared reserve exceeds the total cap")
        for key in ("local_actor_requests_cap", "shared_initial_max_attempts_per_task", "verification_max_attempts_per_task"):
            _integer(m[key], key, 1)
        judge = m["science"]
        _require(isinstance(judge, dict) and set(judge) == {"judge"}, "Verification has only judge science, never training-arm settings")
        judge = judge["judge"]
        _require(isinstance(judge, dict) and set(judge) == {"model", "prompt_sha256", "canonical_reward_source_sha256", "http_protocol_sha256"}
                 and judge["model"] == "gpt-4.1", "Exact verification judge pins required")
        for key in ("prompt_sha256", "canonical_reward_source_sha256"):
            _sha(judge[key], key)
        ref, http = m["approval_reference"], m["judge_http_protocol"]
        if ref is None:
            _require(http is None and judge["http_protocol_sha256"] is None, "Draft must not carry invented API authority")
        else:
            _ref(ref)
            _require(isinstance(http, dict), "Exact judge HTTP protocol required")
            protocol = JudgeHTTPProtocol(**http)
            _require(protocol.sha256 == judge["http_protocol_sha256"] and protocol.model == "gpt-4.1",
                     "Verification HTTP protocol differs from its pinned judge")


def _proposal(manifest):
    m = manifest.to_dict()
    value = json.loads(_read_ref(m["proposal_reference"]))
    v = value["validation"]["separately_proposed_online_verification"]
    shared = v["shared_initial_full300_reserve"]
    # A proposal fixes limits and purpose. It is never an approval receipt.
    _require(v["no_execution_until_api_authority_bound"] is True and v["includes_shared_initial_full300"] is True
             and shared["tasks"] == 300, "Proposal does not define this verification scope")
    expected = dict(gpu_seconds=0, api_microdollars=_microdollars(v["judge_usd_cap"]),
                    api_requests=v["judge_http_attempts_cap"], browser_dispatches=v["browser_dispatches_cap"])
    earmark = dict(gpu_seconds=0, api_microdollars=_microdollars(shared["usd"]),
                   api_requests=shared["judge_http_attempts"], browser_dispatches=shared["browser_dispatches"])
    _cost(expected); _cost(earmark)
    _integer(v["local_actor_requests_cap"], "proposed local actor requests", 1)
    _require(m["budget_caps"] == expected and m["shared_initial_caps"] == earmark
             and m["local_actor_requests_cap"] == v["local_actor_requests_cap"], "Verification limits differ from the frozen proposal")


def _microdollars(value):
    try:
        amount = Decimal(str(value)) * 1_000_000
        _require(amount.is_finite() and amount > 0 and amount == amount.to_integral_value(),
                 "Proposal USD must be a positive exact integer number of microdollars")
        return int(amount)
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise StateError("Invalid proposed USD cap") from exc


def prepare_verification_manifest(*, campaign_id, proposal_reference, original_actor_inventory,
        source_inventory, evaluation_task_manifest, excluded_training_task_ids, verification_task_ids,
        verification_protocol, shared_initial_max_attempts_per_task, verification_max_attempts_per_task,
        judge_prompt_sha256, canonical_reward_source_sha256, approval_reference=None, judge_http_protocol=None):
    """Build an immutable draft or approved manifest; never create a store.

    The caller supplies actual frozen scientific pins and retry limits. Resource
    limits are copied exactly from the pinned proposal, not inferred from any
    existing campaign's unused balance. Approved preparation verifies authority.
    """
    proposal = json.loads(_read_ref(proposal_reference))
    v = proposal["validation"]["separately_proposed_online_verification"]
    shared = v["shared_initial_full300_reserve"]
    protocol = JudgeHTTPProtocol(**judge_http_protocol) if judge_http_protocol is not None else None
    manifest = VerificationManifest.from_dict(dict(schema_version=1, purpose=PURPOSE,
        campaign_id=campaign_id, proposal_reference=proposal_reference,
        original_actor_inventory=original_actor_inventory, source_inventory=source_inventory,
        evaluation_task_manifest=evaluation_task_manifest, excluded_training_task_ids=excluded_training_task_ids,
        verification_task_ids=verification_task_ids, verification_protocol=verification_protocol,
        verification_protocol_sha256=digest(verification_protocol),
        budget_caps=dict(gpu_seconds=0, api_microdollars=_microdollars(v["judge_usd_cap"]),
            api_requests=v["judge_http_attempts_cap"], browser_dispatches=v["browser_dispatches_cap"]),
        shared_initial_caps=dict(gpu_seconds=0, api_microdollars=_microdollars(shared["usd"]),
            api_requests=shared["judge_http_attempts"], browser_dispatches=shared["browser_dispatches"]),
        local_actor_requests_cap=v["local_actor_requests_cap"],
        shared_initial_max_attempts_per_task=shared_initial_max_attempts_per_task,
        verification_max_attempts_per_task=verification_max_attempts_per_task,
        science=dict(judge=dict(model="gpt-4.1", prompt_sha256=judge_prompt_sha256,
            canonical_reward_source_sha256=canonical_reward_source_sha256,
            http_protocol_sha256=protocol.sha256 if protocol else None)),
        approval_reference=approval_reference, judge_http_protocol=judge_http_protocol))
    _proposal(manifest)
    if approval_reference is not None:
        approved_authority(manifest)
    return manifest


def approved_authority(manifest):
    """Verify exact receipt bytes and scope; no API or scheduler operation."""
    _require(type(manifest) is VerificationManifest, "Typed verification manifest required")
    _proposal(manifest)
    m = manifest.to_dict()
    _require(m["approval_reference"] is not None, "Verification API approval is pending; draft cannot create or dispatch a store")
    receipt = json.loads(_read_ref(m["approval_reference"]))
    protocol = JudgeHTTPProtocol(**m["judge_http_protocol"])
    exact = dict(status="approved", campaign_id=m["campaign_id"], purpose=PURPOSE, model="gpt-4.1",
        budget_caps=m["budget_caps"], shared_initial_caps=m["shared_initial_caps"],
        local_actor_requests_cap=m["local_actor_requests_cap"],
        proposal_reference=m["proposal_reference"],
        api_microdollars_cap=m["budget_caps"]["api_microdollars"], api_requests_cap=m["budget_caps"]["api_requests"])
    _require(all(receipt.get(k) == v for k, v in exact.items())
             and isinstance(receipt.get("user_authorization_text"), str) and bool(receipt["user_authorization_text"].strip())
             and isinstance(receipt.get("user_authorization_reference"), str) and bool(receipt["user_authorization_reference"].strip())
             and digest(receipt) == protocol.approval_receipt_sha256,
             "An exact separately approved verification-purpose receipt is required; pilot authority cannot be borrowed")
    return receipt


class VerificationStore:
    """Independent shared0/verification ledger; no training APIs are exposed.

    Empty legacy structural maps in its snapshot are a read-only accounting
    compatibility detail, not a synthetic arm, training plan or campaign.
    """

    _lock = CampaignStore._lock
    _write = CampaignStore._write
    _transaction = CampaignStore._transaction
    snapshot = CampaignStore.snapshot
    settle_subrequest = CampaignStore.settle_subrequest
    register_evaluation_cohort = CampaignStore.register_evaluation_cohort
    start_evaluation_attempt = CampaignStore.start_evaluation_attempt
    finish_evaluation_attempt = CampaignStore.finish_evaluation_attempt
    reserve_evaluation_actor_request = CampaignStore.reserve_evaluation_actor_request
    settle_evaluation_actor_request = CampaignStore.settle_evaluation_actor_request

    def __init__(self, root, manifest):
        approved_authority(manifest)
        self.root, self.manifest = _safe_path(root), manifest
        _require(self.root.is_dir(), "Verification store is absent; never reset a missing ledger")
        self._verify_references()
        with self._lock():
            self._read()

    @classmethod
    def create(cls, root, manifest):
        approved_authority(manifest)  # Must precede every filesystem mutation.
        root = _safe_path(root)
        instance = object.__new__(cls)
        instance.root, instance.manifest = root, manifest
        instance._verify_references()
        root.mkdir(exist_ok=False)
        parent = os.open(root.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
        with instance._lock():
            instance._write(dict(schema_version=1, purpose=PURPOSE, manifest_sha256=manifest.sha256,
                manifest=manifest.to_dict(), revision=0, updates={}, requests={}, draws={}, attempts={},
                readiness={}, halt=None, verification_attempts={}, evaluation_cohorts={}, evaluation_attempts={}))
        return cls(root, manifest)

    def _verify_references(self):
        from .action_credit_evaluation import _ids as evaluation_ids
        m = self.manifest.to_dict()
        for key in ("original_actor_inventory", "source_inventory"):
            _read_ref(m[key])
        ids = evaluation_ids(m["evaluation_task_manifest"])
        _require(not set(ids) & set(m["excluded_training_task_ids"]), "Full300 tasks overlap pinned training/verification tasks")

    def _read(self):
        state = CampaignStore._read(self)
        _require(state.get("purpose") == PURPOSE and all(state[k] == {} for k in ("updates", "requests", "draws", "attempts", "readiness")),
                 "Verification store cannot contain training state")
        return state

    @staticmethod
    def _totals(state):
        total = CampaignStore._totals(state)
        for attempt in state.get("verification_attempts", {}).values():
            for key in COST_DIMENSIONS:
                total[key] += attempt["charged"][key]
        return total

    @staticmethod
    def _purpose_totals(state, purpose):
        _require(purpose in (SHARED, VERIFICATION), "Unknown accounting purpose")
        attempts = state.get("evaluation_attempts" if purpose == SHARED else "verification_attempts", {})
        entries = [*attempts.values(), *(c for c in state.get("subrequests", {}).values()
                    if c["parent_attempt_id"] in attempts)]
        return {k: sum(c["charged"][k] for c in entries) for k in COST_DIMENSIONS}

    def _scope_cap(self, purpose):
        m = self.manifest.to_dict()
        return m["shared_initial_caps"] if purpose == SHARED else {
            k: m["budget_caps"][k] - m["shared_initial_caps"][k] for k in COST_DIMENSIONS}

    def _admit(self, state, purpose, reserved):
        _require(not state["halt"], "Verification store halted")
        total, used = self._totals(state), self._purpose_totals(state, purpose)
        caps, scope = self.manifest.to_dict()["budget_caps"], self._scope_cap(purpose)
        if any(total[k] + reserved[k] > caps[k] or used[k] + reserved[k] > scope[k] for k in COST_DIMENSIONS):
            raise BudgetExceeded("Verification reservation exceeds total or purpose cap; earmarked balances cannot be transferred")

    def validate_accounting_snapshot(self, state):
        """Final audits must validate both purpose caps as well as the total."""
        _require(state["manifest_sha256"] == self.manifest.sha256, "Snapshot belongs to another verification ledger")
        self._admit(state, SHARED, {k: 0 for k in COST_DIMENSIONS})
        self._admit(state, VERIFICATION, {k: 0 for k in COST_DIMENSIONS})
        _require(len(state.get("online_actor_requests", {})) <= self.manifest.to_dict()["local_actor_requests_cap"],
                 "Shared verification/full300 actor cap exceeded")

    def validate_evaluation_request(self, request):
        approved_authority(self.manifest)
        m = self.manifest.to_dict()
        from .action_credit_evaluation import evaluation_protocol
        _require(request["iteration"] == 0 and request["checkpoint_inventory"] == m["original_actor_inventory"]
                 and request["task_manifest"] == m["evaluation_task_manifest"] and request["source_inventory"] == m["source_inventory"]
                 and request["protocol"] == evaluation_protocol(), "Verification ledger permits exactly the pinned shared initial full300")

    def reserve_subrequest(self, request_id, *, parent_attempt_id, payload_sha256, reserved_cost):
        approved_authority(self.manifest)
        _ident(request_id, "request_id"); _ident(parent_attempt_id, "parent_attempt_id")
        _sha(payload_sha256, "payload_sha256")
        reserved = _cost(reserved_cost)
        protocol = JudgeHTTPProtocol(**self.manifest.to_dict()["judge_http_protocol"])
        _require(reserved == dict(gpu_seconds=0, api_requests=1, browser_dispatches=0,
                                 api_microdollars=protocol.reserved_microdollars_per_http),
                 "Verification must reserve the exact approved per-HTTP upper bound")
        identity = dict(parent_attempt_id=parent_attempt_id, payload_sha256=payload_sha256, reserved=reserved)
        with self._transaction() as state:
            calls = state.setdefault("subrequests", {})
            if request_id in calls:
                _require(all(calls[request_id][k] == v for k, v in identity.items()), "HTTP identity cannot be reused")
                return dict(created=False, request_id=request_id, status=calls[request_id]["status"])
            shared = parent_attempt_id in state["evaluation_attempts"]
            parent = state["evaluation_attempts"].get(parent_attempt_id) if shared else state["verification_attempts"].get(parent_attempt_id)
            _require(parent is not None and parent["status"] == "active", "Active verification/evaluation parent required")
            self._admit(state, SHARED if shared else VERIFICATION, reserved)
            calls[request_id] = dict(identity, charged=reserved.copy(), actual=None, status="reserved", completion=None)
        return dict(created=True, request_id=request_id, status="reserved")

    def start_verification_attempt(self, task_id, attempt_id, *, protocol_sha256):
        approved_authority(self.manifest)
        _ident(attempt_id, "verification_attempt_id")
        m = self.manifest.to_dict()
        _require(task_id in m["verification_task_ids"] and protocol_sha256 == m["verification_protocol_sha256"],
                 "Verification task/protocol was not frozen")
        identity = dict(task_id=task_id, protocol_sha256=protocol_sha256, policy_sha256=m["original_actor_inventory"]["sha256"])
        reserved = dict(gpu_seconds=0, api_requests=0, api_microdollars=0, browser_dispatches=1)
        with self._transaction() as state:
            _require(attempt_id not in state["evaluation_attempts"], "Attempt ID belongs to shared full300")
            attempts = state["verification_attempts"]
            if attempt_id in attempts:
                _require(all(attempts[attempt_id][k] == v for k, v in identity.items()), "Verification attempt ID reused")
                return dict(created=False, attempt_id=attempt_id, status=attempts[attempt_id]["status"])
            previous = [a for a in attempts.values() if a["task_id"] == task_id]
            _require(not any(a["status"] in ("active", "passed") for a in previous), "Reconcile active attempt; never repeat passed verification")
            _require(len(previous) < m["verification_max_attempts_per_task"], "Verification task retry cap exhausted")
            self._admit(state, VERIFICATION, reserved)
            attempts[attempt_id] = dict(identity, reserved=reserved, charged=reserved.copy(), actual=None,
                                       status="active", completion=None)
        return dict(created=True, attempt_id=attempt_id, status="active")

    def finish_verification_attempt(self, attempt_id, *, status, evidence, reason=None):
        _require(status in ("passed", "failed", "unknown"), "Explicit verification disposition required")
        _ref(evidence); record = json.loads(_read_ref(evidence))
        _require((status == "passed" and reason is None) or (status != "passed" and isinstance(reason, str) and bool(reason.strip())),
                 "Failed/unknown verification needs a reason and retained evidence")
        completion = dict(status=status, evidence=evidence, reason=reason)
        with self._transaction() as state:
            attempt = state["verification_attempts"].get(attempt_id)
            _require(attempt is not None, "Unknown verification attempt")
            _require(isinstance(record, dict) and record.get("attempt_id") == attempt_id
                     and record.get("task_id") == attempt["task_id"]
                     and record.get("policy_sha256") == attempt["policy_sha256"]
                     and record.get("protocol_sha256") == attempt["protocol_sha256"]
                     and record.get("manifest_sha256") == self.manifest.sha256 and record.get("status") == status,
                     "Verification evidence must bind this exact attempt/policy/protocol/disposition")
            if attempt["completion"] is not None:
                _require(attempt["completion"] == completion, "Verification disposition cannot be rewritten")
                return dict(recorded=False, attempt_id=attempt_id)
            children = [c for c in state.get("subrequests", {}).values() if c["parent_attempt_id"] == attempt_id]
            children += [c for c in state.get("online_actor_requests", {}).values() if c["attempt_id"] == attempt_id]
            _require(not any(c["status"] == "reserved" for c in children), "Reconcile every actor/HTTP attempt before closing verification")
            attempt.update(status=status, completion=completion, actual=attempt["reserved"].copy())
        return dict(recorded=True, attempt_id=attempt_id)

    def reserve_verification_actor_request(self, actor_attempt_id, *, parent_attempt_id, task_id,
                                           policy_sha256, payload_sha256, turn):
        approved_authority(self.manifest)
        _ident(actor_attempt_id, "actor_attempt_id"); _sha(payload_sha256, "payload_sha256")
        m = self.manifest.to_dict()
        _require(policy_sha256 == m["original_actor_inventory"]["sha256"] and type(turn) is int
                 and 0 <= turn < m["verification_protocol"]["max_turns"], "Verification actor policy/turn changed")
        identity = dict(kind="verification", attempt_id=parent_attempt_id, task_id=task_id,
                        policy_sha256=policy_sha256, payload_sha256=payload_sha256, turn=turn)
        with self._transaction() as state:
            parent = state["verification_attempts"].get(parent_attempt_id)
            _require(parent is not None and parent["task_id"] == task_id, "Verification actor parent/task changed")
            calls = state.setdefault("online_actor_requests", {})
            if actor_attempt_id in calls:
                _require(all(calls[actor_attempt_id].get(k) == v for k, v in identity.items()), "Actor transport identity reused")
                return dict(created=False, actor_attempt_id=actor_attempt_id, status=calls[actor_attempt_id]["status"])
            _require(not state["halt"] and parent["status"] == "active", "Verification actor parent not active or ledger halted")
            _require(not any(c["attempt_id"] == parent_attempt_id and c["status"] == "reserved" for c in calls.values()),
                     "Reconcile existing actor transport before another dispatch")
            if len(calls) >= m["local_actor_requests_cap"]:
                raise BudgetExceeded("Shared verification/full300 actor transport cap exhausted")
            calls[actor_attempt_id] = dict(identity, status="reserved", completion=None)
        return dict(created=True, actor_attempt_id=actor_attempt_id, status="reserved")

    def settle_verification_actor_request(self, actor_attempt_id, *, status, response_sha256=None, reason=None):
        _require(status in ("received", "failed", "unknown"), "Explicit actor disposition required")
        if status == "received":
            _sha(response_sha256, "actor_response_sha256")
            _require(reason is None, "Received actor response cannot have failure reason")
        else:
            _ident(reason, "actor failure/unknown reason")
            if response_sha256 is not None:
                _sha(response_sha256, "actor_response_sha256")
        completion = dict(status=status, response_sha256=response_sha256, reason=reason)
        with self._transaction() as state:
            call = state.get("online_actor_requests", {}).get(actor_attempt_id)
            _require(call is not None and call["kind"] == "verification", "Unknown verification actor transport")
            if call["completion"] is not None:
                _require(call["completion"] == completion, "Actor disposition cannot be rewritten")
                return dict(recorded=False, actor_attempt_id=actor_attempt_id)
            call.update(status=status, completion=completion, response_sha256=response_sha256, error=reason)
        return dict(recorded=True, actor_attempt_id=actor_attempt_id)


def open_accounting_store(root, manifest_dict):
    """Reopen exact purpose; unknown purpose never falls back to pilot billing."""
    _require(isinstance(manifest_dict, dict), "Accounting manifest must be an object")
    if "purpose" in manifest_dict:
        return VerificationStore(root, VerificationManifest.from_dict(manifest_dict))
    return CampaignStore(root, CampaignManifest.from_dict(manifest_dict))
