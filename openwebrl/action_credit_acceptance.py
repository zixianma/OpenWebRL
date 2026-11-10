"""Read-only saved-collection audit and immutable native-admission receipt.

One owner scans the scored batch and sequentially reloads each saved episode;
no eight-rank media/checkpoint scan is requested here. Source/file checks bind
trusted local production records, not cryptographic signatures or a new live
browser replay. Observable replay equivalence does not clone remote servers.
CPU fixtures must request diagnostic_only=True and emit cpu_mock receipts,
which the native production controller rejects.
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict, replace
import inspect
import json
import math
from pathlib import Path
import time

from .action_credit import digest
from .action_credit_campaign import _artifact_file
from .action_credit_collect import FrozenActor, physical_verifier_sha256
from .action_credit_episode import load_pending_episode
from .action_credit_judge import JudgeHTTPProtocol
from .action_credit_native_controller import _hash, _json, _write, load_scored_pending, validator_source_sha256
from .action_credit_receipts import ReplayReceipt
from .action_credit_state import COST_DIMENSIONS, _safe_path
from .rl_recipe import parse_judge_verdict


def _sources(store, reference):
    value = _json(reference)
    if digest(value) != store.manifest.to_dict()["code_revision_sha256"]:
        raise ValueError("source inventory differs from campaign")
    root = _safe_path(value["source_root"])
    running = Path(__file__).resolve().parents[1]
    required = {p.relative_to(running).as_posix() for p in (running/"openwebrl").glob("action_credit*.py")}
    required |= {"openwebrl/generate_browser.py", "openwebrl/reward_browser.py",
                 "openwebrl/arm_branch_protocol.py", "openwebrl/arm_branch_browser.py",
                 "openwebrl/rl_recipe.py", "openwebrl/controlled_sft_worker.py", "openwebrl/controlled_browser_server.py",
                 "openwebrl/response_format.py"}
    if not required <= set(value["files"]):
        raise ValueError("production source inventory omits an audited implementation")
    for name, sha in value["files"].items():
        path = _safe_path(root/name)
        if root not in path.parents or _hash(path) != sha:
            raise ValueError("pinned source file changed or escaped its root")
        if name in required and _hash(running/name) != sha:
            raise ValueError("auditor is running a different source implementation")
    return value


def audit_allocation(reference, manifest, *, now=None):
    """Parse retained four-column sacct records, not just owner summary flags."""
    now = time.time() if now is None else now
    value = _json(reference)
    cap = manifest.to_dict()["budget_caps"]["gpu_seconds"]
    if (value.get("schema_version") != 1 or value.get("manifest_sha256") != manifest.sha256
            or value.get("approved_gpu_seconds_cap") != cap
            or type(value.get("observed_unix")) not in (int, float)
            or not 0 <= now-value["observed_unix"] <= 300):
        raise ValueError("allocation accounting is stale or differs from its approved cap")
    ledger = _json(value["attempt_ledger"])
    if ledger.get("manifest_sha256") != manifest.sha256 or ledger.get("approved_gpu_seconds_cap") != cap:
        raise ValueError("allocation attempt ledger changed campaign or cap")
    ids = [str(row["job_id"]) for row in ledger["attempts"]]
    if not ids or len(set(ids)) != len(ids) or any(not key.isdigit() for key in ids):
        raise ValueError("unique exact parent scheduler IDs required")
    raw = {}
    for ref in value["scheduler_records"]:
        for line in _artifact_file(ref).read_text().splitlines():
            parts = line.split("|")
            if len(parts) == 5 and parts[-1] == "": parts.pop()
            if not line.strip() or parts[0] == "JobIDRaw": continue
            if len(parts) != 4 or not parts[0].isdigit() or parts[0] in raw:
                raise ValueError("unexpected or duplicate scheduler record")
            job, state, elapsed, tres = parts
            state = state.split()[0].rstrip("+")
            if not elapsed.isdigit() or state not in ("PENDING", "RUNNING", "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED"):
                raise ValueError("unrecognized scheduler elapsed/state")
            resources = dict(item.split("=", 1) for item in tres.split(",") if "=" in item)
            if state == "PENDING":
                if int(elapsed) != 0: raise ValueError("pending allocation consumed nonzero runtime")
            elif state == "CANCELLED" and int(elapsed) == 0 and not tres.strip():
                # Never-started cancellation: gpus=8 is the approved cost
                # rate times zero, not evidence of an actual allocation.
                pass
            elif resources.get("gres/gpu") != "8":
                raise ValueError("scheduler allocation differs from approved eight GPUs")
            raw[job] = dict(job_id=job, gpus=8, elapsed_seconds=int(elapsed), state=state)
    normalized = {str(row["job_id"]): dict(row, job_id=str(row["job_id"])) for row in value["attempts"]}
    if set(raw) != set(ids) or len(normalized) != len(value["attempts"]) or normalized != raw:
        raise ValueError("scheduler evidence does not cover every preserved allocation attempt exactly")
    used = sum(row["gpus"]*row["elapsed_seconds"] for row in raw.values())
    if used != value.get("gpu_seconds_used") or used > cap:
        raise ValueError("allocation time was reset, omitted or exceeded")
    return dict(observed_unix=value["observed_unix"], gpu_seconds_used=used, approved_gpu_seconds_cap=cap,
                running_gpus=sum(row["gpus"] for row in raw.values() if row["state"] == "RUNNING"),
                attempt_ids=sorted(ids))


def _journals(directory):
    records, references = [], []
    for path in sorted(directory.glob("[0-9]*.json")):
        reference = dict(path=str(path), sha256=_hash(path))
        records.append(_json(reference)); references.append(reference)
    return records, references


def _http_costs(state, attempts, store, protocol):
    references, unknown, checked = [], 0, set()
    for attempt in attempts:
        directory = _safe_path(store.root/"online_artifacts"/digest(attempt)/"judge")
        events, refs = _journals(directory); references.extend(refs)
        reserved, responses, errors = {}, {}, {}
        for event in events:
            kind, record = event["event"], event["record"]
            table = {"judge_http_reserved":reserved, "judge_http_response":responses, "judge_http_error":errors}.get(kind)
            if table is not None:
                key = record["request_id"]
                if key in table: raise ValueError("duplicate HTTP journal event")
                table[key] = record
        calls = {key: row for key, row in state.get("subrequests", {}).items() if row["parent_attempt_id"] == attempt}
        if set(reserved) != set(calls) or set(responses)-set(calls) or set(errors)-set(calls):
            raise ValueError("judge journal and charged HTTP inventory disagree")
        for key, call in calls.items():
            request = reserved[key]
            if (call["payload_sha256"] != digest(request["payload"])
                    or request["protocol_sha256"] != protocol.sha256
                    or request["payload"].get("model") != protocol.model
                    or request["payload"].get("max_completion_tokens") != protocol.max_completion_tokens
                    or call["reserved"] != dict(api_requests=1, api_microdollars=protocol.reserved_microdollars_per_http, gpu_seconds=0, browser_dispatches=0)
                    or call["status"] == "reserved" or call["completion"] is None):
                raise ValueError("HTTP admission, protocol or final disposition mismatch")
            if call["status"] == "succeeded":
                record = responses.get(key)
                if not record or call["completion"]["response_sha256"] != digest(record["response"]):
                    raise ValueError("successful judge response missing or changed")
                usage = record["response"].get("usage", {})
                counts = [usage.get(k) for k in ("prompt_tokens", "completion_tokens")]
                if any(type(n) is not int or n < 0 for n in counts): raise ValueError("judge usage absent")
                amount = math.ceil((counts[0]*protocol.input_microdollars_per_million_tokens
                    + counts[1]*protocol.output_microdollars_per_million_tokens)/1_000_000)
                expected = dict(api_requests=1, api_microdollars=amount, gpu_seconds=0, browser_dispatches=0)
                if record["accounted_microdollars"] != amount or call["actual"] != expected or call["charged"] != expected:
                    raise ValueError("successful HTTP cost does not match retained token usage")
            elif call["status"] in ("unknown", "failed"):
                if key not in errors or call["charged"] != {k:max(call["actual"][k],call["reserved"][k]) for k in COST_DIMENSIONS}:
                    raise ValueError("failed/ambiguous HTTP liability was lost")
                unknown += call["status"] == "unknown"
            else: raise ValueError("unrecognized HTTP status")
            checked.add(key)
    return references, unknown, checked


def _freeze(store, contract, actor, state, prior_inventory, cache):
    ref = state.get("driver_freezes", {}).get(actor.freeze_receipt_sha256)
    if not ref or ref["sha256"] != actor.freeze_receipt_sha256: raise ValueError("missing real driver transfer receipt")
    freeze = cache.get(actor.freeze_receipt_sha256)
    if freeze is None: freeze = _json(ref)
    if (freeze.get("all_eight_transfer_rpcs_awaited") is not True or freeze["policy_sha256"] != contract.policy_sha256
            or freeze["graph_contract_sha256"] != contract.sha256 or freeze["weight_version"] != actor.weight_version
            or actor.weight_version != str(contract.update_id+1)
            or actor.endpoint_sha256 not in {digest(url) for url in freeze["endpoints"]}
            or freeze["rng_reference"]["sha256"] != actor.restore_receipt_sha256
            or freeze["checkpoint_inventory"] != prior_inventory):
        raise ValueError("actor transfer receipt differs from exact restored policy/endpoints")
    if actor.freeze_receipt_sha256 not in cache:
        rng = _json(freeze["rng_reference"])
        if (rng["policy_sha256"] != contract.policy_sha256 or rng["graph_contract_sha256"] != contract.sha256
                or rng["previous_checkpoint_inventory"] != prior_inventory):
            raise ValueError("actor transfer restore/RNG identity changed")
        cache[actor.freeze_receipt_sha256] = freeze


def _episode(store, contract, state, attempt, protocol, prior_inventory, freeze_cache):
    draw_id = state["attempts"][attempt]["draw_id"]
    request_ids = [key for key, value in state["requests"].items() if value == draw_id]
    if len(request_ids) != 1: raise ValueError("episode draw does not have one immutable request")
    envelope = state["online_requests"][request_ids[0]]
    actor = FrozenActor(**envelope["actor"])
    actor.validate(contract)
    _freeze(store, contract, actor, state, prior_inventory, freeze_cache)
    episode = load_pending_episode(store, contract, actor, attempt, verifier_sha256=physical_verifier_sha256())
    events, refs = _journals(Path(episode.artifact_directory))
    accepted = [e["record"] for e in events if e["event"] in ("physical_dispatch_accepted", "unexecuted_response_retained")]
    if len(accepted) != len(episode.turns): raise ValueError("saved response lacks physical observation/dispatch evidence")
    for captured, event in zip(episode.turns, accepted):
        edge = captured.edge
        receipt = ReplayReceipt(**event["receipt"])
        receipt.validate()
        evidence = event["evidence"]
        if (receipt.accepted is not True or receipt.evidence_sha256 != digest(evidence)
                or receipt.verifier_sha256 != physical_verifier_sha256()
                or event["edge"]["edge_id"] != edge.edge_id
                or evidence["dispatch_check"].get("passed") is not True
                or (evidence["replayed_prefix"] and evidence["replay_comparison"].get("passed") is not True)):
            raise ValueError("saved replay/dispatch comparison did not pass")
        n = sum(edge.sampled_mask)
        if n == 0: raise ValueError("empty sampled response mask")
        suffix = (198,) if edge.response_ids[n-1] == 151645 else (151645,198)
        if (n == 0 or edge.sampled_mask != (True,)*n+(False,)*len(suffix) or edge.response_ids[n:] != suffix):
            raise ValueError("synthetic transport suffix entered the sampled response mask")
    entry = state["terminal_verdicts"].get(attempt)
    if not entry: raise ValueError("complete episode has no durable terminal verdict")
    record = _json(entry); result = record["verdict"]; refs.append(entry)
    completion = state["attempts"][attempt]["completion"]
    expected = dict(schema_version=1, manifest_sha256=store.manifest.sha256, request_sha256=episode.request_sha256,
                    draw_id=draw_id, attempt_id=attempt, saved_episode_sha256=state["pending_episodes"][attempt]["sha256"],
                    terminal_evidence_sha256=digest(episode.terminal_evidence))
    if (any(record.get(k) != v for k,v in expected.items())
            or result.get("source_sha256") != store.manifest.to_dict()["science"]["judge"]["canonical_reward_source_sha256"]
            or result["status"] != completion["status"] or entry["status"] != result["status"]
            or entry["reward"] != result["reward"] or completion["outcome"]["artifact_sha256"] != entry["sha256"]):
        raise ValueError("canonical terminal verdict differs from the settled outcome")
    native = result["native_reward"]
    if native.get("judge_prompt_variant") != "action_history": raise ValueError("wrong canonical judge variant")
    judge_events, judge_refs = _journals(Path(episode.artifact_directory)/"judge"); refs.extend(judge_refs)
    if result["status"] == "valid":
        if (native.get("valid") is not True or native.get("judge_timeout") is not False
                or result["reward"] != float(native["judge"] == 1.)
                or completion["outcome"]["reward"] != result["reward"]):
            raise ValueError("valid terminal binary reward differs from canonical diagnostic")
        text = native["judge_text"]
        if text.startswith("Judge not run for "):
            last = episode.native_samples[-1]
            reason = str(last.metadata.get("terminate_reason", "")).lower()
            expected_text = ("Judge not run for terminate_reason=format_error_failed" if reason == "format_error_failed"
                             else f"Judge not run for status={last.status}")
            if text != expected_text or result["reward"] != 0. or (reason != "format_error_failed" and str(last.status).rsplit(".",1)[-1] == "COMPLETED"):
                raise ValueError("unjudged success or invalid native no-judge disposition")
        else:
            responses = [e["record"]["response"] for e in judge_events if e["event"] == "judge_http_response"]
            texts = [r["choices"][0]["message"]["content"] or "" for r in responses if r.get("choices")]
            if text not in texts or parse_judge_verdict(text) != native["judge"]:
                raise ValueError("terminal reward does not match retained provider verdict text")
    elif result["status"] != "invalid" or result["reward"] is not None:
        raise ValueError("invalid judge outcome carries a reward")
    return episode, refs


def _missing(store, state, attempt):
    """Verify the only automatically admitted missing-attempt path, pre-actor."""
    entry = state.get("infrastructure_dispositions", {}).get(attempt)
    if not entry: raise ValueError("missing attempt lacks preserved infrastructure/cleanup proof")
    root = _safe_path(store.root/"online_artifacts"/digest(attempt))
    failure_ref = dict(path=str(root/"clean-preactor-failure.json"),sha256=entry["failure_sha256"])
    record, cleanup = _json(failure_ref), _json(entry["process_cleanup_reference"])
    teardown = dict(path=str(root/"collection-teardown.json"),sha256=record["teardown_sha256"])
    closed = _json(teardown)
    request_ids = [key for key,value in state["requests"].items() if value == state["attempts"][attempt]["draw_id"]]
    if len(request_ids) != 1: raise ValueError("missing attempt has ambiguous request")
    envelope = state["online_requests"][request_ids[0]]
    if (record["manifest_sha256"] != store.manifest.sha256 or cleanup.get("attempt_id") != attempt
            or record["request_sha256"] != digest(dict(envelope["request"],attempt_id=attempt))
            or closed.get("request_sha256") != digest(envelope) or closed.get("attempt_id") != attempt
            or closed.get("manifest_sha256") != store.manifest.sha256
            or cleanup.get("owned_processes_closed") is not True or closed.get("captured_turns") != 0
            or closed.get("pending_response") is not False or closed.get("owned_clients_closed") is not True
            or closed.get("policy_failure") is not None or closed.get("verifier_sha256") != physical_verifier_sha256()
            or any(row["attempt_id"] == attempt for row in state.get("online_actor_requests", {}).values())
            or any(row["parent_attempt_id"] == attempt for row in state.get("subrequests", {}).values())):
        raise ValueError("missing attempt has unresolved sampled/API work or incomplete teardown")
    return [failure_ref,teardown,entry["process_cleanup_reference"]]


def audit_saved_collection(store, contract, pointer, *, source_inventory_ref, native_acceptance_ref,
                          judge_protocol, allocation_accounting_ref, diagnostic_only=False):
    """Audit real retained files before admitting this exact pending batch."""
    if type(diagnostic_only) is not bool or type(judge_protocol) is not JudgeHTTPProtocol:
        raise ValueError("explicit diagnostic mode and typed frozen judge protocol required")
    source = _sources(store, source_inventory_ref)
    native = _json(native_acceptance_ref)
    kind = "cpu_mock" if diagnostic_only else "production_runtime"
    required = ("actual_model","full_response_masks","tp_dp_gradient_parity","ppo2_windows",
                "native_checkpoint_restore","per_rank_rng_restore","native_controller_integration")
    if (native.get("verification_kind") != kind or native.get("manifest_sha256") != store.manifest.sha256
            or native.get("validator_source_sha256") != validator_source_sha256() or native.get("passed") is not True
            or any(native.get(key) is not True for key in required)):
        raise ValueError("source-bound native physical acceptance is absent; CPU diagnostics cannot authorize production")
    accounting = audit_allocation(allocation_accounting_ref,store.manifest)
    current, payload, _ = load_scored_pending(store,contract.update_id)
    if current != pointer or payload["graph"].contract != contract:
        raise ValueError("acceptance requested for a different pending batch")
    manifest = store.manifest.to_dict(); state = store.snapshot()
    if (state["halt"] or manifest["science"]["judge"]["http_protocol_sha256"] != judge_protocol.sha256
            or manifest["science"]["judge"]["canonical_reward_source_sha256"] != source["files"]["openwebrl/reward_browser.py"]):
        raise ValueError("halted campaign or changed judge source/protocol")
    prior = _json(payload["optimizer_rng_artifact"])["previous_checkpoint_inventory"]
    draw_ids = set(payload["leaf_to_draw"].values())
    attempts = {key:row for key,row in state["attempts"].items() if row["draw_id"] in draw_ids}
    if any(row["status"] not in ("valid","invalid","missing") or row["completion"] is None for row in attempts.values()):
        raise ValueError("collection retains an active or unsettled attempt")
    edges = {edge.edge_id:edge for edge in payload["graph"].edges}
    checked_edges, freezes, references, selected = set(), {}, [], {}
    for key, row in attempts.items():
        actual = row["completion"]["actual_cost"]
        if (actual != dict(gpu_seconds=0,api_requests=0,api_microdollars=0,browser_dispatches=1)
                or row["actual"] != actual or row["charged"] != {k:max(actual[k],row["reserved"][k]) for k in COST_DIMENSIONS}):
            raise ValueError("episode allocation/API/browser costs were duplicated or lost")
        if row["status"] == "missing": references.extend(_missing(store,state,key)); continue
        episode, refs = _episode(store,contract,state,key,judge_protocol,prior,freezes)
        references.extend(refs)
        if state["draws"][row["draw_id"]]["selected_attempt"] == key:
            selected[row["draw_id"]] = episode.leaf(status=row["status"],reward=row["completion"]["outcome"].get("reward"))
            for turn in episode.turns:
                final = edges.get(turn.edge.edge_id)
                if (final is None or replace(final,old_log_probs=turn.edge.old_log_probs,old_log_prob_source=turn.edge.old_log_prob_source) != turn.edge
                        or payload["rows"][final.edge_id] != turn.row or payload["replay_receipts"][final.edge_id] != turn.replay_receipt):
                    raise ValueError("training graph changed a selected saved response/context/mask")
                if final.edge_id not in checked_edges:
                    payload["rows"][final.edge_id].validate(final,contract,payload["tensor_payloads"][final.edge_id])
                    checked_edges.add(final.edge_id)
        del episode
    for leaf in payload["graph"].leaves:
        draw = payload["leaf_to_draw"][leaf.leaf_id]
        if draw in selected and selected[draw] != leaf:
            raise ValueError("saved terminal path/status/reward differs from scored inventory")
        if draw not in selected and (leaf.status != "missing" or leaf.edge_ids or leaf.reward is not None):
            raise ValueError("training leaf has no corresponding saved episode")
    if checked_edges != set(edges): raise ValueError("scored graph contains unverified saved responses")
    refs, unknown, http_ids = _http_costs(state,attempts,store,judge_protocol); references.extend(refs)
    for key,row in state.get("online_actor_requests",{}).items():
        if row["attempt_id"] in attempts and row["status"] == "reserved":
            raise ValueError("unresolved actor dispatch remains in completed collection")
    totals = store._totals(state)
    if any(totals[k] > manifest["budget_caps"][k] for k in COST_DIMENSIONS):
        raise ValueError("all-attempt campaign accounting exceeds approved caps")
    upper = accounting["gpu_seconds_used"]+math.ceil(time.time()-accounting["observed_unix"])*accounting["running_gpus"]
    if upper > accounting["approved_gpu_seconds_cap"]: raise ValueError("allocation cap exhausted during artifact audit")
    # Collection must remain frozen throughout auditing. Native training has
    # not started and must not mutate this ledger concurrently.
    if store.snapshot()["revision"] != state["revision"]: raise ValueError("collection ledger changed during audit")
    result = dict(schema_version=1,verification_kind=kind,diagnostic_only=diagnostic_only,passed=True,
        manifest_sha256=store.manifest.sha256,policy_sha256=contract.policy_sha256,update_id=contract.update_id,
        pending_file_sha256=pointer["file_sha256"],binding_sha256=pointer["binding_sha256"],
        ppo_schedule_sha256=pointer["ppo_schedule_sha256"],validator_source_sha256=validator_source_sha256(),
        actor_weight_transfer_verified=True,saved_episode_artifacts_verified=True,browser_replay_verified=True,
        sample_masks_verified=True,terminal_judge_artifacts_verified=True,all_attempt_costs_reconciled=True,
        source_inventory=source_inventory_ref,native_acceptance=native_acceptance_ref,
        artifact_verifier_sha256=_hash(__file__),allocation_accounting=allocation_accounting_ref,
        allocation_gpu_seconds_upper_bound=upper,ledger_revision=state["revision"],charged_resources=totals,
        attempts_checked=len(attempts),edges_checked=len(checked_edges),http_attempts_checked=len(http_ids),
        unknown_http_liabilities_retained=unknown,retained_journal_files_sha256=digest(references),
        replay_scope="Saved observable-equivalence evidence; remote server state was not cloned")
    path = store.root/"collection_audits"/str(contract.update_id)/(digest(result)+".json")
    return dict(path=str(path),sha256=_hash(path)) if path.exists() else _write(path,result)


class CollectionAcceptanceAuditor:
    """Callable driver dependency; owner supplies a fresh scheduler audit view."""
    def __init__(self, *, source_inventory_ref, native_acceptance_ref, judge_protocol,
                 allocation_accounting, diagnostic_only=False):
        self.dependencies = dict(source_inventory_ref=source_inventory_ref,native_acceptance_ref=native_acceptance_ref,
                                 judge_protocol=judge_protocol,diagnostic_only=diagnostic_only)
        self.allocation_accounting = allocation_accounting

    async def __call__(self,store,contract,pointer):
        reference = self.allocation_accounting() if callable(self.allocation_accounting) else self.allocation_accounting
        if inspect.isawaitable(reference): reference = await reference
        return await asyncio.to_thread(audit_saved_collection,store,contract,pointer,
            allocation_accounting_ref=reference,**self.dependencies)
