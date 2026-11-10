"""One isolated real browser episode and metered canonical terminal verdict.

The caller owns the allocation, immutable campaign, frozen actor transfer and
worker lifetime. This module does not submit jobs or authorize API spending.
Use separate processes for concurrent episodes because generate_browser uses
process globals. A failed or ambiguous attempt is never redispatched here.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import urlsplit

from openwebrl.action_credit import digest
from openwebrl.action_credit_collect import (
    SGLangActorTransport, run_native_episode, physical_verifier_sha256,
)
from openwebrl.action_credit_episode import load_pending_episode
from openwebrl.action_credit_judge import (
    JudgeHTTPProtocol, MeteredJudgeClient, PaidDispatchBlocked,
    canonical_action_history_judge,
)
from openwebrl.action_credit_judge_budget import (
    gpt41_context_upper_bound, input_bound_protocol_sha256,
)
from openwebrl.action_credit_state import _safe_path


def _json_artifact(path):
    raw = _safe_path(path).read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def verify_worker_environment():
    """Refuse a worker whose common libraries may default to home caches."""
    for key in (
        "HF_HOME", "HUGGINGFACE_HUB_CACHE", "TORCH_HOME", "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME", "XDG_DATA_HOME", "WANDB_CACHE_DIR", "WANDB_CONFIG_DIR",
        "WANDB_DIR", "TMPDIR", "TORCH_EXTENSIONS_DIR", "TRITON_CACHE_DIR",
        "CUDA_CACHE_PATH", "FLASHINFER_WORKSPACE_BASE",
    ):
        value = os.environ.get(key)
        if not value or not Path(value).is_absolute():
            raise ValueError("Explicit writable worker environment required: " + key)
        _safe_path(value)
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise ValueError("Worker bytecode writes must be disabled")


async def served_weight_version(generate_url):
    """Query the exact assigned engine, without router fallback or retries."""
    import httpx
    parsed = urlsplit(generate_url)
    if (parsed.scheme != "http" or not parsed.hostname or parsed.port is None
            or parsed.path != "/generate" or parsed.query or parsed.fragment
            or parsed.username or parsed.password):
        raise ValueError("Expected an explicit local SGLang engine /generate URL")
    async with httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(retries=0),
                                 timeout=30, follow_redirects=False,
                                 trust_env=False) as client:
        result = await client.get(parsed._replace(path="/get_weight_version").geturl())
        result.raise_for_status()
        version = result.json().get("weight_version")
        if type(version) not in (int, str) or not str(version):
            raise ValueError("Actor engine omitted its weight version")
        return str(version)


def _approval(store, approval_reference, protocol):
    receipt, sha = _json_artifact(approval_reference["path"])
    if (sha != approval_reference["sha256"]
            or digest(receipt) != protocol.approval_receipt_sha256):
        raise PaidDispatchBlocked("API approval file differs from the frozen protocol")
    manifest = store.manifest.to_dict()
    if (receipt.get("status") != "approved"
            or receipt.get("campaign_id") != manifest["campaign_id"]
            or receipt.get("model") != "gpt-4.1"
            or receipt.get("api_microdollars_cap") != manifest["budget_caps"]["api_microdollars"]
            or receipt.get("api_requests_cap") != manifest["budget_caps"]["api_requests"]
            or not receipt.get("user_authorization_text")
            or not receipt.get("user_authorization_reference")):
        raise PaidDispatchBlocked("An exact user-authorized campaign API receipt is required")
    return receipt


def _commit_terminal(store, episode, record, verdict, *, tree_terminal):
    """Idempotent artifact-to-ledger commit; never call a browser or judge."""
    if type(tree_terminal) is not bool or tree_terminal != (store.manifest.to_dict()["arm_id"] in ("tree_outcome_sampling", "tree_grpo")):
        raise ValueError("Terminal tree identity differs from the frozen arm")
    pointer = store.snapshot()["pending_episodes"][episode.request.attempt_id]
    result = record["verdict"]
    if (record.get("schema_version") != 1
            or record.get("manifest_sha256") != store.manifest.sha256
            or record.get("request_sha256") != episode.request_sha256
            or record.get("draw_id") != episode.draw_id
            or record.get("attempt_id") != episode.request.attempt_id
            or record.get("saved_episode_sha256") != pointer["sha256"]
            or record.get("terminal_evidence_sha256") != digest(episode.terminal_evidence)
            or result.get("status") not in ("valid", "invalid")
            or (result["status"] == "valid" and result.get("reward") not in (0., 1.))
            or (result["status"] == "invalid" and result.get("reward") is not None)):
        raise ValueError("Saved terminal verdict does not bind this exact episode")
    expected_path = _safe_path(episode.artifact_directory) / "judge/terminal-verdict.json"
    saved, saved_sha = _json_artifact(expected_path)
    if (saved != record or verdict != dict(path=str(expected_path), sha256=saved_sha)):
        raise ValueError("Terminal verdict file differs from its retained identity")
    entry = dict(verdict, saved_episode_sha256=pointer["sha256"],
                 status=result["status"], reward=result["reward"])
    with store._transaction() as state:
        saved_entries = state.setdefault("terminal_verdicts", {})
        previous = saved_entries.get(episode.request.attempt_id)
        if previous is not None and previous != entry:
            raise ValueError("Terminal verdict cannot be replaced")
        saved_entries[episode.request.attempt_id] = entry
    outcome = dict(artifact_sha256=verdict["sha256"])
    if result["status"] == "valid":
        outcome["reward"] = result["reward"]
    # Allocation wall time is charged by the owning allocation ledger, not
    # multiplied by concurrent episodes. Each HTTP attempt has its own charge.
    cost = dict(gpu_seconds=0, api_requests=0, api_microdollars=0, browser_dispatches=1)
    store.finish_attempt(episode.request.attempt_id, status=result["status"],
        actual_cost=cost, outcome=outcome,
        reason=None if result["status"] == "valid" else "canonical_terminal_judge_invalid",
        sampling_terminal_node_id=(digest([episode.request.request_id, "terminal_observation"])
                                   if tree_terminal else None))
    return episode


def reconcile_saved_verdict(store, episode, *, tree_terminal=False):
    """Finish a commit interrupted after the durable canonical verdict write.

    This accepts only the exact private, pointer-bound episode and retained
    canonical source. Missing/partial verdicts require separate reconciliation;
    they never trigger a second paid HTTP request here.
    """
    path = _safe_path(episode.artifact_directory) / "judge/terminal-verdict.json"
    record, sha = _json_artifact(path)
    if record.get("verdict", {}).get("source_sha256") != store.manifest.to_dict()["science"]["judge"].get("canonical_reward_source_sha256"):
        raise ValueError("Saved verdict came from a different canonical judge")
    return _commit_terminal(store, episode, record, dict(path=str(path), sha256=sha),
                            tree_terminal=tree_terminal)


def _settle_terminal(store, episode, result, *, journal, tree_terminal):
    """Commit the verdict artifact before closing its charged episode slot."""
    pointer = store.snapshot()["pending_episodes"][episode.request.attempt_id]
    if result["status"] not in ("valid", "invalid"):
        raise ValueError("Canonical judge returned no terminal disposition")
    if result["status"] == "valid" and result["reward"] not in (0., 1.):
        raise ValueError("Training reward must be terminal binary success")
    record = dict(schema_version=1, manifest_sha256=store.manifest.sha256,
        request_sha256=episode.request_sha256, draw_id=episode.draw_id,
        attempt_id=episode.request.attempt_id, saved_episode_sha256=pointer["sha256"],
        terminal_evidence_sha256=digest(episode.terminal_evidence), verdict=result)
    verdict = journal.blob("terminal-verdict.json", json.dumps(record, sort_keys=True,
        allow_nan=False).encode() + b"\n")
    return _commit_terminal(store, episode, record, verdict, tree_terminal=tree_terminal)


async def run_collect_and_judge(*, store, contract, request, actor, args, sample,
        sampling, browser_config, browser_manifest, reserved_cost,
        approval_reference, judge_protocol, canonical_reward_source_sha256,
        prefix=(), tree_terminal=False):
    """Concrete dispatch callback: collect, persist, judge, charge, then return.

    No injectable browser or provider callbacks are accepted by this production
    entry point. CPU unit tests exercise its pure admission/commit helpers;
    real token, browser, actor-transfer and judge acceptance is still required.
    """
    verify_worker_environment()
    if type(judge_protocol) is not JudgeHTTPProtocol:
        raise ValueError("Typed frozen judge protocol required")
    _approval(store, approval_reference, judge_protocol)
    judge = store.manifest.to_dict()["science"]["judge"]
    if (judge.get("http_protocol_sha256") != judge_protocol.sha256
            or judge.get("canonical_reward_source_sha256") != canonical_reward_source_sha256
            or judge_protocol.input_bound_protocol_sha256 != input_bound_protocol_sha256()):
        raise ValueError("Worker judge source/transport/cost protocol mismatch")
    from openwebrl import reward_browser
    if hashlib.sha256(Path(reward_browser.__file__).read_bytes()).hexdigest() != canonical_reward_source_sha256:
        raise ValueError("Canonical reward source changed before browser dispatch")
    if reserved_cost != dict(gpu_seconds=0, api_requests=0, api_microdollars=0, browser_dispatches=1):
        raise ValueError("Reserve one browser; allocation and each judge HTTP are separately charged")
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    if digest(url) != actor.endpoint_sha256:
        raise ValueError("Worker must use its exact assigned frozen SGLang engine")
    if tree_terminal != (store.manifest.to_dict()["arm_id"] in ("tree_outcome_sampling", "tree_grpo")):
        raise ValueError("Tree terminal node disposition differs from frozen arm")
    # Read credentials only into process memory; never write them to artifacts.
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("JUDGE_API_KEY")
    if not key:
        raise PaidDispatchBlocked("Approved judge has no configured API credential")
    async def version():
        return await served_weight_version(url)
    from .action_credit_collect import CollectionRejected
    try:
        episode = await run_native_episode(store=store, contract=contract, request=request,
            actor=actor, args=args, sample=sample, sampling=sampling,
            browser_config=browser_config, browser_manifest=browser_manifest,
            reserved_cost=reserved_cost, actor_call=SGLangActorTransport(),
            weight_version=version, verifier_sha256=physical_verifier_sha256(), prefix=prefix)
    except CollectionRejected as exc:
        # Only this narrow no-model/no-judge failure may close as missing.
        # The subprocess owner still has to prove detached browser cleanup
        # before recording the charged disposition in the campaign ledger.
        if prefix or not str(exc).startswith("native generator returned infrastructure failure:"):
            raise
        record_clean_pre_actor_failure(store, request, str(exc))
        return None
    # Reconstruct from the committed private pointer; never judge a merely
    # in-memory or partially written trajectory.
    return await judge_existing_pending_episode(store=store, contract=contract,
        actor=actor, attempt_id=request.attempt_id, args=args,
        approval_reference=approval_reference, judge_protocol=judge_protocol,
        canonical_reward_source_sha256=canonical_reward_source_sha256,
        tree_terminal=tree_terminal)


def _pre_actor_failure_checks(store, request):
    state = store.snapshot()
    attempt = state["attempts"].get(request.attempt_id)
    if not attempt or request.attempt_id in state.get("pending_episodes", {}):
        raise ValueError("Pre-actor failure requires its own uncompleted collection attempt")
    if any(call["attempt_id"] == request.attempt_id
           for call in state.get("online_actor_requests", {}).values()):
        raise ValueError("Actor dispatch exists; preserve partial response for reconciliation")
    if any(call["parent_attempt_id"] == request.attempt_id
           for call in state.get("subrequests", {}).values()):
        raise ValueError("Paid dispatch exists; reconcile its charge and verdict")
    root = _safe_path(store.root / "online_artifacts" / digest(request.attempt_id))
    teardown, sha = _json_artifact(root / "collection-teardown.json")
    envelope = state.get("online_requests", {}).get(request.request_id)
    if (not envelope or teardown.get("request_sha256") != digest(envelope)
            or teardown.get("attempt_id") != request.attempt_id
            or teardown.get("manifest_sha256") != store.manifest.sha256
            or teardown.get("owned_clients_closed") is not True
            or teardown.get("captured_turns") != 0 or teardown.get("pending_response") is not False
            or teardown.get("policy_failure") is not None
            or teardown.get("verifier_sha256") != physical_verifier_sha256()):
        raise ValueError("Collection teardown does not prove an unsampled infrastructure failure")
    return root, sha


def record_clean_pre_actor_failure(store, request, reason):
    """Child artifact only. Missing status awaits parent process cleanup proof."""
    from openwebrl.controlled_sft_worker import AttemptJournal
    if not reason.startswith("native generator returned infrastructure failure:"):
        raise ValueError("Only the explicit native infrastructure disposition is admitted")
    root, sha = _pre_actor_failure_checks(store, request)
    record = dict(schema_version=1, request_sha256=digest(asdict(request)),
        manifest_sha256=store.manifest.sha256, teardown_sha256=sha, reason=reason)
    AttemptJournal(root).blob("clean-preactor-failure.json", json.dumps(record,
        sort_keys=True, allow_nan=False).encode()+b"\n")


def settle_clean_pre_actor_failure(store, request, *, process_cleanup_reference):
    """Parent-only settlement after its real registered-child cleanup receipt.

    The process owner creates and checks that receipt; this function links it
    to the retained no-model/no-judge failure evidence and charges the attempt.
    """
    root, teardown_sha = _pre_actor_failure_checks(store, request)
    record, sha = _json_artifact(root / "clean-preactor-failure.json")
    cleanup, cleanup_sha = _json_artifact(process_cleanup_reference["path"])
    if (cleanup_sha != process_cleanup_reference["sha256"]
            or cleanup.get("attempt_id") != request.attempt_id
            or cleanup.get("owned_processes_closed") is not True
            or record.get("request_sha256") != digest(asdict(request))
            or record.get("manifest_sha256") != store.manifest.sha256
            or record.get("teardown_sha256") != teardown_sha
            or not record.get("reason", "").startswith("native generator returned infrastructure failure:")):
        raise ValueError("Missing disposition requires exact artifact and process cleanup proof")
    with store._transaction() as state:
        entries = state.setdefault("infrastructure_dispositions", {})
        entry = dict(failure_sha256=sha, process_cleanup_reference=process_cleanup_reference)
        if request.attempt_id in entries and entries[request.attempt_id] != entry:
            raise ValueError("Infrastructure disposition cannot be replaced")
        entries[request.attempt_id] = entry
    store.finish_attempt(request.attempt_id, status="missing",
        actual_cost=dict(gpu_seconds=0, api_requests=0, api_microdollars=0, browser_dispatches=1),
        reason="native_infrastructure_failure_before_actor_dispatch")


async def judge_existing_pending_episode(*, store, contract, actor, attempt_id,
        args, approval_reference, judge_protocol, canonical_reward_source_sha256,
        tree_terminal=False):
    """Judge a committed, previously unjudged episode without recollecting it.

    An existing complete verdict is committed idempotently. Any earlier paid
    request without a complete verdict requires explicit reconciliation from
    its retained response/error, rather than a hidden retry after restart.
    """
    verify_worker_environment()
    judge = store.manifest.to_dict()["science"]["judge"]
    if (type(judge_protocol) is not JudgeHTTPProtocol
            or judge.get("http_protocol_sha256") != judge_protocol.sha256
            or judge_protocol.input_bound_protocol_sha256 != input_bound_protocol_sha256()):
        raise ValueError("Recovered judge transport/cost protocol differs from frozen campaign")
    approval = _approval(store, approval_reference, judge_protocol)
    if tree_terminal != (store.manifest.to_dict()["arm_id"] in ("tree_outcome_sampling", "tree_grpo")):
        raise ValueError("Terminal tree identity differs from the frozen arm")
    if canonical_reward_source_sha256 != store.manifest.to_dict()["science"]["judge"].get("canonical_reward_source_sha256"):
        raise ValueError("Canonical judge source differs from campaign")
    episode = load_pending_episode(store, contract, actor, attempt_id,
                                   verifier_sha256=physical_verifier_sha256())
    verdict_path = _safe_path(episode.artifact_directory) / "judge/terminal-verdict.json"
    if verdict_path.exists():
        return reconcile_saved_verdict(store, episode, tree_terminal=tree_terminal)
    state = store.snapshot()
    if (state["attempts"][attempt_id]["status"] != "active"
            or any(call["parent_attempt_id"] == attempt_id
                   for call in state.get("subrequests", {}).values())):
        raise PaidDispatchBlocked("Existing judge attempt needs response reconciliation; do not resend")
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("JUDGE_API_KEY")
    if not key:
        raise PaidDispatchBlocked("Approved judge has no configured API credential")
    from openwebrl.controlled_sft_worker import AttemptJournal
    from openai import AsyncOpenAI
    import httpx
    journal = AttemptJournal(_safe_path(episode.artifact_directory) / "judge")
    async with httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(retries=0),
                                 timeout=180, follow_redirects=False, trust_env=False) as http:
        async with AsyncOpenAI(api_key=key, base_url="https://api.openai.com/v1",
                               max_retries=0, http_client=http) as provider:
            metered = MeteredJudgeClient(store=store, parent_attempt_id=attempt_id,
                protocol=judge_protocol, approval_receipt=approval,
                provider_create=provider.chat.completions.create, provider_max_retries=0,
                input_token_upper_bound=gpt41_context_upper_bound,
                input_bound_protocol_sha256=input_bound_protocol_sha256(), journal=journal)
            result = await canonical_action_history_judge(args=args,
                samples=episode.native_samples, client=metered, journal=journal,
                source_sha256=canonical_reward_source_sha256)
    return _settle_terminal(store, episode, result, journal=journal, tree_terminal=tree_terminal)
