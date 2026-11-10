"""Opt-in native browser collection boundary for action-credit experiments.

This module calls the existing browser generator; no running controller imports
it. It records requests before dispatch, preserves exact processor tensors and
sampled token masks, and rejects state changes before executing an action. A
replayed prefix must match both observable browser evidence and actor context.
An observation check is not a clone of hidden remote state. Returned graphs
remain unscored drafts until the separately frozen native actor scores them.

One worker process may own one native episode at a time: the upstream generator
uses module globals. Separate workers can run concurrently. Budget admission,
policy freeze/restore verification, terminal judging, physical acceptance and
optimizer commit are separate required controller responsibilities. This file
neither requests resources nor supplies default budgets or judge decisions.
"""
from __future__ import annotations

import asyncio
import base64
from copy import copy, deepcopy
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Mapping

from openwebrl.action_credit import CollectionContract, Leaf, ResponseEdge, ResponseGraph, _sha, digest
from openwebrl.action_credit_binding import _check_contract
from openwebrl.action_credit_megatron import BackendRow, tensor_payload_sha256
from openwebrl.action_credit_receipts import ReplayReceipt
from openwebrl.action_credit_state import CampaignStore, StateError, _safe_path
from openwebrl.action_credit_tensor_refs import create_tensor_blob, tensor_inventory
from openwebrl.arm_branch_protocol import compare_observations, observation_evidence


class CollectionRejected(ValueError):
    pass


class ReconcileRequired(RuntimeError):
    """A persisted attempt may have dispatched; never silently send it again."""


@dataclass(frozen=True)
class FrozenActor:
    policy_sha256: str
    weight_version: str
    restore_receipt_sha256: str
    freeze_receipt_sha256: str
    endpoint_sha256: str

    def validate(self, contract):
        for name in ("policy_sha256", "restore_receipt_sha256", "freeze_receipt_sha256", "endpoint_sha256"):
            _sha(getattr(self, name), name)
        if self.policy_sha256 != contract.policy_sha256 or not self.weight_version:
            raise CollectionRejected("actor identity differs from the collection policy")


@dataclass(frozen=True)
class EpisodeRequest:
    """Before-dispatch identity. Digests bind actual runtime inputs, not aliases."""
    request_id: str
    attempt_id: str
    leaf_id: str
    task_id: str
    state_id: str
    action_id: str
    draw_slot: int
    draw_role: str
    group_id: str
    tree_id: str
    task_data_sha256: str
    tools_sha256: str
    environment_policy_sha256: str
    browser_config_sha256: str
    browser_manifest_sha256: str
    generator_protocol_sha256: str
    prefix_sha256: str
    inventory_role: str = "base"
    generation_seed: int = 0
    planned_candidate_edge_id: str | None = None
    candidate_prefix_length: int | None = None

    def spec(self):
        return {k: getattr(self, k) for k in
                ("task_id", "state_id", "action_id", "draw_slot", "draw_role", "group_id", "tree_id", "inventory_role")}


def physical_verifier_sha256():
    """Actual replay/capture implementation identity, not an authority receipt."""
    root = Path(__file__).resolve().parent
    names = ("action_credit_collect.py", "action_credit_episode.py", "action_credit_tensor_refs.py", "generate_browser.py",
             "arm_branch_protocol.py", "arm_branch_browser.py",
             "controlled_browser_server.py", "controlled_sft_worker.py")
    return digest({name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names})


def generator_protocol(args):
    """Pin explicit history/context arguments and resolved thinking-tag mode."""
    from openwebrl.response_format import get_browser_response_format_mode
    keys = ("max_steps", "context_num_screenshots", "turn_history_reasoning_mode",
            "browser_history_reasoning_max_turns", "browser_include_tool_response",
            "browser_history_env_feedback_max_chars", "rollout_max_context_len",
            "max_consecutive_parse_failures", "serial_action_variant")
    if any(not hasattr(args, k) for k in keys) or not hasattr(args, "browser_response_format_mode"):
        raise CollectionRejected("set all native context/history/format controls explicitly")
    return dict(arguments={k: getattr(args, k) for k in keys},
                response_mode=asdict(get_browser_response_format_mode(args.browser_response_format_mode)))


def _admit(store, contract, request, actor, sampling, max_turns, reserved_cost):
    contract.validate()
    actor.validate(contract)
    for name, value in asdict(request).items():
        if name in ("generation_seed", "planned_candidate_edge_id", "candidate_prefix_length"):
            continue
        if name.endswith("sha256"):
            _sha(value, name)
        elif name != "draw_slot" and (not isinstance(value, str) or not value.strip()):
            raise CollectionRejected("missing request identity: " + name)
    if type(request.generation_seed) is not int or not 0 <= request.generation_seed < 2**63:
        raise CollectionRejected("explicit nonnegative integer generation seed required")
    if request.planned_candidate_edge_id is not None:
        if (request.planned_candidate_edge_id != request.action_id
                or request.inventory_role not in ("q_fit", "q_holdout")
                or type(request.candidate_prefix_length) is not int
                or not 0 <= request.candidate_prefix_length < max_turns):
            raise CollectionRejected("candidate edge must equal the preplanned Q action at an explicit prefix boundary")
    elif request.candidate_prefix_length is not None:
        raise CollectionRejected("candidate boundary without a planned candidate edge")
    manifest = store.manifest.to_dict()
    _check_contract(manifest, contract)
    science = manifest["science"]
    generation = science["generation"]
    expected = dict(temperature=generation["temperature"], top_p=generation["top_p"],
                    top_k=generation["top_k"], max_new_tokens=generation["max_response_tokens"])
    if (digest(sampling) != contract.generation_sha256
            or any(sampling.get(k) != v for k, v in expected.items())
            or type(max_turns) is not int or max_turns != generation["max_turns"]):
        raise CollectionRejected("actual generation settings differ from frozen science")
    # A browser dispatch is one launched episode, including a replay attempt.
    # Individual step/tool interactions are separately retained diagnostics.
    if reserved_cost.get("browser_dispatches", 0) < 1:
        raise CollectionRejected("reserve a browser episode before collection")
    actor_cap = science["algorithm"].get("local_actor_requests_cap")
    if type(actor_cap) is not int or actor_cap <= 0:
        raise CollectionRejected("explicit positive local actor request cap required")
    # Attempt identity is separate: a terminal failed attempt may be retried in
    # the same slot with another fully charged attempt, never another request.
    request_identity = asdict(request)
    request_identity.pop("attempt_id")
    envelope = dict(request=request_identity, contract_sha256=contract.sha256,
                    actor=asdict(actor), sampling=sampling, max_turns=max_turns)
    # Extra ledger fields use the same checksummed atomic-write/flock boundary.
    # Reserve the slot first; a crash between these transactions spends nothing
    # and cannot dispatch until the immutable envelope and attempt exist.
    result = store.reserve_draw(request.request_id, update_id=contract.update_id,
                                policy_sha256=contract.policy_sha256, **request.spec())
    with store._transaction() as state:
        requests = state.setdefault("online_requests", {})
        previous = requests.get(request.request_id)
        if previous is not None and previous != envelope:
            raise CollectionRejected("request ID reused with incompatible runtime inputs")
        requests[request.request_id] = deepcopy(envelope)
    attempt = store.start_attempt(result["draw_id"], request.attempt_id, reserved_cost=reserved_cost)
    if not attempt["created"]:
        raise ReconcileRequired("persisted attempt must be reconciled, not redispatched")
    return result["draw_id"], digest(envelope)


@dataclass(frozen=True)
class ActorOutput:
    """Raw sampler tokens, before the browser transport appends suffix tokens."""
    text: str
    token_ids: tuple[int, ...]
    sampler_log_probs: tuple[float, ...]
    finish_type: str
    input_ids: tuple[int, ...]

    def normalized(self):
        if (not self.token_ids or len(self.token_ids) != len(self.sampler_log_probs)
                or any(type(t) is not int or t < 0 for t in self.token_ids)
                or any(isinstance(p, bool) or not math.isfinite(p) or p > 1e-7 for p in self.sampler_log_probs)
                or self.finish_type not in ("stop", "length") or not isinstance(self.text, str)):
            raise CollectionRejected("malformed or interrupted actor response")
        # Native Qwen transport convention. The tokenizer digest and actual GPU
        # acceptance must verify these IDs; appended tokens are never sampled.
        suffix = (198,) if self.token_ids[-1] == 151645 else (151645, 198)
        text = self.text.rstrip()
        if not text.endswith("<|im_end|>"):
            text += "<|im_end|>"
        ids = self.token_ids + suffix
        return ((text + "\n", list(ids), list(self.sampler_log_probs + (0.,) * len(suffix)), self.finish_type),
                (True,) * len(self.token_ids) + (False,) * len(suffix))


class SGLangActorTransport:
    """Concrete raw /generate transport; native teacher-forced scores come later.

    Full input logprob token IDs must agree with local processor IDs. If a
    backend cannot expose those IDs accurately, collection fails closed until
    its separately tested token-identity adapter is supplied.

    The default owns one direct HTTP request with transport retries disabled.
    Do not pass slime.http_utils.post: its retry/fallback loops hide additional
    charged actor requests. A custom transport requires a separately validated
    single-attempt contract; a declaration alone is not runtime evidence.
    """
    def __init__(self, post=None, *, transport_attempts=None):
        if post is not None and transport_attempts != 1:
            raise ValueError("custom actor transport must expose exactly one HTTP attempt")
        self.post = self._single_post if post is None else post

    @staticmethod
    async def _single_post(url, payload):
        import httpx
        async with httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(retries=0),
                                     timeout=None, follow_redirects=False, trust_env=False) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return response.json()

    async def __call__(self, *, url, text, sampling, images, input_ids, timeout):
        payload = dict(text=text, sampling_params=sampling, return_logprob=True, logprob_start_len=0)
        if images:
            payload["image_data"] = images
        call = self.post(url, payload)
        result = await asyncio.wait_for(call, timeout) if timeout is not None else await call
        meta = result.get("meta_info", {})
        output, inputs = meta.get("output_token_logprobs"), meta.get("input_token_logprobs")
        if not output or not inputs:
            raise CollectionRejected("actor omitted exact input/output token provenance")
        ids = tuple(item[1] for item in inputs)
        if ids != tuple(input_ids):
            raise CollectionRejected("server token IDs differ from captured processor context")
        return ActorOutput(result["text"], tuple(item[1] for item in output),
                           tuple(item[0] for item in output), meta.get("finish_reason", {}).get("type"), ids)


def _media(images):
    result = []
    for image in images:
        if isinstance(image, str):
            image = base64.b64decode(image.split(",", 1)[-1] if image.startswith("data:") else image, validate=True)
        if not isinstance(image, bytes):
            raise CollectionRejected("source media bytes/base64 required")
        result.append(image)
    return tuple(result)


@dataclass(frozen=True)
class CapturedResponse:
    edge: ResponseEdge
    row: BackendRow
    replay_receipt: ReplayReceipt
    output: tuple
    before_evidence_json: str
    before_image: bytes
    tensors: Mapping[str, object]
    evidence_sha256: str

    @property
    def sha256(self):
        return digest(dict(edge=asdict(self.edge), row=asdict(self.row), output=self.output,
                           evidence_sha256=self.evidence_sha256))


def prefix_sha256(prefix):
    return digest(tuple(turn.sha256 for turn in prefix))


class NativeCollectionPolicy:
    """The existing generate_browser selector hook captures unretokenized rows."""
    def __init__(self, *, store, contract, request, actor, actor_call, weight_version,
                 sampling, max_turns, prefix, journal, verifier_sha256):
        self.store = store
        self.contract, self.request, self.actor = contract, request, actor
        self.actor_call, self.weight_version = actor_call, weight_version
        self.sampling, self.max_turns = deepcopy(sampling), max_turns
        self.prefix, self.journal, self.verifier_sha256 = tuple(prefix), journal, verifier_sha256
        _sha(verifier_sha256, "verifier_sha256")
        self.turns, self.failure, self.pending, self.context = [], None, None, None
        self.capture_current = None

    def set_training_context(self, context):
        import torch
        ids = tuple(context["prompt_tokens"])
        if not ids or any(type(t) is not int or t < 0 for t in ids):
            raise CollectionRejected("invalid processor input IDs")
        tensors = context.get("multimodal_train_inputs") or {}
        if any(not isinstance(x, torch.Tensor) for x in tensors.values()):
            raise CollectionRejected("non-tensor processor payload")
        self.context = (ids, {k: v.detach().cpu().clone() for k, v in tensors.items()})

    async def check_actor(self):
        if str(await self.weight_version()) != self.actor.weight_version:
            raise CollectionRejected("actor weights changed during frozen collection")

    def actor_dispatch(self, index, *, response=None, error=None):
        key = digest((self.request.attempt_id, index, "actor_transport"))
        with self.store._transaction() as state:
            calls = state.setdefault("online_actor_requests", {})
            if response is None and error is None:
                cap = self.store.manifest.to_dict()["science"]["algorithm"]["local_actor_requests_cap"]
                if (state["halt"] or key in calls or len(calls) >= cap
                        or state["attempts"][self.request.attempt_id]["status"] != "active"):
                    raise CollectionRejected("actor request cap/identity/attempt admission failed")
                calls[key] = dict(attempt_id=self.request.attempt_id, turn=index, status="reserved")
            else:
                if key not in calls or calls[key]["status"] != "reserved":
                    raise ReconcileRequired("actor transport completion is not a fresh reserved request")
                calls[key].update(status="received" if response is not None else "failed",
                                  response_sha256=digest(asdict(response)) if response is not None else None,
                                  error=error)

    async def __call__(self, **kwargs):
        try:
            return await self._decide(**kwargs)
        except BaseException as exc:
            self.failure = type(exc).__name__ + ": " + str(exc)
            self.journal("collection_rejected", dict(reason=self.failure))
            raise

    async def _decide(self, *, infer, url, input_text, sampling_params, images,
                      observation, history, task, task_id, turn, timeout):
        if self.pending is not None:
            # Native parse failure consumes a policy turn but executes no tool.
            # Preserve that sampled response instead of silently dropping the
            # failure or changing the group denominator.
            if self.capture_current is None:
                raise CollectionRejected("pending response lacks an owned browser")
            await self.before_dispatch(await self.capture_current(), action_will_dispatch=False)
        index = len(self.turns)
        if (self.pending is not None or self.context is None or task_id != self.request.task_id
                or turn != index or index >= self.max_turns or sampling_params != self.sampling
                or digest(url) != self.actor.endpoint_sha256):
            raise CollectionRejected("unexpected turn/task/settings or undelivered action")
        ids, tensors = self.context
        self.context = None
        before = observation_evidence(observation)
        media = _media(images)
        media_sha = tuple(hashlib.sha256(x).hexdigest() for x in media)
        if not media or media[-1] != observation["screenshot"]:
            raise CollectionRejected("current screenshot differs from actual actor media")
        await self.check_actor()
        replay_comparison = None
        if index < len(self.prefix):
            source = self.prefix[index]
            source.row.validate(source.edge, self.contract, source.tensors)
            if (source.edge.contract_sha256 != self.contract.sha256
                    or source.edge.input_ids != ids or source.edge.media_sha256 != media_sha
                    or source.edge.tree_id != self.request.tree_id
                    or source.row.tensor_payload_sha256 != tensor_payload_sha256(tensors)
                    or tuple(source.output[1]) != source.edge.response_ids
                    or source.edge.parent_edge_id != (self.turns[-1].edge.edge_id if self.turns else None)
                    or source.edge.remaining_turns != self.max_turns-index):
                raise CollectionRejected("replay prefix is not the same policy/context/tree")
            replay_comparison = compare_observations(json.loads(source.before_evidence_json), before,
                                                    source.before_image, observation["screenshot"])
            output, mask = deepcopy(source.output), source.edge.sampled_mask
        else:
            self.actor_dispatch(index)
            try:
                transport_sampling = dict(self.sampling, sampling_seed=int(digest(
                    [self.request.generation_seed, index, "actor_sampling_seed"])[:8], 16) % (2**31))
                raw = await self.actor_call(url=url, text=input_text, sampling=transport_sampling, images=images,
                                            input_ids=ids, timeout=timeout)
                if not isinstance(raw, ActorOutput):
                    raise CollectionRejected("typed raw actor response required")
                self.actor_dispatch(index, response=raw)
            except BaseException as exc:
                if self.store.snapshot()["online_actor_requests"][digest((self.request.attempt_id, index, "actor_transport"))]["status"] == "reserved":
                    self.actor_dispatch(index, error=type(exc).__name__)
                raise
            if not isinstance(raw, ActorOutput) or raw.input_ids != ids:
                raise CollectionRejected("actor did not attest exact processor input IDs")
            output, mask = raw.normalized()
        await self.check_actor()
        edge_id = (self.prefix[index].edge.edge_id if index < len(self.prefix)
                   else digest((self.request.attempt_id, index, "sampled_response")))
        if index == self.request.candidate_prefix_length:
            if index < len(self.prefix) and edge_id != self.request.planned_candidate_edge_id:
                raise CollectionRejected("replayed candidate differs from preplanned candidate edge")
            edge_id = self.request.planned_candidate_edge_id
        evidence = dict(before=before, replayed_prefix=index < len(self.prefix),
                        replay_comparison=replay_comparison, actor=asdict(self.actor),
                        context_sha256=digest(dict(input_ids=ids, media_sha256=media_sha)))
        self.journal("response_sampled_before_dispatch", dict(edge_id=edge_id, output=output,
            sampled_mask=mask, input_ids=ids, media_sha256=media_sha, evidence=evidence))
        # Retain source pixels and exact processor tensors, even on later failure.
        self.journal.blob(f"{index:04d}-before.png", observation["screenshot"])
        if index < len(self.prefix):
            # The fresh replay context already passed exact tensor equality.
            # Retain its evidence, but reuse the original sampled edge's owner.
            tensors = self.prefix[index].tensors
            inventory = tensor_inventory(self.store, (tensors,))
            self.journal.blob(f"{index:04d}-processor-reference.json",
                              json.dumps(inventory, sort_keys=True).encode())
        else:
            tensors = create_tensor_blob(self.store, self.request.attempt_id, index, tensors)
        self.journal("processor_owner_retained", dict(reference=tensors.ref.to_dict()))
        self.pending = dict(edge_id=edge_id, ids=ids, media_sha=media_sha, tensors=tensors,
                            before=before, image=observation["screenshot"], output=output,
                            mask=mask, evidence=evidence, index=index)
        return deepcopy(output), dict(action_credit_request=self.request.request_id, edge_id=edge_id,
                                      sampled_mask=list(mask), old_scores="pending_frozen_actor_score")

    async def before_dispatch(self, fresh, *, action_will_dispatch=True):
        """Accept a sampled response, distinguishing actions from parse failures."""
        try:
            if self.pending is None:
                raise CollectionRejected("browser dispatch without sampled/replayed response")
            p = self.pending
            actual = observation_evidence(fresh)
            comparison = compare_observations(p["before"], actual, p["image"], fresh["screenshot"])
            await self.check_actor()
            evidence = dict(p["evidence"], dispatch_check=comparison, dispatch_observation=actual,
                            action_will_dispatch=action_will_dispatch)
            evidence_sha = digest(evidence)
            state_sha = digest(p["before"]["snapshot"])
            receipt = ReplayReceipt(p["edge_id"], self.contract.sha256, state_sha,
                digest(dict(input_ids=p["ids"], media_sha256=p["media_sha"])),
                self.max_turns - p["index"], self.verifier_sha256, evidence_sha, True)
            edge = ResponseEdge(p["edge_id"], self.request.task_id, self.request.tree_id,
                self.turns[-1].edge.edge_id if self.turns else None, self.contract.sha256,
                p["ids"], p["media_sha"], tuple(p["output"][1]), tuple(p["mask"]),
                (0.,) * len(p["mask"]), self.contract.policy_sha256, state_sha, receipt.sha256,
                self.max_turns-p["index"], "pending_frozen_actor_score")
            row = BackendRow(edge.edge_id, edge.input_ids, edge.response_ids, edge.sampled_mask,
                             edge.media_sha256, self.contract.processor_sha256,
                             tensor_payload_sha256(p["tensors"]))
            row.validate(edge, self.contract, p["tensors"])
            self.journal("physical_dispatch_accepted" if action_will_dispatch else "unexecuted_response_retained",
                         dict(edge=asdict(edge), row=asdict(row),
                                                           receipt=asdict(receipt), evidence=evidence))
            captured = CapturedResponse(edge, row, receipt, p["output"], json.dumps(p["before"], sort_keys=True),
                                        p["image"], p["tensors"], evidence_sha)
            if p["index"] < len(self.prefix):
                # Keep the original sampled edge/receipt, while preserving the
                # new replay check separately. No duplicate PPO sample is made.
                captured = self.prefix[p["index"]]
            self.turns.append(captured)
            self.pending = None
        except BaseException as exc:
            self.failure = type(exc).__name__ + ": " + str(exc)
            self.journal("dispatch_rejected", dict(reason=self.failure))
            raise


@dataclass(frozen=True)
class PendingEpisode:
    draw_id: str
    request_sha256: str
    request: EpisodeRequest
    turns: tuple[CapturedResponse, ...]
    terminal_evidence: dict
    native_samples: tuple
    artifact_directory: str

    def leaf(self, *, status, reward):
        return Leaf(self.request.leaf_id, self.request.task_id, self.request.tree_id,
                    tuple(x.edge.edge_id for x in self.turns), status, reward,
                    self.request.draw_role, self.request.group_id)


_ACTIVE_GENERATORS = set()


async def run_native_episode(*, store: CampaignStore, contract: CollectionContract,
        request: EpisodeRequest, actor: FrozenActor, args, sample, sampling,
        browser_config, browser_manifest, reserved_cost, actor_call, weight_version,
        verifier_sha256, prefix=(), generation=None, capture=None, local_transport=None):
    """Run one admitted native episode; judging/scoring/settlement remain explicit.

    The returned pending episode is not a valid training record. The controller
    must persist actual terminal verdicts/costs, score the unchanged actor, seal
    the complete planned panel, and validate its binding before optimization.
    A crash leaves its charge and dispatch intent in the ledger for reconciliation.
    """
    if generation is None:
        from openwebrl import generate_browser as generation
    if capture is None:
        from openwebrl.controlled_browser_server import fresh_observation as capture
    if id(generation) in _ACTIVE_GENERATORS:
        raise CollectionRejected("one episode per worker; use separate worker processes")
    if verifier_sha256 != physical_verifier_sha256():
        raise CollectionRejected("physical verifier sources differ from their receipt identity")
    actual_generator = generator_protocol(args)
    if (digest(actual_generator) != request.generator_protocol_sha256
            or store.manifest.to_dict()["science"]["generation"].get("native_protocol_sha256") != digest(actual_generator)):
        raise CollectionRejected("actual native history/context protocol differs from frozen campaign/request")
    if (request.prefix_sha256 != prefix_sha256(prefix) or digest(browser_config) != request.browser_config_sha256
            or digest(browser_manifest) != request.browser_manifest_sha256):
        raise CollectionRejected("prefix or browser config changed after request planning")
    if request.planned_candidate_edge_id is not None:
        boundary = request.candidate_prefix_length
        if (type(boundary) is not int or len(prefix) not in (boundary, boundary+1)
                or (len(prefix) == boundary+1 and prefix[-1].edge.edge_id != request.planned_candidate_edge_id)):
            raise CollectionRejected("candidate dispatch must sample or replay exactly its declared prefix boundary")
    if (browser_config.get("mode") != "local_process"
            or browser_config.get("local_process", {}).get("server_module") != "openwebrl.arm_branch_browser"):
        raise CollectionRejected("strict branch-evidence local browser required")
    local_cfg = browser_config["local_process"]
    from openwebrl import arm_branch_browser
    branch_source_sha = hashlib.sha256(Path(arm_branch_browser.__file__).read_bytes()).hexdigest()
    if local_cfg.get("server_module_sha256") != branch_source_sha:
        raise CollectionRejected("branch server source must be pinned")
    expected_browser = dict(width=1280, height=720, dpr=1, use_screenshot=True,
                            use_a11ytree=False, resize_output_coords=True, resize_scale=1000)
    if any(browser_config.get(k) != v for k, v in expected_browser.items()):
        raise CollectionRejected("browser observation/coordinate contract changed")
    for name in ("log_dir", "port_lock_dir"):
        if not isinstance(local_cfg.get(name), str) or not Path(local_cfg[name]).is_absolute():
            raise CollectionRejected("explicit private browser paths required")
        _safe_path(local_cfg[name])
    if browser_manifest.get("branch_state_version") not in ("observable-replay-v1", "visible-replay-v2"):
        raise CollectionRejected("explicit frozen branch replay protocol required")
    if getattr(args, "browser_action_selector", None) is not None:
        raise CollectionRejected("cannot replace an existing action selector")
    max_turns = store.manifest.to_dict()["science"]["generation"]["max_turns"]
    if getattr(args, "max_steps", None) != max_turns:
        raise CollectionRejected("native max_steps must match the frozen turn horizon")
    # Reject foreign/missing prefix owners before admitting another dispatch.
    tensor_inventory(store, (turn.tensors for turn in prefix))
    draw_id, request_sha = _admit(store, contract, request, actor, sampling, max_turns, reserved_cost)
    from openwebrl.controlled_sft_worker import AttemptJournal
    artifacts = _safe_path(store.root / "online_artifacts" / digest(request.attempt_id))
    journal = AttemptJournal(artifacts)
    journal("dispatch_intent", dict(request_sha256=request_sha, draw_id=draw_id))
    from openwebrl.controlled_browser_server import validate_browser_manifest
    child_manifest = dict(browser_manifest, artifact_directory=str(_safe_path(artifacts / "browser")))
    validate_browser_manifest(child_manifest)
    manifest_blob = journal.blob("browser-manifest.json", json.dumps(child_manifest, sort_keys=True).encode())
    child_env = dict(OPENWEBRL_CONTROLLED_BROWSER_MANIFEST=manifest_blob["path"],
                     OPENWEBRL_CONTROLLED_BROWSER_MANIFEST_SHA256=manifest_blob["sha256"])
    if local_transport is None:
        from openwebrl.env import local_process_env as local_transport
    policy = NativeCollectionPolicy(store=store, contract=contract, request=request, actor=actor,
        actor_call=actor_call, weight_version=weight_version, sampling=sampling,
        max_turns=max_turns, prefix=prefix, journal=journal, verifier_sha256=verifier_sha256)
    call_args, terminal = copy(args), {}
    call_args.browser_action_selector = policy
    call_args.path_to_save_generated_samples = str(_safe_path(artifacts / "native_rollouts"))
    original_create, original_resources = generation._create_env, generation._load_local_resources
    original_override = generation._apply_local_process_env_overrides
    original_server_module = local_transport._server_module
    saved_env = {key: os.environ.get(key) for key in child_env}
    clients = []

    class OwnedClient:
        def __init__(self, client):
            self.client, self.closed, self.closing = client, False, None
        def __getattr__(self, name):
            return getattr(self.client, name)
        async def step(self, actions):
            try:
                await policy.before_dispatch(await capture(self.client))
                return await self.client.step(actions)
            except BaseException as exc:
                policy.failure = type(exc).__name__ + ": " + str(exc)
                raise
        async def _close(self):
            try:
                observation = await capture(self.client)
                if policy.pending is not None and policy.failure is None:
                    await policy.before_dispatch(observation, action_will_dispatch=False)
                terminal["observation"] = observation_evidence(observation)
                journal.blob("terminal.png", observation["screenshot"])
                journal("terminal_observation", terminal["observation"])
            finally:
                await self.client.exit()
                self.closed = True
        async def exit(self):
            if self.closing is None:
                self.closing = asyncio.create_task(self._close())
            await asyncio.shield(self.closing)

    def resources(*pos, **kw):
        loaded = original_resources(*pos, **kw)
        if (digest(loaded[0]) != request.task_data_sha256 or digest(loaded[1]) != request.tools_sha256
                or digest(loaded[2]) != request.environment_policy_sha256):
            raise CollectionRejected("actual task/tools/environment policy differ from frozen request")
        return loaded

    async def create(*pos, **kw):
        if pos[0] != request.task_id or digest(pos[1]) != request.browser_config_sha256:
            raise CollectionRejected("native browser task/config mismatch")
        await policy.check_actor()
        client, task_data = await original_create(*pos, **kw)
        wrapped = OwnedClient(client)
        clients.append(wrapped)
        policy.capture_current = lambda: capture(client)
        return wrapped, task_data

    def server_module(config):
        # Existing transport allows the controlled server only. Extend the
        # isolated worker instance to the independently hash-pinned branch
        # subclass, keeping the shared transport's launch/ownership/cleanup.
        if (config.get("server_module") != "openwebrl.arm_branch_browser"
                or config.get("server_module_sha256") != branch_source_sha
                or hashlib.sha256(Path(arm_branch_browser.__file__).read_bytes()).hexdigest() != branch_source_sha):
            raise CollectionRejected("unexpected or changed local server source")
        return "openwebrl.arm_branch_browser"

    _ACTIVE_GENERATORS.add(id(generation))
    generation._create_env, generation._load_local_resources = create, resources
    generation._apply_local_process_env_overrides = lambda _: deepcopy(browser_config)
    local_transport._server_module = server_module
    os.environ.update(child_env)
    try:
        await policy.check_actor()
        samples = await generation.generate_turn_sample(call_args, sample, deepcopy(sampling))
        # Upstream catches many exceptions; our failure flag is never inferred
        # from an apparently successful returned Sample or default reward.
        if policy.failure or policy.pending is not None or len(policy.turns) < len(prefix):
            raise CollectionRejected(policy.failure or "incomplete response dispatch/replayed prefix")
        for native_sample in samples:
            reason = str((getattr(native_sample, "metadata", None) or {}).get("terminate_reason", ""))
            if (reason.startswith("generation_error") or reason in
                    ("generation_abort", "env_step_error", "no_generated_turns")):
                raise CollectionRejected("native generator returned infrastructure failure: " + reason)
            # These native Samples are diagnostic artifacts. The custom trainer
            # consumes scored BackendRows, never their sampler log probabilities.
            native_sample.remove_sample = True
            metadata = native_sample.metadata or {}
            mask = metadata.get("arm", {}).get("sampled_mask")
            if mask is not None:
                native_sample.loss_mask = [int(x) for x in mask]
                native_sample.rollout_log_probs = [0.] * len(mask)
            metadata["action_credit_training_admission"] = "pending_frozen_scores_and_sealed_panel"
            native_sample.metadata = metadata
        await policy.check_actor()
        if not samples or not clients or any(not c.closed for c in clients) or "observation" not in terminal:
            raise CollectionRejected("native collection lacks an awaited close and terminal capture")
        if len(samples) != len(policy.turns):
            raise CollectionRejected("native samples do not cover every captured processor row")
        for native_sample, captured in zip(samples, policy.turns):
            native_tensors = getattr(native_sample, "multimodal_train_inputs", None) or {}
            if tensor_payload_sha256(native_tensors) != captured.row.tensor_payload_sha256:
                raise CollectionRejected("native processor tensors differ from captured actor context")
            native_sample.multimodal_train_inputs = captured.tensors
        episode = PendingEpisode(draw_id, request_sha, request, tuple(policy.turns), terminal,
                                 tuple(samples), str(artifacts))
        from openwebrl.action_credit_episode import save_pending_episode
        save_pending_episode(store, contract, actor, episode, verifier_sha256=verifier_sha256)
        return episode
    except BaseException as exc:
        journal("episode_unsettled", dict(error_type=type(exc).__name__, message=str(exc)))
        raise
    finally:
        # Own/await cleanup, including when native timeout code detaches it.
        try:
            for client in clients:
                await client.exit()
        finally:
            generation._create_env, generation._load_local_resources = original_create, original_resources
            generation._apply_local_process_env_overrides = original_override
            local_transport._server_module = original_server_module
            for key, value in saved_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            _ACTIVE_GENERATORS.remove(id(generation))
            journal.blob("collection-teardown.json", json.dumps(dict(
                schema_version=1, request_sha256=request_sha, attempt_id=request.attempt_id,
                manifest_sha256=store.manifest.sha256,
                owned_clients=len(clients), owned_clients_closed=all(c.closed for c in clients),
                captured_turns=len(policy.turns), pending_response=policy.pending is not None,
                policy_failure=policy.failure, verifier_sha256=verifier_sha256),
                sort_keys=True, allow_nan=False).encode()+b"\n")
