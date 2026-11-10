"""Trusted private episode persistence; reload never dispatches a request.

Torch payloads contain native Sample objects and may only be loaded from this
campaign's checksummed pointer after file/source verification. This is an
integrity boundary, not authentication of arbitrary externally supplied pickle
files or independent proof that browser/actor verification was performed.
"""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

from openwebrl.action_credit import ResponseGraph, digest
from openwebrl.action_credit_binding import _check_contract
from openwebrl.action_credit_collect import PendingEpisode, FrozenActor, physical_verifier_sha256
from openwebrl.action_credit_megatron import structural_graph
from openwebrl.action_credit_receipts import validate_receipts
from openwebrl.action_credit_state import _safe_path
from openwebrl.action_credit_tensor_refs import (tensor_inventory, bind_tensor_inventory,
                                                episode_tensor_mappings)


def _hash_file(path):
    path = _safe_path(path)
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            hasher.update(block)
    return hasher.hexdigest()


def _validate(store, contract, actor, episode, verifier_sha256):
    if type(episode) is not PendingEpisode or type(actor) is not FrozenActor:
        raise ValueError("typed saved native episode and frozen actor required")
    contract.validate()
    actor.validate(contract)
    _check_contract(store.manifest.to_dict(), contract)
    if verifier_sha256 != physical_verifier_sha256():
        raise ValueError("saved episode verifier source differs from current pinned code")
    request = episode.request
    snapshot = store.snapshot()
    envelope = snapshot.get("online_requests", {}).get(request.request_id)
    request_identity = asdict(request)
    request_identity.pop("attempt_id")
    if (not envelope or envelope["request"] != request_identity
            or envelope["actor"] != asdict(actor) or envelope["contract_sha256"] != contract.sha256
            or digest(envelope) != episode.request_sha256):
        raise ValueError("saved episode differs from pre-dispatch request/actor/runtime pins")
    attempt = snapshot["attempts"].get(request.attempt_id)
    identity = store._draw_identity(contract.update_id, contract.policy_sha256, request.spec())
    draw_id = digest(identity)
    if (episode.draw_id != draw_id or snapshot["requests"].get(request.request_id) != draw_id
            or not attempt or attempt["draw_id"] != draw_id):
        raise ValueError("saved episode draw/attempt differs from charged ledger")
    directory = _safe_path(store.root / "online_artifacts" / digest(request.attempt_id))
    if _safe_path(episode.artifact_directory) != directory:
        raise ValueError("saved episode artifacts escaped the private attempt directory")
    if not episode.turns or len(episode.turns) != len(episode.native_samples):
        raise ValueError("saved native samples must cover every captured response exactly")
    graph = structural_graph(ResponseGraph(contract, tuple(turn.edge for turn in episode.turns),
        (episode.leaf(status="invalid", reward=None),), (request.leaf_id,)))
    validate_receipts(graph, {turn.edge.edge_id: turn.replay_receipt for turn in episode.turns})
    dispositions = []
    for index, (turn, native) in enumerate(zip(episode.turns, episode.native_samples)):
        edge = turn.edge
        turn.row.validate(edge, contract, turn.tensors)
        turn.row.validate(edge, contract, getattr(native, "multimodal_train_inputs", None) or {})
        before = json.loads(turn.before_evidence_json)
        if (turn.replay_receipt.verifier_sha256 != verifier_sha256
                or turn.replay_receipt.evidence_sha256 != turn.evidence_sha256
                or digest(before["snapshot"]) != edge.state_sha256
                or hashlib.sha256(turn.before_image).hexdigest() != before["screenshot_sha256"]
                or not edge.media_sha256 or edge.media_sha256[-1] != before["screenshot_sha256"]
                or tuple(turn.output[1]) != edge.response_ids):
            raise ValueError("saved response/source media/replay evidence mismatch")
        metadata = native.metadata
        arm = metadata.get("arm", {})
        disposition = dict(status=str(native.status), terminate_reason=metadata.get("terminate_reason"),
            turn_index=metadata.get("turn_index"), is_last_turn=metadata.get("is_last_turn"))
        if (tuple(native.tokens) != edge.input_ids + edge.response_ids
                or native.response_length != len(edge.response_ids) or native.response != turn.output[0]
                or tuple(bool(x) for x in native.loss_mask) != edge.sampled_mask
                or len(native.rollout_log_probs) != len(edge.response_ids)
                or any(value != 0. for value in native.rollout_log_probs)
                or arm.get("edge_id") != edge.edge_id or tuple(arm.get("sampled_mask", ())) != edge.sampled_mask
                or metadata.get("action_credit_training_admission") != "pending_frozen_scores_and_sealed_panel"
                or native.remove_sample is not True or disposition["turn_index"] != index
                or disposition["is_last_turn"] is not (index == len(episode.turns)-1)
                or not disposition["terminate_reason"]
                or str(native.status).rsplit(".", 1)[-1].upper() not in ("COMPLETED", "FAILED", "TRUNCATED")):
            raise ValueError("saved native tokens/masks or complete terminal disposition mismatch")
        dispositions.append(disposition)
    terminal = episode.terminal_evidence.get("observation", {})
    if not terminal or _hash_file(directory / "terminal.png") != terminal.get("screenshot_sha256"):
        raise ValueError("saved terminal observation/screenshot is missing or changed")
    return dict(request_sha256=episode.request_sha256, draw_id=draw_id, attempt_id=request.attempt_id,
        contract_sha256=contract.sha256, policy_sha256=contract.policy_sha256,
        actor_sha256=digest(asdict(actor)), verifier_sha256=verifier_sha256,
        turns_sha256=digest([turn.sha256 for turn in episode.turns]),
        terminal_evidence_sha256=digest(episode.terminal_evidence),
        native_dispositions_sha256=digest(dispositions))


def save_pending_episode(store, contract, actor, episode, *, verifier_sha256):
    """Commit one complete private episode and pointer; do not settle its judge."""
    import torch
    owners = tensor_inventory(store, episode_tensor_mappings(episode))
    identity = _validate(store, contract, actor, episode, verifier_sha256)
    snapshot = store.snapshot()
    existing = snapshot.get("pending_episodes", {}).get(episode.request.attempt_id)
    if existing is not None:
        if existing["identity"] != identity:
            raise ValueError("durable pending episode cannot be rewritten")
        load_pending_episode(store, contract, actor, episode.request.attempt_id, verifier_sha256=verifier_sha256)
        return existing
    directory = _safe_path(episode.artifact_directory)
    target = _safe_path(directory / "pending-episode.pt")
    # A file without its final ledger pointer may be a partial crash write.
    # Never overwrite/recollect or auto-adopt it based on existence alone.
    if target.exists():
        raise ValueError("uncommitted episode file requires explicit crash reconciliation")
    artifacts = [dict(path=str(path), sha256=_hash_file(path))
                 for path in sorted(directory.rglob("*")) if path.is_file()]
    payload = dict(schema_version=2, manifest_sha256=store.manifest.sha256,
                   identity=identity, episode=episode, retained_artifacts=artifacts,
                   tensor_owners=owners)
    with target.open("xb") as handle:
        torch.save(payload, handle)
        handle.flush(); os.fsync(handle.fileno())
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    pointer = dict(path=str(target), sha256=_hash_file(target), identity=identity,
                   manifest_sha256=store.manifest.sha256, collection_complete=True, judge_settled=False)
    with store._transaction() as state:
        saved = state.setdefault("pending_episodes", {})
        if episode.request.attempt_id in saved:
            raise ValueError("concurrent pending episode writer requires reconciliation")
        saved[episode.request.attempt_id] = pointer
    return pointer


def load_pending_episode(store, contract, actor, attempt_id, *, verifier_sha256):
    """Reload the exact committed private episode; never resend ambiguous work."""
    import torch
    pointer = store.snapshot().get("pending_episodes", {}).get(attempt_id)
    if not pointer:
        raise ValueError("no committed complete episode; reconcile the existing attempt without redispatch")
    expected = _safe_path(store.root / "online_artifacts" / digest(attempt_id) / "pending-episode.pt")
    if (pointer.get("manifest_sha256") != store.manifest.sha256
            or _safe_path(pointer["path"]) != expected or _hash_file(expected) != pointer["sha256"]):
        raise ValueError("pending episode pointer/file identity mismatch")
    # Private, locally produced pickle only, checked against the atomic ledger
    # before deserialization. No caller-supplied external path is accepted.
    payload = torch.load(expected, weights_only=False, map_location="cpu", mmap=True)
    if (payload.get("schema_version") != 2 or payload.get("manifest_sha256") != store.manifest.sha256
            or payload.get("identity") != pointer["identity"]):
        raise ValueError("pending episode payload/ledger mismatch")
    for artifact in payload["retained_artifacts"]:
        path = _safe_path(artifact["path"])
        if expected.parent not in path.parents or _hash_file(path) != artifact["sha256"]:
            raise ValueError("saved episode rollout/evidence artifact missing or changed")
    episode = payload["episode"]
    bind_tensor_inventory(store, episode_tensor_mappings(episode), payload["tensor_owners"])
    if episode.request.attempt_id != attempt_id or _validate(store, contract, actor, episode, verifier_sha256) != pointer["identity"]:
        raise ValueError("reconstructed episode differs from frozen durable identity")
    return episode
