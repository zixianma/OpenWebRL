"""Frozen chain/GiGPO update planning and durable pre-optimizer batches.

No scheduler, browser or optimizer is started here. A controller dispatches the
planned independent episodes through action_credit_collect, judges them with
the metered canonical adapter, and supplies native frozen-score receipts. The
entire planned inventory must be terminal before this module assembles/seals
it. Optimizer windows and checkpoint commits are separately verified stages.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

from openwebrl.action_credit import Leaf, ResponseGraph, digest, _sha
from openwebrl.action_credit_binding import SealedCollection, _check_contract
from openwebrl.action_credit_megatron import finalize_frozen_scores
from openwebrl.action_credit_objective import build_objective
from openwebrl.action_credit_state import _safe_path
from openwebrl.action_credit_windows import WindowConfig, plan_ppo_windows


def _artifact_file(reference):
    if (not isinstance(reference, dict) or not {"path", "sha256"} <= set(reference)
            or not isinstance(reference["path"], str) or not Path(reference["path"]).is_absolute()):
        raise ValueError("explicit absolute artifact path and SHA required")
    _sha(reference["sha256"], "artifact_sha256")
    path = _safe_path(reference["path"])
    if path != path.resolve(strict=False):
        raise ValueError("native artifact paths must be canonical, without traversal aliases")
    allowed = (Path("/gpfs/projects/krishna/zixianma/OpenWebRL"),
               Path("/gpfs/scrubbed/zixianma"), Path("/tmp"))
    if not any(path == root or root in path.parents for root in allowed):
        raise ValueError("native artifact must remain in approved project/scrubbed/tmp storage")
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            hasher.update(block)
    if hasher.hexdigest() != reference["sha256"]:
        raise ValueError("native artifact SHA mismatch")
    return path


def validate_optimizer_rng_artifact(store, contract, reference):
    """Check saved native checkpoint/RNG identities, not physical GPU restore.

    The controller produces this sidecar from actual per-rank native states
    immediately before the update. Synthetic CPU test files cannot satisfy the
    separate production restore/save/resume readiness gate.
    """
    sidecar_path = _artifact_file(reference)
    saved = json.loads(sidecar_path.read_text())
    manifest = store.manifest.to_dict()
    algorithm, checkpoint = manifest["science"]["algorithm"], manifest["science"]["checkpoint"]
    if (saved.get("schema_version") != 1 or saved.get("manifest_sha256") != store.manifest.sha256
            or saved.get("policy_sha256") != contract.policy_sha256
            or saved.get("graph_contract_sha256") != contract.sha256
            or saved.get("update_id") != contract.update_id
            or type(saved.get("world_size")) is not int or saved["world_size"] != 8
            or algorithm.get("training_world_size") != saved["world_size"]):
        raise ValueError("RNG artifact manifest, policy, update or explicit eight-rank world mismatch")
    validator = checkpoint.get("native_checkpoint_validator_sha256")
    _sha(validator, "native_checkpoint_validator_sha256")
    if saved.get("validator_source_sha256") != validator:
        raise ValueError("RNG artifact validator source differs from frozen runtime code")
    ranks = saved.get("rank_rng_files")
    if (not isinstance(ranks, list) or len(ranks) != saved["world_size"]
            or any(not isinstance(x, dict) or type(x.get("rank")) is not int for x in ranks)
            or {x["rank"] for x in ranks} != set(range(saved["world_size"]))
            or len({x.get("path") for x in ranks}) != len(ranks)):
        raise ValueError("RNG artifact must preserve each unique rank exactly once")
    for rank in ranks:
        _artifact_file(rank)
    previous = saved.get("previous_checkpoint_inventory")
    inventory_path = _artifact_file(previous)
    inventory = json.loads(inventory_path.read_text())
    if (inventory.get("schema_version") != 1 or inventory.get("policy_sha256") != contract.policy_sha256
            or inventory.get("completed_updates") != contract.update_id):
        raise ValueError("Previous committed native checkpoint identity/cursor mismatch")
    files = inventory.get("files")
    required = {"model", "optimizer", "scheduler", "cursor", "rng"}
    if (not isinstance(files, list) or not files
            or any(not isinstance(x, dict) or not isinstance(x.get("components"), list)
                   or not x["components"] or any(not isinstance(k, str) for k in x["components"])
                   for x in files)
            or not required <= {kind for x in files for kind in x["components"]}
            or len({x.get("path") for x in files}) != len(files)):
        raise ValueError("Previous checkpoint must preserve full model/optimizer/scheduler/cursor/RNG inventory")
    file_paths = [_artifact_file(entry) for entry in files]
    if contract.update_id == 0:
        initial_root = checkpoint.get("native_initial_state_root")
        if not isinstance(initial_root, str) or not Path(initial_root).is_absolute():
            raise ValueError("Pin a distinct native initial-state root")
        initial_root = _safe_path(initial_root)
        if (inventory.get("checkpoint_kind") != "native_initialization"
                or not all(inventory.get(k) is True for k in ("fresh_optimizer", "fresh_scheduler", "fresh_cursor"))
                or initial_root not in inventory_path.parents
                or not all(initial_root in path.parents for path in file_paths)):
            raise ValueError("First update requires its own fresh full native initialization checkpoint")
    else:
        prior = store.snapshot()["updates"].get(str(contract.update_id-1), {})
        receipt = prior.get("training_receipt") or {}
        components = inventory.get("component_state_sha256") or {}
        if (inventory.get("checkpoint_kind") != "native_completed_update"
                or prior.get("phase") != "consumed"
                or receipt.get("new_policy_sha256") != contract.policy_sha256
                or receipt.get("new_checkpoint_sha256") != previous["sha256"]
                or any(components.get(name) != receipt.get(name+"_state_sha256")
                       for name in ("optimizer", "scheduler", "cursor"))):
            raise ValueError("Continuation checkpoint must match the exact consumed prior training receipt")
    return dict(path=str(sidecar_path), sha256=reference["sha256"],
                previous_checkpoint_inventory=dict(previous), rank_rng_inventory_sha256=digest(ranks),
                validator_source_sha256=validator, physical_restore_verified_by_this_function=False)


def plan_independent_update(store, contract, task_order_path):
    """Materialize all prespecified groups/replicates, with no reward filtering."""
    manifest = store.manifest.to_dict()
    if manifest["arm_id"] not in ("chain_outcome_grpo", "gigpo"):
        raise ValueError("independent-chain planner is only for chain/GiGPO")
    _check_contract(manifest, contract)
    raw = Path(task_order_path).read_bytes()
    order_sha = hashlib.sha256(raw).hexdigest()
    dataset, algorithm = manifest["science"]["dataset"], manifest["science"]["algorithm"]
    if dataset.get("task_order_sha256") != order_sha:
        raise ValueError("task-order file differs from frozen campaign")
    order = json.loads(raw)
    groups_count, chains = algorithm.get("groups_per_iteration"), algorithm.get("chains_per_group")
    if (type(groups_count) is not int or groups_count <= 0 or type(chains) is not int or chains <= 1
            or order["source_sha256"] != dataset["source_sha256"]
            or order["groups_per_iteration"] != groups_count
            or order.get("outcome_dependent_replacement") is not False):
        raise ValueError("explicit matched fixed-task/group protocol required")
    entry = order["iterations"][contract.update_id]
    groups = entry["groups"]
    if entry["iteration"] != contract.update_id+1 or len(groups) != groups_count:
        raise ValueError("task cursor or complete group count mismatch")
    if len({g["group_id"] for g in groups}) != len(groups):
        raise ValueError("duplicate prespecified task group")
    draws = []
    for group in groups:
        if (group["task_id"] not in contract.train_task_ids
                or order["original_manifest_task_order"][group["source_index"]] != group["task_id"]):
            raise ValueError("task membership or dataset source cursor changed")
        for slot in range(chains):
            tree = f'{group["group_id"]}:chain{slot:02d}'
            spec = dict(task_id=group["task_id"], state_id=group["group_id"]+":root",
                action_id="independent_chain", draw_slot=slot, draw_role="train",
                group_id=group["group_id"], tree_id=tree, inventory_role="base")
            draws.append(dict(leaf_id=tree+":terminal", spec=spec))
    result = store.plan_update(contract.update_id, policy_sha256=contract.policy_sha256,
                               draws=[x["spec"] for x in draws])
    plan = dict(update_id=contract.update_id, policy_sha256=contract.policy_sha256,
        task_order_sha256=order_sha, graph_contract_sha256=contract.sha256,
        plan_sha256=result["plan_sha256"], draws=draws)
    with store._transaction() as state:
        plans = state.setdefault("independent_collection_plans", {})
        previous = plans.get(str(contract.update_id))
        if previous is not None and previous != plan:
            raise ValueError("independent collection plan changed")
        plans[str(contract.update_id)] = plan
    return plan


def draft_independent_update(store, contract, plan, episodes):
    """Assemble exact complete terminal inventory; missing slots stay explicit.

    Episode records are supplied from retained private artifacts. This identity
    bridge does not authenticate caller-created Python objects; the controller
    must verify artifact hashes and reconstruct them from the saved records.
    """
    state = store.snapshot()
    if state.get("independent_collection_plans", {}).get(str(contract.update_id)) != plan:
        raise ValueError("use the persisted immutable independent collection plan")
    expected = state["updates"][str(contract.update_id)]["expected_draws"]
    episodes = tuple(episodes)
    by_leaf = {x.request.leaf_id: x for x in episodes}
    if len(by_leaf) != len(episodes) or set(by_leaf) - {x["leaf_id"] for x in plan["draws"]}:
        raise ValueError("duplicate or unplanned collected episode")
    edges, rows, tensors, receipts, leaves, leaf_to_draw = {}, {}, {}, {}, [], {}
    for item in plan["draws"]:
        leaf_id, spec = item["leaf_id"], item["spec"]
        identity = store._draw_identity(contract.update_id, contract.policy_sha256, spec)
        draw_id = digest(identity)
        draw = state["draws"].get(draw_id)
        if expected.get(draw_id) != identity or not draw or draw["status"] not in ("valid", "invalid", "missing"):
            raise ValueError("complete planned terminal inventory is required; no partial sibling training")
        episode, reward = by_leaf.get(leaf_id), None
        selected = draw["selected_attempt"]
        if selected is not None:
            outcome = state["attempts"][selected]["completion"]["outcome"]
            reward = outcome.get("reward") if outcome else None
        if draw["status"] == "valid" and episode is None:
            raise ValueError("valid draw lacks exact saved native episode")
        if episode is not None:
            if (episode.draw_id != draw_id or episode.request.spec() != spec
                    or episode.request.attempt_id != selected):
                raise ValueError("episode differs from selected immutable draw/attempt")
            leaf = episode.leaf(status=draw["status"], reward=reward)
            for captured in episode.turns:
                edge = captured.edge
                if edge.edge_id in edges:
                    raise ValueError("independent chains cannot reuse a sampled response")
                captured.row.validate(edge, contract, captured.tensors)
                edges[edge.edge_id], rows[edge.edge_id] = edge, captured.row
                tensors[edge.edge_id], receipts[edge.edge_id] = captured.tensors, captured.replay_receipt
        else:
            leaf = Leaf(leaf_id, spec["task_id"], spec["tree_id"], (), draw["status"], None,
                        spec["draw_role"], spec["group_id"])
        leaves.append(leaf)
        leaf_to_draw[leaf_id] = draw_id
    graph = ResponseGraph(contract, tuple(edges.values()), tuple(leaves), tuple(x["leaf_id"] for x in plan["draws"]))
    from openwebrl.action_credit_megatron import structural_graph
    structural_graph(graph).validate(independent_chains=True)
    return dict(graph=graph, rows=rows, tensor_payloads=tensors, replay_receipts=receipts, leaf_to_draw=leaf_to_draw)


def save_scored_pending_update(store, config, draft, frozen_scores, *, scorer_source_sha256, window_config,
                              optimizer_rng_artifact, panels=(), selection=None):
    """Persist the full scored/sealed batch before any optimizer intent/step.

    Tensor payloads use torch's trusted-private-file format. Their exact bytes
    are separately bound by BackendRow. No checkpoint is marked consumed here.
    """
    import torch
    from .action_credit_tensor_refs import tensor_inventory
    owners = tensor_inventory(store, draft['tensor_payloads'].values())
    if (type(window_config) is not WindowConfig or
            store.manifest.to_dict()["science"]["algorithm"].get("ppo_window_config") != asdict(window_config)):
        raise ValueError("full PPO window policy must equal the frozen campaign configuration")
    window_config.validate()
    rng_reference = validate_optimizer_rng_artifact(store, draft["graph"].contract, optimizer_rng_artifact)
    graph = finalize_frozen_scores(draft["graph"], frozen_scores, rows=draft["rows"],
                                   scorer_source_sha256=scorer_source_sha256)
    for edge in graph.edges:
        draft["rows"][edge.edge_id].validate(edge, graph.contract, draft["tensor_payloads"][edge.edge_id])
    store.seal_update(graph.contract.update_id)
    binding = SealedCollection.from_store(store, graph.contract, draft["leaf_to_draw"])
    panels = tuple(panels)
    prepared = build_objective(graph, config, collection_binding=binding, replay_receipts=draft["replay_receipts"],
                               panels=panels, selection=selection)
    schedule = plan_ppo_windows(prepared, window_config)
    payload = dict(graph=graph, config=config, binding=binding, frozen_scores=frozen_scores,
        rows=draft["rows"], tensor_payloads=draft["tensor_payloads"], replay_receipts=draft["replay_receipts"],
        leaf_to_draw=draft["leaf_to_draw"], ppo_schedule=schedule,
        panels=panels, selection=selection,
        optimizer_rng_artifact=rng_reference, tensor_owners=owners,
        checkpoint_policy="restore_previous_full_update_and_replay_pending_schedule_from_zero")
    input_sha = digest(dict(graph=asdict(graph), config=asdict(config), binding_sha256=binding.sha256,
                           rows={k: asdict(v) for k, v in draft["rows"].items()},
                           score_receipts={k: v.sha256 for k, v in frozen_scores.items()},
                           panels=[asdict(panel) for panel in panels],
                           selection_sha256=selection.sha256 if selection is not None else None,
                           ppo_schedule_sha256=schedule.sha256, optimizer_rng_artifact=rng_reference))
    directory = _safe_path(store.root / "pending_updates" / str(graph.contract.update_id))
    directory.mkdir(parents=True, exist_ok=True)
    target = _safe_path(directory / (input_sha+".pt"))
    prior = store.snapshot().get("pending_batches", {}).get(str(graph.contract.update_id))
    if target.exists():
        if (prior is None or prior["input_sha256"] != input_sha or prior["path"] != str(target)):
            raise ValueError("uncommitted or different pending batch file requires explicit reconciliation")
        _artifact_file(dict(path=str(target), sha256=prior["file_sha256"]))
        return prior, prepared
    else:
        # Exclusive write is never an overwrite. A partial file left by a crash
        # fails its digest and must be reconciled; it cannot be used for PPO.
        with target.open("xb") as handle:
            torch.save(payload, handle)
            handle.flush(); os.fsync(handle.fileno())
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    hasher = hashlib.sha256()
    with target.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            hasher.update(block)
    file_sha = hasher.hexdigest()
    pointer = dict(update_id=graph.contract.update_id, input_sha256=input_sha,
        file_sha256=file_sha, path=str(target), manifest_sha256=store.manifest.sha256,
        binding_sha256=binding.sha256, policy_sha256=graph.contract.policy_sha256,
        ppo_schedule_sha256=schedule.sha256, window_count=len(schedule.windows),
        optimizer_rng_artifact=rng_reference,
        checkpoint_policy=payload["checkpoint_policy"],
        optimizer_started=False)
    with store._transaction() as state:
        pointers = state.setdefault("pending_batches", {})
        old = pointers.get(str(graph.contract.update_id))
        if old is not None and old != pointer:
            raise ValueError("saved pending batch pointer changed")
        pointers[str(graph.contract.update_id)] = pointer
    return pointer, prepared
