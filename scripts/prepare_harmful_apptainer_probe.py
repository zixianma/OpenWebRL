#!/usr/bin/env python3
"""Freeze an offline shopping-admin Apptainer probe proposal; never launch work.

The source metadata is reused read-only from the independent WebArena setup.
This prepares an exact resource request and freezes an import/inspection worker.
It does not authorize, download, import, execute or reset an image itself.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re

UPSTREAM_COMMIT = "6473f72db5dcefc97b5725b59e734504edc28a21"
REPOSITORY = "am1n3e/webarena-verified-shopping_admin"
MANIFEST_SHA256 = "d0531dd27ed98d0c459ff9e88118bf2ed8b660b0ed99c38837db46c065a5be13"
CONFIG_SHA256 = "b97846428776ac553c411507ad7cfd3b5409946eee0cee3bbdba65bb3de72d1d"
COMPRESSED_LAYER_BYTES = 1245375030
LAYER_COUNT = 7
DEFAULT_SOURCE = Path("/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/recipe-sources")
FORBIDDEN = Path("/gpfs/home/zixianma")
APPROVED_ROOTS = tuple(Path(p) for p in (
    "/gpfs/projects/krishna/zixianma/OpenWebRL",
    "/gpfs/scrubbed/zixianma/openwebrl-runtime",
    "/gpfs/scrubbed/zixianma/checkpoints/web",
    "/tmp",
))


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def within(path, root):
    return path == root or root in path.parents


def writable_path(value):
    path = Path(value)
    if not path.is_absolute():
        raise ValueError("Output path must be absolute")
    # Reject lexical traversal into the protected tree, even if a symlink exits it.
    lexical = Path(os.path.abspath(path))
    resolved = path.resolve()
    if within(lexical, FORBIDDEN) or within(resolved, FORBIDDEN):
        raise ValueError("Output crosses protected home boundary")
    if not any(within(resolved, root.resolve()) for root in APPROVED_ROOTS):
        raise ValueError("Output is outside approved writable roots")
    # Existing symlink components are ambiguous mutable identities; refuse them.
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValueError("Output contains a symlink component")
    return resolved


def resource_request():
    return {
        "schema_version": 1,
        "status": "proposed_not_authorized",
        "scope": "Shopping-admin only: pinned image import and inspection, then reviewed Apptainer startup/login and baseline-reset validation if admission gates pass",
        "nodes": 1,
        "cpus": 4,
        "memory_gib": 16,
        "gpus": 0,
        "total_allocation_seconds_including_all_attempts": 7200,
        "api_budget_usd": 0,
        "actor_and_judge_calls": 0,
        "node_local_scratch_cap_bytes": 100 * 1024**3,
        "retained_artifacts_cap_bytes": 30 * 1024**3,
        "all_attempts_transfer_reservation_cap_bytes": 4000000000,
        "transfer_accounting": "Reserve body bytes before network reads, including anonymous registry metadata and incomplete/failed attempts; never refund unknown outcomes or reset counters on retry",
        "older_allocation_seconds_added": 0,
        "older_transfer_budget_added": 0,
        "shared_instance_mutation": False,
        "live_agent_supervision_required": True,
        "full_safety_evaluation_in_scope": False,
        "gitlab_in_scope": False,
        "resource_minimum_measured": False,
        "purpose": "Bounded compatibility diagnosis; successful application startup is not assumed",
        "approval_binding": "An eventual authorization must bind the exact request and frozen proposal hashes; preparation is not authorization",
    }


def read_pins(source):
    manifest_raw = (source / "registry-shopping_admin-amd64.json").read_bytes()
    config_raw = (source / "registry-shopping_admin-config.json").read_bytes()
    if sha256(manifest_raw) != MANIFEST_SHA256:
        raise ValueError("Shopping-admin registry manifest digest mismatch")
    if sha256(config_raw) != CONFIG_SHA256:
        raise ValueError("Shopping-admin image config digest mismatch")
    manifest, config = json.loads(manifest_raw), json.loads(config_raw)
    if (manifest.get("schemaVersion") != 2 or
            manifest.get("mediaType") != "application/vnd.docker.distribution.manifest.v2+json" or
            manifest.get("config", {}).get("digest") != "sha256:" + CONFIG_SHA256 or
            manifest["config"].get("size") != len(config_raw)):
        raise ValueError("Pinned manifest/config relationship invalid")
    if config.get("architecture") != "amd64" or config.get("os") != "linux":
        raise ValueError("Only pinned linux/amd64 image is admitted")
    layers = manifest.get("layers", [])
    diff_ids = config.get("rootfs", {}).get("diff_ids", [])
    if len(layers) != LAYER_COUNT or len(diff_ids) != LAYER_COUNT:
        raise ValueError("Pinned layer count differs")
    for layer, diff_id in zip(layers, diff_ids):
        if (layer.get("mediaType") != "application/vnd.docker.image.rootfs.diff.tar.gzip" or
                type(layer.get("size")) is not int or layer["size"] <= 0 or
                not re.fullmatch(r"sha256:[0-9a-f]{64}", layer.get("digest", "")) or
                not re.fullmatch(r"sha256:[0-9a-f]{64}", diff_id)):
            raise ValueError("Unexpected layer descriptor")
    if sum(layer["size"] for layer in layers) != COMPRESSED_LAYER_BYTES:
        raise ValueError("Pinned compressed image size differs")
    return manifest_raw, config_raw, manifest, config


def stages():
    return [
        {
            "id": "authorization_and_allocation",
            "ready": False,
            "requires": ["Exact new resource approval", "Frozen tested controller and cross-attempt ledger", "Actual scheduler allocation identity and remaining aggregate time verified"],
            "receipt": "allocation.json with job ID, approved caps, prior consumed resources and absolute deadline",
        },
        {
            "id": "bounded_import_and_inspection",
            "implementation_ready": True,
            "ready": False,
            "requires": ["Exact allocation approval and authorization receipt", "Import worker verifies layer size/compressed digest and decompressed diff_id", "Worker streams selected startup files without extracting rootfs or executing any image command"],
            "receipt": "inspection.json with actual final startup files, OS/libc/users, writable database/service paths, service commands and verification receipts",
        },
        {
            "id": "reviewed_startup",
            "ready": False,
            "requires": ["Agent review of actual pinned final filesystem and proposed runtime changes", "Matching native fakeroot library and UID-switch probe if required; do not assume Reddit Alpine addon compatibility", "New state claim, owned process supervisor and isolated network namespace", "Separate namespace-local browser/control access; agent cannot reach database/control/audit endpoints", "All writable paths and caches below approved roots; no protected home bind or write", "Local delivery capture and no external egress before any mutation"],
            "receipt": "startup-review.json binds inspected files, exact launcher source, runtime changes and image identity; runtime.json proves login and service health",
        },
        {
            "id": "baseline_reset_validation",
            "ready": False,
            "requires": ["Immutable baseline preserved", "Application-specific database inventory and consistent snapshot procedure", "Own one explicit reversible fixture mutation; preserve all evidence", "Fresh independently claimed writable instance restores baseline and removes fixture mutation", "Reauthenticated browser; owned descendants closed and final budget reconciliation"],
            "receipt": "reset.json reports exact compared database fields and fixture results, never asserts exhaustive equality from a few sentinels",
        },
    ]


def prepare(source, output):
    source, output = Path(source).resolve(), writable_path(output)
    manifest_raw, config_raw, manifest, config = read_pins(source)
    controller_raw = Path(__file__).resolve().read_bytes()
    runner_raw = Path(__file__).resolve().with_name("run_harmful_apptainer_probe.py").read_bytes()
    request = resource_request()
    review = {
        "schema_version": 1,
        "status": "template_only_not_a_review_receipt",
        "reviewed": False,
        "image": REPOSITORY + "@sha256:" + MANIFEST_SHA256,
        "inspection_receipt_sha256": None,
        "final_startup_file_sha256": None,
        "launcher_source_sha256": None,
        "runtime_changes": None,
        "review_requirement": "Inspect the actual pinned final filesystem, inherited entrypoint and supervisor configs before executing any image command",
    }
    artifacts = {
        "registry-manifest.json": manifest_raw,
        "image-config.json": config_raw,
        "prepare_harmful_apptainer_probe.py": controller_raw,
        "run_harmful_apptainer_probe.py": runner_raw,
        "resource-request.json": encoded(request),
        "startup-review-template.json": encoded(review),
    }
    proposal = {
        "schema_version": 1,
        "status": "import_inspection_worker_frozen_pending_exact_resource_approval",
        "launch_ready": False,
        "import_inspection_implementation_ready": True,
        "application_startup_implementation_ready": False,
        "authorization_status": "pending_exact_resource_approval",
        "experiment_ready": False,
        "site": "shopping_admin",
        "upstream_commit": UPSTREAM_COMMIT,
        "image": REPOSITORY + "@sha256:" + MANIFEST_SHA256,
        "source_metadata_directory": str(source),
        "manifest_sha256": MANIFEST_SHA256,
        "config_sha256": CONFIG_SHA256,
        "compressed_layer_bytes": COMPRESSED_LAYER_BYTES,
        "layer_count": LAYER_COUNT,
        "layer_downloads_verified": False,
        "uncompressed_sizes_known": False,
        "image_metadata_entrypoint": config.get("config", {}).get("Entrypoint"),
        "image_metadata_command": config.get("config", {}).get("Cmd"),
        "metadata_is_not_startup_review": True,
        "frozen_artifacts_sha256": {name: sha256(raw) for name, raw in artifacts.items()},
        "resource_request": request,
        "stages": stages(),
        "reuse": {
            "source": str(DEFAULT_SOURCE.parent),
            "reference_only": ["Namespace isolation", "Fresh-state claims", "Parent-owned process tree", "Hash-bound provenance and explicit reset receipts"],
            "not_reused": ["Reddit allocation or budget", "Reddit mutable database or instance", "Hardcoded job/node/deadline controllers", "Alpine native fakeroot binaries without compatibility validation"],
        },
        "required_cap_implementation": {
            "walltime": "Actual Slurm allocation TRES/start/limit checked; all script attempts bound to the same <=2h job; work stops 120 seconds before job end. Cross-job retries refused pending independent accounting",
            "transfer": "Exclusive cross-attempt durable ledger reserves bounded body reads before requests; limits metadata and blobs; never unmetered apptainer docker:// pull",
            "disk": "Import-only worker writes <=4GB reserved compressed bytes plus <=32MiB inspected source per attempt, at most32 attempts and bounded metadata; no rootfs extraction or node-scratch files. Full runtime import/startup needs separately enforced disk caps",
            "processes": "Import-only worker uses no child service/container processes; scontrol has a20-second timeout. Actual startup needs owned process-tree supervision",
            "api": "No provider clients or credentials loaded; no actor/judge calls",
        },
        "implemented_runtime_cap_enforcement": "import_and_static_inspection_only",
        "earliest_meaningful_trial": "After exact approval: execute frozen import/inspection worker on shopping-admin; stop before startup until actual startup review and runtime adapter are ready",
        "worker_argv_template": ["python3", "-B", "FROZEN_PROPOSAL/run_harmful_apptainer_probe.py", "--proposal-dir", "FROZEN_PROPOSAL", "--authorization", "PRIVATE_AUTHORIZATION_JSON", "--run-dir", "NEW_EXCLUSIVE_RUN_DIRECTORY"],
        "authorization_receipt_required_fields": ["approved=true", "job_id of the exact allocation", "proposal_sha256", "resource_request_sha256"],
        "success_criteria": "Verified image identity and diagnosed runtime compatibility, with login/reset evidence only if the corresponding gates pass; not model evaluation or admitted safety tasks",
        "execution_this_command": {"allocations": 0, "network_requests": 0, "image_download_bytes": 0, "containers_started": 0, "model_calls": 0},
    }
    artifacts["proposal.json"] = encoded(proposal)
    # Check the complete existing set before writing; never overwrite a proposal.
    if output.exists() and not output.is_dir():
        raise ValueError("Output exists and is not a directory")
    if output.exists():
        extra = set(p.name for p in output.iterdir()) - set(artifacts)
        if extra:
            raise ValueError("Output is not an exclusive preparation directory")
    for name, raw in artifacts.items():
        dest = writable_path(output / name)
        if dest.exists() and (not dest.is_file() or dest.read_bytes() != raw):
            raise ValueError("Existing preparation differs; choose a fresh output directory: " + name)
    output.mkdir(parents=True, exist_ok=True)
    for name, raw in artifacts.items():
        path = output / name
        if path.exists():
            continue
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    return {"status": proposal["status"], "launch_ready": False,
            "proposal_path": str(output / "proposal.json"),
            "proposal_sha256": sha256(artifacts["proposal.json"]),
            "resource_request_sha256": sha256(artifacts["resource-request.json"]),
            "frozen_artifact_count": len(artifacts)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-metadata", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = prepare(args.source_metadata, args.output)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
