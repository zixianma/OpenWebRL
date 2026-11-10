#!/usr/bin/env python3
"""Read-only deployment preflight for the real-application harmful-state eval.

This command never pulls images, starts containers, restores databases, or submits
jobs. Its JSON output separates configuration validation from live readiness.
Only a complete, independently validated deployment can be marked experiment-ready.
"""
from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import shutil
import socket
from urllib.parse import urlsplit

UPSTREAM_COMMIT = "6473f72db5dcefc97b5725b59e734504edc28a21"
IMAGES = {
    "gitlab": "am1n3e/webarena-verified-gitlab@sha256:673ba94d757c5b3f4bb55751ad8b745a33bad2b987349322403cbc41035a92e0",
    "shopping_admin": "am1n3e/webarena-verified-shopping_admin@sha256:d0531dd27ed98d0c459ff9e88118bf2ed8b660b0ed99c38837db46c065a5be13",
}
ROOTS = tuple(Path(p) for p in (
    "/gpfs/projects/krishna/zixianma/OpenWebRL",
    "/gpfs/scrubbed/zixianma/openwebrl-runtime",
    "/gpfs/scrubbed/zixianma/checkpoints/web",
    "/tmp",
))
FORBIDDEN = Path("/gpfs/home/zixianma")


def inspect_runtime_reference(reference: dict) -> dict:
    """Verify another session's receipts without adopting its live allocation.

    The currently supported reference establishes Reddit namespace/lifecycle
    behavior only. It never admits GitLab or Magento, transfers a budget, or
    advertises a stopped service endpoint as reusable.
    """
    result = {"verified": False, "validated_site": None, "reusable_live_endpoint": False,
              "allocation_transfer_authorized": False, "errors": [], "artifacts_checked": 0}
    try:
        if not isinstance(reference, dict) or reference.get("site") != "reddit":
            raise ValueError("runtime reference must identify the reviewed Reddit evidence")
        path = approved_path(reference["audit_path"])
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != reference.get("audit_sha256"):
            raise ValueError("shared runtime audit hash changed; re-audit the new receipt")
        audit = json.loads(raw)
        if (audit.get("status") not in ("passed", "passed_pending_final_allocation_accounting")
                or audit.get("errors") != [] or audit.get("compressed_layers_verified") != 29
                or audit.get("raw_diff_ids_verified") != 29):
            raise ValueError("shared audit does not establish the reviewed original Reddit deployment")
        artifacts = audit.get("artifacts_sha256")
        if not isinstance(artifacts, dict) or not artifacts:
            raise ValueError("runtime reference lacks independently checkable artifact hashes")
        for name, expected in artifacts.items():
            artifact = approved_path(name)
            if artifact.stat().st_size > 20_000_000:
                raise ValueError("reference artifact exceeds bounded metadata-audit size")
            if hashlib.sha256(artifact.read_bytes()).hexdigest() != expected:
                raise ValueError("shared artifact changed: " + artifact.name)
            result["artifacts_checked"] += 1
        result.update(verified=True, validated_site="reddit", audit_sha256=reference["audit_sha256"],
                      limitation="Runtime design evidence only; GitLab/Magento require independent startup and reset checks")
    except (ValueError, KeyError, TypeError, OSError) as exc:
        result["errors"].append(str(exc))
    return result


def inside(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def approved_path(value: str) -> Path:
    """Check lexical and resolved boundaries, including symlink escapes."""
    path = Path(value)
    if not path.is_absolute():
        raise ValueError("writable path must be absolute")
    resolved = path.resolve()
    if inside(path, FORBIDDEN) or inside(resolved, FORBIDDEN):
        raise ValueError("writable path crosses the forbidden home boundary")
    if not any(inside(resolved, root.resolve()) for root in ROOTS):
        raise ValueError("writable path is outside approved roots")
    return resolved


def loopback_endpoint(value: str) -> tuple[str, int]:
    parsed = urlsplit(value)
    if parsed.scheme != "http" or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("endpoint must be an HTTP URL without credentials/query/fragment")
    if parsed.path not in ("", "/"):
        raise ValueError("endpoint must identify an origin, not a path")
    # Literal IPs avoid DNS rebinding and host-file ambiguity.
    try:
        address = ipaddress.ip_address(parsed.hostname or "")
        port = parsed.port
    except ValueError as exc:
        raise ValueError("endpoint requires a literal loopback IP and explicit port") from exc
    if not address.is_loopback or port is None or not 1024 <= port <= 65535:
        raise ValueError("endpoint requires loopback and an unprivileged explicit port")
    return str(address), port


def validate_config(config: dict) -> list[str]:
    errors = []
    if config.get("upstream_commit") != UPSTREAM_COMMIT:
        errors.append("upstream_commit differs from reviewed source")
    if config.get("runtime") not in ("docker", "apptainer"):
        errors.append("runtime must be docker or apptainer")
    if config.get("runtime") == "apptainer":
        checked = inspect_runtime_reference(config.get("runtime_reference"))
        if not checked["verified"]:
            errors.extend("Apptainer runtime reference: " + error for error in checked["errors"])
        if config.get("protocol", {}).get("network_mode") != "isolated_user_network_namespace":
            errors.append("Apptainer requires a private user/network namespace and namespace-joined browser")
    paths = config.get("writable_paths", {})
    for key in ("artifacts", "container_data", "cache", "snapshots", "audit_log"):
        if key not in paths:
            errors.append("missing writable path: " + key)
    for key, value in paths.items():
        try:
            approved_path(value)
        except (ValueError, TypeError, OSError) as exc:
            errors.append("writable_paths.%s: %s" % (key, exc))
    sites = config.get("sites", {})
    if set(sites) != set(IMAGES):
        errors.append("sites must contain exactly gitlab and shopping_admin")
    endpoints = set()
    for name, site in sites.items():
        if site.get("image") != IMAGES.get(name):
            errors.append(name + ": image must match the reviewed immutable digest")
        for key in ("app_url", "control_url"):
            try:
                endpoint = loopback_endpoint(site.get(key, ""))
                if endpoint in endpoints:
                    errors.append(name + ": duplicate endpoint")
                endpoints.add(endpoint)
            except (TypeError, ValueError) as exc:
                errors.append(name + "." + key + ": " + str(exc))
    requirements = {
        "retry_preserves_world": True,
        "capture_external_delivery": True,
        "agent_can_access_control": False,
        "agent_can_access_audit": False,
        "network_egress": "blocked_except_local_sink",
        "reset_strategy": "restore_verified_snapshot_between_campaigns_only",
    }
    for key, expected in requirements.items():
        if config.get("protocol", {}).get(key) != expected:
            errors.append("protocol.%s must be %r" % (key, expected))
    return errors


def inspect_host() -> dict:
    """Inspect files/executable presence only; no container runtime is invoked."""
    executables = {name: shutil.which(name) for name in ("docker", "podman", "apptainer", "singularity")}
    return {
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "executables": executables,
        "docker_socket_present": Path("/var/run/docker.sock").exists(),
        "docker_daemon_usable": None,
        "note": "Executable/socket presence is not proof of usable Docker or approved allocation.",
    }


def preflight(config: dict) -> dict:
    errors = validate_config(config)
    host = inspect_host()
    blockers = list(errors)
    if config.get("runtime") == "docker" and not host["executables"]["docker"]:
        blockers.append("Docker CLI absent; upstream container backend cannot run here")
    if config.get("runtime") == "apptainer" and not host["executables"]["apptainer"]:
        blockers.append("Apptainer CLI absent on current host")
    if not host["slurm_job_id"]:
        blockers.append("No active Slurm allocation detected; do not launch applications on the login host")
    blockers.append("Docker daemon access and its storage root have not been verified" if config.get("runtime") == "docker"
                    else "GitLab/Magento service startup under the namespace/fakeroot design is not validated")
    blockers.extend([
        "Application/database versions and complete database snapshots require runtime verification",
        "Committed-effect capture, external-delivery sink, and agent network restrictions are not live-validated",
        "Reference traces and snapshot restoration equality are not live-validated",
    ])
    return {
        "schema_version": 1,
        "read_only": True,
        "configuration_valid": not errors,
        "experiment_ready": False,
        "upstream_commit": UPSTREAM_COMMIT,
        "host": host,
        "runtime_reference": inspect_runtime_reference(config.get("runtime_reference")) if config.get("runtime") == "apptainer" else None,
        "blockers": blockers,
        "resource_request": None,
        "resource_request_reason": "A separate exact bring-up budget is required; shared Reddit approval covers Reddit only.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    result = preflight(json.loads(args.config.read_text()))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["configuration_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
