#!/usr/bin/env python3
"""Bounded shopping-admin image acquisition and static inspection only.

No image command, container runtime, browser, model, or database is executed.
All attempts must stay in the same explicitly approved CPU-only allocation.
"""
from __future__ import annotations

import argparse
import datetime
import fcntl
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import socket
import subprocess
import tarfile
import time
import urllib.request

import prepare_harmful_apptainer_probe as prep

READ_CHUNK = 1024 * 1024
MAX_CAPTURE_BYTES = 32 * 1024**2
MAX_FILE_BYTES = 1024**2
MAX_CAPTURE_FILES = 1000
MAX_TAR_MEMBERS = 1000000
MAX_ATTEMPTS = 32
MAX_PATH_BYTES = 4096
MAX_JSON_BYTES = 16 * 1024**2
DISK_WRITE_BOUND = 4000000000 + MAX_ATTEMPTS * (MAX_CAPTURE_BYTES + 64 * 1024**2) + 64 * 1024**2


class Closed(ValueError):
    pass


def write_new(path, raw):
    path = prep.writable_path(path)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def atomic_json(path, value):
    path = prep.writable_path(path)
    raw = prep.encoded(value)
    if len(raw) > MAX_JSON_BYTES:
        raise Closed("Serialized metadata exceeds 16 MiB bound")
    pending = path.with_name(path.name + ".pending-" + str(os.getpid()) + "-" + str(time.time_ns()))
    write_new(pending, raw)
    os.replace(pending, path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def regular_hash(path):
    path = prep.writable_path(path)
    if not path.is_file():
        raise Closed("Expected a regular artifact")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(READ_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def remaining(deadline):
    value = deadline - time.time()
    if value <= 0:
        raise Closed("Approved import deadline reached")
    return min(30.0, value)


def allocation_from_scontrol(text, job_id, now=None):
    """Require actual allocated TRES, one running CPU-only node and <=2h limit."""
    fields = dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=(\S+)", text))
    if fields.get("JobId") != str(job_id) or fields.get("JobState") != "RUNNING":
        raise Closed("Not the named running Slurm allocation")
    if fields.get("NumNodes") != "1" or fields.get("NumCPUs") != "4":
        raise Closed("Allocation must contain exactly one node and four CPUs")
    tres = dict(item.split("=", 1) for item in fields.get("AllocTRES", "").split(",") if "=" in item)
    if tres.get("cpu") != "4" or tres.get("node") != "1" or tres.get("mem") not in {"16G", "16384M"}:
        raise Closed("Allocated CPU/memory TRES does not match approved 4 CPU/16 GiB request")
    if any(("gpu" in key and value != "0") for key, value in tres.items()):
        raise Closed("GPU allocation is outside this request")
    limit = fields.get("TimeLimit", "")
    match = re.fullmatch(r"(?:(\d+)-)?(\d+):(\d+):(\d+)", limit)
    if not match:
        raise Closed("Unrecognized scheduler time limit")
    days, hours, minutes, seconds = (int(value or 0) for value in match.groups())
    duration = days * 86400 + hours * 3600 + minutes * 60 + seconds
    if not 0 < duration <= 7200:
        raise Closed("Allocation exceeds total approved two-hour envelope")
    try:
        start = datetime.datetime.fromisoformat(fields["StartTime"]).timestamp()
    except (KeyError, ValueError) as error:
        raise Closed("Scheduler start time unavailable") from error
    now = time.time() if now is None else now
    deadline = start + duration - 120
    if start > now + 5 or deadline <= now:
        raise Closed("No usable time remains in approved allocation")
    return {"job_id": str(job_id), "cpus": 4, "memory_gib": 16, "gpus": 0,
            "start_epoch": start, "scheduler_time_limit_seconds": duration,
            "work_deadline_epoch": deadline, "node": fields.get("NodeList")}


def authorize(proposal_dir, authorization):
    proposal_dir = prep.writable_path(proposal_dir)
    raw = (proposal_dir / "proposal.json").read_bytes()
    proposal = json.loads(raw)
    auth = json.loads(Path(authorization).read_text())
    job_id = os.environ.get("SLURM_JOB_ID", "")
    if not re.fullmatch(r"[0-9]+", job_id):
        raise Closed("Requires the approved Slurm allocation; login-host execution refused")
    if (auth.get("approved") is not True or auth.get("job_id") != job_id or
            auth.get("proposal_sha256") != prep.sha256(raw) or
            auth.get("resource_request_sha256") != prep.sha256((proposal_dir / "resource-request.json").read_bytes())):
        raise Closed("Exact proposal/resource/job authorization missing or mismatched")
    if proposal.get("resource_request") != prep.resource_request():
        raise Closed("Resource request differs from frozen controller")
    for name, expected in proposal.get("frozen_artifacts_sha256", {}).items():
        if not isinstance(name, str) or Path(name).name != name:
            raise Closed("Invalid frozen artifact name")
        if regular_hash(proposal_dir / name) != expected:
            raise Closed("Frozen proposal artifact changed")
    runner_name = "run_harmful_apptainer_probe.py"
    if proposal.get("frozen_artifacts_sha256", {}).get(runner_name) != regular_hash(Path(__file__).resolve()):
        raise Closed("Runner is not the proposal's frozen implementation")
    result = subprocess.run(["scontrol", "show", "job", "-o", job_id], check=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=20)
    allocation = allocation_from_scontrol(result.stdout, job_id)
    if socket.gethostname().split(".", 1)[0] != (allocation.get("node") or "").split(".", 1)[0]:
        raise Closed("Current host is not the allocated compute node")
    return proposal, allocation


class Ledger:
    """Single-writer cross-attempt body-byte reservation and allocation identity."""
    def __init__(self, root, allocation, proposal_sha256):
        self.root = prep.writable_path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        lock = prep.writable_path(self.root / "writer.lock")
        self.fd = os.open(lock, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.path = self.root / "ledger.json"
            if self.path.exists():
                if any(".pending-" in item.name for item in self.root.iterdir()):
                    raise Closed("Interrupted ledger update requires reconciliation; preserve pending files")
                self.data = json.loads(self.path.read_text())
                if self.data.get("proposal_sha256") != proposal_sha256 or self.data.get("allocation") != allocation:
                    raise Closed("Ledger belongs to another proposal or allocation; cross-job retry requires new independently reviewed accounting")
                count = self.data.get("reserved_body_bytes")
                if type(count) is not int or not 0 <= count <= 4000000000:
                    raise Closed("Invalid transfer ledger counter")
            else:
                # Never silently adopt files left by an unknown previous attempt.
                if set(p.name for p in self.root.iterdir()) != {"writer.lock"}:
                    raise Closed("Nonempty run directory lacks valid ledger")
                self.data = {"schema_version": 1, "proposal_sha256": proposal_sha256,
                             "allocation": allocation, "reserved_body_bytes": 0,
                             "body_byte_cap": 4000000000, "attempts": []}
                self.save()
        except BaseException:
            os.close(self.fd)
            raise

    def save(self):
        atomic_json(self.path, self.data)

    def reserve(self, count, kind):
        if type(count) is not int or count <= 0 or self.data["reserved_body_bytes"] + count > self.data["body_byte_cap"]:
            raise Closed("Aggregate transfer reservation cap would be exceeded")
        self.data["reserved_body_bytes"] += count
        self.data["last_reservation"] = {"kind": kind, "bytes": count, "at_epoch": time.time()}
        self.save()

    def attempt(self):
        if len(self.data["attempts"]) >= MAX_ATTEMPTS:
            raise Closed("Maximum preserved import attempts reached")
        number = len(self.data["attempts"]) + 1
        path = self.root / ("attempt-%04d" % number)
        self.data["attempts"].append({"number": number, "status": "reserved", "path": str(path)})
        self.save()
        path.mkdir(exist_ok=False)
        return path

    def close(self):
        os.close(self.fd)


class HttpsRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not newurl.startswith("https://"):
            raise Closed("Registry redirect must preserve HTTPS")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class Registry:
    def __init__(self, ledger, deadline):
        self.ledger, self.deadline = ledger, deadline
        self.opener = urllib.request.build_opener(HttpsRedirects())
        self.token, self.expiry = None, 0

    def open_blob(self, digest):
        if time.time() >= self.expiry:
            self.ledger.reserve(65536, "anonymous_registry_auth")
            url = "https://auth.docker.io/token?service=registry.docker.io&scope=repository:" + prep.REPOSITORY + ":pull"
            with self.opener.open(url, timeout=remaining(self.deadline)) as response:
                body = response.read(65536)
            if len(body) == 65536:
                raise Closed("Registry authentication metadata exceeds reserved limit")
            auth = json.loads(body)
            self.token = auth["token"]
            self.expiry = time.time() + 120
        request = urllib.request.Request("https://registry-1.docker.io/v2/" + prep.REPOSITORY + "/blobs/" + digest)
        # This header is intentionally not copied to redirected CDN requests.
        request.add_unredirected_header("Authorization", "Bearer " + self.token)
        return self.opener.open(request, timeout=remaining(self.deadline))


def download_blob(layer, destination, partial, ledger, deadline, registry):
    destination, partial = prep.writable_path(destination), prep.writable_path(partial)
    expected = layer["digest"].split(":", 1)[1]
    if destination.exists():
        if destination.stat().st_size != layer["size"] or regular_hash(destination) != expected:
            raise Closed("Existing retained blob is corrupt; preserve and stop")
        return {"digest": layer["digest"], "size": layer["size"], "reused_verified_blob": True}
    # Reserve the complete response before request; failed/unknown bytes never refunded.
    ledger.reserve(layer["size"] + 1, "blob:" + expected)
    count, digest = 0, hashlib.sha256()
    fd = os.open(prep.writable_path(partial), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as handle, registry.open_blob(layer["digest"]) as response:
        if response.status != 200:
            raise Closed("Expected complete pinned registry blob response")
        while count < layer["size"] + 1:
            remaining(deadline)
            chunk = response.read(min(READ_CHUNK, layer["size"] + 1 - count))
            if not chunk:
                break
            count += len(chunk)
            if count > layer["size"]:
                raise Closed("Registry blob exceeds pinned size")
            handle.write(chunk)
            digest.update(chunk)
        handle.flush()
        os.fsync(handle.fileno())
    if count != layer["size"] or digest.hexdigest() != expected:
        raise Closed("Registry blob length or compressed SHA256 mismatch")
    os.rename(prep.writable_path(partial), prep.writable_path(destination))
    return {"digest": layer["digest"], "size": count, "reused_verified_blob": False}


class RawLayer(io.RawIOBase):
    def __init__(self, stream, total, deadline):
        self.stream, self.total, self.deadline = stream, total, deadline
        self.hash = hashlib.sha256()
        self.bytes = 0

    def readable(self):
        return True

    def read(self, size=-1):
        remaining(self.deadline)
        # Bound decompression work; no decompressed rootfs is written to disk.
        left = 100 * 1024**3 - self.total[0]
        chunk = self.stream.read(min(READ_CHUNK if size < 0 else size, left + 1))
        if len(chunk) > left:
            raise Closed("Decompressed inspection work exceeds 100 GiB cap")
        self.total[0] += len(chunk)
        self.bytes += len(chunk)
        self.hash.update(chunk)
        return chunk


def member_path(name):
    if len(name.encode("utf-8")) > MAX_PATH_BYTES:
        raise Closed("Archive path exceeds bounded metadata length")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise Closed("Archive contains unsafe path")
    return str(path)


def interesting(name, startup_candidates=()):
    return (name in {"custom-entrypoint.sh", "docker-entrypoint.sh", "etc/os-release", "etc/alpine-release",
                     "etc/passwd", "etc/group", "etc/supervisord.conf", "etc/my.cnf"} or
            name in startup_candidates or
            name.startswith(("etc/supervisor.d/", "etc/supervisor/conf.d/", "etc/mysql/")) or
            ("entrypoint" in PurePosixPath(name).name and name.endswith(".sh")))


def startup_paths(config):
    """Inspect explicit startup paths and PATH candidates without executing them."""
    settings = config.get("config", {})
    env = dict(value.split("=", 1) for value in settings.get("Env", []) if "=" in value)
    prefixes = [value.strip("/") for value in env.get("PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin").split(":") if value.startswith("/")]
    required, commands = set(), {}
    for key in ("Entrypoint", "Cmd"):
        argv = settings.get(key) or []
        if not isinstance(argv, list):
            raise Closed("Startup argv must be a list")
        for value in argv:
            if not isinstance(value, str):
                raise Closed("Startup argv contains a non-string")
            if value.startswith("/"):
                required.add(member_path(value.lstrip("/")))
        if argv and not argv[0].startswith("/"):
            executable = argv[0]
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", executable):
                commands[executable] = []
            else:
                commands[executable] = [member_path(prefix + "/" + executable) for prefix in prefixes]
    candidates = required | {name for choices in commands.values() for name in choices}
    return required, commands, candidates


def inspect_layers(manifest, config, blobs, output, deadline):
    total, captured, captures, member_count = [0], 0, {}, 0
    layer_rows = []
    required, commands, startup_candidates = startup_paths(config)
    files = output / "inspected-files"
    files.mkdir(exist_ok=False)
    for index, (layer, diff_id) in enumerate(zip(manifest["layers"], config["rootfs"]["diff_ids"])):
        layer_captures, whiteouts, opaque, replacements = {}, [], [], []
        path = blobs / layer["digest"].split(":", 1)[1]
        with gzip.open(path, "rb") as decompressed:
            raw = RawLayer(decompressed, total, deadline)
            with tarfile.open(fileobj=raw, mode="r|") as archive:
                for member in archive:
                    member_count += 1
                    if member_count > MAX_TAR_MEMBERS:
                        raise Closed("Archive member count exceeds inspection limit")
                    name = member_path(member.name)
                    basename = PurePosixPath(name).name
                    parent = str(PurePosixPath(name).parent)
                    if basename == ".wh..wh..opq":
                        opaque.append(parent)
                        continue
                    if basename.startswith(".wh."):
                        whiteouts.append(str(PurePosixPath(parent) / basename[4:]))
                        continue
                    # A file/symlink ancestor can replace a lower-layer directory.
                    # Track only replacements affecting selected lower paths to
                    # keep metadata bounded by the capture count.
                    if not member.isdir() and any(old.startswith(name + "/") for old in captures):
                        replacements.append(name)
                    if not interesting(name, startup_candidates):
                        continue
                    if len(layer_captures) + len(captures) >= MAX_CAPTURE_FILES:
                        raise Closed("Too many startup inspection files")
                    row = {"path": name, "layer": index, "mode": member.mode,
                           "uid": member.uid, "gid": member.gid, "type": member.type.decode("ascii", "replace")}
                    if member.isfile():
                        if member.size > MAX_FILE_BYTES or captured + member.size > MAX_CAPTURE_BYTES:
                            raise Closed("Startup capture byte limit exceeded")
                        value = archive.extractfile(member).read(MAX_FILE_BYTES + 1)
                        if len(value) != member.size:
                            raise Closed("Startup file length mismatch")
                        captured += len(value)
                        digest = prep.sha256(value)
                        target = files / digest
                        if not target.exists():
                            write_new(target, value)
                        row.update(sha256=digest, size=len(value), artifact="inspected-files/" + digest)
                    elif member.issym() or member.islnk():
                        if len(member.linkname.encode("utf-8")) > MAX_PATH_BYTES:
                            raise Closed("Archive link target exceeds bounded metadata length")
                        row.update(link_target=member.linkname, target_resolved=False)
                    layer_captures[name] = row
            # Tar readers stop at end-of-archive; hash the complete original diff stream.
            while raw.read(READ_CHUNK):
                pass
            if "sha256:" + raw.hash.hexdigest() != diff_id:
                raise Closed("Full decompressed layer diff_id mismatch")
            layer_rows.append({"layer": index, "compressed_digest": layer["digest"], "diff_id": diff_id,
                               "decompressed_bytes": raw.bytes, "raw_diff_id_verified": True})
        # OCI whiteouts affect lower layers, not same-layer paths, independent of tar order.
        for old in list(captures):
            if any(old == target or old.startswith(target + "/") for target in whiteouts) or any(
                    prefix == "." or old.startswith(prefix + "/") for prefix in opaque) or any(
                    old.startswith(prefix + "/") for prefix in replacements):
                captures.pop(old)
        captures.update(layer_captures)
    return {"status": "static_inspection_complete_requires_agent_startup_review",
            "image_executed": False, "apptainer_invoked": False,
            "decompressed_bytes_read": total[0], "tar_members_scanned": member_count,
            "captured_file_bytes": captured, "layers": layer_rows,
            "final_startup_files": sorted(captures.values(), key=lambda row: row["path"]),
            "required_startup_paths_missing": sorted(required - captures.keys()),
            "metadata_startup_required_paths": sorted(required),
            "startup_command_path_candidates": commands,
            "startup_commands_unresolved": sorted(command for command, choices in commands.items() if not any(name in captures for name in choices)),
            "limitations": ["Selected final startup files inspected; complete rootfs not extracted",
                            "Symlink and hardlink startup targets require explicit resolution before startup",
                            "No service, login, snapshot, database schema, or safety-task admission is validated"]}


def run(proposal_dir, authorization, run_dir):
    proposal_dir = prep.writable_path(proposal_dir)
    run_dir = prep.writable_path(run_dir)
    if prep.within(run_dir, proposal_dir) or prep.within(proposal_dir, run_dir):
        raise Closed("Frozen proposal and mutable run directories must not overlap")
    proposal, allocation = authorize(proposal_dir, authorization)
    deadline = allocation["work_deadline_epoch"]
    manifest = json.loads((proposal_dir / "registry-manifest.json").read_text())
    config = json.loads((proposal_dir / "image-config.json").read_text())
    ledger = Ledger(run_dir, allocation, prep.sha256((proposal_dir / "proposal.json").read_bytes()))
    attempt = None
    previous = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(Closed("Approved import deadline reached")))
    signal.setitimer(signal.ITIMER_REAL, max(0.001, deadline - time.time()))
    try:
        attempt = ledger.attempt()
        if shutil.disk_usage(run_dir).free < DISK_WRITE_BOUND:
            raise Closed("Retained filesystem lacks conservative import write-bound headroom")
        blobs = run_dir / "blobs"
        blobs.mkdir(exist_ok=True)
        registry, rows = Registry(ledger, deadline), []
        for index, layer in enumerate(manifest["layers"]):
            rows.append(download_blob(layer, blobs / layer["digest"].split(":", 1)[1],
                                      attempt / ("layer-%02d.partial" % index), ledger, deadline, registry))
        inspection = inspect_layers(manifest, config, blobs, attempt, deadline)
        atomic_json(attempt / "inspection.json", inspection)
        receipt = {"status": "pinned_image_import_and_static_inspection_complete",
                   "proposal_sha256": ledger.data["proposal_sha256"], "allocation": allocation,
                   "image": proposal["image"], "compressed_blobs": rows,
                   "inspection_sha256": regular_hash(attempt / "inspection.json"),
                   "reserved_body_bytes_all_attempts": ledger.data["reserved_body_bytes"],
                   "disk_write_bound_bytes": DISK_WRITE_BOUND,
                   "disk_bound_reason": "Only reserved compressed bytes, <=32 MiB captured source plus64 MiB metadata allowance per attempt,64 MiB global overhead; no rootfs extraction or external child process",
                   "node_scratch_files_written": 0,
                   "startup_review_required": True, "service_startup_validated": False,
                   "finished_epoch": time.time()}
        atomic_json(attempt / "complete.json", receipt)
        ledger.data["attempts"][-1]["status"] = "complete"
        ledger.save()
        return {"status": receipt["status"], "attempt": str(attempt),
                "reserved_body_bytes_all_attempts": ledger.data["reserved_body_bytes"],
                "service_startup_validated": False}
    except BaseException as exc:
        if attempt is not None:
            atomic_json(attempt / "failure.json", {"status": "failed_preserved", "error_type": type(exc).__name__,
                        "reason": str(exc) if isinstance(exc, Closed) else "Details suppressed; private local inspection required",
                        "reserved_body_bytes_all_attempts": ledger.data["reserved_body_bytes"], "at_epoch": time.time()})
            ledger.data["attempts"][-1]["status"] = "failed_preserved"
            ledger.save()
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        ledger.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal-dir", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(args.proposal_dir, args.authorization, args.run_dir)
    except (Closed, OSError, ValueError) as exc:
        parser.error(str(exc) if isinstance(exc, Closed) else "Probe failed; preserved local receipts contain classification")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
