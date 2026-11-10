"""Restore the sealed initialized native baseline inside an approved worker.

These blocking file operations belong in a supervised worker, never the deadline
watchdog. Restoring files proves neither native reset equivalence nor task safety.
The caller supplies the prospective allocation context and owns worker closure.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import time

from openwebrl.harmful_native_recovery import binding, checked_path, digest, need


def _deadline(deadline):
    need(time.monotonic() < deadline, "Filesystem verification deadline exhausted")


def _relative(value):
    path = Path(value)
    need(value == "." or (not path.is_absolute() and ".." not in path.parts and
                          str(path) == value), "Invalid manifest path")
    return path


def verify_filesystem(root, manifest, *, expected_manifest_sha256, deadline):
    """Independently check every object, metadata field and hardlink group.

    Enumeration and file reads use directory descriptors with O_NOFOLLOW. The
    expected ordering is canonical depth-first lexical order, as in the sealed
    historical manifest. Files, sockets and symlinks are inspected, never run.
    """
    root = Path(root)
    need(root.is_absolute() and not any(p.is_symlink() for p in (root, *root.parents)),
         "Unsafe filesystem root")
    _deadline(deadline)
    data = Path(manifest).read_bytes()
    need(hashlib.sha256(data).hexdigest() == expected_manifest_sha256,
         "Canonical manifest changed")
    rows = [json.loads(line) for line in data.splitlines()]
    need(rows and rows[0]["path"] == ".", "Manifest must begin with root")
    names = [row["path"] for row in rows]
    need(len(names) == len(set(names)), "Repeated manifest object")
    for name in names:
        _relative(name)
    iterator, hardlinks, examined = iter(rows), {}, []
    regular_bytes, sockets, hardlink_members = 0, 0, 0

    def identity(value):
        return (value.st_dev, value.st_ino, value.st_mode, value.st_uid, value.st_gid,
                value.st_size, value.st_mtime_ns, value.st_ctime_ns, value.st_nlink)

    def visit(name, parent_fd=None, entry=None):
        nonlocal regular_bytes, sockets, hardlink_members
        _deadline(deadline)
        row = next(iterator, None)
        need(row is not None and row["path"] == name, "Filesystem coverage or ordering differs")
        target = root if parent_fd is None else entry
        before = os.stat(target, dir_fd=parent_fd, follow_symlinks=False)
        need((stat.S_IMODE(before.st_mode), before.st_uid, before.st_gid, before.st_mtime_ns) ==
             (row["mode"], row["uid"], row["gid"], row["mtime_ns"]), "Filesystem metadata differs: " + name)
        kind = row["kind"]
        if kind == "directory":
            need(stat.S_ISDIR(before.st_mode), "Directory type differs")
            fd = os.open(target, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
            try:
                need(identity(os.fstat(fd)) == identity(before), "Directory changed while opening")
                for child in sorted(os.listdir(fd)):
                    visit(child if name == "." else name + "/" + child, fd, child)
                need(identity(os.fstat(fd)) == identity(before), "Directory changed during inspection")
            finally:
                os.close(fd)
        elif kind == "file":
            need(stat.S_ISREG(before.st_mode) and before.st_size == row["bytes"], "Regular file differs")
            first = hardlinks.setdefault((before.st_dev, before.st_ino), name)
            need(first == row["hardlink_first_path"], "Hardlink structure differs")
            fd = os.open(target, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent_fd)
            with os.fdopen(fd, "rb") as stream:
                need(identity(os.fstat(stream.fileno())) == identity(before), "File changed while opening")
                value = hashlib.sha256()
                for chunk in iter(lambda: stream.read(1024**2), b""):
                    _deadline(deadline)
                    value.update(chunk)
                need(value.hexdigest() == row["sha256"], "Regular file content differs: " + name)
                need(identity(os.fstat(stream.fileno())) == identity(before), "File changed during read")
            regular_bytes += row["bytes"]
            hardlink_members += int(first != name)
        elif kind == "symlink":
            need(stat.S_ISLNK(before.st_mode) and
                 os.readlink(target, dir_fd=parent_fd) == row["target"], "Symlink differs")
        elif kind == "socket":
            need(stat.S_ISSOCK(before.st_mode) and before.st_rdev == row["rdev"], "Socket differs")
            sockets += 1
        else:
            raise ValueError("Unsupported native filesystem object")
        after = os.stat(target, dir_fd=parent_fd, follow_symlinks=False)
        need(identity(after) == identity(before), "Filesystem changed during inspection")
        examined.append((name, identity(after)))

    visit(".")
    need(next(iterator, None) is None, "Filesystem omitted canonical objects")
    # Recheck identities after the full traversal; do not accept a moving tree.
    for name, expected in examined:
        _deadline(deadline)
        path = root / name
        need(not any(p.is_symlink() for p in path.parents if p == root or root in p.parents),
             "Parent changed to symlink")
        need(identity(path.lstat()) == expected, "Object changed after inspection")
    need(digest(manifest) == expected_manifest_sha256, "Manifest changed during inspection")
    return {"status": "full_filesystem_equal", "equal": True, "entries": len(rows),
            "baseline_manifest_sha256": expected_manifest_sha256,
            "fresh_manifest_sha256": expected_manifest_sha256,
            "regular_file_bytes": regular_bytes, "socket_entries": sockets,
            "hardlink_members": hardlink_members, "native_reset_verified": False}


def restore_baseline(*, context, contract_binding, original_baseline_binding,
                     deadline, destination=None):
    """Return a new provenance receipt after exact restore and independent check.

    context.authorize() must check the exact approval, scheduler identity, ledger,
    source bindings and deadline before this function writes a fresh destination.
    This function never starts a service or copies an old allocation receipt.
    """
    allocation = context.authorize()
    runtime = context.runtime_root
    historical = context.historical_root
    scratch = context.scratch_root
    destination = checked_path(destination or scratch / "initialized-baseline", scratch)
    need(destination != scratch and not destination.exists(), "Restore needs a fresh child destination")
    need(destination.parent.is_dir(), "Restore parent absent")
    need(deadline <= time.monotonic() + allocation["work_deadline_epoch"] - time.time(),
         "Restore deadline exceeds allocation work budget")
    before = time.time()
    paths = {}
    for label, spec in (("contract", contract_binding), ("original_baseline", original_baseline_binding)):
        path = checked_path(spec["path"], historical)
        need(digest(path) == spec["sha256"], "Historical restore binding changed: " + label)
        paths[label] = path
    contract = json.loads(paths["contract"].read_text())
    original = json.loads(paths["original_baseline"].read_text())
    need(contract["status"] == "durable_baseline_restore_contract_sealed", "Restore contract unsealed")
    need(original["kind"] == "initialized_build" and original["status"] == "complete"
         and original["closure_verified"] is True, "Original baseline provenance invalid")
    for label in ("helper", "manifest", "archive", "evidence", "owner_closure"):
        item = contract[label]
        path = checked_path(item["path"], historical)
        _deadline(deadline)
        need(digest(path) == item["sha256"], "Sealed restore component changed: " + label)
        if "bytes" in item:
            need(path.stat().st_size == item["bytes"], "Restore component length differs")
        paths[label] = path
    archive_identity = paths["archive"].stat()
    source = importlib.util.spec_from_file_location("sealed_native_archive_helper", paths["helper"])
    helper = importlib.util.module_from_spec(source)
    source.loader.exec_module(helper)
    context.authorize()
    helper.restore(paths["archive"], paths["manifest"], destination,
                   expected_archive_sha256=contract["archive"]["sha256"],
                   expected_manifest_sha256=contract["manifest"]["sha256"],
                   deadline=time.time() + deadline - time.monotonic())
    check = verify_filesystem(destination, paths["manifest"],
                              expected_manifest_sha256=contract["manifest"]["sha256"], deadline=deadline)
    after = paths["archive"].stat()
    need((after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns) ==
         (archive_identity.st_dev, archive_identity.st_ino, archive_identity.st_size,
          archive_identity.st_mtime_ns, archive_identity.st_ctime_ns), "Archive changed during restore")
    for label in ("helper", "manifest", "contract", "original_baseline"):
        spec = contract[label] if label in contract else (contract_binding if label == "contract" else original_baseline_binding)
        need(digest(paths[label]) == spec["sha256"], "Restore custody changed")
    context.authorize()
    return {"schema_version": 1, "status": "complete", "kind": "restored_initialized_baseline",
            "job_id": allocation["job_id"], "runtime_root": str(runtime),
            "source": str(paths["archive"]), "destination": str(destination),
            "destination_identity": {"device": destination.stat().st_dev, "inode": destination.stat().st_ino},
            "restore_contract": contract_binding, "original_baseline_receipt": original_baseline_binding,
            "archive_sha256": contract["archive"]["sha256"],
            "initialized_baseline_manifest_sha256": contract["manifest"]["sha256"],
            "cached_baseline_sha256": original["cached_baseline_sha256"],
            "image_sha256": original["image_sha256"], "full_files_comparison": check,
            "never_start_or_modify": True, "services_started": False,
            "native_reset_verified": False, "started_epoch": before, "finished_epoch": time.time(),
            "controller": binding(Path(__file__))}


def verify_fresh_clone(*, context, baseline_binding, manifest_binding,
                       destination, deadline, owned_copy_receipt):
    """Verify a caller-owned cp -a clone without starting native applications.

    The caller must exclusively create and record its instance request *before*
    copying, charge the global all-attempt world counter, reserve storage growth,
    and preserve the supervised copy receipt even on failure. This function
    cannot replace those launch-time duties with a post-hoc file comparison.
    """
    allocation = context.authorize()
    destination = checked_path(destination, context.scratch_root)
    baseline_path = checked_path(baseline_binding["path"], context.runtime_root)
    need(digest(baseline_path) == baseline_binding["sha256"], "Fresh baseline receipt changed")
    baseline = json.loads(baseline_path.read_text())
    need(baseline["kind"] == "restored_initialized_baseline" and baseline["status"] == "complete"
         and baseline["job_id"] == allocation["job_id"], "Wrong prospective baseline lineage")
    source = checked_path(baseline["destination"], context.scratch_root)
    need(source == context.scratch_root / "initialized-baseline" and destination != source,
         "Clone must be an independent baseline descendant")
    expected_argv = ["cp", "-a", "--reflink=auto", "--", str(source), str(destination)]
    need(owned_copy_receipt.get("argv") == expected_argv and
         owned_copy_receipt.get("returncode") == 0 and
         owned_copy_receipt.get("closure_verified") is True,
         "Fresh clone lacks verified owned copy closure")
    identity = lambda path: {"device": path.stat().st_dev, "inode": path.stat().st_ino}
    need(identity(source) == baseline["destination_identity"] and
         identity(destination) != identity(source), "Baseline or clone identity differs")
    manifest_path = checked_path(manifest_binding["path"], context.historical_root)
    need(manifest_binding["sha256"] == baseline["initialized_baseline_manifest_sha256"],
         "Clone manifest does not match restored baseline")
    comparison = verify_filesystem(destination, manifest_path,
                                   expected_manifest_sha256=manifest_binding["sha256"], deadline=deadline)
    # Reflink sharing is allowed; an ordinary hardlink to the immutable baseline
    # is not. Root-directory identity alone cannot prove independent files.
    for line in manifest_path.read_text().splitlines():
        row = json.loads(line)
        _deadline(deadline)
        if row["kind"] != "file":
            continue
        left, right = (source / _relative(row["path"])).lstat(), (destination / row["path"]).lstat()
        need((left.st_dev, left.st_ino) != (right.st_dev, right.st_ino),
             "Clone aliases an immutable baseline file")
    need(digest(baseline_path) == baseline_binding["sha256"], "Baseline custody changed")
    context.authorize()
    return {"schema_version": 1, "status": "complete", "kind": "copy",
            "job_id": allocation["job_id"], "name": destination.name,
            "source": str(source), "destination": str(destination),
            "destination_identity": identity(destination),
            "initialized_baseline_receipt_sha256": baseline_binding["sha256"],
            "initialized_baseline_manifest_sha256": manifest_binding["sha256"],
            "cached_baseline_sha256": baseline["cached_baseline_sha256"],
            "image_sha256": baseline["image_sha256"],
            "baseline_provenance_kind": baseline["kind"],
            "owned_copy_receipt": owned_copy_receipt, "full_files_comparison": comparison,
            "native_reset_verified": False, "services_started": False,
            "controller": binding(Path(__file__))}
