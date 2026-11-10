import hashlib
import json
import os
from pathlib import Path
import socket
import stat
import subprocess
import tempfile
import time
import unittest
from types import SimpleNamespace

from openwebrl.harmful_native_restore import verify_filesystem, verify_fresh_clone


class FilesystemRestoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="native-restore-test-", dir="/tmp")
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root = self.base / "world"
        self.root.mkdir()
        (self.root / "a").write_bytes(b"actual native fixture\x00")
        os.link(self.root / "a", self.root / "b")
        (self.root / "dir").mkdir()
        (self.root / "dir/c").write_bytes(b"nested")
        (self.root / "link").symlink_to("/outside-is-never-followed")
        with socket.socket(socket.AF_UNIX) as value:
            value.bind(str(self.root / "sock"))
        self.manifest = self.base / "manifest.jsonl"
        self.rows = []
        links = {}
        # Test fixture implements the external canonical format directly.
        for path in [self.root, self.root / "a", self.root / "b", self.root / "dir",
                     self.root / "dir/c", self.root / "link", self.root / "sock"]:
            meta = path.lstat()
            name = str(path.relative_to(self.root))
            row = dict(path=name, mode=stat.S_IMODE(meta.st_mode), uid=meta.st_uid,
                       gid=meta.st_gid, mtime_ns=meta.st_mtime_ns)
            if stat.S_ISDIR(meta.st_mode):
                row.update(kind="directory")
            elif stat.S_ISREG(meta.st_mode):
                row.update(kind="file", bytes=meta.st_size,
                           sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                           hardlink_first_path=links.setdefault(meta.st_ino, name))
            elif stat.S_ISLNK(meta.st_mode):
                row.update(kind="symlink", target=os.readlink(path))
            else:
                row.update(kind="socket", rdev=meta.st_rdev)
            self.rows.append(row)
        self.seal()

    def seal(self):
        self.manifest.write_text("".join(json.dumps(row) + "\n" for row in self.rows))
        self.sha = hashlib.sha256(self.manifest.read_bytes()).hexdigest()

    def verify(self, **kwargs):
        return verify_filesystem(self.root, self.manifest,
                                 expected_manifest_sha256=self.sha,
                                 deadline=kwargs.get("deadline", time.monotonic() + 5))

    def test_complete_inventory_including_socket_symlink_and_hardlink(self):
        result = self.verify()
        self.assertTrue(result["equal"])
        self.assertEqual(result["entries"], 7)
        self.assertEqual(result["socket_entries"], 1)
        self.assertEqual(result["hardlink_members"], 1)
        self.assertFalse(result["native_reset_verified"])

    def test_same_length_content_tamper_fails_even_with_preserved_mtime(self):
        path = self.root / "a"
        before = path.stat()
        path.write_bytes(b"X" * before.st_size)
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        with self.assertRaisesRegex(ValueError, "content differs"):
            self.verify()

    def test_hardlink_copy_with_identical_content_and_metadata_fails(self):
        original = self.root / "b"
        old = self.base / "preserved-b"
        original.rename(old)
        original.write_bytes(old.read_bytes())
        meta = old.stat()
        os.chmod(original, stat.S_IMODE(meta.st_mode))
        os.utime(original, ns=(meta.st_atime_ns, meta.st_mtime_ns))
        root_row = self.rows[0]
        os.utime(self.root, ns=(root_row["mtime_ns"], root_row["mtime_ns"]))
        with self.assertRaisesRegex(ValueError, "Hardlink structure"):
            self.verify()

    def test_extra_file_and_wrong_type_fail(self):
        (self.root / "unexpected").write_text("extra")
        row = self.rows[0]
        os.utime(self.root, ns=(row["mtime_ns"], row["mtime_ns"]))
        with self.assertRaisesRegex(ValueError, "coverage"):
            self.verify()
        self.rows[-1]["kind"] = "directory"
        self.seal()
        with self.assertRaisesRegex(ValueError, "Directory type"):
            self.verify()

    def test_path_escape_duplicate_and_source_seal_refused(self):
        self.rows[1]["path"] = "../a"
        self.seal()
        with self.assertRaisesRegex(ValueError, "Invalid manifest path"):
            self.verify()
        self.rows[1]["path"] = "."
        self.seal()
        with self.assertRaisesRegex(ValueError, "Repeated manifest"):
            self.verify()
        self.manifest.write_text("changed")
        with self.assertRaisesRegex(ValueError, "manifest changed"):
            self.verify()

    def test_expired_deadline_and_symlink_root_refused(self):
        with self.assertRaisesRegex(ValueError, "deadline"):
            self.verify(deadline=time.monotonic() - 1)
        path = self.base / "alias"
        path.symlink_to(self.root)
        with self.assertRaisesRegex(ValueError, "Unsafe filesystem root"):
            verify_filesystem(path, self.manifest, expected_manifest_sha256=self.sha,
                              deadline=time.monotonic() + 5)

    def test_new_clone_keeps_original_lineage_but_needs_new_job_and_closure(self):
        baseline = self.base / "initialized-baseline"
        self.root.rename(baseline)
        runtime = self.base / "runtime"
        runtime.mkdir()
        historical = self.base / "old"
        historical.mkdir()
        manifest = historical / "manifest.jsonl"
        manifest.write_bytes(self.manifest.read_bytes())
        receipt_path = runtime / "baseline.json"
        receipt = {"kind": "restored_initialized_baseline", "status": "complete",
                   "job_id": "900001", "destination": str(baseline),
                   "destination_identity": {"device": baseline.stat().st_dev, "inode": baseline.stat().st_ino},
                   "initialized_baseline_manifest_sha256": self.sha,
                   "cached_baseline_sha256": "a" * 64, "image_sha256": "b" * 64}
        receipt_path.write_text(json.dumps(receipt))
        bound = lambda path: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        ctx = SimpleNamespace(runtime_root=runtime, historical_root=historical, scratch_root=self.base,
                              authorize=lambda: {"job_id": "900001"})
        destination = self.base / "instance-1"
        argv = ["cp", "-a", "--reflink=auto", "--", str(baseline), str(destination)]
        subprocess.run(argv, check=True)
        owned = {"argv": argv, "returncode": 0, "closure_verified": True}
        def verify():
            return verify_fresh_clone(context=ctx, baseline_binding=bound(receipt_path),
                    manifest_binding=bound(manifest), destination=destination,
                    deadline=time.monotonic() + 5, owned_copy_receipt=owned)
        result = verify()
        self.assertEqual(result["baseline_provenance_kind"], "restored_initialized_baseline")
        self.assertFalse(result["native_reset_verified"])
        owned["closure_verified"] = False
        with self.assertRaisesRegex(ValueError, "owned copy closure"):
            verify()
        owned["closure_verified"] = True
        receipt["job_id"] = "355424"
        receipt_path.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, "prospective baseline lineage"):
            verify()
        receipt["job_id"] = "900001"
        receipt_path.write_text(json.dumps(receipt))
        for name in ("a", "b"):
            (destination / name).rename(self.base / ("preserved-copy-" + name))
            os.link(baseline / name, destination / name)
        root_time = self.rows[0]["mtime_ns"]
        os.utime(destination, ns=(root_time, root_time))
        with self.assertRaisesRegex(ValueError, "aliases an immutable baseline"):
            verify()


if __name__ == "__main__":
    unittest.main()
