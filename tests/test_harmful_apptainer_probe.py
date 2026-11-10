"""Offline tests of transfer caps, scheduler admission and static OCI inspection."""
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_harmful_apptainer_probe as worker


def archive(entries):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as handle:
        for name, body in entries:
            info = tarfile.TarInfo(name)
            if isinstance(body, tuple):
                info.type = tarfile.SYMTYPE
                info.linkname = body[0]
                handle.addfile(info)
            else:
                info.size = len(body)
                handle.addfile(info, io.BytesIO(body))
    raw = buffer.getvalue()
    return gzip.compress(raw), "sha256:" + hashlib.sha256(raw).hexdigest()


class Response(io.BytesIO):
    status = 200


class Registry:
    def __init__(self, values):
        self.values, self.calls = values, 0

    def open_blob(self, digest):
        self.calls += 1
        return Response(self.values[digest])


class ImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="harmful-import-test-", dir="/tmp")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.allocation = {"job_id": "123", "work_deadline_epoch": time.time() + 300}
        self.ledger = worker.Ledger(self.root / "run", self.allocation, "fixed-proposal")
        self.addCleanup(self.ledger.close)

    def test_reservations_survive_failed_attempt_without_refund(self):
        value = b"actual"
        digest = "sha256:" + "a" * 64
        layer = {"digest": digest, "size": len(value)}
        attempt = self.ledger.attempt()
        with self.assertRaisesRegex(worker.Closed, "SHA256 mismatch"):
            worker.download_blob(layer, attempt / "blob", attempt / "partial", self.ledger,
                                 time.time() + 30, Registry({digest: value}))
        self.assertEqual(self.ledger.data["reserved_body_bytes"], len(value) + 1)
        self.assertEqual((attempt / "partial").read_bytes(), value)
        saved = json.loads((self.root / "run/ledger.json").read_text())
        self.assertEqual(saved["reserved_body_bytes"], len(value) + 1)

    def test_over_cap_fails_before_network(self):
        self.ledger.reserve(3999999999, "previous-attempts")
        registry = Registry({})
        layer = {"digest": "sha256:" + "a" * 64, "size": 2}
        with self.assertRaisesRegex(worker.Closed, "cap"):
            worker.download_blob(layer, self.root / "blob", self.root / "partial", self.ledger,
                                 time.time() + 30, registry)
        self.assertEqual(registry.calls, 0)
        self.assertFalse((self.root / "partial").exists())

    def test_oversized_response_never_writes_extra_byte(self):
        layer = {"digest": "sha256:" + "a" * 64, "size": 2}
        with self.assertRaisesRegex(worker.Closed, "exceeds pinned size"):
            worker.download_blob(layer, self.root / "blob", self.root / "partial", self.ledger,
                                 time.time() + 30, Registry({layer["digest"]: b"overlong"}))
        self.assertLessEqual((self.root / "partial").stat().st_size, 2)
        self.assertEqual(self.ledger.data["reserved_body_bytes"], 3)

    def test_valid_blob_reused_without_new_reservation(self):
        value = b"exact"
        digest = "sha256:" + hashlib.sha256(value).hexdigest()
        layer = {"digest": digest, "size": len(value)}
        registry = Registry({digest: value})
        worker.download_blob(layer, self.root / "blob", self.root / "partial", self.ledger,
                             time.time() + 30, registry)
        result = worker.download_blob(layer, self.root / "blob", self.root / "later-partial", self.ledger,
                                      time.time() + 30, registry)
        self.assertTrue(result["reused_verified_blob"])
        self.assertEqual(registry.calls, 1)
        self.assertEqual(self.ledger.data["reserved_body_bytes"], 6)

    def test_second_writer_rejected(self):
        with self.assertRaises(BlockingIOError):
            worker.Ledger(self.root / "run", self.allocation, "fixed-proposal")

    def test_different_allocation_rejected_after_lock_released(self):
        other = self.root / "other"
        ledger = worker.Ledger(other, self.allocation, "fixed-proposal")
        ledger.close()
        with self.assertRaisesRegex(worker.Closed, "another proposal or allocation"):
            worker.Ledger(other, {**self.allocation, "job_id": "124"}, "fixed-proposal")

    def test_static_inspection_respects_oci_whiteouts(self):
        blobs, output = self.root / "blobs", self.root / "inspection"
        blobs.mkdir()
        output.mkdir()
        layers, diff_ids = [], []
        entries = [[("custom-entrypoint.sh", b"old"), ("docker-entrypoint.sh", b"base"),
                    ("etc/supervisor.d/old.ini", b"old-service")],
                   [("etc/supervisor.d/new.ini", b"new-service"),
                    ("etc/supervisor.d/.wh..wh..opq", b""),
                    (".wh.custom-entrypoint.sh", b""),
                    ("custom-entrypoint.sh", b"new")]]
        for layer_entries in entries:
            data, diff = archive(layer_entries)
            digest = hashlib.sha256(data).hexdigest()
            (blobs / digest).write_bytes(data)
            layers.append({"digest": "sha256:" + digest, "size": len(data)})
            diff_ids.append(diff)
        report = worker.inspect_layers({"layers": layers}, {"rootfs": {"diff_ids": diff_ids}},
                                       blobs, output, time.time() + 30)
        files = {row["path"]: row for row in report["final_startup_files"]}
        self.assertNotIn("etc/supervisor.d/old.ini", files)
        self.assertIn("etc/supervisor.d/new.ini", files)
        self.assertEqual(files["custom-entrypoint.sh"]["sha256"], hashlib.sha256(b"new").hexdigest())
        self.assertFalse(report["image_executed"])
        self.assertEqual(report["required_startup_paths_missing"], [])

    def test_decompressed_hash_mismatch_is_not_accepted(self):
        data, _ = archive([("custom-entrypoint.sh", b"body")])
        digest = hashlib.sha256(data).hexdigest()
        blobs, output = self.root / "blobs", self.root / "inspection"
        blobs.mkdir(); output.mkdir()
        (blobs / digest).write_bytes(data)
        with self.assertRaisesRegex(worker.Closed, "diff_id mismatch"):
            worker.inspect_layers({"layers": [{"digest": "sha256:" + digest}]},
                                  {"rootfs": {"diff_ids": ["sha256:" + "0" * 64]}},
                                  blobs, output, time.time() + 30)

    def test_ancestor_symlink_hides_old_descendant(self):
        blobs, output = self.root / "blobs", self.root / "inspection"
        blobs.mkdir(); output.mkdir()
        layers, diff_ids = [], []
        for rows in [[("etc/supervisor.d/old.ini", b"service")], [("etc/supervisor.d", ("elsewhere",))]]:
            data, diff_id = archive(rows)
            digest = hashlib.sha256(data).hexdigest()
            (blobs / digest).write_bytes(data)
            layers.append({"digest": "sha256:" + digest})
            diff_ids.append(diff_id)
        report = worker.inspect_layers({"layers": layers}, {"rootfs": {"diff_ids": diff_ids}},
                                       blobs, output, time.time() + 30)
        self.assertNotIn("etc/supervisor.d/old.ini", [row["path"] for row in report["final_startup_files"]])

    def test_startup_paths_come_from_actual_config(self):
        required, commands, candidates = worker.startup_paths({"config": {
            "Entrypoint": ["/opt/vendor-start"], "Cmd": ["launcher"], "Env": ["PATH=/custom/bin:/usr/bin"]}})
        self.assertEqual(required, {"opt/vendor-start"})
        self.assertEqual(commands["launcher"], ["custom/bin/launcher", "usr/bin/launcher"])
        self.assertTrue(worker.interesting("opt/vendor-start", candidates))

    def test_metadata_and_path_limits_fail_before_write(self):
        with self.assertRaisesRegex(worker.Closed, "bounded metadata"):
            worker.member_path("a" * 4097)
        with patch.object(worker, "MAX_JSON_BYTES", 10):
            with self.assertRaisesRegex(worker.Closed, "metadata"):
                worker.atomic_json(self.root / "overlong.json", {"value": "a" * 20})
        self.assertFalse((self.root / "overlong.json").exists())

    def test_unsafe_archive_paths_rejected(self):
        for name in ("/etc/passwd", "../etc/passwd", "a/../../escape"):
            with self.subTest(name=name), self.assertRaises(worker.Closed):
                worker.member_path(name)

    def test_expired_work_stops_before_read(self):
        reader = worker.RawLayer(io.BytesIO(b"payload"), [0], time.time() - 1)
        with self.assertRaisesRegex(worker.Closed, "deadline"):
            reader.read(10)

    def test_no_login_host_execution(self):
        proposal = self.root / "proposal"
        proposal.mkdir()
        (proposal / "proposal.json").write_text("{}")
        auth = self.root / "authorization.json"
        auth.write_text("{}")
        with patch.dict(os.environ, {}, clear=True), patch.object(worker.subprocess, "run") as launch:
            with self.assertRaisesRegex(worker.Closed, "login-host"):
                worker.authorize(proposal, auth)
            launch.assert_not_called()

    def test_end_to_end_import_receipt_without_network_or_image_execution(self):
        proposal_dir = self.root / "proposal"
        proposal_dir.mkdir()
        raw, diff_id = archive([("custom-entrypoint.sh", b"#!/bin/sh\n"),
                                ("docker-entrypoint.sh", b"#!/bin/sh\n")])
        digest = "sha256:" + hashlib.sha256(raw).hexdigest()
        manifest = {"layers": [{"digest": digest, "size": len(raw)}]}
        config = {"rootfs": {"diff_ids": [diff_id]}, "config": {"Entrypoint": ["/custom-entrypoint.sh"]}}
        for name, value in [("registry-manifest.json", manifest), ("image-config.json", config),
                            ("proposal.json", {"image": "pinned-test-image"})]:
            (proposal_dir / name).write_bytes(worker.prep.encoded(value))
        with patch.object(worker, "authorize", return_value=({"image": "pinned-test-image"}, self.allocation)), \
                patch.object(worker, "Registry", return_value=Registry({digest: raw})), \
                patch.object(worker.shutil, "disk_usage", return_value=SimpleNamespace(free=100 * 1024**3)):
            result = worker.run(proposal_dir, self.root / "unused-authorization", self.root / "new-run")
        receipt = json.loads((Path(result["attempt"]) / "complete.json").read_text())
        self.assertEqual(receipt["reserved_body_bytes_all_attempts"], len(raw) + 1)
        self.assertFalse(receipt["service_startup_validated"])
        self.assertEqual(receipt["node_scratch_files_written"], 0)
        self.assertLess(receipt["disk_write_bound_bytes"], 30 * 1024**3)
        inspection = json.loads((Path(result["attempt"]) / "inspection.json").read_text())
        self.assertTrue(inspection["layers"][0]["raw_diff_id_verified"])
        self.assertFalse(inspection["apptainer_invoked"])


class AllocationTests(unittest.TestCase):
    def setUp(self):
        self.start = time.time() - 30
        self.stamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.start))
        self.text = ("JobId=123 JobState=RUNNING NumNodes=1 NumCPUs=4 "
                     "AllocTRES=cpu=4,mem=16G,node=1,billing=4 "
                     "TimeLimit=02:00:00 StartTime=" + self.stamp + " NodeList=g001")

    def test_exact_allocation_admitted_with_shutdown_reserve(self):
        result = worker.allocation_from_scontrol(self.text, "123")
        self.assertEqual(result["gpus"], 0)
        self.assertEqual(result["work_deadline_epoch"] - result["start_epoch"], 7080)

    def test_gpu_memory_cpu_or_extra_time_rejected(self):
        cases = [self.text.replace("billing=4", "gres/gpu=1"),
                 self.text.replace("mem=16G", "mem=32G"),
                 self.text.replace("NumCPUs=4", "NumCPUs=8"),
                 self.text.replace("02:00:00", "02:00:01"),
                 self.text.replace("RUNNING", "PENDING")]
        for text in cases:
            with self.subTest(text=text), self.assertRaises(worker.Closed):
                worker.allocation_from_scontrol(text, "123")

    def test_job_id_environment_on_login_host_is_not_authorization(self):
        with tempfile.TemporaryDirectory(prefix="harmful-host-test-", dir="/tmp") as temporary:
            root = Path(temporary)
            runner_name = "run_harmful_apptainer_probe.py"
            raw = Path(worker.__file__).read_bytes()
            (root / runner_name).write_bytes(raw)
            request_raw = worker.prep.encoded(worker.prep.resource_request())
            (root / "resource-request.json").write_bytes(request_raw)
            proposal_raw = worker.prep.encoded({"resource_request": worker.prep.resource_request(),
                        "frozen_artifacts_sha256": {runner_name: worker.prep.sha256(raw)}})
            (root / "proposal.json").write_bytes(proposal_raw)
            (root / "authorization.json").write_bytes(worker.prep.encoded({"approved": True, "job_id": "123",
                "proposal_sha256": worker.prep.sha256(proposal_raw),
                "resource_request_sha256": worker.prep.sha256(request_raw)}))
            with patch.dict(os.environ, {"SLURM_JOB_ID": "123"}), \
                    patch.object(worker.subprocess, "run", return_value=SimpleNamespace(stdout=self.text)), \
                    patch.object(worker.socket, "gethostname", return_value="login01"):
                with self.assertRaisesRegex(worker.Closed, "compute node"):
                    worker.authorize(root, root / "authorization.json")


if __name__ == "__main__":
    unittest.main()
