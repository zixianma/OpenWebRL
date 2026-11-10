"""Preparation invariants; these tests do not validate a live application."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/prepare_harmful_apptainer_probe.py"
spec = importlib.util.spec_from_file_location("harmful_probe_prepare", SCRIPT)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="harmful-probe-test-", dir="/tmp")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "metadata"
        self.source.mkdir()
        # Tiny local metadata fixture; production SHA pins stay exact.
        config = {"architecture": "amd64", "os": "linux",
                  "rootfs": {"diff_ids": ["sha256:" + "a" * 64] * 7},
                  "config": {"Entrypoint": ["/custom-entrypoint.sh"], "Cmd": ["supervisord"]}}
        config_raw = probe.encoded(config)
        manifest = {"schemaVersion": 2,
                    "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
                    "config": {"digest": "sha256:" + probe.sha256(config_raw), "size": len(config_raw)},
                    "layers": [{"mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
                                "size": 1, "digest": "sha256:" + "b" * 64}] * 7}
        manifest_raw = probe.encoded(manifest)
        (self.source / "registry-shopping_admin-amd64.json").write_bytes(manifest_raw)
        (self.source / "registry-shopping_admin-config.json").write_bytes(config_raw)
        for name, value in [("MANIFEST_SHA256", probe.sha256(manifest_raw)),
                            ("CONFIG_SHA256", probe.sha256(config_raw)),
                            ("COMPRESSED_LAYER_BYTES", 7)]:
            mocker = patch.object(probe, name, value)
            mocker.start()
            self.addCleanup(mocker.stop)

    def test_freezes_inputs_and_never_claims_runtime_readiness(self):
        output = self.root / "proposal"
        result = probe.prepare(self.source, output)
        data = json.loads((output / "proposal.json").read_text())
        self.assertFalse(result["launch_ready"])
        self.assertEqual(data["implemented_runtime_cap_enforcement"], "import_and_static_inspection_only")
        self.assertFalse(data["experiment_ready"])
        self.assertTrue(all(not stage["ready"] for stage in data["stages"]))
        self.assertFalse(data["layer_downloads_verified"])
        for name, digest in data["frozen_artifacts_sha256"].items():
            self.assertEqual(probe.sha256((output / name).read_bytes()), digest)
        self.assertEqual(result, probe.prepare(self.source, output))

    def test_request_preserves_caps_and_excludes_other_jobs(self):
        request = probe.resource_request()
        self.assertEqual((request["cpus"], request["memory_gib"], request["gpus"]), (4, 16, 0))
        self.assertEqual(request["total_allocation_seconds_including_all_attempts"], 7200)
        self.assertEqual(request["api_budget_usd"], 0)
        self.assertEqual(request["node_local_scratch_cap_bytes"], 100 * 1024**3)
        self.assertEqual(request["retained_artifacts_cap_bytes"], 30 * 1024**3)
        self.assertEqual(request["all_attempts_transfer_reservation_cap_bytes"], 4000000000)
        self.assertEqual(request["older_allocation_seconds_added"], 0)
        self.assertFalse(request["gitlab_in_scope"])
        self.assertEqual(request["status"], "proposed_not_authorized")

    def test_modified_metadata_fails_before_creating_output(self):
        with (self.source / "registry-shopping_admin-amd64.json").open("ab") as handle:
            handle.write(b" ")
        output = self.root / "proposal"
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            probe.prepare(self.source, output)
        self.assertFalse(output.exists())

    def test_different_existing_artifact_is_preserved(self):
        output = self.root / "proposal"
        probe.prepare(self.source, output)
        request = output / "resource-request.json"
        request.write_text("preserve me")
        with self.assertRaisesRegex(ValueError, "Existing preparation differs"):
            probe.prepare(self.source, output)
        self.assertEqual(request.read_text(), "preserve me")

    def test_unrelated_directory_is_not_adopted(self):
        output = self.root / "proposal"
        output.mkdir()
        (output / "owner.json").write_text("other task")
        with self.assertRaisesRegex(ValueError, "exclusive preparation"):
            probe.prepare(self.source, output)
        self.assertEqual(list(output.iterdir()), [output / "owner.json"])

    def test_symlink_artifact_rejected_without_writing_target(self):
        output = self.root / "proposal"
        output.mkdir()
        target = self.root / "retained"
        target.write_text("preserve")
        (output / "proposal.json").symlink_to(target)
        with self.assertRaisesRegex(ValueError, "symlink"):
            probe.prepare(self.source, output)
        self.assertEqual(target.read_text(), "preserve")

    def test_symlink_directory_rejected(self):
        (self.root / "alias").symlink_to(self.root / "new")
        with self.assertRaisesRegex(ValueError, "symlink"):
            probe.prepare(self.source, self.root / "alias" / "proposal")

    def test_protected_and_unapproved_paths_rejected(self):
        for path in ("/gpfs/home/zixianma/new", "/etc/harmful-probe", "relative/path"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                probe.writable_path(path)


if __name__ == "__main__":
    unittest.main()
