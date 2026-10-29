"""Offline tests of source reconstruction and fail-closed write boundaries."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("api_source_materializer", ROOT / "materialize.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MaterializeTests(unittest.TestCase):
    def test_all_campaigns_restore_original_hashes_and_names(self):
        index = json.loads((ROOT / "source-provenance.json").read_text())
        with tempfile.TemporaryDirectory() as temporary:
            for campaign, entry in index["snapshots"].items():
                output = Path(temporary) / campaign
                MODULE.materialize(campaign, output)
                self.assertEqual({p.name for p in output.iterdir()}, {r["file"] for r in entry["files"]})
                for row in entry["files"]:
                    self.assertEqual(hashlib.sha256((output / row["file"]).read_bytes()).hexdigest(), row["sha256"])

    def test_existing_output_is_untouched(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "existing"
            output.mkdir()
            sentinel = output / "keep"
            sentinel.write_text("unchanged")
            with self.assertRaises(FileExistsError):
                MODULE.materialize("browsercode_luna_effort", output)
            self.assertEqual(sentinel.read_text(), "unchanged")
            self.assertEqual(list(output.iterdir()), [sentinel])

    def test_bad_hash_or_source_escape_creates_no_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            source.mkdir()
            (source / "sample.py").write_text("pass\n")
            row = {"file": "sample.py", "published_path": "sample.py", "sha256": "0" * 64}
            manifest = {"snapshots": {"example": {"files": [row]}}}
            output = Path(temporary) / "output"
            for location in ["sample.py", "../outside.py"]:
                row["published_path"] = location
                (source / "source-provenance.json").write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):
                    MODULE.materialize("example", output, source_root=source)
                self.assertFalse(output.exists())

    def test_unknown_campaign_creates_no_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output"
            with self.assertRaises(ValueError):
                MODULE.materialize("missing", output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
