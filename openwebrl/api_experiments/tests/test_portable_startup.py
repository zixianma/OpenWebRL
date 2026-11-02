"""Generated-runtime and startup failure checks; no real browser or provider."""
import ast
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


def module(name, source):
    spec = importlib.util.spec_from_file_location(name, source)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


B = module("startup_builder", ROOT / "reproduce/prepare_browsercode.py")
S = module("startup_poll", ROOT / "reproduce/local_browser_startup.py")
VALID = b"12345\n/devtools/browser/01234567-89ab-cdef-0123-456789abcdef\n"


class PortCompletion(unittest.TestCase):
    def test_partial_writes_complete_in_the_same_process_and_deadline(self):
        clock = [0.0]
        reads = iter((b"", b"12345\n", VALID[:-10], VALID))
        proc = SimpleNamespace(poll=mock.Mock(return_value=None))
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            diagnostic = Path(tmp) / "observations.jsonl"
            with mock.patch.object(Path, "read_bytes", side_effect=lambda: next(reads)), \
                    mock.patch.object(S.time, "monotonic", side_effect=lambda: clock[0]), \
                    mock.patch.object(S.time, "sleep", side_effect=lambda seconds: clock.__setitem__(0, clock[0] + seconds)):
                self.assertEqual(S.complete_port(proc, Path(tmp) / "port", 45, diagnostic), VALID.decode().splitlines())
            records = [json.loads(line) for line in diagnostic.read_text().splitlines()]
            self.assertEqual(len(records), 4)
            self.assertTrue(all(row["deadline"] == 45 for row in records))
            self.assertAlmostEqual(clock[0], .3)

    def test_invalid_or_incomplete_metadata_is_never_accepted(self):
        for raw in (b"", b"12345", b"0\n/devtools/browser/01234567-89ab-cdef-0123-456789abcdef", VALID.replace(b"12345", b"65536"), VALID + b"extra\n", VALID[:-10], b"12345\n/devtools/browser/\xff"):
            with self.subTest(raw=raw):
                self.assertIsNone(S.port_lines(raw))

    def test_complete_bytes_after_deadline_or_after_process_exit_fail(self):
        for times, statuses in (([44.9, 44.99, 45.01], [None] * 3), ([1, 1, 1], [None, None, 1])):
            with self.subTest(times=times), tempfile.TemporaryDirectory(dir="/tmp") as tmp:
                proc = SimpleNamespace(poll=mock.Mock(side_effect=statuses))
                with mock.patch.object(Path, "read_bytes", return_value=VALID), \
                        mock.patch.object(S.time, "monotonic", side_effect=times):
                    with self.assertRaisesRegex(RuntimeError, "startup failed"):
                        S.complete_port(proc, Path(tmp) / "port", 45, Path(tmp) / "diagnostic")

    def test_incomplete_bytes_do_not_reset_the_original_deadline(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            proc = SimpleNamespace(poll=lambda: None)
            with mock.patch.object(Path, "read_bytes", return_value=b"12345\n"), \
                    mock.patch.object(S.time, "monotonic", return_value=45), \
                    mock.patch.object(S.time, "sleep") as sleep:
                with self.assertRaisesRegex(RuntimeError, "original45s"):
                    S.complete_port(proc, Path(tmp) / "port", 45, Path(tmp) / "diagnostic")
                sleep.assert_not_called()

    def test_read_error_is_preserved_without_a_retry(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            diagnostic = Path(tmp) / "diagnostic"
            with mock.patch.object(Path, "read_bytes", side_effect=PermissionError(13, "fixture")) as read:
                with self.assertRaises(PermissionError):
                    S.complete_port(SimpleNamespace(poll=lambda: None), Path(tmp) / "port", 45, diagnostic)
            self.assertEqual(read.call_count, 1)
            self.assertEqual(json.loads(diagnostic.read_text())["read_error"], "PermissionError")


class GeneratedRuntime(unittest.TestCase):
    def test_every_campaign_has_safe_owned_startup_and_hash_receipts(self):
        index = json.loads((ROOT / "source-provenance.json").read_text())
        archived = {entry["published_path"]: B.digest(ROOT / entry["published_path"])
                    for campaign in index["snapshots"].values() for entry in campaign["files"]}
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            root = Path(tmp)
            cua, browser, nodes = root / "cua", root / "browser", root / "nodes"
            for path in (cua / "cuawright/webwright/agents/default.py", browser / "packages/bcode-browser/src/browser-execute.ts", nodes / "effect/package.json", root / "bun", root / "chrome"):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture")
            tasks = root / "tasks.json"
            tasks.write_text(json.dumps([dict(task_id="offline", intent="Offline build only", start_url="https://example.invalid/", benchmark="online-mind2web")]))
            for campaign in B.CAMPAIGNS:
                with self.subTest(campaign=campaign), mock.patch.object(B, "verify_dependencies"):
                    output = B.build(campaign, root / campaign, cuawright_source=cua, browsercode_source=browser,
                                     node_modules=nodes, bun=root / "bun", chromium=root / "chrome", tasks_json=tasks)
                    environment = ast.parse((output / "environments.py").read_text())
                    self.assertFalse(any(isinstance(node, ast.FunctionDef) and node.name == "live_page" for node in environment.body))
                    self.assertTrue(any(isinstance(node, ast.ImportFrom) and node.module == "portable_browser_lifecycle" for node in environment.body))
                    owned = next(node for node in environment.body if isinstance(node, ast.ClassDef) and node.name == "OwnedBrowser")
                    start = next(node for node in owned.body if isinstance(node, ast.FunctionDef) and node.name == "start")
                    calls = [node for node in ast.walk(start) if isinstance(node, ast.Call)]
                    self.assertEqual(sum(isinstance(node.func, ast.Attribute) and node.func.attr == "Popen" for node in calls), 1)
                    self.assertEqual(sum(isinstance(node.func, ast.Name) and node.func.id == "record_owned" for node in calls), 1)
                    self.assertEqual(sum(isinstance(node.func, ast.Name) and node.func.id == "complete_port" for node in calls), 1)
                    self.assertIn("deadline=time.monotonic()+45", ast.get_source_segment((output / "environments.py").read_text(), start))
                    manifest = json.loads((output / "source-manifest.json").read_text())
                    receipt = json.loads((output / "portable-build.json").read_text())
                    for helper, source in (("portable_browser_lifecycle.py", "browser_lifecycle.py"), ("portable_browser_startup.py", "local_browser_startup.py")):
                        self.assertEqual(B.digest(output / helper), B.digest(ROOT / "reproduce" / source))
                        self.assertEqual(manifest["source_hashes"][str(output / helper)], B.digest(output / helper))
                    changes = [row for row in receipt["adaptations"] if row["file"] == "environments.py"]
                    for previous, following in zip(changes, changes[1:]):
                        self.assertEqual(previous["adapted_sha256"], following["original_sha256"])
                    self.assertEqual(changes[-1]["adapted_sha256"], B.digest(output / "environments.py"))
        self.assertEqual(archived, {name: B.digest(ROOT / name) for name in archived})


if __name__ == "__main__":
    unittest.main()
