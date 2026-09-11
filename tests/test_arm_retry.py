"""Protect original outcomes and prevent premature/duplicate retry execution."""
import importlib.util
import json
import os
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
spec = importlib.util.spec_from_file_location("arm_retry", Path(__file__).parents[1] / "scripts/run_arm_retry.py")
retry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(retry)


def fixture(root):
    source, output = root / "original/full", root / "separate-retry"
    tasks = root / "tasks.jsonl"
    tasks.write_text("{}\n")
    rows = {"baseline": [("a", False, None), ("b", True, 1), ("c", True, 0)],
            "scalar": [("a", True, 1), ("b", False, None), ("c", True, 0)],
            "selection": [("a", True, 1), ("b", True, 1), ("c", True, 0)]}
    manifest = dict(actor="/pinned/actor", task_file=str(tasks), task_file_sha256=retry.digest(tasks),
                    task_ids=["a", "b", "c"], seed=42, sampling={"temperature": .7, "top_p": .9, "max_new_tokens": 1024},
                    max_steps=30, history="full", context_num_screenshots=1, browser_format="browser_env",
                    judge="o4-mini", judge_protocol="online_mind2web/AgentTrek", task_timeout=1800)
    for mode, data in rows.items():
        directory = source / mode
        (directory / "results").mkdir(parents=True)
        (directory / "manifest.json").write_text(json.dumps(dict(manifest, mode=mode)))
        (directory / "summary.json").write_text(json.dumps({"scheduled": 3, "attempted": 3}))
        for task, valid, reward in data:
            (directory / "results" / (task + ".json")).write_text(json.dumps({"task_id": task, "valid": valid, "reward": reward}))
    queue = dict(authorized=True, source_full=str(source), output=str(output), modes=["baseline", "scalar"],
                 attempts_per_arm_per_task=1, expected_source_tasks=3, expected_retry_tasks=2,
                 task_ids=["a", "b"], task_indices=[0, 1], parallel_by_mode={"baseline": 8, "scalar": 16},
                 allocation="42", slurm_step="0", gpu_uuid="GPU-assigned", resident_actor_processes=[])
    queue["original_result_sha256"] = {mode: {p.name: retry.digest(p) for p in (source / mode / "results").glob("*.json")}
                                       for mode in ["baseline", "scalar"]}
    return queue


class RetryTests(unittest.TestCase):
    def test_exact_union_repeats_valid_counterparts_without_touching_original(self):
        with tempfile.TemporaryDirectory() as directory:
            q = fixture(Path(directory))
            source, output, manifests, rows = retry.validate_source(q)
            self.assertTrue(rows["baseline"]["b"]["valid"])
            self.assertTrue(rows["scalar"]["a"]["valid"])
            self.assertFalse(output.exists())
            argv = retry.build_eval_command(q, "scalar", manifests["scalar"])
            for arg, value in [("--task-indices", "0,1"), ("--seed", "42"), ("--parallel", "16"),
                               ("--max-new-tokens", "1024"), ("--task-timeout", "1800")]:
                self.assertEqual(argv[argv.index(arg) + 1], value)

    def test_incomplete_selection_prevents_any_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            q = fixture(Path(directory))
            f = Path(q["source_full"]) / "selection/summary.json"
            f.write_text(json.dumps({"scheduled": 3, "attempted": 2}))
            with self.assertRaisesRegex(ValueError, "selection.*incomplete"):
                retry.validate_source(q)
            self.assertFalse(Path(q["output"]).exists())

    def test_original_outcome_change_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            q = fixture(Path(directory))
            f = Path(q["source_full"]) / "baseline/results/c.json"
            f.write_text(json.dumps({"task_id": "c", "valid": True, "reward": 1}))
            with self.assertRaisesRegex(ValueError, "changed since"):
                retry.validate_source(q)

    def test_wrong_cohort_and_wrong_index_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            q = fixture(Path(directory))
            wrong = dict(q, task_ids=["a", "c"])
            with self.assertRaisesRegex(ValueError, "unavailable-task union"):
                retry.validate_source(wrong)
            with self.assertRaisesRegex(ValueError, "indices"):
                retry.validate_source(dict(q, task_indices=[0, 2]))

    def test_original_output_and_descendants_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            q = fixture(Path(directory))
            source = Path(q["source_full"])
            for path in [source, source / "retry", source.parent, source.parent / "retry"]:
                with self.assertRaisesRegex(ValueError, "separate"):
                    retry.validate_layout(dict(q, output=str(path)))

    def test_changed_protocol_or_dataset_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            q = fixture(Path(directory))
            f = Path(q["source_full"]) / "scalar/manifest.json"
            m = json.loads(f.read_text()); m["max_steps"] = 60; f.write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError, "different max_steps"):
                retry.validate_source(q)
            m["max_steps"] = 30; f.write_text(json.dumps(m))
            Path(m["task_file"]).write_text("changed\n")
            with self.assertRaisesRegex(ValueError, "dataset changed"):
                retry.validate_source(q)

    def test_missing_authorization_and_repeated_attempts_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            q = fixture(Path(directory))
            with self.assertRaisesRegex(ValueError, "authorization"):
                retry.validate_layout(dict(q, authorized=False))
            with self.assertRaisesRegex(ValueError, "one fixed attempt"):
                retry.validate_layout(dict(q, attempts_per_arm_per_task=2))
            output = Path(q["output"]); output.mkdir()
            (output / "retry-status.json").write_text(json.dumps({"phase": "running_scalar"}))
            queue = Path(directory) / "queue.json"; queue.write_text(json.dumps(q))
            with patch.object(retry, "validate_allocation") as gpu:
                with self.assertRaisesRegex(ValueError, "already started"):
                    retry.execute(queue)
                gpu.assert_not_called()
            (output / "retry-status.json").write_text(json.dumps({"phase": "complete"}))
            with patch.object(retry, "validate_allocation") as gpu:
                retry.execute(queue)
                gpu.assert_not_called()

    def test_retry_report_separates_denominators_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            q = fixture(root)
            source, output, _, original = retry.validate_source(q)
            before = {str(p): retry.digest(p) for p in source.rglob("*.json")}
            for mode, rows in {"baseline": [("a", True, 1), ("b", False, None)],
                               "scalar": [("a", True, 1), ("b", True, 0)]}.items():
                (output / mode / "results").mkdir(parents=True)
                for task, valid, reward in rows:
                    (output / mode / "results" / (task + ".json")).write_text(json.dumps(
                        {"task_id": task, "valid": valid, "reward": reward}))
            docs = root / "repo/openwebrl/docs"
            docs.mkdir(parents=True)
            with patch.object(retry, "REPO", root / "repo"):
                retry.summarize_retry(q, original)
            report = json.loads((output / "comparison.json").read_text())
            self.assertEqual(report["summaries"]["baseline"]["success_rate_all_scheduled"], .5)
            self.assertEqual(report["summaries"]["baseline"]["success_rate_valid"], 1.)
            self.assertEqual(report["recovery"]["baseline"]["previously_valid_now_unavailable"], 1)
            self.assertEqual(report["paired"]["common_valid_tasks"], 1)
            self.assertEqual(before, {str(p): retry.digest(p) for p in source.rglob("*.json")})
            self.assertIn("not a replacement", (docs / "ARM_INFERENCE.md").read_text())
            self.assertFalse((docs / "ARM_INFERENCE_RETRY_RESULTS.md").exists())

    def test_project_quota_does_not_lose_completed_retry_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            q = fixture(root)
            _, output, _, original = retry.validate_source(q)
            for mode in ["baseline", "scalar"]:
                (output / mode / "results").mkdir(parents=True)
                for task in q["task_ids"]:
                    (output / mode / "results" / (task + ".json")).write_text(json.dumps(
                        {"task_id": task, "valid": True, "reward": 1}))
            document = root / "repo/openwebrl/docs/ARM_INFERENCE.md"
            document.parent.mkdir(parents=True)
            document.write_text("Original pending document")
            original_write = Path.write_text
            def write(path, text, *args, **kwargs):
                if path == document.with_suffix(".md.tmp"):
                    raise OSError("Disk quota exceeded")
                return original_write(path, text, *args, **kwargs)
            with patch.object(retry, "REPO", root / "repo"), patch.object(Path, "write_text", write):
                report = retry.summarize_retry(q, original)
            self.assertIn("Disk quota", report["project_markdown_write_error"])
            self.assertTrue((output / "RETRY_RESULTS.md").exists())
            self.assertEqual(document.read_text(), "Original pending document")
            self.assertEqual(json.loads((output / "comparison.json").read_text())["summaries"]["scalar"]["successes"], 2)

    def test_original_report_finishes_before_queued_retry_and_survives_retry_failure(self):
        import subprocess
        from types import SimpleNamespace
        report_spec = importlib.util.spec_from_file_location("arm_report", Path(__file__).parents[1] / "scripts/summarize_arm_reproduction.py")
        report = importlib.util.module_from_spec(report_spec)
        report_spec.loader.exec_module(report)
        with tempfile.TemporaryDirectory() as directory:
            q = fixture(Path(directory))
            source = Path(q["source_full"])
            (source / "queued-retry.json").write_text(json.dumps(q))
            def child(*args, **kwargs):
                self.assertTrue((source / "comparison.json").exists())
                self.assertEqual(args[0][-2:], ["--queue", str(source / "queued-retry.json")])
                return SimpleNamespace(returncode=7)
            with patch("sys.argv", ["report", str(source)]), patch.object(subprocess, "run", side_effect=child) as run, patch("builtins.print"):
                report.main()
            run.assert_called_once()
            self.assertTrue((source / "comparison.json").exists())

    def test_held_queue_completes_original_report_without_launching_retry(self):
        import subprocess
        report_spec = importlib.util.spec_from_file_location("arm_report", Path(__file__).parents[1] / "scripts/summarize_arm_reproduction.py")
        report = importlib.util.module_from_spec(report_spec)
        report_spec.loader.exec_module(report)
        with tempfile.TemporaryDirectory() as directory:
            q = fixture(Path(directory))
            q["authorized"] = False
            source = Path(q["source_full"])
            queue = source / "queued-retry.json"
            queue.write_text(json.dumps(q))
            original_queue = queue.read_bytes()
            with patch("sys.argv", ["report", str(source)]), patch.object(subprocess, "run") as run, patch("builtins.print"):
                report.main()
            run.assert_not_called()
            self.assertTrue((source / "comparison.json").exists())
            self.assertFalse(Path(q["output"]).exists())
            self.assertEqual(queue.read_bytes(), original_queue)

    def test_wrong_step_fails_before_gpu_inspection(self):
        with tempfile.TemporaryDirectory() as directory:
            q = fixture(Path(directory))
            with patch.dict(os.environ, {"SLURM_JOB_ID": "42", "SLURM_STEP_ID": "other"}), patch.object(retry, "command") as command:
                with self.assertRaisesRegex(ValueError, "Slurm step"):
                    retry.validate_allocation(q)
                command.assert_not_called()

    def test_other_gpu_or_extra_processes_are_rejected(self):
        q = {"allocation": "42", "slurm_step": "0", "gpu_uuid": "GPU-assigned", "resident_actor_processes": []}
        with patch.dict(os.environ, {"SLURM_JOB_ID": "42", "SLURM_STEP_ID": "0"}), \
             patch.object(Path, "read_text", return_value="0::/job_42/step_0/user/task_0"), \
             patch.object(retry.getpass, "getuser", return_value="owner"):
            with patch.object(retry, "command", side_effect=["UserId=owner(1) JobState=RUNNING", "GPU-other"]):
                with self.assertRaisesRegex(ValueError, "Assigned GPU"):
                    retry.validate_allocation(q)
            with patch.object(retry, "command", side_effect=["UserId=owner(1) JobState=RUNNING", "GPU-assigned", "123"]):
                with self.assertRaisesRegex(ValueError, "GPU processes differ"):
                    retry.validate_allocation(q)


if __name__ == "__main__":
    unittest.main()
