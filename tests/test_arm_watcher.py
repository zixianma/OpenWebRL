"""Safety and promotion checks for the allocation-bound ARM watcher."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("arm_watcher", Path(__file__).parents[1] / "scripts/watch_arm_reproduction.py")
watcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watcher)


def write_smoke(root):
    for mode in ("baseline", "scalar", "selection"):
        target = root / mode
        (target / "selections").mkdir(parents=True)
        (target / "summary.json").write_text(json.dumps({"attempted": 3, "scheduled": 3, "valid": 2, "successes": 0}))
        trace = {"candidates": [{}] * (1 if mode == "baseline" else 5),
                 "selected_index": 0, "scores": [0, 1, 2, 3, 4], "fallback": None}
        (target / "selections/task.jsonl").write_text(json.dumps(trace) + "\n")


class WatcherTests(unittest.TestCase):
    def test_wait_requires_consecutive_empty_checks_and_keeps_busy_workloads(self):
        args = SimpleNamespace(job_id="42", poll_seconds=30, gpu_count=2)
        free = [{"processes": [], "memory_MiB": 0}] * 2
        busy = [{"processes": ["123"], "memory_MiB": 80000}] * 2
        with patch.object(watcher, "allocation"), patch.object(watcher, "record"), \
             patch.object(watcher, "gpu_status", side_effect=[busy, free, busy, free, free]) as query, \
             patch.object(watcher.time, "sleep") as sleep:
            watcher.wait_free(args)
        self.assertEqual(query.call_count, 5)
        self.assertEqual(sleep.call_count, 4)

    def test_failed_gpu_query_never_allows_launch(self):
        args = SimpleNamespace(job_id="42", poll_seconds=30, gpu_count=2)
        with patch.object(watcher, "allocation"), patch.object(watcher, "record"), \
             patch.object(watcher, "gpu_status", side_effect=RuntimeError("nvidia-smi failed")):
            with self.assertRaisesRegex(RuntimeError, "nvidia-smi failed"):
                watcher.wait_free(args)

    def test_single_gpu_allocation_and_wrong_device_count(self):
        with patch.object(watcher, "command", side_effect=["0, GPU-dedicated, 0", ""]):
            devices = watcher.gpu_status(1)
        self.assertEqual(devices, [{"index": 0, "uuid": "GPU-dedicated", "memory_MiB": 0, "processes": []}])
        with patch.object(watcher, "command", return_value="0, GPU-dedicated, 0"):
            with self.assertRaisesRegex(RuntimeError, "Expected 2"):
                watcher.gpu_status(2)

    def test_supervisor_cannot_depend_on_other_training_allocation(self):
        with patch.object(Path, "read_text", return_value="0::/slurm/job_100/step_interactive/user/task_0"):
            with self.assertRaisesRegex(RuntimeError, "dedicated allocation"):
                watcher.validate_supervisor_allocation("200")
        with patch.object(Path, "read_text", return_value="0::/slurm/job_200/step_extern/user/task_0"):
            watcher.validate_supervisor_allocation("200")
        with patch.object(Path, "read_text", return_value="0::/user.slice/session.scope"):
            watcher.validate_supervisor_allocation("200")

    def test_smoke_gate_requires_functional_inference_but_not_task_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_smoke(root)
            self.assertEqual(watcher.smoke_gate(root)["selection"]["valid_tasks"], 2)
            path = root / "selection/selections/task.jsonl"
            trace = json.loads(path.read_text())
            trace["fallback"] = "Malformed selector output"
            path.write_text(json.dumps(trace) + "\n")
            with self.assertRaisesRegex(ValueError, "malformed"):
                watcher.smoke_gate(root)

    def test_all_invalid_smoke_stops_promotion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_smoke(root)
            (root / "baseline/summary.json").write_text(json.dumps({"attempted": 3, "scheduled": 3, "valid": 0}))
            with self.assertRaisesRegex(ValueError, "valid judged"):
                watcher.smoke_gate(root)


if __name__ == "__main__":
    unittest.main()
