import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.resume_arm_c2_training import (
    allocation_deadline,
    checked_checkpoint,
    parse_job_record,
)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class ResumeArmC2TrainingTest(unittest.TestCase):
    def make_run(self, root):
        run = Path(root)
        dataset = run / "training.jsonl"
        dataset.write_text('{"example": 1}\n')
        digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
        dump(
            run / "dataset-audit.json",
            {"complete_collection": True, "dataset": str(dataset), "dataset_sha256": digest},
        )
        checkpoint = run / "student" / "paused-000007-1"
        progress = {"epoch": 1, "next_position": 32, "updates": 7, "dataset_sha256": digest}
        dump(checkpoint / "progress.json", progress)
        for name in ("adapter_config.json", "adapter_model.safetensors", "optimizer.pt"):
            (checkpoint / name).write_bytes(b"present")
        dump(run / "student" / "latest-checkpoint.json", {"path": str(checkpoint), **progress})
        return run, checkpoint, progress

    def test_deadline_uses_five_minute_margin(self):
        record = parse_job_record(
            "JobId=285131 JobState=RUNNING EndTime=2026-09-09T15:29:33-07:00 NodeList=g022"
        )
        self.assertEqual(allocation_deadline(record).isoformat(), "2026-09-09T22:24:33+00:00")

    def test_accepts_matching_latest_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            run, checkpoint, progress = self.make_run(temporary)
            actual, state, audit = checked_checkpoint(run)
            self.assertEqual(actual, checkpoint.resolve())
            self.assertEqual(state, progress)
            self.assertEqual(state["dataset_sha256"], audit["dataset_sha256"])

    def test_rejects_checkpoint_outside_student_lineage(self):
        with tempfile.TemporaryDirectory() as temporary:
            run, _, progress = self.make_run(temporary)
            foreign = run / "foreign"
            foreign.mkdir()
            dump(run / "student" / "latest-checkpoint.json", {"path": str(foreign), **progress})
            with self.assertRaisesRegex(ValueError, "outside"):
                checked_checkpoint(run)

    def test_rejects_changed_dataset(self):
        with tempfile.TemporaryDirectory() as temporary:
            run, _, _ = self.make_run(temporary)
            (run / "training.jsonl").write_text("changed\n")
            with self.assertRaisesRegex(ValueError, "audit"):
                checked_checkpoint(run)


if __name__ == "__main__":
    unittest.main()
