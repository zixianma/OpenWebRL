#!/usr/bin/env python3
"""Wait for preference training and use remaining job 286384 time for fixed-100 eval."""

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

from resume_arm_c2_training import allocation_deadline, parse_job_record, write_json


ROOT = Path("/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/runs/arm-preference-calibrated-286384-r4")
REPO = Path("/gpfs/projects/krishna/zixianma/OpenWebRL")
PYTHON = Path("/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/venv/bin/python")


def main():
    record = parse_job_record(subprocess.check_output(["scontrol", "show", "job", "286384", "-o"], text=True))
    deadline = allocation_deadline(record, margin_minutes=5)
    endpoint = ROOT / "student/endpoint"
    while not (endpoint / "progress.json").exists():
        if datetime.now(timezone.utc) >= deadline:
            raise TimeoutError("Preference endpoint was not ready before allocation end")
        time.sleep(20)
    pair_audit = json.loads((ROOT / "pair-audit.json").read_text())
    progress = json.loads((endpoint / "progress.json").read_text())
    progress["dataset_sha256"] = pair_audit["manifest_sha256"]
    write_json(endpoint / "progress.json", progress)
    audit = json.loads((ROOT / "dataset-audit.json").read_text())
    audit["dataset_sha256"] = pair_audit["manifest_sha256"]
    write_json(ROOT / "dataset-audit.json", audit)
    sample = ROOT / "evaluation/checkpoint-scaling-100/sample.json"
    sample.parent.mkdir(parents=True, exist_ok=True)
    sample.write_text((REPO / "openwebrl/docs/arm_c2_scaling_100.json").read_text())
    subprocess.run([str(PYTHON), str(REPO / "scripts/run_arm_checkpoint_eval.py"),
        "--job-id", "286384", "--run-root", str(ROOT), "--checkpoint", str(endpoint),
        "--label", "endpoint-000131", "--sample", str(sample), "--actor-port", "19110",
        "--browser-port-start", "19200", "--browser-port-end", "19399"], cwd=REPO, check=True)


if __name__ == "__main__":
    main()
