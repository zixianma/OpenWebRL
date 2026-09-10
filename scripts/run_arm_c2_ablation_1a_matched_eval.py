#!/usr/bin/env python3
"""Start the update-250 matched-exposure eval beside the 1A endpoint eval."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request

from resume_arm_c2_training import allocation_deadline, parse_job_record, write_json


REPO = Path(__file__).resolve().parents[1]
TRAINING_PYTHON = Path(
    "/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/venv/bin/python"
)


def endpoint_ready(port: int) -> bool:
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/health_generate", timeout=5
        ) as response:
            response.read()
        return True
    except OSError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--run-root", required=True, type=Path)
    args = parser.parse_args()

    if os.getenv("SLURM_JOB_ID") != args.job_id:
        raise ValueError("Matched evaluation must run inside the authorized allocation")
    record = parse_job_record(
        subprocess.check_output(
            ["scontrol", "show", "job", args.job_id, "-o"], text=True, timeout=20
        )
    )
    deadline = allocation_deadline(record, margin_minutes=5)
    root = args.run_root.resolve()
    status_path = root / "evaluation/checkpoint-scaling-100/update-000250/waiter-status.json"
    checkpoint = root / "student/update-000250"
    complete = root / "student/complete.json"

    write_json(
        status_path,
        {
            "phase": "waiting_for_training_and_endpoint_server",
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "allocation": args.job_id,
            "checkpoint": str(checkpoint),
            "exposure_match": {
                "new_update": 250,
                "new_effective_batch": 32,
                "new_examples": 8000,
                "baseline_update": 500,
                "baseline_effective_batch": 16,
                "baseline_examples": 8000,
            },
        },
    )
    while datetime.now(timezone.utc) < deadline:
        root_status_path = root / "status.json"
        root_status = json.loads(root_status_path.read_text()) if root_status_path.exists() else {}
        if root_status.get("phase") == "training_paused_or_failed":
            write_json(
                status_path,
                {
                    "phase": "not_started_training_incomplete",
                    "updated_utc": datetime.now(timezone.utc).isoformat(),
                    "allocation": args.job_id,
                },
            )
            return
        if complete.exists() and (checkpoint / "progress.json").exists() and endpoint_ready(19110):
            break
        time.sleep(15)
    else:
        raise TimeoutError("Training and endpoint evaluator were not ready before cutoff")

    command = [
        str(TRAINING_PYTHON),
        str(REPO / "scripts/run_arm_checkpoint_eval.py"),
        "--job-id",
        args.job_id,
        "--run-root",
        str(root),
        "--checkpoint",
        str(checkpoint),
        "--label",
        "update-000250",
        "--sample",
        str(root / "evaluation/checkpoint-scaling-100/sample.json"),
        "--actor-port",
        "19111",
        "--browser-port-start",
        "19400",
        "--browser-port-end",
        "19599",
        "--allow-shared-gpu",
        "--mem-fraction-static",
        "0.3",
        "--parallel",
        "6",
    ]
    write_json(
        status_path,
        {
            "phase": "running",
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "allocation": args.job_id,
            "command": command,
        },
    )
    with (root / "matched-eval-controller.log").open("a") as log:
        code = subprocess.call(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
    write_json(
        status_path,
        {
            "phase": "complete" if code == 0 else "paused_or_failed",
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "allocation": args.job_id,
            "returncode": code,
        },
    )
    raise SystemExit(code)


if __name__ == "__main__":
    main()
