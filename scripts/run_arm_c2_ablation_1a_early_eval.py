#!/usr/bin/env python3
"""Use post-primary idle capacity for the available early 1A checkpoint."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

from resume_arm_c2_training import allocation_deadline, parse_job_record, write_json


REPO = Path(__file__).resolve().parents[1]
TRAINING_PYTHON = Path(
    "/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/venv/bin/python"
)
TERMINAL_PHASES = {"complete", "paused", "evaluation_paused_or_failed"}


def read_status(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--run-root", required=True, type=Path)
    args = parser.parse_args()
    if os.getenv("SLURM_JOB_ID") != args.job_id:
        raise ValueError("Early evaluation must run inside the authorized allocation")

    record = parse_job_record(
        subprocess.check_output(
            ["scontrol", "show", "job", args.job_id, "-o"], text=True, timeout=20
        )
    )
    deadline = allocation_deadline(record, margin_minutes=5)
    root = args.run_root.resolve()
    work = root / "evaluation/checkpoint-scaling-100"
    waiter_status = work / "update-000066/waiter-status.json"
    endpoint_status = work / "endpoint-000263/status.json"
    matched_status = work / "update-000250/status.json"
    write_json(
        waiter_status,
        {
            "phase": "waiting_for_a_primary_server_to_finish",
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "allocation": args.job_id,
            "comparison_caveat": "2112 examples versus 1600 at original C2 update 100",
        },
    )

    while datetime.now(timezone.utc) < deadline:
        endpoint = read_status(endpoint_status).get("phase")
        matched = read_status(matched_status).get("phase")
        if endpoint in TERMINAL_PHASES or matched in TERMINAL_PHASES:
            break
        time.sleep(15)
    else:
        raise TimeoutError("Neither primary evaluator finished before cutoff")

    # The completed evaluator writes its terminal status immediately before
    # stopping its actor server. Give that cleanup a short head start.
    time.sleep(15)
    command = [
        str(TRAINING_PYTHON),
        str(REPO / "scripts/run_arm_checkpoint_eval.py"),
        "--job-id",
        args.job_id,
        "--run-root",
        str(root),
        "--checkpoint",
        str(root / "student/update-000066"),
        "--label",
        "update-000066",
        "--sample",
        str(work / "sample.json"),
        "--actor-port",
        "19112",
        "--browser-port-start",
        "19600",
        "--browser-port-end",
        "19799",
        "--allow-shared-gpu",
        "--mem-fraction-static",
        "0.35",
        "--parallel",
        "6",
    ]
    write_json(
        waiter_status,
        {
            "phase": "running",
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "allocation": args.job_id,
            "command": command,
        },
    )
    with (root / "early-eval-controller.log").open("a") as log:
        code = subprocess.call(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
    write_json(
        waiter_status,
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
