#!/usr/bin/env python3
"""Run fixed-subset C2 checkpoint evaluations sequentially in one allocation."""

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from resume_arm_c2_training import allocation_deadline, parse_job_record, write_json


REPO = Path(__file__).resolve().parents[1]
ARM_PYTHON = Path("/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/venv/bin/python")


def subset_final(full, sample, target):
    rows = []
    for task_id in sample["task_ids"]:
        path = full / "results" / (hashlib.sha256(task_id.encode()).hexdigest()[:20] + ".json")
        rows.append(json.loads(path.read_text()))
    valid = [row for row in rows if row.get("valid")]
    successes = sum(row.get("reward") == 1 for row in valid)
    value = {
        "source": str(full),
        "checkpoint": "student/epoch-2",
        "updates": 1050,
        "sample_seed": sample["seed"],
        "sample_count": sample["count"],
        "attempted": len(rows),
        "valid": len(valid),
        "unavailable": len(rows) - len(valid),
        "successes": successes,
        "success_rate_all_scheduled": successes / len(rows),
        "success_rate_valid": successes / len(valid) if valid else None,
        "derived_without_duplicate_rollouts": True,
    }
    target.mkdir(parents=True, exist_ok=True)
    write_json(target / "summary.json", value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--checkpoint", action="append", default=[], metavar="LABEL=PATH",
                        help="Evaluate this explicit checkpoint series instead of waiting for epoch 2")
    args = parser.parse_args()
    if os.getenv("SLURM_JOB_ID") != args.job_id or f"/job_{args.job_id}/" not in Path("/proc/self/cgroup").read_text():
        raise ValueError("Scaling queue must run inside the authorized allocation")
    record = parse_job_record(subprocess.check_output(
        ["scontrol", "show", "job", args.job_id, "-o"], text=True, timeout=20))
    deadline = allocation_deadline(record, margin_minutes=5)
    root = args.run_root.resolve()
    scaling = root / "evaluation" / "checkpoint-scaling-100"
    sample_path = scaling / "sample.json"
    sample = json.loads(sample_path.read_text())
    full = root / "evaluation" / "c2-student-online-mind2web-300"
    status = scaling / "queue-status.json"
    if args.checkpoint:
        entries = []
        for spec in args.checkpoint:
            label, separator, value = spec.partition("=")
            if not separator or not label:
                parser.error("--checkpoint must be LABEL=PATH")
            entries.append((label, Path(value).resolve()))
        for label, checkpoint in entries:
            if datetime.now(timezone.utc) >= deadline - timedelta(minutes=10):
                write_json(status, {"phase": "paused_before_next_checkpoint", "allocation": args.job_id,
                                    "updated_utc": datetime.now(timezone.utc).isoformat(), "next": label})
                return
            write_json(status, {"phase": "evaluating_checkpoint", "allocation": args.job_id,
                                "updated_utc": datetime.now(timezone.utc).isoformat(), "label": label,
                                "checkpoint": str(checkpoint)})
            command = [str(ARM_PYTHON), str(REPO / "scripts/run_arm_checkpoint_eval.py"),
                       "--job-id", args.job_id, "--run-root", str(root),
                       "--checkpoint", str(checkpoint), "--label", label,
                       "--sample", str(sample_path), "--actor-port", "19110",
                       "--browser-port-start", "19200", "--browser-port-end", "19399"]
            code = subprocess.call(command, cwd=REPO)
            checkpoint_status = json.loads((scaling / label / "status.json").read_text())
            if code or checkpoint_status["phase"] != "complete":
                write_json(status, {"phase": "checkpoint_paused_or_failed", "allocation": args.job_id,
                                    "updated_utc": datetime.now(timezone.utc).isoformat(), "label": label,
                                    "returncode": code, "checkpoint_status": checkpoint_status})
                return
        write_json(status, {"phase": "complete", "allocation": args.job_id,
                            "updated_utc": datetime.now(timezone.utc).isoformat(),
                            "labels": [label for label, _ in entries]})
        return
    write_json(status, {"phase": "waiting_for_primary_epoch_2_evaluation", "allocation": args.job_id,
                        "updated_utc": datetime.now(timezone.utc).isoformat(), "deadline_utc": deadline.isoformat()})
    while not (full / "summary.json").exists():
        if datetime.now(timezone.utc) >= deadline - timedelta(minutes=5):
            write_json(status, {"phase": "primary_evaluation_incomplete_at_deadline", "allocation": args.job_id,
                                "updated_utc": datetime.now(timezone.utc).isoformat()})
            return
        time.sleep(15)
    subset_final(full, sample, scaling / "update-001050-from-full")
    while subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True, timeout=20
    ).strip():
        if datetime.now(timezone.utc) >= deadline - timedelta(minutes=5):
            return
        time.sleep(5)

    for update in (100, 500, 900):
        if datetime.now(timezone.utc) >= deadline - timedelta(minutes=10):
            write_json(status, {"phase": "paused_before_next_checkpoint", "allocation": args.job_id,
                                "updated_utc": datetime.now(timezone.utc).isoformat(), "next_update": update})
            return
        label = f"update-{update:06d}"
        write_json(status, {"phase": "evaluating_checkpoint", "allocation": args.job_id,
                            "updated_utc": datetime.now(timezone.utc).isoformat(), "update": update})
        command = [str(ARM_PYTHON), str(REPO / "scripts/run_arm_checkpoint_eval.py"),
                   "--job-id", args.job_id, "--run-root", str(root),
                   "--checkpoint", str(root / "student" / label), "--label", label,
                   "--sample", str(sample_path), "--actor-port", "19110",
                   "--browser-port-start", "19200", "--browser-port-end", "19399"]
        code = subprocess.call(command, cwd=REPO)
        checkpoint_status = json.loads((scaling / label / "status.json").read_text())
        if code or checkpoint_status["phase"] != "complete":
            write_json(status, {"phase": "checkpoint_paused_or_failed", "allocation": args.job_id,
                                "updated_utc": datetime.now(timezone.utc).isoformat(), "update": update,
                                "returncode": code, "checkpoint_status": checkpoint_status})
            return
    write_json(status, {"phase": "complete", "allocation": args.job_id,
                        "updated_utc": datetime.now(timezone.utc).isoformat(),
                        "updates": [100, 500, 900, 1050]})


if __name__ == "__main__":
    main()
