#!/usr/bin/env python3
"""Safely resume an audited C2 LoRA student inside an existing Slurm allocation.

This launcher never submits a job. Run it through ``srun`` in compute that has
already been assigned. It derives the visible GPU and allocation deadline,
validates the durable checkpoint lineage, freezes a per-allocation config, and
then invokes the pinned ARM-branch trainer.
"""

import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import getpass
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def parse_job_record(text):
    return dict(field.split("=", 1) for field in text.strip().split() if "=" in field)


def allocation_deadline(record, margin_minutes=5):
    value = record.get("EndTime")
    if not value or value in {"Unknown", "N/A"}:
        raise ValueError("The allocation has no finite EndTime")
    local = datetime.fromisoformat(value)
    if local.tzinfo is None:
        local = local.astimezone()
    return (local.astimezone(timezone.utc) - timedelta(minutes=margin_minutes))


def checked_checkpoint(run_root):
    student = run_root / "student"
    if (student / "complete.json").exists():
        raise ValueError("C2 training is already complete; do not resume it")
    pointer_path = student / "latest-checkpoint.json"
    if not pointer_path.exists():
        raise ValueError("Missing student/latest-checkpoint.json")
    pointer = json.loads(pointer_path.read_text())
    checkpoint = Path(pointer["path"]).resolve()
    if checkpoint.parent != student.resolve() or not checkpoint.is_dir():
        raise ValueError("Latest checkpoint is outside this run's student directory")
    required = {"adapter_config.json", "adapter_model.safetensors", "optimizer.pt", "progress.json"}
    missing = sorted(name for name in required if not (checkpoint / name).is_file())
    if missing:
        raise ValueError(f"Incomplete resume checkpoint: {missing}")
    progress = json.loads((checkpoint / "progress.json").read_text())
    for field in ("epoch", "next_position", "updates", "dataset_sha256"):
        if pointer.get(field) != progress.get(field):
            raise ValueError(f"Checkpoint pointer/progress mismatch: {field}")
    audit = json.loads((run_root / "dataset-audit.json").read_text())
    dataset = Path(audit["dataset"])
    if not audit.get("complete_collection") or sha256(dataset) != audit["dataset_sha256"]:
        raise ValueError("The complete C2 dataset audit no longer matches its dataset")
    if progress["dataset_sha256"] != audit["dataset_sha256"]:
        raise ValueError("Checkpoint and dataset have different lineages")
    return checkpoint, progress, audit


def pinned_trainer(config):
    checkout = config.get("training_checkout")
    if not checkout:
        raise ValueError("A pinned ARM training checkout is required")
    root = Path(checkout["root"]).resolve()
    entrypoint = (root / checkout["entrypoint"]).resolve()
    if root not in entrypoint.parents:
        raise ValueError("Trainer entrypoint escapes the pinned checkout")
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, timeout=20
    ).strip()
    if revision != checkout["commit"] or sha256(entrypoint) != checkout["sha256"]:
        raise ValueError("Pinned ARM trainer revision or checksum changed")
    python = Path(config["training_python"]).resolve()
    if not python.is_file():
        raise ValueError(f"Training Python is missing: {python}")
    return python, entrypoint, root


def runtime_allocation(job_id):
    if os.getenv("SLURM_JOB_ID") != job_id:
        raise ValueError("Run this launcher inside an srun step for the authorized job")
    if f"/job_{job_id}/" not in Path("/proc/self/cgroup").read_text():
        raise ValueError("Process cgroup does not belong to the requested allocation")
    raw = subprocess.check_output(
        ["scontrol", "show", "job", job_id, "-o"], text=True, timeout=20
    ).strip()
    record = parse_job_record(raw)
    if record.get("JobState") != "RUNNING":
        raise ValueError("Allocation is not running")
    if not record.get("UserId", "").startswith(getpass.getuser() + "("):
        raise ValueError("Allocation belongs to another user")
    uuids = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        text=True,
        timeout=20,
    ).strip().splitlines()
    if len(uuids) != 1:
        raise ValueError(f"C2 training requires exactly one visible GPU, found {len(uuids)}")
    pids = subprocess.check_output(
        ["nvidia-smi", "--id=" + uuids[0], "--query-compute-apps=pid", "--format=csv,noheader"],
        text=True,
        timeout=20,
    ).strip()
    if pids:
        raise ValueError("The assigned GPU already has a compute process")
    return record, uuids[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True, help="Already-running Slurm allocation")
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--launch", action="store_true", help="Start training after validation")
    parser.add_argument("--margin-minutes", type=int, default=5)
    args = parser.parse_args()
    if args.margin_minutes < 3:
        parser.error("--margin-minutes must be at least 3")

    run_root = args.run_root.resolve()
    record, gpu_uuid = runtime_allocation(args.job_id)
    stop = allocation_deadline(record, args.margin_minutes)
    if datetime.now(timezone.utc) >= stop - timedelta(minutes=5):
        raise ValueError("Less than five minutes remain before the resume deadline")
    checkpoint, progress, audit = checked_checkpoint(run_root)
    base_config = json.loads((run_root / "frozen-config.json").read_text())
    python, entrypoint, checkout_root = pinned_trainer(base_config)

    session = run_root / "execution-sessions" / args.job_id
    config_path = session / "training-resume-config.json"
    config = dict(base_config)
    config.update(
        allocation=args.job_id,
        gpu_uuid=gpu_uuid,
        node=record.get("NodeList", os.getenv("SLURMD_NODENAME")),
        stop_utc=stop.isoformat(),
        resume_authorization={
            "recorded_utc": datetime.now(timezone.utc).isoformat(),
            "source": "existing user-assigned allocation; this launcher never submits jobs",
            "allocation_end": record["EndTime"],
            "resume_checkpoint": {"path": str(checkpoint), **progress},
        },
    )
    receipt = {
        "phase": "validated" if not args.launch else "launching",
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "allocation": args.job_id,
        "node": config["node"],
        "gpu_uuid": gpu_uuid,
        "stop_utc": config["stop_utc"],
        "run_root": str(run_root),
        "checkpoint": str(checkpoint),
        "checkpoint_progress": progress,
        "dataset_sha256": audit["dataset_sha256"],
        "trainer": str(entrypoint),
        "trainer_sha256": sha256(entrypoint),
    }
    write_json(config_path, config)
    write_json(session / "training-resume-receipt.json", receipt)
    print(json.dumps(receipt, indent=2), flush=True)
    if not args.launch:
        return

    lock_stream = (run_root / "student" / "training-resume.lock").open("a")
    fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    log_path = run_root / f"training-resume-{args.job_id}.log"
    command = [str(python), str(entrypoint), "--config", str(config_path), "--resume", str(checkpoint)]
    with log_path.open("a") as log:
        child = subprocess.Popen(command, cwd=checkout_root, stdout=log, stderr=subprocess.STDOUT)

        def forward(signum, _frame):
            if child.poll() is None:
                child.send_signal(signum)

        signal.signal(signal.SIGINT, forward)
        signal.signal(signal.SIGTERM, forward)
        returncode = child.wait()
    latest = json.loads((run_root / "student" / "latest-checkpoint.json").read_text())
    receipt.update(
        phase="complete" if (run_root / "student" / "complete.json").exists() else "paused",
        finished_utc=datetime.now(timezone.utc).isoformat(),
        returncode=returncode,
        latest_checkpoint=latest,
        log=str(log_path),
    )
    write_json(session / "training-resume-receipt.json", receipt)
    if returncode:
        raise SystemExit(returncode)


if __name__ == "__main__":
    main()
