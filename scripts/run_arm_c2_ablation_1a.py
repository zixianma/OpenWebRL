#!/usr/bin/env python3
"""Run approved C2 ablation 1A training and fixed-100 evaluation on one H200."""

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from resume_arm_c2_training import allocation_deadline, parse_job_record, write_json


REPO = Path(__file__).resolve().parents[1]
RUNTIME = Path("/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction")
SOURCE_RUN = RUNTIME / "runs/c2-full-282782-20260908T075414Z"
RUN_ROOT = RUNTIME / "runs/c2-ablation-1a"
ARM_CHECKOUT = RUNTIME / "action-reward-models-c2"
TRAINER = ARM_CHECKOUT / "actor_distillation/train_openwebrl_c2.py"
TRAINING_PYTHON = RUNTIME / "venv/bin/python"
TRAINER_COMMIT = "f3496dec08d416348900614e0183f2fa9fca08e6"
TRAINER_SHA256 = "704dc25f95a807428c12def2866a71f66dd1f7e74d62107c1be5b2a5ae0b0363"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def validate_allocation(job_id):
    if os.getenv("SLURM_JOB_ID") != job_id or f"/job_{job_id}/" not in Path("/proc/self/cgroup").read_text():
        raise ValueError("1A controller must run inside its approved allocation")
    raw = subprocess.check_output(["scontrol", "show", "job", job_id, "-o"], text=True, timeout=20)
    record = parse_job_record(raw)
    if record.get("JobState") != "RUNNING" or int(record.get("NumNodes", 0)) != 1:
        raise ValueError("1A allocation is not one running node")
    if int(record.get("NumCPUs", 0)) < 4:
        raise ValueError("1A allocation has fewer than four CPUs")
    gpus = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid,name", "--format=csv,noheader"], text=True, timeout=20
    ).strip().splitlines()
    if len(gpus) != 1 or "H200" not in gpus[0]:
        raise ValueError(f"1A requires exactly one visible H200, found {gpus}")
    if subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True, timeout=20
    ).strip():
        raise ValueError("Assigned 1A GPU is already occupied")
    return record, gpus[0].split(",", 1)[0]


def prepare(job_id, record, gpu_uuid):
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    source_audit = json.loads((SOURCE_RUN / "dataset-audit.json").read_text())
    if source_audit["retained_turns"] != 8394 or sha256(source_audit["dataset"]) != source_audit["dataset_sha256"]:
        raise ValueError("Source C2 dataset no longer matches the audited 8,394 rows")
    audit = dict(source_audit, ablation="1A", source_audit=str(SOURCE_RUN / "dataset-audit.json"))
    write_json(RUN_ROOT / "dataset-audit.json", audit)
    if subprocess.check_output(["git", "-C", str(ARM_CHECKOUT), "rev-parse", "HEAD"], text=True).strip() != TRAINER_COMMIT:
        raise ValueError("ARM training checkout moved from the pinned 1A commit")
    if sha256(TRAINER) != TRAINER_SHA256:
        raise ValueError("Pinned 1A trainer checksum changed")
    end = allocation_deadline(record, margin_minutes=0)
    training_stop = end - timedelta(minutes=70)
    config = {
        "allocation": job_id,
        "gpu_uuid": gpu_uuid,
        "node": record.get("NodeList", os.getenv("SLURMD_NODENAME")),
        "output": str(RUN_ROOT),
        "task_count": 2091,
        "stop_utc": training_stop.isoformat(),
        "training_python": str(TRAINING_PYTHON),
        "teacher_manifest": {"actor": "/gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT"},
        "training_checkout": {"root": str(ARM_CHECKOUT), "entrypoint": str(TRAINER.relative_to(ARM_CHECKOUT)),
                              "commit": TRAINER_COMMIT, "sha256": TRAINER_SHA256,
                              "branch": "openwebrl/c2-filtered-sft"},
        "training": {
            "ablation": "1A",
            "lora_rank": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "learning_rate": 1e-5,
            "epochs": 1,
            "gradient_accumulation": 32,
            "max_tokens": 32768,
            "seed": 42,
            "all_eligible_turns": True,
            "schedule": "example_indexed_first_epoch",
            "warmup_examples": 512,
            "cosine_horizon_examples": 16788,
            "save_updates": [66, 132, 198, 250],
            "endpoint_selection": "predeclared epoch-1/update-263 endpoint; no evaluation-driven selection",
        },
        "authorization": {"request": "1 GPU x 5 hours for run 1A plus fixed 100-task validation",
                          "approved_in_conversation": True, "max_h200_hours": 5,
                          "allocation_end_utc": end.isoformat(), "training_stop_utc": training_stop.isoformat()},
    }
    config_path = RUN_ROOT / "frozen-config.json"
    if config_path.exists():
        prior = json.loads(config_path.read_text())
        # Allocation-bound fields may change only for a deliberate resume; this first launcher refuses it.
        if prior != config:
            raise ValueError("A different 1A frozen config already exists")
    else:
        write_json(config_path, config)
    scaling = RUN_ROOT / "evaluation/checkpoint-scaling-100"
    scaling.mkdir(parents=True, exist_ok=True)
    sample = json.loads((REPO / "openwebrl/docs/arm_c2_scaling_100.json").read_text())
    write_json(scaling / "sample.json", sample)
    write_json(RUN_ROOT / "status.json", {"phase": "prepared", "updated_utc": datetime.now(timezone.utc).isoformat(),
                                           "allocation": job_id, "gpu_uuid": gpu_uuid,
                                           "training_stop_utc": training_stop.isoformat(),
                                           "allocation_end_utc": end.isoformat()})
    return config_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()
    record, gpu_uuid = validate_allocation(args.job_id)
    config = prepare(args.job_id, record, gpu_uuid)
    processes = []
    stopping = False

    def stop(*_ignored):
        nonlocal stopping
        stopping = True
        for process in processes:
            if process.poll() is None:
                process.terminate()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    train_log = (RUN_ROOT / "training.log").open("a")
    trainer = subprocess.Popen([str(TRAINING_PYTHON), str(TRAINER), "--config", str(config)],
                               cwd=REPO, stdout=train_log, stderr=subprocess.STDOUT)
    processes.append(trainer)
    write_json(RUN_ROOT / "status.json", {"phase": "training", "updated_utc": datetime.now(timezone.utc).isoformat(),
                                           "allocation": args.job_id, "pid": trainer.pid})
    wandb = None
    try:
        while trainer.poll() is None and not stopping:
            if wandb is None and (RUN_ROOT / "student/metrics.jsonl").exists():
                wandb_log = (RUN_ROOT / "wandb-sync.log").open("a")
                wandb = subprocess.Popen(
                    [str(TRAINING_PYTHON), str(REPO / "scripts/sync_arm_c2_wandb.py"),
                     "--run-root", str(RUN_ROOT), "--env-file", str(REPO / ".env"),
                     "--name", "arm-c2-ablation-1a", "--group", "arm-c2-ablation",
                     "--tag", "ablation-1a", "--poll-seconds", "20"],
                    cwd=REPO, stdout=wandb_log, stderr=subprocess.STDOUT)
                processes.append(wandb)
            time.sleep(10)
        train_code = trainer.wait()
        train_log.close()
        if train_code or not (RUN_ROOT / "student/complete.json").exists():
            write_json(RUN_ROOT / "status.json", {"phase": "training_paused_or_failed",
                                                   "updated_utc": datetime.now(timezone.utc).isoformat(),
                                                   "allocation": args.job_id, "returncode": train_code})
            return
        if wandb is not None:
            try:
                wandb.wait(timeout=120)
            except subprocess.TimeoutExpired:
                wandb.terminate()
                wandb.wait(timeout=30)
        checkpoint = RUN_ROOT / "student/epoch-1"
        write_json(RUN_ROOT / "status.json", {"phase": "evaluating_fixed_100",
                                               "updated_utc": datetime.now(timezone.utc).isoformat(),
                                               "allocation": args.job_id, "checkpoint": str(checkpoint)})
        command = [str(TRAINING_PYTHON), str(REPO / "scripts/run_arm_checkpoint_eval.py"),
                   "--job-id", args.job_id, "--run-root", str(RUN_ROOT),
                   "--checkpoint", str(checkpoint), "--label", "endpoint-000263",
                   "--sample", str(RUN_ROOT / "evaluation/checkpoint-scaling-100/sample.json"),
                   "--actor-port", "19110", "--browser-port-start", "19200", "--browser-port-end", "19399"]
        with (RUN_ROOT / "evaluation-controller.log").open("a") as log:
            eval_code = subprocess.call(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
        eval_status_path = RUN_ROOT / "evaluation/checkpoint-scaling-100/endpoint-000263/status.json"
        eval_status = json.loads(eval_status_path.read_text()) if eval_status_path.exists() else None
        write_json(RUN_ROOT / "status.json", {"phase": "complete" if eval_code == 0 and eval_status and eval_status.get("phase") == "complete" else "evaluation_paused_or_failed",
                                               "updated_utc": datetime.now(timezone.utc).isoformat(),
                                               "allocation": args.job_id, "training_updates": 263,
                                               "evaluation_returncode": eval_code,
                                               "evaluation": eval_status})
    finally:
        stop()
        for process in processes:
            if process.poll() is None:
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()
        if not train_log.closed:
            train_log.close()


if __name__ == "__main__":
    main()
