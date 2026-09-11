#!/usr/bin/env python3
"""Own the calibrated ARM preference run in already-authorized job 286384."""

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from resume_arm_c2_training import allocation_deadline, parse_job_record, write_json


REPO = Path(__file__).resolve().parents[1]
RUNTIME = Path("/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction")
ROOT = RUNTIME / "runs/arm-preference-calibrated-286384-r4"
PYTHON = RUNTIME / "venv/bin/python"
SOURCE_AUDIT = RUNTIME / "runs/c2-full-282782-20260908T075414Z/dataset-audit.json"
ACTOR = Path("/gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT")


def main():
    job = "286384"
    if os.getenv("SLURM_JOB_ID") != job or f"/job_{job}/" not in Path("/proc/self/cgroup").read_text():
        raise ValueError("Controller must run inside job 286384")
    record = parse_job_record(subprocess.check_output(["scontrol", "show", "job", job, "-o"], text=True))
    if record.get("JobState") != "RUNNING":
        raise ValueError("Allocation is not running")
    devices = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,name", "--format=csv,noheader"], text=True).splitlines()
    if len(devices) != 2 or any("H200" not in device for device in devices):
        raise ValueError(f"Expected two H200s, found {devices}")
    if subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip():
        raise ValueError("Allocated GPUs are already occupied")
    end = allocation_deadline(record, margin_minutes=0)
    ROOT.mkdir(parents=True, exist_ok=True)
    config = {
        "allocation": 286384,
        "node": record.get("NodeList"),
        "output": str(ROOT),
        "actor": str(ACTOR),
        "source_audit": str(SOURCE_AUDIT),
        "seed": 42,
        "pair_count": 4192,
        "calibration_pairs": 4,
        "target_gradient_ratio": 0.20,
        "beta": 0.1,
        "gradient_accumulation_per_rank": 16,
        "effective_batch_pairs": 32,
        "save_updates": [33, 66, 99],
        "stop_unix": (end - timedelta(minutes=8)).timestamp(),
        "allocation_end_utc": end.isoformat(),
        "authorization": "User supplied 2 H200s x 4 hours in existing allocation 286384",
    }
    config_path = ROOT / "frozen-config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Existing frozen configuration differs")
    write_json(config_path, config)
    write_json(ROOT / "dataset-audit.json", {"retained_turns": 4192, "scheduled": 4192,
               "successful_tasks_with_usable_turns": None, "source_audit": str(SOURCE_AUDIT)})
    write_json(ROOT / "status.json", {"phase": "starting", "updated_utc": datetime.now(timezone.utc).isoformat(),
               "allocation": 286384, "gpus": devices, "allocation_end_utc": end.isoformat()})
    processes = []
    stopping = False
    def stop(*_):
        nonlocal stopping
        stopping = True
        for process in processes:
            if process.poll() is None:
                process.terminate()
    signal.signal(signal.SIGINT, stop); signal.signal(signal.SIGTERM, stop)
    log = (ROOT / "training.log").open("a")
    command = [str(PYTHON), "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
               str(REPO / "scripts/train_arm_preference.py"), "--config", str(config_path)]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO) + (os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else "")
    environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    trainer = subprocess.Popen(command, cwd=REPO, env=environment, stdout=log, stderr=subprocess.STDOUT)
    processes.append(trainer)
    write_json(ROOT / "status.json", {"phase": "training", "updated_utc": datetime.now(timezone.utc).isoformat(),
               "allocation": 286384, "pid": trainer.pid, "allocation_end_utc": end.isoformat()})
    wandb = None
    try:
        while trainer.poll() is None and not stopping:
            if wandb is None and (ROOT / "student/metrics.jsonl").exists():
                wandb_log = (ROOT / "wandb-sync.log").open("a")
                wandb = subprocess.Popen([str(PYTHON), str(REPO / "scripts/sync_arm_c2_wandb.py"),
                    "--run-root", str(ROOT), "--env-file", str(REPO / ".env"),
                    "--project", "openwebrl-arm", "--name", "arm-preference-calibrated-286384",
                    "--group", "arm-preference-distillation", "--tag", "preference-calibrated"],
                    cwd=REPO, stdout=wandb_log, stderr=subprocess.STDOUT)
                processes.append(wandb)
            time.sleep(10)
        code = trainer.wait()
        phase = "trained" if code == 0 and (ROOT / "student/complete.json").exists() else "paused_or_failed"
        write_json(ROOT / "status.json", {"phase": phase, "updated_utc": datetime.now(timezone.utc).isoformat(),
                   "allocation": 286384, "returncode": code,
                   "complete": json.loads((ROOT / "student/complete.json").read_text()) if (ROOT / "student/complete.json").exists() else None})
    finally:
        stop()
        for process in processes:
            if process.poll() is None:
                try: process.wait(timeout=60)
                except subprocess.TimeoutExpired: process.kill(); process.wait()
        log.close()


if __name__ == "__main__":
    main()
