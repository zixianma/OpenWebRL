#!/usr/bin/env python3
"""Evaluate C2 update 500 and ablation 1A endpoint on the OM2W holdout 200."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from resume_arm_c2_training import parse_job_record, write_json


REPO = Path(__file__).resolve().parents[1]
RUNTIME = Path("/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/runs")
C2_ROOT = RUNTIME / "c2-full-282782-20260908T075414Z"
A1_ROOT = RUNTIME / "c2-ablation-1a"
GROUP = A1_ROOT / "evaluation/holdout-200-vs-c2"
COHORTS = REPO / "openwebrl/docs/arm_c2_full300_cohorts.json"
COHORT_NAME = "checkpoint_selection_holdout_200"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def result_count(label: str) -> int:
    path = GROUP / label / "online-mind2web/results"
    return len(list(path.glob("*.json"))) if path.exists() else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()
    if os.getenv("SLURM_JOB_ID") != args.job_id or f"/job_{args.job_id}/" not in Path("/proc/self/cgroup").read_text():
        raise ValueError("Holdout-200 controller must run inside its approved allocation")
    record = parse_job_record(
        subprocess.check_output(["scontrol", "show", "job", args.job_id, "-o"], text=True, timeout=20)
    )
    if int(record.get("NumNodes", "0")) != 1 or int(record.get("NumCPUs", "0")) < 16:
        raise ValueError("Holdout-200 pair requires one node and at least 16 CPUs")
    devices = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid,name", "--format=csv,noheader"], text=True, timeout=20
    ).strip().splitlines()
    if len(devices) != 2 or any("H200" not in device for device in devices):
        raise ValueError("Holdout-200 pair requires exactly two visible H200s")
    if subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True, timeout=20
    ).strip():
        raise ValueError("Assigned holdout-200 GPU is already occupied")

    gpu_uuids = [device.split(",", 1)[0] for device in devices]
    specs = [
        {
            "label": "c2-update-000500",
            "model": C2_ROOT / "evaluation/checkpoint-scaling-100/update-000500/merged-model",
            "checkpoint": C2_ROOT / "student/update-000500",
            "audit": C2_ROOT / "dataset-audit.json",
            "actor_port": 19110,
            "browser_start": 19200,
            "browser_end": 19399,
            "gpu_uuid": gpu_uuids[0],
        },
        {
            "label": "1a-endpoint-000263",
            "model": A1_ROOT / "evaluation/checkpoint-scaling-100/endpoint-000263/merged-model",
            "checkpoint": A1_ROOT / "student/epoch-1",
            "audit": A1_ROOT / "dataset-audit.json",
            "actor_port": 19410,
            "browser_start": 19500,
            "browser_end": 19699,
            "gpu_uuid": gpu_uuids[1],
        },
    ]
    for spec in specs:
        manifest = json.loads((spec["model"] / "merge-manifest.json").read_text())
        if Path(manifest["checkpoint"]).resolve() != spec["checkpoint"].resolve():
            raise ValueError(f"Merged-model provenance mismatch for {spec['label']}")
    GROUP.mkdir(parents=True, exist_ok=True)
    config = {
        "allocation": args.job_id,
        "gpu_uuids": gpu_uuids,
        "task_count_per_policy": 200,
        "task_cohort_manifest": str(COHORTS.resolve()),
        "task_cohort_manifest_sha256": sha256(COHORTS),
        "cohort_name": COHORT_NAME,
        "parallel_per_policy": 12,
        "mem_fraction_static_per_server": 0.45,
        "protocol": {
            "seed": 42,
            "temperature": 0.7,
            "top_p": 0.9,
            "max_new_tokens": 1024,
            "max_steps": 30,
            "history": "full",
            "screenshots": 1,
            "judge": "o4-mini",
            "judge_protocol": "online_mind2web/AgentTrek",
        },
        "policies": [
            {
                "label": spec["label"],
                "model": str(spec["model"].resolve()),
                "checkpoint": str(spec["checkpoint"].resolve()),
                "merge_manifest_sha256": sha256(spec["model"] / "merge-manifest.json"),
            }
            for spec in specs
        ],
    }
    write_json(GROUP / "run-config.json", config)
    status = GROUP / "pair-status.json"
    worker = REPO / "scripts/run_arm_full300_checkpoint_eval.py"
    processes = []
    logs = []
    stopping = False

    def stop(*_ignored):
        nonlocal stopping
        stopping = True
        for process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    def launch(spec):
        log = (GROUP / f"controller-{spec['label']}.log").open("a")
        command = [
            sys.executable,
            str(worker),
            "--job-id",
            args.job_id,
            "--run-root",
            str(A1_ROOT),
            "--label",
            spec["label"],
            "--model",
            str(spec["model"]),
            "--expected-checkpoint",
            str(spec["checkpoint"]),
            "--dataset-audit",
            str(spec["audit"]),
            "--output-group",
            str(GROUP),
            "--task-cohort",
            str(COHORTS),
            "--cohort-name",
            COHORT_NAME,
            "--actor-port",
            str(spec["actor_port"]),
            "--browser-port-start",
            str(spec["browser_start"]),
            "--browser-port-end",
            str(spec["browser_end"]),
            "--parallel",
            "12",
            "--mem-fraction-static",
            "0.45",
        ]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = spec["gpu_uuid"]
        process = subprocess.Popen(
            command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, env=environment
        )
        processes.append(process)
        logs.append(log)

    try:
        for spec in specs:
            launch(spec)
        while any(process.poll() is None for process in processes):
            write_json(
                status,
                {
                    "phase": "evaluating",
                    "updated_utc": datetime.now(timezone.utc).isoformat(),
                    **config,
                    "completed": {spec["label"]: result_count(spec["label"]) for spec in specs},
                },
            )
            time.sleep(30)
        codes = {specs[i]["label"]: process.returncode for i, process in enumerate(processes)}
        complete = all(
            (GROUP / spec["label"] / "online-mind2web/summary.json").exists() for spec in specs
        )
        analysis_code = None
        if complete and all(code == 0 for code in codes.values()):
            with (GROUP / "summarize.log").open("a") as log:
                analysis_code = subprocess.call(
                    [
                        sys.executable,
                        str(REPO / "scripts/summarize_arm_c2_vs_1a_full300.py"),
                        "--group",
                        str(GROUP),
                        "--cohorts",
                        str(REPO / "openwebrl/docs/arm_c2_full300_cohorts.json"),
                    ],
                    cwd=REPO,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
        succeeded = complete and all(code == 0 for code in codes.values()) and analysis_code == 0
        write_json(
            status,
            {
                "phase": "complete" if succeeded else "paused",
                "updated_utc": datetime.now(timezone.utc).isoformat(),
                **config,
                "completed": {spec["label"]: result_count(spec["label"]) for spec in specs},
                "returncodes": codes,
                "analysis_returncode": analysis_code,
            },
        )
    finally:
        stop()
        for process in processes:
            try:
                process.wait(timeout=90)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        for log in logs:
            log.close()


if __name__ == "__main__":
    main()
