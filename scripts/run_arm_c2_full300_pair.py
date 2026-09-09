#!/usr/bin/env python3
"""Run update-500 and update-700 full OM2W evaluations together on one H200."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from resume_arm_c2_training import parse_job_record, write_json


REPO = Path(__file__).resolve().parents[1]
RUNTIME_BASELINE = Path(
    "/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/runs/"
    "dedicated-282209-20260908T005547Z/full/baseline"
)


def result_count(path):
    results = path / "online-mind2web/results"
    return len(list(results.glob("*.json"))) if results.exists() else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--run-root", required=True, type=Path)
    args = parser.parse_args()
    if os.getenv("SLURM_JOB_ID") != args.job_id or f"/job_{args.job_id}/" not in Path("/proc/self/cgroup").read_text():
        raise ValueError("Full300 pair controller must run inside the approved allocation")
    record = parse_job_record(subprocess.check_output(
        ["scontrol", "show", "job", args.job_id, "-o"], text=True, timeout=20))
    if int(record.get("NumNodes", "0")) != 1 or int(record.get("NumCPUs", "0")) < 4:
        raise ValueError("Full300 pair requires one node and at least four CPUs")
    devices = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True, timeout=20
    ).strip().splitlines()
    if len(devices) != 1:
        raise ValueError("Full300 pair requires exactly one visible H200")
    root = args.run_root.resolve()
    group = root / "evaluation/checkpoint-full-300"
    group.mkdir(parents=True, exist_ok=True)
    status = group / "pair-status.json"
    worker = REPO / "scripts/run_arm_full300_checkpoint_eval.py"
    specs = [
        ("update-000500", root / "evaluation/checkpoint-scaling-100/update-000500/merged-model", 19110, 19200, 19399),
        ("update-000700", root / "evaluation/checkpoint-scaling-100/update-000700/merged-model", 19410, 19500, 19699),
    ]
    config = {"allocation": args.job_id, "gpu_uuid": devices[0], "task_count_per_checkpoint": 300,
              "parallel_per_checkpoint": 6, "mem_fraction_static_per_server": 0.3,
              "protocol": {"seed": 42, "temperature": 0.7, "top_p": 0.9, "max_new_tokens": 1024,
                           "max_steps": 30, "history": "full", "screenshots": 1,
                           "judge": "o4-mini", "judge_protocol": "online_mind2web/AgentTrek"},
              "checkpoints": [label for label, *_ in specs]}
    write_json(group / "run-config.json", config)
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
        label, model, actor_port, browser_start, browser_end = spec
        log = (group / f"controller-{label}.log").open("a")
        command = [sys.executable, str(worker), "--job-id", args.job_id, "--run-root", str(root),
                   "--label", label, "--model", str(model), "--actor-port", str(actor_port),
                   "--browser-port-start", str(browser_start), "--browser-port-end", str(browser_end),
                   "--parallel", "6", "--mem-fraction-static", "0.3"]
        process = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        processes.append(process); logs.append(log)

    try:
        launch(specs[0])
        first_status = group / specs[0][0] / "status.json"
        startup_deadline = time.time() + 900
        while time.time() < startup_deadline and not stopping:
            if processes[0].poll() is not None:
                raise RuntimeError("Update-500 worker exited before paired launch")
            if first_status.exists() and json.loads(first_status.read_text()).get("phase") == "evaluating":
                break
            time.sleep(2)
        else:
            raise TimeoutError("Update-500 server did not become ready")
        launch(specs[1])
        while any(process.poll() is None for process in processes):
            snapshot = {label: result_count(group / label) for label, *_ in specs}
            write_json(status, {"phase": "evaluating", "updated_utc": datetime.now(timezone.utc).isoformat(),
                                **config, "completed": snapshot})
            time.sleep(30)
        codes = {specs[i][0]: process.returncode for i, process in enumerate(processes)}
        complete = all((group / label / "online-mind2web/summary.json").exists() for label, *_ in specs)
        succeeded = complete and all(code == 0 for code in codes.values())
        analysis_code = None
        if succeeded:
            with (group / "summarize.log").open("a") as log:
                analysis_code = subprocess.call(
                    [sys.executable, str(REPO / "scripts/summarize_arm_c2_full300.py"),
                     "--run-root", str(root),
                     "--cohorts", str(REPO / "openwebrl/docs/arm_c2_full300_cohorts.json"),
                     "--baseline", str(RUNTIME_BASELINE)],
                    cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
        write_json(status, {"phase": "complete" if succeeded and analysis_code == 0 else "paused",
                            "updated_utc": datetime.now(timezone.utc).isoformat(), **config,
                            "completed": {label: result_count(group / label) for label, *_ in specs},
                            "returncodes": codes, "analysis_returncode": analysis_code})
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
