#!/usr/bin/env python3
"""Evaluate one durable C2 checkpoint on a fixed Online-Mind2Web subset."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request

from resume_arm_c2_training import allocation_deadline, parse_job_record, write_json


REPO = Path(__file__).resolve().parents[1]
RUNTIME = Path("/gpfs/scrubbed/zixianma/openwebrl-runtime")
ARM_RUNTIME = RUNTIME / "arm-reproduction"
ARM_PYTHON = ARM_RUNTIME / "venv/bin/python"
SERVER_PYTHON = RUNTIME / "venv/bin/python"


def fetch_json(url):
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.load(response)


def reachable(url):
    with urllib.request.urlopen(url, timeout=10) as response:
        response.read()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--sample", required=True, type=Path)
    parser.add_argument("--actor-port", required=True, type=int)
    parser.add_argument("--browser-port-start", required=True, type=int)
    parser.add_argument("--browser-port-end", required=True, type=int)
    parser.add_argument("--allow-shared-gpu", action="store_true",
                        help="Allow another recorded evaluator server on the assigned GPU")
    parser.add_argument("--mem-fraction-static", type=float, default=0.4)
    parser.add_argument("--parallel", type=int, default=8)
    args = parser.parse_args()
    if not 0.1 <= args.mem_fraction_static <= 0.8:
        parser.error("--mem-fraction-static must be between 0.1 and 0.8")
    if args.parallel < 1:
        parser.error("--parallel must be positive")
    if os.getenv("SLURM_JOB_ID") != args.job_id or f"/job_{args.job_id}/" not in Path("/proc/self/cgroup").read_text():
        raise ValueError("Checkpoint evaluation must run in its authorized allocation")
    record = parse_job_record(subprocess.check_output(
        ["scontrol", "show", "job", args.job_id, "-o"], text=True, timeout=20))
    deadline = allocation_deadline(record, margin_minutes=5)
    devices = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True, timeout=20
    ).strip().splitlines()
    if len(devices) != 1:
        raise ValueError("Checkpoint worker requires exactly one visible GPU")
    root = args.run_root.resolve()
    checkpoint = args.checkpoint.resolve()
    while not (checkpoint / "progress.json").exists():
        if datetime.now(timezone.utc) >= deadline:
            raise TimeoutError("Checkpoint did not become available in this allocation")
        time.sleep(15)
    sample = json.loads(args.sample.read_text())
    if sample["count"] != 100 or len(sample["indices"]) != 100 or len(set(sample["indices"])) != 100:
        raise ValueError("Expected one fixed 100-task sample")
    work = root / "evaluation" / "checkpoint-scaling-100" / args.label
    work.mkdir(parents=True, exist_ok=True)
    merged = work / "merged-model"
    status = work / "status.json"
    completed_summary = work / "online-mind2web" / "summary.json"
    completed_manifest = work / "online-mind2web" / "manifest.json"
    if completed_summary.exists() and completed_manifest.exists():
        summary = json.loads(completed_summary.read_text())
        manifest = json.loads(completed_manifest.read_text())
        if (
            summary.get("scheduled") != 100
            or summary.get("attempted") != 100
            or manifest.get("task_ids") != sample["task_ids"]
            or os.path.realpath(manifest.get("actor", "")) != os.path.realpath(merged)
            or manifest.get("judge") != "o4-mini"
            or manifest.get("judge_protocol") != "online_mind2web/AgentTrek"
        ):
            raise ValueError("Existing completed evaluation does not match the fixed cohort/protocol")
        write_json(status, {"phase": "complete", "updated_utc": datetime.now(timezone.utc).isoformat(),
                            "allocation": args.job_id, "completed": 100, "scheduled": 100,
                            "returncode": 0, "summary": summary, "reused_complete": True})
        return
    pids = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True, timeout=20
    ).strip()
    if pids and not args.allow_shared_gpu:
        raise ValueError("Assigned checkpoint-evaluation GPU is occupied")
    write_json(status, {"phase": "merging", "updated_utc": datetime.now(timezone.utc).isoformat(),
                        "allocation": args.job_id, "gpu_uuid": devices[0], "checkpoint": str(checkpoint),
                        "allow_shared_gpu": args.allow_shared_gpu,
                        "mem_fraction_static": args.mem_fraction_static})
    with (work / "merge.log").open("a") as log:
        subprocess.run(
            [str(ARM_PYTHON), str(REPO / "scripts/merge_arm_c2_student.py"),
             "--run-root", str(root), "--checkpoint", str(checkpoint), "--output", str(merged)],
            cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)

    server_log = (work / "actor-server.log").open("a")
    server = subprocess.Popen(
        [str(SERVER_PYTHON), "-m", "sglang.launch_server", "--model-path", str(merged),
         "--host", "127.0.0.1", "--port", str(args.actor_port), "--dtype", "bfloat16", "--tp", "1",
         "--mem-fraction-static", str(args.mem_fraction_static), "--context-length", "32768",
         "--max-running-requests", "24",
         "--chunked-prefill-size", "4096", "--disable-cuda-graph"],
        cwd=REPO, stdout=server_log, stderr=subprocess.STDOUT, start_new_session=True)

    def stop(*_ignored):
        if server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        startup = min(deadline.timestamp(), time.time() + 900)
        while True:
            if server.poll() is not None:
                raise RuntimeError("Actor server exited during startup")
            try:
                info = fetch_json(f"http://127.0.0.1:{args.actor_port}/get_model_info")
                reachable(f"http://127.0.0.1:{args.actor_port}/health_generate")
                if os.path.realpath(info.get("model_path", "")) != os.path.realpath(merged):
                    raise ValueError("Actor server model differs from merged checkpoint")
                break
            except (OSError, ValueError):
                if time.time() >= startup:
                    raise TimeoutError("Actor startup timed out")
                time.sleep(2)
        output = work / "online-mind2web"
        write_json(status, {"phase": "evaluating", "updated_utc": datetime.now(timezone.utc).isoformat(),
                            "allocation": args.job_id, "gpu_uuid": devices[0], "output": str(output)})
        remaining = int((deadline - datetime.now(timezone.utc)).total_seconds())
        command = ["timeout", "--signal=TERM", "--kill-after=120", str(remaining),
                   str(ARM_PYTHON), "-m", "openwebrl.arm_eval", "--mode", "baseline",
                   "--actor", str(merged), "--actor-port", str(args.actor_port), "--output", str(output),
                   "--parallel", str(args.parallel), "--task-indices", ",".join(map(str, sample["indices"])),
                   "--seed", "42", "--temperature", "0.7", "--top-p", "0.9",
                   "--max-new-tokens", "1024", "--max-steps", "30", "--task-timeout", "1800",
                   "--judge-model", "o4-mini", "--env-file", ".env",
                   "--browser-port-start", str(args.browser_port_start),
                   "--browser-port-end", str(args.browser_port_end)]
        with (work / "eval.log").open("a") as log:
            code = subprocess.call(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
        count = len(list((output / "results").glob("*.json"))) if (output / "results").exists() else 0
        phase = "complete" if code == 0 and (output / "summary.json").exists() else "paused"
        write_json(status, {"phase": phase, "updated_utc": datetime.now(timezone.utc).isoformat(),
                            "allocation": args.job_id, "completed": count, "scheduled": 100,
                            "returncode": code, "summary": json.loads((output / "summary.json").read_text())
                            if (output / "summary.json").exists() else None})
        if code not in (0, 124):
            raise SystemExit(code)
    finally:
        stop()
        server_log.close()


if __name__ == "__main__":
    main()
