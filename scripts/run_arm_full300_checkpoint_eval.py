#!/usr/bin/env python3
"""Evaluate one merged C2 checkpoint on all 300 Online-Mind2Web tasks."""

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
ARM_PYTHON = RUNTIME / "arm-reproduction/venv/bin/python"
SERVER_PYTHON = RUNTIME / "venv/bin/python"
TASK_FILE = REPO / "openwebrl/data/online-mind2web.jsonl"


def get_json(url):
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--label", required=True, choices=("update-000500", "update-000700"))
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--actor-port", required=True, type=int)
    parser.add_argument("--browser-port-start", required=True, type=int)
    parser.add_argument("--browser-port-end", required=True, type=int)
    parser.add_argument("--parallel", type=int, default=6)
    parser.add_argument("--mem-fraction-static", type=float, default=0.3)
    args = parser.parse_args()
    if os.getenv("SLURM_JOB_ID") != args.job_id or f"/job_{args.job_id}/" not in Path("/proc/self/cgroup").read_text():
        raise ValueError("Full checkpoint evaluation must run inside its authorized allocation")
    if not 1 <= args.parallel <= 16 or not 0.25 <= args.mem_fraction_static <= 0.45:
        parser.error("unsupported parallelism or server memory fraction")
    record = parse_job_record(subprocess.check_output(
        ["scontrol", "show", "job", args.job_id, "-o"], text=True, timeout=20))
    deadline = allocation_deadline(record, margin_minutes=5)
    devices = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True, timeout=20
    ).strip().splitlines()
    if len(devices) != 1:
        raise ValueError("Full300 paired worker requires exactly one visible GPU")
    root = args.run_root.resolve()
    model = args.model.resolve()
    expected_checkpoint = (root / "student" / args.label).resolve()
    merge = json.loads((model / "merge-manifest.json").read_text())
    if Path(merge["checkpoint"]).resolve() != expected_checkpoint:
        raise ValueError("Merged-model checkpoint provenance does not match label")
    audit = json.loads((root / "dataset-audit.json").read_text())
    if merge.get("dataset_sha256") != audit.get("dataset_sha256"):
        raise ValueError("Merged-model and C2 dataset provenance differ")
    work = root / "evaluation/checkpoint-full-300" / args.label
    output = work / "online-mind2web"
    status = work / "status.json"
    work.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    manifest_path = output / "manifest.json"
    if summary_path.exists() and manifest_path.exists():
        summary, manifest = json.loads(summary_path.read_text()), json.loads(manifest_path.read_text())
        if (summary.get("scheduled"), summary.get("attempted"), len(manifest.get("task_ids", []))) != (300, 300, 300):
            raise ValueError("Existing full300 evaluation is incomplete or malformed")
        if os.path.realpath(manifest.get("actor", "")) != os.path.realpath(model):
            raise ValueError("Existing full300 evaluation used another model")
        write_json(status, {"phase": "complete", "updated_utc": datetime.now(timezone.utc).isoformat(),
                            "allocation": args.job_id, "completed": 300, "scheduled": 300,
                            "summary": summary, "reused_complete": True})
        return
    server_log = (work / "actor-server.log").open("a")
    server = subprocess.Popen(
        [str(SERVER_PYTHON), "-m", "sglang.launch_server", "--model-path", str(model),
         "--host", "127.0.0.1", "--port", str(args.actor_port), "--dtype", "bfloat16", "--tp", "1",
         "--mem-fraction-static", str(args.mem_fraction_static), "--context-length", "32768",
         "--max-running-requests", "24", "--chunked-prefill-size", "4096", "--disable-cuda-graph"],
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
        write_json(status, {"phase": "starting_server", "updated_utc": datetime.now(timezone.utc).isoformat(),
                            "allocation": args.job_id, "gpu_uuid": devices[0], "model": str(model),
                            "parallel": args.parallel, "mem_fraction_static": args.mem_fraction_static})
        startup = min(deadline.timestamp(), time.time() + 900)
        while True:
            if server.poll() is not None:
                raise RuntimeError("Full300 actor server exited during startup")
            try:
                info = get_json(f"http://127.0.0.1:{args.actor_port}/get_model_info")
                with urllib.request.urlopen(f"http://127.0.0.1:{args.actor_port}/health_generate", timeout=10) as response:
                    response.read()
                if os.path.realpath(info.get("model_path", "")) != os.path.realpath(model):
                    raise ValueError("Full300 actor server model mismatch")
                break
            except (OSError, ValueError):
                if time.time() >= startup:
                    raise TimeoutError("Full300 actor startup timed out")
                time.sleep(2)
        write_json(status, {"phase": "evaluating", "updated_utc": datetime.now(timezone.utc).isoformat(),
                            "allocation": args.job_id, "gpu_uuid": devices[0], "model": str(model),
                            "parallel": args.parallel, "output": str(output)})
        remaining = max(1, int((deadline - datetime.now(timezone.utc)).total_seconds()))
        command = ["timeout", "--signal=TERM", "--kill-after=120", str(remaining), str(ARM_PYTHON),
                   "-m", "openwebrl.arm_eval", "--mode", "baseline", "--actor", str(model),
                   "--actor-port", str(args.actor_port), "--output", str(output), "--parallel", str(args.parallel),
                   "--task-file", str(TASK_FILE), "--seed", "42", "--temperature", "0.7", "--top-p", "0.9",
                   "--max-new-tokens", "1024", "--max-steps", "30", "--task-timeout", "1800",
                   "--judge-model", "o4-mini", "--env-file", ".env",
                   "--browser-port-start", str(args.browser_port_start), "--browser-port-end", str(args.browser_port_end)]
        with (work / "eval.log").open("a") as log:
            code = subprocess.call(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
        count = len(list((output / "results").glob("*.json"))) if (output / "results").exists() else 0
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
        phase = "complete" if code == 0 and summary and summary.get("attempted") == 300 else "paused"
        write_json(status, {"phase": phase, "updated_utc": datetime.now(timezone.utc).isoformat(),
                            "allocation": args.job_id, "completed": count, "scheduled": 300,
                            "returncode": code, "summary": summary})
        if code not in (0, 124):
            raise SystemExit(code)
    except Exception as exc:
        write_json(status, {"phase": "failed", "updated_utc": datetime.now(timezone.utc).isoformat(),
                            "allocation": args.job_id, "error_type": type(exc).__name__, "error": str(exc)})
        raise
    finally:
        stop()
        server_log.close()


if __name__ == "__main__":
    main()
