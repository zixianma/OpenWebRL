#!/usr/bin/env python3
"""Retry only the first-pass unavailable tasks for one C2 checkpoint."""

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


def get_json(url):
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--original-output", required=True, type=Path)
    parser.add_argument("--sample", required=True, type=Path)
    parser.add_argument("--actor-port", required=True, type=int)
    parser.add_argument("--browser-port-start", required=True, type=int)
    parser.add_argument("--browser-port-end", required=True, type=int)
    parser.add_argument("--parallel", type=int, default=3)
    parser.add_argument("--mem-fraction-static", type=float, default=0.25)
    args = parser.parse_args()
    if os.getenv("SLURM_JOB_ID") != args.job_id or f"/job_{args.job_id}/" not in Path("/proc/self/cgroup").read_text():
        raise ValueError("Retry must run inside the authorized allocation")
    record = parse_job_record(subprocess.check_output(
        ["scontrol", "show", "job", args.job_id, "-o"], text=True, timeout=20))
    deadline = allocation_deadline(record, margin_minutes=2)
    sample = json.loads(args.sample.read_text())
    task_to_index = dict(zip(sample["task_ids"], sample["indices"], strict=True))
    originals = [json.loads(path.read_text()) for path in (args.original_output / "results").glob("*.json")]
    unavailable = sorted(row["task_id"] for row in originals if not row.get("valid"))
    indices = [task_to_index[task_id] for task_id in unavailable]
    root = args.run_root.resolve()
    work = root / "evaluation/checkpoint-scaling-100/unavailable-retries" / args.label
    output = work / "online-mind2web"
    status = work / "status.json"
    work.mkdir(parents=True, exist_ok=True)
    write_json(work / "retry-manifest.json", {
        "label": args.label, "source": str(args.original_output.resolve()),
        "model": str(args.model.resolve()), "original_unavailable_task_ids": unavailable,
        "task_indices": indices, "scheduled": len(indices), "aggregation": "fill_original_unavailable_only",
        "judge": "o4-mini", "judge_protocol": "online_mind2web/AgentTrek",
    })
    server_log = (work / "actor-server.log").open("a")
    server = subprocess.Popen(
        [str(SERVER_PYTHON), "-m", "sglang.launch_server", "--model-path", str(args.model.resolve()),
         "--host", "127.0.0.1", "--port", str(args.actor_port), "--dtype", "bfloat16", "--tp", "1",
         "--mem-fraction-static", str(args.mem_fraction_static), "--context-length", "32768",
         "--max-running-requests", "12", "--chunked-prefill-size", "4096", "--disable-cuda-graph"],
        cwd=REPO, stdout=server_log, stderr=subprocess.STDOUT, start_new_session=True)

    def stop(*_ignored):
        if server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        write_json(status, {"phase": "starting_server", "updated_utc": datetime.now(timezone.utc).isoformat(),
                            "scheduled": len(indices), "allocation": args.job_id})
        startup = min(deadline.timestamp(), time.time() + 600)
        while True:
            if server.poll() is not None:
                raise RuntimeError("Retry actor server exited during startup")
            try:
                info = get_json(f"http://127.0.0.1:{args.actor_port}/get_model_info")
                with urllib.request.urlopen(f"http://127.0.0.1:{args.actor_port}/health_generate", timeout=10) as response:
                    response.read()
                if os.path.realpath(info.get("model_path", "")) != os.path.realpath(args.model):
                    raise ValueError("Retry actor server model mismatch")
                break
            except (OSError, ValueError):
                if time.time() >= startup:
                    raise TimeoutError("Retry actor startup timed out")
                time.sleep(2)
        write_json(status, {"phase": "evaluating", "updated_utc": datetime.now(timezone.utc).isoformat(),
                            "scheduled": len(indices), "allocation": args.job_id})
        remaining = max(1, int((deadline - datetime.now(timezone.utc)).total_seconds()))
        command = ["timeout", "--signal=TERM", "--kill-after=60", str(remaining), str(ARM_PYTHON),
                   "-m", "openwebrl.arm_eval", "--mode", "baseline", "--actor", str(args.model.resolve()),
                   "--actor-port", str(args.actor_port), "--output", str(output), "--parallel", str(args.parallel),
                   "--task-indices", ",".join(map(str, indices)), "--seed", "42", "--temperature", "0.7",
                   "--top-p", "0.9", "--max-new-tokens", "1024", "--max-steps", "30",
                   "--task-timeout", "1800", "--judge-model", "o4-mini", "--env-file", ".env",
                   "--browser-port-start", str(args.browser_port_start), "--browser-port-end", str(args.browser_port_end)]
        with (work / "eval.log").open("a") as log:
            code = subprocess.call(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
        count = len(list((output / "results").glob("*.json"))) if (output / "results").exists() else 0
        summary_path = output / "summary.json"
        write_json(status, {"phase": "complete" if code == 0 and summary_path.exists() else "paused",
                            "updated_utc": datetime.now(timezone.utc).isoformat(), "scheduled": len(indices),
                            "completed": count, "returncode": code,
                            "summary": json.loads(summary_path.read_text()) if summary_path.exists() else None})
    except Exception as exc:
        write_json(status, {"phase": "failed", "updated_utc": datetime.now(timezone.utc).isoformat(),
                            "scheduled": len(indices), "error_type": type(exc).__name__,
                            "error": str(exc)})
        raise
    finally:
        stop()
        server_log.close()


if __name__ == "__main__":
    main()
