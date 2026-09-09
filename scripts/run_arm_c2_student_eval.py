#!/usr/bin/env python3
"""Hand a completed C2 SFT checkpoint directly to matched Online-Mind2Web eval."""

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request

from scripts.resume_arm_c2_training import allocation_deadline, parse_job_record, write_json


REPO = Path(__file__).resolve().parents[1]
RUNTIME = Path("/gpfs/scrubbed/zixianma/openwebrl-runtime")
ARM_RUNTIME = RUNTIME / "arm-reproduction"
ARM_PYTHON = ARM_RUNTIME / "venv/bin/python"
SERVER_PYTHON = RUNTIME / "venv/bin/python"


def job_record(job_id):
    if os.getenv("SLURM_JOB_ID") != job_id or f"/job_{job_id}/" not in Path("/proc/self/cgroup").read_text():
        raise ValueError("Evaluation handoff must run inside its authorized allocation")
    raw = subprocess.check_output(["scontrol", "show", "job", job_id, "-o"], text=True, timeout=20)
    record = parse_job_record(raw)
    if record.get("JobState") != "RUNNING":
        raise ValueError("Allocation is not running")
    return record


def visible_gpu():
    devices = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True, timeout=20
    ).strip().splitlines()
    if len(devices) != 1:
        raise ValueError("Student evaluation requires exactly one visible GPU")
    return devices[0]


def gpu_processes(uuid):
    return subprocess.check_output(
        ["nvidia-smi", "--id=" + uuid, "--query-compute-apps=pid", "--format=csv,noheader"],
        text=True,
        timeout=20,
    ).strip().splitlines()


def fetch_json(url):
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.load(response)


def reachable(url):
    with urllib.request.urlopen(url, timeout=10) as response:
        response.read()


def record(run_root, job_id, phase, **extra):
    value = {
        "phase": phase,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "allocation": job_id,
        **extra,
    }
    write_json(run_root / "evaluation" / "status.json", value)
    print(phase, json.dumps(extra), flush=True)


def run_eval(run_root, actor, output, parallel, task_indices, deadline, log_name):
    remaining = int((deadline - datetime.now(timezone.utc)).total_seconds())
    if remaining < 300:
        return 124
    command = [
        "timeout", "--signal=TERM", "--kill-after=120", str(remaining),
        str(ARM_PYTHON), "-m", "openwebrl.arm_eval",
        "--mode", "baseline", "--actor", str(actor), "--output", str(output),
        "--parallel", str(parallel), "--task-indices", task_indices,
        "--seed", "42", "--temperature", "0.7", "--top-p", "0.9",
        "--max-new-tokens", "1024", "--max-steps", "30",
        "--task-timeout", "1800", "--judge-model", "o4-mini", "--env-file", ".env",
    ]
    with (run_root / "evaluation" / log_name).open("a") as log:
        return subprocess.call(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--wait", action="store_true", help="Wait for student/complete.json")
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    allocation = job_record(args.job_id)
    deadline = allocation_deadline(allocation, margin_minutes=5)
    gpu = visible_gpu()
    evaluation = run_root / "evaluation"
    evaluation.mkdir(exist_ok=True)
    record(run_root, args.job_id, "waiting_for_epoch_2", gpu_uuid=gpu, deadline_utc=deadline.isoformat())
    completion = run_root / "student" / "complete.json"
    while not completion.exists():
        if not args.wait:
            raise ValueError("Training is incomplete; pass --wait for automatic handoff")
        if datetime.now(timezone.utc) >= deadline - timedelta(minutes=5):
            record(run_root, args.job_id, "training_incomplete_at_handoff_deadline")
            return
        time.sleep(15)
    while gpu_processes(gpu):
        if datetime.now(timezone.utc) >= deadline - timedelta(minutes=5):
            raise TimeoutError("GPU did not become free after epoch-2 completion")
        time.sleep(5)

    merged = run_root / "student" / "epoch-2-merged"
    record(run_root, args.job_id, "merging_epoch_2", checkpoint=json.loads(completion.read_text())["checkpoint"])
    merge_log = evaluation / "merge.log"
    with merge_log.open("a") as log:
        subprocess.run(
            [str(ARM_PYTHON), str(REPO / "scripts/merge_arm_c2_student.py"),
             "--run-root", str(run_root), "--output", str(merged)],
            cwd=REPO,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    if gpu_processes(gpu):
        raise ValueError("Merge process exited but did not release the GPU")

    server_log = (evaluation / "actor-server.log").open("a")
    server = subprocess.Popen(
        [
            str(SERVER_PYTHON), "-m", "sglang.launch_server", "--model-path", str(merged),
            "--host", "127.0.0.1", "--port", "19100", "--dtype", "bfloat16", "--tp", "1",
            "--mem-fraction-static", "0.4", "--context-length", "32768",
            "--max-running-requests", "24", "--chunked-prefill-size", "4096", "--disable-cuda-graph",
        ],
        cwd=REPO,
        stdout=server_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    def stop_server(*_args):
        if server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait()

    signal.signal(signal.SIGINT, stop_server)
    signal.signal(signal.SIGTERM, stop_server)
    try:
        startup_deadline = min(deadline.timestamp(), time.time() + 900)
        while True:
            if server.poll() is not None:
                raise RuntimeError("Student actor server exited during startup")
            try:
                info = fetch_json("http://127.0.0.1:19100/get_model_info")
                reachable("http://127.0.0.1:19100/health_generate")
                if os.path.realpath(info.get("model_path", "")) != os.path.realpath(merged):
                    raise ValueError("SGLang loaded a different actor path")
                break
            except (OSError, ValueError):
                if time.time() >= startup_deadline:
                    raise TimeoutError("Student actor server startup timed out")
                time.sleep(2)

        smoke = evaluation / "c2-student-smoke"
        record(run_root, args.job_id, "student_eval_smoke", output=str(smoke))
        code = run_eval(run_root, merged, smoke, 3, "0,50,100", deadline, "smoke-eval.log")
        if code:
            raise RuntimeError(f"Student evaluation smoke exited {code}")
        summary = json.loads((smoke / "summary.json").read_text())
        if summary["attempted"] != 3 or summary["valid"] < 1:
            raise ValueError("Student evaluation smoke did not produce a usable judged trajectory")

        full = evaluation / "c2-student-online-mind2web-300"
        record(run_root, args.job_id, "student_evaluation", output=str(full), parallel=8)
        code = run_eval(run_root, merged, full, 8, "", deadline, "full-eval.log")
        if code == 124:
            count = len(list((full / "results").glob("*.json"))) if (full / "results").exists() else 0
            record(run_root, args.job_id, "student_evaluation_paused", completed=count, scheduled=300)
            return
        if code:
            raise RuntimeError(f"Full student evaluation exited {code}")
        summary = json.loads((full / "summary.json").read_text())
        record(run_root, args.job_id, "student_evaluation_complete", summary=summary)
    finally:
        stop_server()
        server_log.close()


if __name__ == "__main__":
    main()
