#!/usr/bin/env python3
"""Wait inside an existing Slurm allocation, smoke-test ARM, then run the benchmark."""
import argparse
import csv
from datetime import datetime, timezone
import fcntl
import getpass
import json
import math
import os
import re
from pathlib import Path
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
TERMINAL = {"complete", "smoke_failed", "run_failed", "stopped_allocation_ended", "stopped"}
ACTIVE = None


def command(argv):
    return subprocess.check_output(argv, text=True, stderr=subprocess.STDOUT, timeout=20).strip()


def allocation(job_id):
    info = command(["scontrol", "show", "job", str(job_id), "-o"])
    if f"UserId={getpass.getuser()}(" not in info or "JobState=RUNNING" not in info:
        raise RuntimeError("The specified allocation is no longer your running job")
    return info


def gpu_status(expected_count=2):
    rows = list(csv.reader(command([
        "nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"
    ]).splitlines()))
    if len(rows) != expected_count:
        raise RuntimeError(f"Expected {expected_count} assigned GPUs; found {len(rows)}")
    devices = []
    for index, uuid, memory in rows:
        # A failed/ambiguous query raises; it never counts as an idle GPU.
        processes = command([
            "nvidia-smi", "-i", index.strip(), "--query-compute-apps=pid", "--format=csv,noheader"
        ])
        devices.append({"index": int(index), "uuid": uuid.strip(), "memory_MiB": int(memory),
                        "processes": processes.splitlines() if processes else []})
    return devices


def smoke_gate(root):
    """Require real judged rollouts and working selectors, not task success."""
    checks = {}
    for mode in ("baseline", "scalar", "selection"):
        summary = json.loads((root / mode / "summary.json").read_text())
        traces = [json.loads(line)
                  for path in (root / mode / "selections").glob("*.jsonl")
                  for line in path.read_text().splitlines() if line.strip()]
        if summary["attempted"] != summary["scheduled"] or summary["valid"] < 1:
            raise ValueError(f"{mode}: smoke did not finish with at least one valid judged task")
        if not traces:
            raise ValueError(f"{mode}: no executed-action traces")
        expected = 1 if mode == "baseline" else 5
        for row in traces:
            if len(row["candidates"]) != expected or not 0 <= row["selected_index"] < expected:
                raise ValueError(f"{mode}: invalid candidate/selection trace")
            if mode == "scalar":
                scores = row["scores"]
                if len(scores) != 5 or not all(math.isfinite(x) for x in scores):
                    raise ValueError("scalar: missing/nonfinite scores")
            if mode == "selection" and row.get("fallback"):
                raise ValueError("selection: malformed output fallback in smoke")
        checks[mode] = {"valid_tasks": summary["valid"], "scheduled": summary["scheduled"],
                        "action_traces": len(traces)}
    return checks


def record(args, phase, **detail):
    path = args.output / "watcher-status.json"
    previous = json.loads(path.read_text()) if path.exists() else {}
    if phase == "starting_allocation_step":
        for key in ("reason", "failed_phase", "returncode", "devices", "consecutive_free_checks",
                    "phase_output", "smoke_checks", "comparison"):
            previous.pop(key, None)
    result = dict(previous, phase=phase, allocation=args.job_id,
                  updated_utc=datetime.now(timezone.utc).isoformat(), **detail)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(result, indent=2) + "\n")
    temp.replace(path)
    if previous.get("phase") != phase:
        print(json.dumps(result), flush=True)
        with (args.output / "watcher-events.jsonl").open("a") as stream:
            stream.write(json.dumps(result) + "\n")


def interrupted(signum, _frame):
    raise InterruptedError(f"Received signal {signum}")


def run_child(argv, logfile, env=None):
    global ACTIVE
    with logfile.open("a") as log:
        ACTIVE = subprocess.Popen(argv, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                                  stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return ACTIVE.wait()
        finally:
            if ACTIVE.poll() is None:
                # Only this watcher's process group; never existing training.
                os.killpg(ACTIVE.pid, signal.SIGTERM)
                try:
                    ACTIVE.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(ACTIVE.pid, signal.SIGKILL)
                    ACTIVE.wait()
            ACTIVE = None


def wait_free(args):
    consecutive = 0
    while True:
        allocation(args.job_id)
        devices = gpu_status(args.gpu_count)
        free = all(not d["processes"] and d["memory_MiB"] < 1024 for d in devices)
        consecutive = consecutive + 1 if free else 0
        record(args, "waiting_for_free_gpus", devices=devices, consecutive_free_checks=consecutive)
        if consecutive >= 2:
            return
        time.sleep(args.poll_seconds)


def worker(args):
    if os.environ.get("SLURM_JOB_ID") != args.job_id or not os.environ.get("SLURM_STEP_ID"):
        raise RuntimeError("Worker must run inside the specified Slurm allocation step")
    record(args, "waiting_for_free_gpus", worker_pid=os.getpid(),
           slurm_step=os.environ["SLURM_STEP_ID"], node=command(["hostname", "-s"]))
    for phase, indices, parallel in (("smoke", args.smoke_indices, args.smoke_parallel),
                                      ("full", "", args.parallel)):
        if phase == "smoke" and args.reuse_completed_smoke:
            checks = smoke_gate(args.output / "smoke")
            record(args, "smoke_passed", smoke_checks=checks, smoke_reused=True)
            continue
        wait_free(args)
        target = args.output / phase
        target.mkdir(exist_ok=True)
        record(args, f"running_{phase}", phase_output=str(target))
        env = dict(os.environ, ARM_JOB_ID=args.job_id, OUTPUT_ROOT=str(target),
                   TASK_INDICES=indices, N_PARALLEL=str(parallel), EVAL_SEED="42",
                   ARM_NUM_GPUS=str(args.gpu_count), ACTOR_MEMORY_FRACTION="0.4")
        code = run_child(["bash", "scripts/run_arm_reproduction.sh"],
                         args.output / f"{phase}-launcher.log", env)
        if code:
            record(args, "run_failed", failed_phase=phase, returncode=code)
            return code
        if phase == "smoke":
            try:
                checks = smoke_gate(target)
            except Exception as exc:
                record(args, "smoke_failed", reason=str(exc))
                return 1
            record(args, "smoke_passed", smoke_checks=checks)
    record(args, "complete", comparison=str(args.output / "full/comparison.json"))
    return 0


def validate_supervisor_allocation(job_id):
    # A supervisor inside another job would die when that job expires, even
    # when its srun child targets the correct GPUs.
    cgroup = Path("/proc/self/cgroup").read_text()
    match = re.search(r"/job_(\d+)(?:/|$)", cgroup)
    if match and match.group(1) != job_id:
        raise RuntimeError("Start the supervisor on a login node or inside the dedicated allocation")
    return cgroup.strip()


def supervise(args):
    supervisor_cgroup = validate_supervisor_allocation(args.job_id)
    info = allocation(args.job_id)
    record(args, "starting_allocation_step", supervisor_pid=os.getpid(), allocation_info=info,
           supervisor_cgroup=supervisor_cgroup,
           smoke_task_indices=args.smoke_indices, smoke_parallel=args.smoke_parallel,
           full_tasks_per_arm=300, full_parallel=args.parallel, gpu_count=args.gpu_count,
           dedicated_allocation=args.dedicated_allocation,
           allocation_policy="Existing allocation only; stop on expiry; never stop other workloads")
    argv = ["srun", f"--jobid={args.job_id}", "--overlap", f"--gres=gpu:h200:{args.gpu_count}", "--ntasks=1",
            "--immediate=30", sys.executable, str(Path(__file__).resolve()), "--worker",
            "--job-id", args.job_id, "--output", str(args.output),
            "--poll-seconds", str(args.poll_seconds), "--smoke-indices", args.smoke_indices,
            "--smoke-parallel", str(args.smoke_parallel), "--parallel", str(args.parallel),
            "--gpu-count", str(args.gpu_count), "--dedicated-allocation"]
    if args.reuse_completed_smoke:
        argv.append("--reuse-completed-smoke")
    # The invoking shell may belong to a different training allocation. Let
    # srun populate every allocation/device variable for the selected job.
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("SLURM_", "SBATCH_", "SRUN_")) and k != "CUDA_VISIBLE_DEVICES"}
    code = run_child(argv, args.output / "allocation-step.log", env)
    current = json.loads((args.output / "watcher-status.json").read_text())
    if code or current["phase"] != "complete":
        try:
            allocation(args.job_id)
        except Exception:
            record(args, "stopped_allocation_ended", returncode=code)
        else:
            if current["phase"] not in TERMINAL:
                record(args, "run_failed", returncode=code, reason="Allocation step exited")
    return code


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--job-id", required=True)
    ap.add_argument("--gpu-count", type=int, choices=(1, 2), default=2)
    ap.add_argument("--dedicated-allocation", action="store_true",
                    help="The user explicitly assigned this allocation to ARM evaluation")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--poll-seconds", type=int, default=30)
    ap.add_argument("--smoke-indices", default="0,50,100")
    ap.add_argument("--smoke-parallel", type=int, default=3)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--reuse-completed-smoke", action="store_true",
                    help="Revalidate and reuse a completed smoke without changing its protocol")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if not args.dedicated_allocation:
        ap.error("Require an allocation explicitly assigned to ARM; empty GPUs alone are insufficient")
    if not args.job_id.isdigit() or min(args.poll_seconds, args.smoke_parallel, args.parallel) < 1:
        ap.error("Expected numeric job ID and positive polling interval/concurrency")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    lock = None
    if not args.worker:
        # Prevent duplicate watchers even when their output directories differ.
        lock_path = Path("/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction") / f"watch-{args.job_id}.lock"
        lock = lock_path.open("a")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        return worker(args) if args.worker else supervise(args)
    except InterruptedError as exc:
        record(args, "stopped", reason=str(exc))
        return 130
    except Exception as exc:
        record(args, "run_failed", error_type=type(exc).__name__, reason=str(exc))
        return 1
    finally:
        if lock:
            lock.close()


if __name__ == "__main__":
    sys.exit(main())
