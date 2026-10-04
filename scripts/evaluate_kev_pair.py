#!/usr/bin/env python3
"""Prepare or execute a paired Kev 0.8B/27B pilot in one approved allocation."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from openwebrl import jev_eval as io, kev_eval as kev
import evaluate_jev_ultrafast as cohort

RUNTIME = cohort.RUNTIME
PREPARATION = RUNTIME / "kev-preparation-20261004"
VARIANTS = ("0.8b", "27b")


def cohort_args(args, variant):
    return argparse.Namespace(task_file=args.task_file, upstream=RUNTIME / "jev-ultrafast-upstream",
        output=args.output / variant, env_file=args.env_file, limit=10, workers=2,
        wall_budget_seconds=2800, browser="browser-use", text_provider="openai",
        decision_provider="kev", kev_variant=variant, kev_endpoint=args.endpoint)


def source_identity():
    upstream = RUNTIME / "kev-upstream"
    head = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if head != kev.UPSTREAM_COMMIT:
        raise ValueError("Kev upstream commit changed")
    if subprocess.check_output(["git", "-C", str(upstream), "diff", "HEAD", "--", "kev", "uv.lock"]):
        raise ValueError("Kev source differs from the reviewed upstream commit")
    paths = subprocess.check_output(["git", "-C", str(upstream), "ls-files", "kev"], text=True).splitlines()
    return dict(commit=head, files_sha256={p: kev.file_hash(upstream / p) for p in paths},
                lock_sha256=kev.file_hash(upstream / "uv.lock"))


def prepare(args):
    kev.local_endpoint(args.endpoint)
    args.output = args.output.resolve()
    if args.output.is_relative_to(REPO):
        raise ValueError("Private plans must stay outside the repository")
    baseline = json.loads((args.baseline / "plan.json").read_text())
    tasks = io.load_tasks(args.task_file, 10)
    if tasks != baseline["tasks"] or kev.file_hash(args.task_file) != baseline["task_file_sha256"]:
        raise ValueError("The paired cohort must match the completed Jev pilot exactly")
    model_audit = json.loads((PREPARATION / "verified-models.json").read_text())
    for variant in VARIANTS:
        audit = model_audit[variant]
        if audit["spec"] != kev.model_spec(variant) or not audit["verified"]:
            raise ValueError("Pinned Kev weights have not passed the independent file audit")
        for path, info in audit["files"].items():
            stat = Path(path).stat()
            if stat.st_size != info["size"] or stat.st_mtime_ns != info["mtime_ns"]:
                raise ValueError("Downloaded model files changed since their hash audit")
    dependencies = json.loads((PREPARATION / "environment.json").read_text())
    command = [str(RUNTIME / "kev-eval-venv/bin/python"), "-c",
        "import importlib.metadata as m,json,sys; print(json.dumps({p:m.version(p) for p in sys.argv[1:]}))",
        *dependencies["packages"]]
    if json.loads(subprocess.check_output(command, text=True)) != dependencies["packages"]:
        raise ValueError("Kev dependencies changed after the CPU preflight")
    plans = {v: cohort.prepare(cohort_args(args, v)) for v in VARIANTS}
    requests = []
    for path in sorted(args.baseline.glob("tasks/*/api-attempts.jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["provider"] == "jev":
                requests.append(dict(row["request"], model="kev-latest"))
    if not requests:
        raise ValueError("The baseline has no saved decision requests for warmup")
    requests.sort(key=lambda body: len(json.dumps(body)))
    warmup = [requests[0], requests[-1]]
    plan = dict(version=1, variants=list(VARIANTS), cohort_plan_sha256={v: io.digest(p) for v, p in plans.items()},
        resources=dict(gpus=1, gpu_type="H200", cpus=8, memory_gib=120, total_seconds=7200),
        controller_budget_seconds=6900, endpoint=args.endpoint, source=source_identity(),
        model_audit_sha256=io.digest(model_audit), dependencies=dependencies,
        warmup_sha256=io.digest(warmup), baseline_plan_sha256=io.digest(baseline),
        code_sha256={str(p.relative_to(REPO)): kev.file_hash(p) for p in (
            Path(__file__), REPO / "scripts/evaluate_kev_pair_1gpu.sbatch")},
        limits=dict(browser_sessions=20, browser_minutes=240, concurrent_browsers=2,
                    local_kev_requests=3604, text_http_attempts=3600, judge_http_attempts=80),
        execution_status="prepared_not_submitted", wandb_project="openwebrl-evals")
    path = args.output / "pair-plan.json"
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError("Prepared pair changed; use a new output directory")
    io.write_json(path, plan)
    io.write_json(args.output / "warmup-requests.json", warmup)
    return plan


def server_environment():
    # Pin every upstream serving override; do not inherit an unrelated experiment.
    env = {k: v for k, v in os.environ.items() if not k.startswith("KEV_")}
    env.update(HF_HOME=str(RUNTIME / "hf-cache"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               HF_XET_CACHE=str(RUNTIME / "hf-cache/xet"),
               TRITON_CACHE_DIR=str(PREPARATION / "triton-cache"),
               TORCHINDUCTOR_CACHE_DIR=str(PREPARATION / "inductor-cache"),
               KEV_BACKEND="torch", KEV_DTYPE="bf16", KEV_FUSED="1", KEV_CUDA_GRAPHS="1",
               KEV_MERGE="1", KEV_LORA_SCALE="1", KEV_TRUNCATE_STATES="0", KEV_DATE_FACTS="0",
               WANDB_PROJECT="openwebrl-evals", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1")
    return env


def stop_process(process, seconds):
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def check_port_available(endpoint):
    import socket
    from urllib.parse import urlsplit
    with socket.socket() as probe:
        # Match the server's bind policy: a closed prior listener may leave
        # TIME_WAIT connections. A live listener must still fail this check.
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", urlsplit(endpoint).port))


def run(args, plan):
    import httpx
    if not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Execution requires an explicitly approved H200 allocation")
    with (args.output / "pair-owner.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = args.output / "pair-time-ledger.json"
        ledger = json.loads(path.read_text()) if path.exists() else dict(attempts=[])
        if any(not a["finished"] for a in ledger["attempts"]):
            raise ValueError("Interrupted allocation: reconcile scheduler elapsed time before recovery")
        remaining = plan["controller_budget_seconds"] - sum(a["elapsed_seconds"] for a in ledger["attempts"])
        if remaining < 1200:
            raise ValueError("Approved pair budget is exhausted; never reset it for a retry")
        started = time.monotonic()
        attempt = dict(job_id=os.environ["SLURM_JOB_ID"], started_unix=time.time(), elapsed_seconds=0, finished=False)
        ledger["attempts"].append(attempt)
        io.write_json(path, ledger)
        stopped = False

        def stop_signal(*_):
            nonlocal stopped
            stopped = True

        previous = {s: signal.signal(s, stop_signal) for s in (signal.SIGTERM, signal.SIGINT)}

        def heartbeat(stage):
            elapsed = time.monotonic() - started
            attempt["elapsed_seconds"] = elapsed
            io.write_json(path, ledger)
            io.write_json(args.output / "heartbeat.json", dict(stage=stage, updated_unix=time.time(),
                          job_id=attempt["job_id"], elapsed_seconds=elapsed, remaining_seconds=remaining-elapsed))
            if stopped or elapsed >= remaining:
                raise TimeoutError("Allocation shutdown or approved pair budget reached")

        server = browser = None
        try:
            for variant in VARIANTS:
                root = args.output / variant
                summary = root / "summary.json"
                if summary.exists() and json.loads(summary.read_text())["verified_complete"]:
                    continue
                if any(root.glob("tasks/*/task.json")):
                    raise ValueError("Partial cohort needs diagnosis and a preserved separate retry")
                heartbeat("starting_" + variant)
                # Refuse an occupied endpoint rather than connecting to somebody
                # else's model, even if it advertises the same checkpoint.
                from urllib.parse import urlsplit
                check_port_available(args.endpoint)
                spec = kev.model_spec(variant)
                command = [str(RUNTIME / "kev-eval-venv/bin/python"), "-m", "kev.serve", "--run", spec["server_run"],
                           "--host", "127.0.0.1", "--port", str(urlsplit(args.endpoint).port)]
                with (root / "server.log").open("a") as log:
                    server = subprocess.Popen(command, cwd=RUNTIME / "kev-upstream", env=server_environment(), stdout=log, stderr=log)
                    try:
                        deadline = time.monotonic() + 600
                        while True:
                            heartbeat("loading_" + variant)
                            if server.poll() is not None:
                                raise RuntimeError("Kev server exited before readiness; inspect server.log")
                            try:
                                card = kev.check_server(args.endpoint, spec)
                                break
                            except httpx.TransportError:
                                if time.monotonic() >= deadline:
                                    raise TimeoutError("Kev server startup exceeded 600 seconds")
                                time.sleep(5)
                        io.write_json(root / "server-identity.json", card)
                        warmup = json.loads((args.output / "warmup-requests.json").read_text())
                        if (root / "warmup-attempts.jsonl").exists():
                            raise ValueError("Warmup already attempted: preserve request budget before retry")
                        for request in warmup:
                            heartbeat("warmup_" + variant)
                            io.append_json(root / "warmup-attempts.jsonl", dict(request_sha256=io.digest(request), started_unix=time.time()))
                            response = httpx.post(args.endpoint, json=request, timeout=240, trust_env=False)
                            response.raise_for_status()
                            kev.validate_answers(request, response.json())
                            io.append_json(root / "warmup-responses.jsonl", response.json())
                        io.write_json(root / "server-warm.json", kev.check_server(args.endpoint, spec))
                        ca = cohort_args(args, variant)
                        command = [sys.executable, str(REPO / "scripts/evaluate_jev_ultrafast.py"), "--execute",
                                   "--decision-provider", "kev", "--kev-variant", variant,
                                   "--kev-endpoint", args.endpoint, "--output", str(root),
                                   "--task-file", str(ca.task_file), "--env-file", str(args.env_file),
                                   "--wall-budget-seconds", str(ca.wall_budget_seconds)]
                        with (root / "controller.log").open("a") as browser_log:
                            browser = subprocess.Popen(command, cwd=REPO, stdout=browser_log, stderr=browser_log)
                            while browser.poll() is None:
                                heartbeat("browsers_" + variant)
                                if server.poll() is not None:
                                    raise RuntimeError("Kev server stopped during the cohort")
                                time.sleep(5)
                            if browser.returncode:
                                raise RuntimeError("Kev browser cohort failed; inspect preserved artifacts")
                        io.write_json(root / "server-final.json", kev.check_server(args.endpoint, spec))
                    finally:
                        stop_process(browser, 150)
                        stop_process(server, 20)
                        browser = server = None
            summaries = {v: json.loads((args.output / v / "summary.json").read_text()) for v in VARIANTS}
            completed = all(s["verified_complete"] for s in summaries.values())
            io.write_json(args.output / "summary.json", dict(verified_complete=completed, variants=summaries))
            return 0 if completed else 2
        except Exception as exc:
            io.write_json(args.output / "failure.json", dict(error_type=type(exc).__name__, error=str(exc), time=time.time()))
            raise
        finally:
            stop_process(browser, 150)
            stop_process(server, 20)
            attempt.update(elapsed_seconds=time.monotonic()-started, finished=True)
            io.write_json(path, ledger)
            for sig, handler in previous.items():
                signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RUNTIME / "evaluations/kev-pair-om2w-pilot-20261004")
    parser.add_argument("--baseline", type=Path, default=RUNTIME / "evaluations/jev-ultrafast-om2w-pilot-20261004")
    parser.add_argument("--task-file", type=Path, default=REPO / "openwebrl/data/eval/online-mind2web.jsonl")
    parser.add_argument("--env-file", type=Path, default=REPO / ".env")
    parser.add_argument("--endpoint", default="http://127.0.0.1:18761/v1/systemone")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan = prepare(args)
    if args.execute:
        return run(args, plan)
    print(json.dumps(plan, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
