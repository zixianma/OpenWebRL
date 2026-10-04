#!/usr/bin/env python3
"""Prepare or run a resumable Jev Ultrafast Online-Mind2Web cohort (no GPU)."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from openwebrl import jev_eval as evaluation

RUNTIME = Path("/gpfs/scrubbed/zixianma/openwebrl-runtime")


def configuration(args):
    from openwebrl import jev_harness
    mercury = args.text_provider == "openrouter"
    config = dict(version=1, browser=args.browser, jev_model="jev-1.13.0",
                harness_revision=getattr(args, "harness_revision", "upstream-v1"),
                text_provider=args.text_provider,
                text_model="inception/mercury-2.5" if mercury else "gpt-4.1-mini-2025-04-14",
                text_base_url="https://openrouter.ai/api/v1" if mercury else "https://api.openai.com/v1",
                text_sampling="upstream provider defaults; reasoning disabled" if mercury else dict(temperature=0.6, top_p=0.95),
                text_max_tokens=1024, judge_model="o4-mini", judge_prompt_variant="agenttrek",
                judge_max_completion_tokens=4096,
                judge_base_url="https://api.openai.com/v1", judge_seed=42,
                max_steps=30, max_decisions=60, task_timeout_seconds=600,
                viewport=dict(width=1120, height=780), actor_input="DOM text; no screenshots",
                actor_sampling="argmax operation/target probabilities; T/top-p/top-k do not apply",
                wandb_project="openwebrl-evals", upstream=evaluation.source_identity(args.upstream),
                judge_prompt_sha256=evaluation.digest(evaluation.judge_protocol()),
                code_sha256={str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in (Path(__file__), Path(evaluation.__file__), Path(jev_harness.__file__))})
    if getattr(args, "decision_provider", "jev") == "kev":
        from openwebrl import kev_eval
        config.update(decision_provider="kev", jev_model="kev-latest",
                      decision_timeout_seconds=120,
                      decision_endpoint=kev_eval.local_endpoint(args.kev_endpoint),
                      kev=kev_eval.model_spec(args.kev_variant))
        path = Path(kev_eval.__file__)
        config["code_sha256"][str(path.relative_to(REPO))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return config


def prepare(args):
    tasks = evaluation.load_tasks(args.task_file, args.limit)
    config = configuration(args)
    output = args.output.resolve()
    if output.is_relative_to(REPO):
        raise ValueError("Keep task payloads and trajectories outside the repository")
    plan = dict(config=config, tasks=tasks, task_file_sha256=hashlib.sha256(args.task_file.read_bytes()).hexdigest(),
                workers=args.workers, wall_budget_seconds=args.wall_budget_seconds,
                proposed_resources=dict(gpus=0, cpus=max(4, 2 * args.workers),
                                        memory_gib=max(8, 4 * args.workers),
                                        hours=math.ceil((args.wall_budget_seconds + 300) / 3600)),
                limits=dict(browser_sessions=len(tasks), remote_browser_minutes=12 * len(tasks),
                            jev_http_attempts=180 * len(tasks), text_http_attempts=180 * len(tasks),
                            judge_http_attempts=4 * len(tasks)),
                approval_status="not_submitted", training_updates=0)
    if config.get("decision_provider") == "kev":
        plan["limits"]["kev_http_attempts"] = plan["limits"].pop("jev_http_attempts")
        plan["proposed_resources"] = dict(shared_pair_allocation=True, gpus=1, cpus=8, memory_gib=120, hours=2)
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "plan.json"
    if manifest.exists() and json.loads(manifest.read_text()) != plan:
        raise ValueError("Existing output has a different plan; use a new output directory")
    if not manifest.exists():
        evaluation.write_json(manifest, plan)
    return plan


def check_worker_source(config):
    for relative, expected in config["code_sha256"].items():
        if hashlib.sha256((REPO / relative).read_bytes()).hexdigest() != expected:
            raise ValueError("Evaluation code changed after preparation: " + relative)
    if evaluation.digest(evaluation.judge_protocol()) != config["judge_prompt_sha256"]:
        raise ValueError("Canonical judge prompts changed after preparation")
    installed = Path(evaluation.importlib.metadata.distribution("jev-ultrafast").locate_file("jev_ultrafast"))
    for name, expected in config["upstream"]["files_sha256"].items():
        if hashlib.sha256((installed / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Installed Jev source changed after preparation: " + name)


def worker(args):
    plan = json.loads((args.output / "plan.json").read_text())
    config = plan["config"]
    check_worker_source(config)
    evaluation.load_credentials(args.env_file, config)
    if config.get("decision_provider") == "kev":
        from openwebrl.kev_eval import check_server
        check_server(config["decision_endpoint"], config["kev"])
    task = next(t for t in plan["tasks"] if evaluation.digest(t["task_id"]) == args.worker)
    root = args.output / "tasks" / args.worker
    # Completed attempts are immutable. Interrupted attempts require a separate retry cohort.
    if (root / "task.json").exists():
        raise ValueError("Attempt already exists; preserve it and use a separate retry cohort")
    result = evaluation.run_task(task, root, config)
    return 3 if result["provider_blocked"] else 0


def stop_owned_worker(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def execute(args, plan):
    check_worker_source(plan["config"])
    evaluation.load_credentials(args.env_file, plan["config"])
    with (args.output / "owner.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger_path = args.output / "time-ledger.json"
        ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else dict(attempts=[])
        # Persisted heartbeats also charge interrupted invocations on restart.
        for attempt in ledger["attempts"]:
            if not attempt.get("finished"):
                attempt["elapsed_seconds"] = max(attempt.get("elapsed_seconds", 0),
                    time.time() - attempt["started_unix"])
        used = sum(a["elapsed_seconds"] for a in ledger["attempts"])
        remaining = plan["wall_budget_seconds"] - used
        if remaining <= 0:
            raise ValueError("Cohort wall budget exhausted; no automatic budget reset")
        attempt = dict(started_unix=time.time(), elapsed_seconds=0, finished=False,
                       slurm_job_id=os.environ.get("SLURM_JOB_ID"))
        ledger["attempts"].append(attempt)
        evaluation.write_json(ledger_path, ledger)
        started = time.monotonic()
        pending = []
        previous_provider_block = False
        for task in plan["tasks"]:
            root = args.output / "tasks" / evaluation.digest(task["task_id"])
            if (root / "result.json").exists():
                previous_provider_block |= json.loads((root / "result.json").read_text())["provider_blocked"]
                continue
            if (root / "task.json").exists():
                raise ValueError("Interrupted task exists; diagnose and preserve it before a separate retry")
            pending.append(task)
        if previous_provider_block:
            raise ValueError("Previous provider failure needs diagnosis; preserve this cohort before retrying")
        active, failures = {}, []
        stop, shutdown_at = False, None

        def interrupted(*_):
            nonlocal stop, shutdown_at
            stop = True
            shutdown_at = time.monotonic() + 120

        previous = {s: signal.signal(s, interrupted) for s in (signal.SIGTERM, signal.SIGINT)}
        try:
            while pending or active:
                elapsed = time.monotonic() - started
                # Reserve enough time to finish one worst-case task and clean up.
                reserve = plan["config"]["task_timeout_seconds"] + 4 * 120 + 120
                if elapsed + reserve >= remaining:
                    stop = True
                while pending and len(active) < plan["workers"] and not stop:
                    task = pending.pop(0)
                    task_hash = evaluation.digest(task["task_id"])
                    root = args.output / "tasks" / task_hash
                    root.mkdir(parents=True, exist_ok=True)
                    log = (root / "worker.log").open("a")
                    command = [sys.executable, str(Path(__file__)), "--output", str(args.output),
                               "--env-file", str(args.env_file), "--worker", task_hash]
                    process = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=log,
                                               start_new_session=True)
                    active[process.pid] = (process, log, root, time.monotonic())
                for pid, (process, log, root, task_started) in list(active.items()):
                    if time.monotonic() - task_started > reserve or (shutdown_at is not None and time.monotonic() > shutdown_at):
                        stop_owned_worker(process)
                    code = process.poll()
                    if code is not None:
                        log.close()
                        active.pop(pid)
                        cleanup_errors = evaluation.stop_recorded_session(root)
                        if cleanup_errors:
                            failures.append(dict(task_directory=root.name, cleanup_errors=cleanup_errors))
                            stop = True
                        if code != 0:
                            failures.append(dict(task_directory=root.name, exit_code=code))
                            stop = True
                summary = evaluation.summarize(plan["tasks"], args.output, plan["config"].get("decision_provider", "jev"))
                summary.update(active_workers=len(active), pending_tasks=len(pending), failures=failures,
                               state="running" if active or (pending and not stop) else "complete" if summary["verified_complete"] else "partial")
                evaluation.write_json(args.output / "summary.json", summary)
                attempt["elapsed_seconds"] = time.monotonic() - started
                evaluation.write_json(ledger_path, ledger)
                if not active and (stop or not pending):
                    break
                time.sleep(5)
        finally:
            for process, log, root, _ in active.values():
                stop_owned_worker(process)
                log.close()
                evaluation.stop_recorded_session(root)
            for s, handler in previous.items():
                signal.signal(s, handler)
            attempt.update(elapsed_seconds=time.monotonic() - started, finished=True)
            evaluation.write_json(ledger_path, ledger)
        print(json.dumps(summary, sort_keys=True))
        return 0 if summary["verified_complete"] and not failures else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-file", type=Path, default=REPO / "openwebrl/data/eval/online-mind2web.jsonl")
    parser.add_argument("--upstream", type=Path, default=RUNTIME / "jev-ultrafast-upstream")
    parser.add_argument("--output", type=Path, default=RUNTIME / "evaluations/jev-ultrafast-om2w-pilot-20261004")
    parser.add_argument("--env-file", type=Path, default=REPO / ".env")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--wall-budget-seconds", type=int, default=6900)
    parser.add_argument("--browser", choices=("browser-use", "local"), default="browser-use")
    parser.add_argument("--text-provider", choices=("openai", "openrouter"), default="openai")
    parser.add_argument("--decision-provider", choices=("jev", "kev"), default="jev")
    parser.add_argument("--harness-revision", choices=("upstream-v1", "actionable-v2"), default="upstream-v1")
    parser.add_argument("--kev-variant", choices=("0.8b", "27b"))
    parser.add_argument("--kev-endpoint", default="http://127.0.0.1:18761/v1/systemone")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return worker(args)
    if args.decision_provider == "kev" and not args.kev_variant:
        parser.error("--kev-variant is required for the Kev decision provider")
    if args.workers < 1 or args.wall_budget_seconds < 1200:
        parser.error("Positive worker count and at least 1200 seconds are required")
    plan = prepare(args)
    if not args.execute:
        print(json.dumps({k: v for k, v in plan.items() if k != "tasks"}, indent=2))
        print(f"Prepared {len(plan['tasks'])} tasks; no browser/model calls or allocation submission.")
        return 0
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Run the evaluation inside an explicitly approved allocation")
    return execute(args, plan)


if __name__ == "__main__":
    raise SystemExit(main())
