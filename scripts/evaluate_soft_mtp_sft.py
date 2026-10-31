"""One actor-only Online-Mind2Web cohort; owns its server and all 300 tasks."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from urllib.request import urlopen


RUNTIME = Path("/gpfs/scrubbed/zixianma/openwebrl-runtime")
PYTHON = RUNTIME / "venv/bin/python"


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def wait_for_actor(process: subprocess.Popen, checkpoint: Path, port: int) -> None:
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"SGLang actor exited: {process.returncode}")
        try:
            with urlopen(f"http://127.0.0.1:{port}/get_model_info", timeout=3) as response:
                info = json.load(response)
            if Path(info["model_path"]).resolve() != checkpoint.resolve():
                raise ValueError("actor server loaded another checkpoint")
            return
        except (OSError, KeyError, ValueError):
            time.sleep(3)
    raise TimeoutError("SGLang actor did not become ready in 15 minutes")


def audit(output: Path, expected_ids: set[str]) -> dict:
    directories = list(output.glob("eval_*/"))
    if len(directories) != 1:
        raise ValueError("expected one saved evaluation directory")
    root = directories[0]
    results = [load(path) for path in root.glob("results_task_*.jsonl")]
    abnormal = [load(path) for path in root.glob("abnormal_task_*.jsonl")]
    ids = [record["task_id"] for record in results + abnormal]
    if len(ids) != 300 or len(set(ids)) != 300 or set(ids) != expected_ids:
        raise ValueError("saved results do not cover the exact 300 tasks")
    valid = [record for record in results if record.get("reward") is not None and
             "ABORTED" not in record.get("status", "") and
             not record.get("metadata", {}).get("judge_invalid") and
             not record.get("metadata", {}).get("judge_timeout")]
    successes = sum(record.get("reward") == 1.0 for record in results)
    return {"tasks": 300, "recorded": len(ids), "successes": successes,
            "overall_success_rate": successes / 300, "valid": len(valid),
            "valid_successes": sum(record.get("reward") == 1.0 for record in valid),
            "valid_only_success_rate": sum(record.get("reward") == 1.0 for record in valid) / len(valid) if valid else None,
            "abnormal": len(abnormal), "saved_output": str(root)}


def smoke_judge(env: dict[str, str], budget_path: Path) -> None:
    """Validate direct o4-mini dispatch within the shared paid-call ledger."""
    from openai import OpenAI
    from openwebrl.eval.judge_api_budget import JudgeBudget

    budget = JudgeBudget(budget_path, 20.0, 2480)
    ticket = budget.reserve()
    try:
        response = OpenAI(api_key=env["OPENAI_API_KEY"], max_retries=0).chat.completions.create(
            model="o4-mini", messages=[{"role": "user", "content": "Reply with READY."}],
            max_completion_tokens=64,
        )
    except BaseException:
        budget.settle(ticket)
        raise
    budget.settle(ticket, getattr(response, "usage", None))
    if not response.choices:
        raise ValueError("o4-mini smoke returned no choice")


def reserve_browser_sessions(path: Path, condition: str, job_id: str) -> None:
    """Conservatively charge all 300 task slots before browser dispatch."""
    with path.with_suffix(".lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        state = load(path) if path.exists() else {"cap": 620, "reserved": 0, "attempts": []}
        if state["cap"] != 620 or state["reserved"] + 300 > 620:
            raise ValueError("approved browser-session cap would be exceeded")
        if any(attempt["job_id"] == job_id for attempt in state["attempts"]):
            raise ValueError("this scheduler job already reserved browser sessions")
        state["reserved"] += 300
        state["attempts"].append({"condition": condition, "job_id": job_id, "reserved": 300})
        temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=("soft-mtp", "released-sft"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    approval = load(args.output.parent / "approval.json")
    plan = args.output.parent.parent / "plan.json"
    if approval.get("plan_sha256") != hashlib.sha256(plan.read_bytes()).hexdigest():
        raise ValueError("evaluation approval is bound to another plan")
    if approval.get("approved") is not True or approval.get("evaluation") != {
        "jobs": 2, "gpus_per_job": 1, "cpus_per_job": 8, "memory_gib_per_job": 240,
        "total_seconds_per_job": 25200, "browser_sessions": 620,
        "judge_attempts": 2480, "judge_usd": 20.0,
    }:
        raise ValueError("exact evaluation resource/API approval is missing")
    if not args.checkpoint.is_dir() or not (args.checkpoint / "config.json").is_file():
        raise FileNotFoundError("actor checkpoint is missing")
    tasks = Path(__file__).resolve().parents[1] / "openwebrl/data/eval/online-mind2web.jsonl"
    rows = [json.loads(line) for line in tasks.read_text().splitlines() if line.strip()]
    expected_ids = {row["metadata"]["task_id"] for row in rows}
    if len(rows) != 300 or len(expected_ids) != 300:
        raise ValueError("evaluation task file is not the 300-task cohort")
    args.output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    from dotenv import dotenv_values

    for key, value in dotenv_values("/gpfs/projects/krishna/zixianma/OpenWebRL/.env").items():
        if value and key in ("OPENAI_API_KEY", "JUDGE_API_KEY", "WANDB_API_KEY"):
            env.setdefault(key, value)
    if not env.get("OPENAI_API_KEY"):
        raise ValueError("o4-mini judge credential is unavailable")
    env.update(WANDB_PROJECT="openwebrl-evals", SLIME_BROWSER_ENV_MODE="local_process",
               OPENWEBRL_DIRECT_OPENAI="1",
               SLIME_BROWSER_LOCAL_PROCESS_MAX_PROCESSES="3",
               SLIME_BROWSER_LOCAL_PROCESS_PYTHON=str(PYTHON),
               OPENWEBRL_JUDGE_BUDGET_PATH=str(args.output.parent / "judge-budget.json"),
               OPENWEBRL_JUDGE_BUDGET_USD="20.0", OPENWEBRL_JUDGE_ATTEMPT_CAP="2480",
               PYTHONDONTWRITEBYTECODE="1")
    if args.condition == "soft-mtp":
        env.update(SLIME_BROWSER_LOCAL_PROCESS_PORT_START="18000", SLIME_BROWSER_LOCAL_PROCESS_PORT_END="18499")
    else:
        env.update(SLIME_BROWSER_LOCAL_PROCESS_PORT_START="18500", SLIME_BROWSER_LOCAL_PROCESS_PORT_END="18999")
    smoke_judge(env, args.output.parent / "judge-budget.json")
    server_command = [str(PYTHON), "-m", "sglang.launch_server", "--model-path", str(args.checkpoint),
                      "--host", "127.0.0.1", "--port", str(args.port), "--dtype", "bfloat16", "--tp", "1",
                      "--mem-fraction-static", "0.8", "--context-length", "32768",
                      "--max-running-requests", "12", "--chunked-prefill-size", "4096", "--disable-cuda-graph"]
    eval_command = [str(PYTHON), "-m", "openwebrl.run_evaluate", "--task-file", str(tasks),
                    "--n-parallel", "3", "--sglang-ip", "127.0.0.1", "--sglang-port", str(args.port),
                    "--hf-checkpoint", str(args.checkpoint), "--max-steps", "30",
                    "--temperature", "0.6", "--top-p", "0.95", "--top-k", "20",
                    "--max-new-tokens", "4096", "--judge-max-completion-tokens", "4096",
                    "--context-num-screenshots", "1", "--browser-response-format-mode", "browser_env",
                    "--judge-model", "o4-mini", "--judge-api-mode", "api_key",
                    "--judge-timeout-secs", "120", "--eval-protocol", "online_mind2web",
                    "--inference-step-timeout-secs", "180", "--task-timeout-secs", "1800",
                    "--output", str(args.output)]
    server = None
    with (args.output / "actor-server.log").open("a") as server_log:
        try:
            server = subprocess.Popen(server_command, env=env, stdout=server_log,
                                      stderr=subprocess.STDOUT, start_new_session=True)
            wait_for_actor(server, args.checkpoint, args.port)
            reserve_browser_sessions(args.output.parent / "browser-budget.json", args.condition,
                                     os.environ.get("SLURM_JOB_ID", "manual"))
            with (args.output / "evaluation.log").open("a") as eval_log:
                subprocess.run(eval_command, env=env, stdout=eval_log, stderr=subprocess.STDOUT, check=True)
            summary = audit(args.output, expected_ids)
            summary.update(condition=args.condition, checkpoint=str(args.checkpoint),
                           protocol="local browser, OM2W/AgentTrek o4-mini, T0.6 p0.95 k20, 4096 tokens, 30 turns")
            (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            import wandb

            os.environ["WANDB_PROJECT"] = "openwebrl-evals"
            if env.get("WANDB_API_KEY"):
                os.environ["WANDB_API_KEY"] = env["WANDB_API_KEY"]
            run = wandb.init(project="openwebrl-evals", name=f"soft-mtp-sft-{args.condition}-full300",
                             group="soft-mtp-sft-paired-full300-20261031", dir=str(args.output))
            run.log({"success/overall": summary["overall_success_rate"],
                     "success/valid_only": summary["valid_only_success_rate"],
                     "count/valid": summary["valid"], "count/abnormal": summary["abnormal"]})
            run.finish()
            print(json.dumps(summary, indent=2))
        finally:
            if server is not None and server.poll() is None:
                os.killpg(server.pid, signal.SIGTERM)
                try:
                    server.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    server.wait()


if __name__ == "__main__":
    main()
