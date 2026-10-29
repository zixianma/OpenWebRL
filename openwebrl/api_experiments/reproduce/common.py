"""Single-process BrowserCode reproduction using an existing frozen runtime."""
from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import signal
import sys
import time
import uuid

ARMS = {
    "luna": ("luna_single_xhigh", "luna_single_medium", "luna_single_low", "luna_n5_medium", "luna_n5_low"),
    "claude": ("opus_single", "sonnet_single", "sonnet_n3", "sonnet_n5"),
}


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def positive_money(value):
    try:
        amount = Decimal(value)
    except InvalidOperation as error:
        raise argparse.ArgumentTypeError("Use a positive finite USD amount") from error
    if not amount.is_finite() or amount <= 0:
        raise argparse.ArgumentTypeError("Use a positive finite USD amount")
    return str(amount)


def parser(provider):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime", type=Path, required=True, help="Existing configured frozen campaign directory")
    p.add_argument("--arm", choices=ARMS[provider], required=True)
    p.add_argument("--output", type=Path, required=True, help="New output directory outside the source runtime")
    p.add_argument("--limit", type=int, default=1, help="First N frozen tasks; default: 1")
    p.add_argument("--model-usd", type=positive_money, required=True)
    p.add_argument("--judge-usd", type=positive_money, required=True)
    p.add_argument("--model-calls", type=int, default=10000, help="Includes all candidates, selectors and counters")
    p.add_argument("--judge-calls", type=int, default=1100)
    p.add_argument("--browser-starts", type=int, help="Default: limit + one mechanical preflight")
    p.add_argument("--env-file", type=Path, help="Optional local key file; otherwise use process environment")
    p.add_argument("--wandb-mode", choices=("offline", "online", "disabled"), default="offline")
    p.add_argument("--execute", action="store_true", help="Run browser/API work; omitted means read-only dry run")
    return p


def prepare(args, provider):
    runtime = args.runtime.resolve()
    output = args.output.resolve()
    if output.is_relative_to(Path("/gpfs/home/zixianma")):
        raise ValueError("Output cannot be below the read-only home directory")
    if output.exists() or args.output.is_symlink():
        raise ValueError("Output must be a new directory; existing evidence is never overwritten")
    if output.is_relative_to(runtime):
        raise ValueError("Place output outside the frozen source runtime")
    plan, config, tasks = (read(runtime / name) for name in ("campaign-plan.json", "runtime-config.json", "tasks.json"))
    if not 1 <= args.limit <= len(tasks):
        raise ValueError("limit must be between 1 and the frozen task count")
    if min(args.model_calls, args.judge_calls, args.browser_starts if args.browser_starts is not None else args.limit + 1) <= 0:
        raise ValueError("Call and browser caps must be positive")
    spec = next((a for a in plan["arms"] if a["arm"] == args.arm), None)
    if args.arm not in ARMS[provider] or spec is None:
        raise ValueError("Arm does not belong to this provider/runtime")
    expected = "gpt-6-luna" if provider == "luna" else ("claude-opus-5-5" if args.arm == "opus_single" else "claude-sonnet-5-5")
    if spec["model"] != expected or plan["protocol"]["max_decisions"] != 30 or spec["output_tokens"] != 12288:
        raise ValueError("Runtime is not the frozen BrowserCode model/12288/30 protocol")
    if config["judge_protocol"]["model"] != "gpt-5.6-terra":
        raise ValueError("Runtime must use the frozen Terra judge")
    manifest = read(runtime / "source-manifest.json")
    if manifest["plan_sha256"] != digest(runtime / "campaign-plan.json"):
        raise ValueError("Frozen plan hash changed")
    for path, expected_hash in manifest["source_hashes"].items():
        if digest(path) != expected_hash:
            raise ValueError("Frozen source changed: " + path)
    if plan["task_manifest_sha256"] != digest(runtime / "tasks.json"):
        raise ValueError("Frozen task manifest changed")
    for name in ("cuawright_source", "frozen_support_source", "chromium", "python"):
        if not Path(config[name]).exists():
            raise ValueError("Missing runtime dependency: " + name)
    for name in ("browser_bridge.ts", "browsercode", "node_modules", "javascript-runtime/bun", "sandbox.py"):
        if not (runtime / name).exists():
            raise ValueError("Missing configured browser dependency: " + name)
    bridge = (runtime / "browser_bridge.ts").read_text()
    guard = re.search(r"workspace\.startsWith\('([^']+)'\)", bridge)
    if guard and not output.is_relative_to(Path(guard.group(1))):
        raise ValueError("Configured bridge requires output below " + guard.group(1))
    chosen = tasks[:args.limit]
    ids = [t["task_id"] for t in chosen]
    if len(set(ids)) != len(ids) or any(not re.fullmatch(r"[A-Za-z0-9_.-]+", x) or x in (".", "..") for x in ids):
        raise ValueError("Task IDs must be unique safe filenames")
    return {"provider": provider, "arm": args.arm, "model": expected, "candidate_count": spec["proposal_count"],
            "candidate_effort": spec["reasoning_effort"], "selector_effort": spec.get("selector_reasoning_effort", spec["reasoning_effort"] if spec["proposal_count"] > 1 else None),
            "task_count": len(chosen), "max_output_tokens": 12288, "max_decisions": 30,
            "runtime": str(runtime), "output": str(output), "source_manifest_sha256": digest(runtime / "source-manifest.json"),
            "task_manifest_sha256": digest(runtime / "tasks.json"), "model_usd_cap": args.model_usd, "judge_usd_cap": args.judge_usd,
            "model_call_cap": args.model_calls, "judge_call_cap": args.judge_calls,
            "browser_start_cap": args.browser_starts if args.browser_starts is not None else args.limit + 1,
            "scheduling": "one episode at a time; original per-decision proposal parallelism", "execute": args.execute}, chosen


def summary(rows, pools, cleanup_verified):
    valid = [r for r in rows if r["valid"]]
    def mean(key):
        values = [float(r[key]) for r in valid if r.get(key) is not None]
        return {"mean": sum(values) / len(values) if values else None, "observations": len(values)}
    return {"recorded": len(rows), "valid": len(valid), "invalid": len(rows) - len(valid),
            "successes": sum(r.get("reward") == 1 for r in valid),
            "valid_only_success": sum(r.get("reward") == 1 for r in valid) / len(valid) if valid else None,
            "valid_task_latency_seconds": mean("task_latency_seconds"), "valid_selected_browser_batches": mean("selected_code_batches"),
            "all_attempt_pools": pools, "owned_cleanup_verified": cleanup_verified}


def execute(args, plan, tasks):
    keys = dict(os.environ)
    if args.env_file:
        from dotenv import dotenv_values
        keys.update({k: v for k, v in dotenv_values(args.env_file).items() if v})
    actor_key = "OPENAI_API_KEY" if plan["provider"] == "luna" else "ANTHROPIC_API_KEY"
    if not keys.get(actor_key) or not (keys.get("JUDGE_API_KEY") or keys.get("OPENAI_API_KEY")):
        raise ValueError("Set " + actor_key + " and OPENAI_API_KEY or JUDGE_API_KEY")
    root = Path(plan["output"])
    root.mkdir(parents=True, exist_ok=False)
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    for name in ("XDG_CACHE_HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME", "WANDB_DIR", "WANDB_CONFIG_DIR", "WANDB_CACHE_DIR", "WANDB_DATA_DIR", "HF_HOME", "TORCH_HOME", "MPLCONFIGDIR", "PYTHONPYCACHEPREFIX"):
        path = root / "runtime-cache" / name.lower()
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    for name in ("WANDB_RUN_ID", "WANDB_RESUME", "WANDB_NAME", "WANDB_RUN_GROUP", "WANDB_SWEEP_ID", "WANDB_JOB_TYPE"):
        os.environ.pop(name, None)
    os.environ["WANDB_PROJECT"] = "openwebrl-evals"
    if keys.get("WANDB_API_KEY"):
        os.environ["WANDB_API_KEY"] = keys["WANDB_API_KEY"]
    sys.path.insert(0, plan["runtime"])
    worker = importlib.import_module("worker")
    environments = importlib.import_module("environments")
    budget_module = importlib.import_module("request_budget" if plan["provider"] == "luna" else "claude_budget")
    gate = budget_module.SharedHTTPGate(root / "http-gate", 8)
    original = worker.PairedModel
    class ScopedModel(original):
        def __init__(self, *positional, **kwargs):
            kwargs["http_gate"] = gate
            super().__init__(*positional, **kwargs)
    worker.PairedModel = ScopedModel
    budgets = {"actor": worker.ReservationBudget(root / "budget/model", args.model_usd, args.model_calls),
               "judge": worker.ReservationBudget(root / "budget/judge", args.judge_usd, args.judge_calls),
               "browser": worker.Budget(root / "budget/browser", plan["browser_start_cap"], plan["browser_start_cap"])}
    from openai import OpenAI
    judge = OpenAI(api_key=keys.get("JUDGE_API_KEY") or keys["OPENAI_API_KEY"], max_retries=0, timeout=180)
    try:
        if plan["provider"] == "luna":
            actor = OpenAI(api_key=keys[actor_key], max_retries=0, timeout=180)
        else:
            from anthropic import Anthropic
            actor = Anthropic(api_key=keys[actor_key], max_retries=0, timeout=180)
    except BaseException:
        judge.close()
        raise
    (root / "reproduction-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    (root / "selected-task-ids.json").write_text(json.dumps([t["task_id"] for t in tasks]) + "\n")
    rows, tracking = [], None
    cleanup_ok, stopped = False, None
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(SystemExit(143)))
    try:
        if args.wandb_mode != "disabled":
            import wandb
            tracking = wandb.init(id=uuid.uuid4().hex, resume="never", project="openwebrl-evals", mode=args.wandb_mode, dir=str(root), job_type="api_browsercode_reproduction", config=plan, settings=wandb.Settings(disable_git=True), save_code=False)
        worker.preflight(root, budgets["browser"])
        for task in tasks:
            try:
                row = worker.episode(root, task, {"harness": "browsercode", "policy": args.arm}, actor, judge, budgets, 0)
            except BaseException:
                saved = root / "episodes" / (task["task_id"] + "-browsercode-" + args.arm) / "result.json"
                if saved.is_file():
                    rows.append(read(saved))
                stopped = "Episode interrupted; inspect preserved evidence before continuation"
                raise
            rows.append(row)
            if tracking:
                tracking.log({"valid": int(row["valid"]), "success": int(row.get("reward") == 1), "task_latency_seconds": row.get("task_latency_seconds"), "browser_batches": row.get("selected_code_batches")})
            (root / "progress.json").write_text(json.dumps({"recorded": len(rows), "epoch": time.time()}) + "\n")
            if any((b.root / "halt.json").exists() for b in budgets.values()):
                stopped = "Pool halted; preserve evidence and diagnose before any continuation"
                break
    finally:
        try:
            environments.cleanup_registered_groups(root)
            cleanup_ok = True
        finally:
            finalization_errors = []
            for component, close in (("tracking", tracking.finish if tracking else lambda: None), ("actor_client", actor.close), ("judge_client", judge.close)):
                try:
                    close()
                except Exception as error:
                    finalization_errors.append({"component": component, "error_type": type(error).__name__})
            pools = {name: {"calls": b.snapshot()["calls"], "charged_or_reserved_usd": b.snapshot()["charged_or_reserved_usd"]} for name, b in budgets.items() if name != "browser"}
            result = summary(rows, pools, cleanup_ok)
            result["requested"] = len(tasks)
            result["unrecorded"] = len(tasks) - len(rows)
            result["stopped_reason"] = stopped
            result["finalization_errors"] = finalization_errors
            (root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 1 if stopped or finalization_errors else 0


def main(provider):
    p = parser(provider)
    args = p.parse_args()
    try:
        plan, tasks = prepare(args, provider)
        if not args.execute:
            print(json.dumps(plan, indent=2))
            print("Dry run: no output directory, browser, SDK client or API request created. Add --execute to run.")
            return 0
        return execute(args, plan, tasks)
    except (ValueError, FileNotFoundError) as error:
        p.error(str(error))
