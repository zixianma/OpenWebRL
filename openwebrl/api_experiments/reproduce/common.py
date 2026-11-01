"""Reproduce a prepared BrowserCode runtime with isolated all-attempt budgets."""
from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
import hashlib
import importlib
import importlib.util
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
    "gpt": ("serial_high", "onecall3_low", "single_low", "single_medium", "parallel3_select_low", "parallel3_select_high", "parallel5_select_high", "luna_single_high"),
}


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def required_cloud(runtime):
    """Bind the prospective cloud marker to actual runtime/adapter bytes.

    Read the runtime again at execution: a caller cannot remove a field from a
    previously prepared dictionary and accidentally select the local backend.
    """
    runtime = Path(runtime).resolve()
    marker = runtime / 'cloud-experiment.json'
    campaign, config = read(runtime/'campaign-plan.json'), read(runtime/'runtime-config.json')
    marked = marker.exists() or campaign.get('protocol',{}).get('browser_backend') == 'browser-use-v4' or 'cloud_protocol' in config
    if not marked:
        return None
    if not marker.is_file():
        raise ValueError('Cloud runtime requires its original explicit backend marker')
    receipt = read(marker)
    expected = dict(browser_backend='browser-use-v4',proxy_country='us',paid_authorization=False)
    if receipt.get('schema') != 'prospective-browsercode-cloud-study-v1' or any(type(receipt.get(k)) is not type(v) or receipt.get(k)!=v for k,v in expected.items()):
        raise ValueError('Invalid prospective cloud backend receipt')
    if campaign.get('protocol',{}).get('browser_backend') != 'browser-use-v4' or config.get('cloud_protocol',{}).get('browser_backend') != 'browser-use-v4':
        raise ValueError('Runtime cloud backend declarations differ')
    manifest = read(runtime/'source-manifest.json')
    for path in (marker,runtime/'campaign-plan.json',runtime/'runtime-config.json'):
        if manifest['source_hashes'].get(str(path)) != digest(path):
            raise ValueError('Cloud runtime source binding changed')
    if manifest.get('plan_sha256') != digest(runtime/'campaign-plan.json'):
        raise ValueError('Cloud runtime plan binding changed')
    sources = receipt.get('adapter_source_sha256')
    names = {'common.py','cloud_browser.py','cloud_browsercode.py'}
    if type(sources) is not dict or set(sources) != names or any(sources[name] != digest(Path(__file__).with_name(name)) for name in names):
        raise ValueError('Cloud adapter source binding changed; prepare a fresh runtime')
    return dict(browser_backend='browser-use-v4',proxy_country='us',marker_sha256=digest(marker),adapter_source_sha256=sources)


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
    p.add_argument("--runtime", type=Path, required=True, help="Isolated runtime produced by prepare_browsercode.py")
    choices = (*ARMS[provider], 'luna_single_high') if provider == 'luna' else ARMS[provider]
    p.add_argument("--arm", choices=choices, required=True)
    p.add_argument("--harness", choices=("browsercode", "cuawright"), default="browsercode", help="Earlier paired-harness campaigns support both executors")
    p.add_argument("--benchmark", choices=("online-mind2web", "webvoyager", "deepshop"), help="Filter the prepared task roster before applying --limit")
    p.add_argument("--task-ids", type=Path, help="JSON list of unique ordered frozen task IDs; select after --benchmark and before --limit")
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
    prospective = None
    if (runtime / 'prospective-experiment.json').exists() or plan.get('prospective_protocol') or config.get('prospective_protocol'):
        if provider != 'luna' or args.arm not in ('luna_single_xhigh', 'luna_single_high') or getattr(args, 'harness', 'browsercode') != 'browsercode':
            raise ValueError('Prospective output override requires Luna single high/xhigh BrowserCode')
        spec = importlib.util.spec_from_file_location('prospective_browsercode', Path(__file__).with_name('prospective_browsercode.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        prospective = module.validate(runtime)
        if prospective['arm'] != args.arm:
            raise ValueError('Prospective arm/effort does not match selected arm')
    benchmark = getattr(args, "benchmark", None)
    if benchmark:
        tasks = [t for t in tasks if t.get("benchmark", "online-mind2web") == benchmark]
    selection = getattr(args, 'task_ids', None)
    selection_sha = None
    if selection is not None:
        selection = Path(selection).resolve()
        raw_selection = selection.read_bytes()
        ids = json.loads(raw_selection)
        if not isinstance(ids, list) or not ids or any(not isinstance(value, str) for value in ids) or len(set(ids)) != len(ids):
            raise ValueError('--task-ids requires a nonempty JSON list of unique task ID strings')
        if any(not re.fullmatch(r'[A-Za-z0-9_. -]+', value) or value in ('.', '..') for value in ids):
            raise ValueError('--task-ids contains an unsafe task filename')
        lookup = {task['task_id']: task for task in tasks}
        if any(value not in lookup for value in ids):
            raise ValueError('--task-ids contains a task absent from the frozen roster/benchmark')
        tasks = [lookup[value] for value in ids]
        selection_sha = hashlib.sha256(raw_selection).hexdigest()
    if not 1 <= args.limit <= len(tasks):
        raise ValueError("limit must be between 1 and the frozen task count")
    if min(args.model_calls, args.judge_calls, args.browser_starts if args.browser_starts is not None else args.limit + 1) <= 0:
        raise ValueError("Call and browser caps must be positive")
    spec = next((a for a in plan["arms"] if a["arm"] == args.arm), None)
    if (args.arm not in ARMS[provider] and not (prospective and args.arm == prospective['arm'])) or spec is None:
        raise ValueError("Arm does not belong to this provider/runtime")
    expected = spec["model"]
    allowed_models = {"luna": {"gpt-6-luna"}, "claude": {"claude-opus-5-5", "claude-sonnet-5-5"}, "gpt": {"gpt-6-luna", "gpt-6-sol", "gpt-6.1-sol"}}
    if expected not in allowed_models[provider]:
        raise ValueError("Unexpected model for provider")
    if provider == 'luna' and args.arm == 'luna_single_high' and prospective is None:
        raise ValueError('Luna single high requires a validated prospective override')
    harness = getattr(args, "harness", "browsercode")
    supported_harnesses = {cell["harness"] for cell in plan.get("cells", [])} or {"browsercode"}
    if harness not in supported_harnesses:
        raise ValueError("Harness is not supported by this frozen campaign")
    expected_output = spec["output_tokens"]
    allowed_outputs = (prospective['output_tokens'],) if prospective else (4096, 12288)
    expected_decisions = prospective['max_decisions'] if prospective else 30
    if plan["protocol"]["max_decisions"] != expected_decisions or expected_output not in allowed_outputs:
        raise ValueError("Runtime is not the frozen BrowserCode model/12288/30 protocol")
    if provider != "gpt" and config["judge_protocol"]["model"] != "gpt-5.6-terra":
        raise ValueError("Runtime must use the frozen Terra judge")
    manifest = read(runtime / "source-manifest.json")
    if manifest["plan_sha256"] != digest(runtime / "campaign-plan.json"):
        raise ValueError("Frozen plan hash changed")
    for path, expected_hash in manifest["source_hashes"].items():
        if digest(path) != expected_hash:
            raise ValueError("Frozen source changed: " + path)
    cloud = required_cloud(runtime)
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
    if len(set(ids)) != len(ids) or any(not re.fullmatch(r"[A-Za-z0-9_. -]+", x) or x in (".", "..") for x in ids):
        raise ValueError("Task IDs must be unique safe filenames")
    result = {"provider": provider, "arm": args.arm, "model": expected, "candidate_count": spec["proposal_count"],
            "candidate_effort": spec.get("reasoning_effort", "high"), "selector_effort": spec.get("selector_reasoning_effort", spec.get("reasoning_effort", "high") if spec["proposal_count"] > 1 else None),
            "selector_max_output_tokens": spec.get("selector_output_tokens", plan["protocol"].get("selector_max_output_tokens")) if spec["proposal_count"] > 1 else None,
            "task_count": len(chosen), "benchmark": benchmark, "harness": harness, "max_output_tokens": expected_output, "max_decisions": expected_decisions,
            "runtime": str(runtime), "output": str(output), "source_manifest_sha256": digest(runtime / "source-manifest.json"),
            "task_manifest_sha256": digest(runtime / "tasks.json"), "model_usd_cap": args.model_usd, "judge_usd_cap": args.judge_usd,
            "model_call_cap": args.model_calls, "judge_call_cap": args.judge_calls,
            "browser_start_cap": args.browser_starts if args.browser_starts is not None else args.limit + 1,
            "scheduling": "one episode at a time; original per-decision proposal parallelism", "execute": args.execute}
    if cloud is not None:
        result['required_cloud'] = cloud
    if prospective:
        result.update(prospective_protocol=prospective['schema'], output_multiplier=prospective['output_multiplier'],
                      prospective_receipt_sha256=digest(runtime / 'prospective-experiment.json'),
                      benchmark_roster_counts=prospective['benchmark_counts'])
    if selection_sha:
        result.update(task_selection_sha256=selection_sha, task_selection_count=len(tasks),
                      selected_task_ids=ids, task_selection_path=str(selection))
    return result, chosen


def summary(rows, pools, cleanup_verified):
    valid = [r for r in rows if r["valid"]]
    def mean(key):
        values = [float(r[key]) for r in valid if r.get(key) is not None]
        return {"mean": sum(values) / len(values) if values else None, "observations": len(values)}
    return {"recorded": len(rows), "valid": len(valid), "invalid": len(rows) - len(valid),
            "successes": sum(r.get("reward") == 1 for r in valid),
            "recorded_success_rate": sum(r.get("reward") == 1 for r in valid) / len(rows) if rows else None,
            "valid_only_success": sum(r.get("reward") == 1 for r in valid) / len(valid) if valid else None,
            "valid_task_latency_seconds": mean("task_latency_seconds"), "valid_selected_browser_batches": mean("selected_code_batches"),
            "all_attempt_pools": pools, "owned_cleanup_verified": cleanup_verified}


def execute(args, plan, tasks, *, worker_setup=None):
    cloud = required_cloud(plan['runtime'])
    if cloud is not None or plan.get('required_cloud') is not None:
        if cloud is None or plan.get('required_cloud') != cloud or worker_setup is None:
            raise ValueError('This runtime requires its bound Browser Use cloud owner; the local run frontdoor cannot execute it')
    keys = dict(os.environ)
    if args.env_file:
        from dotenv import dotenv_values
        keys.update({k: v for k, v in dotenv_values(args.env_file).items() if v})
    actor_key = "ANTHROPIC_API_KEY" if plan["provider"] == "claude" else "OPENAI_API_KEY"
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
    os.environ["OPENWEBRL_REPRO_OUTPUT"] = str(root)
    if keys.get("WANDB_API_KEY"):
        os.environ["WANDB_API_KEY"] = keys["WANDB_API_KEY"]
    sys.path.insert(0, plan["runtime"])
    worker = importlib.import_module("worker")
    environments = importlib.import_module("environments")
    integration = worker_setup(worker, environments) if worker_setup is not None else None
    try:
        if cloud is not None:
            module = sys.modules.get(type(integration).__module__)
            if (module is None or getattr(module,'Registry',None) is not type(integration)
                or Path(module.__file__).resolve() != Path(__file__).with_name('cloud_browsercode.py').resolve()
                or integration.worker is not worker):
                raise ValueError('Bound Browser Use cloud Registry required before SDK initialization')
        if plan["provider"] != "gpt":
            budget_module = importlib.import_module("request_budget" if plan["provider"] == "luna" else "claude_budget")
            gate = budget_module.SharedHTTPGate(root / "http-gate", 8)
            original = worker.PairedModel
            class ScopedModel(original):
                def __init__(self, *positional, **kwargs):
                    kwargs["http_gate"] = gate
                    super().__init__(*positional, **kwargs)
            worker.PairedModel = ScopedModel
        budget_type = getattr(worker, "ReservationBudget", worker.Budget)
        model_cap = Decimal(args.model_usd) if plan["provider"] == "gpt" else args.model_usd
        judge_cap = Decimal(args.judge_usd) if plan["provider"] == "gpt" else args.judge_usd
        budgets = {"actor": budget_type(root / "budget/model", model_cap, args.model_calls),
                   "judge": (getattr(integration, "judge_budget_type", None) or budget_type)(root / "budget/judge", judge_cap, args.judge_calls),
                   "browser": worker.Budget(root / "budget/browser", plan["browser_start_cap"], plan["browser_start_cap"])}
        from openai import OpenAI
        judge = OpenAI(api_key=keys.get("JUDGE_API_KEY") or keys["OPENAI_API_KEY"], max_retries=0, timeout=180)
        try:
            if plan["provider"] != "claude":
                actor = OpenAI(api_key=keys[actor_key], max_retries=0, timeout=180)
            else:
                from anthropic import Anthropic
                actor = Anthropic(api_key=keys[actor_key], max_retries=0, timeout=180)
        except BaseException:
            judge.close()
            raise
    except BaseException:
        if integration is not None:
            try:
                integration.close_all()
            finally:
                integration.restore()
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
                row = worker.episode(root, task, {"harness": plan.get("harness", "browsercode"), "policy": args.arm}, actor, judge, budgets, 0)
            except BaseException:
                saved = root / "episodes" / (task["task_id"] + "-" + plan.get("harness", "browsercode") + "-" + args.arm) / "result.json"
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
        cloud_cleanup = None
        cloud_error = None
        if integration is not None:
            try:
                cloud_cleanup = integration.close_all()
            except BaseException as exc:
                cloud_error = type(exc).__name__
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
            pools = {}
            for name, budget in budgets.items():
                if name == "browser":
                    continue
                state = budget.snapshot() if hasattr(budget, "snapshot") else (read(budget.root / "ledger.json") if (budget.root / "ledger.json").exists() else {"calls": 0, "charged_or_reserved_usd": 0})
                pools[name] = {key: state[key] for key in ("calls", "charged_or_reserved_usd")}
            result = summary(rows, pools, cleanup_ok)
            result["requested"] = len(tasks)
            result["unrecorded"] = len(tasks) - len(rows)
            result["stopped_reason"] = stopped
            if cloud_error is not None:
                cleanup_ok = False
                result['owned_cleanup_verified'] = False
                finalization_errors.append({'component':'cloud_ownership','error_type':cloud_error})
            if integration is not None:
                result['cloud_cleanup'] = cloud_cleanup
                integration.restore()
            result["finalization_errors"] = finalization_errors
            (root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 1 if stopped or finalization_errors or not cleanup_ok or len(rows) != len(tasks) or result["invalid"] else 0


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
