"""Reproducible, resumable Online-Mind2Web inference comparison (no policy training)."""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path


def summarize(records, scheduled):
    valid = [r for r in records if r.get("valid")]
    successes = sum(r.get("reward") == 1.0 for r in valid)
    return {
        "scheduled": scheduled, "attempted": len(records), "valid": len(valid),
        "successes": successes, "unavailable": len(records) - len(valid),
        "success_rate_all_scheduled": successes / scheduled if scheduled else None,
        "success_rate_valid": successes / len(valid) if valid else None,
    }


async def run(args):
    import httpx
    from dotenv import load_dotenv
    from openwebrl import generate_browser as generation
    from openwebrl import run_evaluate as evaluation
    from openwebrl.arm_inference import ActionSelector
    from slime.utils.http_utils import init_http_client

    if args.env_file:
        load_dotenv(args.env_file, override=False)
    os.environ.setdefault("JUDGE_API_BASE", "https://api.openai.com/v1")
    os.environ["SLIME_BROWSER_ENV_MODE"] = "local_process"
    os.environ["SLIME_BROWSER_LOCAL_PROCESS_LOG_DIR"] = str(Path(args.output) / "browser_logs")
    os.environ["SLIME_BROWSER_LOCAL_PROCESS_PORT_START"] = "19200"
    os.environ["SLIME_BROWSER_LOCAL_PROCESS_PORT_END"] = "19399"
    os.environ["SLIME_BROWSER_LOCAL_PROCESS_MAX_PROCESSES"] = str(args.parallel)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    # The training generator records navigation-failure hosts; keep evaluation's
    # list inside its own output directory, never modify the training blacklist.
    generation._BROWSER_HOST_BLACKLIST_PATH = str(output / "navigation_failures.txt")
    evaluation.reward_func = evaluation._load_reward_func("online_mind2web")

    tasks = evaluation.load_tasks_from_jsonl(args.task_file)
    if args.task_indices:
        wanted = {int(i) for i in args.task_indices.split(",")}
        tasks = [t for t in tasks if t["index"] in wanted]
        if len(tasks) != len(wanted):
            raise ValueError("Some requested task indices were not found")
    if len({t["task_id"] for t in tasks}) != len(tasks):
        raise ValueError("Duplicate task IDs")
    sampling = {"temperature": args.temperature, "top_p": args.top_p,
                "max_new_tokens": args.max_new_tokens}
    manifest = {
        "mode": args.mode, "actor": args.actor, "task_file": str(Path(args.task_file).resolve()),
        "task_file_sha256": hashlib.sha256(Path(args.task_file).read_bytes()).hexdigest(),
        "task_ids": [t["task_id"] for t in tasks], "seed": args.seed,
        "sampling": sampling, "max_steps": args.max_steps,
        "context_num_screenshots": 1, "history": "full",
        "browser_format": "browser_env", "judge": args.judge_model,
        "judge_protocol": "online_mind2web/AgentTrek", "candidate_count": 1 if args.mode == "baseline" else 5,
        "selector_endpoint": args.selector_endpoint if args.mode != "baseline" else None,
        "task_timeout": args.task_timeout,
    }
    async with httpx.AsyncClient(timeout=30, trust_env=False) as client:
        if args.mode != "baseline":
            resp = await client.get(args.selector_endpoint.rstrip("/") + "/health")
            resp.raise_for_status()
            manifest["selector_health"] = resp.json()
            if manifest["selector_health"]["mode"] != args.mode:
                raise ValueError("Wrong selector service mode")
        resp = await client.get(f"http://{args.actor_host}:{args.actor_port}/get_model_info")
        resp.raise_for_status()
        info = resp.json()
        manifest["actor_server_model"] = info.get("model_path")
        if os.path.realpath(info.get("model_path", "")) != os.path.realpath(args.actor):
            raise ValueError("Actor server does not match pinned frozen actor path")

    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Output manifest differs; use a new directory")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "results").mkdir(exist_ok=True)
    eval_args = evaluation.EvalArgs(
        sglang_router_ip=args.actor_host, sglang_router_port=args.actor_port,
        hf_checkpoint=args.actor, max_steps=args.max_steps,
        context_num_screenshots=1, judge_api_model=args.judge_model,
        judge_api_mode="served", judge_timeout_secs=180,
        browser_response_format_mode="browser_env", turn_history_reasoning_mode="full",
        browser_include_tool_response=1, inference_step_timeout_secs=180,
        task_timeout_secs=args.task_timeout, path_to_save_generated_samples=str(output / "samples"),
    )
    eval_args.rollout_temperature = args.temperature
    eval_args.rollout_max_response_len = args.max_new_tokens
    eval_args.browser_action_selector = ActionSelector(
        args.mode, args.selector_endpoint, output / "selections", seed=args.seed)
    init_http_client(eval_args)
    sem = asyncio.Semaphore(args.parallel)

    async def one(task):
        path = output / "results" / (hashlib.sha256(task["task_id"].encode()).hexdigest()[:20] + ".json")
        if path.exists():
            return json.loads(path.read_text())
        async with sem:
            print(f"START {args.mode} {task['task_id']}", flush=True)
            try:
                record = await evaluation.evaluate_single_task(eval_args, task, sampling, True)
                record["valid"] = record.get("reward") is not None and "ABORTED" not in record.get("status", "")
                if record.get("metadata", {}).get("judge_invalid") or record.get("metadata", {}).get("judge_timeout"):
                    record["valid"] = False
            except Exception as exc:
                record = {"task_id": task["task_id"], "reward": None, "valid": False,
                          "error_type": type(exc).__name__, "error": str(exc)}
            temp = path.with_suffix(".tmp")
            temp.write_text(json.dumps(record, ensure_ascii=False, indent=2))
            temp.replace(path)
            print(f"DONE {args.mode} {task['task_id']} reward={record.get('reward')} valid={record['valid']}", flush=True)
            return record

    records = await asyncio.gather(*(one(t) for t in tasks))
    summary = summarize(records, len(tasks))
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", required=True, choices=["baseline", "selection", "scalar"])
    ap.add_argument("--actor", default="/gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT")
    ap.add_argument("--actor-host", default="127.0.0.1")
    ap.add_argument("--actor-port", type=int, default=19100)
    ap.add_argument("--selector-endpoint", default="http://127.0.0.1:19101")
    ap.add_argument("--task-file", default="openwebrl/data/eval/online-mind2web.jsonl")
    ap.add_argument("--task-indices", default="")
    ap.add_argument("--output", required=True)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--task-timeout", type=int, default=1800)
    ap.add_argument("--judge-model", default="o4-mini")
    ap.add_argument("--env-file", default=".env")
    args = ap.parse_args()
    if args.parallel < 1:
        ap.error("--parallel must be positive")
    # SGLang configures the event-loop policy during import. Do so before
    # creating the loop, otherwise browser subprocesses can lose their watcher.
    from openwebrl import run_evaluate  # noqa: F401
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
