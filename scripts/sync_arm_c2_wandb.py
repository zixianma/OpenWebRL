#!/usr/bin/env python3
"""Backfill and follow an active C2 trainer's durable metrics in Weights & Biases."""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import secrets
import signal
import time


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    from dotenv import load_dotenv
    import wandb

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--project", default=None)
    parser.add_argument("--entity", default=None)
    parser.add_argument("--name", default="arm-c2-filtered-sft-openwebrl-4b")
    parser.add_argument("--group", default="arm-c2-filtered-sft")
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument("--poll-seconds", type=float, default=10)
    args = parser.parse_args()
    load_dotenv(args.env_file, override=False)
    root = args.run_root.resolve()
    student = root / "student"
    wandb_root = student / "wandb"
    wandb_root.mkdir(parents=True, exist_ok=True)
    cache = wandb_root / "cache"
    cache.mkdir(exist_ok=True)
    os.environ.setdefault("WANDB_CACHE_DIR", str(cache))
    metrics_path = student / "metrics.jsonl"
    if not metrics_path.exists():
        raise ValueError("Student metrics do not exist")
    lock = (student / "wandb-sync.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = student / "wandb-sync.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
    else:
        state = {
            "run_id": secrets.token_hex(4),
            "last_logged_update": 0,
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        write_json(state_path, state)

    training = json.loads((student / "training-config.json").read_text())
    audit = json.loads((root / "dataset-audit.json").read_text())
    # A resumed run must stay in the project recorded with its durable run ID.
    # New ARM runs can select their dedicated project with --project.
    project = state.get("project") or args.project or os.getenv("WANDB_PROJECT", "openwebrl")
    run = wandb.init(
        project=project,
        entity=args.entity,
        id=state["run_id"],
        resume="allow",
        name=args.name,
        group=args.group,
        job_type="sft",
        tags=["arm", "c2", "action-level-sft", "openwebrl-4b", *args.tag],
        dir=str(wandb_root),
        config={
            **training,
            "examples": audit.get("retained_turns"),
            "successful_trajectories": audit.get("successful_tasks_with_usable_turns"),
            "task_count": audit.get("scheduled"),
            "metric_source": str(metrics_path),
            "logging_mode": "durable-file sidecar with historical backfill",
        },
        settings=wandb.Settings(init_timeout=60),
    )
    run.define_metric("optimizer/update")
    run.define_metric("train/*", step_metric="optimizer/update")
    state.update(project=project, entity=run.entity, url=run.url, status="running")
    write_json(state_path, state)
    print(run.url, flush=True)
    stopped = False

    def stop(*_args):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        while not stopped:
            rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
            pending = [row for row in rows if row["updates"] > state["last_logged_update"]]
            for row in pending:
                epoch_size = audit["retained_turns"]
                payload = {
                        "optimizer/update": row["updates"],
                        "train/cross_entropy": row["loss"],
                        "train/perplexity": math.exp(row["loss"]),
                        "train/gradient_norm": row["grad_norm"],
                        "train/learning_rate": row["lr"],
                        "train/epoch_index": row["epoch"],
                        "train/epoch_number": row["epoch"] + 1,
                        "train/epoch_position": row["next_position"],
                        "train/epoch_fraction": row["next_position"] / epoch_size,
                    }
                for source, target in (
                    ("sft_loss", "train/sft_loss"),
                    ("preference_loss", "train/preference_loss"),
                    ("policy_margin", "train/policy_margin"),
                    ("reference_margin", "train/reference_margin"),
                    ("preference_weight", "train/preference_weight"),
                ):
                    if source in row:
                        payload[target] = row[source]
                run.log(payload, step=row["updates"])
                state["last_logged_update"] = row["updates"]
                state["last_logged_utc"] = row["utc"]
                write_json(state_path, state)
            if (student / "complete.json").exists() and not pending:
                state.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat())
                write_json(state_path, state)
                break
            time.sleep(args.poll_seconds)
    finally:
        if state.get("status") != "complete":
            state.update(status="paused", paused_utc=datetime.now(timezone.utc).isoformat())
            write_json(state_path, state)
        run.finish()


if __name__ == "__main__":
    main()
