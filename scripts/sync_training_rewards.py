"""Expose existing rollout rewards under train/ in a live shared W&B run.

Reads already-uploaded metrics and copies exact values once per rollout. Does
not touch the trainer, request compute, or change the primary run's finish state.
For future launches the rollout logger emits these metrics directly.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

from slime.utils.training_reward_metrics import (
    REWARD_AXIS, define_training_reward_metrics, training_reward_metrics,
)


def pending_rewards(history, published):
    grouped = {}
    existing = set(published)
    for row in history:
        if "train/reward" in row and REWARD_AXIS in row:
            existing.add(int(row[REWARD_AXIS]))
        if "rollout/raw_reward_mean" in row and "rollout/iteration" in row:
            iteration = int(row["rollout/iteration"])
            grouped.setdefault(iteration, {}).update(row)
    return [training_reward_metrics(grouped[i]) for i in sorted(grouped) if i not in existing]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="entity/project/run_id")
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--deadline", required=True, help="ISO-8601 timestamp with timezone")
    args = parser.parse_args()
    deadline = datetime.fromisoformat(args.deadline).timestamp()
    from dotenv import dotenv_values
    for key, value in dotenv_values(args.env_file).items():
        if key.startswith("WANDB") and value:
            os.environ[key] = value
    import wandb
    entity, project, run_id = args.run.split("/")
    api = wandb.Api(timeout=20)
    run = wandb.init(
        entity=entity, project=project, id=run_id,
        dir=str(args.run_directory),
        settings=wandb.Settings(
            mode="shared", x_primary=False, x_update_finish_state=False,
            x_label="training-reward-mirror", x_disable_stats=True,
        ),
    )
    define_training_reward_metrics(run)
    published = set()
    try:
        with (args.run_directory / "training_reward_sync.jsonl").open("a", buffering=1) as audit:
            while time.time() < deadline:
                try:
                    remote = api.run(args.run)
                    history = list(remote.scan_history(page_size=100))
                    for metrics in pending_rewards(history, published):
                        if not metrics:
                            continue
                        run.log(metrics)
                        published.add(metrics[REWARD_AXIS])
                        record = {"utc": datetime.now(timezone.utc).isoformat(), "metrics": metrics}
                        audit.write(json.dumps(record) + "\n")
                        print(json.dumps(record), flush=True)
                    if (args.run_directory / "exit_status.json").exists():
                        break
                except Exception as exc:
                    print(f"Reward sync retry: {type(exc).__name__}: {exc}", flush=True)
                time.sleep(min(30, max(0, deadline - time.time())))
    finally:
        run.finish()


if __name__ == "__main__":
    main()
