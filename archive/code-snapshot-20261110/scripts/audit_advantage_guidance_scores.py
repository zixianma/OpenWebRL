#!/usr/bin/env python3
"""Aggregate action-only AG validation scores without publishing private states."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random

from prepare_action_chunk_outcome_pilot import digest


def mean(values):
    return sum(values) / len(values)


def audit(dataset: Path, scores: Path):
    manifest = json.loads((dataset.parent / "manifest.json").read_text())
    if digest(dataset) != manifest["panels_sha256"]:
        raise ValueError("Dataset hash differs from frozen manifest")
    validation = {row["state_id"]: row for line in dataset.open() if line.strip()
                  for row in [json.loads(line)] if row["fold"] == 4}
    predictions = {}
    for line in scores.open():
        row = json.loads(line)
        if row["state_id"] in predictions:
            raise ValueError("Duplicate scored state")
        predictions[row["state_id"]] = row
    if set(predictions) != set(validation):
        raise ValueError("Validation scoring is incomplete or contains extra states")
    by_task = defaultdict(list)
    for state, source in validation.items():
        row = predictions[state]
        if row["task_group"] != source["task_group"] or row["holdout"] != source["pooled_action_holdout_rate"]:
            raise ValueError("Prediction and frozen outcome identity mismatch")
        holdout = row["holdout"]
        value = {"uniform": mean(holdout),
                 "actor_first": holdout[source.get("actor_first_action_index", 0)]}
        for name in ("actor", "positive", "implicit"):
            logits = row[name]
            if len(logits) != len(source["actions"]):
                raise ValueError("Score count differs from the saved candidate set")
            maximum = max(logits)
            winners = [i for i, score in enumerate(logits) if score == maximum]
            value[name] = mean([holdout[i] for i in winners])
        by_task[row["task_group"]].append(value)
    names = ("uniform", "actor_first", "actor", "positive", "implicit")
    states = [state for group in by_task.values() for state in group]
    point = {name: 100 * mean([state[name] for state in states]) for name in names}
    groups = list(by_task.values())
    rng = random.Random(20261021)
    deltas = {name: [] for name in ("implicit_minus_actor", "implicit_minus_positive",
                                    "positive_minus_actor")}
    for _ in range(10000):
        sample = [state for _ in groups for state in rng.choice(groups)]
        deltas["implicit_minus_actor"].append(100 * mean([s["implicit"] - s["actor"] for s in sample]))
        deltas["implicit_minus_positive"].append(100 * mean([s["implicit"] - s["positive"] for s in sample]))
        deltas["positive_minus_actor"].append(100 * mean([s["positive"] - s["actor"] for s in sample]))
    for values in deltas.values():
        values.sort()
    return {"states": len(states), "tasks": len(groups), "dataset_sha256": digest(dataset),
            "scores_sha256": digest(scores), "success_pct": point,
            "paired_pp": {name: {"estimate": 100 * mean([
                state[name.split("_minus_")[0]] - state[name.split("_minus_")[1]] for state in states]),
                "95pct_task_bootstrap": [values[250], values[9749]]}
                for name, values in deltas.items()},
            "endpoint": "valid-only held-out continuation success on saved distinct actions; draws " +
                        ",".join(map(str, manifest["holdout_draws"])),
            "limits": "Teacher-forced complete-action ranking only; no free-running or live guided decoding"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.dataset, args.scores)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
