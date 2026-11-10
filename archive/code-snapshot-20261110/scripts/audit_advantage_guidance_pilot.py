#!/usr/bin/env python3
"""Audit action-only AG pilot labels against untouched valid-only holdout draws."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random

from prepare_action_chunk_outcome_pilot import digest


def mean(values):
    return sum(values) / len(values)


def evaluate(rows):
    if not rows:
        return None
    grouped = defaultdict(list)
    for row in rows:
        if not all(item["valid"] > 0 for item in row["fit"] + row["holdout"]):
            raise ValueError("Missing valid action evidence")
        fit = row["pooled_action_fit_rate"]
        hold = row["pooled_action_holdout_rate"]
        if any(value is None for value in fit + hold):
            raise ValueError("Action-equivalent pooled target is missing")
        winners = [i for i, value in enumerate(fit) if value == max(fit)]
        first = row.get("actor_first_action_index", 0)
        grouped[row["task_group"]].append({"actor_first": hold[first],
            "uniform": mean(hold), "fit_best": mean([hold[i] for i in winners]),
            "fit_ties": len(winners) > 1, "all_equal_fit": len(winners) == len(fit)})
    flat = [state for states in grouped.values() for state in states]
    names = ("actor_first", "uniform", "fit_best")
    point = {name: mean([state[name] for state in flat]) for name in names}
    rng = random.Random(20261021)
    groups = list(grouped.values())
    delta = []
    for _ in range(10000):
        sample = [state for _ in groups for state in rng.choice(groups)]
        delta.append(mean([state["fit_best"] - state["uniform"] for state in sample]))
    delta.sort()
    return {"states": len(flat), "task_groups": len(groups),
            "success_pct": {name: 100 * value for name, value in point.items()},
            "fit_best_minus_uniform_pp": 100 * (point["fit_best"] - point["uniform"]),
            "fit_best_minus_uniform_95pct_task_bootstrap_pp":
                [100 * delta[250], 100 * delta[9749]],
            "fit_ties": sum(state["fit_ties"] for state in flat),
            "all_equal_fit": sum(state["all_equal_fit"] for state in flat)}


def audit(path: Path):
    manifest = json.loads((path.parent / "manifest.json").read_text())
    if digest(path) != manifest["panels_sha256"]:
        raise ValueError("Panel file differs from pinned manifest")
    rows = [json.loads(line) for line in path.open() if line.strip()]
    admitted = manifest["counts"].get("admitted_states",
                                     manifest["counts"].get("train_admitted_states", 0) +
                                     manifest["counts"].get("test_admitted_states", 0))
    if len(rows) != admitted:
        raise ValueError("Panel count differs from pinned manifest")
    roles = {"all": rows, "train_folds_0_to_3": [row for row in rows if row["fold"] < 4],
             "validation_fold_4": [row for row in rows if row["fold"] == 4]}
    if {row["task_group"] for row in roles["train_folds_0_to_3"]} & \
       {row["task_group"] for row in roles["validation_fold_4"]}:
        raise ValueError("Task group leaks across training/validation")
    return {"scope": manifest.get("source", "pinned Piotr-matched states"),
            "holdout_draws": manifest["holdout_draws"],
            "panels_sha256": digest(path),
            "counts": manifest["counts"],
            "roles": {name: evaluate(data) for name, data in roles.items()},
            "caveat": "Fit labels estimate panel-relative action success, not true policy advantage; the empty-reasoning prefix is untested"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.panels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["roles"], sort_keys=True))


if __name__ == "__main__":
    main()
