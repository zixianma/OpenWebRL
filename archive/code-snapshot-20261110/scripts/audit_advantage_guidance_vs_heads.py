#!/usr/bin/env python3
"""Paired offline action-choice audit on the verified SFT-action test states."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random

from prepare_action_chunk_outcome_pilot import digest


HEADS = ("outcome", "teacher_k3", "teacher_k1", "teacher_pre")


def mean(values):
    return sum(values) / len(values)


def choice(scores):
    winners = [i for i, value in enumerate(scores) if value == max(scores)]
    return winners


def audit(dataset: Path, score_file: Path, head_root: Path):
    manifest = json.loads((dataset.parent / "manifest.json").read_text())
    if digest(dataset) != manifest["panels_sha256"]:
        raise ValueError("Dataset hash differs from manifest")
    states = {row["state_id"]: row for line in dataset.open() if line.strip()
              for row in [json.loads(line)] if row["fold"] == 4}
    scores = {row["state_id"]: row for line in score_file.open() if line.strip()
              for row in [json.loads(line)]}
    if set(states) != set(scores):
        raise ValueError("Incomplete or extra actor scores")
    picks = {}
    hashes = {}
    for head in HEADS:
        path = head_root / head / "picks.json"
        record = json.loads(path.read_text())
        picks[head] = record["picks"]
        hashes[head] = digest(path)
        if not set(states) <= set(picks[head]):
            raise ValueError("Head lacks some verified test states: " + head)
    by_task = defaultdict(list)
    for sid, state in states.items():
        holdout = state["pooled_action_holdout_rate"]
        scored = scores[sid]
        if scored["holdout"] != holdout or scored["task_group"] != state["task_group"]:
            raise ValueError("Score and dataset identity mismatch")
        row = {"uniform": mean(holdout),
               "actor_first": holdout[state["actor_first_action_index"]]}
        for model in ("actor", "positive", "implicit"):
            if len(scored[model]) != len(holdout):
                raise ValueError("Actor candidate count mismatch")
            row[model] = mean([holdout[i] for i in choice(scored[model])])
        for head in HEADS:
            selected = picks[head][sid]
            if not isinstance(selected, int) or not 0 <= selected < len(holdout):
                raise ValueError("Head pick outside candidate set")
            row[head] = holdout[selected]
        by_task[state["task_group"]].append(row)
    flat = [row for group in by_task.values() for row in group]
    groups = list(by_task.values())
    models = ["actor_first", "uniform", "actor", "positive", "implicit", *HEADS]
    rng = random.Random(20261023)
    differences = {(model, reference): [] for model in models
                   for reference in ("actor", "uniform") if model != reference}
    for _ in range(10000):
        sample = [row for _ in groups for row in rng.choice(groups)]
        for (model, reference), values in differences.items():
            values.append(100 * mean([row[model] - row[reference] for row in sample]))
    return {"states": len(flat), "tasks": len(groups),
            "endpoint": "valid-only outcome per state on continuation draws 3-7, same distinct action keys",
            "candidate_view": "2-5 distinct SFT-proposed actions; head sees original candidate reasoning, AG sees empty-reasoning prefix",
            "dataset_sha256": digest(dataset), "actor_scores_sha256": digest(score_file),
            "head_pick_sha256": hashes,
            "success_pct": {model: 100 * mean([row[model] for row in flat]) for model in models},
            "paired_pp": {model + "_minus_" + reference: {
                "estimate": 100 * mean([row[model] - row[reference] for row in flat]),
                "95pct_task_bootstrap": [sorted(values)[250], sorted(values)[9749]]}
                for (model, reference), values in differences.items()},
            "limits": "Teacher-forced offline ranking; no free-running or live guided decoding"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--heads", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.dataset, args.scores, args.heads)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"states": report["states"], "tasks": report["tasks"],
                      "success_pct": report["success_pct"],
                      "implicit_minus_actor": report["paired_pp"]["implicit_minus_actor"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
