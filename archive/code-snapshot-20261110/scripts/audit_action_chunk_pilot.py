#!/usr/bin/env python3
"""Independent CPU audit of frozen tool-name choices and their uncertainty.

The aggregate is safe for project docs. Per-state diagnostics stay in runtime.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import statistics

import numpy as np
from scipy.linalg import solve
import torch


def rows(path):
    with path.open() as stream:
        # The archived screenshots are large and irrelevant to this CPU audit.
        # Drop them as each line is parsed to keep login-node memory bounded.
        keep = ("state_id", "task_group", "fold", "source_plan", "actions", "outcomes")
        return [{key: value[key] for key in keep} for line in stream if line.strip()
                for value in [json.loads(line)]]


def outcome_rates(row, draw_ids):
    counts = defaultdict(lambda: [0, 0, 0])
    for action, outcomes in zip(row["actions"], row["outcomes"]):
        for repeat in draw_ids:
            result = outcomes[repeat]
            counts[action["name"]][2] += 1
            if result["valid"]:
                counts[action["name"]][0] += int(result["success"])
                counts[action["name"]][1] += 1
    return {name: dict(successes=value[0], valid=value[1], recorded=value[2],
                       rate=value[0] / value[1] if value[1] else None)
            for name, value in counts.items()}


def quantiles(values):
    if not values:
        return None
    return {key: float(np.quantile(values, q)) for key, q in
            (("min", 0), ("p10", .1), ("median", .5), ("p90", .9), ("max", 1))}


def fit_predictions(panels):
    """Recompute the five task-disjoint ridge fits; verify saved picks later."""
    predictions, alphas = {}, {}
    for fold in range(5):
        train = [p for p in panels if p["fold"] != fold]
        test = [p for p in panels if p["fold"] == fold]
        X = np.concatenate([p["h"] for p in train])
        Y = np.concatenate([p["target"] for p in train])
        groups = np.concatenate([[p["task_group"]] * len(p["names"]) for p in train])
        weights = np.concatenate([[1 / len(p["names"])] * len(p["names"]) for p in train])
        mu, sd = X.mean(0), X.std(0) + 1e-5
        X = (X - mu) / sd
        gram = X @ X.T
        tasks = sorted(set(groups), key=lambda name: hashlib.sha256(name.encode()).hexdigest())

        def dual(indices, alpha):
            root = np.sqrt(weights[indices])
            matrix = gram[np.ix_(indices, indices)] * root[:, None] * root[None, :]
            matrix.flat[::len(indices) + 1] += alpha
            return root * solve(matrix, Y[indices] * root,
                                assume_a="pos", check_finite=False)

        losses = {}
        for alpha in (1e2, 1e4, 1e6):
            scores = []
            for k in range(3):
                held_tasks = set(tasks[k::3])
                held = np.array([i for i, task in enumerate(groups) if task in held_tasks])
                fit = np.array([i for i, task in enumerate(groups) if task not in held_tasks])
                prediction = gram[np.ix_(held, fit)] @ dual(fit, alpha)
                scores.append(np.average((prediction - Y[held]) ** 2,
                                         weights=weights[held]))
            losses[alpha] = float(np.mean(scores))
        alpha = min(losses, key=lambda value: (losses[value], value))
        alphas[str(fold)] = alpha
        head = X.T @ dual(np.arange(len(X)), alpha)
        for panel in test:
            scores = ((panel["h"] - mu) / sd) @ head
            predictions[panel["state_id"]] = dict(zip(panel["names"], map(float, scores)))
    return predictions, alphas


def audit(root):
    panels = rows(root / "panels.jsonl")
    features = [item for shard in sorted((root / "results").glob("features-*.pt"))
                for item in torch.load(shard, map_location="cpu", weights_only=False)]
    picks = json.loads((root / "results" / "picks.json").read_text())
    metrics = json.loads((root / "results" / "metrics.json").read_text())
    if len(panels) != len(features) or len(panels) != len(picks):
        raise ValueError("Saved panel, feature and pick counts differ")
    by_pick = {row["state_id"]: row for row in picks}
    task_folds = defaultdict(set)
    fitted = []
    for row, feature in zip(panels, features):
        if row["state_id"] != feature["state_id"]:
            raise ValueError("Feature identity mismatch")
        task_folds[row["task_group"]].add(row["fold"])
        by_name = {}
        for option in feature["options"]:
            name = option["name"]
            if name not in by_name or option["logprob"] > by_name[name]["logprob"]:
                by_name[name] = option
        fit, hold = outcome_rates(row, range(5)), outcome_rates(row, range(5, 8))
        names = sorted(by_name)
        if any(fit[name]["rate"] is None or hold[name]["rate"] is None for name in names):
            raise ValueError("A tool name lacks valid fit or held-out evidence")
        h = np.stack([by_name[name]["h"].float().numpy() for name in names])
        h -= h.mean(0, keepdims=True)
        target = np.array([fit[name]["rate"] for name in names])
        target -= target.mean()
        fitted.append(dict(state_id=row["state_id"], task_group=row["task_group"],
                           fold=row["fold"], source=row["source_plan"],
                           names=names, actor={name: by_name[name]["logprob"] for name in names},
                           h=h, target=target, fit=fit, hold=hold,
                           multiple_calls=any(len(json.loads(a["action_key"])) > 1
                                              for a in row["actions"])))
    if any(len(value) != 1 for value in task_folds.values()):
        raise ValueError("Task group crosses folds")
    predictions, alphas = fit_predictions(fitted)
    if alphas != metrics["fold_alpha"]:
        raise ValueError("Regularization selection differs from original fit")

    detail = []
    for panel in fitted:
        names = panel["names"]
        actor_order = sorted(names, key=lambda name: (-panel["actor"][name], names.index(name)))
        value_order = sorted(names, key=lambda name: (-predictions[panel["state_id"]][name], names.index(name)))
        actor, value = actor_order[0], value_order[0]
        saved = by_pick[panel["state_id"]]
        if actor != saved["actor_name"] or value != saved["value_name"]:
            raise ValueError("Recomputed name choice differs from saved pick")
        a, v = panel["hold"][actor], panel["hold"][value]
        if abs(a["rate"] - saved["actor_success"]) > 1e-9 or abs(v["rate"] - saved["value_success"]) > 1e-9:
            raise ValueError("Recomputed success differs from saved pick")
        detail.append(dict(state_id=panel["state_id"], task_group=panel["task_group"],
                           fold=panel["fold"], source=panel["source"],
                           multiple_calls=panel["multiple_calls"], names=len(names),
                           actor_name=actor, value_name=value, changed=actor != value,
                           actor_margin_nats=panel["actor"][actor_order[0]] - panel["actor"][actor_order[1]],
                           value_margin_rate=predictions[panel["state_id"]][value_order[0]]
                                             - predictions[panel["state_id"]][value_order[1]],
                           actor_hold=a, value_hold=v,
                           actor_fit=panel["fit"][actor], value_fit=panel["fit"][value],
                           delta_pp=100 * (v["rate"] - a["rate"])))
    if sum(row["changed"] for row in detail) != metrics["changed_name_states"]:
        raise ValueError("Changed-choice count differs from saved report")
    if abs(statistics.mean(row["delta_pp"] for row in detail) - metrics["value_minus_actor_pp"][0]) > 1e-9:
        raise ValueError("Paired mean differs from saved report")

    changed = [row for row in detail if row["changed"]]
    deltas = [row["delta_pp"] for row in changed]
    task_deltas = defaultdict(list)
    for row in detail:
        task_deltas[row["task_group"]].append(row["delta_pp"])
    contributions = [sum(values) / len(detail) for values in task_deltas.values()]
    absolute = sorted((abs(value) for value in contributions), reverse=True)
    task_mean = [statistics.mean(values) for values in task_deltas.values()]
    rng = random.Random(0)
    equal_task_bootstrap = sorted(statistics.mean(rng.choices(task_mean, k=len(task_mean)))
                                  for _ in range(5000))
    leave_one = [statistics.mean(other["delta_pp"] for other in detail
                                 if other["task_group"] != task)
                 for task in task_deltas]
    by_source = defaultdict(list)
    for row in detail:
        by_source[Path(row["source"]).parent.parent.name].append(row)

    def summary(subset):
        if not subset:
            return None
        return dict(states=len(subset), task_groups=len({r["task_group"] for r in subset}),
                    changed=sum(r["changed"] for r in subset),
                    value_minus_actor_pp=statistics.mean(r["delta_pp"] for r in subset),
                    actor_held_valid_fraction=statistics.mean(r["actor_hold"]["valid"] /
                                                                r["actor_hold"]["recorded"] for r in subset),
                    value_held_valid_fraction=statistics.mean(r["value_hold"]["valid"] /
                                                                r["value_hold"]["recorded"] for r in subset))

    aggregate = dict(status="independent_cpu_audit", states=len(detail),
                     task_groups=len(task_deltas), changed_states=len(changed),
                     task_equal_value_minus_actor_pp=statistics.mean(task_mean),
                     task_equal_bootstrap_95pct_pp=[equal_task_bootstrap[125], equal_task_bootstrap[4875]],
                     changed_outcomes=dict(value_better=sum(x > 0 for x in deltas),
                                           actor_better=sum(x < 0 for x in deltas),
                                           tied=sum(x == 0 for x in deltas)),
                     overall=summary(detail), changed_only=summary(changed),
                     by_source={name: summary(group) for name, group in sorted(by_source.items())},
                     single_call=summary([r for r in detail if not r["multiple_calls"]]),
                     multiple_call=summary([r for r in detail if r["multiple_calls"]]),
                     actor_margin_nats=quantiles([r["actor_margin_nats"] for r in detail]),
                     value_margin_rate=quantiles([r["value_margin_rate"] for r in detail]),
                     changed_actor_margin_nats=quantiles([r["actor_margin_nats"] for r in changed]),
                     changed_value_margin_rate=quantiles([r["value_margin_rate"] for r in changed]),
                     task_concentration=dict(tasks_with_positive_net=sum(sum(v) > 0 for v in task_deltas.values()),
                                             tasks_with_negative_net=sum(sum(v) < 0 for v in task_deltas.values()),
                                             tasks_with_zero_net=sum(sum(v) == 0 for v in task_deltas.values()),
                                             largest_task_states=max(map(len, task_deltas.values())),
                                             median_task_states=statistics.median(map(len, task_deltas.values())),
                                             largest_absolute_contribution_pp=max(absolute),
                                             top_five_share_of_absolute_contribution=sum(absolute[:5]) / sum(absolute),
                                             leave_one_task_out_delta_pp_range=[min(leave_one), max(leave_one)]),
                     caveat="Exploratory audit of the same selected states; no new validation or causal return labels")
    return aggregate, detail


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--private-detail", type=Path, required=True)
    args = parser.parse_args()
    aggregate, detail = audit(args.root)
    args.aggregate.parent.mkdir(parents=True, exist_ok=True)
    args.private_detail.parent.mkdir(parents=True, exist_ok=True)
    args.aggregate.write_text(json.dumps(aggregate, indent=2) + "\n")
    args.private_detail.write_text(json.dumps(detail, indent=2) + "\n")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
