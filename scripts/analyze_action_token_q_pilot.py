#!/usr/bin/env python3
"""Audit the frozen action-token-Q pilot with task-clustered paired intervals."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random

import torch
from torch.nn import functional as F


def mean(values):
    return sum(values) / len(values)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def interval(differences, clusters, seed=0, draws=10000):
    by_group = defaultdict(list)
    for difference, cluster in zip(differences, clusters):
        by_group[cluster].append(difference)
    groups = sorted(by_group)
    rng = random.Random(seed)
    boots = []
    for _ in range(draws):
        sampled = [rng.choice(groups) for _ in groups]
        boots.append(mean([x for group in sampled for x in by_group[group]]))
    boots.sort()
    return [boots[int(.025 * draws)], boots[int(.975 * draws)]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = {row["state_id"]: row for line in (args.root / "forks.jsonl").open()
              if (row := json.loads(line))}
    rows = [row for path in sorted((args.root / "results").glob("features-*.pt"))
            for row in torch.load(path, map_location="cpu", weights_only=False)]
    model = torch.load(args.root / "results" / "token-q.pt", map_location="cpu", weights_only=False)
    weights = model["state_dict"]
    embeddings = torch.load(args.root / "results" / "token-embeddings.pt", map_location="cpu", weights_only=False)
    tokens = {token: vector.float() for token, vector in zip(embeddings["token_ids"], embeddings["embeddings"])}
    dimension = model["hidden_size"]
    projected = {token: F.linear(F.layer_norm(vector, (dimension,)), weights["token.weight"])
                 for token, vector in tokens.items()}
    bias = {token: float(F.linear(F.layer_norm(vector, (dimension,)), weights["token_bias.weight"]))
            for token, vector in tokens.items()}

    def scores(fork):
        h = F.linear(F.layer_norm(fork["h"].float(), (dimension,)), weights["state.weight"])
        q = {token: float((h * projected[token]).sum() / math.sqrt(32)) + bias[token]
             for _, _, a, b in fork["comparisons"] for token in (a, b)}
        return {"actor": fork["actor_logits"], "q": q}

    validation, branch = [], []
    for row in rows:
        original = source[row["state_id"]]
        if row["split"] == "validation":
            correct = {"actor": [], "q": []}
            margins = []
            for fork in row["forks"]:
                score = scores(fork)
                for i, j, a, b in fork["comparisons"]:
                    if i in row["positive_indices"] and j in row["negative_indices"]:
                        winner, loser = a, b
                    elif j in row["positive_indices"] and i in row["negative_indices"]:
                        winner, loser = b, a
                    else:
                        continue
                    for mode in correct:
                        difference = score[mode][winner] - score[mode][loser]
                        correct[mode].append(1.0 if difference > 0 else .5 if difference == 0 else 0.0)
                    margins.append((score["actor"][winner] - score["actor"][loser],
                                    score["q"][winner] - score["q"][loser]))
            if correct["actor"]:
                validation.append({"task_group": original["task_group"],
                                   "actor": mean(correct["actor"]), "q": mean(correct["q"]),
                                   "margins": margins})
        elif row["split"] == "branch124" and row["forks"]:
            votes = {mode: [0.0] * 5 for mode in ("actor", "q")}
            for fork in row["forks"]:
                score = scores(fork)
                for i, j, a, b in fork["comparisons"]:
                    for mode in votes:
                        difference = score[mode][a] - score[mode][b]
                        votes[mode][i] += 1.0 if difference > 0 else .5 if difference == 0 else 0.0
                        votes[mode][j] += 1.0 if difference < 0 else .5 if difference == 0 else 0.0
            outcomes = row["outcomes"]
            if not all(x["valid"] > 0 for x in outcomes):
                continue
            success = [x["successes"] / x["valid"] for x in outcomes]
            record = {"task_group": original["task_group"], "prompt_verified": row["prompt_verified"],
                      "uniform": mean(success)}
            for mode, count in votes.items():
                tied = [i for i, value in enumerate(count) if value == max(count)]
                record[mode] = success[tied[0]]
                record[mode + "_tie_averaged"] = mean([success[i] for i in tied])
                record[mode + "_tied"] = len(tied) > 1
            branch.append(record)

    assert len(validation) == 74 and len({r["task_group"] for r in validation}) == 38
    assert len(branch) == 101
    previous = json.loads((args.root / "results" / "metrics.json").read_text())
    for mode in ("actor", "q"):
        assert abs(mean([r[mode] for r in validation]) - previous[f"validation_{mode if mode == 'q' else 'actor'}_only"][0]) < 1e-6
        assert abs(mean([r[mode] for r in branch]) - previous["branch124_observed_fork_outcomes"]["fork_scored"][mode]["fully_valid_state_success_per_valid_draw"]) < 1e-6

    def contrast(items, left, right):
        differences = [r[left] - r[right] for r in items]
        return {"difference": mean(differences), "ci95_task_cluster": interval(differences, [r["task_group"] for r in items])}

    def guided_accuracy(beta):
        def correctness(actor, q):
            margin = actor + beta * q
            return 1.0 if margin > 0 else .5 if margin == 0 else 0.0
        return mean([mean([correctness(actor, q) for actor, q in row["margins"]])
                     for row in validation])

    result = {"job_id": "365307", "scope": "offline observed-fork probe; no decoding intervention",
              "source": {"forks_sha256": sha256(args.root / "forks.jsonl"),
                         "metrics_sha256": sha256(args.root / "results" / "metrics.json"),
                         "checkpoint_sha256": sha256(args.root / "results" / "token-q.pt")},
              "validation": {"panels": len(validation), "task_groups": len({r["task_group"] for r in validation}),
                             "actor_accuracy": mean([r["actor"] for r in validation]),
                             "q_accuracy": mean([r["q"] for r in validation]),
                             "q_minus_actor": contrast(validation, "q", "actor")},
              "posthoc_guidance_score_diagnostic": {
                  "definition": "Actor next-token logit plus beta times Q at observed forks; same held-out panels inspected after Q failed; no beta selection or free-running decoding",
                  "beta_panel_accuracy": [{"beta": beta, "accuracy": guided_accuracy(beta)}
                                          for beta in (0, .01, .03, .1, .3, 1, 3)]},
              "historical_branch_fully_valid": {"states": len(branch),
                  "actor_success": mean([r["actor"] for r in branch]),
                  "q_success": mean([r["q"] for r in branch]),
                  "uniform_success": mean([r["uniform"] for r in branch]),
                  "q_minus_actor": contrast(branch, "q", "actor"),
                  "q_minus_uniform": contrast(branch, "q", "uniform"),
                  "tie_averaged_q_minus_actor": contrast(branch, "q_tie_averaged", "actor_tie_averaged"),
                  "actor_ties": sum(r["actor_tied"] for r in branch),
                  "q_ties": sum(r["q_tied"] for r in branch),
                  "prompt_verified_states": sum(r["prompt_verified"] for r in branch)},
              "bootstrap": "10,000 paired task-cluster resamples; seed 0; percentile interval; fixed trained checkpoint",
              "limitations": "Teacher labels judged original complete responses. Historical outcomes reused three-draw states and are not an untouched test."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
