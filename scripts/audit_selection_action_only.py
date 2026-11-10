#!/usr/bin/env python3
"""Audit whether Piotr's completed-response panels support action-only replay.

Emits aggregate counts and an input hash, never candidate text or task data.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def shared_reasoning_comparison(candidates, target, normalize=False):
    groups = defaultdict(list)
    for i, response in enumerate(candidates):
        if "</think>" not in response:
            continue
        reasoning, _ = response.split("</think>", 1)
        groups[" ".join(reasoning.split()) if normalize else reasoning].append(i)
    branches = [indices for indices in groups.values() if len(indices) > 1 and
                len({candidates[i].split("</think>", 1)[1] for i in indices}) > 1]
    comparison = any(any(target[i] > 0 for i in indices) and
                     any(target[i] == 0 for i in indices) for indices in branches)
    return bool(branches), comparison


def audit(path):
    digest = hashlib.sha256()
    counts = defaultdict(Counter)
    with Path(path).open("rb") as stream:
        for raw in stream:
            digest.update(raw)
            row = json.loads(raw)
            split = row["split"]
            values = counts[split]
            candidates, target, parseable = row["candidates"], row["target"], row["parseable"]
            winner = row["selection"]
            assert len(candidates) == len(target) == len(parseable) == 5
            values["panels"] += 1
            values["informative_action_equivalence"] += bool(row["informative"])
            usable = bool(parseable[winner] and any(parseable[i] and target[i] == 0 for i in range(5)))
            values["parseable_winner_distinct_parseable_loser"] += usable
            values["strict_all_five_think_boundaries"] += usable and all("</think>" in x for x in candidates)
            for label, normalize in (("exact", False), ("whitespace_normalized", True)):
                branch, comparison = shared_reasoning_comparison(candidates, target, normalize)
                values[f"{label}_same_reason_different_actions"] += branch
                values[f"{label}_teacher_comparison"] += comparison
    return {"source_sha256": digest.hexdigest(), "counts": {k: dict(v) for k, v in sorted(counts.items())},
            "definitions": {"strict_all_five_think_boundaries":
                            "Parseable teacher winner and distinct parseable loser; all five responses contain </think>.",
                            "teacher_comparison":
                            "One selected-class and one nonselected candidate share the same reasoning prefix but have different action suffixes."}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.panels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
