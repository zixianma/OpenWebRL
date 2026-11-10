#!/usr/bin/env python3
"""Prepare action-only advantage-guidance data from audited branching panels.

This is a CPU-only preflight. It never trains a model or calls a browser/API.
The private output contains prompts, images and actions; publish aggregates only.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from prepare_action_chunk_outcome_pilot import (
    digest, outcomes, read, repeat_zero_prefixes, task_group,
)


DEFAULT_INDEX = Path(
    "/gpfs/scrubbed/zixianma/openwebrl-runtime/selection-head-20261008/"
    "state-inventory-ids-rl-20261017.json"
)
SEED = "advantage-guidance-action-only-v4-20261021"
FIT_DRAWS = range(6)
HOLD_DRAWS = range(6, 8)


def action_suffix(response: str) -> str:
    if response.count("</think>") != 1:
        raise ValueError("Action lacks one reasoning boundary")
    suffix = response.split("</think>", 1)[1]
    if "<tool_call>" not in suffix or "</tool_call>" not in suffix or "<|im_end|>" not in suffix:
        raise ValueError("Action suffix lacks a complete tool call")
    if suffix.find("<tool_call>") > suffix.find("</tool_call>"):
        raise ValueError("Tool-call delimiters are reversed")
    return suffix[:suffix.index("<|im_end|>") + len("<|im_end|>")]


def action_only_prompt(prompt: str) -> str:
    # The saved SFT prompt ends inside <think>; no candidate's original
    # reasoning is reused to represent another candidate's outcome.
    if not prompt.endswith("<think>\n"):
        raise ValueError("Unexpected actor prompt boundary")
    return prompt + "</think>"


def evidence(cell: dict, candidate: int, draws: range) -> dict:
    values = [cell[candidate, draw] for draw in draws]
    valid = [value for value in values if value["valid"]]
    return {"recorded": len(values), "valid": len(valid),
            "invalid": len(values) - len(valid),
            "successes": sum(value["success"] for value in valid),
            "rate": sum(value["success"] for value in valid) / len(valid) if valid else None}


def pooled_rates(action_keys: list[str], evidence_by_candidate: list[dict]) -> list[float | None]:
    if len(action_keys) != 5 or len(evidence_by_candidate) != 5:
        raise ValueError("Expected five aligned action records")
    pooled = {}
    for key in set(action_keys):
        indices = [index for index, value in enumerate(action_keys) if value == key]
        valid = sum(evidence_by_candidate[index]["valid"] for index in indices)
        pooled[key] = (sum(evidence_by_candidate[index]["successes"] for index in indices) / valid
                       if valid else None)
    return [pooled[key] for key in action_keys]


def balance_folds(rows: list[dict]) -> None:
    groups = defaultdict(list)
    for row in rows:
        groups[row["task_group"]].append(row)
    sizes = [0] * 5
    for group in sorted(groups, key=lambda value: (-len(groups[value]),
                        hashlib.sha256(f"{SEED}|{value}".encode()).hexdigest())):
        fold = min(range(5), key=lambda value: (sizes[value], value))
        for row in groups[group]:
            row["fold"] = fold
        sizes[fold] += len(groups[group])


def prepare(index_path: Path, output: Path, limit: int | None = None,
            scope: str = "piotr") -> dict:
    if scope not in ("piotr", "all"):
        raise ValueError("Scope must be piotr or all")
    inventory = read(index_path)
    if "meta" not in inventory or "summary" not in inventory:
        raise ValueError("Expected an independently audited state inventory")
    meta = inventory["meta"]
    chosen = sorted((state, item) for state, item in meta.items()
                    if (scope == "all" or item["piotr"]) and item["draw0"] and item["informative"])
    if limit is not None:
        chosen = chosen[:limit]
    entries = [{"state": state, "root": item["root"], "piotr": item["piotr"]}
               for state, item in chosen]
    cells = outcomes(entries)
    counts = Counter()
    rows = []
    for entry in entries:
        counts["indexed_states"] += 1
        state = entry["state"]
        cell = cells[state]
        if any((candidate, draw) not in cell for candidate in range(5) for draw in range(8)):
            counts["excluded_incomplete_recorded_panel"] += 1
            continue
        ref_path = Path(entry["root"]) / "states" / state / "anchor-reference.json"
        if not ref_path.exists():
            counts["excluded_missing_anchor"] += 1
            continue
        ref = read(ref_path)
        if digest(ref["path"]) != ref["sha256"]:
            raise ValueError("Anchor hash mismatch: " + state)
        anchor = read(ref["path"])
        if len(anchor["candidates"]) != 5:
            counts["excluded_nonfive_candidates"] += 1
            continue
        try:
            prompt, images = repeat_zero_prefixes(entry, anchor)
            common = action_only_prompt(prompt)
            actions = [action_suffix(candidate["output"][0])
                       for candidate in anchor["candidates"]]
            if any(candidate["output"][3] != "stop" for candidate in anchor["candidates"]):
                raise ValueError("Truncated candidate")
        except (KeyError, IndexError, ValueError) as error:
            counts["excluded_" + type(error).__name__ + "_" +
                   str(error).lower().replace(" ", "_")[:55]] += 1
            continue
        fit = [evidence(cell, candidate, FIT_DRAWS) for candidate in range(5)]
        hold = [evidence(cell, candidate, HOLD_DRAWS) for candidate in range(5)]
        if any(item["valid"] == 0 for item in fit):
            counts["excluded_candidate_without_valid_fit"] += 1
            continue
        if any(item["valid"] == 0 for item in hold):
            counts["excluded_candidate_without_valid_holdout"] += 1
            continue
        group = task_group(anchor["task"])
        action_keys = [candidate["action_key"] for candidate in anchor["candidates"]]
        rates = pooled_rates(action_keys, fit)
        hold_rates = pooled_rates(action_keys, hold)
        baseline = sum(rates) / 5
        record = {"state_id": state, "task_id": anchor["task_id"],
                  "task_group": group, "fold": None, "source_root": entry["root"],
                  "piotr_task_matched": entry["piotr"],
                  "source_anchor_sha256": ref["sha256"],
                  "actor_input_sha256": anchor["prompt_input_sha256"],
                  "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                  "action_only_prompt": common, "images": images,
                  "actions": actions, "action_keys": action_keys,
                  "fit": fit, "holdout": hold,
                  "pooled_action_fit_rate": rates, "pooled_action_holdout_rate": hold_rates,
                  "panel_relative_advantage": [value - baseline for value in rates]}
        rows.append(record)
        counts["admitted_states"] += 1
        counts["admitted_piotr_task_matched"] += bool(entry["piotr"])
        counts["admitted_valid_fit"] += sum(item["valid"] for item in fit)
        counts["admitted_invalid_fit"] += sum(item["invalid"] for item in fit)
        counts["admitted_valid_holdout"] += sum(item["valid"] for item in hold)
        counts["admitted_invalid_holdout"] += sum(item["invalid"] for item in hold)
        counts["admitted_distinct_executable_actions"] += len(set(record["action_keys"]))
        if max(rates) > min(rates):
            counts["admitted_informative_fit"] += 1
    balance_folds(rows)
    task_folds = defaultdict(set)
    for row in rows:
        task_folds[row["task_group"]].add(row["fold"])
    if any(len(folds) != 1 for folds in task_folds.values()):
        raise AssertionError("Task leaks across folds")
    output.mkdir(parents=True, exist_ok=True)
    path = output / "panels.jsonl"
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "status": "cpu_prepared_no_gpu_or_api_calls",
        "scope": scope,
        "input_index": str(index_path), "input_sha256": digest(index_path),
        "panels_sha256": digest(path), "seed": SEED,
        "fit_draws": list(FIT_DRAWS), "holdout_draws": list(HOLD_DRAWS),
        "target": "valid-only action-key-pooled fit success fraction minus equal-weight five-slot mean",
        "condition_model": "proposed Gaussian label Y|Ahat ~ N(Ahat, 0.25^2); not a claim about true advantage",
        "prefix": "same saved pre-action VLM input plus empty <think>...</think> for all five actions",
        "caveat": "Action outcomes were collected with original candidate reasoning; action-only prefix is a new input distribution",
        "counts": dict(sorted(counts.items())),
        "folds": {str(fold): {"states": sum(row["fold"] == fold for row in rows),
                              "tasks": len({row["task_group"] for row in rows if row["fold"] == fold})}
                  for fold in range(5)},
        "dry_run_limit": limit,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--scope", choices=("piotr", "all"), default="piotr")
    args = parser.parse_args()
    print(json.dumps(prepare(args.index, args.output, args.limit, args.scope), sort_keys=True))


if __name__ == "__main__":
    main()
