#!/usr/bin/env python3
"""Prepare the selection-head common SFT-action cohort for action-only AG.

The existing task split and distinct executable-action panels are preserved.
Only outcome draws 0-2 label actions; draws 3-7 remain evaluation-only.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from prepare_advantage_guidance_pilot import action_only_prompt, action_suffix
from prepare_action_chunk_outcome_pilot import digest


def read_jsonl(path: Path):
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def evidence(action: dict, draws: range) -> dict:
    cells = [cell for draw in draws for cell in action["outcomes"][str(draw)]]
    if any(cell is None for cell in cells):
        raise ValueError("Missing recorded continuation cell")
    valid = [cell for cell in cells if cell[0]]
    successes = sum(bool(cell[1]) for cell in valid)
    return {"recorded": len(cells), "valid": len(valid),
            "invalid": len(cells) - len(valid), "successes": successes,
            "rate": successes / len(valid) if valid else None}


def prepare(data: Path, common_panels: Path, output: Path) -> dict:
    actions = {row["state_id"]: row for row in read_jsonl(data / "actions.jsonl")}
    prompts = {row["state_id"]: row["prompt_text"] for row in read_jsonl(data / "states.jsonl")}
    common = {row["base_state_id"] for row in read_jsonl(common_panels)}
    if not common <= actions.keys() or not common <= prompts.keys():
        raise ValueError("Common head states are missing from the SFT-action export")
    counts = Counter()
    rows = []
    for state in sorted(common):
        source = actions[state]
        split = source["split"]
        if split not in ("train", "test"):
            raise ValueError("Unrecognized source split")
        counts[f"{split}_common_states"] += 1
        if not source["prompt_verified"]:
            counts[f"{split}_excluded_unverified_prompt"] += 1
            continue
        prompt = prompts[state]
        if hashlib.sha256(prompt.encode()).hexdigest() != source["prompt_sha256"]:
            raise ValueError("Saved actor prompt digest mismatch: " + state)
        common_prefix = action_only_prompt(prompt)
        candidates = source["actions"]
        suffixes = [action_suffix(action["rep_text"]) for action in candidates]
        if not 2 <= len(candidates) <= 5 or len({action["key"] for action in candidates}) != len(candidates):
            raise ValueError("Expected 2-5 distinct executable actions: " + state)
        fit = [evidence(action, range(3)) for action in candidates]
        holdout = [evidence(action, range(3, 8)) for action in candidates]
        if any(item["valid"] == 0 for item in fit):
            counts[f"{split}_excluded_missing_valid_fit"] += 1
            continue
        if any(item["valid"] == 0 for item in holdout):
            counts[f"{split}_excluded_missing_valid_holdout"] += 1
            continue
        image = Path(source["image"])
        if not image.is_file():
            raise ValueError("Saved screenshot missing: " + state)
        fit_rates = [item["rate"] for item in fit]
        mean_rate = sum(fit_rates) / len(fit_rates)
        first = min(range(len(candidates)), key=lambda i: min(candidates[i]["sft_draws"]))
        row = {"state_id": state, "task_id": source["task_id"],
               "task_group": source["task_id"], "fold": 0 if split == "train" else 4,
               "source_split": split, "source_root": source["root"],
               "prompt_sha256": source["prompt_sha256"],
               "action_only_prompt": common_prefix,
               "image_path": str(image), "image_sha256": digest(image),
               "actions": suffixes, "action_keys": [action["key"] for action in candidates],
               "actor_first_action_index": first,
               "sft_proposal_counts": [len(action["sft_draws"]) for action in candidates],
               "fit": fit, "holdout": holdout,
               "pooled_action_fit_rate": fit_rates,
               "pooled_action_holdout_rate": [item["rate"] for item in holdout],
               "panel_relative_advantage": [rate - mean_rate for rate in fit_rates]}
        rows.append(row)
        counts[f"{split}_admitted_states"] += 1
        counts["admitted_actions"] += len(candidates)
        counts["valid_fit_draws"] += sum(item["valid"] for item in fit)
        counts["invalid_fit_draws"] += sum(item["invalid"] for item in fit)
        counts["valid_holdout_draws"] += sum(item["valid"] for item in holdout)
        counts["invalid_holdout_draws"] += sum(item["invalid"] for item in holdout)
        counts["informative_fit_states"] += max(fit_rates) > min(fit_rates)
    train_tasks = {row["task_group"] for row in rows if row["fold"] == 0}
    test_tasks = {row["task_group"] for row in rows if row["fold"] == 4}
    if train_tasks & test_tasks:
        raise ValueError("Task leaks across the inherited head split")
    output.mkdir(parents=True, exist_ok=True)
    panel_path = output / "panels.jsonl"
    with panel_path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "status": "cpu_prepared_no_gpu_or_api_calls",
        "source": "selection-head common SFT-action states; one row per base state",
        "actions_sha256": digest(data / "actions.jsonl"),
        "states_sha256": digest(data / "states.jsonl"),
        "head_panels_sha256": digest(common_panels),
        "panels_sha256": digest(panel_path),
        "fit_draws": [0, 1, 2], "holdout_draws": [3, 4, 5, 6, 7],
        "target": "valid-only pooled continuation success minus mean across distinct executable actions",
        "condition_model": "proposed Gaussian label around this panel-relative surrogate; not true advantage",
        "prefix": "same saved pre-action VLM input plus empty reasoning for all actions",
        "caveat": "Continuation outcomes were collected with each proposal's original reasoning, not the empty reasoning prefix",
        "counts": dict(sorted(counts.items())),
        "task_groups": {"train": len(train_tasks), "test": len(test_tasks)},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--common-panels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.data, args.common_panels, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
