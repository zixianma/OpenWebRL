#!/usr/bin/env python3
"""Recontextualize Piotr action suffixes under candidate 0's reasoning.

This is surrogate supervision: GPT-5.5 chose among the original whole responses.
Keep the resulting raw action/reasoning text in the private runtime, not docs.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def build(path, output):
    output.mkdir(parents=True, exist_ok=True)
    source_hash = hashlib.sha256()
    counts = Counter()
    output_hash = hashlib.sha256()
    with path.open("rb") as source, (output / "panels.jsonl").open("wb") as destination:
        for raw in source:
            source_hash.update(raw)
            row = json.loads(raw)
            split = row["split"]
            counts[f"{split}_input"] += 1
            candidates, target, parseable = row["candidates"], row["target"], row["parseable"]
            if len(candidates) != 5 or not all("</think>" in response for response in candidates):
                counts[f"{split}_missing_boundary"] += 1
                continue
            if not parseable[row["selection"]]:
                counts[f"{split}_unparseable_winner"] += 1
                continue
            positive = [i for i in range(5) if target[i] > 0 and parseable[i]]
            negative = [i for i in range(5) if target[i] == 0 and parseable[i]]
            if not positive or not negative:
                counts[f"{split}_no_parseable_comparison"] += 1
                continue
            shared = candidates[0].split("</think>", 1)[0] + "</think>"
            suffixes = [response.split("</think>", 1)[1] for response in candidates]
            record = {"state_id": row["state_id"], "base_state_id": row["base_state_id"],
                      "task_group": row["task_group"], "split": split,
                      "prompt_sha256": row["prompt_sha256"], "image": row["image"],
                      "shared_reasoning_prefix": shared, "action_suffixes": suffixes,
                      "positive_indices": positive, "negative_indices": negative,
                      "teacher_selection_original_full_response": row["selection"],
                      "label_scope": "surrogate_original_full_response_choice"}
            encoded = (json.dumps(record, ensure_ascii=False) + "\n").encode()
            destination.write(encoded)
            output_hash.update(encoded)
            counts[f"{split}_output"] += 1
    manifest = {"status": "prepared_no_new_teacher_labels_or_gpu_work",
                "source_sha256": source_hash.hexdigest(),
                "output_sha256": output_hash.hexdigest(),
                "shared_reasoning_rule": "candidate index 0, independent of teacher selection",
                "pair_rule": "parseable selected-action class versus distinct parseable losers",
                "label_caveat": "GPT-5.5 judged original whole responses; preferences are inherited after reasoning replacement",
                "counts": dict(sorted(counts.items()))}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.panels, args.output)["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
