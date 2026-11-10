#!/usr/bin/env python3
"""Freeze a small, task-disjoint action-token fork pilot without GPU or API calls."""
import argparse
from collections import defaultdict, Counter
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

SFT = "/gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT"
STATES = Path("/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/"
              "joint-data-inputs-0d83b48/openwebrl_actor/states_full.jsonl")
BRANCH = Path("/gpfs/scrubbed/zixianma/openwebrl-runtime/selection-head-20261008/branch124")
SEED = "action-token-q-surrogate-v1-20261014"
END = "<|im_end|>"


def order(value):
    return hashlib.sha256(f"{SEED}|{value}".encode()).hexdigest()


def read_jsonl(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def choose_surrogate(path):
    groups = defaultdict(lambda: defaultdict(list))
    for row in read_jsonl(path):
        groups[row["split"]][row["task_group"]].append(row)
    selected = []
    for split, maximum in (("train", 256), ("validation", 38)):
        names = sorted(groups[split], key=order)[:maximum]
        for group in names:
            bases = defaultdict(list)
            for row in groups[split][group]:
                bases[row["base_state_id"]].append(row)
            base = min(bases, key=order)
            selected.extend(sorted(bases[base], key=lambda r: order(r["state_id"]))[:2])
    return selected


def first_fork(left, right, action_start):
    index = next((i for i, (a, b) in enumerate(zip(left, right)) if a != b), None)
    if index is None or index < action_start:
        return None
    return tuple(left[:index]), left[index], right[index]


def forks(tokenizer, shared, suffixes):
    responses = [shared + suffix for suffix in suffixes]
    ids = [tokenizer.encode(response if response.endswith(END) else response + END,
                            add_special_tokens=False) for response in responses]
    close = tokenizer.convert_tokens_to_ids("</think>")
    closes = [[i for i, token in enumerate(sequence) if token == close] for sequence in ids]
    if any(len(indices) != 1 for indices in closes):
        raise ValueError("Expected one </think> token per recontextualized response")
    action_start = closes[0][0] + 1
    if any(sequence[:action_start] != ids[0][:action_start] for sequence in ids):
        raise ValueError("Recontextualized responses do not share the tokenized reasoning prefix")
    grouped = defaultdict(list)
    for i in range(5):
        for j in range(i + 1, 5):
            branch = first_fork(ids[i], ids[j], action_start)
            if branch is None:
                continue
            prefix, left, right = branch
            grouped[prefix].append([i, j, left, right])
    return [{"prefix_ids": list(prefix), "comparisons": comparisons}
            for prefix, comparisons in sorted(grouped.items(), key=lambda item: item[0])]


def prepare(surrogate, output, tokenizer_path=SFT):
    output.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    selected = choose_surrogate(surrogate)
    bases = {r["base_state_id"] for r in selected}
    prompts = {r["state_id"]: r["prompt_text"] for r in read_jsonl(STATES) if r["state_id"] in bases}
    branch_prompts = {r["state_id"]: r["prompt_text"] for r in read_jsonl(BRANCH / "states.jsonl")}
    counts = Counter()
    rows = []
    for row in selected:
        prompt = prompts[row["base_state_id"]]
        if hashlib.sha256(prompt.encode()).hexdigest() != row["prompt_sha256"]:
            raise ValueError("Piotr prompt hash mismatch")
        fs = forks(tok, row["shared_reasoning_prefix"], row["action_suffixes"])
        positives, negatives = set(row["positive_indices"]), set(row["negative_indices"])
        comparisons = sum(sum(((i in positives and j in negatives) or
                               (j in positives and i in negatives)) for i, j, *_ in f["comparisons"])
                          for f in fs)
        if not comparisons:
            counts[f"{row['split']}_no_token_fork"] += 1
            continue
        rows.append({"state_id": row["state_id"], "task_group": row["task_group"],
                     "split": row["split"], "prompt_text": prompt, "image": row["image"],
                     "prompt_sha256": row["prompt_sha256"], "positive_indices": sorted(positives),
                     "negative_indices": sorted(negatives), "forks": fs})
        counts[f"{row['split']}_panels"] += 1
        counts[f"{row['split']}_labeled_token_comparisons"] += comparisons
    for row in read_jsonl(BRANCH / "panels.jsonl"):
        prompt = branch_prompts[row["state_id"]]
        if row.get("prompt_verified") and hashlib.sha256(prompt.encode()).hexdigest() != row["prompt_sha256"]:
            raise ValueError("Branch prompt hash mismatch")
        if not all("</think>" in x for x in row["candidates"]):
            counts["branch_missing_boundary"] += 1
            continue
        shared = row["candidates"][0].split("</think>", 1)[0] + "</think>"
        suffixes = [x.split("</think>", 1)[1] for x in row["candidates"]]
        rows.append({"state_id": row["state_id"], "task_group": row["task_group"],
                     "split": "branch124", "prompt_text": prompt, "image": row["image"],
                     "prompt_sha256": row["prompt_sha256"], "prompt_verified": row["prompt_verified"],
                     "outcomes": row["outcomes"], "forks": forks(tok, shared, suffixes)})
        counts["branch_panels"] += 1
    path = output / "forks.jsonl"
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {"status": "cpu_prepared_no_gpu_or_api_approval", "seed": SEED,
                "surrogate_sha256": hashlib.sha256(Path(surrogate).read_bytes()).hexdigest(),
                "forks_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "tokenizer_path": str(tokenizer_path), "train_task_groups_max": 256,
                "panels_per_task_group_max": 2, "validation_task_groups_max": 38,
                "counts": dict(sorted(counts.items())),
                "objective": "Observed first action-token fork; inherited original GPT-5.5 full-response preference"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surrogate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.surrogate, args.output)["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
