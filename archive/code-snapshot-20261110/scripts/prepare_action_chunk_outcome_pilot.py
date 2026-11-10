#!/usr/bin/env python3
"""Freeze a private, outcome-labelled tool-name chunk pilot without GPU/API calls.

The fixed five candidate actions are replayed after candidate 0's reasoning.
This preflight requires an identical archived actor prompt and image sequence in
all five repeat-0 branch rollouts. Draws 0-4 are fit evidence; 5-7 are held out.
Invalid or missing outcomes never become failures or successes.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import unicodedata


RUNTIME = Path("/gpfs/scrubbed/zixianma/openwebrl-runtime")
ROOT = RUNTIME / "selection-head-20261008"
PLANS = (ROOT / "lookahead-v2-20261013/k3/plan.json",
         ROOT / "lookahead-fast1k-20261013/k3/plan.json",
         ROOT / "lookahead-new-20261016/s-pre/plan.json")
CHUNK = re.compile(r'<tool_call>\s*\{\s*"name"\s*:\s*"((?:[^"\\]|\\.)*)"', re.S)
SEED = "action-chunk-outcome-tool-name-v3-fingerprinted-20261017"


def task_group(instruction):
    normalized = re.sub(r"[\W_]+", " ", unicodedata.normalize("NFKC", instruction).casefold())
    normalized = " ".join(normalized.split())
    return hashlib.sha256(normalized.encode()).hexdigest()


def balance_folds(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["task_group"]].append(row)
    sizes = [0] * 5
    for group in sorted(groups, key=lambda name: (-len(groups[name]),
                        hashlib.sha256(f"{SEED}|{name}".encode()).hexdigest())):
        fold = min(range(5), key=lambda candidate: (sizes[candidate], candidate))
        for row in groups[group]:
            row["fold"] = fold
        sizes[fold] += len(groups[group])
    return sizes


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def prompt_fingerprint(text, images):
    """Exact contract used by the frozen branching actor at candidate sampling."""
    image_hashes = [hashlib.sha256(json.dumps(image, sort_keys=True,
                        ensure_ascii=False).encode()).hexdigest() for image in images]
    payload = dict(text=text, images=image_hashes)
    return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                     ensure_ascii=False).encode()).hexdigest()


def candidates(anchor):
    if len(anchor["candidates"]) != 5:
        raise ValueError("Expected five fixed actions")
    shared = anchor["candidates"][0]["output"][0].split("</think>", 1)[0] + "</think>"
    actions = []
    for candidate in anchor["candidates"]:
        response = candidate["output"][0]
        if "</think>" not in response:
            raise ValueError("Missing reasoning boundary")
        suffix = response.split("</think>", 1)[1]
        match = CHUNK.search(suffix)
        calls = json.loads(candidate["action_key"])
        if not match or not calls or json.loads('"' + match.group(1) + '"') != calls[0]["name"]:
            raise ValueError("Tool-name chunk does not match executable action")
        actions.append(dict(chunk=suffix[:match.end()], name=calls[0]["name"],
                            action_key=candidate["action_key"]))
    return shared, actions


def repeat_zero_prefixes(entry, anchor):
    release = read(Path(entry["root"]) / "states" / entry["state"] / "release.json")
    seen = {}
    for directory in release["attempt_directories"]:
        directory = Path(directory)
        immediate = directory / "immediate.json"
        if not immediate.exists():
            continue
        info = read(immediate)
        if info["repeat"] != 0:
            continue
        index = info["candidate"]
        files = list(directory.rglob("rollouts/**/*.json"))
        if index in seen or len(files) != 1:
            raise ValueError("Duplicate or missing repeat-0 rollout")
        rollout = read(files[0])
        response = anchor["candidates"][index]["output"][0]
        text = rollout["llm_input_texts"]
        if text.count(response) != 1:
            raise ValueError("Candidate output missing or ambiguous in saved actor input")
        prefix = text[:text.index(response)]
        user_messages = [segment.split("<|im_end|>", 1)[0]
                         for segment in prefix.split("<|im_start|>user\n")[1:]]
        if len(rollout["images"]) < len(user_messages):
            raise ValueError("Archived user-turn images are incomplete")
        images = []
        for user_index, message in enumerate(user_messages):
            placeholders = message.count("<|image_pad|>")
            if placeholders:
                source = rollout["images"][user_index]
                if len(source) != placeholders:
                    raise ValueError("Image placeholders and archived images disagree")
                images.extend(source)
        expected = anchor.get("prompt_input_sha256")
        if not expected:
            raise ValueError("Missing saved pre-action actor fingerprint")
        if prompt_fingerprint(prefix, images) != expected:
            # The final-episode rollout has re-filtered prior screenshots. At
            # candidate sampling the current observation was the last image;
            # restore its one placeholder and verify the frozen actor hash.
            marker = "\n\n</observation>"
            at = prefix.rfind(marker)
            current = rollout["images"][len(user_messages) - 1]
            if at < 0 or len(current) != 1:
                raise ValueError("Cannot restore current screenshot in actor input")
            prefix = (prefix[:at] + "\nscreenshot:\n<|vision_start|><|image_pad|><|vision_end|>\n</observation>"
                      + prefix[at + len(marker):])
            images.extend(current)
        if prompt_fingerprint(prefix, images) != expected:
            raise ValueError("Reconstructed actor input fails saved fingerprint")
        seen[index] = (prefix, images)
    if sorted(seen) != list(range(5)):
        raise ValueError("Repeat-0 prompt is absent for a fixed action")
    if any(value != seen[0] for value in seen.values()):
        raise ValueError("The five actions do not share an exact actor input")
    return seen[0]


def outcomes(entries):
    by_root = defaultdict(set)
    for entry in entries:
        by_root[entry["root"]].add(entry["state"])
    panel = defaultdict(dict)
    for root, states in by_root.items():
        with sqlite3.connect(f"file:{root}/state.sqlite3?mode=ro", uri=True) as db:
            for (raw,) in db.execute("SELECT result_json FROM attempts WHERE status='finished'"):
                value = json.loads(raw)
                state = value.get("state")
                if state not in states or value.get("mode") != "branch" or not value.get("released"):
                    continue
                key = (int(value["candidate"]), int(value["repeat"]))
                item = {"valid": bool(value["valid"]),
                        "success": bool((value.get("reward") or 0) >= 1) if value["valid"] else None}
                if key in panel[state] and panel[state][key] != item:
                    raise ValueError("Contradictory released outcome for " + state)
                panel[state][key] = item
    return panel


def prepare(plans, output):
    output.mkdir(parents=True, exist_ok=True)
    entries = []
    for plan in plans:
        entries.extend(dict(entry, source_plan=str(plan)) for entry in read(plan)["states"])
    if len({entry["state"] for entry in entries}) != len(entries):
        raise ValueError("A state appears in more than one frozen source plan")
    cells = outcomes(entries)
    counts = Counter()
    rows = []
    for entry in entries:
        counts["planned_states"] += 1
        sid = entry["state"]
        cell = cells[sid]
        if any((i, r) not in cell for i in range(5) for r in range(8)):
            counts["incomplete_recorded_panel"] += 1
            continue
        ref = read(Path(entry["root"]) / "states" / sid / "anchor-reference.json")
        if digest(ref["path"]) != ref["sha256"]:
            raise ValueError("Anchor hash mismatch for " + sid)
        anchor = read(ref["path"])
        try:
            shared, actions = candidates(anchor)
            prompt, images = repeat_zero_prefixes(entry, anchor)
        except ValueError as error:
            counts["excluded_" + str(error).lower().replace(" ", "_")[:65]] += 1
            continue
        names = {action["name"] for action in actions}
        if len(names) < 2:
            counts["one_tool_name"] += 1
            continue
        if any(not any(cell[i, r]["valid"] for i, action in enumerate(actions)
                       if action["name"] == name for r in range(5)) for name in names):
            counts["tool_name_without_valid_fit_draw"] += 1
            continue
        if any(not any(cell[i, r]["valid"] for i, action in enumerate(actions)
                       if action["name"] == name for r in range(5, 8)) for name in names):
            counts["tool_name_without_valid_hold_draw"] += 1
            continue
        task = anchor["task_id"]
        group = task_group(anchor["task"])
        rows.append(dict(state_id=sid, task_id=task, task_group=group, fold=None,
            source_plan=entry["source_plan"],
            prompt_text=prompt, images=images, shared_reasoning=shared,
            actions=actions, outcomes=[[cell[i, r] for r in range(8)] for i in range(5)],
            anchor_sha256=ref["sha256"], prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest()))
        counts["admitted_states"] += 1
        counts["admitted_valid_draws"] += sum(cell[i, r]["valid"] for i in range(5) for r in range(8))
        counts["admitted_invalid_draws"] += sum(not cell[i, r]["valid"] for i in range(5) for r in range(8))
    balance_folds(rows)
    path = output / "panels.jsonl"
    with path.open("w") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts["task_groups"] = len({row["task_group"] for row in rows})
    manifest = dict(status="cpu_prepared_no_gpu_or_api_approval", seed=SEED,
                    source_plans={str(plan): digest(plan) for plan in plans},
                    panels_sha256=digest(path), counts=dict(sorted(counts.items())),
                    folds={str(fold): {"states": sum(row["fold"] == fold for row in rows),
                                       "task_groups": len({row["task_group"] for row in rows if row["fold"] == fold})}
                           for fold in range(5)},
                    target="valid-only successes on draws 0-4; draws 5-7 held out",
                    caveat="Tool-name chunks only; fixed candidates and candidate-0 substituted reasoning")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def rebalance_existing(output):
    """Upgrade only task folds in a previously verified panel; preserve payloads."""
    path, manifest_path = output / "panels.jsonl", output / "manifest.json"
    manifest = read(manifest_path)
    if digest(path) != manifest["panels_sha256"]:
        raise ValueError("Existing panel hash mismatch")
    with path.open() as stream:
        rows = [json.loads(line) for line in stream]
    if len(rows) != manifest["counts"]["admitted_states"]:
        raise ValueError("Existing panel count mismatch")
    balance_folds(rows)
    partial = path.with_suffix(path.suffix + ".staged")
    with partial.open("w") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(partial, path)
    manifest["seed"] = SEED
    manifest["panels_sha256"] = digest(path)
    manifest["folds"] = {str(fold): {
        "states": sum(row["fold"] == fold for row in rows),
        "task_groups": len({row["task_group"] for row in rows if row["fold"] == fold})}
        for fold in range(5)}
    staged_manifest = manifest_path.with_suffix(manifest_path.suffix + ".staged")
    staged_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(staged_manifest, manifest_path)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plans", type=Path, nargs="+", default=PLANS)
    parser.add_argument("--rebalance-existing", action="store_true")
    args = parser.parse_args()
    result = rebalance_existing(args.output) if args.rebalance_existing else prepare(args.plans, args.output)
    print(json.dumps({"counts": result["counts"], "folds": result["folds"]}, sort_keys=True))


if __name__ == "__main__":
    main()
