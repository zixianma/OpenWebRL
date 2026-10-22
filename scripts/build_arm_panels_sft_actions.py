#!/usr/bin/env python3
"""Five-candidate SelectionARM panels and per-slot targets for the SFT-action states (CPU only).

Input: the SFT-action panel builder's output (`--data`: actions.jsonl + panels.jsonl, scripts/build_sft_action_panels.py)
and the round labels (`--labels`, as scripts/train_head_round.py reads them). Output (`--output`): panels.jsonl and
targets.json in the format scripts/build_selection_sft_ours.py consumes, so the LLaMA-Factory dataset is rendered by
the same prompt path the live `piotr5` selector uses.

`--view distinct` (default): the selection heads' training view -- one candidate per distinct action (the action's
representative SFT text, 2-5 candidates), written in `--orderings` shuffled orders per state (a head trained for 20
epochs with candidate order reshuffled each epoch; 10 orderings x Piotr's 2 epochs gives the same 20 exposures).
`--view natural`: five of the state's SFT proposals, drawn without replacement from its pool of SFT draws (with
replacement when the pool has fewer than five), in random order -- the five iid actor samples a live decision sees.
Each state gets `--panels-per-state` seeded panels (`<state>#<k>`; Piotr's release has about 13 per state). A slot's
target is its action's label value: Luna vote share (teacher_pre / teacher_k1 / teacher_k3) or the tie-split best
observed success on draws 0-2 (outcome), as for the selection heads. Training states are the heads' common training
states (all four labels present); test states get panels for offline scoring. Fold 1 = train, fold 0 = test.
"""
import argparse
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_sft_action_panels import EXPORT  # noqa: E402
from train_head_round import SOURCES, load_actions, targets  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--export", type=Path, default=EXPORT)
    ap.add_argument("--view", choices=("distinct", "natural"), default="distinct")
    ap.add_argument("--orderings", type=int, default=10, help="distinct view: shuffled orders per training state")
    ap.add_argument("--panels-per-state", type=int, default=13, help="natural view: panels per training state")
    ap.add_argument("--test-panels-per-state", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20261021)
    args = ap.parse_args()

    records = {x["state_id"]: x for x in load_actions(args.data)}
    texts = {}
    for line in open(args.data / "panels.jsonl"):
        p = json.loads(line)
        texts[p["state_id"]] = dict(zip(p["feature_draws"], p["candidates"]))
    sources = [*SOURCES, "outcome"]
    y = {s: targets(list(records.values()), args.labels, s) for s in sources}
    common = set.intersection(*(set(v) for v in y.values()))

    anchors = {}
    for line in open(args.export):
        r = json.loads(line)
        if r["state"] in records and r["checkpoint"] == "SFT":
            ref = json.loads((Path(r["release_path"]).parent / "anchor-reference.json").read_text())["path"]
            anchors[r["state"]] = json.loads(Path(ref).read_text())

    rng = random.Random(args.seed)
    out_panels, out_targets, counts = [], {}, {"train_states": 0, "test_states": 0, "train_panels": 0, "test_panels": 0}
    for sid in sorted(records):
        x = records[sid]
        if sid not in common or sid not in anchors:
            continue
        anchor = anchors[sid]
        split = x["split"]
        pool = [(d, a) for a, act in enumerate(x["actions"]) for d in act["sft_draws"]]
        reps = [(act["rep_draw"], a) for a, act in enumerate(x["actions"])]
        if args.view == "distinct":
            n = args.orderings if split == "train" else 1
        else:
            n = args.panels_per_state if split == "train" else args.test_panels_per_state
        counts[f"{split}_states"] += 1
        for k in range(n):
            if args.view == "distinct":
                pick = list(reps)
            else:
                pick = rng.sample(pool, 5) if len(pool) >= 5 else [rng.choice(pool) for _ in range(5)]
            rng.shuffle(pick)
            pid = f"{sid}#{k}"
            out_panels.append(dict(
                state_id=pid, base_state_id=sid, task_id=x["task_id"], split=split, image=x["image"],
                screenshot_sha256=anchor["before_observation"]["screenshot"].get("sha256"), task=anchor["task"],
                url=anchor["before_observation"].get("active_tab_url", ""),
                history=[e["output"][0] for e in anchor["prefix"]],
                candidates=[texts[sid][d] if args.view == "natural" else x["actions"][a]["rep_text"] for d, a in pick], draws=[d for d, _ in pick], actions=[a for _, a in pick]))
            out_targets[pid] = dict(task=x["task_id"], fold=1 if split == "train" else 0,
                                    targets={s: [y[s][sid][a] for _, a in pick] for s in sources})
            counts[f"{split}_panels"] += 1
    args.output.mkdir(parents=True, exist_ok=True)
    with open(args.output / "panels.jsonl", "w") as f:
        for p in out_panels:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    (args.output / "targets.json").write_text(json.dumps(dict(sources=sources, states=out_targets)))
    counts.update(view=args.view, orderings=args.orderings, seed=args.seed, panels_per_state=args.panels_per_state, test_panels_per_state=args.test_panels_per_state)
    (args.output / "summary.json").write_text(json.dumps(counts, indent=1))
    print(json.dumps(counts, indent=1))


if __name__ == "__main__":
    main()
