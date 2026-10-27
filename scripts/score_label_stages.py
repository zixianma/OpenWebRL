#!/usr/bin/env python3
"""Score teacher labels and the heads trained on them on the same SFT-action states, both splits, held-out draws 3-7.

For each label source and each split (train = the heads' training states, test = the held-out test states), on the
fixed-five view (each state's first five SFT proposals by draw index, duplicates kept; success on draws 3-7, valid draws
only; states with a held-out rate for every action and a target from every source):
  * teacher: the action with the largest target among the five (Luna vote share, or for outcome the best success on
    draws 0-2); tied actions are averaged (expected value of a random tie-break);
  * head: the head's pick, averaged per state over the given seeds;
  * baseline: one SFT sample from the same five (mean over the five).
Differences from the baseline are averaged over states with a whole-task bootstrap (selection_outcome_experiments.cluster_boot).
Training states are seen by the head (their labels), but draws 3-7 never are.

  score_label_stages.py --data D --labels L --sources outcome,teacher_k3 --heads outcome=DIR[,DIR] ... --output OUT.json
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_head_round import fixed_five, held_rate, load_actions, load_features, targets  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True); ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--sources", required=True); ap.add_argument("--heads", nargs="*", default=[])
    ap.add_argument("--train-states", type=Path, help="train-states.json of a head (default: first head's)")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    import torch
    from openwebrl.selection_head import CandidateSetHead, HeadConfig
    from selection_outcome_experiments import cluster_boot
    records = load_actions(args.data); rec = {x["state_id"]: x for x in records}
    sources = args.sources.split(",")
    heads = {k: [Path(p) for p in v.split(",")] for k, v in (h.split("=", 1) for h in args.heads)}
    y = {s: targets(records, args.labels, s) for s in sources}
    ts = args.train_states or next(iter(heads.values()))[0] / "train-states.json"
    train = set(json.loads(ts.read_text()))
    features = {}

    def slots(sid):
        rows, acts = fixed_five(rec[sid]); return rows, acts

    rates = {}
    for s, x in rec.items():
        r = [held_rate(a, range(3, 8)) for a in x["actions"]]
        if None not in r and all(s in y[src] for src in sources):
            rates[s] = r
    split_ids = {"train": sorted(s for s in rates if s in train),
                 "test": sorted(s for s in rates if rec[s]["split"] == "test")}
    picks = {}                                              # (source, split) -> per-state head success, seed-averaged
    for src, dirs in heads.items():
        per_seed = []
        for d in dirs:
            saved = torch.load(d / "head.pt", map_location="cpu"); config = HeadConfig.from_json(saved["config"])
            if config.layers not in features:
                features[config.layers] = load_features(args.data / "features", config.layers, torch)
            cand, state = features[config.layers]
            head = CandidateSetHead(config); head.load_state_dict(saved["state_dict"]); head.eval()
            got = {}
            with torch.no_grad():
                for sid in (s for ids in split_ids.values() for s in ids):
                    rows, acts = slots(sid)
                    c = cand[sid][rows].float()[None]; m = torch.ones(1, len(rows), dtype=torch.bool)
                    got[sid] = rates[sid][acts[head(c, state[sid].float()[None], m)[0].argmax(-1).item()]]
            per_seed.append(got)
        picks[src] = {s: sum(g[s] for g in per_seed) / len(per_seed) for s in per_seed[0]}
    out = dict(note=__doc__.split("\n\n")[0], sources=sources, heads={k: [str(d) for d in v] for k, v in heads.items()},
               splits={k: dict(states=len(v), tasks=len({rec[s]["task_id"] for s in v})) for k, v in split_ids.items()})
    mean = lambda v: round(100 * sum(v) / len(v), 2)
    for split, ids in split_ids.items():
        g = [rec[s]["task_id"] for s in ids]
        base = {s: sum(rates[s][a] for a in slots(s)[1]) / len(slots(s)[1]) for s in ids}
        out[split] = dict(sft_sample=mean([base[s] for s in ids]))
        for src in sources:
            teach = {}
            for s in ids:
                acts = slots(s)[1]; best = max(y[src][s][a] for a in acts)
                tied = [a for a in acts if y[src][s][a] == best]
                teach[s] = sum(rates[s][a] for a in tied) / len(tied)
            row = dict(teacher=mean([teach[s] for s in ids]),
                       teacher_minus_sft=[round(v, 2) for v in cluster_boot([teach[s] - base[s] for s in ids], g)])
            if src in picks:
                row.update(head=mean([picks[src][s] for s in ids]), seeds=len(heads[src]),
                           head_minus_sft=[round(v, 2) for v in cluster_boot([picks[src][s] - base[s] for s in ids], g)])
            out[split][src] = row
    args.output.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
