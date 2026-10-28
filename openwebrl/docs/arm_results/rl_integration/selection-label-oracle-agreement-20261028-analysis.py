#!/usr/bin/env python3
"""Agreement of teacher picks and head (student) picks with the oracle action, same states as score_label_stages.py.

Fixed-five view (first five SFT proposals by draw index), held-out draws 3-7, valid draws only. Oracle set = the
actions among the five with the best held-out success (all tied best actions count). Per state:
  * teacher: share of the teacher's tied top actions that are in the oracle set (expected value of a random tie-break);
  * head: 1 if the head's argmax is in the oracle set (seed-averaged when several heads are given);
  * chance: share of the five slots whose action is in the oracle set (one SFT sample);
  * teacher_head: 1 if the head's pick is one of the teacher's top actions (seed- and tie-averaged).
Reported on all states and on informative states (oracle set not covering all five slots, i.e. the pick can be wrong).
Rates in %, with whole-task bootstrap CIs of the difference from chance (selection_outcome_experiments.cluster_boot).

  score_label_agreement.py --data D --labels L --sources outcome,teacher_k3 --heads outcome=DIR[,DIR] ... --output OUT.json
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_head_round import fixed_five, held_rate, load_actions, load_features, targets  # noqa: E402

TOL = 1e-9


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True); ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--sources", required=True); ap.add_argument("--heads", nargs="*", default=[])
    ap.add_argument("--train-states", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    import torch
    from openwebrl.selection_head import CandidateSetHead, HeadConfig
    from selection_outcome_experiments import cluster_boot
    records = load_actions(args.data); rec = {x["state_id"]: x for x in records}
    sources = args.sources.split(",")
    heads = {k: [Path(p) for p in v.split(",")] for k, v in (h.split("=", 1) for h in args.heads)}
    y = {s: targets(records, args.labels, s) for s in sources}
    train = set(json.loads(args.train_states.read_text()))
    rates = {}
    for s, x in rec.items():
        r = [held_rate(a, range(3, 8)) for a in x["actions"]]
        if None not in r and all(s in y[src] for src in sources):
            rates[s] = r
    split_ids = {"train": sorted(s for s in rates if s in train),
                 "test": sorted(s for s in rates if rec[s]["split"] == "test")}
    acts = {s: fixed_five(rec[s]) for ids in split_ids.values() for s in ids}       # (feature rows, action per slot)
    oracle = {}
    for s, (_, a) in acts.items():
        best = max(rates[s][i] for i in a); oracle[s] = {i for i in a if rates[s][i] >= best - TOL}
    features, head_picks = {}, {}                          # source -> state -> list of picked actions (one per seed)
    for src, dirs in heads.items():
        head_picks[src] = {}
        for d in dirs:
            saved = torch.load(d / "head.pt", map_location="cpu"); config = HeadConfig.from_json(saved["config"])
            if config.layers not in features:
                features[config.layers] = load_features(args.data / "features", config.layers, torch)
            cand, state = features[config.layers]
            head = CandidateSetHead(config); head.load_state_dict(saved["state_dict"]); head.eval()
            with torch.no_grad():
                for s, (rows, a) in acts.items():
                    c = cand[s][rows].float()[None]; m = torch.ones(1, len(rows), dtype=torch.bool)
                    head_picks[src].setdefault(s, []).append(a[head(c, state[s].float()[None], m)[0].argmax(-1).item()])
    pct = lambda v: round(100 * sum(v) / len(v), 2) if v else None
    out = dict(note=__doc__.split("\n\n")[0], sources=sources, heads={k: [str(d) for d in v] for k, v in heads.items()})
    for split, ids in split_ids.items():
        out[split] = {}
        for subset, keep in (("all", lambda s: True), ("informative", lambda s: len([i for i in acts[s][1] if i in oracle[s]]) < 5)):
            sub = [s for s in ids if keep(s)]
            g = [rec[s]["task_id"] for s in sub]
            chance = {s: sum(i in oracle[s] for i in acts[s][1]) / 5 for s in sub}
            res = dict(states=len(sub), tasks=len(set(g)), chance=pct(list(chance.values())))
            for src in sources:
                top = {}
                for s in sub:
                    a = acts[s][1]; best = max(y[src][s][i] for i in a)
                    top[s] = sorted({i for i in a if y[src][s][i] >= best - TOL})
                teach = {s: sum(i in oracle[s] for i in top[s]) / len(top[s]) for s in sub}
                row = dict(teacher=pct(list(teach.values())),
                           teacher_minus_chance=[round(v, 2) for v in cluster_boot([teach[s] - chance[s] for s in sub], g)])
                if src in head_picks:
                    hp = {s: sum(p in oracle[s] for p in head_picks[src][s]) / len(head_picks[src][s]) for s in sub}
                    th = {s: sum(p in top[s] for p in head_picks[src][s]) / len(head_picks[src][s]) for s in sub}
                    row.update(head=pct(list(hp.values())), seeds=len(heads[src]),
                               head_minus_chance=[round(v, 2) for v in cluster_boot([hp[s] - chance[s] for s in sub], g)],
                               head_picks_teacher_top=pct(list(th.values())))
                res[src] = row
            out[split][subset] = res
    args.output.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
