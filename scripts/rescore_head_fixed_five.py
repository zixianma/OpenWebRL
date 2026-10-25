#!/usr/bin/env python3
"""Re-score saved selection heads on the fixed-five-proposal view of the SFT-action test states (no retraining).

Each test state's first five SFT proposals by draw index, duplicates kept; the head picks one; baseline = one SFT sample
from the same five. Success on held-out draws 3-7, valid draws only; task-clustered bootstrap. Writes fixed-five.json
next to each head.pt and prints one JSON line per head.

  rescore_head_fixed_five.py --data D HEAD_DIR [HEAD_DIR ...]
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_head_round import held_rate, load_actions, load_features, score_fixed_five  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True); ap.add_argument("--features", type=Path)
    ap.add_argument("heads", type=Path, nargs="+")
    args = ap.parse_args()
    import torch
    from openwebrl.selection_head import CandidateSetHead, HeadConfig
    from selection_outcome_experiments import cluster_boot
    rec = {x["state_id"]: x for x in load_actions(args.data)}
    features = {}
    mean = lambda v: round(100 * sum(v) / len(v), 2)
    for d in args.heads:
        saved = torch.load(d / "head.pt", map_location="cpu")
        config = HeadConfig.from_json(saved["config"])
        if config.layers not in features:
            features[config.layers] = load_features(args.features or args.data / "features", config.layers, torch)
        cand, state = features[config.layers]
        head = CandidateSetHead(config); head.load_state_dict(saved["state_dict"]); head.eval()
        rates, scored = {}, []
        for s, x in sorted(rec.items()):
            if x["split"] != "test" or s not in cand:
                continue
            r = [held_rate(a, range(3, 8)) for a in x["actions"]]
            if None not in r:
                rates[s] = r; scored.append(s)
        five = score_fixed_five(head, scored, rec, cand, state, rates, torch)
        g = [rec[s]["task_id"] for s in scored]
        res = dict(head_dir=str(d), test_states=len(scored), test_tasks=len(set(g)),
                   head=mean([five[s][0] for s in scored]), sft_sample=mean([five[s][1] for s in scored]),
                   oracle=mean([five[s][2] for s in scored]),
                   head_minus_sft_sample=cluster_boot([five[s][0] - five[s][1] for s in scored], g))
        (d / "fixed-five.json").write_text(json.dumps(res, indent=1))
        print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
