#!/usr/bin/env python3
"""Fit the candidate-set selection head on cached frozen-actor features.

Reports held-out-task agreement with the GPT-5.5 teacher (argmax lands on a
teacher-equivalent candidate) next to uniform, first-candidate and plurality
baselines. Checkpoint selection uses validation agreement, so the reported
validation score is a selected-checkpoint score, not an untouched test score.
"""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def plurality_target(candidates):
    """Indicator of the largest action-equivalence class (ties shared)."""
    from openwebrl.selection_head import equivalent, parse_calls
    calls = [parse_calls(c) for c in candidates]
    sizes = [sum(equivalent(a, b) for b in calls) if a is not None else 0 for a in calls]
    best = max(sizes)
    return [float(s == best and best > 0) for s in sizes]


def load(args, torch, panels=None, features=None):
    from openwebrl.selection_head import CANDIDATE_POOLS
    rows = {}
    with open(panels or args.panels) as stream:
        for line in stream:
            row = json.loads(line)
            rows[row["state_id"]] = row
    pools = [CANDIDATE_POOLS.index(p) for p in args.pools.split(",")]
    data = {}
    layers_all = None
    for path in sorted((features or args.features).glob("shard*-chunk*.pt")):
        chunk = torch.load(path, map_location="cpu")
        layers_all = chunk["layers"]
        keep = [layers_all.index(int(l)) for l in args.layers.split(",")]
        for sid, panel in chunk["panels"].items():
            row = rows[sid]
            # panel [5, L*P, H] is layer-major; keep the requested layers/pools.
            panel = panel.view(5, len(layers_all), len(CANDIDATE_POOLS), -1)[:, keep][:, :, pools]
            state = chunk["states"][row["base_state_id"]][keep]
            data.setdefault(row["split"], []).append((panel.reshape(5, -1), state.reshape(-1), row))
    out = {}
    for split, items in data.items():
        out[split] = {
            "candidates": torch.stack([x[0] for x in items]),
            "state": torch.stack([x[1] for x in items]),
            "plurality": torch.tensor([plurality_target(x[2]["candidates"]) for x in items]),
            "rows": [x[2] for x in items],
        }
        if "target" in items[0][2]:
            out[split]["target"] = torch.tensor([x[2]["target"] for x in items])
            out[split]["informative"] = torch.tensor([x[2]["informative"] for x in items])
    return out, layers_all


def predict(head, split, device, torch, batch=512, score_head="select"):
    head.eval()
    logits = []
    with torch.no_grad():
        for i in range(0, len(split["state"]), batch):
            selection, value = head(split["candidates"][i:i + batch].to(device),
                                    split["state"][i:i + batch].to(device))
            logits.append((value if score_head == "value" else selection).float().cpu())
    return torch.cat(logits)


def evaluate_outcomes(picks, split, external=None):
    """Mean continuation success of the chosen candidate (invalid = 0) on branch states.

    picks: per-state chosen index from the head; external: {name: {state_id: index}}.
    Reported on all states, prompt-verified states, and verified states whose task
    does not occur in Piotr's release.
    """
    rows = split["rows"]
    rate = lambda row, i: row["outcomes"][i]["successes"] / row["outcomes"][i]["trials"]
    plural = [max(range(5), key=lambda i: (p[i], -i)) for p in split["plurality"].tolist()]
    choosers = {"head": dict(zip([r["state_id"] for r in rows], picks)),
                "actor_first": {r["state_id"]: 0 for r in rows},
                "plurality_first_tie": dict(zip([r["state_id"] for r in rows], plural)),
                **(external or {})}
    subsets = {"all": lambda r: True, "prompt_verified": lambda r: r["prompt_verified"],
               "verified_no_piotr_task": lambda r: r["prompt_verified"] and not r["piotr_task_overlap"]}
    out = {}
    for name, keep in subsets.items():
        chosen = [r for r in rows if keep(r)]
        result = {"states": len(chosen),
                  "uniform": sum(sum(rate(r, i) for i in range(5)) / 5 for r in chosen) / len(chosen),
                  "hindsight_oracle": sum(max(rate(r, i) for i in range(5)) for r in chosen) / len(chosen)}
        for chooser, picked in choosers.items():
            covered = [r for r in chosen if r["state_id"] in picked]
            if covered:
                result[chooser] = sum(rate(r, picked[r["state_id"]]) for r in covered) / len(covered)
                result[chooser + "_states"] = len(covered)
        out[name] = result
    return out


def evaluate_agreement(choosers, split, reference):
    """Mean share of a reference teacher's judgments matched by each chooser's action.

    reference: {state_id: [candidate indices]} from a teacher no chooser trained on.
    Matching uses the same action equivalence as the training targets.
    """
    from openwebrl.selection_head import equivalent, parse_calls
    rows = [r for r in split["rows"] if r["state_id"] in reference]
    calls = {r["state_id"]: [parse_calls(c) for c in r["candidates"]] for r in rows}
    same = lambda sid, a, b: a == b or equivalent(calls[sid][a], calls[sid][b])
    agree = lambda sid, i: sum(same(sid, i, j) for j in reference[sid]) / len(reference[sid])
    out = {"states": len(rows),
           "uniform": sum(sum(agree(r["state_id"], i) for i in range(5)) / 5 for r in rows) / len(rows)}
    for name, picked in choosers.items():
        covered = [r["state_id"] for r in rows if r["state_id"] in picked]
        if covered:
            out[name] = sum(agree(sid, picked[sid]) for sid in covered) / len(covered)
            out[name + "_states"] = len(covered)
    return out


def evaluate(head, split, device, torch, batch=512, score_head="select"):
    from openwebrl.selection_head import topk_agreement
    logits = predict(head, split, device, torch, batch, score_head)
    t, inf = split["target"], split["informative"]
    hit = lambda l, m: topk_agreement(l[m], t[m])
    plural = split["plurality"] + 1e-3 * torch.rand_like(split["plurality"])  # random tie-break
    return {"panels": len(t), "informative_panels": int(inf.sum()),
            "head": hit(logits, slice(None)), "head_informative": hit(logits, inf),
            "uniform": (t > 0).float().mean(-1).mean().item(),
            "uniform_informative": (t[inf] > 0).float().mean(-1).mean().item(),
            "first_candidate": (t[:, 0] > 0).float().mean().item(),
            "plurality": hit(plural, slice(None)), "plurality_informative": hit(plural, inf)}


def score_branch(head, args, torch, result):
    """Branch-state picks and outcomes; branch states never enter training or checkpoint selection."""
    branch, _ = load(args, torch, args.branch_panels, args.branch_features)
    (branch,) = branch.values()   # one split per branch panel file
    picks = predict(head, branch, args.device, torch, score_head=args.score_head).argmax(-1).tolist()
    result["branch_head_picks"] = dict(zip([r["state_id"] for r in branch["rows"]], picks))
    if args.branch_picks_only:
        return
    external = {}
    for path in args.external_picks:
        with open(path) as stream:
            # Unparseable replies are unavailable, reported via the coverage count.
            external[path.stem] = {r["state_id"]: r["selected_index"] for r in map(json.loads, stream)
                                   if r["selected_index"] is not None}
    result["branch_outcomes"] = evaluate_outcomes(picks, branch, external)
    if args.reference_picks:
        with open(args.reference_picks) as stream:
            reference = {r["state_id"]: r["selected_indices"] for r in map(json.loads, stream)}
        choosers = {"head": result["branch_head_picks"],
                    "actor_first": {r["state_id"]: 0 for r in branch["rows"]}, **external}
        result["branch_reference_agreement"] = evaluate_agreement(choosers, branch, reference)
        print(json.dumps({"reference_agreement": result["branch_reference_agreement"]}), flush=True)
    print(json.dumps(result["branch_outcomes"]), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panels", type=Path, required=True)
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--branch-panels", type=Path, help="Outcome-labeled branching panels")
    ap.add_argument("--branch-features", type=Path)
    ap.add_argument("--branch-picks-only", action="store_true",
                    help="save picks without computing historical invalid-as-failure outcome rates")
    ap.add_argument("--external-picks", type=Path, action="append", default=[],
                    help="JSONL of {state_id, selected_index} from another selector (name = file stem)")
    ap.add_argument("--reference-picks", type=Path,
                    help="JSONL {state_id, selected_indices} from an independent teacher, for agreement")
    ap.add_argument("--layers", default="18,27,36")
    ap.add_argument("--pools", default="thought_mean,action_mean,last")
    ap.add_argument("--pointwise", action="store_true")
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--objective", choices=("soft_ce", "pairwise_teacher"), default="soft_ce",
                    help="Pairwise uses only the GPT-5.5 winner class versus non-equivalent candidates")
    ap.add_argument("--load-head", type=Path,
                    help="score the branch panels with this saved head.pt instead of training (--panels/--features unused)")
    args = ap.parse_args()

    import torch
    from openwebrl.selection_head import CandidateSetHead, HeadConfig, pairwise_teacher_loss, selection_loss

    if args.load_head:
        saved = torch.load(args.load_head, map_location="cpu")
        args.objective = saved.get("objective", "soft_ce")
        args.score_head = "value" if args.objective == "pairwise_teacher" else "select"
        config = HeadConfig.from_json(saved["config"])
        args.layers, args.pools = ",".join(map(str, config.layers)), ",".join(config.pools)
        head = CandidateSetHead(config)
        head.load_state_dict(saved["state_dict"])
        head.to(args.device)
        result = {"args": {k: str(v) for k, v in vars(args).items()}}
        score_branch(head, args, torch, result)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "metrics.json").write_text(json.dumps(result, indent=1))
        return

    torch.manual_seed(args.seed)
    args.score_head = "value" if args.objective == "pairwise_teacher" else "select"
    data, _ = load(args, torch)
    train, val = data["train"], data["validation"]
    hidden = train["state"].shape[-1] // len(args.layers.split(","))
    config = HeadConfig(hidden_size=hidden, layers=tuple(int(l) for l in args.layers.split(",")),
                        pools=tuple(args.pools.split(",")), d_model=args.d_model, depth=args.depth,
                        dropout=args.dropout, listwise=not args.pointwise)
    head = CandidateSetHead(config)
    head.fit_standardizers(train["candidates"], train["state"])
    head.to(args.device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = args.epochs * ((len(train["target"]) + args.batch_size - 1) // args.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps, pct_start=0.05)
    args.output.mkdir(parents=True, exist_ok=True)
    mask = torch.ones(args.batch_size, 5, dtype=torch.bool, device=args.device)
    history, best = [], None
    for epoch in range(args.epochs):
        head.train()
        order, started, total = torch.randperm(len(train["target"])), time.time(), 0.
        for i in range(0, len(order), args.batch_size):
            idx = order[i:i + args.batch_size]
            # Panels are permutation-equivariant; shuffling candidates guards against
            # any slot-order leakage in the cached features.
            perm = torch.argsort(torch.rand(len(idx), 5), -1)
            cands = train["candidates"][idx].gather(1, perm[..., None].expand(-1, -1, train["candidates"].shape[-1]))
            target = train["target"][idx].gather(1, perm)
            logits, values = head(cands.to(args.device), train["state"][idx].to(args.device))
            loss = (pairwise_teacher_loss(values, target.to(args.device), mask[:len(idx)])
                    if args.objective == "pairwise_teacher"
                    else selection_loss(logits, target.to(args.device), mask[:len(idx)]))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
            opt.step(); sched.step()
            total += loss.item() * len(idx)
        metrics = {"epoch": epoch + 1, "train_loss": total / len(order),
                   "seconds": round(time.time() - started, 1),
                   "validation": evaluate(head, val, args.device, torch, score_head=args.score_head)}
        history.append(metrics)
        print(json.dumps(metrics), flush=True)
        if best is None or metrics["validation"]["head"] > best["validation"]["head"]:
            best = metrics
            torch.save({"config": config.to_json(), "state_dict": head.state_dict(),
                        "objective": args.objective}, args.output / "head.pt")
    head.load_state_dict(torch.load(args.output / "head.pt")["state_dict"])
    result = {"args": {k: str(v) for k, v in vars(args).items()}, "best": best, "history": history,
              "train": evaluate(head, train, args.device, torch, score_head=args.score_head),
              "note": "validation selects the checkpoint; not an untouched test score"}
    if args.branch_panels:
        score_branch(head, args, torch, result)
    (args.output / "metrics.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
