#!/usr/bin/env python3
"""Supervision ablation for the selection head: the same frozen-feature head trained on different label sources.

One task-grouped split: fold `--test-fold` of `targets.json` is the test set; every other fold trains. All sources
train on exactly the same states (those with every requested target). Fixed epochs (no validation split, so neither
labels nor outcomes of test tasks are seen), several seeds per source. Writes each seed's pick on every test state;
held-out outcomes (draws 3-7) are scored separately.

Usage: train_supervision_ablation.py --panels P --features F --targets T --output O [--sources teacher_pre,teacher_k3]
"""
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panels", type=Path, required=True); ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--targets", type=Path, required=True); ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--sources", default="teacher_pre,teacher_k3"); ap.add_argument("--test-fold", type=int, default=0)
    ap.add_argument("--seeds", default="0,1,2"); ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05); ap.add_argument("--device", default="cpu")
    ap.add_argument("--save-heads", action="store_true", help="save each trained head (deployment); --test-fold -1 trains on every fold")
    ap.add_argument("--train-frac", type=float, default=1.0, help="learning curve: keep this fraction of training tasks (seeded)")
    ap.add_argument("--frac-seed", type=int, default=20261017)
    ap.add_argument("--layers", default="18,27,36", help="actor layers the head reads (e.g. 14,21,28 for a 28-layer actor)")
    args = ap.parse_args()
    import torch
    from openwebrl.selection_head import CandidateSetHead, HeadConfig, selection_loss
    from train_selection_head import load

    targets = json.loads(args.targets.read_text())["states"]
    sources = args.sources.split(",")
    cfg = SimpleNamespace(panels=args.panels, layers=args.layers, pools="thought_mean,action_mean,last")
    layers = tuple(int(x) for x in args.layers.split(","))
    data, _ = load(cfg, torch, panels=args.panels, features=args.features)
    rows = [r for split in data.values() for r in split["rows"]]
    cand = torch.cat([split["candidates"] for split in data.values()]); state = torch.cat([split["state"] for split in data.values()])
    index = {r["state_id"]: i for i, r in enumerate(rows)}
    usable = [s for s in index if s in targets and all(src in targets[s]["targets"] for src in sources)]
    train_ids = [s for s in usable if targets[s]["fold"] != args.test_fold]
    if args.train_frac < 1:                                # subsample whole training tasks, nested across fractions
        import random
        tasks = sorted({targets[s]["task"] for s in train_ids})
        random.Random(args.frac_seed).shuffle(tasks)
        keep = set(tasks[:max(1, round(args.train_frac * len(tasks)))])
        train_ids = [s for s in train_ids if targets[s]["task"] in keep]
    test_ids = [s for s in index if s in targets and targets[s]["fold"] == args.test_fold]
    tr, te = torch.tensor([index[s] for s in train_ids]), torch.tensor([index[s] for s in test_ids])
    hidden = state.shape[-1] // len(layers)
    out = dict(train_states=len(train_ids), test_states=len(test_ids), test_fold=args.test_fold, sources=sources,
               train_frac=args.train_frac, train_tasks=len({targets[s].get("task") for s in train_ids}),
               missing_features=sum(s not in index for s in targets), picks={})
    for src in sources:
        y = torch.tensor([targets[s]["targets"][src] for s in train_ids], dtype=torch.float32)
        for seed in [int(x) for x in args.seeds.split(",")]:
            torch.manual_seed(seed)
            config = HeadConfig(hidden_size=hidden, layers=layers, pools=("thought_mean", "action_mean", "last"),
                                d_model=512, depth=2, dropout=0.1, listwise=True)
            head = CandidateSetHead(config); head.fit_standardizers(cand[tr], state[tr]); head.to(args.device)
            opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            steps = args.epochs * ((len(tr) + args.batch_size - 1) // args.batch_size)
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps, pct_start=0.05)
            mask = torch.ones(args.batch_size, 5, dtype=torch.bool, device=args.device)
            started = time.time()
            for _ in range(args.epochs):
                head.train()
                order = torch.randperm(len(tr))
                for i in range(0, len(order), args.batch_size):
                    idx = order[i:i + args.batch_size]; b = tr[idx]
                    perm = torch.argsort(torch.rand(len(idx), 5), -1)          # candidate order is never informative
                    c = cand[b].gather(1, perm[..., None].expand(-1, -1, cand.shape[-1]))
                    logits, _ = head(c.to(args.device), state[b].to(args.device))
                    loss = selection_loss(logits, y[idx].gather(1, perm).to(args.device), mask[:len(idx)])
                    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 1.); opt.step(); sched.step()
            head.eval()
            with torch.no_grad():
                logits = head(cand[te].to(args.device), state[te].to(args.device))[0] if len(te) else torch.zeros(0, 5)
            out["picks"][f"{src}/seed{seed}"] = {s: int(k) for s, k in zip(test_ids, logits.argmax(-1).tolist())} if test_ids else {}
            if args.save_heads:
                d = args.output / "heads" / src / f"seed{seed}"; d.mkdir(parents=True, exist_ok=True)
                torch.save({"config": config.to_json(), "state_dict": {k: v.cpu() for k, v in head.state_dict().items()}, "objective": "soft_ce",
                            "source": src, "seed": seed, "train_states": len(train_ids)}, d / "head.pt")
            print(json.dumps(dict(source=src, seed=seed, seconds=round(time.time() - started, 1))), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "student-picks.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "picks"}))


if __name__ == "__main__":
    main()
