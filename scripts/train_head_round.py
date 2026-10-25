#!/usr/bin/env python3
"""Selection-head ablation round on SFT-only action panels (`build_sft_action_panels.py`).

One fixed task-grouped split (actions.jsonl `split`), one seed, fixed epochs (no validation; test labels and outcomes
are never seen). Every source trains on the same states: those with a target from every source in --common-sources.

Label sources (one target per action): teacher_pre / teacher_k1 / teacher_k3 = Luna vote share over the state's
actions (judgments under --labels/<pre|k1|k3>); outcome = the action(s) with the best observed success on draws 0-2,
valid draws only, ties split. Held-out draws 3-7 only score.

Panel views: "distinct" = one representative per action (2-5 candidates); "natural" = five SFT proposals drawn at
the actor's own frequencies (each SFT proposal of an action's key; a state with fewer labelled proposals uses all),
resampled every epoch; duplicates of an action share its target mass.

Offline score on test states (all actions with >= 1 valid held-out draw), on held-out draws 3-7:
  * fixed five (primary, as served): the state's first five SFT proposals by draw index, duplicates kept (all of them
    when it has fewer); the head picks one; baseline = one SFT sample from the same five (mean over the five);
  * distinct actions (diagnostic): one representative per action; baseline uniform over actions; also the SFT-sample
    baseline over all proposals and the oracle; natural view: 20 seeded five-proposal panels per test state.
Task-clustered bootstrap differences.

  train_head_round.py --data D --labels L --source teacher_k1 --output O [--interaction ... --scorer ... --loss ...
                      --mlp-depth 2 --mlp-width 1686 --readout linear|pointer --train-view distinct|natural --train-frac 1]

Teacher-data scaling (--extra-only): train on the extra panels alone (none of our states), on the first
--extra-max-states base states of whole tasks in a seeded order (nested across sizes), and keep the epoch with the best
agreement on the extra panels' own validation split (tasks of our test split excluded there too), as the live
Piotr-data head did. Our test states are only scored.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
SOURCES = {"teacher_pre": "pre", "teacher_k1": "k1", "teacher_k3": "k3"}


def load_actions(data):
    return [json.loads(l) for l in open(data / "actions.jsonl")]


def held_rate(action, draws):
    cells = [c for d in draws for c in action["outcomes"][str(d)] if c and c[0]]
    return sum(c[1] for c in cells) / len(cells) if cells else None


def targets(records, labels, source):
    out = {}
    if source == "outcome":
        for x in records:
            rates = [held_rate(a, range(3)) for a in x["actions"]]
            if None in rates:
                continue
            best = max(rates); w = [float(r == best) for r in rates]
            out[x["state_id"]] = [v / sum(w) for v in w]
        return out
    judged = labels / SOURCES[source] / "judgments"
    for x in records:
        f = judged / f"{x['state_id']}.json"
        if not f.exists():
            continue
        where = {a["teacher_exec"]: i for i, a in enumerate(x["actions"])}
        votes = [0.0] * len(x["actions"])
        for j in json.loads(f.read_text()):
            if j.get("valid"):
                votes[where[j["candidate"]]] += 1
        if sum(votes):
            out[x["state_id"]] = [v / sum(votes) for v in votes]
    return out


def fixed_five(x):
    """Feature rows and action indices of the state's first five SFT proposals by draw index (duplicates kept)."""
    pos = {d: i for i, d in enumerate(x["feature_draws"])}
    slots = sorted((d, a) for a, act in enumerate(x["actions"]) for d in act["sft_draws"])[:5]
    return [pos[d] for d, _ in slots], [a for _, a in slots]


def score_fixed_five(head, ids, rec, cand, state, rates, torch):
    """Per state: (head's pick success, one-SFT-sample success, best in panel) on the fixed five proposals."""
    out = {}
    with torch.no_grad():
        for sid in ids:
            rows, acts = fixed_five(rec[sid])
            c = cand[sid][rows].float()[None]; m = torch.ones(1, len(rows), dtype=torch.bool)
            pick = acts[head(c, state[sid].float()[None], m)[0].argmax(-1).item()]
            r = rates[sid]
            out[sid] = (r[pick], sum(r[a] for a in acts) / len(acts), max(r[a] for a in acts))
    return out


def load_features(features, layers, torch):
    from openwebrl.selection_head import CANDIDATE_POOLS
    cand, state, stored = {}, {}, None
    for path in sorted(features.glob("shard*-chunk*.pt")):
        chunk = torch.load(path, map_location="cpu")
        stored = list(chunk["layers"]); keep = [stored.index(l) for l in layers]
        for sid, panel in chunk["panels"].items():
            cand[sid] = panel.view(panel.shape[0], len(stored), len(CANDIDATE_POOLS), -1)[:, keep].reshape(panel.shape[0], -1)
            state[sid] = chunk["states"][sid][keep].reshape(-1)
    return cand, state


def load_extra(args, rec, layers, torch, split="train"):
    """Five-candidate panels of one split with teacher targets, excluding every task of our test split.
    --extra-max-states keeps whole tasks, in a seeded order, until that many base states are covered."""
    from build_selection_head_branch_eval import norm
    from openwebrl.selection_head import CANDIDATE_POOLS
    import re
    test_tasks = {norm(x["task"]) for x in rec.values() if x["split"] == "test"}
    task = {}
    with open(args.extra_states) as stream:
        for line in stream:
            r = json.loads(line)
            found = re.findall(r"<user_request>(.*?)</user_request>", r["prompt_text"], re.S)
            task[r["state_id"]] = norm(found[-1]) if found else None
    rows = {}
    excluded = 0
    for line in open(args.extra_panels):
        r = json.loads(line)
        if r["split"] != split:
            continue
        if task.get(r["base_state_id"]) in test_tasks:
            excluded += 1; continue
        rows[r["state_id"]] = r
    if split == "train" and args.extra_max_states:
        by_task = defaultdict(set)
        for r in rows.values():
            by_task[task.get(r["base_state_id"]) or r["task_group"]].add(r["base_state_id"])
        order = sorted(by_task); random.Random(args.extra_subset_seed).shuffle(order)
        keep = set()
        for t in order:
            if len(keep) >= args.extra_max_states:
                break
            keep |= by_task[t]
        rows = {k: r for k, r in rows.items() if r["base_state_id"] in keep}
    if split == "train" and args.extra_panels_per_state:          # seeded choice of panels within each base state
        per = defaultdict(list)
        for k in sorted(rows):
            per[rows[k]["base_state_id"]].append(k)
        pick = random.Random(args.extra_subset_seed)
        rows = {k: rows[k] for b in sorted(per) for k in pick.sample(per[b], min(len(per[b]), args.extra_panels_per_state))}
    c, s, t = [], [], []
    for path in sorted(args.extra_features.glob("shard*-chunk*.pt")):
        chunk = torch.load(path, map_location="cpu")
        stored = list(chunk["layers"]); keep = [stored.index(l) for l in layers]
        for sid, panel in chunk["panels"].items():
            if sid in rows:
                c.append(panel.view(5, len(stored), len(CANDIDATE_POOLS), -1)[:, keep].reshape(5, -1))
                s.append(chunk["states"][rows[sid]["base_state_id"]][keep].reshape(-1)); t.append(rows[sid]["target"])
    info = dict(panels=len(c), base_states=len({rows[k]["base_state_id"] for k in rows}), excluded_test_task_panels=excluded,
                tasks=len({task.get(r["base_state_id"]) or r["task_group"] for r in rows.values()}),
                base_state_ids=sorted({r["base_state_id"] for r in rows.values()}))
    return torch.stack(c), torch.stack(s), torch.tensor(t, dtype=torch.float32), info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True); ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--features", type=Path); ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--source", required=True, choices=[*SOURCES, "outcome"])
    ap.add_argument("--common-sources", default="teacher_pre,teacher_k1,teacher_k3,outcome")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05); ap.add_argument("--layers", default="18,27,36")
    ap.add_argument("--interaction", choices=("joint", "independent"), default="joint")
    ap.add_argument("--scorer", choices=("transformer", "mlp"), default="transformer")
    ap.add_argument("--mlp-depth", type=int, default=1); ap.add_argument("--mlp-width", type=int)
    ap.add_argument("--readout", choices=("linear", "pointer"), default="linear")
    ap.add_argument("--loss", choices=("softce", "bce"), default="softce")
    ap.add_argument("--train-view", choices=("distinct", "natural"), default="distinct")
    ap.add_argument("--train-frac", type=float, default=1.0); ap.add_argument("--frac-seed", type=int, default=20261021)
    ap.add_argument("--extra-panels", type=Path, help="add five-candidate teacher panels (e.g. Piotr's GPT-5.5 picks, "
                    "`target` per row) from their train split to training; tasks of our test split are excluded")
    ap.add_argument("--extra-features", type=Path); ap.add_argument("--extra-states", type=Path,
                    help="states_full.jsonl of the extra panels (task text, to exclude our test tasks)")
    ap.add_argument("--extra-only", action="store_true", help="train on the extra panels alone; select the epoch on "
                    "their validation split")
    ap.add_argument("--extra-max-states", type=int, help="first N base states of whole extra tasks (seeded order)")
    ap.add_argument("--extra-subset-seed", type=int, default=20261023)
    ap.add_argument("--extra-panels-per-state", type=int, help="keep at most this many panels per extra base state")
    ap.add_argument("--last-epoch", action="store_true", help="--extra-only: keep the last epoch (as our branch-state "
                    "heads do) instead of the best on the extra validation split; the curve is still logged")
    args = ap.parse_args()
    if args.extra_only and not args.extra_panels:
        ap.error("--extra-only needs --extra-panels")
    import torch
    from openwebrl.selection_head import CandidateSetHead, HeadConfig, pointwise_bce_loss, selection_loss
    from selection_outcome_experiments import cluster_boot
    torch.set_num_threads(int(__import__("os").environ.get("OMP_NUM_THREADS", "1")))
    loss_fn = selection_loss if args.loss == "softce" else pointwise_bce_loss
    layers = tuple(int(x) for x in args.layers.split(","))

    records = load_actions(args.data)
    cand, state = load_features(args.features or args.data / "features", layers, torch)
    common = None
    for src in args.common_sources.split(","):
        ids = set(targets(records, args.labels, src))
        common = ids if common is None else common & ids
    y = targets(records, args.labels, args.source)
    rec = {x["state_id"]: x for x in records}
    train_ids = sorted(s for s in common if rec[s]["split"] == "train" and s in cand)
    if args.train_frac < 1:                                       # whole training tasks, nested across fractions
        tasks = sorted({rec[s]["task_id"] for s in train_ids})
        random.Random(args.frac_seed).shuffle(tasks)
        keep = set(tasks[:max(1, round(args.train_frac * len(tasks)))])
        train_ids = [s for s in train_ids if rec[s]["task_id"] in keep]
    test_ids = sorted(s for s in rec if rec[s]["split"] == "test" and s in cand)
    extra_c, extra_s, extra_t, extra_info = load_extra(args, rec, layers, torch) if args.extra_panels else (None, None, None, {})
    extra_ids = extra_info.pop("base_state_ids", [])
    if args.extra_only:
        train_ids = []
        val_c, val_s, val_t, val_info = load_extra(args, rec, layers, torch, split="validation")
        val_info.pop("base_state_ids")

    def slot_draws(x):
        """Feature row of every SFT proposal, with its action index."""
        pos = {d: i for i, d in enumerate(x["feature_draws"])}
        return [(pos[d], a) for a, act in enumerate(x["actions"]) for d in act["sft_draws"]]

    def panel(x, view, rng):
        """(feature rows, action index per slot) for one panel."""
        n = len(x["actions"])
        if view == "distinct":
            rows = list(range(n)); acts = list(range(n))       # representatives are feature rows 0..n-1
        else:
            pool = slot_draws(x)
            pick = pool if len(pool) <= 5 else rng.sample(pool, 5)
            rows, acts = [p for p, _ in pick], [a for _, a in pick]
        order = list(range(len(rows))); rng.shuffle(order)
        return [rows[i] for i in order], [acts[i] for i in order]

    def batch(ids, view, rng, with_target=True):
        N = 5
        c = torch.zeros(len(ids), N, cand[ids[0]].shape[-1]); m = torch.zeros(len(ids), N, dtype=torch.bool)
        t = torch.zeros(len(ids), N); acts = []
        for b, s in enumerate(ids):
            rows, a = panel(rec[s], view, rng)
            c[b, :len(rows)] = cand[s][rows].float(); m[b, :len(rows)] = True; acts.append(a)
            if with_target:
                share = y[s]; count = defaultdict(int)
                for k in a:
                    count[k] += 1
                for i, k in enumerate(a):
                    # softce: an action's mass is split over its duplicates; bce: each copy carries the action's value
                    t[b, i] = share[k] / count[k] if args.loss == "softce" else share[k]
        return c, torch.stack([state[s].float() for s in ids]), m, t, acts

    torch.manual_seed(args.seed); rng = random.Random(args.seed)
    config = HeadConfig(hidden_size=state[test_ids[0]].shape[-1] // len(layers), layers=layers,
                        pools=("thought_mean", "action_mean", "last"), d_model=512, depth=2, dropout=0.1,
                        listwise=args.interaction == "joint", interaction=args.interaction, scorer=args.scorer,
                        mlp_depth=args.mlp_depth, mlp_width=args.mlp_width, readout=args.readout)
    head = CandidateSetHead(config)
    fit = [cand[s][:len(rec[s]["actions"])].float() for s in train_ids]
    fit_state = [state[s].float() for s in train_ids]
    if extra_c is not None:
        fit.append(extra_c.reshape(-1, extra_c.shape[-1]).float()); fit_state += list(extra_s.float())
    head.fit_standardizers(torch.cat(fit), torch.stack(fit_state))
    items = [("ours", s) for s in train_ids] + [("extra", i) for i in range(0 if extra_c is None else len(extra_c))]
    params = sum(p.numel() for p in head.parameters())
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = args.epochs * ((len(items) + args.batch_size - 1) // args.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps, pct_start=0.05)
    started = time.time()
    curve, best = [], None
    for _ in range(args.epochs):
        head.train(); order = items[:]; rng.shuffle(order)
        for i in range(0, len(order), args.batch_size):
            chunk = order[i:i + args.batch_size]
            ours = [x for k, x in chunk if k == "ours"]; extra = torch.tensor([x for k, x in chunk if k == "extra"], dtype=torch.long)
            parts = [batch(ours, args.train_view, rng)[:4]] if ours else []
            if len(extra):
                perm = torch.argsort(torch.rand(len(extra), 5), -1)          # candidate order is never informative
                ec = extra_c[extra].gather(1, perm[..., None].expand(-1, -1, extra_c.shape[-1])).float()
                parts.append((ec, extra_s[extra].float(), torch.ones(len(extra), 5, dtype=torch.bool), extra_t[extra].gather(1, perm)))
            c, s, m, t = (torch.cat(x) for x in zip(*parts))
            logits, _ = head(c, s, m)
            loss = loss_fn(logits, t, m)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 1.); opt.step(); sched.step()
        if args.extra_only:                                       # teacher agreement on the extra validation split
            head.eval()
            with torch.no_grad():
                hit = []
                for i in range(0, len(val_c), 1024):
                    lg = head(val_c[i:i + 1024].float(), val_s[i:i + 1024].float(), torch.ones(len(val_c[i:i + 1024]), 5, dtype=torch.bool))[0]
                    hit.append(val_t[i:i + 1024].gather(1, lg.argmax(-1, keepdim=True))[:, 0] > 0)
            agree = torch.cat(hit).float().mean().item()
            curve.append(round(agree, 4))
            if best is None or agree > best[0]:
                best = (agree, len(curve), {k: v.clone() for k, v in head.state_dict().items()})
    if args.extra_only and not args.last_epoch:
        head.load_state_dict(best[2])
    head.eval()

    # offline scoring on held-out draws 3-7
    scored, per = [], {}
    for s in test_ids:
        rates = [held_rate(a, range(3, 8)) for a in rec[s]["actions"]]
        if None in rates:
            continue
        freq = [len(a["sft_draws"]) for a in rec[s]["actions"]]
        scored.append(s); per[s] = dict(rates=rates, sft=sum(f * r for f, r in zip(freq, rates)) / sum(freq))
    picks, natural = {}, {}
    with torch.no_grad():
        for i in range(0, len(scored), 256):
            ids = scored[i:i + 256]
            c, s, m, _, acts = batch(ids, "distinct", random.Random(0), with_target=False)
            for sid, row, a in zip(ids, head(c, s, m)[0].argmax(-1).tolist(), acts):
                picks[sid] = a[row]
        nrng = random.Random(20261021)
        for sid in scored:
            got = []
            for _ in range(20):
                c, s, m, _, acts = batch([sid], "natural", nrng, with_target=False)
                got.append(per[sid]["rates"][acts[0][head(c, s, m)[0].argmax(-1).item()]])
            natural[sid] = sum(got) / len(got)
    five = score_fixed_five(head, scored, rec, cand, state, {s: per[s]["rates"] for s in scored}, torch)
    g = [rec[s]["task_id"] for s in scored]
    head_r = [per[s]["rates"][picks[s]] for s in scored]
    uni = [sum(per[s]["rates"]) / len(per[s]["rates"]) for s in scored]
    sft = [per[s]["sft"] for s in scored]; nat = [natural[s] for s in scored]
    mean = lambda v: round(100 * sum(v) / len(v), 2)
    res = dict(source=args.source, interaction=args.interaction, scorer=args.scorer, loss=args.loss,
               mlp_depth=args.mlp_depth, mlp_width=args.mlp_width, readout=args.readout, train_view=args.train_view,
               train_frac=args.train_frac, seed=args.seed, params=params, train_states=len(train_ids), extra=extra_info,
               train_tasks=len({rec[s]["task_id"] for s in train_ids}), test_states=len(scored), test_tasks=len(set(g)),
               extra_only=args.extra_only, extra_max_states=args.extra_max_states, extra_subset_seed=args.extra_subset_seed,
               extra_panels_per_state=args.extra_panels_per_state, last_epoch=args.last_epoch,
               batch_size=args.batch_size, epochs=args.epochs,
               extra_validation=(dict(val_info, agreement_by_epoch=curve, best_epoch=best[1], best_agreement=round(best[0], 4))
                                 if args.extra_only else None),
               train_seconds=round(time.time() - started, 1),
               fixed_five=dict(head=mean([five[s][0] for s in scored]), sft_sample=mean([five[s][1] for s in scored]),
                               oracle=mean([five[s][2] for s in scored]),
                               head_minus_sft_sample=cluster_boot([five[s][0] - five[s][1] for s in scored], g)),
               success=dict(head=mean(head_r), head_natural_panels=mean(nat), uniform=mean(uni), sft_sample=mean(sft),
                            oracle=mean([max(per[s]["rates"]) for s in scored])),
               head_minus_uniform=cluster_boot([a - b for a, b in zip(head_r, uni)], g),
               head_minus_sft_sample=cluster_boot([a - b for a, b in zip(head_r, sft)], g),
               natural_minus_sft_sample=cluster_boot([a - b for a, b in zip(nat, sft)], g))
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save({"config": config.to_json(), "state_dict": {k: v.cpu() for k, v in head.state_dict().items()},
                "objective": "soft_ce", "loss": args.loss, "params": params, "source": args.source, "seed": args.seed,
                "train_states": len(train_ids), "train_view": args.train_view,
                "extra_base_states": len(extra_ids), "extra_only": args.extra_only}, args.output / "head.pt")
    (args.output / "train-states.json").write_text(json.dumps(train_ids + extra_ids))
    (args.output / "picks.json").write_text(json.dumps(dict(picks=picks, natural=natural)))
    (args.output / "scores.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
