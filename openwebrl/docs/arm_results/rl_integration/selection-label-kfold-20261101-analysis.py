#!/usr/bin/env python3
"""Pool the task-level 5-fold selection-label comparison (out-of-fold heads, teacher picks) into kfold_results.json.

Inputs: scores/f{k}-per-state.json from score.sbatch (score_label_stages_kfold.py) for k = 0-4. Per state (fixed five:
first five SFT proposals by draw index, duplicates kept; success on held-out draws 3-7, valid draws only):
  base = one SFT sample from the five; teacher = argmax label among the five (ties averaged); head = the out-of-fold head's
  pick (seeds 0-2 averaged per state, and seed 0 alone). Gains are minus base, in pp.
Intervals: paired 95% percentile bootstrap over whole tasks (10,000 resamples, numpy default_rng(20261101)); the mean
over resampled states (ratio of sums), as selection_outcome_experiments.cluster_boot. Also: per-fold, fold 0 alone, the
teacher on each original split, pairwise source contrasts, and an exact reproduction check of fold-0 seed-0 against the
published selection-label-same-state-20261027.json (that file used cluster_boot(seed=0), recomputed the same way here).

Adding a source (e.g. teacher_k5): train/score with SOURCES including it (see train.sbatch); it is picked up here from
the per-state files automatically.

  analyze_kfold.py
"""
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

W = Path(__file__).resolve().parent
sys.path.insert(0, str(W / "source" / "scripts"))
from selection_outcome_experiments import cluster_boot  # noqa: E402

R = Path("/gpfs/scrubbed/zixianma/openwebrl-runtime/selection-head-20261008/sft-actions-20261021")
OLD = Path("/gpfs/scrubbed/zixianma/openwebrl-runtime/docs-worktree-arm-selection/openwebrl/docs/arm_results/"
           "rl_integration/selection-label-same-state-20261027.json")
K, B, SEED = 5, 10000, 20261101


def boot(diff, groups, rng_seed=SEED, n=B):
    """[mean, lo, hi] in pp; task-clustered percentile bootstrap (ratio of sums over resampled tasks)."""
    tasks = sorted(set(groups)); idx = {t: i for i, t in enumerate(tasks)}
    s = np.zeros(len(tasks)); c = np.zeros(len(tasks))
    for d, g in zip(diff, groups):
        s[idx[g]] += d; c[idx[g]] += 1
    rng = np.random.default_rng(rng_seed)
    draw = rng.integers(0, len(tasks), size=(n, len(tasks)))
    m = s[draw].sum(1) / c[draw].sum(1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    return [round(100 * float(np.mean(diff)), 2), round(100 * float(lo), 2), round(100 * float(hi), 2)]


def main():
    per = [json.loads((W / "scores" / f"f{k}-per-state.json").read_text()) for k in range(K)]
    sources = list(per[0]["teacher"])
    orig_split = {}
    for line in open(R / "actions.jsonl"):
        x = json.loads(line); orig_split[x["state_id"]] = x["split"]
    rows = {}                                                     # state -> dict
    for k, p in enumerate(per):
        for s, b in p["base"].items():
            assert s not in rows, s
            r = dict(fold=k, task=p["task"][s], split=orig_split[s], base=b)
            for src in sources:
                r[f"teacher/{src}"] = p["teacher"][src][s]
                vals = [v[s] for v in p["head_seeds"][src]["values"]]
                r[f"head/{src}"] = sum(vals) / len(vals); r[f"head_seed0/{src}"] = vals[0]; r[f"nseeds/{src}"] = len(vals)
            rows[s] = r
    ids = sorted(rows)
    kinds = ["teacher", "head", "head_seed0"]

    def block(sel):
        g = [rows[s]["task"] for s in sel]
        out = dict(states=len(sel), tasks=len(set(g)), sft_sample=round(100 * np.mean([rows[s]["base"] for s in sel]), 2))
        for src in sources:
            o = {}
            for kind in kinds:
                v = [rows[s][f"{kind}/{src}"] for s in sel]
                o[kind] = round(100 * np.mean(v), 2)
                o[f"{kind}_minus_sft"] = boot([a - rows[s]["base"] for a, s in zip(v, sel)], g)
            out[src] = o
        return out

    res = dict(note=__doc__.split("\n\n")[0], bootstrap=dict(resamples=B, seed=SEED, cluster="task_id", interval="percentile 95%"),
               sources=sources, seeds={src: rows[ids[0]][f"nseeds/{src}"] for src in sources},
               fold_sizes={k: dict(states=sum(rows[s]["fold"] == k for s in ids),
                                   tasks=len({rows[s]["task"] for s in ids if rows[s]["fold"] == k})) for k in range(K)})
    res["pooled"] = block(ids)
    res["fold0"] = block([s for s in ids if rows[s]["fold"] == 0])
    res["per_fold"] = {k: block([s for s in ids if rows[s]["fold"] == k]) for k in range(K)}
    res["teacher_by_original_split"] = {
        sp: {**{k: v for k, v in block([s for s in ids if rows[s]["split"] == sp]).items() if k in ("states", "tasks", "sft_sample")},
             **{src: dict(teacher=b[src]["teacher"], teacher_minus_sft=b[src]["teacher_minus_sft"])
                for b in [block([s for s in ids if rows[s]["split"] == sp])] for src in sources}}
        for sp in ("train", "test")}
    g = [rows[s]["task"] for s in ids]
    res["pooled_contrasts"] = {
        kind: {f"{a}-{b}": boot([rows[s][f"{kind}/{a}"] - rows[s][f"{kind}/{b}"] for s in ids], g)
               for i, a in enumerate(sources) for b in sources[i + 1:]}
        for kind in ("teacher", "head")}
    res["pooled_head_minus_teacher"] = {src: boot([rows[s][f"head/{src}"] - rows[s][f"teacher/{src}"] for s in ids], g)
                                        for src in sources}

    # exact reproduction check: fold 0, seed 0, cluster_boot(seed=0) as in the published file
    old = json.loads(OLD.read_text())["test"]
    f0 = [s for s in ids if rows[s]["fold"] == 0]; g0 = [rows[s]["task"] for s in f0]
    check = dict(states=len(f0), published_states=json.loads(OLD.read_text())["splits"]["test"]["states"],
                 sft_sample=round(100 * np.mean([rows[s]["base"] for s in f0]), 2), published_sft_sample=old["sft_sample"])
    for src in sources:
        if src not in old:
            continue
        mine_t = [round(v, 2) for v in cluster_boot([rows[s][f"teacher/{src}"] - rows[s]["base"] for s in f0], g0)]
        mine_h = [round(v, 2) for v in cluster_boot([rows[s][f"head_seed0/{src}"] - rows[s]["base"] for s in f0], g0)]
        check[src] = dict(teacher=mine_t, published_teacher=old[src]["teacher_minus_sft"], head_seed0=mine_h,
                          published_head=old[src]["head_minus_sft"],
                          match=bool(np.allclose(mine_t, old[src]["teacher_minus_sft"], atol=0.011)
                                     and np.allclose(mine_h, old[src]["head_minus_sft"], atol=0.011)))
    res["fold0_seed0_reproduction"] = check
    res["per_state"] = {s: {k: v for k, v in rows[s].items() if not k.startswith("nseeds/")} for s in ids}
    (W / "kfold_results.json").write_text(json.dumps(res, indent=1))

    def fmt(x):
        return f"{x[0]:+.1f} [{x[1]:+.1f}, {x[2]:+.1f}]"
    print(f"pooled {res['pooled']['states']} states / {res['pooled']['tasks']} tasks; sft {res['pooled']['sft_sample']}")
    for src in sources:
        p = res["pooled"][src]; f = res["fold0"][src]
        print(f"{src:13s} teacher {fmt(p['teacher_minus_sft'])}  head3 {fmt(p['head_minus_sft'])}  head0 {fmt(p['head_seed0_minus_sft'])}"
              f"  | fold0 teacher {fmt(f['teacher_minus_sft'])} head3 {fmt(f['head_minus_sft'])} head0 {fmt(f['head_seed0_minus_sft'])}")
    print("per-fold head3:", {src: [res["per_fold"][k][src]["head_minus_sft"][0] for k in range(K)] for src in sources})
    print("per-fold teacher:", {src: [res["per_fold"][k][src]["teacher_minus_sft"][0] for k in range(K)] for src in sources})
    print("repro:", json.dumps({k: (v if not isinstance(v, dict) else v["match"]) for k, v in check.items()}))


if __name__ == "__main__":
    main()
