"""Training-seed replication of the 4.4M (d128-L2) and 22.6M (d512-L2, default) Recipe 3 heads, live with same-window SFT (Nov 11 evening).

Joint transformer, soft CE, Recipe 3 GPT-5.5 H = 4 labels, lr 3e-4; seeds 1 and 2 newly trained (2 eval runs each); seed 0 heads
and SFT 4B alone re-evaluated in the same window (1 run each). SFT 4B actor, N = 5, T 1.0 / top-p 0.9, 30 steps, all tasks,
rolled out judge-deferred (16:00-21:00 PST). Pooled size rows average per-task success over all seeds and runs of a size.
Paired bootstrap over tasks common to every condition (10,000 resamples, seed 20261029, benchmarks resampled independently).
Refuses records still DEFERRED or with a dead-key judge message. Writes the adjacent JSON.
"""
import glob
import json
from pathlib import Path

import numpy as np

ROOT = "/gpfs/scrubbed/zixianma/openwebrl-runtime/recipe3-ablations-20261104/eval-seed"
BENCHES = ["online_mind2web", "webvoyager", "deepshop"]
HEADS = {"sft1": ("SFT 4B alone (1 sample)", None, None), "head5-5fbddd8b2031": ("4.4M seed 0", "4.4M", 0),
         "head5-8211c6dc15c7": ("4.4M seed 1", "4.4M", 1), "head5-50b0eab40285": ("4.4M seed 2", "4.4M", 2),
         "head5-91e7a5a67417": ("22.6M seed 0 (default)", "22.6M", 0), "head5-62b65af3bdf0": ("22.6M seed 1", "22.6M", 1),
         "head5-993c5d16b600": ("22.6M seed 2", "22.6M", 2)}
B, SEED = 10000, 20261029


def load(d):
    out, refused = {}, 0
    for f in glob.glob(f"{d}/collection/records/*.json"):
        x = json.load(open(f))
        jt = str(x.get("judge_text") or "")
        if jt.startswith("DEFERRED") or jt.strip() == "All judge API retry attempts exhausted.":
            refused += 1
            continue
        valid = str(x["valid"]) == "True"
        out[x["task_id"]] = (valid, valid and float(x["reward"] or 0) >= 1)
    return out, refused


rng = np.random.default_rng(SEED)
res = {"conditions": {}, "pooled": {}, "contrasts": {}}
arr = {}                                   # (bench, key) -> per-task [valid, success, episodes]
for b in BENCHES:
    runs = {c: [load(d) for d in sorted(glob.glob(f"{ROOT}/run*/{b}/{c}/{b}"))] for c in HEADS}
    assert all(rf == 0 for c in runs for _, rf in runs[c]), f"unjudged records in {b}"
    tasks = sorted(set.intersection(*[set(r) for c in runs for r, _ in runs[c]]))
    for c, rs in runs.items():
        rs = [r for r, _ in rs]
        arr[(b, c)] = np.array([[sum(r[t][0] for r in rs), sum(r[t][1] for r in rs), len(rs)] for t in tasks], float)
    for size in ("4.4M", "22.6M"):
        arr[(b, size)] = sum(arr[(b, c)] for c, v in HEADS.items() if v[1] == size)
    res.setdefault("tasks", {})[b] = len(tasks)
    res.setdefault("_idx", {})[b] = rng.integers(0, len(tasks), (B, len(tasks)))


def rates(a):
    return a[:, 1].sum() / a[:, 2].sum(), a[:, 1].sum() / a[:, 0].sum()


def boot(a, idx):
    s, v, n = a[idx, 1].sum(1), a[idx, 0].sum(1), a[idx, 2].sum(1)
    return s / n, s / v


keys = list(HEADS) + ["4.4M", "22.6M"]
for k in keys:
    row = {}
    for b in BENCHES:
        a = arr[(b, k)]; ov, vo = rates(a)
        row[b] = dict(valid=int(a[:, 0].sum()), episodes=int(a[:, 2].sum()), successes=int(a[:, 1].sum()), overall=ov, valid_only=vo)
    row["mean_overall"] = float(np.mean([row[b]["overall"] for b in BENCHES]))
    row["mean_valid_only"] = float(np.mean([row[b]["valid_only"] for b in BENCHES]))
    (res["conditions"] if k in HEADS else res["pooled"])[k] = dict(label=HEADS[k][0] if k in HEADS else f"{k}, 3 seeds pooled", **row)


def contrast(x, y):
    out = {}
    for j, k in ((0, "overall"), (1, "valid_only")):
        d = 100 * np.mean([boot(arr[(b, x)], res["_idx"][b])[j] - boot(arr[(b, y)], res["_idx"][b])[j] for b in BENCHES], 0)
        pt = 100 * np.mean([rates(arr[(b, x)])[j] - rates(arr[(b, y)])[j] for b in BENCHES])
        out[f"mean_{k}_pp"] = [pt, *np.percentile(d, [2.5, 97.5])]
    for b in BENCHES:
        d = 100 * (boot(arr[(b, x)], res["_idx"][b])[1] - boot(arr[(b, y)], res["_idx"][b])[1])
        out[f"{b}_valid_only_pp"] = [100 * (rates(arr[(b, x)])[1] - rates(arr[(b, y)])[1]), *np.percentile(d, [2.5, 97.5])]
    return out


for k in keys:
    if k != "sft1":
        res["contrasts"][f"{k} - sft1"] = contrast(k, "sft1")
res["contrasts"]["4.4M - 22.6M (pooled)"] = contrast("4.4M", "22.6M")
for s in (0, 1, 2):
    a = [c for c, v in HEADS.items() if v[1] == "4.4M" and v[2] == s][0]; z = [c for c, v in HEADS.items() if v[1] == "22.6M" and v[2] == s][0]
    res["contrasts"][f"4.4M - 22.6M (seed {s})"] = contrast(a, z)
res.pop("_idx")
out = json.loads(json.dumps(res, default=float))
Path(__file__).with_suffix(".json").write_text(json.dumps(out, indent=1) + "\n")
for k, r in {**out["conditions"], **out["pooled"]}.items():
    print(f"{r['label']:28s}", [f"{100 * r[b]['valid_only']:.1f}" for b in BENCHES], f"{100 * r['mean_overall']:.1f} {100 * r['mean_valid_only']:.1f}")
for k, v in out["contrasts"].items():
    print(k, [round(x, 1) for x in v["mean_overall_pp"]])
