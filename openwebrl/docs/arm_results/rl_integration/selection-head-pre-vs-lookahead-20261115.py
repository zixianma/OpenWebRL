"""Pre-execution vs lookahead labels across selection-head sizes, one window (November 14-15): heads of four sizes (23K linear
probe, d128-L2 4.4M, d512-L2 22.6M, d4096-L8 1.8B), each trained on the same 18,781 Recipe 3 panels with GPT-5.5 lookahead (H = 4,
K = 3) labels and with GPT-5.5 pre-execution labels (same recipe, lr 3e-4, training seed 0), plus SFT 4B alone; 3 runs each on all
1,045 tasks, SFT 4B actor, N = 5, T 1.0 / top-p 0.9, vlm/normal, rolled out judge-deferred and judged afterwards. Paired bootstrap
over tasks common to every condition (10,000 resamples, seed 20261029, benchmarks resampled independently, per-task success
averaged over runs). Contrasts: pre - lookahead at each size and pooled over sizes (per-task mean of the four sizes), each head -
SFT alone. Refuses DEFERRED and dead-key records. Writes the adjacent JSON."""
import glob, json
from pathlib import Path
import numpy as np
ROOT = "/gpfs/scrubbed/zixianma/openwebrl-runtime/recipe3-ablations-20261104/eval-pre"
BENCH = ["online_mind2web", "webvoyager", "deepshop"]
B, SEED = 10000, 20261029
SIZES = [("probe", "linear probe", 23041), ("d128", "d128-L2", 4362754), ("d512", "d512-L2", 22562818), ("d4096", "d4096-L8", 1770463234)]
SLUG = {("probe", "lookahead"): "head5-ef597d45f3b1", ("probe", "pre"): "head5-c8dc992eb5ad",
        ("d128", "lookahead"): "head5-5fbddd8b2031", ("d128", "pre"): "head5-cf9c67a0d0f2",
        ("d512", "lookahead"): "head5-91e7a5a67417", ("d512", "pre"): "head5-cecccef33bba",
        ("d4096", "lookahead"): "head5-7bbad4567d44", ("d4096", "pre"): "head5-cd581ffe7dd0"}
CONDS = {"sft": "sft1", **{f"{s}-{l}": v for (s, l), v in SLUG.items()}}


def load(slug, b):
    rs = []
    for r in sorted(glob.glob(f"{ROOT}/run*")):
        d = {}
        for f in glob.glob(f"{r}/{b}/{slug}/{b}/collection/records/*.json"):
            x = json.load(open(f)); jt = str(x.get("judge_text") or "")
            assert not jt.startswith("DEFERRED") and jt.strip() != "All judge API retry attempts exhausted.", f
            v = str(x["valid"]) == "True"; d[x["task_id"]] = (v, v and float(x["reward"] or 0) >= 1)
        rs.append(d)
    return rs


rng = np.random.default_rng(SEED)
res, boot = {"conditions": {}, "contrasts": {}}, {}
for b in BENCH:
    data = {k: load(s, b) for k, s in CONDS.items()}
    tasks = sorted(set.intersection(*[set(r) for k in data for r in data[k]]))
    idx = rng.integers(0, len(tasks), (B, len(tasks)))
    for k, rs in data.items():
        s = np.array([np.mean([r[t][1] for r in rs]) for t in tasks]); v = np.array([np.mean([r[t][0] for r in rs]) for t in tasks])
        res["conditions"].setdefault(k, {"slug": CONDS[k], "runs": len(rs)})[b] = dict(
            tasks=len(tasks), episodes=len(rs) * len(tasks), valid=int(round(v.sum() * len(rs))), successes=int(round(s.sum() * len(rs))),
            overall=float(s.mean()), valid_only=float(s.sum() / v.sum()))
        boot[(k, b)] = (s, v, idx)
for k, r in res["conditions"].items():
    r["mean_overall"] = float(np.mean([r[b]["overall"] for b in BENCH])); r["mean_valid_only"] = float(np.mean([r[b]["valid_only"] for b in BENCH]))


def stats(arms_a, arms_b):
    """Mean over arms_a minus mean over arms_b (per-task mean over arms), on mean overall and mean valid-only."""
    pt, bo, pv, bv = [], [], [], []
    for b in BENCH:
        idx = boot[(arms_a[0], b)][2]
        sa = np.mean([boot[(k, b)][0] for k in arms_a], 0); va = np.mean([boot[(k, b)][1] for k in arms_a], 0)
        sb = np.mean([boot[(k, b)][0] for k in arms_b], 0); vb = np.mean([boot[(k, b)][1] for k in arms_b], 0)
        pt.append(sa.mean() - sb.mean()); bo.append(sa[idx].mean(1) - sb[idx].mean(1))
        pv.append(sa.sum() / va.sum() - sb.sum() / vb.sum()); bv.append(sa[idx].sum(1) / va[idx].sum(1) - sb[idx].sum(1) / vb[idx].sum(1))
    f = lambda p, q: [100 * float(np.mean(p)), *map(float, np.percentile(100 * np.mean(q, 0), [2.5, 97.5]))]
    return dict(mean_overall_pp=f(pt, bo), mean_valid_only_pp=f(pv, bv))


for s, name, _ in SIZES:
    res["contrasts"][f"{s}: pre - lookahead"] = stats([f"{s}-pre"], [f"{s}-lookahead"])
res["contrasts"]["all sizes: pre - lookahead"] = stats([f"{s}-pre" for s, *_ in SIZES], [f"{s}-lookahead" for s, *_ in SIZES])
for k in CONDS:
    if k != "sft":
        res["conditions"][k]["minus_sft"] = stats([k], ["sft"])
res["sizes"] = [dict(key=s, name=n, params=p) for s, n, p in SIZES]
Path(__file__).with_suffix(".json").write_text(json.dumps(res, indent=1) + "\n")
for k, r in res["conditions"].items():
    print(k, r["runs"], [f"{r[b]['valid']}/{r[b]['episodes']} {r[b]['successes']} {100*r[b]['overall']:.1f}" for b in BENCH],
          f"{100*r['mean_overall']:.1f} {100*r['mean_valid_only']:.1f}", [round(x, 1) for x in r.get("minus_sft", {}).get("mean_overall_pp", [])])
for k, v in res["contrasts"].items():
    print(k, [round(x, 1) for x in v["mean_overall_pp"]], [round(x, 1) for x in v["mean_valid_only_pp"]])
