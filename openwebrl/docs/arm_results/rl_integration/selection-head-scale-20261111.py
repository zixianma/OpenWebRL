"""Selection-head model scale (Recipe 3 GPT-5.5 H = 4 labels), live on all three benchmarks with same-window SFT (November 11).

Joint (bidirectional) transformer heads with soft-CE loss, same data, recipe and lr 3e-4, differing only in width / depth:
d128-L2 4.4M, d256-L2 9.6M, default d512-L2 22.6M (ADM-22M), d2048-L4 272.8M (the 1.8B d4096-L8 head is not in this run:
its CPU scoring OOM-killed the 120G eval job). Evaluated as head5 on the SFT 4B actor, N = 5, T 1.0 / top-p 0.9, 30 steps,
all tasks, 2 runs per condition, rolled out judge-deferred on November 11 (00:59-09:30 PST) and judged afterwards. Paired bootstrap over tasks common to every run and condition
(10,000 resamples, seed 20261029, benchmarks resampled independently, per-task success averaged over runs). Refuses records still
DEFERRED or with a dead-key judge message. Writes the adjacent JSON.
"""
import glob
import json
from pathlib import Path

import numpy as np

ROOT = "/gpfs/scrubbed/zixianma/openwebrl-runtime/recipe3-ablations-20261104/eval-scale"
BENCHES = ["online_mind2web", "webvoyager", "deepshop"]
REF = "sft1"
CONDITIONS = {REF: "SFT 4B alone (1 sample)", "head5-5fbddd8b2031": "d128-L2 (4.4M)", "head5-fa31796f1fe0": "d256-L2 (9.6M)",
              "head5-91e7a5a67417": "d512-L2 (22.6M, default)", "head5-938477dffad2": "d2048-L4 (272.8M)"}
DEF = "head5-91e7a5a67417"
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
        out[x["task_id"]] = (valid, valid and float(x["reward"] or 0) >= 1, x.get("judge_model"))
    return out, refused


runs = sorted(glob.glob(f"{ROOT}/run*"))
rng = np.random.default_rng(SEED)
res, boot = {"runs": len(runs), "benchmarks": BENCHES, "conditions": {}, "judge_models": {}}, {}
for b in BENCHES:
    data = {c: [load(f"{r}/{b}/{c}/{b}") for r in runs] for c in CONDITIONS}
    assert all(rf == 0 for c in data for _, rf in data[c]), f"unjudged records in {b}"
    tasks = sorted(set.intersection(*[set(d) for c in data for d, _ in data[c]]))
    res["judge_models"][b] = sorted({d[t][2] for c in data for d, _ in data[c] for t in d if d[t][2]})
    idx = rng.integers(0, len(tasks), (B, len(tasks)))
    for c in CONDITIONS:
        rs = [d for d, _ in data[c]]
        a = np.array([[sum(r[t][0] for r in rs), sum(r[t][1] for r in rs)] for t in tasks], float)
        n_ep = len(rs) * len(tasks)
        res["conditions"].setdefault(c, {"label": CONDITIONS[c]})[b] = {
            "tasks": len(tasks), "episodes": n_ep, "valid": int(a[:, 0].sum()), "successes": int(a[:, 1].sum()),
            "overall": a[:, 1].sum() / n_ep, "valid_only": a[:, 1].sum() / a[:, 0].sum()}
        boot[(b, c)] = (a[idx, 1].sum(1) / n_ep, a[idx, 1].sum(1) / a[idx, 0].sum(1))


def contrast(c, ref):
    out = {}
    for b in BENCHES:
        d = 100 * (boot[(b, c)][1] - boot[(b, ref)][1])
        out[f"{b}_valid_only_pp"] = [100 * (res["conditions"][c][b]["valid_only"] - res["conditions"][ref][b]["valid_only"]), *np.percentile(d, [2.5, 97.5])]
    for k, j in (("overall", 0), ("valid_only", 1)):
        d = 100 * np.mean([boot[(b, c)][j] - boot[(b, ref)][j] for b in BENCHES], 0)
        out[f"mean_{k}_pp"] = [100 * (res["conditions"][c][f"mean_{k}"] - res["conditions"][ref][f"mean_{k}"]), *np.percentile(d, [2.5, 97.5])]
    return out


for c in CONDITIONS:
    r = res["conditions"][c]
    r["mean_overall"] = float(np.mean([r[b]["overall"] for b in BENCHES]))
    r["mean_valid_only"] = float(np.mean([r[b]["valid_only"] for b in BENCHES]))
for c in CONDITIONS:
    if c != REF:
        res["conditions"][c]["minus_sft"] = contrast(c, REF)
        if c != DEF:
            res["conditions"][c]["minus_default"] = contrast(c, DEF)
out = json.loads(json.dumps(res, default=float))
Path(__file__).with_suffix(".json").write_text(json.dumps(out, indent=1) + "\n")
for c, r in out["conditions"].items():
    print(r["label"], [f"{r[b]['valid']}/{r[b]['episodes']} {r[b]['successes']} {100*r[b]['valid_only']:.1f}" for b in BENCHES],
          f"{100*r['mean_overall']:.1f} {100*r['mean_valid_only']:.1f}",
          {k: [round(x, 1) for x in v] for k, v in r.get("minus_sft", {}).items() if k.startswith("mean")},
          {k: [round(x, 1) for x in v] for k, v in r.get("minus_default", {}).items() if k.startswith("mean")})
print(out["judge_models"])
