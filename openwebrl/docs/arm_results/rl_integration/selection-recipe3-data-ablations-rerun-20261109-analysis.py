"""Recipe 3 GPT-5.5 H = 4 4B LoRA data ablations, same-window rerun (November 8-9): the 4-panels-per-state, 1-panel-per-state
and 1,000-state subsets re-evaluated alongside the full-data LoRA, 3 runs each, after the November 4-6 batch showed subsets
ahead of the full data across windows.

SFT 4B actor, N = 5, T 1.0 / top-p 0.9, 30 steps, all 1,045 tasks, judges as in the Recipe 3 evals (Online-Mind2Web
gpt-5.6-terra, WebVoyager and DeepShop GPT-4o), rolled out judge-deferred and judged afterwards. Subsets whose records are not
all judged yet are listed under "pending" and left out. Paired bootstrap over tasks common to every run and condition
(10,000 resamples, seed 20261029, benchmarks resampled independently, per-task success averaged over runs). Refuses records
still DEFERRED or with a dead-key judge message. Writes the adjacent JSON.
"""
import glob
import json
from pathlib import Path

import numpy as np

ROOT = "/gpfs/scrubbed/zixianma/openwebrl-runtime/recipe3-ablations-20261104/eval-1108"
SLUGS = {"full": "arm5-9fadc8dc4655", "pps4": "arm5-9117fa5a37bb", "pps1": "arm5-9f25a8bd0da7", "s1000": "arm5-fc760b23e227"}
BENCHES = ["online_mind2web", "webvoyager", "deepshop"]
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


runs = sorted(glob.glob(f"{ROOT}/run*"))
data = {k: {b: [load(f"{r}/{b}/{s}/{b}") for r in runs] for b in BENCHES} for k, s in SLUGS.items()}
pending = [k for k in SLUGS if any(rf for b in BENCHES for _, rf in data[k][b])]
assert "full" not in pending
keys = [k for k in SLUGS if k not in pending]
rng = np.random.default_rng(SEED)
res = {"runs": len(runs), "slugs": SLUGS, "pending": pending, "rates": {}}
boot = {}
for b in BENCHES:
    tasks = sorted(set.intersection(*[set(d) for k in keys for d, _ in data[k][b]]))
    idx = rng.integers(0, len(tasks), (B, len(tasks)))
    for k in keys:
        rs = [d for d, _ in data[k][b]]
        a = np.array([[sum(r[t][0] for r in rs), sum(r[t][1] for r in rs)] for t in tasks], float)
        n = len(rs) * len(tasks)
        res["rates"].setdefault(k, {})[b] = {"tasks": len(tasks), "episodes": n, "valid": int(a[:, 0].sum()), "successes": int(a[:, 1].sum()),
                                              "overall": 100 * a[:, 1].sum() / n, "valid_only": 100 * a[:, 1].sum() / a[:, 0].sum()}
        boot[(b, k)] = (a[idx, 1].sum(1) / n, a[idx, 1].sum(1) / a[idx, 0].sum(1))
for k in keys:
    r = res["rates"][k]
    r["mean_overall"] = float(np.mean([r[b]["overall"] for b in BENCHES]))
    r["mean_valid_only"] = float(np.mean([r[b]["valid_only"] for b in BENCHES]))
    if k == "full":
        continue
    for b in BENCHES:
        d = 100 * (boot[(b, k)][0] - boot[(b, "full")][0])
        r[b]["diff_overall_vs_full_pp"] = [r[b]["overall"] - res["rates"]["full"][b]["overall"], *np.percentile(d, [2.5, 97.5])]
    for m, j in (("overall", 0), ("valid_only", 1)):
        d = 100 * np.mean([boot[(b, k)][j] - boot[(b, "full")][j] for b in BENCHES], 0)
        r[f"diff_mean_{m}_vs_full_pp"] = [r[f"mean_{m}"] - res["rates"]["full"][f"mean_{m}"], *np.percentile(d, [2.5, 97.5])]
out = json.loads(json.dumps(res, default=float))
Path(__file__).with_suffix(".json").write_text(json.dumps(out, indent=1) + "\n")
print("pending", pending)
for k, r in out["rates"].items():
    print(k, [f"{r[b]['valid']}/{r[b]['episodes']} {r[b]['valid_only']:.1f}" for b in BENCHES], f"{r['mean_overall']:.1f} {r['mean_valid_only']:.1f}",
          {m: [round(x, 1) for x in v] for m, v in r.items() if m.startswith("diff")})
