"""RL 8B actor (OpenWebRL/OpenWebRL-8B @ 824a54f, 4,096 response tokens) on Browser Use stealth browsers with US residential
proxy, November 12: RL 8B alone (1 sample) vs RL 8B + default ADM (R3+WA 4B LoRA, arm5-d470d7e52a38, N = 5), T 1.0 / top-p 0.9,
30 steps, 1 run, rolled out judge-deferred and judged afterwards. All three benchmarks, all 1,045 tasks per arm (Browser Use credit topped up twice; the Online-Mind2Web ADM job resumed after a credit pause).
Paired bootstrap over tasks common to both arms (10,000 resamples, seed 20261029, benchmarks resampled independently).
Refuses DEFERRED and dead-key records. Writes the adjacent JSON. Runtime: recipe3-ablations-20261104/stealth/rl8b-proxy-1112/."""
import glob, json, sys
from pathlib import Path
import numpy as np
ROOT = "/gpfs/scrubbed/zixianma/openwebrl-runtime/recipe3-ablations-20261104/stealth/rl8b-proxy-1112/run1"
BENCH = [b for b in ["online_mind2web", "webvoyager", "deepshop"] if b in (sys.argv[1:] or ["online_mind2web", "webvoyager", "deepshop"])]
COND = {"sft1": "RL 8B alone (1 sample)", "arm5-d470d7e52a38": "RL 8B + ADM (R3+WA 4B LoRA, N = 5)"}
B, SEED = 10000, 20261029


def load(c, b):
    out, unjudged = {}, 0
    for f in glob.glob(f"{ROOT}/{b}/{c}/{b}/collection/records/*.json"):
        x = json.load(open(f)); jt = str(x.get("judge_text") or "")
        if jt.startswith("DEFERRED") or jt.strip() == "All judge API retry attempts exhausted.":
            unjudged += 1; continue
        v = str(x["valid"]) == "True"
        out[x["task_id"]] = (v, v and float(x["reward"] or 0) >= 1, x.get("judge_model"))
    return out, unjudged


rng = np.random.default_rng(SEED); res = {"benchmarks": BENCH, "conditions": {c: {"label": l} for c, l in COND.items()}, "judges": {}}
boot = {}
for b in BENCH:
    d = {c: load(c, b) for c in COND}
    tasks = sorted(set(d["sft1"][0]) & set(d["arm5-d470d7e52a38"][0]))
    res["judges"][b] = sorted({v[2] for c in COND for v in d[c][0].values() if v[2]})
    idx = rng.integers(0, len(tasks), (B, len(tasks)))
    for c in COND:
        a = np.array([[d[c][0][t][0], d[c][0][t][1]] for t in tasks], float)
        res["conditions"][c][b] = dict(tasks=len(tasks), unjudged=d[c][1], valid=int(a[:, 0].sum()), successes=int(a[:, 1].sum()),
                                       overall=a[:, 1].mean(), valid_only=a[:, 1].sum() / a[:, 0].sum())
        boot[(b, c)] = (a[idx, 1].mean(1), a[idx, 1].sum(1) / a[idx, 0].sum(1))
diff = {}
for b in BENCH:
    for k, j in (("overall", 0), ("valid_only", 1)):
        dd = 100 * (boot[(b, "arm5-d470d7e52a38")][j] - boot[(b, "sft1")][j])
        diff[f"{b}_{k}_pp"] = [100 * (res["conditions"]["arm5-d470d7e52a38"][b][k] - res["conditions"]["sft1"][b][k]), *np.percentile(dd, [2.5, 97.5])]
for c in COND:
    r = res["conditions"][c]; r["mean_overall"] = float(np.mean([r[b]["overall"] for b in BENCH])); r["mean_valid_only"] = float(np.mean([r[b]["valid_only"] for b in BENCH]))
for k, j in (("overall", 0), ("valid_only", 1)):
    dd = 100 * np.mean([boot[(b, "arm5-d470d7e52a38")][j] - boot[(b, "sft1")][j] for b in BENCH], 0)
    diff[f"mean_{k}_pp"] = [100 * (res["conditions"]["arm5-d470d7e52a38"][f"mean_{k}"] - res["conditions"]["sft1"][f"mean_{k}"]), *np.percentile(dd, [2.5, 97.5])]
res["adm_minus_alone"] = diff
out = json.loads(json.dumps(res, default=float)); Path(__file__).with_suffix(".json").write_text(json.dumps(out, indent=1) + "\n")
for c in COND:
    r = out["conditions"][c]; print(r["label"], {b: f"{r[b]['valid']}/{r[b]['tasks']} {r[b]['successes']} vo {100*r[b]['valid_only']:.1f} ov {100*r[b]['overall']:.1f} unjudged {r[b]['unjudged']}" for b in BENCH}, f"mean ov {100*r['mean_overall']:.1f} vo {100*r['mean_valid_only']:.1f}")
print({k: [round(x, 1) for x in v] for k, v in diff.items()}); print(out["judges"])
