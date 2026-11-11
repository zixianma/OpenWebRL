"""Kev-style decision pass on Recipe 3 GPT-5.5 H = 4 labels, live on all three benchmarks with same-window SFT.

Kev-A: LoRA on every token of the 4B actor plus a decision-token pointer head; Kev-B: the same, but prompt tokens run through
the unadapted actor. Both evaluated as head5x (separate scorer pass). SFT 4B actor, N = 5, T 1.0 / top-p 0.9, 30 steps, all tasks,
2 runs per condition, rolled out judge-deferred. Online-Mind2Web ran November 8 (judge gpt-5.6-terra); WebVoyager and DeepShop
November 10, 16:00-21:00 PST (judge GPT-4o); SFT ran in each window. Paired bootstrap over tasks common to every run and condition
(10,000 resamples, seed 20261029, benchmarks resampled independently, per-task success averaged over runs). Refuses records still
DEFERRED or with a dead-key judge message. Writes the adjacent JSON.
"""
import glob
import json
from pathlib import Path

import numpy as np

ROOT = "/gpfs/scrubbed/zixianma/openwebrl-runtime/recipe3-ablations-20261104/eval-kev"
BENCHES = ["online_mind2web", "webvoyager", "deepshop"]
REF = "sft1"
CONDITIONS = {REF: "SFT 4B alone (1 sample)", "head5x-9ff81e7c894f": "Kev-A (LoRA on all tokens)",
              "head5x-18ab53442be0": "Kev-B (no LoRA on prompt tokens)"}
A, KB = "head5x-9ff81e7c894f", "head5x-18ab53442be0"
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
for c in (A, KB):
    res["conditions"][c]["minus_sft"] = contrast(c, REF)
res["kevB_minus_kevA"] = contrast(KB, A)
out = json.loads(json.dumps(res, default=float))
Path(__file__).with_suffix(".json").write_text(json.dumps(out, indent=1) + "\n")
for c, r in out["conditions"].items():
    print(r["label"], [f"{r[b]['valid']}/{r[b]['episodes']} {r[b]['successes']} {100*r[b]['valid_only']:.1f}" for b in BENCHES],
          f"{100*r['mean_overall']:.1f} {100*r['mean_valid_only']:.1f}",
          {k: [round(x, 1) for x in v] for k, v in r.get("minus_sft", {}).items()})
print("B - A", {k: [round(x, 1) for x in v] for k, v in out["kevB_minus_kevA"].items()}, out["judge_models"])
