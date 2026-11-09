"""Kev-style decision pass on Recipe 3 GPT-5.5 H = 4 labels, live on Online-Mind2Web (November 8-9), with same-window SFT.

Kev-A: LoRA on every token of the 4B actor plus a decision-token pointer head; Kev-B: the same, but prompt tokens run through
the unadapted actor (so the actor's generation cache could be reused). Both evaluated as head5x (separate scorer pass).
SFT 4B actor, N = 5, T 1.0 / top-p 0.9, 30 steps, 300 tasks, 2 runs, judge gpt-5.6-terra, rolled out judge-deferred.
Paired bootstrap over tasks (10,000 resamples, seed 20261029, per-task success averaged over runs). Refuses records still
DEFERRED or with a dead-key judge message. Writes the adjacent JSON.
"""
import glob
import json
from pathlib import Path

import numpy as np

ROOT = "/gpfs/scrubbed/zixianma/openwebrl-runtime/recipe3-ablations-20261104/eval-kev"
B = "online_mind2web"
REF = "sft1"
CONDITIONS = {REF: "SFT 4B alone (1 sample)", "head5x-9ff81e7c894f": "Kev-A (LoRA on all tokens)",
              "head5x-18ab53442be0": "Kev-B (no LoRA on prompt tokens)"}
NB, SEED = 10000, 20261029


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
data = {c: [load(f"{r}/{B}/{c}/{B}") for r in runs] for c in CONDITIONS}
assert all(rf == 0 for c in data for _, rf in data[c]), "unjudged records"
tasks = sorted(set.intersection(*[set(d) for c in data for d, _ in data[c]]))
judges = sorted({d[t][2] for c in data for d, _ in data[c] for t in d if d[t][2]})
rng = np.random.default_rng(SEED)
idx = rng.integers(0, len(tasks), (NB, len(tasks)))
res = {"benchmark": B, "tasks": len(tasks), "runs": len(runs), "judge_models": judges, "conditions": {}}
boot = {}
for c, label in CONDITIONS.items():
    rs = [d for d, _ in data[c]]
    a = np.array([[sum(r[t][0] for r in rs), sum(r[t][1] for r in rs)] for t in tasks], float)
    n_ep = len(rs) * len(tasks)
    per_run = [sum(r[t][1] for t in tasks) / len(tasks) for r in rs]
    res["conditions"][c] = {"label": label, "episodes": n_ep, "valid": int(a[:, 0].sum()), "successes": int(a[:, 1].sum()),
                            "overall": a[:, 1].sum() / n_ep, "valid_only": a[:, 1].sum() / a[:, 0].sum(), "overall_per_run": per_run}
    boot[c] = (a[idx, 1].sum(1) / n_ep, a[idx, 1].sum(1) / a[idx, 0].sum(1))
for c in CONDITIONS:
    if c == REF:
        continue
    r = res["conditions"][c]
    for k, j in (("overall", 0), ("valid_only", 1)):
        d = 100 * (boot[c][j] - boot[REF][j])
        r[f"diff_{k}_pp"] = [100 * (r[k] - res["conditions"][REF][k]), *np.percentile(d, [2.5, 97.5])]
a, b = "head5x-9ff81e7c894f", "head5x-18ab53442be0"
res["kevB_minus_kevA_overall_pp"] = [100 * (res["conditions"][b]["overall"] - res["conditions"][a]["overall"]),
                                     *np.percentile(100 * (boot[b][0] - boot[a][0]), [2.5, 97.5])]
out = json.loads(json.dumps(res, default=float))
Path(__file__).with_suffix(".json").write_text(json.dumps(out, indent=1) + "\n")
for c, r in out["conditions"].items():
    print(r["label"], r["valid"], r["successes"], round(100 * r["overall"], 1), round(100 * r["valid_only"], 1),
          [round(100 * x, 1) for x in r["overall_per_run"]], {k: [round(x, 1) for x in v] for k, v in r.items() if k.startswith("diff")})
print("B - A", [round(x, 1) for x in out["kevB_minus_kevA_overall_pp"]], out["judge_models"])
