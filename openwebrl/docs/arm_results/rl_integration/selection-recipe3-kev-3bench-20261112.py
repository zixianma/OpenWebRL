"""Kev-style decision pass on Recipe 3 GPT-5.5 H = 4 labels: 3 runs on all three benchmarks, plus a same-window
comparison with the Recipe 3 GPT-5.5 K = 3 4B LoRA decoder (arm5-9fadc8dc4655) in run 3.

Kev-A: LoRA on every token of the 4B actor plus a decision-token pointer head; Kev-B: prompt tokens through the unadapted
actor. SFT 4B actor, N = 5, T 1.0 / top-p 0.9, 30 steps, all tasks, rolled out judge-deferred. Runs 1-2: Online-Mind2Web
November 8, WebVoyager/DeepShop November 10; run 3 (all four conditions, all benchmarks): November 11-12, 23:43-04:30 PST.
Judges: Online-Mind2Web gpt-5.6-terra, WebVoyager/DeepShop GPT-4o. Every task in plan.json task_ids is in the denominator;
a task with no record (browser failed all attempts) counts as an invalid failure. Paired bootstrap over tasks (10,000
resamples, seed 20261029, benchmarks resampled independently, per-task success averaged over runs). Refuses records still
DEFERRED or with a dead-key judge message. Writes the adjacent JSON.
"""
import glob
import json
from pathlib import Path

import numpy as np

ROOT = "/gpfs/scrubbed/zixianma/openwebrl-runtime/recipe3-ablations-20261104/eval-kev"
BENCHES = ["online_mind2web", "webvoyager", "deepshop"]
SFT, DEC, KA, KB = "sft1", "arm5-9fadc8dc4655", "head5x-9ff81e7c894f", "head5x-18ab53442be0"
LABELS = {SFT: "SFT 4B alone (1 sample)", DEC: "Recipe 3 decoder (GPT-5.5 K = 3, 4B LoRA)",
          KA: "Kev-A (LoRA on all tokens)", KB: "Kev-B (no LoRA on prompt tokens)"}
NB, SEED = 10000, 20261029


def tasks_of(d):
    return sorted(json.load(open(f"{d}/plan.json"))["task_ids"])


def load(d):
    out = {}
    for f in glob.glob(f"{d}/collection/records/*.json"):
        x = json.load(open(f))
        jt = str(x.get("judge_text") or "")
        assert not jt.startswith("DEFERRED") and jt.strip() != "All judge API retry attempts exhausted.", f
        valid = str(x["valid"]) == "True"
        out[x["task_id"]] = (valid, valid and float(x["reward"] or 0) >= 1, x.get("judge_model"))
    return out


def block(runs, conds, rng):
    res, boot, judges = {}, {}, {}
    for b in BENCHES:
        dirs = {c: [f"{ROOT}/{r}/{b}/{c}/{b}" for r in runs] for c in conds}
        tasks = tasks_of(dirs[SFT][0])
        assert all(tasks_of(d) == tasks for c in conds for d in dirs[c]), b
        data = {c: [load(d) for d in dirs[c]] for c in conds}
        judges[b] = sorted({v[2] for c in data for r in data[c] for v in r.values() if v[2]})
        idx = rng.integers(0, len(tasks), (NB, len(tasks)))
        for c in conds:
            a = np.array([[sum(r.get(t, (False, False))[0] for r in data[c]),
                           sum(r.get(t, (False, False))[1] for r in data[c])] for t in tasks], float)
            n_ep = len(runs) * len(tasks)
            res.setdefault(c, {"label": LABELS[c]})[b] = {
                "tasks": len(tasks), "episodes": n_ep, "recorded": sum(len(r) for r in data[c]),
                "valid": int(a[:, 0].sum()), "successes": int(a[:, 1].sum()),
                "overall": a[:, 1].sum() / n_ep, "valid_only": a[:, 1].sum() / a[:, 0].sum(),
                "overall_per_run": [sum(v[1] for v in r.values()) / len(tasks) for r in data[c]]}
            boot[(b, c)] = (a[idx, 1].sum(1) / n_ep, a[idx, 1].sum(1) / a[idx, 0].sum(1))
    for c in conds:
        r = res[c]
        r["mean_overall"] = float(np.mean([r[b]["overall"] for b in BENCHES]))
        r["mean_valid_only"] = float(np.mean([r[b]["valid_only"] for b in BENCHES]))

    def contrast(c, ref):
        out = {}
        for b in BENCHES:
            for k, j in (("overall", 0), ("valid_only", 1)):
                d = 100 * (boot[(b, c)][j] - boot[(b, ref)][j])
                out[f"{b}_{k}_pp"] = [100 * (res[c][b][k] - res[ref][b][k]), *np.percentile(d, [2.5, 97.5])]
        for k, j in (("overall", 0), ("valid_only", 1)):
            d = 100 * np.mean([boot[(b, c)][j] - boot[(b, ref)][j] for b in BENCHES], 0)
            out[f"mean_{k}_pp"] = [100 * (res[c][f"mean_{k}"] - res[ref][f"mean_{k}"]), *np.percentile(d, [2.5, 97.5])]
        return out
    return res, contrast, judges


rng = np.random.default_rng(SEED)
pooled, con3, judges = block(["run1", "run2", "run3"], [SFT, KA, KB], rng)
for c in (KA, KB):
    pooled[c]["minus_sft"] = con3(c, SFT)
pooled_bma = con3(KB, KA)
same, con1, _ = block(["run3"], [SFT, DEC, KA, KB], rng)
for c in (DEC, KA, KB):
    same[c]["minus_sft"] = con1(c, SFT)
for c in (KA, KB):
    same[c]["minus_decoder"] = con1(c, DEC)
out = json.loads(json.dumps({"three_runs": {"runs": 3, "conditions": pooled, "kevB_minus_kevA": pooled_bma},
                             "run3_same_window": {"runs": 1, "conditions": same},
                             "judge_models": judges}, default=float))
Path(__file__).with_suffix(".json").write_text(json.dumps(out, indent=1) + "\n")
for name in ("three_runs", "run3_same_window"):
    print("==", name)
    for c, r in out[name]["conditions"].items():
        print(f"{r['label']:44s}", [f"{r[b]['valid']}/{r[b]['episodes']} {r[b]['successes']} o{100*r[b]['overall']:.1f} v{100*r[b]['valid_only']:.1f}" for b in BENCHES],
              f"mean o{100*r['mean_overall']:.1f} v{100*r['mean_valid_only']:.1f}")
        for k in ("minus_sft", "minus_decoder"):
            if k in r:
                print("   ", k, {kk: [round(x, 1) for x in v] for kk, v in r[k].items()})
print("B - A (3 runs)", {k: [round(x, 1) for x in v] for k, v in out["three_runs"]["kevB_minus_kevA"].items()})
print(out["judge_models"])
