"""Recipe 3 GPT-5.5 H = 4 4B decoder, same-window controls (November 9): step-matched LoRA and full fine-tune vs the full LoRA.

All conditions: SFT 4B actor, N = 5, T 1.0 / top-p 0.9, 30 steps, all 1,045 tasks, 3 runs, judges as in the Recipe 3 evals
(Online-Mind2Web gpt-5.6-terra, WebVoyager and DeepShop GPT-4o), rolled out judge-deferred and judged afterwards.
Paired bootstrap over tasks common to every run and condition (10,000 resamples, seed 20261029, benchmarks resampled
independently, per-task success averaged over runs). Refuses records still DEFERRED or with a dead-key judge message.
Writes the adjacent JSON.
"""
import glob
import json
from pathlib import Path

import numpy as np

ROOT = "/gpfs/scrubbed/zixianma/openwebrl-runtime/recipe3-ablations-20261104/eval-1108"
REF = "arm5-9fadc8dc4655"
CONDITIONS = {
    REF: "Full data, LoRA (4,122 steps; reference)",
    "arm5-5f368697014f": "Full data, LoRA, step-matched (1,692 steps)",
    "arm5-31cb7266dae5": "Full data, full fine-tune (4,122 steps)",
}
TRAINING = {
    REF: "rl-actor-recollect-20261016/p16-k3-20261031/lora/lora (rank 32, lr 1e-4, 2 epochs, batch 16)",
    "arm5-5f368697014f": "recipe3-ablations-20261104/lora/full-s1692 (identical LoRA, max_steps 1692 = 4-panels-per-state subset's steps, ~0.82 epoch)",
    "arm5-31cb7266dae5": "recipe3-ablations-20261104/lora/ft-full (all language-model weights; vision tower and projector frozen; lr 1e-5, 2 epochs, batch 16)",
}
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
rng = np.random.default_rng(SEED)
res = {"conditions": {}, "runs": len(runs), "reference": REF}
boot = {}
for b in BENCHES:
    data = {c: [load(f"{r}/{b}/{c}/{b}") for r in runs] for c in CONDITIONS}
    assert all(rf == 0 for c in data for _, rf in data[c]), f"unjudged records in {b}"
    tasks = sorted(set.intersection(*[set(d) for c in data for d, _ in data[c]]))
    idx = rng.integers(0, len(tasks), (B, len(tasks)))
    for c in CONDITIONS:
        rs = [d for d, _ in data[c]]
        a = np.array([[sum(r[t][0] for r in rs), sum(r[t][1] for r in rs)] for t in tasks], float)
        n_ep = len(rs) * len(tasks)
        res["conditions"].setdefault(c, {"label": CONDITIONS[c], "training": TRAINING[c]})[b] = {
            "tasks": len(tasks), "episodes": n_ep, "valid": int(a[:, 0].sum()), "successes": int(a[:, 1].sum()),
            "overall": a[:, 1].sum() / n_ep, "valid_only": a[:, 1].sum() / a[:, 0].sum()}
        boot[(b, c)] = (a[idx, 1].sum(1) / n_ep, a[idx, 1].sum(1) / a[idx, 0].sum(1))
for c in CONDITIONS:
    r = res["conditions"][c]
    r["mean_overall"] = float(np.mean([r[b]["overall"] for b in BENCHES]))
    r["mean_valid_only"] = float(np.mean([r[b]["valid_only"] for b in BENCHES]))
    if c == REF:
        continue
    for b in BENCHES:
        d = 100 * (boot[(b, c)][1] - boot[(b, REF)][1])
        r[b]["diff_valid_only_pp"] = [100 * (r[b]["valid_only"] - res["conditions"][REF][b]["valid_only"]), *np.percentile(d, [2.5, 97.5])]
    for k, j in (("overall", 0), ("valid_only", 1)):
        d = 100 * np.mean([boot[(b, c)][j] - boot[(b, REF)][j] for b in BENCHES], 0)
        r[f"diff_mean_{k}_pp"] = [100 * (r[f"mean_{k}"] - res["conditions"][REF][f"mean_{k}"]), *np.percentile(d, [2.5, 97.5])]
out = json.loads(json.dumps(res, default=float))
Path(__file__).with_suffix(".json").write_text(json.dumps(out, indent=1) + "\n")
for c, r in out["conditions"].items():
    print(r["label"], [f"{r[b]['valid']}/{r[b]['episodes']} {r[b]['successes']} {100*r[b]['valid_only']:.1f}" for b in BENCHES],
          f"{100*r['mean_overall']:.1f} {100*r['mean_valid_only']:.1f}",
          {k: [round(x, 1) for x in v] for k, v in r.items() if k.startswith("diff")},
          {b: [round(x, 1) for x in r[b]["diff_valid_only_pp"]] for b in BENCHES if "diff_valid_only_pp" in r[b]})
