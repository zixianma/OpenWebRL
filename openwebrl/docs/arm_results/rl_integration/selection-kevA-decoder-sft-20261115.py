"""Kev-A vs the Recipe 3 GPT-5.5 K=3 4B LoRA decoder vs SFT 4B alone, 3 runs each in one window (November 15-16). Kev-A: kev/kevA-r3
(Recipe 3 GPT-5.5 K=3 labels, LoRA on all tokens; head5x-9ff81e7c894f); decoder: arm5-9fadc8dc4655 (eval-kevA-1114/run1-3); SFT 4B
alone: eval-pre/run1-3 sft1 (another session's batch, same tasks and protocol). SFT 4B actor, N = 5, T 1.0 / top-p 0.9, 30 steps, all
1,045 tasks, judge-deferred. One row per method against SFT 4B alone; paired bootstrap over tasks on run-averaged per-task success
(10,000 resamples, seed 20261029, benchmarks resampled independently). Refuses DEFERRED and dead-key records unless the
judge-sidecar verdict file has them: on Nov 15 /gpfs/scrubbed refused writes, so the Kev-A/decoder episodes were judged by
deferjudge/xbench_defer_sidecar.py with the same native judges and verdicts appended to SIDECAR (not written back into the
records). Writes the adjacent JSON."""
import glob, json
from pathlib import Path
import numpy as np
R = "/gpfs/scrubbed/zixianma/openwebrl-runtime/recipe3-ablations-20261104"
BENCH = ["online_mind2web", "webvoyager", "deepshop"]; NT = {"online_mind2web": 300, "webvoyager": 595, "deepshop": 150}
M = {"sft": ("SFT 4B alone (1 sample)", f"{R}/eval-pre", "sft1"),
     "kevA": ("+ Kev-A (Recipe 3 GPT-5.5 K = 3, LoRA on all tokens)", f"{R}/eval-kevA-1114", "head5x-9ff81e7c894f"),
     "decoder": ("+ 4B LoRA decoder (Recipe 3 GPT-5.5 K = 3)", f"{R}/eval-kevA-1114", "arm5-9fadc8dc4655")}
B, SEED = 10000, 20261029
SIDECAR = Path("/gpfs/projects/krishna/zixianma/judge-sidecar-20261115/verdicts.jsonl")
SIDE = {}
if SIDECAR.exists():
    for l in open(SIDECAR):
        x = json.loads(l); SIDE[(x["plan"], x["task_id"])] = x


def load(root, slug, b):
    runs = []
    for r in sorted(glob.glob(f"{root}/run[123]")):
        d = {}
        for f in glob.glob(f"{r}/{b}/{slug}/{b}/collection/records/*.json"):
            x = json.load(open(f)); jt = str(x.get("judge_text") or "")
            if jt.startswith("DEFERRED") or jt.strip() == "All judge API retry attempts exhausted.":
                sv = SIDE.get((f"{r}/{b}/{slug}/{b}", x["task_id"]))
                assert sv is not None, f
                v = bool(sv["valid"]); d[x["task_id"]] = (v, v and float(sv["reward"] or 0) >= 1); continue
            v = str(x["valid"]) == "True"; d[x["task_id"]] = (v, v and float(x["reward"] or 0) >= 1)
        assert len(d) == NT[b], (r, b, slug, len(d))
        runs.append(d)
    return runs


data = {(k, b): load(v[1], v[2], b) for k, v in M.items() for b in BENCH}
res = {}
for k, (label, _, _) in M.items():
    r = dict(label=label, runs=len(data[(k, BENCH[0])]))
    for b in BENCH:
        rs = data[(k, b)]; n = sum(len(x) for x in rs); v = sum(t[0] for x in rs for t in x.values()); s = sum(t[1] for x in rs for t in x.values())
        r[b] = dict(episodes=n, valid=v, successes=s, overall=s / n, valid_only=s / v)
    r["mean_overall"] = float(np.mean([r[b]["overall"] for b in BENCH])); r["mean_valid_only"] = float(np.mean([r[b]["valid_only"] for b in BENCH]))
    for ref in ("sft", "decoder"):
        if k == ref or (ref == "decoder" and k != "kevA"): continue
        rng = np.random.default_rng(SEED); bo, bv, po, pv = [], [], [], []; per = {}
        for b in BENCH:
            A, C = data[(k, b)], data[(ref, b)]; tasks = sorted(set.intersection(*[set(x) for x in A + C]))
            f = lambda rs, j: np.array([np.mean([x[t][j] for x in rs]) for t in tasks])
            av, as_, cv, cs = f(A, 0), f(A, 1), f(C, 0), f(C, 1); idx = rng.integers(0, len(tasks), (B, len(tasks)))
            d_o = as_[idx].mean(1) - cs[idx].mean(1); bo.append(d_o); bv.append(as_[idx].sum(1) / av[idx].sum(1) - cs[idx].sum(1) / cv[idx].sum(1))
            po.append(as_.mean() - cs.mean()); pv.append(as_.sum() / av.sum() - cs.sum() / cv.sum())
            per[f"{b}_overall_pp"] = [100 * po[-1], *np.percentile(100 * d_o, [2.5, 97.5])]
        r[f"minus_{ref}"] = dict(per, mean_overall_pp=[100 * float(np.mean(po)), *np.percentile(100 * np.mean(bo, 0), [2.5, 97.5])],
                                 mean_valid_only_pp=[100 * float(np.mean(pv)), *np.percentile(100 * np.mean(bv, 0), [2.5, 97.5])])
    res[k] = r
    print(label, r["runs"], " ".join(f"{b[:4]} {100*r[b]['overall']:.1f}" for b in BENCH), f"| ov {100*r['mean_overall']:.1f} vo {100*r['mean_valid_only']:.1f}",
          {x: [round(y, 1) for y in r[x]["mean_overall_pp"]] for x in ("minus_sft", "minus_decoder") if x in r})
Path(__file__).with_suffix(".json").write_text(json.dumps(json.loads(json.dumps(res, default=float)), indent=1) + "\n")
