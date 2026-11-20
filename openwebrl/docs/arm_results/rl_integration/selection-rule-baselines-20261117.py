"""Rule-selector baselines (November 16-17): random, majority vote (action-distribution) and teacher action-type prior over the
SFT 4B actor's N = 5 proposals, 3 runs each on all 1,045 tasks (T 1.0 / top-p 0.9, judge-deferred, judged afterwards). References
from the November 14-15 eval-pre window (cross-date): SFT 4B alone and the GPT-5.5 lookahead ADM heads (23K probe, 4.4M, 22.6M,
1.8B). Paired bootstrap over tasks common to every condition (10,000 resamples, seed 20261029, benchmarks resampled independently,
per-task success averaged over runs). Refuses DEFERRED and dead-key records.

November 17: every invalid episode of all eight conditions was re-run once (local browser, same protocol) and judged; the
original invalid records were moved to quarantine Q. Default = post-rerun records (selection-rule-baselines-20261117.json);
ORIGINAL=1 reads the quarantined originals instead (selection-rule-baselines-original-20261117.json). BASELINES and BENCHES
restrict conditions/benchmarks for interim reads. Usage: python3 <this> <output stem>."""
import glob, json, sys
from pathlib import Path
import numpy as np
R = "/gpfs/scrubbed/zixianma/openwebrl-runtime/recipe3-ablations-20261104"
import os
BENCH = os.environ.get("BENCHES", "online_mind2web,webvoyager,deepshop").split(",")
B, SEED = 10000, 20261029
CONDS = {"random": ("eval-baselines", "head5-93c301779c9b"), "majority": ("eval-baselines", "head5-5e4432f806ad"),
         "typeprior": ("eval-baselines", "head5-7b86fc5cb182"), "sft": ("eval-pre", "sft1"),
         "adm-23k": ("eval-pre", "head5-ef597d45f3b1"), "adm-4.4m": ("eval-pre", "head5-5fbddd8b2031"),
         "adm-22.6m": ("eval-pre", "head5-91e7a5a67417"), "adm-1.8b": ("eval-pre", "head5-7bbad4567d44")}
Q = "/gpfs/scrubbed/zixianma/openwebrl-runtime/quarantine/invalid-rerun-20261117"
ADM = ["adm-23k", "adm-4.4m", "adm-22.6m", "adm-1.8b"]
import os
RULES = [k for k in ["random", "majority", "typeprior"] if k in os.environ.get("BASELINES", "random,majority,typeprior").split(",")]
CONDS = {k: v for k, v in CONDS.items() if k in RULES or k == "sft" or k in ADM}


def load(root, slug, b):
    rs = []
    for r in sorted(glob.glob(f"{R}/{root}/run*")):
        d = {}
        cur = {Path(f).name: f for f in glob.glob(f"{r}/{b}/{slug}/{b}/collection/records/*.json")}
        if os.environ.get("ORIGINAL") == "1":  # pre-rerun view: quarantined originals replace their rerun records
            q = f"{Q}/{Path(r).relative_to(R)}/{b}/{slug}/{b}/records"
            cur.update({Path(f).name: f for f in glob.glob(f"{q}/*.json")})
        for f in cur.values():
            x = json.load(open(f)); jt = str(x.get("judge_text") or "")
            assert not jt.startswith("DEFERRED") and jt.strip() != "All judge API retry attempts exhausted.", f
            v = str(x["valid"]) == "True"; d[x["task_id"]] = (v, v and float(x["reward"] or 0) >= 1)
        rs.append(d)
    return rs


rng = np.random.default_rng(SEED)
res, boot = {"conditions": {}, "contrasts": {}}, {}
for b in BENCH:
    data = {k: load(*v, b) for k, v in CONDS.items()}
    tasks = sorted(set.intersection(*[set(r) for k in data for r in data[k]]))
    idx = rng.integers(0, len(tasks), (B, len(tasks)))
    for k, rs in data.items():
        s = np.array([np.mean([r[t][1] for r in rs]) for t in tasks]); v = np.array([np.mean([r[t][0] for r in rs]) for t in tasks])
        res["conditions"].setdefault(k, {"source": CONDS[k], "runs": len(rs)})[b] = dict(
            tasks=len(tasks), episodes=len(rs) * len(tasks), valid=int(round(v.sum() * len(rs))), successes=int(round(s.sum() * len(rs))),
            overall=float(s.mean()), valid_only=float(s.sum() / v.sum()))
        boot[(k, b)] = (s, v, idx)
for k, r in res["conditions"].items():
    r["mean_overall"] = float(np.mean([r[b]["overall"] for b in BENCH])); r["mean_valid_only"] = float(np.mean([r[b]["valid_only"] for b in BENCH]))


def stats(arms_a, arms_b):
    pt, bo, pv, bv = [], [], [], []
    for b in BENCH:
        idx = boot[(arms_a[0], b)][2]
        sa = np.mean([boot[(k, b)][0] for k in arms_a], 0); va = np.mean([boot[(k, b)][1] for k in arms_a], 0)
        sb = np.mean([boot[(k, b)][0] for k in arms_b], 0); vb = np.mean([boot[(k, b)][1] for k in arms_b], 0)
        pt.append(sa.mean() - sb.mean()); bo.append(sa[idx].mean(1) - sb[idx].mean(1))
        pv.append(sa.sum() / va.sum() - sb.sum() / vb.sum()); bv.append(sa[idx].sum(1) / va[idx].sum(1) - sb[idx].sum(1) / vb[idx].sum(1))
    per = {b: 100 * float(x) for b, x in zip(BENCH, pt)}
    f = lambda p, q: [100 * float(np.mean(p)), *map(float, np.percentile(100 * np.mean(q, 0), [2.5, 97.5]))]
    return dict(mean_overall_pp=f(pt, bo), mean_valid_only_pp=f(pv, bv), per_benchmark_overall_pp=per)


for k in [*RULES, *ADM]:
    res["contrasts"][f"{k} - sft"] = stats([k], ["sft"])
res["contrasts"]["ADM mean of 4 sizes - sft"] = stats(ADM, ["sft"])
for base in RULES:
    for k in [r for r in RULES if r != base and RULES.index(r) > RULES.index(base)] + ADM + ["ADM"]:
        res["contrasts"][f"{k} - {base}"] = stats(ADM if k == "ADM" else [k], [base])
out = Path(sys.argv[1] if len(sys.argv) > 1 else __file__).with_suffix(".json"); out.write_text(json.dumps(res, indent=1) + "\n")
for k, r in res["conditions"].items():
    print(f"{k:10s} runs={r['runs']}", " | ".join(f"{b[:6]} {r[b]['valid']}/{r[b]['episodes']} ov {100*r[b]['overall']:.1f} vo {100*r[b]['valid_only']:.1f}" for b in BENCH),
          f"| mean ov {100*r['mean_overall']:.1f} vo {100*r['mean_valid_only']:.1f}")
for k, v in res["contrasts"].items():
    print(f"{k:32s}", [round(x, 1) for x in v["mean_overall_pp"]], "vo", [round(x, 1) for x in v["mean_valid_only_pp"]], {b[:6]: round(x, 1) for b, x in v["per_benchmark_overall_pp"].items()})
