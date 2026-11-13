"""Head model scale, every eval run pooled (November 11-12): 3-benchmark success of the 4.4M, 9.6M, 22.6M, 272.8M and 1.8B heads
(joint transformer, soft CE, Recipe 3 GPT-5.5 H = 4 labels, lr 3e-4, training seed 0) and SFT 4B alone, pooling every run of each
from the three windows that evaluated them: eval-scale (2 runs, Nov 11 morning; 1.8B evening), eval-seed (1 run, Nov 11 evening;
4.4M, 22.6M, SFT) and eval-sizes3 (1 run, Nov 12 night; 9.6M, 22.6M, 272.8M, 1.8B, SFT). Runs: heads 3 each (22.6M 4), SFT 4.
Cross-window pooling: windows differ by up to ~6 pp for SFT, and sizes are not in every window equally. Paired bootstrap over tasks
common to every condition (10,000 resamples, seed 20261029, benchmarks resampled independently, per-task success averaged over a
condition's runs). Refuses DEFERRED and dead-key records. Writes the adjacent JSON."""
import glob, json
from pathlib import Path
import numpy as np
R = "/gpfs/scrubbed/zixianma/openwebrl-runtime/recipe3-ablations-20261104"
BENCH = ["online_mind2web", "webvoyager", "deepshop"]
WIN = ["eval-scale", "eval-seed", "eval-sizes3"]
COND = {"sft1": ("SFT 4B alone (1 sample)", None), "head5-5fbddd8b2031": ("d128-L2", 4.36e6), "head5-fa31796f1fe0": ("d256-L2", 9.58e6),
        "head5-91e7a5a67417": ("d512-L2 (default)", 22.56e6), "head5-938477dffad2": ("d2048-L4", 272.76e6), "head5-7bbad4567d44": ("d4096-L8", 1.77e9)}
DEF, B, SEED = "head5-91e7a5a67417", 10000, 20261029


def runs(c, b):
    out = []
    for r in sorted(p for w in WIN for p in glob.glob(f"{R}/{w}/run*")):
        fs = glob.glob(f"{r}/{b}/{c}/{b}/collection/records/*.json")
        if not fs: continue
        d = {}
        for f in fs:
            x = json.load(open(f)); jt = str(x.get("judge_text") or "")
            assert not jt.startswith("DEFERRED") and jt.strip() != "All judge API retry attempts exhausted.", f
            v = str(x["valid"]) == "True"; d[x["task_id"]] = (v, v and float(x["reward"] or 0) >= 1)
        out.append((r.split("/")[-2] + "/" + r.split("/")[-1], d))
    return out


rng = np.random.default_rng(SEED)
res = {"windows": WIN, "conditions": {c: {"label": l, "params": p} for c, (l, p) in COND.items()}}
boot = {}
for b in BENCH:
    data = {c: runs(c, b) for c in COND}
    tasks = sorted(set.intersection(*[set(d) for c in COND for _, d in data[c]]))
    idx = rng.integers(0, len(tasks), (B, len(tasks)))
    for c in COND:
        rs = [d for _, d in data[c]]
        succ = np.array([np.mean([r[t][1] for r in rs]) for t in tasks]); val = np.array([np.mean([r[t][0] for r in rs]) for t in tasks])
        res["conditions"][c][b] = dict(tasks=len(tasks), runs=[n for n, _ in data[c]], episodes=len(rs) * len(tasks),
                                       valid=int(round(val.sum() * len(rs))), successes=int(round(succ.sum() * len(rs))),
                                       overall=float(succ.mean()), valid_only=float(succ.sum() / val.sum()))
        boot[(b, c)] = succ[idx].mean(1)
for c, r in res["conditions"].items():
    r["mean_overall"] = float(np.mean([r[b]["overall"] for b in BENCH])); r["mean_valid_only"] = float(np.mean([r[b]["valid_only"] for b in BENCH]))
for c in COND:
    for ref, key in (("sft1", "minus_sft_mean_overall_pp"), (DEF, "minus_default_mean_overall_pp")):
        if c == ref or (ref == DEF and c == "sft1"): continue
        d = 100 * np.mean([boot[(b, c)] - boot[(b, ref)] for b in BENCH], 0)
        res["conditions"][c][key] = [100 * (res["conditions"][c]["mean_overall"] - res["conditions"][ref]["mean_overall"]), *map(float, np.percentile(d, [2.5, 97.5]))]
Path(__file__).with_suffix(".json").write_text(json.dumps(res, indent=1) + "\n")
for c, r in res["conditions"].items():
    print(r["label"], len(r["webvoyager"]["runs"]), [f"{r[b]['valid']}/{r[b]['episodes']} {r[b]['successes']} {100*r[b]['valid_only']:.1f}" for b in BENCH],
          f"{100*r['mean_overall']:.1f} {100*r['mean_valid_only']:.1f}", [round(x, 1) for x in r.get("minus_sft_mean_overall_pp", [])], [round(x, 1) for x in r.get("minus_default_mean_overall_pp", [])])
