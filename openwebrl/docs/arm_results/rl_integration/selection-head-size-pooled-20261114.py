"""Head-size summary, one row per method: every run of each method pooled across its evaluation windows, against the actor
alone as control. SFT 4B actor: all runs November 11-13 (eval-scale, eval-seed, eval-sizes3, eval-small, eval-probe; SFT 4B
alone and the 4B LoRA also from eval-1111). RL 8B actor: heads November 13-14 (eval-rl8b-heads, frozen SFT-4B re-encoding),
RL 8B alone and + 4B LoRA November 4-5 (evaluations/sameday-rl8b-20261104). Heads: Recipe 3 GPT-5.5 H = 4, training seed 0.
Per-task success is averaged over a method's runs; paired bootstrap over tasks common to both arms (10,000 resamples, seed
20261029, benchmarks resampled independently). Pooling spans windows, so differences mix window drift with method effects.
Refuses DEFERRED and dead-key records. Writes the adjacent JSON."""
import glob, json
from pathlib import Path
import numpy as np
R = "/gpfs/scrubbed/zixianma/openwebrl-runtime/recipe3-ablations-20261104"
L = "/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations/sameday-rl8b-20261104"
BENCH = ["online_mind2web", "webvoyager", "deepshop"]
H = lambda *e: [f"{R}/{x}" for x in e]
SFT = {"sft": ("SFT 4B alone (1 sample)", "—", H("eval-scale", "eval-seed", "eval-sizes3", "eval-1111"), "sft1"),
       "probe": ("+ linear probe", "23K", H("eval-probe"), "head5-ef597d45f3b1"),
       "d8": ("+ d8-L1 head", "0.25M", H("eval-small"), "head5-f08c932b4269"),
       "d32": ("+ d32-L1 head", "1.0M", H("eval-small"), "head5-8b28e54a228f"),
       "d128": ("+ d128-L2 head", "4.4M", H("eval-scale", "eval-seed"), "head5-5fbddd8b2031"),
       "d256": ("+ d256-L2 head", "9.6M", H("eval-scale", "eval-sizes3"), "head5-fa31796f1fe0"),
       "d512": ("+ d512-L2 head (default ADM-22M)", "22.6M", H("eval-scale", "eval-seed", "eval-sizes3", "eval-small", "eval-probe"), "head5-91e7a5a67417"),
       "d2048": ("+ d2048-L4 head", "272.8M", H("eval-scale", "eval-sizes3"), "head5-938477dffad2"),
       "d4096": ("+ d4096-L8 head", "1.8B", H("eval-scale", "eval-sizes3"), "head5-7bbad4567d44"),
       "lora": ("+ 4B LoRA decoder (ADM-4B)", "4.4B", H("eval-1111"), "arm5-9fadc8dc4655")}
RL = {"rl": ("RL 8B alone (1 sample)", "—", [f"{L}/rl8b"], "sft1"),
      "probe": ("+ linear probe", "23K", H("eval-rl8b-heads"), "head5x-ef597d45f3b1"),
      "d128": ("+ d128-L2 head", "4.4M", H("eval-rl8b-heads"), "head5x-5fbddd8b2031"),
      "d512": ("+ d512-L2 head (default ADM-22M)", "22.6M", H("eval-rl8b-heads"), "head5x-91e7a5a67417"),
      "d2048": ("+ d2048-L4 head", "272.8M", H("eval-rl8b-heads"), "head5x-938477dffad2"),
      "lora": ("+ 4B LoRA decoder (GPT-5.5 H = 4)", "4.4B", [f"{L}/rl8b-lora"], "arm5-9fadc8dc4655")}
B, SEED = 10000, 20261029


def load(roots, slug, b):
    runs = []
    for root in roots:
        for r in sorted(glob.glob(f"{root}/run*")):
            d = {}
            for f in glob.glob(f"{r}/{b}/{slug}/{b}/collection/records/*.json"):
                x = json.load(open(f)); jt = str(x.get("judge_text") or "")
                assert not jt.startswith("DEFERRED") and jt.strip() != "All judge API retry attempts exhausted.", f
                v = str(x["valid"]) == "True"; d[x["task_id"]] = (v, v and float(x["reward"] or 0) >= 1)
            if len(d) == {"online_mind2web": 300, "webvoyager": 595, "deepshop": 150}[b]: runs.append(d)
    return runs


def block(spec, ctrl):
    data = {(k, b): load(v[2], v[3], b) for k, v in spec.items() for b in BENCH}
    out = {}
    for k, (label, params, roots, slug) in spec.items():
        r = dict(label=label, params=params, runs=len(data[(k, BENCH[0])]), sources=[Path(x).name for x in roots])
        for b in BENCH:
            rs = data[(k, b)]; n = sum(len(x) for x in rs); v = sum(t[0] for x in rs for t in x.values()); s = sum(t[1] for x in rs for t in x.values())
            r[b] = dict(episodes=n, valid=v, successes=s, overall=s / n, valid_only=s / v)
        r["mean_overall"] = float(np.mean([r[b]["overall"] for b in BENCH])); r["mean_valid_only"] = float(np.mean([r[b]["valid_only"] for b in BENCH]))
        if k != ctrl:
            rng = np.random.default_rng(SEED); bo, bv, po, pv = [], [], [], []
            for b in BENCH:
                A, C = data[(k, b)], data[(ctrl, b)]; tasks = sorted(set.intersection(*[set(x) for x in A + C]))
                f = lambda rs, j: np.array([np.mean([x[t][j] for x in rs]) for t in tasks])
                av, as_, cv, cs = f(A, 0), f(A, 1), f(C, 0), f(C, 1); idx = rng.integers(0, len(tasks), (B, len(tasks)))
                bo.append(as_[idx].mean(1) - cs[idx].mean(1)); bv.append(as_[idx].sum(1) / av[idx].sum(1) - cs[idx].sum(1) / cv[idx].sum(1))
                po.append(as_.mean() - cs.mean()); pv.append(as_.sum() / av.sum() - cs.sum() / cv.sum())
            r["minus_control_mean_overall_pp"] = [100 * float(np.mean(po)), *np.percentile(100 * np.mean(bo, 0), [2.5, 97.5])]
            r["minus_control_mean_valid_only_pp"] = [100 * float(np.mean(pv)), *np.percentile(100 * np.mean(bv, 0), [2.5, 97.5])]
        out[k] = r; print(label, r["runs"], f"{100*r['mean_overall']:.1f} {100*r['mean_valid_only']:.1f}",
                          [round(x, 1) for x in r.get("minus_control_mean_overall_pp", [])], [round(x, 1) for x in r.get("minus_control_mean_valid_only_pp", [])], flush=True)
    return out


res = {"sft4b": block(SFT, "sft"), "rl8b": block(RL, "rl")}
Path(__file__).with_suffix(".json").write_text(json.dumps(json.loads(json.dumps(res, default=float)), indent=1) + "\n")
