"""RL 8B actor (OpenWebRL/OpenWebRL-8B @ 824a54f, 4,096 response tokens) + Recipe 3 GPT-5.5 H = 4 heads on local browsers
(November 13-14): 22.6M default, 4.4M (d128-L2), 272.8M (d2048-L4) and the 23K linear probe, training seed 0, 2 runs each on
all 1,045 tasks. The heads read OpenWebRL-4B-SFT features: each step the frozen SFT-4B encoder re-encodes the RL 8B prompt and
its five candidates (head5x). No RL 8B-alone run in this window: the references are RL 8B alone and RL 8B + the GPT-5.5 H = 4
4B LoRA from November 4-5 (evaluations/sameday-rl8b-20261104, 3 runs; cross-window). Paired bootstrap over tasks common to
both arms (10,000 resamples, seed 20261029, benchmarks resampled independently, per-task success averaged over runs).
Refuses DEFERRED and dead-key records. Writes the adjacent JSON."""
import glob, json
from pathlib import Path
import numpy as np
H = "/gpfs/scrubbed/zixianma/openwebrl-runtime/recipe3-ablations-20261104/eval-rl8b-heads"
L = "/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations/sameday-rl8b-20261104"
BENCH = ["online_mind2web", "webvoyager", "deepshop"]
COND = {"head-22.6M": ("RL 8B + 22.6M head (default)", H, "head5x-91e7a5a67417"), "head-4.4M": ("RL 8B + 4.4M head", H, "head5x-5fbddd8b2031"),
        "head-272.8M": ("RL 8B + 272.8M head", H, "head5x-938477dffad2"), "probe-23K": ("RL 8B + 23K linear probe", H, "head5x-ef597d45f3b1"),
        "alone-nov4": ("RL 8B alone (1 sample; Nov 4-5)", f"{L}/rl8b", "sft1"),
        "lora-nov4": ("RL 8B + 4B LoRA, GPT-5.5 H = 4 (Nov 4-5)", f"{L}/rl8b-lora", "arm5-9fadc8dc4655")}
B, SEED = 10000, 20261029


def load(root, slug, b):
    runs = []
    for r in sorted(glob.glob(f"{root}/run*")):
        d = {}
        for f in glob.glob(f"{r}/{b}/{slug}/{b}/collection/records/*.json"):
            x = json.load(open(f)); jt = str(x.get("judge_text") or "")
            assert not jt.startswith("DEFERRED") and jt.strip() != "All judge API retry attempts exhausted.", f
            v = str(x["valid"]) == "True"; d[x["task_id"]] = (v, v and float(x["reward"] or 0) >= 1)
        if d: runs.append(d)
    return runs


data = {(c, b): load(root, slug, b) for c, (_, root, slug) in COND.items() for b in BENCH}
res = {"conditions": {}, "contrasts": {}}
for c, (label, _, _) in COND.items():
    r = {"label": label, "runs": len(data[(c, BENCH[0])])}
    for b in BENCH:
        rs = data[(c, b)]; tasks = set.intersection(*[set(x) for x in rs])
        v = sum(x[t][0] for x in rs for t in tasks); s = sum(x[t][1] for x in rs for t in tasks); n = len(rs) * len(tasks)
        r[b] = dict(episodes=n, valid=v, successes=s, overall=s / n, valid_only=s / v)
    r["mean_overall"] = float(np.mean([r[b]["overall"] for b in BENCH])); r["mean_valid_only"] = float(np.mean([r[b]["valid_only"] for b in BENCH]))
    res["conditions"][c] = r


def contrast(a, ref):
    rng = np.random.default_rng(SEED); out = {}; bo, bv, po, pv = [], [], [], []
    for b in BENCH:
        A, R = data[(a, b)], data[(ref, b)]; tasks = sorted(set.intersection(*[set(x) for x in A + R]))
        f = lambda rs, k: np.array([np.mean([x[t][k] for x in rs]) for t in tasks])
        av, as_, rv, rs_ = f(A, 0), f(A, 1), f(R, 0), f(R, 1); idx = rng.integers(0, len(tasks), (B, len(tasks)))
        d_ov = as_[idx].mean(1) - rs_[idx].mean(1); d_vo = as_[idx].sum(1) / av[idx].sum(1) - rs_[idx].sum(1) / rv[idx].sum(1)
        bo.append(d_ov); bv.append(d_vo); po.append(as_.mean() - rs_.mean()); pv.append(as_.sum() / av.sum() - rs_.sum() / rv.sum())
        out[f"{b}_valid_only_pp"] = [100 * (as_.sum() / av.sum() - rs_.sum() / rv.sum()), *np.percentile(100 * d_vo, [2.5, 97.5])]
    for k, bs, ps in (("mean_overall_pp", bo, po), ("mean_valid_only_pp", bv, pv)):
        m = 100 * np.mean(bs, 0); out[k] = [100 * float(np.mean(ps)), *np.percentile(m, [2.5, 97.5])]
    return out


for c in ("head-4.4M", "head-272.8M", "probe-23K"):
    res["contrasts"][f"{c} - head-22.6M"] = contrast(c, "head-22.6M")
for c in ("head-22.6M", "head-4.4M", "head-272.8M", "probe-23K"):
    res["contrasts"][f"{c} - alone-nov4 (cross-window)"] = contrast(c, "alone-nov4")
    res["contrasts"][f"{c} - lora-nov4 (cross-window)"] = contrast(c, "lora-nov4")
out = json.loads(json.dumps(res, default=float)); Path(__file__).with_suffix(".json").write_text(json.dumps(out, indent=1) + "\n")
for c, r in out["conditions"].items():
    print(f"{r['label']:45s} runs {r['runs']}", " ".join(f"{b[:4]} {100*r[b]['valid_only']:.1f}" for b in BENCH), f"| ov {100*r['mean_overall']:.1f} vo {100*r['mean_valid_only']:.1f}")
for k, v in out["contrasts"].items():
    print(f"{k:50s}", {kk: [round(x, 1) for x in vv] for kk, vv in v.items() if kk.startswith("mean")})
