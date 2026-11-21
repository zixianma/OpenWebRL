"""Matched GPT-5.5 pre vs K=3 4B LoRA comparison (November 20-21). Two SelectionARM LoRAs (rank 32, 2 epochs, OpenWebRL-4B-SFT)
trained on the same 8,802 panels / 2,311 states: targets from (a) GPT-5.5 K=3 lookahead labels relabelled in the pre prompt format
(gpt-5.5, 2,048 output tokens, no effort set, written rationale) and (b) the GPT-5.5 pre-execution labels. Evaluated the same day,
3 runs x all 1,045 tasks, SFT 4B actor, N = 5, T 1.0 / top-p 0.9, judge-deferred then graded. Cross-date references (Nov 14): SFT 4B
alone (eval-pre, after the Nov 17 invalid reruns) and the earlier Recipe 3 GPT-5.5 K=3 decoder trained on all 20,525 panels with the
unmatched-format labels (eval-kevA-1114). Paired bootstrap over tasks common to every condition (10,000 resamples, seed 20261029,
benchmarks resampled independently, per-task success averaged over runs). Refuses DEFERRED and dead-key records.
Usage: g55m_lora_20261120.py [output.json]"""
import glob, json, sys
from pathlib import Path
import numpy as np
R = "/gpfs/scrubbed/zixianma/openwebrl-runtime/recipe3-ablations-20261104"
BENCH = ["online_mind2web", "webvoyager", "deepshop"]
B, SEED = 10000, 20261029
CONDS = {"k3m": ("eval-g55m-1120", "arm5-73e75c310fd9"), "pre": ("eval-g55m-1120", "arm5-9fa0d4b0c0d3"),
         "sft": ("eval-pre", "sft1")}
if "--with-sft" in sys.argv:   # same-day SFT control added 2026-11-20 17:48 (user "can we add a sft control eval runs too now")
    CONDS["sft-sameday"] = ("eval-g55m-1120", "sft1")
if "--with-heads" in sys.argv:   # matched-label heads, trained and evaluated 2026-11-20/21 (user: "train 23 and 4M ADM heads with these data")
    CONDS.update({"k3m-4M": ("eval-g55m-1120", "head5-0b07409ba903"), "k3m-23M": ("eval-g55m-1120", "head5-d7f5169f30e5"),
                  "pre-4M": ("eval-g55m-1120", "head5-2fac14e09dce"), "pre-23M": ("eval-g55m-1120", "head5-a29c62a28980")})
if "--with-old" in sys.argv:   # needs eval-kevA-1114's decoder records graded (still judge-deferred on 2026-11-20)
    CONDS["old-k3-full"] = ("eval-kevA-1114", "arm5-9fadc8dc4655")
PAIRS = [("k3m", "pre"), ("k3m", "sft"), ("pre", "sft")] + ([("k3m-4M", "pre-4M"), ("k3m-23M", "pre-23M"), ("k3m-4M", "sft-sameday"), ("pre-4M", "sft-sameday"),
         ("k3m-23M", "sft-sameday"), ("pre-23M", "sft-sameday"), ("k3m", "k3m-23M"), ("pre", "pre-23M")] if "k3m-4M" in CONDS else []) + ([("k3m", "sft-sameday"), ("pre", "sft-sameday"), ("sft-sameday", "sft")] if "sft-sameday" in CONDS else []) + ([("old-k3-full", "sft"), ("k3m", "old-k3-full"), ("pre", "old-k3-full")] if "old-k3-full" in CONDS else [])


def load(root, slug, b):
    rs = []
    for r in sorted(glob.glob(f"{R}/{root}/run*")):
        d = {}
        for f in glob.glob(f"{r}/{b}/{slug}/{b}/collection/records/*.json"):
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


def stats(a, b_):
    pt, bo, pv, bv = [], [], [], []
    for b in BENCH:
        sa, va, idx = boot[(a, b)]; sb, vb, _ = boot[(b_, b)]
        pt.append(sa.mean() - sb.mean()); bo.append(sa[idx].mean(1) - sb[idx].mean(1))
        pv.append(sa.sum() / va.sum() - sb.sum() / vb.sum()); bv.append(sa[idx].sum(1) / va[idx].sum(1) - sb[idx].sum(1) / vb[idx].sum(1))
    f = lambda p, q: [100 * float(np.mean(p)), *map(float, np.percentile(100 * np.mean(q, 0), [2.5, 97.5]))]
    return dict(mean_overall_pp=f(pt, bo), mean_valid_only_pp=f(pv, bv), per_benchmark_overall_pp={b: 100 * float(x) for b, x in zip(BENCH, pt)},
                per_benchmark_valid_only_pp={b: 100 * float(x) for b, x in zip(BENCH, pv)})


for a, b in PAIRS:
    res["contrasts"][f"{a} - {b}"] = stats(a, b)
args = [a for a in sys.argv[1:] if not a.startswith("--")]
out = Path(args[0] if args else Path(__file__).with_suffix(".json")); out.write_text(json.dumps(res, indent=1) + "\n")
for k, r in res["conditions"].items():
    print(f"{k:12s} runs={r['runs']}", " | ".join(f"{b[:6]} {r[b]['valid']}/{r[b]['episodes']} ov {100*r[b]['overall']:.1f} vo {100*r[b]['valid_only']:.1f}" for b in BENCH),
          f"| mean ov {100*r['mean_overall']:.1f} vo {100*r['mean_valid_only']:.1f}")
for k, v in res["contrasts"].items():
    print(f"{k:24s}", [round(x, 1) for x in v["mean_overall_pp"]], "vo", [round(x, 1) for x in v["mean_valid_only_pp"]],
          {b[:6]: round(x, 1) for b, x in v["per_benchmark_valid_only_pp"].items()})
