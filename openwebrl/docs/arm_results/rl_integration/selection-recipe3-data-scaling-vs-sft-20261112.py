"""Recipe 3 GPT-5.5 H = 4 4B LoRA data scaling against SFT 4B alone: 3-benchmark mean overall success of every data-scaling LoRA
(3 runs each) minus SFT 4B alone (1 sample, 3 runs). Same-window SFT where the window had one (Nov 11 eval-1111, Nov 11
evening eval-1111b vs the Nov 11 eval-1111 SFT, Nov 12 eval-s250ep); the Nov 8-9 (eval-1108) and Nov 10 (eval-1110) windows ran
no SFT, so their points use the Nov 11 eval-1111 SFT (cross-window, flagged). Full-data LoRA in each SFT window as the reference.
Paired bootstrap over tasks common to both arms (10,000 resamples, seed 20261029, benchmarks resampled independently, per-task
success averaged over runs). Refuses DEFERRED and dead-key records. Writes the adjacent JSON."""
import glob, json
from pathlib import Path
import numpy as np
R = "/gpfs/scrubbed/zixianma/openwebrl-runtime/recipe3-ablations-20261104"
BENCH = ["online_mind2web", "webvoyager", "deepshop"]
B, SEED = 10000, 20261029
SFT11, SFT12 = ("eval-1111", "sft1"), ("eval-s250ep", "sft1")
# key: (label, panels, optimizer steps, one panel per state, arm, SFT reference, same window)
POINTS = {
    "pps4": ("4 / state", 7417, 1700, False, ("eval-1108", "arm5-9117fa5a37bb"), SFT11, False),
    "s1000": ("1,000 states", 8875, 2000, False, ("eval-1108", "arm5-fc760b23e227"), SFT11, False),
    "pps1": ("1 / state", 2112, 489, True, ("eval-1108", "arm5-9f25a8bd0da7"), SFT11, False),
    "s250": ("250 states", 2234, 482, False, ("eval-1110", "arm5-d6c492ad4c40"), SFT11, False),
    "s1000-pps1": ("1K × 1, 2 ep", 1001, 238, True, ("eval-1111", "arm5-5d57cf0e0de4"), SFT11, True),
    "s1000-pps1-e4": ("1K × 1, 4 ep", 1001, 476, True, ("eval-1111b", "arm5-a82f7bbded0c"), SFT11, True),
    "s250-pps1": ("250 × 1, 2 ep", 252, 62, True, ("eval-1111", "arm5-c5ec9b4862f6"), SFT11, True),
    "s250-pps1-e4": ("250 × 1, 4 ep", 252, 124, True, ("eval-s250ep", "arm5-9c955fbdf7eb"), SFT12, True),
    "s250-pps1-e16": ("250 × 1, 16 ep", 252, 495, True, ("eval-s250ep", "arm5-88632a7097e6"), SFT12, True),
    "full-1111": ("full data (Nov 11)", 18786, 4122, False, ("eval-1111", "arm5-9fadc8dc4655"), SFT11, True),
    "full-1112": ("full data (Nov 12)", 18786, 4122, False, ("eval-s250ep", "arm5-9fadc8dc4655"), SFT12, True),
}


def load(e, slug, b):
    out = {}
    for run in sorted(glob.glob(f"{R}/{e}/run*")):
        d = {}
        for f in glob.glob(f"{run}/{b}/{slug}/{b}/collection/records/*.json"):
            x = json.load(open(f)); jt = str(x.get("judge_text") or "")
            assert not jt.startswith("DEFERRED") and jt.strip() != "All judge API retry attempts exhausted.", f
            d[x["task_id"]] = str(x["valid"]) == "True" and float(x["reward"] or 0) >= 1
        if d: out[run] = d
    return list(out.values())


cache, res = {}, {}
rng = np.random.default_rng(SEED)
get = lambda k, b: cache.setdefault((k, b), load(*k, b))
for key, (label, panels, steps, one, arm, ref, same) in POINTS.items():
    pts, boots, rates = [], [], {}
    for b in BENCH:
        A, S = get(arm, b), get(ref, b)
        tasks = sorted(set.intersection(*[set(r) for r in A + S]))
        a = np.array([np.mean([r[t] for r in A]) for t in tasks]); s = np.array([np.mean([r[t] for r in S]) for t in tasks])
        idx = rng.integers(0, len(tasks), (B, len(tasks)))
        pts.append(a.mean() - s.mean()); boots.append(a[idx].mean(1) - s[idx].mean(1))
        rates[b] = dict(tasks=len(tasks), arm_runs=len(A), sft_runs=len(S), arm_overall=a.mean(), sft_overall=s.mean())
    d = 100 * np.mean(boots, 0)
    res[key] = dict(label=label, panels=panels, steps=steps, one_panel_per_state=one, arm=list(arm), sft=list(ref), same_window=same,
                    per_benchmark=rates, minus_sft_mean_overall_pp=[100 * float(np.mean(pts)), *map(float, np.percentile(d, [2.5, 97.5]))])
    print(label, [round(x, 1) for x in res[key]["minus_sft_mean_overall_pp"]], "same" if same else "cross", flush=True)
Path(__file__).with_suffix(".json").write_text(json.dumps(res, indent=1, default=float) + "\n")
