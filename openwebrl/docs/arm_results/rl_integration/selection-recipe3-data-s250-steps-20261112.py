"""250-state x 1-panel LoRA step controls (November 12, 00:00-05:00 PST): the 252-panel / 495-example subset trained 4 epochs
(~124 steps) and 16 epochs (~495 steps), 3 runs each, with the full-data LoRA and SFT 4B alone re-run in the same window (1 run
each; eval-s250ep). The 2-epoch LoRA on the same data (~62 steps; eval-1111, 3 runs, Nov 11) is in a second group (cross-window).
Paired bootstrap over tasks (10,000 resamples, seed 20261029, benchmarks resampled independently, per-task success averaged over
a condition's runs). Refuses records still DEFERRED or with a dead-key judge message. Writes the adjacent JSON.
"""
import glob
import json
from pathlib import Path

import numpy as np

R = "/gpfs/scrubbed/zixianma/openwebrl-runtime/recipe3-ablations-20261104"
BENCHES = ["online_mind2web", "webvoyager", "deepshop"]
GROUPS = {
    "same_window": {"ref": "full", "conds": {
        "s250-pps1, 16 epochs": ("eval-s250ep", "arm5-88632a7097e6"), "s250-pps1, 4 epochs": ("eval-s250ep", "arm5-9c955fbdf7eb"),
        "full": ("eval-s250ep", "arm5-9fadc8dc4655"), "SFT 4B alone": ("eval-s250ep", "sft1")}},
    "vs_2_epochs": {"ref": "s250-pps1, 2 epochs", "conds": {
        "s250-pps1, 16 epochs": ("eval-s250ep", "arm5-88632a7097e6"), "s250-pps1, 4 epochs": ("eval-s250ep", "arm5-9c955fbdf7eb"),
        "s250-pps1, 2 epochs": ("eval-1111", "arm5-c5ec9b4862f6")}},
}
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


rng = np.random.default_rng(SEED)
res = {}
for g, spec in GROUPS.items():
    arr, idx, rows = {}, {}, {}
    for b in BENCHES:
        runs = {n: [load(d) for d in sorted(glob.glob(f"{R}/{root}/run*/{b}/{c}/{b}"))] for n, (root, c) in spec["conds"].items()}
        assert all(rf == 0 for n in runs for _, rf in runs[n]), f"unjudged records in {g} {b}"
        tasks = sorted(set.intersection(*[set(r) for n in runs for r, _ in runs[n]]))
        idx[b] = rng.integers(0, len(tasks), (B, len(tasks)))
        for n, rs in runs.items():
            rs = [r for r, _ in rs]
            arr[(b, n)] = np.array([[sum(r[t][0] for r in rs), sum(r[t][1] for r in rs), len(rs)] for t in tasks], float)
    rt = lambda a: (a[:, 1].sum() / a[:, 2].sum(), a[:, 1].sum() / a[:, 0].sum())
    bt = lambda a, i: (a[i, 1].sum(1) / a[i, 2].sum(1), a[i, 1].sum(1) / a[i, 0].sum(1))
    for n in spec["conds"]:
        row = {b: dict(zip(("valid", "episodes", "successes"), map(int, (arr[(b, n)][:, 0].sum(), arr[(b, n)][:, 2].sum(), arr[(b, n)][:, 1].sum()))),
                       overall=rt(arr[(b, n)])[0], valid_only=rt(arr[(b, n)])[1]) for b in BENCHES}
        row["runs"] = int(arr[(BENCHES[0], n)][0, 2])
        row["mean_overall"] = float(np.mean([row[b]["overall"] for b in BENCHES])); row["mean_valid_only"] = float(np.mean([row[b]["valid_only"] for b in BENCHES]))
        for m in spec["conds"]:
            if m == n:
                continue
            for j, k in ((0, "overall"), (1, "valid_only")):
                d = 100 * np.mean([bt(arr[(b, n)], idx[b])[j] - bt(arr[(b, m)], idx[b])[j] for b in BENCHES], 0)
                pt = 100 * np.mean([rt(arr[(b, n)])[j] - rt(arr[(b, m)])[j] for b in BENCHES])
                row[f"minus {m}: mean_{k}_pp"] = [pt, *np.percentile(d, [2.5, 97.5])]
        rows[n] = row
    res[g] = dict(reference=spec["ref"], tasks={b: int(len(idx[b][0])) for b in BENCHES}, conditions=rows)
out = json.loads(json.dumps(res, default=float))
Path(__file__).with_suffix(".json").write_text(json.dumps(out, indent=1) + "\n")
for g, r in out.items():
    print("==", g)
    for n, row in r["conditions"].items():
        print(f"{n:28s} runs {row['runs']}", [f"{100 * row[b]['valid_only']:.1f}" for b in BENCHES], f"{100 * row['mean_overall']:.1f}",
              {k.split(':')[0]: [round(x, 1) for x in v] for k, v in row.items() if k.endswith('mean_overall_pp')})
