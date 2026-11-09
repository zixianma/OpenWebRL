"""SFT 4B vs SFT 4B + Recipe 3 GPT-5.5 K=3 decoder on Browser Use stealth browsers (US residential proxy), November 8-9.

One run per condition on all tasks (rolled out November 9 in two windows; credits ran out in between and the same plans
resumed). Paired comparisons use each benchmark's tasks finished by both conditions (all of them at full coverage). Also compares with local-browser runs on the same tasks: stealth - local per condition, and the selector gain
(decoder - SFT) local vs stealth. Local SFT references are from other days (Nov 6 / Nov 8); the local decoder ran Nov 9.
Refuses records whose judge_text is still DEFERRED or a dead-key retry message. Writes the adjacent JSON.
"""
import glob
import json
from pathlib import Path

import numpy as np

RT = "/gpfs/scrubbed/zixianma/openwebrl-runtime/recipe3-ablations-20261104"
BENCHES = ["online_mind2web", "webvoyager", "deepshop"]
DEC = "arm5-9fadc8dc4655"
STEALTH = {c: [f"{RT}/stealth/run1/{{b}}/{c}/{{b}}"] for c in ("sft1", DEC)}
LOCAL = {  # local-browser references on the same tasks (different days)
    "sft1": {"online_mind2web": [f"{RT}/eval-kev/run{r}/{{b}}/sft1/{{b}}" for r in (1, 2)],  # Nov 8
             "webvoyager": [f"{RT}/eval-ctrl/run{r}/{{b}}/sft1/{{b}}" for r in (1, 2, 3)],  # Nov 6
             "deepshop": [f"{RT}/eval-ctrl/run{r}/{{b}}/sft1/{{b}}" for r in (1, 2, 3)]},
    DEC: {b: [f"{RT}/eval-1108/run{r}/{{b}}/{DEC}/{{b}}" for r in (1, 2, 3)] for b in BENCHES},  # Nov 9
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


def rates(runs, tasks):
    a = np.array([[sum(r[t][0] for r in runs), sum(r[t][1] for r in runs)] for t in tasks], float)
    return a, a[:, 1].sum() / (len(runs) * len(tasks)), a[:, 1].sum() / a[:, 0].sum()


rng = np.random.default_rng(SEED)
res, boot = {"benchmarks": {}}, {}
for b in BENCHES:
    st = {c: load(STEALTH[c][0].format(b=b)) for c in STEALTH}
    assert all(r == 0 for _, r in st.values()), "unjudged stealth records"
    tasks = sorted(set(st["sft1"][0]) & set(st[DEC][0]))
    idx = rng.integers(0, len(tasks), (B, len(tasks)))
    row = {"common_tasks": len(tasks), "collected": {c: len(st[c][0]) for c in st}}
    for c in st:
        a, ov, vo = rates([st[c][0]], tasks)
        row[c] = {"valid": int(a[:, 0].sum()), "successes": int(a[:, 1].sum()), "overall": ov, "valid_only": vo}
        boot[(b, c)] = (a[idx, 1].sum(1) / len(tasks), a[idx, 1].sum(1) / a[idx, 0].sum(1))
        loc = [load(p.format(b=b))[0] for p in LOCAL[c][b]]
        lt = [t for t in tasks if all(t in r for r in loc)]
        _, lov, lvo = rates(loc, lt)
        row[c]["local_same_tasks"] = {"runs": len(loc), "tasks": len(lt), "overall": lov, "valid_only": lvo,
                                      "dirs": [p.format(b=b) for p in LOCAL[c][b]]}
    for k, j in (("overall", 0), ("valid_only", 1)):
        d = 100 * (boot[(b, DEC)][j] - boot[(b, "sft1")][j])
        row[f"diff_{k}_pp"] = [100 * (row[DEC][k] - row["sft1"][k]), *np.percentile(d, [2.5, 97.5])]
    res["benchmarks"][b] = row
for k, j in (("overall", 0), ("valid_only", 1)):
    m = {c: float(np.mean([res["benchmarks"][b][c][k] for b in BENCHES])) for c in ("sft1", DEC)}
    d = 100 * np.mean([boot[(b, DEC)][j] - boot[(b, "sft1")][j] for b in BENCHES], 0)
    res[f"mean_{k}"] = {**m, "diff_pp": [100 * (m[DEC] - m["sft1"]), *np.percentile(d, [2.5, 97.5])]}
# Stealth - local, each condition on all tasks it finished under stealth (paired over tasks; local runs pooled).
res["stealth_minus_local"], sl = {}, {}
for c in STEALTH:
    for b in BENCHES:
        s_runs = [load(STEALTH[c][0].format(b=b))[0]]
        l_runs = [load(p.format(b=b))[0] for p in LOCAL[c][b]]
        tasks = sorted(set(s_runs[0]).intersection(*l_runs))
        idx = rng.integers(0, len(tasks), (B, len(tasks)))
        (sa, sov, svo), (la, lov, lvo) = rates(s_runs, tasks), rates(l_runs, tasks)
        bs = (sa[idx, 1].sum(1) / len(tasks), sa[idx, 1].sum(1) / sa[idx, 0].sum(1))
        bl = (la[idx, 1].sum(1) / (len(l_runs) * len(tasks)), la[idx, 1].sum(1) / la[idx, 0].sum(1))
        sl[(c, b)] = (bs, bl)
        row = {"tasks": len(tasks), "local_runs": len(l_runs), "local": {"valid": int(la[:, 0].sum()), "successes": int(la[:, 1].sum()),
               "overall": lov, "valid_only": lvo}, "stealth": {"valid": int(sa[:, 0].sum()), "successes": int(sa[:, 1].sum()),
               "overall": sov, "valid_only": svo}}
        for k, j in (("overall", 0), ("valid_only", 1)):
            row[f"diff_{k}_pp"] = [100 * (row["stealth"][k] - row["local"][k]), *np.percentile(100 * (bs[j] - bl[j]), [2.5, 97.5])]
        res["stealth_minus_local"].setdefault(c, {})[b] = row
    for k, j in (("overall", 0), ("valid_only", 1)):
        rr = res["stealth_minus_local"][c]
        d = 100 * np.mean([sl[(c, b)][0][j] - sl[(c, b)][1][j] for b in BENCHES], 0)
        rr[f"mean_{k}"] = {"local": float(np.mean([rr[b]["local"][k] for b in BENCHES])), "stealth": float(np.mean([rr[b]["stealth"][k] for b in BENCHES])),
                           "diff_pp": [100 * float(np.mean([rr[b]["stealth"][k] - rr[b]["local"][k] for b in BENCHES])), *np.percentile(d, [2.5, 97.5])]}
# Selector gain (decoder - SFT), local vs stealth, on tasks common to all four cells.
res["gain_local_vs_stealth"], gb = {}, {}
for b in BENCHES:
    cells = {("stealth", c): [load(STEALTH[c][0].format(b=b))[0]] for c in STEALTH}
    cells.update({("local", c): [load(p.format(b=b))[0] for p in LOCAL[c][b]] for c in STEALTH})
    tasks = sorted(set.intersection(*[set(r) for rs in cells.values() for r in rs]))
    idx = rng.integers(0, len(tasks), (B, len(tasks)))
    pt, bt = {}, {}
    for key, runs in cells.items():
        a, ov, vo = rates(runs, tasks); pt[key] = (ov, vo)
        bt[key] = (a[idx, 1].sum(1) / (len(runs) * len(tasks)), a[idx, 1].sum(1) / a[idx, 0].sum(1))
    row = {"tasks": len(tasks)}
    for k, j in (("overall", 0), ("valid_only", 1)):
        g = {m: (pt[(m, DEC)][j] - pt[(m, "sft1")][j], bt[(m, DEC)][j] - bt[(m, "sft1")][j]) for m in ("local", "stealth")}
        gb[(b, j)] = g
        row[k] = {m: [100 * g[m][0], *np.percentile(100 * g[m][1], [2.5, 97.5])] for m in g}
        row[k]["stealth_minus_local"] = [100 * (g["stealth"][0] - g["local"][0]), *np.percentile(100 * (g["stealth"][1] - g["local"][1]), [2.5, 97.5])]
    res["gain_local_vs_stealth"][b] = row
for k, j in (("overall", 0), ("valid_only", 1)):
    m = {}
    for mode in ("local", "stealth"):
        m[mode] = [100 * float(np.mean([gb[(b, j)][mode][0] for b in BENCHES])),
                   *np.percentile(100 * np.mean([gb[(b, j)][mode][1] for b in BENCHES], 0), [2.5, 97.5])]
    d = 100 * np.mean([gb[(b, j)]["stealth"][1] - gb[(b, j)]["local"][1] for b in BENCHES], 0)
    m["stealth_minus_local"] = [m["stealth"][0] - m["local"][0], *np.percentile(d, [2.5, 97.5])]
    res["gain_local_vs_stealth"][f"mean_{k}"] = m
res["notes"] = {"bootstrap": f"paired over common tasks, {B} resamples, seed {SEED}, benchmarks resampled independently",
                "stealth": "Browser Use cloud browsers, stealth + US residential proxy + CAPTCHA solving, viewport pinned 1280x1000 "
                           "at DPR 1 via CDP, screenshots Lanczos-downscaled to 1280x1000; 1 run per condition",
                "local_references": "SFT: Online-Mind2Web Nov 8 (eval-kev, 2 runs), WebVoyager/DeepShop Nov 6 (eval-ctrl, 3 runs); decoder: Nov 9 (eval-1108, 3 runs)",
                "judges": "Online-Mind2Web gpt-5.6-terra (AgentTrek), WebVoyager and DeepShop gpt-4o; judged after rollout"}
out = json.loads(json.dumps(res, default=float))
Path(__file__).with_suffix(".json").write_text(json.dumps(out, indent=1) + "\n")
for b in BENCHES:
    r = out["benchmarks"][b]
    print(b, r["common_tasks"], {c: (r[c]["valid"], r[c]["successes"], round(100 * r[c]["valid_only"], 1),
                                     round(100 * r[c]["local_same_tasks"]["overall"], 1)) for c in ("sft1", DEC)},
          [round(x, 1) for x in r["diff_valid_only_pp"]])
print({k: [round(100 * out[k]["sft1"], 1), round(100 * out[k][DEC], 1), [round(x, 1) for x in out[k]["diff_pp"]]]
       for k in ("mean_overall", "mean_valid_only")})
for c in STEALTH:
    r = out["stealth_minus_local"][c]
    print(c, {b: (r[b]["tasks"], [round(x, 1) for x in r[b]["diff_valid_only_pp"]]) for b in BENCHES},
          {k: [round(x, 1) for x in r[k]["diff_pp"]] for k in ("mean_overall", "mean_valid_only")})
g = out["gain_local_vs_stealth"]
print("gain", {k: {m: [round(x, 1) for x in g[k][m]] for m in g[k]} for k in ("mean_overall", "mean_valid_only")})
