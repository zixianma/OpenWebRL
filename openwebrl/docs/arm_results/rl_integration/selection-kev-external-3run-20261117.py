"""External Kev selectors over the SFT 4B actor, 3 runs each, against the November 14-15 SFT 4B / Kev-A / Recipe 3 decoder cohort
(selection-kevA-decoder-sft-20261115). Original Kev-4B (text/page-view decision contract, `kev5:4b`): run 1 = Nov 11
(jev-kev-crossbench-eval-md-20261111/run1), runs 2-3 = Nov 16-17 (kevweb/results-kev4b, node-local runs with live judging).
Kev-4B-Web-Selector (XunmeiLiu/Kev-4B-Web-Selector @ d15adf84, behind serve_kevweb_selector.py, `arm5` contract): runs 1-3 =
Nov 15-16 (kevweb/results); its copy-back lost episodes when the project fileset filled, and missing tasks are excluded (not
counted as failures). SFT 4B actor, N = 5, T 1.0 / top-p 0.9, seed 20261018, 30 steps. Paired bootstrap over tasks present in
every run of both methods (10,000 resamples, seed 20261029, benchmarks resampled independently, per-task success averaged over
runs). DEFERRED / dead-key records are refused unless the Nov 15 judge sidecar has their verdict. Writes the adjacent JSON."""
import glob, json
from pathlib import Path
import numpy as np
R = "/gpfs/scrubbed/zixianma/openwebrl-runtime/recipe3-ablations-20261104"; K = "/gpfs/projects/krishna/zixianma/kevweb"
E = "/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations/jev-kev-crossbench-eval-md-20261111"
BENCH = ["online_mind2web", "webvoyager", "deepshop"]; NT = {"online_mind2web": 300, "webvoyager": 595, "deepshop": 150}
LABEL = {"sft": "SFT 4B alone (1 sample)", "kev4b": "+ Kev-4B (original)", "kevweb": "+ Kev-4B-Web-Selector",
         "kevA": "+ Kev-A (Recipe 3 GPT-5.5 K = 3)", "decoder": "+ 4B LoRA decoder (Recipe 3 GPT-5.5 K = 3)"}
SIDE = {}
for l in open("/gpfs/projects/krishna/zixianma/judge-sidecar-20261115/verdicts.jsonl"):
    x = json.loads(l); SIDE[(x["plan"], x["task_id"])] = x


def read(files, plan=None):
    d, unusable = {}, 0
    for f in files:
        try:
            x = json.load(open(f))
        except (OSError, ValueError):
            unusable += 1; continue
        jt = str(x.get("judge_text") or "")
        if jt.startswith("DEFERRED") or jt.strip() == "All judge API retry attempts exhausted.":
            sv = SIDE.get((plan, x["task_id"])) if plan else None
            if sv is None:
                unusable += 1; continue
            v = bool(sv["valid"]); d[x["task_id"]] = (v, v and float(sv["reward"] or 0) >= 1); continue
        v = str(x["valid"]) == "True"; d[x["task_id"]] = (v, v and float(x["reward"] or 0) >= 1)
    return d, unusable


def runs(k, b):
    if k == "kev4b":
        return [read(glob.glob(f"{E}/run1/{b}/kev5-4b/{b}/collection/records/*.json"))] + \
               [read(glob.glob(f"{K}/results-kev4b/{r}/{b}/records/*.json")) for r in ("run2", "run3")]
    if k == "kevweb":
        return [read(glob.glob(f"{K}/results/{r}/{b}/records/*.json")) for r in ("run1", "run2", "run3")]
    root, slug = {"sft": (f"{R}/eval-pre", "sft1"), "kevA": (f"{R}/eval-kevA-1114", "head5x-9ff81e7c894f"),
                  "decoder": (f"{R}/eval-kevA-1114", "arm5-9fadc8dc4655")}[k]
    out = []
    for r in ("run1", "run2", "run3"):
        p = f"{root}/{r}/{b}/{slug}/{b}"; out.append(read(glob.glob(f"{p}/collection/records/*.json"), p))
    return out


D = {(k, b): runs(k, b) for k in LABEL for b in BENCH}
res = {}
for k, label in LABEL.items():
    r = dict(label=label, runs=3)
    for b in BENCH:
        rs = [x[0] for x in D[(k, b)]]; n = sum(len(x) for x in rs); v = sum(t[0] for x in rs for t in x.values())
        s = sum(t[1] for x in rs for t in x.values())
        r[b] = dict(scheduled=3 * NT[b], episodes=n, missing=3 * NT[b] - n, unusable_files=[x[1] for x in D[(k, b)]], valid=v,
                    successes=s, overall=s / n, valid_only=s / v,
                    per_run_overall=[sum(t[1] for t in x.values()) / len(x) for x in rs])
    r["mean_overall"] = float(np.mean([r[b]["overall"] for b in BENCH])); r["mean_valid_only"] = float(np.mean([r[b]["valid_only"] for b in BENCH]))
    res[k] = r
rng = np.random.default_rng(20261029)
pairs = [(a, c) for a in ("kev4b", "kevweb") for c in ("sft", "kevA", "decoder")] + [("kev4b", "kevweb"), ("kevA", "sft"), ("decoder", "sft")]
res["differences"] = {}
for a, c in pairs:
    bo, bv, po, pv, per = [], [], [], [], {}
    for b in BENCH:
        A = [x[0] for x in D[(a, b)]]; C = [x[0] for x in D[(c, b)]]; tasks = sorted(set.intersection(*[set(x) for x in A + C]))
        f = lambda rs, j: np.array([np.mean([x[t][j] for x in rs]) for t in tasks])
        av, as_, cv, cs = f(A, 0), f(A, 1), f(C, 0), f(C, 1); idx = rng.integers(0, len(tasks), (10000, len(tasks)))
        d = as_[idx].mean(1) - cs[idx].mean(1); bo.append(d); bv.append(as_[idx].sum(1) / av[idx].sum(1) - cs[idx].sum(1) / cv[idx].sum(1))
        po.append(as_.mean() - cs.mean()); pv.append(as_.sum() / av.sum() - cs.sum() / cv.sum())
        per[b] = dict(tasks=len(tasks), overall_pp=[100 * po[-1], *np.percentile(100 * d, [2.5, 97.5])])
    res["differences"][f"{a}_minus_{c}"] = dict(per, mean_overall_pp=[100 * float(np.mean(po)), *np.percentile(100 * np.mean(bo, 0), [2.5, 97.5])],
                                                mean_valid_only_pp=[100 * float(np.mean(pv)), *np.percentile(100 * np.mean(bv, 0), [2.5, 97.5])])
for k in LABEL:
    r = res[k]; print(r["label"], " ".join(f"{b[:4]} {100*r[b]['overall']:.1f} ({r[b]['valid']}/{r[b]['episodes']}, {r[b]['successes']})" for b in BENCH),
                      f"| ov {100*r['mean_overall']:.1f} vo {100*r['mean_valid_only']:.1f}")
for k, v in res["differences"].items():
    print(k, *(f"{b[:4]}(n{v[b]['tasks']}) {v[b]['overall_pp'][0]:+.1f} [{v[b]['overall_pp'][1]:+.1f}, {v[b]['overall_pp'][2]:+.1f}]" for b in BENCH),
          "ov %+.1f [%+.1f, %+.1f]" % tuple(v["mean_overall_pp"]), "vo %+.1f [%+.1f, %+.1f]" % tuple(v["mean_valid_only_pp"]))
Path(__file__).with_suffix(".json").write_text(json.dumps(json.loads(json.dumps(res, default=float)), indent=1) + "\n")
