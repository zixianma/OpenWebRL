"""RL 8B (OpenWebRL/OpenWebRL-8B @ 824a54f, 4,096 response tokens) on DeepShop: local browser vs Browser Use stealth with and without
the US residential proxy. Local RL 8B alone Nov 15 (eval-rl8b-ds-1114) and Nov 4-5 (evaluations/sameday-rl8b-20261104, 3 runs);
stealth no proxy Nov 15 (stealth/noproxy-ds-1114: RL 8B alone, + default ADM R3+WA LoRA); stealth US proxy Nov 12
(stealth/rl8b-proxy-1112). 1 run each unless stated, all 150 tasks, T 1.0 / top-p 0.9, 30 steps, GPT-4o DeepShop judge.
Deferred records are read from the Nov 15 judge sidecar (/gpfs/scrubbed refused writes). Paired bootstrap over tasks
(10,000 resamples, seed 20261029) against local RL 8B alone Nov 15. Writes the adjacent JSON."""
import glob, json
from pathlib import Path
import numpy as np
W = "/gpfs/scrubbed/zixianma/openwebrl-runtime/recipe3-ablations-20261104"
L = "/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations/sameday-rl8b-20261104"
SIDE = {}
for l in open("/gpfs/projects/krishna/zixianma/judge-sidecar-20261115/verdicts.jsonl"):
    x = json.loads(l); SIDE[(x["plan"], x["task_id"])] = x


def load(plan_glob):
    d = {}
    for plan in glob.glob(plan_glob):
        for f in glob.glob(plan + "/collection/records/*.json"):
            r = json.load(open(f)); t = r["task_id"]; jt = str(r.get("judge_text") or "")
            if jt.startswith("DEFERRED") or jt.strip() == "All judge API retry attempts exhausted.":
                s = SIDE[(plan, t)]; v = bool(s["valid"]); ok = v and float(s["reward"] or 0) >= 1
            else:
                v = str(r["valid"]) == "True"; ok = v and float(r["reward"] or 0) >= 1
            d.setdefault(t, []).append((v, ok))
    return d


C = {"local-1115": ("Local, RL 8B alone (Nov 15)", load(W + "/eval-rl8b-ds-1114/run1/deepshop/sft1/deepshop")),
     "local-1104": ("Local, RL 8B alone (Nov 4-5, 3 runs)", load(L + "/rl8b/run*/deepshop/sft1/deepshop")),
     "noproxy-alone": ("Stealth no proxy, RL 8B alone (Nov 15)", load(W + "/stealth/noproxy-ds-1114/run1/deepshop/sft1/deepshop")),
     "noproxy-adm": ("Stealth no proxy, RL 8B + ADM (Nov 15)", load(W + "/stealth/noproxy-ds-1114/run1/deepshop/arm5-d470d7e52a38/deepshop")),
     "proxy-alone": ("Stealth US proxy, RL 8B alone (Nov 12)", load(W + "/stealth/rl8b-proxy-1112/run1/deepshop/sft1/deepshop")),
     "proxy-adm": ("Stealth US proxy, RL 8B + ADM (Nov 12)", load(W + "/stealth/rl8b-proxy-1112/run1/deepshop/arm5-d470d7e52a38/deepshop"))}


def diff(A, R):
    ts = sorted(set(A) & set(R)); a = np.array([np.mean([e[1] for e in A[t]]) for t in ts]); b = np.array([np.mean([e[1] for e in R[t]]) for t in ts])
    idx = np.random.default_rng(20261029).integers(0, len(ts), (10000, len(ts))); dd = 100 * (a[idx].mean(1) - b[idx].mean(1))
    return [100 * (a.mean() - b.mean()), float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))]


res = {}
for k, (label, d) in C.items():
    e = [x for t in d for x in d[t]]; n = len(e); v = sum(x[0] for x in e); s = sum(x[1] for x in e)
    res[k] = dict(label=label, episodes=n, valid=v, successes=s, overall=s / n, valid_only=s / v,
                  minus_local_1115_pp=None if k == "local-1115" else diff(d, C["local-1115"][1]))
res["noproxy_adm_minus_alone_pp"] = diff(C["noproxy-adm"][1], C["noproxy-alone"][1])
res["proxy_adm_minus_alone_pp"] = diff(C["proxy-adm"][1], C["proxy-alone"][1])
Path(__file__).with_suffix(".json").write_text(json.dumps(res, indent=1, default=float) + "\n")
for k, r in res.items():
    print(k, r if not isinstance(r, dict) else (r["episodes"], r["valid"], r["successes"], round(100 * r["overall"], 1), r["minus_local_1115_pp"] and [round(x, 1) for x in r["minus_local_1115_pp"]]))
