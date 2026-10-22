"""Architecture 2x2 at full scale (2026-10-22): interaction (joint vs independent transformer) x loss (soft CE vs pointwise BCE),
all K = 1 Luna labels on the same 1,380 SFT-actor states, frozen SFT-4B features, N = 5 SFT samples, T 1.0 / top-p 0.9, 30 steps,
every task of Online-Mind2Web (300), WebVoyager (595), DeepShop (150), same task order as crossbench-full-t10-20261018.
Means weight the three benchmarks equally. Paired differences: stratified bootstrap (tasks resampled within each benchmark,
10,000 draws, seed 20261022); overall uses every task, valid-only the tasks valid under both. Cost uses analyze.py's accounting."""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

F = Path("/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations/crossbench-full-t10-20261018")
S = Path("/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations/sft-actions-full-20261021")
BEN = ("online_mind2web", "webvoyager", "deepshop")
P4B = 4022468096 + 415347712
CONDS = {"SFT 4B": (F, "sft1"), "Piotr SelectionARM": (F, "piotr5"),
         "Joint transformer, soft CE": (S, "head5-fd5bf58b5fa6"), "Joint transformer, BCE": (S, "head5-eb0b0820bc71"),
         "Independent transformer, soft CE": (S, "head5-1a3c6edd3b17"), "Independent transformer, BCE": (S, "head5-dbfdd42d9af0")}


def episode(p):
    r = json.loads(Path(p).read_text())
    valid = str(r.get("valid")) == "True"
    out = dict(valid=valid, success=valid and float(r.get("reward") or 0) >= 1, steps=int(r.get("browser_steps") or 0),
               sel_tokens=int(r.get("selector_input_tokens") or 0) + int(r.get("selector_output_tokens") or 0),
               actor_tok=0, head_flops=0.0, decisions=[], fallback=0)
    for tf in (r.get("trace_files") or []):
        q = Path(tf["path"] if isinstance(tf, dict) else tf)
        if q.exists():
            for line in open(q):
                x = json.loads(line)
                out["decisions"].append(x["timing"]["decision_seconds"])
                out["head_flops"] += float((x.get("selector_telemetry") or {}).get("flops") or 0)
                out["fallback"] += bool(x.get("fallback"))
            for f in (q.parent.parent / "actor-requests").glob("*.json"):
                m = json.loads(f.read_text()).get("meta") or {}
                out["actor_tok"] += (m.get("prompt_tokens") or 0) - (m.get("cached_tokens") or 0) + (m.get("completion_tokens") or 0)
    return out


def load(root, slug, b):
    d = root / b / slug / b
    keys = json.loads((d / "plan.json").read_text())["keys"]
    with ThreadPoolExecutor(4) as pool:
        eps = list(pool.map(lambda k: episode(d / "collection" / "records" / f"{k}.json"), keys.values()))
    return dict(zip(keys, eps))


data = {(c, b): load(*CONDS[c], b) for c in CONDS for b in BEN}
for b in BEN:
    assert len({frozenset(data[c, b]) for c in CONDS}) == 1, b
rows = {}
for c in CONDS:
    per = {}
    for b in BEN:
        eps = list(data[c, b].values()); n = len(eps); v = sum(e["valid"] for e in eps); k = sum(e["success"] for e in eps)
        steps = sum(e["steps"] for e in eps)
        sel_pf = (2 * P4B * sum(e["sel_tokens"] for e in eps) if c == "Piotr SelectionARM" else sum(e["head_flops"] for e in eps)) / 1e15
        dec = [x for e in eps for x in e["decisions"]]
        per[b] = dict(tasks=n, successes=k, valid=v, overall=round(100 * k / n, 1), valid_only=round(100 * k / v, 1),
                      pflop_per_episode=round((2 * P4B * sum(e["actor_tok"] for e in eps) / 1e15 + sel_pf) / n, 3),
                      model_seconds_per_episode=round(sum(dec) / n, 1), steps_per_episode=round(steps / n, 2),
                      fallbacks=sum(e["fallback"] for e in eps), decisions=len(dec))
    rows[c] = dict(per_benchmark=per, **{f"mean_{m}": round(float(np.mean([per[b][m] for b in BEN])), 1 if m != "pflop_per_episode" else 3)
                                         for m in ("overall", "valid_only", "pflop_per_episode", "model_seconds_per_episode")})
rng = np.random.default_rng(20261022)


def diff(a, c, valid, sign=1):
    """a, c: lists of conditions averaged within each side (for main effects)."""
    per = []
    for b in BEN:
        ts = [t for t in data[a[0], b] if not valid or all(data[x, b][t]["valid"] for x in a + c)]
        per.append(np.array([np.mean([data[x, b][t]["success"] for x in a]) - np.mean([data[x, b][t]["success"] for x in c]) for t in ts]))
    point = np.mean([x.mean() for x in per])
    boots = np.mean([x[rng.integers(0, len(x), (10000, len(x)))].mean(1) for x in per], axis=0)
    return [round(100 * point, 1), round(100 * float(np.percentile(boots, 2.5)), 1), round(100 * float(np.percentile(boots, 97.5)), 1)]


JS, JB, IS, IB = "Joint transformer, soft CE", "Joint transformer, BCE", "Independent transformer, soft CE", "Independent transformer, BCE"
contrasts = {f"{h} - {base}": ([h], [base]) for h in (JS, JB, IS, IB) for base in ("SFT 4B", "Piotr SelectionARM")}
contrasts.update({"joint - independent (avg over losses)": ([JS, JB], [IS, IB]), "soft CE - BCE (avg over interactions)": ([JS, IS], [JB, IB]),
                  "joint - independent, soft CE": ([JS], [IS]), "joint - independent, BCE": ([JB], [IB]),
                  "soft CE - BCE, joint": ([JS], [JB]), "soft CE - BCE, independent": ([IS], [IB])})
pairs = {k: dict(mean_overall=diff(a, c, False), mean_valid_only=diff(a, c, True)) for k, (a, c) in contrasts.items()}
# interaction term: (JS - IS) - (JB - IB), via the same stratified bootstrap on per-task contrasts
per_o, per_v = [], []
for b in BEN:
    f = lambda t: (data[JS, b][t]["success"] - data[IS, b][t]["success"]) - (data[JB, b][t]["success"] - data[IB, b][t]["success"])
    per_o.append(np.array([f(t) for t in data[JS, b]], float))
point = np.mean([x.mean() for x in per_o]); boots = np.mean([x[rng.integers(0, len(x), (10000, len(x)))].mean(1) for x in per_o], axis=0)
pairs["interaction x loss term (overall)"] = [round(100 * point, 1), round(100 * float(np.percentile(boots, 2.5)), 1), round(100 * float(np.percentile(boots, 97.5)), 1)]
(S / "analysis" / "arch-2x2-20261022.json").write_text(json.dumps(dict(note=__doc__, conditions={c: v[1] for c, v in CONDS.items()}, rows=rows, paired=pairs), indent=1))
for c, r in rows.items():
    print(f"{c:<34}", {b: (r['per_benchmark'][b]['overall'], r['per_benchmark'][b]['valid_only']) for b in BEN},
          r["mean_overall"], r["mean_valid_only"], "PF/ep", r["mean_pflop_per_episode"], "s/ep", r["mean_model_seconds_per_episode"],
          "fb", sum(r['per_benchmark'][b]['fallbacks'] for b in BEN))
for k, v in pairs.items():
    print(k, v)
