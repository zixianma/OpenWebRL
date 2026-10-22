"""N = 3 selectors on Online-Mind2Web (300) and SFT 8B single sample on all three benchmarks (2026-10-21), against the
full-set N = 5 / SFT 4B runs of crossbench-full-t10-20261018 (same task order, T 1.0 / top-p 0.9, 30 steps, one episode
per task, one condition-benchmark per H200). Metrics and cost exactly as full_analysis_5cond.py: actor PFLOP = 2 x params x
(uncached prompt + generated tokens); Piotr's selector 2 x 4.44e9 x its tokens; head FLOPs as logged. Model time per
episode = mean decision seconds x decisions / tasks. Paired 95% bootstrap over tasks (10,000 resamples, seed 20261018)."""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

E = Path(__file__).resolve().parents[1]
F = Path("/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations/crossbench-full-t10-20261018")
BENCH = ("online_mind2web", "webvoyager", "deepshop")
P4B, P8B = 4022468096 + 415347712, 8767123696
rng = np.random.default_rng(20261018)
# key -> (label, directory per benchmark, actor params, benchmarks)
CONDS = {
    "sft1": ("SFT 4B, 1 sample", lambda b: F / b / "sft1" / b, P4B, BENCH),
    "sft8b": ("SFT 8B, 1 sample", lambda b: E / "sft8b" / b / "sft1" / b, P8B, BENCH),
    "piotr5": ("Piotr SelectionARM, N = 5", lambda b: F / b / "piotr5" / b, P4B, BENCH),
    "piotr3": ("Piotr SelectionARM, N = 3", lambda b: E / "n3" / "piotr3" / b, P4B, BENCH[:1]),
    "head5-755d4cd49137": ("Head, Piotr's data (joint), N = 5", lambda b: F / b / "head5-755d4cd49137" / b, P4B, BENCH),
    "head3-755d4cd49137": ("Head, Piotr's data (joint), N = 3", lambda b: E / "n3" / "head3-755d4cd49137" / b, P4B, BENCH[:1]),
    "head3-26c57559ccb4": ("Head, Piotr's data (pointwise), N = 3", lambda b: E / "n3" / "head3-26c57559ccb4" / b, P4B, BENCH[:1]),
}


def episode(rec_path):
    r = json.loads(Path(rec_path).read_text())
    valid = str(r.get("valid")) == "True"
    out = dict(valid=valid, success=valid and float(r.get("reward") or 0) >= 1, steps=int(r.get("browser_steps") or 0),
               sel_tokens=int(r.get("selector_input_tokens") or 0) + int(r.get("selector_output_tokens") or 0),
               prompt=0, cached=0, completion=0, head_flops=0.0, decisions=[], selection=[])
    for tf in (r.get("trace_files") or []):
        p = Path(tf["path"] if isinstance(tf, dict) else tf)
        if p.exists():
            for line in open(p):
                x = json.loads(line)
                out["decisions"].append(x["timing"]["decision_seconds"]); out["selection"].append(x["timing"]["selection_seconds"])
                out["head_flops"] += float((x.get("selector_telemetry") or {}).get("flops") or 0)
            for f in (p.parent.parent / "actor-requests").glob("*.json"):
                m = (json.loads(f.read_text()).get("meta") or {})
                out["prompt"] += m.get("prompt_tokens") or 0; out["cached"] += m.get("cached_tokens") or 0
                out["completion"] += m.get("completion_tokens") or 0
    return out


def load(d):
    keys = list(json.loads((d / "plan.json").read_text())["keys"].items())
    with ThreadPoolExecutor(4) as pool:
        eps = list(pool.map(lambda kv: episode(d / "collection" / "records" / f"{kv[1]}.json"), keys))
    return {t: e for (t, _), e in zip(keys, eps)}


def boot(x):
    x = np.asarray(x, float)
    m = x[rng.integers(0, len(x), (10000, len(x)))].mean(1)
    return [round(100 * x.mean(), 1), round(100 * np.percentile(m, 2.5), 1), round(100 * np.percentile(m, 97.5), 1)]


data = {(k, b): load(c[1](b)) for k, c in CONDS.items() for b in c[3]}
rows = []
for k, (label, _, params, benches) in CONDS.items():
    for b in list(benches) + (["all"] if len(benches) > 1 else []):
        eps = [e for bb in (benches if b == "all" else (b,)) for e in data[k, bb].values()]
        n = len(eps); v = sum(e["valid"] for e in eps); s = sum(e["success"] for e in eps); steps = sum(e["steps"] for e in eps)
        actor_tok = sum(e["prompt"] - e["cached"] + e["completion"] for e in eps)
        sel_pf = (2 * P4B * sum(e["sel_tokens"] for e in eps) if k.startswith("piotr") else sum(e["head_flops"] for e in eps)) / 1e15
        dec = [x for e in eps for x in e["decisions"]]; sel = [x for e in eps for x in e["selection"]]
        pf_step = (2 * params * actor_tok / 1e15 + sel_pf) / steps
        rows.append(dict(key=k, condition=label, benchmark=b, tasks=n, successes=s, valid=v, overall=round(100 * s / n, 1),
                         valid_only=round(100 * s / v, 1), steps_per_episode=round(steps / n, 2), pflop_per_step=round(pf_step, 4),
                         pflop_per_episode=round(pf_step * steps / n, 3), decisions=len(dec),
                         decision_seconds_mean=round(float(np.mean(dec)), 2), selection_seconds_mean=round(float(np.mean(sel)), 3),
                         model_seconds_per_episode=round(float(np.sum(dec)) / n, 1)))
PAIRS = [("sft8b", "sft1", BENCH), ("piotr3", "sft1", BENCH[:1]), ("piotr3", "piotr5", BENCH[:1]),
         ("head3-755d4cd49137", "sft1", BENCH[:1]), ("head3-755d4cd49137", "head5-755d4cd49137", BENCH[:1]),
         ("head3-755d4cd49137", "piotr3", BENCH[:1]), ("head3-26c57559ccb4", "sft1", BENCH[:1]),
         ("head3-26c57559ccb4", "head3-755d4cd49137", BENCH[:1]), ("head3-26c57559ccb4", "piotr3", BENCH[:1]),
         ("piotr5", "sft8b", BENCH), ("head5-755d4cd49137", "sft8b", BENCH)]
pairs = []
for a, c, benches in PAIRS:
    for b in list(benches) + (["all"] if len(benches) > 1 else []):
        bs = benches if b == "all" else (b,)
        assert all(data[a, bb].keys() == data[c, bb].keys() for bb in bs)
        d_all = [data[a, bb][t]["success"] - data[c, bb][t]["success"] for bb in bs for t in data[a, bb]]
        d_val = [data[a, bb][t]["success"] - data[c, bb][t]["success"] for bb in bs for t in data[a, bb]
                 if data[a, bb][t]["valid"] and data[c, bb][t]["valid"]]
        pairs.append(dict(a=a, b=c, benchmark=b, tasks=len(d_all), overall=boot(d_all), valid_tasks=len(d_val), valid_only=boot(d_val)))
(Path(__file__).parent / "n3-8b-results.json").write_text(json.dumps(dict(note=__doc__, rows=rows, paired=pairs), indent=1))
for r in rows:
    print(r["key"], r["benchmark"], r["successes"], r["valid"], r["overall"], r["valid_only"], "PF/step", r["pflop_per_step"],
          "PF/ep", r["pflop_per_episode"], "dec", r["decision_seconds_mean"], "sel", r["selection_seconds_mean"],
          "t/ep", r["model_seconds_per_episode"], "steps", r["steps_per_episode"])
for p in pairs:
    print(p["a"], "-", p["b"], p["benchmark"], p["tasks"], p["overall"], p["valid_tasks"], p["valid_only"])
