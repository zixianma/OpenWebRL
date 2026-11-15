"""Cost and success of the 4.4M d128-L2 head (ADM-4M, training seed 0) for the cost figure variant, with the same accounting
as selection-recipe3-vs-cost-20261104-analysis.py: actor PFLOP = 2 x 4.44B x (uncached prompt + completion tokens) of the actor
requests, plus the head's logged selector FLOPs; model time = summed decision seconds; both per episode, run 1 (all 1,045
tasks). Success: overall success, 3 runs pooled, November 14-15 window (eval-pre). Writes the adjacent JSON."""
import glob, json, os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
E = "/gpfs/scrubbed/zixianma/openwebrl-runtime/recipe3-ablations-20261104/eval-pre"
HERE = Path(__file__).resolve().parent
BENCH = ("online_mind2web", "webvoyager", "deepshop")
P4B = 4022468096 + 415347712
HEADS = {"adm-4m": ("head5-5fbddd8b2031", 4362754)}
succ = json.loads((HERE / "selection-head-pre-vs-lookahead-20261115.json").read_text())["conditions"]
SKEY = {"adm-4m": "d128-lookahead"}


def episode(p):
    r = json.loads(Path(p).read_text()); o = dict(prompt=0, cached=0, completion=0, head_flops=0.0, dec=0.0)
    for tf in r.get("trace_files") or []:
        q = Path(tf["path"] if isinstance(tf, dict) else tf)
        if not q.exists(): continue
        for line in open(q):
            x = json.loads(line); o["dec"] += x["timing"]["decision_seconds"]; o["head_flops"] += float((x.get("selector_telemetry") or {}).get("flops") or 0)
        for f in (q.parent.parent / "actor-requests").glob("*.json"):
            m = json.loads(f.read_text()).get("meta") or {}
            o["prompt"] += m.get("prompt_tokens") or 0; o["cached"] += m.get("cached_tokens") or 0; o["completion"] += m.get("completion_tokens") or 0
    return o


out = {}
with ThreadPoolExecutor(int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))) as pool:
    for k, (slug, params) in HEADS.items():
        out[k] = dict(slug=slug, system_parameters=P4B + params, benchmarks={})
        for b in BENCH:
            eps = list(pool.map(episode, sorted(glob.glob(f"{E}/run1/{b}/{slug}/{b}/collection/records/*.json"))))
            n = len(eps); pf = sum(2 * P4B * (e["prompt"] - e["cached"] + e["completion"]) + e["head_flops"] for e in eps) / 1e15
            out[k]["benchmarks"][b] = dict(cost_tasks=n, pflop_per_episode=round(pf / n, 3), model_seconds_per_episode=round(sum(e["dec"] for e in eps) / n, 1),
                                           overall=round(100 * succ[SKEY[k]][b]["overall"], 1))
        out[k]["reported_mean_overall"] = round(100 * succ[SKEY[k]]["mean_overall"], 1)
        print(k, out[k], flush=True)
(HERE / "selection-adm4m-cost-20261115.json").write_text(json.dumps(dict(note=__doc__, systems=out), indent=1) + "\n")
