"""Joint head (Piotr's data, N = 5) on Online-Mind2Web 300 with the actor served under sglang n-gram speculative decoding
(2026-10-24/25: first 100 tasks job 386408, remaining 200 job 386899), against the same head's 2026-10-18 run without it
(crossbench-full-t10-20261018) re-judged by gpt-5.6-terra on the saved trajectories (rejudge-terra-20261023). Both rows use
the terra judge; same task order, T 1.0 / top-p 0.9, 30 steps, one episode per task. Cost as selection-n3-sft8b-full-20261021-analysis.py:
actor PFLOP = 2 x params x (uncached prompt + generated tokens), head FLOPs as logged (n-gram draft verification not counted);
model time per episode = summed decision seconds / tasks. Paired 95% bootstrap over tasks (10,000 resamples, seed 20261018)."""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

EV = Path("/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations")
P4B = 4022468096 + 415347712
rng = np.random.default_rng(20261018)
TERRA = EV / "rejudge-terra-20261023/out/crossbench-full-t10-20261018:head5-755d4cd49137/verdicts"
CONDS = {
    "head5-terra": ("Head, Piotr's data (joint), N = 5, 10-18, terra re-judge",
                    EV / "crossbench-full-t10-20261018/online_mind2web/head5-755d4cd49137/online_mind2web", TERRA),
    "head5-ngram": ("Head, Piotr's data (joint), N = 5, n-gram, terra",
                    EV / "ngram-om2w-20261024/full300/head5-755d4cd49137/online_mind2web", None),
}


def episode(rec_path, rejudge):
    r = json.loads(Path(rec_path).read_text())
    valid = str(r.get("valid")) == "True"
    reward = float(r.get("reward") or 0)
    if rejudge is not None and (rejudge / f"{r['key']}.json").exists():   # replayed judge call: the terra verdict replaces o4-mini's
        w = json.loads((rejudge / f"{r['key']}.json").read_text()).get("verdict")
        valid, reward = valid and w is not None, float(w or 0)
    out = dict(valid=valid, success=valid and reward >= 1, steps=int(r.get("browser_steps") or 0), prompt=0, cached=0,
               completion=0, head_flops=0.0, decisions=[], generation=[], accept=[])
    for tf in (r.get("trace_files") or []):
        p = Path(tf["path"] if isinstance(tf, dict) else tf)
        for line in open(p):
            x = json.loads(line)
            out["decisions"].append(x["timing"]["decision_seconds"]); out["generation"].append(x["timing"]["generation_seconds"])
            out["head_flops"] += float((x.get("selector_telemetry") or {}).get("flops") or 0)
        for f in (p.parent.parent / "actor-requests").glob("*.json"):
            m = (json.loads(f.read_text()).get("meta") or {})
            out["prompt"] += m.get("prompt_tokens") or 0; out["cached"] += m.get("cached_tokens") or 0
            out["completion"] += m.get("completion_tokens") or 0
            if m.get("spec_accept_length") is not None:
                out["accept"].append(m["spec_accept_length"])
    return out


def load(d, rejudge):
    plan = json.loads((d / "plan.json").read_text())
    assert len(plan["task_ids"]) == 300
    keys = [(t, plan["keys"][t]) for t in plan["task_ids"]]
    with ThreadPoolExecutor(2) as pool:
        eps = list(pool.map(lambda kv: episode(d / "collection" / "records" / f"{kv[1]}.json", rejudge), keys))
    return {t: e for (t, _), e in zip(keys, eps)}


def boot(x):
    x = np.asarray(x, float)
    m = x[rng.integers(0, len(x), (10000, len(x)))].mean(1)
    return [round(100 * x.mean(), 1), round(100 * np.percentile(m, 2.5), 1), round(100 * np.percentile(m, 97.5), 1)]


data = {k: load(c[1], c[2]) for k, c in CONDS.items()}
rows = []
for k, (label, _, _) in CONDS.items():
    eps = list(data[k].values())
    n = len(eps); v = sum(e["valid"] for e in eps); s = sum(e["success"] for e in eps); steps = sum(e["steps"] for e in eps)
    actor_tok = sum(e["prompt"] - e["cached"] + e["completion"] for e in eps)
    dec = [x for e in eps for x in e["decisions"]]; gen = [x for e in eps for x in e["generation"]]
    acc = [x for e in eps for x in e["accept"]]
    pf_step = (2 * P4B * actor_tok / 1e15 + sum(e["head_flops"] for e in eps) / 1e15) / steps
    rows.append(dict(key=k, condition=label, tasks=n, successes=s, valid=v, overall=round(100 * s / n, 1),
                     valid_only=round(100 * s / v, 1), steps_per_episode=round(steps / n, 2), pflop_per_step=round(pf_step, 4),
                     pflop_per_episode=round(pf_step * steps / n, 3), decisions=len(dec),
                     decision_seconds_mean=round(float(np.mean(dec)), 2), generation_seconds_mean=round(float(np.mean(gen)), 2),
                     model_seconds_per_episode=round(float(np.sum(dec)) / n, 1),
                     hit_30_steps=sum(e["steps"] >= 30 for e in eps),
                     spec_accept_length_mean=round(float(np.mean(acc)), 2) if acc else None))
a, c = data["head5-ngram"], data["head5-terra"]
assert a.keys() == c.keys()
d_all = [a[t]["success"] - c[t]["success"] for t in a]
d_val = [a[t]["success"] - c[t]["success"] for t in a if a[t]["valid"] and c[t]["valid"]]
paired = dict(a="head5-ngram", b="head5-terra", tasks=len(d_all), overall=boot(d_all), valid_tasks=len(d_val), valid_only=boot(d_val),
              a_only_wins=sum(x == 1 for x in d_val), b_only_wins=sum(x == -1 for x in d_val))
(Path(__file__).parent / "selection-ngram-om2w-full-20261025.json").write_text(json.dumps(dict(note=__doc__, rows=rows, paired=paired), indent=1))
for r in rows:
    print(json.dumps(r))
print(json.dumps(paired))
