"""Outcome-objective heads live on Online-Mind2Web 300 (2026-10-25, jobs 386977-386979; seed-0 heads of
selection-outcome-objectives-20261025): softmax-rate soft CE, centred advantage MSE, count-weighted binomial BCE, against
the current best-action outcome head (2026-10-21, sft-actions-full-20261021) and the 10-18 SFT 4B / Piotr SelectionARM
references. Every row is judged by gpt-5.6-terra: the new runs live, the older runs by replaying their saved judge
requests (rejudge-terra-20261023). Same task order, SFT-4B actor, T 1.0 / top-p 0.9, 30 steps, five candidates.
Overall counts invalid episodes as failures. Paired 95% bootstrap over tasks (10,000 resamples, seed 20261018);
valid-only differences use tasks valid under both runs."""
import json
from pathlib import Path
import numpy as np

EV = Path("/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations")
TERRA = EV / "rejudge-terra-20261023/out"
B = "online_mind2web"
RUNS = {  # key: (label, collection dir, terra re-judge verdicts or None when judged live by terra)
    "sft1": ("SFT 4B, one sample", EV / f"crossbench-full-t10-20261018/{B}/sft1/{B}", TERRA / "crossbench-full-t10-20261018:sft1"),
    "piotr5": ("Piotr SelectionARM", EV / f"crossbench-full-t10-20261018/{B}/piotr5/{B}", TERRA / "crossbench-full-t10-20261018:piotr5"),
    "best": ("Head, best-action outcome target (10-21)", EV / f"sft-actions-full-20261021/{B}/head5-7586978ccf07/{B}",
             TERRA / "sft-actions-full-20261021:head5-7586978ccf07"),
    "soft": ("Head, softmax(rate / 0.25) soft CE", EV / f"advantage-heads-20261025/om2w-r2/head5-9933fb324a52/{B}", None),
    "adv": ("Head, centred advantage MSE", EV / f"advantage-heads-20261025/om2w-r2/head5-8950bcb1b617/{B}", None),
    "binom": ("Head, count-weighted binomial BCE", EV / f"advantage-heads-20261025/om2w-r2/head5-c07e328de4a1/{B}", None),
}
rng = np.random.default_rng(20261018)


def load(directory, rejudge):
    plan = json.loads((directory / "plan.json").read_text())
    out = {}
    for t in plan["task_ids"]:
        r = json.loads((directory / "collection/records" / f"{plan['keys'][t]}.json").read_text())
        valid, reward = str(r["valid"]) == "True", float(r.get("reward") or 0)
        if rejudge is not None and (rejudge / "verdicts" / f"{r['key']}.json").exists():
            w = json.loads((rejudge / "verdicts" / f"{r['key']}.json").read_text()).get("verdict")
            valid, reward = valid and w is not None, float(w or 0)
        out[t] = dict(valid=valid, success=valid and reward >= 1, steps=int(r.get("browser_steps") or 0))
    assert len(out) == 300
    return out


def boot(x):
    x = np.asarray(x, float)
    m = x[rng.integers(0, len(x), (10000, len(x)))].mean(1)
    return [round(100 * x.mean(), 1), round(100 * np.percentile(m, 2.5), 1), round(100 * np.percentile(m, 97.5), 1)]


data = {k: load(d, r) for k, (_, d, r) in RUNS.items()}
rows = []
for k, (label, _, _) in RUNS.items():
    e = list(data[k].values()); s = sum(x["success"] for x in e); v = sum(x["valid"] for x in e)
    rows.append(dict(key=k, condition=label, tasks=len(e), successes=s, valid=v, overall=round(100 * s / len(e), 1),
                     valid_only=round(100 * s / v, 1), steps_per_episode=round(sum(x["steps"] for x in e) / len(e), 2)))
paired = []
for a in ("soft", "adv", "binom"):
    for b in ("best", "sft1", "piotr5"):
        A, C = data[a], data[b]
        assert A.keys() == C.keys()
        d_all = [A[t]["success"] - C[t]["success"] for t in A]
        d_val = [A[t]["success"] - C[t]["success"] for t in A if A[t]["valid"] and C[t]["valid"]]
        paired.append(dict(a=a, b=b, tasks=len(d_all), overall=boot(d_all), valid_tasks=len(d_val), valid_only=boot(d_val)))
d_all = [data["best"][t]["success"] - data["sft1"][t]["success"] for t in data["best"]]
paired.append(dict(a="best", b="sft1", tasks=300, overall=boot(d_all)))
Path(__file__).with_name("selection-outcome-objectives-live-20261025.json").write_text(
    json.dumps(dict(note=__doc__, rows=rows, paired=paired), indent=1))
for r in rows:
    print(json.dumps(r))
for p in paired:
    print(json.dumps(p))
