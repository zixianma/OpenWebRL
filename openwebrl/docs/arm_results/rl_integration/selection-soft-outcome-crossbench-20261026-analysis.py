"""Soft-CE outcome head (softmax(rate / 0.25), seed 0; selection-outcome-objectives-20261025) on all three benchmarks:
Online-Mind2Web 300 (2026-10-25, job 386978), WebVoyager 595 and DeepShop 150 (2026-10-26, jobs 388858 and
388859/388967), against the best-action outcome head (2026-10-21), SFT 4B one sample and Piotr SelectionARM (2026-10-18).
SFT-4B actor, five candidates, T 1.0 / top-p 0.9, 30 steps, full task order. Online-Mind2Web judged by gpt-5.6-terra
(older runs by replaying their saved judge requests), WebVoyager and DeepShop by GPT-4o. DeepShop drifted between run
dates, so it is also compared with the 2026-10-24/25 same-day DeepShop runs. Overall counts invalid episodes as
failures; "mean" weights each benchmark 1/3; paired 95% bootstrap resamples tasks within each benchmark (10,000
resamples, seed 20261018)."""
import json
from pathlib import Path
import numpy as np

EV = Path("/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations")
TERRA = EV / "rejudge-terra-20261023/out"
BENCH = ("online_mind2web", "webvoyager", "deepshop")
SOFT = "head5-9933fb324a52"
X18, X21 = "crossbench-full-t10-20261018", "sft-actions-full-20261021"


def ref(run, cond):
    return {b: (EV / f"{run}/{b}/{cond}/{b}", TERRA / f"{run}:{cond}" if b == BENCH[0] else None) for b in BENCH}


RUNS = {  # key: (label, {benchmark: (collection dir, terra re-judge dir or None)})
    "soft": ("Head, soft CE to softmax(rate / 0.25) (10-25/26)",
             {"online_mind2web": (EV / f"advantage-heads-20261025/om2w-r2/{SOFT}/online_mind2web", None),
              "webvoyager": (EV / f"advantage-heads-20261025/webvoyager-r2/{SOFT}/webvoyager", None),
              "deepshop": (EV / f"advantage-heads-20261025/deepshop-r2/{SOFT}/deepshop", None)}),
    "best": ("Head, best-action outcome target (10-21)", ref(X21, "head5-7586978ccf07")),
    "sft1": ("SFT 4B, one sample (10-18)", ref(X18, "sft1")),
    "piotr5": ("Piotr SelectionARM (10-18)", ref(X18, "piotr5")),
}
DEEPSHOP_SAMEDAY = {
    "sft1-1024": ("SFT 4B, one sample (DeepShop 10-24)", EV / "deepshop-sameday-20261024/deepshop/sft1/deepshop"),
    "early-1024": ("Earlier Piotr-data head (DeepShop 10-24)", EV / "deepshop-sameday-20261024/deepshop/head5-755d4cd49137/deepshop"),
    "early-1025": ("Earlier Piotr-data head (DeepShop 10-25)", EV / "piotr-head-sameday-20261025/deepshop/head5-755d4cd49137/deepshop"),
}
rng = np.random.default_rng(20261018)


def load(directory, rejudge=None):
    plan = json.loads((directory / "plan.json").read_text())
    out = {}
    for t in plan["task_ids"]:
        r = json.loads((directory / "collection/records" / f"{plan['keys'][t]}.json").read_text())
        valid, reward = str(r["valid"]) == "True", float(r.get("reward") or 0)
        if rejudge is not None and (rejudge / "verdicts" / f"{r['key']}.json").exists():
            w = json.loads((rejudge / "verdicts" / f"{r['key']}.json").read_text()).get("verdict")
            valid, reward = valid and w is not None, float(w or 0)
        out[t] = dict(valid=valid, success=valid and reward >= 1)
    return out


def stats(e):
    e = list(e.values()); s = sum(x["success"] for x in e); v = sum(x["valid"] for x in e)
    return dict(tasks=len(e), successes=s, valid=v, overall=round(100 * s / len(e), 1), valid_only=round(100 * s / v, 1))


def ci(m):
    return [round(100 * np.percentile(m, 2.5), 1), round(100 * np.percentile(m, 97.5), 1)]


data = {(k, b): load(*src) for k, (_, srcs) in RUNS.items() for b, src in srcs.items()}
rows = []
for k, (label, _) in RUNS.items():
    row = dict(key=k, condition=label, **{b: stats(data[k, b]) for b in BENCH})
    row["mean_overall"] = round(sum(row[b]["overall"] for b in BENCH) / 3, 1)
    row["mean_valid_only"] = round(sum(row[b]["valid_only"] for b in BENCH) / 3, 1)
    rows.append(row)

paired = []
for c in ("best", "sft1", "piotr5"):
    out, boots, means = dict(a="soft", b=c), [], []
    for b in BENCH:
        A, C = data["soft", b], data[c, b]
        assert A.keys() == C.keys()
        d = np.array([A[t]["success"] - C[t]["success"] for t in A], float)
        m = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
        out[b] = [round(100 * d.mean(), 1), *ci(m)]
        boots.append(m); means.append(d.mean())
    out["mean"] = [round(100 * sum(means) / 3, 1), *ci(sum(boots) / 3)]
    paired.append(out)

deepshop = []
soft_ds = data["soft", "deepshop"]
for k, (label, d) in DEEPSHOP_SAMEDAY.items():
    C = load(d)
    assert C.keys() == soft_ds.keys()
    x = np.array([soft_ds[t]["success"] - C[t]["success"] for t in C], float)
    m = x[rng.integers(0, len(x), (10000, len(x)))].mean(1)
    deepshop.append(dict(key=k, condition=label, **stats(C), soft_minus=[round(100 * x.mean(), 1), *ci(m)]))

Path(__file__).with_name("selection-soft-outcome-crossbench-20261026.json").write_text(
    json.dumps(dict(note=__doc__, rows=rows, paired=paired, deepshop_sameday=deepshop), indent=1))
for r in rows:
    print(r["key"], {b: (r[b]["successes"], r[b]["valid"], r[b]["overall"], r[b]["valid_only"]) for b in BENCH},
          r["mean_overall"], r["mean_valid_only"])
for p in paired + deepshop:
    print(json.dumps(p))
