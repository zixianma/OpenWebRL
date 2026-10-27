"""Post-execution label head live: joint selection head trained on Luna post-execution labels (1,380 SFT-action branch
states, round-1 recipe, seed 0, head5-73757f12d4e2; labels job 392245, head job 392246). Live on Online-Mind2Web 300
(gpt-5.6-terra), WebVoyager 595 and DeepShop 150 (GPT-4o); SFT-4B actor, five candidates, T 1.0 / top-p 0.9, 30 steps
(jobs 392418 / 392419 / 392422, 2026-10-27). Reference = the SFT 4B row used for the other label heads
(selection-data-scale-20261023.json: Online-Mind2Web re-judged by terra, WebVoyager and DeepShop from 2026-10-18).
DeepShop SFT also ran twice on 2026-10-27 (jobs 391483 and 392423, same day as this head) for drift checks. Overall counts
invalid as failure; mean weights benchmarks equally; paired 95% bootstrap over tasks within each benchmark (10,000
resamples, seed 20261018)."""
import json
from pathlib import Path
import numpy as np

EV = Path("/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations")
TERRA = EV / "rejudge-terra-20261023/out"
BENCH = ("online_mind2web", "webvoyager", "deepshop")
P, X18 = EV / "post-label-20261027", EV / "crossbench-full-t10-20261018"
H = "head5-73757f12d4e2"
RUNS = {"post": {b: (P / f"{b}/{H}/{b}", None) for b in BENCH},
        "sft1": {"online_mind2web": (X18 / "online_mind2web/sft1/online_mind2web", TERRA / "crossbench-full-t10-20261018:sft1"),
                 "webvoyager": (X18 / "webvoyager/sft1/webvoyager", None),
                 "deepshop": (X18 / "deepshop/sft1/deepshop", None)}}
DS_SFT = {"sft1_ds_20261027a": EV / "state-scale-1panel-20261026/deepshop/sft1/deepshop",
          "sft1_ds_20261027b": P / "deepshop/sft1/deepshop"}
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


def diff(A, C):
    assert A.keys() == C.keys()
    d = np.array([A[t]["success"] - C[t]["success"] for t in A], float)
    return d.mean(), d[rng.integers(0, len(d), (10000, len(d)))].mean(1)


data = {(k, b): load(*src) for k, srcs in RUNS.items() for b, src in srcs.items()}
out = dict(note=__doc__, rows={})
for k in RUNS:
    row = {b: stats(data[k, b]) for b in BENCH}
    row["mean_overall"] = round(sum(row[b]["overall"] for b in BENCH) / 3, 1)
    out["rows"][k] = row
means, boots, per = {}, {}, {}
for b in BENCH:
    means[b], boots[b] = diff(data["post", b], data["sft1", b])
    per[b] = [round(100 * means[b], 1), *ci(boots[b])]
out["post_minus_sft"] = dict(per, mean=[round(100 * sum(means.values()) / 3, 1), *ci(sum(boots.values()) / 3)])
for k, d in DS_SFT.items():
    e = load(d); out["rows"][k] = {"deepshop": stats(e)}
    m, bt = diff(data["post", "deepshop"], e); out[f"post_minus_{k}"] = [round(100 * m, 1), *ci(bt)]
print(json.dumps({k: v for k, v in out.items() if k != "note"}, indent=1))
Path(__file__).with_suffix("").with_name("selection-post-label-20261027.json").write_text(json.dumps(out, indent=1))
