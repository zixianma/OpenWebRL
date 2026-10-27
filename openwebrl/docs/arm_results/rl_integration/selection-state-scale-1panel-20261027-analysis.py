"""State scaling at one GPT-5.5 panel per state: joint selection head trained on the first N Piotr base states of whole
tasks (nested, subset seed 20261023, our 86 test tasks excluded; N = 254 / 501 / 1,011 / 2,587 = every training state),
one panel per state from a fixed per-state order (--extra-nested-panels); batch 64, last of 20 epochs, seed 0. 1,393 states =
the one-panel head of the data-scaling runs (its single panel drawn by an unnested seeded draw). Live on Online-Mind2Web 300
(gpt-5.6-terra), WebVoyager 595 and DeepShop 150 (GPT-4o); SFT-4B actor, five candidates, T 1.0 / top-p 0.9, 30 steps, full
task order. New heads ran 2026-10-26/27 (jobs 391471-391482); the 1,393-state head ran Online-Mind2Web / WebVoyager on
2026-10-23/24 and DeepShop on 2026-10-26 (job 391288). SFT 4B: Online-Mind2Web / WebVoyager from 2026-10-18 (OM2W re-judged by
terra), DeepShop same-day 2026-10-27 (job 391483). Overall counts invalid as failure; "mean" weights benchmarks equally;
paired 95% bootstrap resamples tasks within each benchmark (10,000 resamples, seed 20261018). Offline: fixed-five score of
three seeds per N (1,393 = panel-scale K = 1 heads, same nested panel)."""
import json
from pathlib import Path
import numpy as np

EV = Path("/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations")
TERRA = EV / "rejudge-terra-20261023/out"
SH = Path("/gpfs/scrubbed/zixianma/openwebrl-runtime/selection-head-20261008")
BENCH = ("online_mind2web", "webvoyager", "deepshop")
SS, PS, DSC = EV / "state-scale-1panel-20261026", EV / "panel-scale-20261026", EV / "data-scale-20261023"
X18 = EV / "crossbench-full-t10-20261018"
NEW = {"n250": (254, "head5-3c1bbd704f85"), "n500": (501, "head5-3f279a03a2fa"), "n1000": (1011, "head5-f481ccabd374"),
       "nall": (2587, "head5-9b7cd2a74b8b")}
K1 = "head5-878e69f9264f"
RUNS = {k: (f"{n:,} states, 1 panel each (10-26/27)", n, {b: (SS / f"{b}/{h}/{b}", None) for b in BENCH})
        for k, (n, h) in NEW.items()}
RUNS["n1393"] = ("1,393 states, 1 panel each (OM2W/WV 10-23/24, DS 10-26)", 1393, {
    "online_mind2web": (DSC / f"online_mind2web/{K1}/online_mind2web", None),
    "webvoyager": (DSC / f"webvoyager/{K1}/webvoyager", None), "deepshop": (PS / f"deepshop/{K1}/deepshop", None)})
RUNS = {k: RUNS[k] for k in ("n250", "n500", "n1000", "n1393", "nall")}
RUNS["sft1"] = ("SFT 4B, one sample", None, {
    "online_mind2web": (X18 / "online_mind2web/sft1/online_mind2web", TERRA / "crossbench-full-t10-20261018:sft1"),
    "webvoyager": (X18 / "webvoyager/sft1/webvoyager", None), "deepshop": (SS / "deepshop/sft1/deepshop", None)})
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


data = {(k, b): load(*src) for k, (_, _, srcs) in RUNS.items() for b, src in srcs.items()}
rows = []
for k, (label, states, _) in RUNS.items():
    row = dict(key=k, condition=label, states=states, **{b: stats(data[k, b]) for b in BENCH})
    row["mean_overall"] = round(sum(row[b]["overall"] for b in BENCH) / 3, 1)
    row["om2w_wv_overall"] = round((row["online_mind2web"]["overall"] + row["webvoyager"]["overall"]) / 2, 1)
    rows.append(row)


def paired(a, c):
    out, boots, means = dict(a=a, b=c), {}, {}
    for b in BENCH:
        A, C = data[a, b], data[c, b]
        assert A.keys() == C.keys(), (a, c, b)
        d = np.array([A[t]["success"] - C[t]["success"] for t in A], float)
        m = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
        out[b] = [round(100 * d.mean(), 1), *ci(m)]; boots[b] = m; means[b] = d.mean()
    out["mean"] = [round(100 * sum(means.values()) / 3, 1), *ci(sum(boots.values()) / 3)]
    two = ("online_mind2web", "webvoyager")
    out["om2w_wv"] = [round(100 * sum(means[b] for b in two) / 2, 1), *ci(sum(boots[b] for b in two) / 2)]
    return out


heads = [k for k in RUNS if k != "sft1"]
pairs = [paired(k, "sft1") for k in heads] + [paired("nall", "n250"), paired("nall", "n1000")]
offline = {}
for k, d in [("n250", SH / "state-scale-1panel-20261026/heads/n250"), ("n500", SH / "state-scale-1panel-20261026/heads/n500"),
             ("n1000", SH / "state-scale-1panel-20261026/heads/n1000"), ("n1393", SH / "panel-scale-20261026/heads/k1"),
             ("nall", SH / "state-scale-1panel-20261026/heads/nall")]:
    s = [json.loads((d / f"seed{i}/scores.json").read_text()) for i in range(3)]
    offline[k] = dict(states=s[0]["extra"]["base_states"], panels=s[0]["extra"]["panels"],
                      fixed_five=[x["fixed_five"]["head"] for x in s], sft_sample=s[0]["fixed_five"]["sft_sample"],
                      mean=round(sum(x["fixed_five"]["head"] for x in s) / 3, 2))
Path(__file__).with_name("selection-state-scale-1panel-20261027.json").write_text(
    json.dumps(dict(note=__doc__, rows=rows, paired=pairs, offline=offline), indent=1))
for r in rows:
    print(r["key"], {b: (r[b]["successes"], r[b]["valid"], r[b]["overall"]) for b in BENCH}, r["mean_overall"], r["om2w_wv_overall"])
for p in pairs:
    print(json.dumps(p))
print(json.dumps({k: (v["mean"], v["fixed_five"]) for k, v in offline.items()}))
