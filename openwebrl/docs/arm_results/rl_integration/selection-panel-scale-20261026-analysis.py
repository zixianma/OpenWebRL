"""Panel scaling at fixed states: joint selection head trained on the 1,393 Piotr states of the one-panel head with at
most K GPT-5.5 panels per state (K = 1: 1,393 panels; 8: 10,181; 16: every panel, 18,587; nested subsets for K = 8, 16;
batch 64, last of 20 epochs, seed 0). Live on Online-Mind2Web 300 (gpt-5.6-terra), WebVoyager 595 and DeepShop 150
(GPT-4o); SFT-4B actor, five candidates, T 1.0 / top-p 0.9, 30 steps, full task order. K = 8 / 16 ran 2026-10-26; K = 1 is
the one-panel head of the data-scaling runs (Online-Mind2Web / WebVoyager 2026-10-23/24; DeepShop rerun same-day 2026-10-26,
job 391288; its first DeepShop run, 2026-10-24, is kept as k1_ds_1024 to show the date drift). SFT 4B: Online-Mind2Web / WebVoyager from 2026-10-18 (OM2W
re-judged by terra), DeepShop same-day 2026-10-26. Overall counts invalid as failure; "mean" weights benchmarks equally;
paired 95% bootstrap resamples tasks within each benchmark (10,000 resamples, seed 20261018). Offline: fixed-five score
of all three seeds per K."""
import json
from pathlib import Path
import numpy as np

EV = Path("/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations")
TERRA = EV / "rejudge-terra-20261023/out"
HEADS = Path("/gpfs/scrubbed/zixianma/openwebrl-runtime/selection-head-20261008/panel-scale-20261026/heads")
BENCH = ("online_mind2web", "webvoyager", "deepshop")
PS, DS = EV / "panel-scale-20261026", EV / "data-scale-20261023"
K8, K16, K1 = "head5-0f16963c46eb", "head5-87adc458c551", "head5-878e69f9264f"
X18 = EV / "crossbench-full-t10-20261018"
RUNS = {  # key: (label, panels, {benchmark: (collection dir, terra re-judge dir or None)})
    "k1": ("1 panel per state (OM2W/WV 10-23/24, DS 10-26)", 1393, {
        "online_mind2web": (DS / f"online_mind2web/{K1}/online_mind2web", None),
        "webvoyager": (DS / f"webvoyager/{K1}/webvoyager", None), "deepshop": (PS / f"deepshop/{K1}/deepshop", None)}),
    "k8": ("<= 8 panels per state (10-26)", 10181, {b: (PS / f"{b}/{K8}/{b}", None) for b in BENCH}),
    "k16": ("<= 16 panels per state, all (10-26)", 18587, {b: (PS / f"{b}/{K16}/{b}", None) for b in BENCH}),
    "sft1": ("SFT 4B, one sample", None, {
        "online_mind2web": (X18 / "online_mind2web/sft1/online_mind2web", TERRA / "crossbench-full-t10-20261018:sft1"),
        "webvoyager": (X18 / "webvoyager/sft1/webvoyager", None),
        "deepshop": (PS / "deepshop/sft1/deepshop", None)}),
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


data = {(k, b): load(*src) for k, (_, _, srcs) in RUNS.items() for b, src in srcs.items()}
rows = []
for k, (label, panels, _) in RUNS.items():
    row = dict(key=k, condition=label, panels=panels, **{b: stats(data[k, b]) for b in BENCH})
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


pairs = [paired(a, c) for a, c in (("k1", "sft1"), ("k8", "sft1"), ("k16", "sft1"), ("k8", "k1"), ("k16", "k1"), ("k16", "k8"))]
offline = {}
for K in (1, 2, 4, 8, 16):
    s = [json.loads((HEADS / f"k{K}/seed{i}/scores.json").read_text()) for i in range(3)]
    offline[K] = dict(panels=s[0]["extra"]["panels"], states=s[0]["extra"]["base_states"],
                      fixed_five=[x["fixed_five"]["head"] for x in s], sft_sample=s[0]["fixed_five"]["sft_sample"],
                      minus_sft_sample=[[round(v, 2) for v in x["fixed_five"]["head_minus_sft_sample"]] for x in s])
old = load(DS / f"deepshop/{K1}/deepshop"); new = data["k1", "deepshop"]
x = np.array([new[t]["success"] - old[t]["success"] for t in new], float)
k1_ds_1024 = dict(**stats(old), rerun_minus_first=[round(100 * x.mean(), 1), *ci(x[rng.integers(0, len(x), (10000, len(x)))].mean(1))])
print("k1_ds_1024", k1_ds_1024)
Path(__file__).with_name("selection-panel-scale-20261026.json").write_text(
    json.dumps(dict(note=__doc__, rows=rows, paired=pairs, offline=offline, k1_ds_1024=k1_ds_1024), indent=1))
for r in rows:
    print(r["key"], {b: (r[b]["successes"], r[b]["valid"], r[b]["overall"]) for b in BENCH}, r["mean_overall"], r["om2w_wv_overall"])
for p in pairs:
    print(json.dumps(p))
print(json.dumps(offline))
