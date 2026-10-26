"""Same-day rerun (2026-10-25, jobs 386994-386999) of the two heads trained on Piotr's GPT-5.5 panels: the earlier
head (heads/listwise, train_selection_head.py, 2,692 states, run 2026-10-18) and the all-data scaling head
(data-scale-20261023/heads/nall/seed0, train_head_round.py --extra-only, 2,587 states with our test tasks excluded, run
2026-10-23). Same protocol as the earlier runs: SFT-4B actor, T 1.0 / top-p 0.9, 30 steps, five candidates, full task
order; Online-Mind2Web judged by gpt-5.6-terra (earlier OM2W runs re-judged by replaying saved requests), WebVoyager and
DeepShop by GPT-4o. Overall counts invalid episodes as failures. "Mean" weights each benchmark 1/3; paired 95% bootstrap
resamples tasks within each benchmark (10,000 resamples, seed 20261018)."""
import json
from pathlib import Path
import numpy as np

EV = Path("/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations")
TERRA = EV / "rejudge-terra-20261023/out"
BENCH = ("online_mind2web", "webvoyager", "deepshop")
EARLY, ALL = "head5-755d4cd49137", "head5-fc589b57c9e0"
RUNS = {  # key: (label, {benchmark: (collection dir, terra re-judge dir or None)})
    "early-1025": ("Earlier Piotr-data head, run 10-25", {b: (EV / f"piotr-head-sameday-20261025/{b}/{EARLY}/{b}", None) for b in BENCH}),
    "all-1025": ("All-data scaling head, run 10-25", {b: (EV / f"piotr-head-sameday-20261025/{b}/{ALL}/{b}", None) for b in BENCH}),
    "early-1018": ("Earlier Piotr-data head, run 10-18", {b: (EV / f"crossbench-full-t10-20261018/{b}/{EARLY}/{b}",
                                                              TERRA / f"crossbench-full-t10-20261018:{EARLY}" if b == BENCH[0] else None)
                                                          for b in BENCH}),
    "all-1023": ("All-data scaling head, run 10-23/24", {b: (EV / f"data-scale-20261023/{b}/{ALL}/{b}", None) for b in BENCH}),
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
        out[t] = dict(valid=valid, success=valid and reward >= 1)
    return out


data = {(k, b): load(*src) for k, (_, srcs) in RUNS.items() for b, src in srcs.items()}
rows = []
for k, (label, _) in RUNS.items():
    row = dict(key=k, condition=label)
    for b in BENCH:
        e = list(data[k, b].values()); s = sum(x["success"] for x in e); v = sum(x["valid"] for x in e)
        row[b] = dict(tasks=len(e), successes=s, valid=v, overall=round(100 * s / len(e), 1), valid_only=round(100 * s / v, 1))
    row["mean_overall"] = round(sum(row[b]["overall"] for b in BENCH) / 3, 1)
    row["mean_valid_only"] = round(sum(row[b]["valid_only"] for b in BENCH) / 3, 1)
    rows.append(row)


def paired(a, c):
    out = {}
    diffs = {}
    for b in BENCH:
        A, C = data[a, b], data[c, b]
        assert A.keys() == C.keys()
        diffs[b] = np.array([A[t]["success"] - C[t]["success"] for t in A], float)
        m = diffs[b][rng.integers(0, len(diffs[b]), (10000, len(diffs[b])))].mean(1)
        out[b] = [round(100 * diffs[b].mean(), 1), round(100 * np.percentile(m, 2.5), 1), round(100 * np.percentile(m, 97.5), 1)]
        out[b + "_boot"] = m
    mean = sum(out.pop(b + "_boot") for b in BENCH) / 3
    out["mean"] = [round(100 * sum(diffs[b].mean() for b in BENCH) / 3, 1), round(100 * np.percentile(mean, 2.5), 1),
                   round(100 * np.percentile(mean, 97.5), 1)]
    return dict(a=a, b=c, **out)


pairs = [paired("all-1025", "early-1025"),       # same day: the question asked
         paired("early-1025", "early-1018"),     # day effect, earlier head
         paired("all-1025", "all-1023"),         # day effect, all-data head
         paired("all-1023", "early-1018")]       # the original cross-day comparison
Path(__file__).with_name("selection-piotr-head-sameday-20261025.json").write_text(
    json.dumps(dict(note=__doc__, rows=rows, paired=pairs), indent=1))
for r in rows:
    print(r["key"], {b: (r[b]["successes"], r[b]["overall"]) for b in BENCH}, r["mean_overall"])
for p in pairs:
    print(json.dumps(p))
