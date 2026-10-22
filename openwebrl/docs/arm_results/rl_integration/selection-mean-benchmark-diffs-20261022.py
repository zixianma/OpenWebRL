"""Equal-weight three-benchmark means (Online-Mind2Web, WebVoyager, DeepShop each one third) of overall and valid-only success,
and paired differences on that mean: within each benchmark, tasks are resampled (10,000 stratified bootstrap draws, seed
20261022); overall uses every task, valid-only the tasks valid under both methods; the three benchmark differences are then
averaged equally. Same runs as selection-head-sft-actions-full-20261021.json and selection-n3-sft8b-full-20261021.json."""
import json
from pathlib import Path
import numpy as np

F = Path("/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations/crossbench-full-t10-20261018")
S = Path("/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations/sft-actions-full-20261021")
E = Path("/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations/n3-8b-20261021")
BEN = ("online_mind2web", "webvoyager", "deepshop")
CONDS = {"SFT 4B": lambda b: F / b / "sft1" / b, "SFT 8B": lambda b: E / "sft8b" / b / "sft1" / b,
         "Piotr SelectionARM": lambda b: F / b / "piotr5" / b, "Head, Piotr's data": lambda b: F / b / "head5-755d4cd49137" / b,
         "Head, outcome": lambda b: S / b / "head5-7586978ccf07" / b, "Head, K = 3": lambda b: S / b / "head5-11d5eff8ffe1" / b,
         "Head, K = 1": lambda b: S / b / "head5-fd5bf58b5fa6" / b, "Head, pre-execution": lambda b: S / b / "head5-2af75856eeed" / b,
         "Head, K = 1, 1,908 SFT + RL states": lambda b: F / b / "head5-2a0fa0782535" / b}


def load(d):
    keys = json.loads((d / "plan.json").read_text())["keys"]
    out = {}
    for t, k in keys.items():
        r = json.loads((d / "collection" / "records" / f"{k}.json").read_text())
        v = str(r.get("valid")) == "True"
        out[t] = (v, v and float(r.get("reward") or 0) >= 1)
    return out


data = {(c, b): load(f(b)) for c, f in CONDS.items() for b in BEN}
rows = {}
for c in CONDS:
    ov = [np.mean([s for _, s in data[c, b].values()]) for b in BEN]
    vo = [np.mean([s for v, s in data[c, b].values() if v]) for b in BEN]
    rows[c] = dict(valid_only={b: round(100 * x, 1) for b, x in zip(BEN, vo)}, overall={b: round(100 * x, 1) for b, x in zip(BEN, ov)},
                   mean_overall=round(100 * float(np.mean(ov)), 1), mean_valid_only=round(100 * float(np.mean(vo)), 1))
rng = np.random.default_rng(20261022)


def diff(a, c, valid):
    per = []
    for b in BEN:
        ts = [t for t in data[a, b] if not valid or (data[a, b][t][0] and data[c, b][t][0])]
        per.append(np.array([float(data[a, b][t][1]) - float(data[c, b][t][1]) for t in ts]))
    point = np.mean([x.mean() for x in per])
    boots = np.mean([x[rng.integers(0, len(x), (10000, len(x)))].mean(1) for x in per], axis=0)
    return [round(100 * point, 1), round(100 * float(np.percentile(boots, 2.5)), 1), round(100 * float(np.percentile(boots, 97.5)), 1)]


pairs = {}
for a in CONDS:
    for c in ("SFT 4B", "Piotr SelectionARM"):
        if a != c:
            pairs[f"{a} - {c}"] = dict(mean_overall=diff(a, c, False), mean_valid_only=diff(a, c, True))
pairs["Head, outcome - Head, Piotr's data"] = dict(mean_overall=diff("Head, outcome", "Head, Piotr's data", False),
                                                   mean_valid_only=diff("Head, outcome", "Head, Piotr's data", True))
pairs["Head, K = 3 - Head, Piotr's data"] = dict(mean_overall=diff("Head, K = 3", "Head, Piotr's data", False),
                                                 mean_valid_only=diff("Head, K = 3", "Head, Piotr's data", True))
pairs["SFT 8B - Head, Piotr's data"] = dict(mean_overall=diff("SFT 8B", "Head, Piotr's data", False),
                                            mean_valid_only=diff("SFT 8B", "Head, Piotr's data", True))
(Path(__file__).parent / "mean-benchmark-diffs.json").write_text(json.dumps(dict(note=__doc__, rows=rows, paired=pairs), indent=1))
for c, r in rows.items():
    print(c, r["valid_only"], r["mean_overall"], r["mean_valid_only"])
for k, v in pairs.items():
    print(k, v)
