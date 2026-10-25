"""Outcome-label objectives for the selection head (2026-10-25), offline. Joint transformer head (22.6 M), the 1,380
SFT-action branch train states, 20 epochs, seeds 0-2; targets from draws 0-2 (valid draws only): best action(s) with
ties split (current), softmax(rate / 0.25), centred advantage MSE, count-weighted binomial BCE on the rate
(scripts/train_head_round.py --source outcome | outcome_soft | outcome_rate --loss advmse | binom). Scored on held-out
draws 3-7 of the 348 test states (86 tasks): distinct-action view (head − uniform over actions) and the fixed five
(first five SFT proposals, head − one SFT sample). Paired difference vs the current objective: per state the held-out
rate of the picked action, averaged over the three seeds, task-clustered bootstrap (5,000 resamples, seed 0)."""
import json
from pathlib import Path
import numpy as np

D = Path("/gpfs/scrubbed/zixianma/openwebrl-runtime/selection-head-20261008/sft-actions-20261021")
RUNS = D / "advantage-20261025"
OBJECTIVES = {"best": "Best action, soft CE (current)", "soft": "Soft CE to softmax(rate / 0.25)",
              "adv": "Centred advantage MSE", "binom": "Count-weighted binomial BCE"}
rec = {x["state_id"]: x for x in map(json.loads, open(D / "actions.jsonl"))}


def held(action):
    cells = [c for d in range(3, 8) for c in action["outcomes"][str(d)] if c and c[0]]
    return sum(c[1] for c in cells) / len(cells) if cells else None


rows, per_state = [], {}
for key, label in OBJECTIVES.items():
    scores = [json.loads((RUNS / f"{key}-s{s}" / "scores.json").read_text()) for s in range(3)]
    picks = [json.loads((RUNS / f"{key}-s{s}" / "picks.json").read_text())["picks"] for s in range(3)]
    per_state[key] = {sid: float(np.mean([held(rec[sid]["actions"][p[sid]]) for p in picks])) for sid in picks[0]}
    rows.append(dict(key=key, objective=label, seeds=3,
                     distinct=[x["success"]["head"] for x in scores], fixed_five=[x["fixed_five"]["head"] for x in scores],
                     distinct_mean=round(float(np.mean([x["success"]["head"] for x in scores])), 2),
                     fixed_five_mean=round(float(np.mean([x["fixed_five"]["head"] for x in scores])), 2),
                     uniform=scores[0]["success"]["uniform"], sft_sample_fixed_five=scores[0]["fixed_five"]["sft_sample"],
                     oracle=scores[0]["success"]["oracle"], test_states=scores[0]["test_states"],
                     train_states=scores[0]["train_states"]))
ids = sorted(per_state["best"])
tasks = sorted({rec[s]["task_id"] for s in ids})
members = {t: [i for i, s in enumerate(ids) if rec[s]["task_id"] == t] for t in tasks}
rng = np.random.default_rng(0)
paired = []
for key in ("soft", "adv", "binom"):
    d = np.array([per_state[key][s] - per_state["best"][s] for s in ids])
    boot = [d[np.concatenate([members[tasks[j]] for j in rng.integers(0, len(tasks), len(tasks))])].mean() for _ in range(5000)]
    paired.append(dict(a=key, b="best", states=len(ids), tasks=len(tasks), diff_pp=round(100 * d.mean(), 2),
                       ci_pp=[round(100 * np.percentile(boot, 2.5), 2), round(100 * np.percentile(boot, 97.5), 2)],
                       states_where_picks_differ=int((np.abs(d) > 1e-9).sum())))
out = Path(__file__).with_name("selection-outcome-objectives-20261025.json")
out.write_text(json.dumps(dict(note=__doc__, rows=rows, paired=paired), indent=1))
for r in rows:
    print(r["key"], r["distinct_mean"], r["fixed_five_mean"], r["distinct"], r["fixed_five"])
for p in paired:
    print(p)
