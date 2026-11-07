#!/usr/bin/env python3
"""Same eval states, four students: what the Recipe 3 ADM-4B LoRAs (GPT-5.5 / Luna labels x pre / K = 3) pick.

States: 8,000 real-choice live decisions (>= 2 distinct actions among the five candidates), 2,000 from each student's own
live runs (November 3-4, SFT 4B actor, 3 runs x 1,045 tasks), stratified 300:595:150 over OM2W / WebVoyager / DeepShop,
seed 20261107. Each state's selector request was rebuilt from the saved trace (task, URL, history, five candidates,
screenshot verified by sha256) and sent to all four students on the live serving stack (sglang, greedy, JSON schema):
runtime pre-post-k3/student-replay (replay.py, replay.sbatch). Picks are compared as actions (same tool calls, points within
5/1000), not candidate positions. No per-candidate outcomes exist on eval states: only the executed pick was played out.

Fidelity: the source student's replayed pick against its live pick. Agreement: share of states where two students pick the
same action, with chance = mean over states of sum_g (|g| / 5)^2 under independent picks proportional to proposal counts.
Type shifts: action type of each student's pick on all states and on the states where the two students of one judge
disagree. Task-clustered 95% bootstrap (10,000 resamples) for paired differences.

  student_replay.py  (CPU only; writes student-replay.json next to this file and examples to the runtime)
"""
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, "/gpfs/scrubbed/zixianma/openwebrl-runtime/selection-head-20261008/data-scale-20261023/source")
from openwebrl.selection_head import equivalent, parse_calls  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from student_live import TYPES, groups, kind  # noqa: E402

HERE = Path(__file__).resolve().parent
RD = Path("/gpfs/scrubbed/zixianma/openwebrl-runtime/selection-head-20261008/label-kfold-20261101/pre-post-k3/student-replay")
S = ("luna-pre", "luna-k3", "gpt55-pre", "gpt55-k3")
PAIRS = (("luna-pre", "luna-k3"), ("gpt55-pre", "gpt55-k3"), ("luna-pre", "gpt55-pre"), ("luna-k3", "gpt55-k3"),
         ("luna-pre", "gpt55-k3"), ("luna-k3", "gpt55-pre"))
B, SEED = 10000, 20261107


def boot(vals, groups_):
    tasks = sorted(set(groups_)); idx = {t: i for i, t in enumerate(tasks)}
    s = np.zeros(len(tasks)); c = np.zeros(len(tasks))
    for v, g in zip(vals, groups_):
        s[idx[g]] += v; c[idx[g]] += 1
    d = np.random.default_rng(SEED).integers(0, len(tasks), size=(B, len(tasks)))
    m = s[d].sum(1) / c[d].sum(1)
    return [round(100 * float(np.mean(vals)), 1), *(round(100 * float(x), 1) for x in np.percentile(m, [2.5, 97.5]))]


def main():
    states = [json.loads(l) for l in open(RD / "states.jsonl")]
    picks = defaultdict(dict); live = {}
    for r in map(json.loads, open(RD / "picks.jsonl")):
        if r.get("student"):
            picks[r["i"]][r["student"]] = r["pick"]; live[r["i"]] = r["live_pick"]
    rows = []
    for i, st in enumerate(states):
        p = picks.get(i, {})
        if len(p) != 4 or any(v is None for v in p.values()):
            continue
        rows.append((i, st, p))
    # Candidate calls per state (one pass over traces, grouped by file).
    need = defaultdict(list)
    for i, st, p in rows:
        need[st["trace"]].append(i)
    cand, prev, streak, reward = {}, {}, {}, {}
    for tr, ids in need.items():
        steps = {s["turn"]: s for s in map(json.loads, open(tr))}
        res = json.loads((Path(tr).parent.parent / "result.json").read_text())
        for i in ids:
            t = states[i]["turn"]; s = steps[t]
            cand[i] = [parse_calls(c.get("action") or "") for c in s["candidates"]]
            done = [parse_calls(steps[u]["candidates"][steps[u]["selected_index"]]["action"]) for u in range(t)]
            prev[i] = done[-1] if done else None
            streak[i] = next((n for n, c in enumerate(reversed(done)) if kind(c) != "scroll"), len(done))  # trailing scrolls
            reward[i] = res.get("reward")
    same = lambda i, a, b: a == b or (cand[i][a] is not None and equivalent(cand[i][a], cand[i][b]))  # noqa: E731
    task = {i: f"{st['bench']}/{st['task']}" for i, st, _ in rows}
    out = dict(note=__doc__.split("\n\n")[0], states=len(rows), sampled=len(states),
               by_source=Counter(st["source"] for _, st, _ in rows))
    out["fidelity"] = {s: round(100 * np.mean([same(i, p[s], live[i]) for i, st, p in rows if st["source"] == s]), 1) for s in S}
    chance = np.mean([sum((len(g[1]) / len(cand[i])) ** 2 for g in groups(cand[i])) for i, _, _ in rows])
    out["agreement"] = dict(chance=round(100 * chance, 1),
                            **{f"{a}~{b}": boot([same(i, p[a], p[b]) for i, _, p in rows], [task[i] for i, _, _ in rows])
                               for a, b in PAIRS})
    out["agreement_by_bench"] = {b: {f"{x}~{y}": round(100 * np.mean([same(i, p[x], p[y]) for i, st, p in rows if st["bench"] == b]), 1)
                                     for x, y in PAIRS} for b in ("online_mind2web", "webvoyager", "deepshop")}
    out["all_four_agree"] = round(100 * np.mean([all(same(i, p["luna-pre"], p[s]) for s in S) for i, _, p in rows]), 1)
    # Label-type effect consistent across judges: both pre students pick X, both K = 3 students pick Y != X.
    split = [(i, st, p) for i, st, p in rows if same(i, p["luna-pre"], p["gpt55-pre"]) and same(i, p["luna-k3"], p["gpt55-k3"])
             and not same(i, p["luna-pre"], p["luna-k3"])]
    out["consistent_label_split"] = dict(states=len(split), share=round(100 * len(split) / len(rows), 1),
                                         switches=[[f"{x} -> {y}", n] for (x, y), n in Counter(
                                             (kind(cand[i][p["luna-pre"]]), kind(cand[i][p["luna-k3"]])) for i, _, p in split).most_common(10)])
    jsplit = [(i, st, p) for i, st, p in rows if same(i, p["luna-pre"], p["luna-k3"]) and same(i, p["gpt55-pre"], p["gpt55-k3"])
              and not same(i, p["luna-pre"], p["gpt55-pre"])]
    out["consistent_judge_split"] = dict(states=len(jsplit), share=round(100 * len(jsplit) / len(rows), 1),
                                         switches=[[f"{x} -> {y}", n] for (x, y), n in Counter(
                                             (kind(cand[i][p["luna-pre"]]), kind(cand[i][p["gpt55-pre"]])) for i, _, p in jsplit).most_common(10)])
    # Behaviour on identical states.
    beh = {}
    for s in S:
        pk = [(i, cand[i][p[s]]) for i, _, p in rows]
        sz = [(i, sum(same(i, p[s], j) for j in range(len(cand[i])))) for i, _, p in rows]
        so = [(i, p) for i, _, p in rows if any(kind(c) == "stop / answer" for c in cand[i])]
        som = [(i, p) for i, p in so if 0 < sum(kind(c) == "stop / answer" for c in cand[i]) <= 2]
        beh[s] = dict(types={t: round(100 * np.mean([kind(c) == t for _, c in pk]), 1) for t in TYPES},
                      proposed_by=round(float(np.mean([n for _, n in sz])), 2),
                      singleton=round(100 * np.mean([n == 1 for _, n in sz]), 1),
                      repeat_prev=round(100 * np.mean([prev[i] is not None and c is not None and equivalent(prev[i], c) for i, c in pk]), 1),
                      stop_given_offered=round(100 * np.mean([kind(cand[i][p[s]]) == "stop / answer" for i, p in so]), 1),
                      stop_given_minority_offer=round(100 * np.mean([kind(cand[i][p[s]]) == "stop / answer" for i, p in som]), 1),
                      first_position=round(100 * np.mean([p[s] == 0 for _, _, p in rows]), 1))
    beh["offered"] = {t: round(100 * np.mean([kind(c) == t for i, _, _ in rows for c in cand[i]]), 1) for t in TYPES}
    beh["states_stop_offered"] = sum(any(kind(c) == "stop / answer" for c in cand[i]) for i, _, _ in rows)
    out["behaviour"] = beh
    # Paired K = 3 - pre differences per judge (task-clustered).
    diff = {}
    for j in ("luna", "gpt55"):
        a, b = f"{j}-pre", f"{j}-k3"; g = [task[i] for i, _, _ in rows]
        f = {"back": lambda i, s, p: kind(cand[i][p[s]]) == "back",
             "scroll": lambda i, s, p: kind(cand[i][p[s]]) == "scroll",
             "stop / answer": lambda i, s, p: kind(cand[i][p[s]]) == "stop / answer",
             "repeat_prev": lambda i, s, p: prev[i] is not None and cand[i][p[s]] is not None and equivalent(prev[i], cand[i][p[s]]),
             "singleton": lambda i, s, p: sum(same(i, p[s], k) for k in range(len(cand[i]))) == 1}
        diff[j] = {k: boot([fn(i, b, p) - fn(i, a, p) for i, _, p in rows], g) for k, fn in f.items()}
    out["k3_minus_pre"] = diff
    # After >= 2 consecutive scrolls, with both a scroll and a go-back among the candidates.
    sc = [(i, p) for i, _, p in rows if streak[i] >= 2 and {"scroll", "back"} <= {kind(c) for c in cand[i]}]
    out["after_scrolls"] = dict(states=len(sc), **{s: {k: round(100 * np.mean([kind(cand[i][p[s]]) == k for i, p in sc]), 1)
                                                      for k in ("scroll", "back", "click")} for s in S})
    # Only outcome on eval states: the logged episode (source student's own pick) on label-split states vs all its states.
    ep = {}
    for s in S:
        own = [i for i, st, _ in rows if st["source"] == s and reward[i] is not None]
        sp = [i for i, st, _ in split if st["source"] == s and reward[i] is not None]
        ep[s] = dict(all_states=len(own), all_success=round(100 * np.mean([reward[i] == 1 for i in own]), 1),
                     split_states=len(sp), split_success=round(100 * np.mean([reward[i] == 1 for i in sp]), 1))
    out["source_episode_success"] = ep
    sub = [(i, p) for i, _, p in split if kind(cand[i][p["luna-pre"]]) != kind(cand[i][p["luna-k3"]])]
    out["consistent_label_split"]["type_changes"] = len(sub)
    (HERE / "student-replay.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    with open(RD / "label-split-examples.jsonl", "w") as fo:
        for i, st, p in split:
            fo.write(json.dumps(dict(i=i, **st, pre=p["luna-pre"], k3=p["luna-k3"], picks=p,
                                     pre_kind=kind(cand[i][p["luna-pre"]]), k3_kind=kind(cand[i][p["luna-k3"]]))) + "\n")
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
