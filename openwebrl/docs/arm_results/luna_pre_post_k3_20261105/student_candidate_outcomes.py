#!/usr/bin/env python3
"""Outcome of every candidate action on replayed eval states, from live eval episodes only (no new rollouts).

Same matching as student_outcomes.py (same task, step and screenshot sha256 across all November 1-4 selector runs; executed
action compared by equivalence; valid episodes only), but for all distinct actions among the five candidates, not only the
two student picks. Question: when a student picks a minority action (proposed by fewer of the five candidates than the
unique most-proposed action), does that pick do better or worse than the majority action, on states where both were played
out? Per-state success averaged, task-clustered bootstrap.

  student_candidate_outcomes.py  (CPU only; writes candidate-outcomes.json next to this file and per-state rows to the runtime)
"""
from collections import defaultdict
import glob
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, "/gpfs/scrubbed/zixianma/openwebrl-runtime/selection-head-20261008/data-scale-20261023/source")
from openwebrl.selection_head import equivalent, parse_calls  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from student_live import groups, kind  # noqa: E402
from student_outcomes import B, E, RD, ROOTS, SEED  # noqa: E402

HERE = Path(__file__).resolve().parent
S = ("luna-pre", "luna-k3", "gpt55-pre", "gpt55-k3")


def main():
    states = [json.loads(l) for l in open(RD / "states.jsonl")]
    picks = defaultdict(dict)
    for r in map(json.loads, open(RD / "picks.jsonl")):
        if r.get("student") and r["pick"] is not None:
            picks[r["i"]][r["student"]] = r["pick"]
    by_trace = defaultdict(list)
    for i, st in enumerate(states):
        if len(picks[i]) == 4:
            by_trace[st["trace"]].append(i)
    tgt = {}
    for tr, ids in by_trace.items():
        steps = {s["turn"]: s for s in map(json.loads, open(tr))}
        for i in ids:
            st = states[i]; s = steps[st["turn"]]
            cands = [parse_calls(c.get("action") or "") for c in s["candidates"]]
            g = groups(cands)
            if len(g) >= 2:
                who = {n: [x for x in S if picks[i][x] in m] for n, (_, m) in enumerate(g)}
                tgt[i] = dict(sha=s["screenshot_sha256"], task=st["task"], bench=st["bench"], turn=st["turn"], groups=g, who=who,
                              source=st["source"], trace=tr)
    want = defaultdict(list)
    for i, t in tgt.items():
        want[(t["bench"], t["task"], t["turn"], t["sha"])].append(i)
    print("target states", len(tgt), flush=True)
    hits, episodes = defaultdict(list), 0
    for root in ROOTS:
        for f in glob.glob(str(E / root / "run*" / "*" / "*" / "*" / "collection" / "records" / "*.json")):
            r = json.loads(Path(f).read_text()); episodes += 1
            tr = [t["path"] for t in r.get("trace_files") or [] if t["path"].endswith(".jsonl")]
            if not tr or not r.get("valid"):
                continue
            for s in map(json.loads, open(tr[0])):
                k = (r["benchmark"], r["task_id"], s["turn"], s.get("screenshot_sha256"))
                if k in want and s["selected_index"] is not None:
                    hits[k].append(dict(root=root, run=f.split("/")[len(str(E).split("/")) + 1], slug=r["condition_slug"],
                                        reward=r.get("reward"), action=s["candidates"][s["selected_index"]]["action"], trace=tr[0]))
    print("episodes scanned", episodes, flush=True)
    rows = []
    for i, t in tgt.items():
        eps = [(e, parse_calls(e["action"] or "")) for e in hits[(t["bench"], t["task"], t["turn"], t["sha"])]]
        gs = []
        for n, (rep, m) in enumerate(t["groups"]):
            mt = [e for e, c in eps if rep is not None and c is not None and equivalent(rep, c)]
            gs.append(dict(members=m, size=len(m), kind=kind(rep), picked_by=t["who"][n], n=len(mt),
                           success=sum(e["reward"] == 1 for e in mt), episodes=mt))
        if sum(g["n"] > 0 for g in gs) >= 2:
            rows.append(dict(i=i, task=f"{t['bench']}/{t['task']}", bench=t["bench"], turn=t["turn"], source=t["source"],
                             trace=t["trace"], groups=gs))
    out = dict(note=__doc__.split("\n\n")[0], roots=ROOTS, episodes_scanned=episodes, states=len(tgt),
               states_two_actions_played=len(rows))
    rng = np.random.default_rng(SEED)
    for s in S + ("any",):
        d, g, ps, ms, sing = [], [], [], [], []
        for r in rows:
            sz = sorted((x["size"] for x in r["groups"]), reverse=True)
            if sz[0] == sz[1]:
                continue                                   # no unique majority action
            maj = max(r["groups"], key=lambda x: x["size"])
            mine = [x for x in r["groups"] if x is not maj and x["n"] and (x["picked_by"] if s == "any" else s in x["picked_by"])]
            if not maj["n"] or not mine:
                continue
            p = np.mean([x["success"] / x["n"] for x in mine]); q = maj["success"] / maj["n"]
            d.append(p - q); g.append(r["task"]); ps.append(p); ms.append(q); sing.append(any(x["size"] == 1 for x in mine))
        tasks = sorted(set(g)); ix = {x: n for n, x in enumerate(tasks)}; gi = np.array([ix[x] for x in g]); d = np.array(d)
        sm, c = np.bincount(gi, d, len(tasks)), np.bincount(gi, minlength=len(tasks))
        bt = rng.integers(0, len(tasks), size=(B, len(tasks))); m = sm[bt].sum(1) / c[bt].sum(1)
        out[s] = dict(states=len(d), tasks=len(tasks), singleton_picks=int(sum(sing)),
                      minority_pick_success=round(100 * float(np.mean(ps)), 1), majority_success=round(100 * float(np.mean(ms)), 1),
                      minority_minus_majority=[round(100 * float(d.mean()), 1), *(round(100 * float(x), 1) for x in np.percentile(m, [2.5, 97.5]))],
                      better=int((d > 0).sum()), worse=int((d < 0).sum()), tied=int((d == 0).sum()))
    (HERE / "candidate-outcomes.json").write_text(json.dumps(out, indent=1))
    with open(Path(os.environ.get("OUT_DIR", RD)) / "candidate-outcomes.jsonl", "w") as fo:
        for r in rows:
            fo.write(json.dumps(r) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
