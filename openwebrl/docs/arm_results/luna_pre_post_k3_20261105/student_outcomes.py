#!/usr/bin/env python3
"""Outcomes for both sides of a pre / K = 3 student split, from live eval episodes only (no new rollouts).

On a replayed eval state (student_replay.py) only the executed action was played out. But every selector evaluated on
November 1-4 ran the same SFT 4B actor on the same 1,045 tasks, so the same screen at the same step recurs across
selectors and runs, and different selectors executed different actions from it. A state is matched by identical
screenshot (sha256 of the logged PNG) on the same task at the same step; the executed action is compared by equivalence
(same calls, points within 5/1000). Each matched episode contributes its judged outcome (valid episodes only).

For each replayed state where the two students of one judge pick different actions (pre vs K = 3), this collects the
episodes that executed the pre pick and those that executed the K = 3 pick from that exact screen, with their outcomes,
then pools over states: success after the pre pick vs after the K = 3 pick on states where both were played out
(per-state success averaged, task-clustered bootstrap).

  student_outcomes.py  (CPU only; writes student-outcomes.json next to this file and per-state matches (both branches played out) to the runtime)
"""
from collections import defaultdict
import glob
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, "/gpfs/scrubbed/zixianma/openwebrl-runtime/selection-head-20261008/data-scale-20261023/source")
from openwebrl.selection_head import equivalent, parse_calls  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from student_live import kind  # noqa: E402

HERE = Path(__file__).resolve().parent
E = Path("/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations")
RD = Path("/gpfs/scrubbed/zixianma/openwebrl-runtime/selection-head-20261008/label-kfold-20261101/pre-post-k3/student-replay")
ROOTS = ("p16-lora-lunapre-20261103", "p16-lora-lunak3-20261103", "p16-heads-luna-20261103", "p16-heads-all-20261102",
         "p16-lora-pre-20261102", "p16-lora-20261101", "controls-sameday-20261103", "sameday-lora-20261104")
JUDGES = (("luna", "luna-pre", "luna-k3"), ("gpt55", "gpt55-pre", "gpt55-k3"))
B, SEED = 10000, 20261107


def main():
    states = [json.loads(l) for l in open(RD / "states.jsonl")]
    picks = defaultdict(dict)
    for r in map(json.loads, open(RD / "picks.jsonl")):
        if r.get("student") and r["pick"] is not None:
            picks[r["i"]][r["student"]] = r["pick"]
    # Targets: states where pre and K = 3 of at least one judge pick different actions.
    by_trace = defaultdict(list)
    for i, st in enumerate(states):
        if len(picks[i]) == 4:
            by_trace[st["trace"]].append(i)
    tgt = {}                                        # i -> dict(sha, task, bench, turn, cands, split per judge)
    for tr, ids in by_trace.items():
        steps = {s["turn"]: s for s in map(json.loads, open(tr))}
        for i in ids:
            st, p = states[i], picks[i]; s = steps[st["turn"]]
            cands = [parse_calls(c.get("action") or "") for c in s["candidates"]]
            same = lambda a, b: a == b or (cands[a] is not None and cands[b] is not None and equivalent(cands[a], cands[b]))  # noqa: E731
            sp = {j: (p[a], p[b]) for j, a, b in JUDGES if not same(p[a], p[b])}
            if sp:
                tgt[i] = dict(sha=s["screenshot_sha256"], task=st["task"], bench=st["bench"], turn=st["turn"], cands=cands,
                              split=sp, source=st["source"], run=st["run"], trace=tr)
    want = defaultdict(list)
    for i, t in tgt.items():
        want[(t["bench"], t["task"], t["turn"], t["sha"])].append(i)
    print("target states", len(tgt), flush=True)
    # One pass over every live episode of those days: executed action from each wanted screen.
    hits = defaultdict(list)                         # key -> [episode dicts]
    episodes = 0
    for root in ROOTS:
        for f in glob.glob(str(E / root / "run*" / "*" / "*" / "*" / "collection" / "records" / "*.json")):
            r = json.loads(Path(f).read_text()); episodes += 1
            tr = [t["path"] for t in r.get("trace_files") or [] if t["path"].endswith(".jsonl")]
            if not tr:
                continue
            bench, task = r["benchmark"], r["task_id"]
            for s in map(json.loads, open(tr[0])):
                k = (bench, task, s["turn"], s.get("screenshot_sha256"))
                if k in want and s["selected_index"] is not None:
                    hits[k].append(dict(root=root, run=f.split("/")[len(str(E).split("/")) + 1], slug=r["condition_slug"],
                                        valid=bool(r.get("valid")), reward=r.get("reward"), judge=r.get("judge_model"),
                                        action=s["candidates"][s["selected_index"]]["action"], trace=tr[0], record=f))
    print("episodes scanned", episodes, flush=True)
    rows, per_state = defaultdict(list), []
    for i, t in tgt.items():
        eps = hits[(t["bench"], t["task"], t["turn"], t["sha"])]
        for j, (a, b) in t["split"].items():
            side = {}
            for name, k in (("pre", a), ("k3", b)):
                ref = t["cands"][k]
                m = [e for e in eps if e["valid"] and ref is not None and (c := parse_calls(e["action"] or "")) is not None
                     and equivalent(ref, c)]
                side[name] = dict(n=len(m), success=sum(e["reward"] == 1 for e in m), kind=kind(ref), episodes=m)
            row = dict(i=i, judge=j, task=f"{t['bench']}/{t['task']}", bench=t["bench"], turn=t["turn"], source=t["source"],
                       matched_episodes=len(eps), pre=side["pre"], k3=side["k3"], trace=t["trace"])
            per_state.append(row)
            if side["pre"]["n"] and side["k3"]["n"]:
                rows[j].append(row)
    out = dict(note=__doc__.split("\n\n")[0], roots=ROOTS, episodes_scanned=episodes, split_states=len(tgt),
               judges_in_matches=sorted({e["judge"] or "" for v in hits.values() for e in v}))
    rng = np.random.default_rng(SEED)
    for j, rs in rows.items():
        d = np.array([r["k3"]["success"] / r["k3"]["n"] - r["pre"]["success"] / r["pre"]["n"] for r in rs])
        tasks = sorted({r["task"] for r in rs}); ix = {x: n for n, x in enumerate(tasks)}
        g = np.array([ix[r["task"]] for r in rs])
        s = np.bincount(g, d, len(tasks)); c = np.bincount(g, minlength=len(tasks))
        bt = rng.integers(0, len(tasks), size=(B, len(tasks)))
        m = s[bt].sum(1) / c[bt].sum(1)
        out[j] = dict(split_states=sum(r["judge"] == j for r in per_state), both_played=len(rs), tasks=len(tasks),
                      pre_success=round(100 * float(np.mean([r["pre"]["success"] / r["pre"]["n"] for r in rs])), 1),
                      k3_success=round(100 * float(np.mean([r["k3"]["success"] / r["k3"]["n"] for r in rs])), 1),
                      k3_minus_pre=[round(100 * float(d.mean()), 1), *(round(100 * float(x), 1) for x in np.percentile(m, [2.5, 97.5]))],
                      pre_episodes=sum(r["pre"]["n"] for r in rs), k3_episodes=sum(r["k3"]["n"] for r in rs))
    # States where both judges split the same way (both pre students on one action, both K = 3 students on another).
    cons = {json.loads(l)["i"] for l in open(RD / "label-split-examples.jsonl")}
    cr = {r["i"]: r for j in rows for r in rows[j] if r["i"] in cons}.values()
    out["consistent_split"] = dict(states=len(cons), both_played=len(cr),
                                   pre_success=round(100 * float(np.mean([r["pre"]["success"] / r["pre"]["n"] for r in cr])), 1),
                                   k3_success=round(100 * float(np.mean([r["k3"]["success"] / r["k3"]["n"] for r in cr])), 1))
    (HERE / "student-outcomes.json").write_text(json.dumps(out, indent=1))
    with open(RD / "split-outcomes.jsonl", "w") as fo:
        for r in per_state:                     # only states with both branches played out (keeps the file small)
            if r["pre"]["n"] and r["k3"]["n"]:
                for side in ("pre", "k3"):
                    r[side]["episodes"] = [{k: e[k] for k in ("root", "run", "slug", "reward", "trace")} for e in r[side]["episodes"]]
                fo.write(json.dumps(r) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
