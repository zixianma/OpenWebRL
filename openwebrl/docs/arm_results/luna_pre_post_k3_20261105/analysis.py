#!/usr/bin/env python3
"""Pre-execution vs post-execution vs K = 3 lookahead teacher picks (GPT-6 Luna) on the same 1,738 SFT-action branch states.

Labels: runtime selection-head-20261008/sft-actions-20261021/labels/{pre,post,k3}/judgments/<state>.json. One judge
(gpt-6-luna, effort high), the same prompt and candidates; the three differ only in evidence: none (pre), the draw-0
immediate execution result (post), or that result plus the actor's next 3 steps (k3). Two calls per state per source,
each with its own random candidate order (teacher_repeat random0 / random1). Answers are an index only (no explanation).

Picks: each call's chosen distinct action. Agreement between sources compares call r of one source with call r of the
other (r = 0, 1) and averages; self-agreement compares a source's two calls. Action types: first parsed call of the
picked action. Outcomes (no label ever sees them): the fixed-five view of selection-label-kfold-20261101 (each state's
first five SFT proposals by draw index, duplicates kept; success on held-out draws 3-7, valid draws only); teacher =
argmax of the two-call vote share among the five (ties averaged); SFT sample = mean over the five; oracle = best
success on draws 0-2. Intervals: task-clustered percentile bootstrap (ratio of sums), 10,000 resamples, seed 20261105.

  analysis.py  (CPU only; writes stats.json next to this file)
"""
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

R = Path("/gpfs/scrubbed/zixianma/openwebrl-runtime/selection-head-20261008/sft-actions-20261021")
HERE = Path(__file__).resolve().parent
SOURCES = ("pre", "post", "k3")
PAIRS = (("pre", "post"), ("post", "k3"), ("pre", "k3"))
B, SEED = 10000, 20261105
TYPE = {"click": "click", "hover": "click", "scroll": "scroll", "go_back": "back", "done": "stop / answer",
        "write": "type", "press_keys": "type", "goto_url": "navigate (URL)", "new_tab": "new tab",
        "close_tab": "tab switch / close", "switch_tab": "tab switch / close", "wait": "wait"}
TYPES = ["click", "scroll", "type", "back", "stop / answer", "navigate (URL)", "new tab", "wait", "tab switch / close"]
DEPTH = (("0", 0, 0), ("1–5", 1, 5), ("6–11", 6, 11), ("12+", 12, 10**9))


def boot(vals, groups):
    """[mean, lo, hi] in pp; task-clustered percentile bootstrap, ratio of sums over resampled tasks."""
    tasks = sorted(set(groups)); idx = {t: i for i, t in enumerate(tasks)}
    s = np.zeros(len(tasks)); c = np.zeros(len(tasks))
    for v, g in zip(vals, groups):
        s[idx[g]] += v; c[idx[g]] += 1
    draw = np.random.default_rng(SEED).integers(0, len(tasks), size=(B, len(tasks)))
    m = s[draw].sum(1) / c[draw].sum(1)
    return [round(100 * float(np.mean(vals)), 2), *(round(100 * float(x), 2) for x in np.percentile(m, [2.5, 97.5]))]


def rate(action, draws):
    cells = [c for d in draws for c in action["outcomes"][str(d)] if c and c[0]]
    return sum(c[1] for c in cells) / len(cells) if cells else None


def fixed_five(x):
    return [a for _, a in sorted((d, a) for a, act in enumerate(x["actions"]) for d in act["sft_draws"])[:5]]


def kind(action):
    calls = json.loads(action["key"])
    return TYPE.get(calls[0][0], "other") if calls and calls[0] else "other"


def main():
    recs = {}
    for line in open(R / "actions.jsonl"):
        x = json.loads(line); recs[x["state_id"]] = x
    picks = {}                                                    # state -> source -> [action index per call]
    for sid, x in recs.items():
        where = {a["teacher_exec"]: i for i, a in enumerate(x["actions"])}
        got = {}
        for src in SOURCES:
            f = R / "labels" / src / "judgments" / f"{sid}.json"
            if not f.exists():
                break
            calls = sorted((j for j in json.loads(f.read_text()) if j.get("valid")), key=lambda j: j["teacher_repeat"])
            if len(calls) != 2:
                break
            got[src] = [where[j["candidate"]] for j in calls]
        if len(got) == len(SOURCES):
            picks[sid] = got
    ids = sorted(picks)
    task = {s: recs[s]["task_id"] for s in ids}
    n_act = {s: len(recs[s]["actions"]) for s in ids}
    depth = {s: recs[s]["depth"] for s in ids}
    out = dict(note=__doc__.split("\n\n")[0], states=len(ids), tasks=len(set(task.values())),
               bootstrap=dict(resamples=B, seed=SEED, cluster="task_id", interval="percentile 95%"))

    # Agreement (call-matched) and self-agreement, overall and by panel size / depth.
    def agree(sel):
        row = dict(states=len(sel))
        for a, b in PAIRS:
            row[f"{a}~{b}"] = round(100 * np.mean([np.mean([picks[s][a][r] == picks[s][b][r] for r in (0, 1)]) for s in sel]), 1)
        for a in SOURCES:
            row[f"{a}~{a}"] = round(100 * np.mean([picks[s][a][0] == picks[s][a][1] for s in sel]), 1)
        row["all three"] = round(100 * np.mean([np.mean([len({picks[s][k][r] for k in SOURCES}) == 1 for r in (0, 1)])
                                                for s in sel]), 1)
        row["chance pair"] = round(100 * np.mean([1 / n_act[s] for s in sel]), 1)
        row["chance three"] = round(100 * np.mean([1 / n_act[s] ** 2 for s in sel]), 1)
        return row
    out["agreement"] = dict(all=agree(ids), by_actions={str(k): agree([s for s in ids if n_act[s] == k]) for k in (2, 3, 4, 5)},
                            by_depth={n: agree([s for s in ids if lo <= depth[s] <= hi]) for n, lo, hi in DEPTH})

    # Action-type share of picks (both calls), all states and states where not all three sources agree.
    def types(sel):
        res = {}
        for src in SOURCES:
            c = Counter(kind(recs[s]["actions"][a]) for s in sel for a in picks[s][src])
            res[src] = {t: round(100 * c[t] / (2 * len(sel)), 1) for t in TYPES}
        c = Counter(kind(a) for s in sel for a in recs[s]["actions"])
        res["offered"] = {t: round(100 * c[t] / sum(c.values()), 1) for t in TYPES}
        return res
    split = [s for s in ids if any(len({picks[s][k][r] for k in SOURCES}) > 1 for r in (0, 1))]
    out["action_types"] = dict(all=types(ids), not_unanimous=dict(states=len(split), **types(split)))
    # Most common switches (call-matched) for each pair.
    sw = {}
    for a, b in PAIRS:
        c = Counter((kind(recs[s]["actions"][picks[s][a][r]]), kind(recs[s]["actions"][picks[s][b][r]]))
                    for s in ids for r in (0, 1) if picks[s][a][r] != picks[s][b][r])
        sw[f"{a}->{b}"] = [[f"{x} → {y}", n / 2] for (x, y), n in c.most_common(8)]
    out["switches"] = sw
    # Stop / answer share by depth (states offering a stop candidate are the denominator's subset shown separately).
    stop = {}
    for n, lo, hi in DEPTH:
        sel = [s for s in ids if lo <= depth[s] <= hi]
        offer = [s for s in sel if any(kind(a) == "stop / answer" for a in recs[s]["actions"])]
        row = dict(states=len(sel), offering_stop=len(offer))
        for src in SOURCES:
            row[src] = round(100 * np.mean([np.mean([kind(recs[s]["actions"][a]) == "stop / answer" for a in picks[s][src]])
                                            for s in sel]), 1)
            row[f"{src}_given_offered"] = round(100 * np.mean([np.mean([kind(recs[s]["actions"][a]) == "stop / answer"
                                                                        for a in picks[s][src]]) for s in offer]), 1) if offer else None
        stop[n] = row
    out["stop_by_depth"] = stop
    # Position bias: shown position of the pick (1-based) per source; picks are over different panel sizes.
    pos = {}
    for src in SOURCES:
        c = Counter()
        for s in ids:
            for j in json.loads((R / "labels" / src / "judgments" / f"{s}.json").read_text()):
                if j.get("valid"):
                    c[j["shown_position"]] += 1
        tot = sum(c.values()); pos[src] = {str(k): round(100 * c[k] / tot, 1) for k in sorted(c)}
    out["shown_position"] = pos

    # Outcomes on the fixed-five view, held-out draws 3-7.
    rows = {}
    for s in ids:
        x = recs[s]; five = fixed_five(x)
        r = [rate(a, range(3, 8)) for a in x["actions"]]
        if any(r[a] is None for a in five):
            continue
        early = [rate(a, range(3)) for a in x["actions"]]
        row = dict(sft=sum(r[a] for a in five) / len(five))
        for src in SOURCES:
            votes = Counter(picks[s][src]); best = max(votes[a] for a in five)
            tied = sorted({a for a in five if votes[a] == best})
            row[src] = sum(r[a] for a in tied) / len(tied); row[f"{src}_set"] = tied
        if all(early[a] is not None for a in five):
            best = max(early[a] for a in five); tied = sorted({a for a in five if early[a] == best})
            row["oracle"] = sum(r[a] for a in tied) / len(tied)
        row["best"] = max(r[a] for a in five)
        rows[s] = row
    sel = sorted(rows); g = [task[s] for s in sel]
    res = dict(states=len(sel), tasks=len(set(g)), sft_sample=round(100 * np.mean([rows[s]["sft"] for s in sel]), 2),
               best_in_five=round(100 * np.mean([rows[s]["best"] for s in sel]), 2))
    for src in (*SOURCES, "oracle"):
        have = [s for s in sel if src in rows[s]]
        res[src] = dict(success=round(100 * np.mean([rows[s][src] for s in have]), 2), states=len(have),
                        minus_sft=boot([rows[s][src] - rows[s]["sft"] for s in have], [task[s] for s in have]))
    # Head to head where the two sources' vote-argmax sets differ.
    h2h = {}
    for a, b in PAIRS:
        d = [s for s in sel if rows[s][f"{a}_set"] != rows[s][f"{b}_set"]]
        h2h[f"{b}-{a}"] = dict(states=len(d), tasks=len({task[s] for s in d}),
                               **{a: round(100 * np.mean([rows[s][a] for s in d]), 2), b: round(100 * np.mean([rows[s][b] for s in d]), 2)},
                               diff=boot([rows[s][b] - rows[s][a] for s in d], [task[s] for s in d]),
                               b_better=sum(rows[s][b] > rows[s][a] for s in d), a_better=sum(rows[s][a] > rows[s][b] for s in d))
    res["head_to_head"] = h2h
    out["outcomes"] = res
    (HERE / "stats.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
