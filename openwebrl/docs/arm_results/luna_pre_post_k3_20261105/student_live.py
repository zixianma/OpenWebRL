#!/usr/bin/env python3
"""What the trained students (ADM-4B LoRAs, Recipe 3) pick live: pre-execution vs K = 3 lookahead labels, GPT-5.5 and Luna.

Every live decision of the four LoRAs on the SFT 4B actor (3 runs x 1,045 tasks: OM2W 300, WebVoyager 595, DeepShop 150;
5 candidates at T 1.0 / top-p 0.9, 30 steps) is read from the saved selection traces. Pairs compared on the same day:
Luna pre vs Luna K = 3 (November 3) and GPT-5.5 pre vs GPT-5.5 K = 3 (November 4 same-day rerun).

Per decision: the five candidates' parsed tool calls are grouped by equivalence (same calls, points within 5/1000, as in
selection_head.equivalent); a decision is a real choice when at least two groups exist. Recorded: the picked action's type,
its group size (how many of the five proposed it), whether it is the plurality action, whether it repeats the previous
executed action, and whether a stop was offered / picked. Per episode: valid-only success, steps, step-limit endings,
and loops (the same action executed three times in a row). Live trajectories diverge, so these are behaviour rates on
each student's own states, not same-state comparisons (see student_offline.py for those).

  student_live.py  (CPU only; writes student-live.json next to this file and the per-decision index to the runtime)
"""
from collections import Counter, defaultdict
import glob
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, "/gpfs/scrubbed/zixianma/openwebrl-runtime/selection-head-20261008/data-scale-20261023/source")
from openwebrl.selection_head import equivalent, parse_calls  # noqa: E402

E = Path("/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations")
HERE = Path(__file__).resolve().parent
INDEX = Path("/gpfs/scrubbed/zixianma/openwebrl-runtime/selection-head-20261008/label-kfold-20261101/pre-post-k3/student-live-index.jsonl")
BENCH = ("online_mind2web", "webvoyager", "deepshop")
STUDENTS = {                      # name -> (day, root, condition slug)
    "luna-pre": ("2026-11-03", "p16-lora-lunapre-20261103", "arm5-3b148a583f85"),
    "luna-k3": ("2026-11-03", "p16-lora-lunak3-20261103", "arm5-2ed62dd7c4dc"),
    "gpt55-pre": ("2026-11-04", "sameday-lora-20261104", "arm5-80ee47643be2"),
    "gpt55-k3": ("2026-11-04", "sameday-lora-20261104", "arm5-9fadc8dc4655"),
}
TYPE = {"click": "click", "hover": "click", "scroll": "scroll", "go_back": "back", "done": "stop / answer",
        "write": "type", "press_keys": "type", "goto_url": "navigate (URL)", "new_tab": "tab", "close_tab": "tab",
        "switch_tab": "tab", "wait": "wait"}
TYPES = ["click", "scroll", "type", "back", "stop / answer", "navigate (URL)", "tab", "wait", "other"]


def kind(calls):
    return TYPE.get(calls[0]["name"], "other") if calls else "other"


def groups(calls):
    out = []                                       # list of [representative calls, member indices]
    for i, c in enumerate(calls):
        for g in out:
            if c is not None and equivalent(g[0], c):
                g[1].append(i); break
        else:
            out.append([c, [i]])
    return out


def episodes(name):
    day, root, slug = STUDENTS[name]
    for run in ("run1", "run2", "run3"):
        for b in BENCH:
            for f in sorted(glob.glob(str(E / root / run / b / slug / b / "collection" / "records" / "*.json"))):
                r = json.loads(Path(f).read_text())
                tr = [t["path"] for t in r.get("trace_files") or [] if t["path"].endswith(".jsonl")]
                yield run, b, r, tr[0] if tr else None


def main():
    INDEX.parent.mkdir(parents=True, exist_ok=True)
    stats = {}
    with open(INDEX, "w") as idx:
        for name in STUDENTS:
            dec, eps = [], []
            for run, b, r, tr in episodes(name):
                steps = [json.loads(l) for l in open(tr)] if tr else []
                prev, streak, loop = None, 1, False
                for s in steps:
                    cands = [parse_calls(c.get("action") or "") for c in s["candidates"]]
                    gs = groups(cands); k = s["selected_index"]
                    pick = cands[k] if k is not None and 0 <= k < len(cands) else None
                    g = next(g for g in gs if k in g[1]) if pick is not None else None
                    top = max(len(x[1]) for x in gs)
                    rep = prev is not None and pick is not None and equivalent(prev, pick)
                    streak = streak + 1 if rep else 1; loop |= streak >= 3
                    d = dict(student=name, run=run, bench=b, task=r["task_id"], turn=s["turn"], url=s.get("url"),
                             n_groups=len(gs), pick_kind=kind(pick), pick_group=len(g[1]) if g else 0,
                             plurality=bool(g) and len(g[1]) == top, unique_plurality=sum(len(x[1]) == top for x in gs) == 1,
                             repeat=rep, stop_offered=any(kind(c) == "stop / answer" for c in cands),
                             offered_kinds=[kind(c) for c in cands], first=k == 0, trace=tr)
                    dec.append(d); idx.write(json.dumps(d) + "\n"); prev = pick
                eps.append(dict(run=run, bench=b, task=r["task_id"], valid=bool(r.get("valid")), reward=r.get("reward"),
                                steps=len(steps), end=r.get("terminate_reason"), loop=loop,
                                stopped=bool(steps) and dec[-1]["pick_kind"] == "stop / answer"))
            real = [d for d in dec if d["n_groups"] >= 2]
            v = [e for e in eps if e["valid"]]
            row = dict(day=STUDENTS[name][0], episodes=len(eps), valid=len(v), decisions=len(dec), real_choices=len(real),
                       success_valid=round(100 * np.mean([e["reward"] == 1 for e in v]), 1),
                       success_by_bench={b: round(100 * np.mean([e["reward"] == 1 for e in v if e["bench"] == b]), 1) for b in BENCH},
                       steps_mean=round(float(np.mean([e["steps"] for e in v])), 2),
                       step_limit=round(100 * np.mean([e["end"] not in ("task_completed",) and not e["stopped"] for e in v]), 1),
                       loop_episodes=round(100 * np.mean([e["loop"] for e in v]), 1),
                       ends={k: c for k, c in Counter(e["end"] for e in v).most_common(6)})
            row["pick_types_real"] = {t: round(100 * np.mean([d["pick_kind"] == t for d in real]), 1) for t in TYPES}
            row["offered_types_real"] = {t: round(100 * np.mean([c == t for d in real for c in d["offered_kinds"]]), 1) for t in TYPES}
            row["plurality_real"] = round(100 * np.mean([d["plurality"] for d in real]), 1)
            row["minority_real"] = round(100 * np.mean([d["pick_group"] == 1 for d in real]), 1)
            row["first_candidate_real"] = round(100 * np.mean([d["first"] for d in real]), 1)
            row["repeat_all"] = round(100 * np.mean([d["repeat"] for d in dec]), 1)
            row["repeat_when_alternative"] = round(100 * np.mean([d["repeat"] for d in real]), 1)
            so = [d for d in dec if d["stop_offered"]]
            row["stop_given_offered"] = round(100 * np.mean([d["pick_kind"] == "stop / answer" for d in so]), 1)
            row["stop_given_offered_minority"] = round(100 * np.mean(
                [d["pick_kind"] == "stop / answer" for d in so if d["offered_kinds"].count("stop / answer") <= 2]), 1)
            by_turn = defaultdict(list)
            for d in so:
                by_turn[min(d["turn"] // 5, 4)].append(d["pick_kind"] == "stop / answer")
            row["stop_given_offered_by_turn"] = {f"{5 * k}-{5 * k + 4}" if k < 4 else "20+": round(100 * np.mean(x), 1)
                                                 for k, x in sorted(by_turn.items())}
            stats[name] = row
            print(name, json.dumps(row)[:600], flush=True)
    (HERE / "student-live.json").write_text(json.dumps(dict(note=__doc__.split("\n\n")[0], students=stats), indent=1))


if __name__ == "__main__":
    main()
