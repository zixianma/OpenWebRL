#!/usr/bin/env python3
"""Shortlist qualitative examples for the pre-exec / post-exec / K = 3 comparison (same 1,727 states as analysis.py).

A state qualifies for a category when each teacher's two calls agree with each other (a firm pick) and the picks split as
the category requires; candidates are ranked by the held-out success gap (draws 3-7, valid only, at least 4 valid draws
per action) between the better and worse pick. Categories:
  post_fixes_pre : pre picks X; post and K = 3 pick Y; Y succeeds more
  k3_fixes_post  : pre and post pick X; K = 3 picks Y; Y succeeds more
  k3_misled      : pre and post pick Y; K = 3 picks X; Y succeeds more
  pre_right      : pre picks Y; post and K = 3 pick X; Y succeeds more
Writes shortlist.json (runtime-only paths; not published).
"""
import json
from pathlib import Path

R = Path("/gpfs/scrubbed/zixianma/openwebrl-runtime/selection-head-20261008/sft-actions-20261021")
OUT = Path("/gpfs/scrubbed/zixianma/openwebrl-runtime/selection-head-20261008/label-kfold-20261101/pre-post-k3/shortlist.json")


def held(action):
    cells = [c for d in range(3, 8) for c in action["outcomes"][str(d)] if c and c[0]]
    return (sum(c[1] for c in cells), len(cells))


def main():
    cats = {k: [] for k in ("post_fixes_pre", "k3_fixes_post", "k3_misled", "pre_right")}
    for line in open(R / "actions.jsonl"):
        x = json.loads(line); sid = x["state_id"]
        where = {a["teacher_exec"]: i for i, a in enumerate(x["actions"])}
        p = {}
        for src in ("pre", "post", "k3"):
            f = R / "labels" / src / "judgments" / f"{sid}.json"
            if not f.exists():
                break
            calls = [where[j["candidate"]] for j in json.loads(f.read_text()) if j.get("valid")]
            if len(calls) != 2 or calls[0] != calls[1]:
                break
            p[src] = calls[0]
        if len(p) != 3:
            continue
        h = [held(a) for a in x["actions"]]
        if any(n < 4 for _, n in h):
            continue
        r = [s / n for s, n in h]
        pre, post, k3 = p["pre"], p["post"], p["k3"]
        spec = {"post_fixes_pre": (pre != post and post == k3, post, pre),
                "k3_fixes_post": (pre == post and k3 != post, k3, post),
                "k3_misled": (pre == post and k3 != post, post, k3),
                "pre_right": (pre != post and post == k3, pre, post)}
        for cat, (ok, good, bad) in spec.items():
            if ok and r[good] - r[bad] >= 0.4:
                cats[cat].append(dict(state=sid, task=x["task"].strip(), url=x["url"], depth=x["depth"], gap=round(r[good] - r[bad], 2),
                                      picks=p, held=h, keys=[a["key"] for a in x["actions"]],
                                      kinds=[json.loads(a["key"])[0][0] for a in x["actions"]], good=good, bad=bad))
    for v in cats.values():
        v.sort(key=lambda e: -e["gap"])
    OUT.write_text(json.dumps(cats, indent=1, ensure_ascii=False))
    for k, v in cats.items():
        print(k, len(v))
        for e in v[:12]:
            print(f"  {e['gap']:+.2f} {e['state']} d{e['depth']} {e['task'][:90]!r} good={e['keys'][e['good']][:70]} {e['held'][e['good']]} bad={e['keys'][e['bad']][:70]} {e['held'][e['bad']]}")


if __name__ == "__main__":
    main()
