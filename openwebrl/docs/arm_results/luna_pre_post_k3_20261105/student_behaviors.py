#!/usr/bin/env python3
"""Pick behaviour of the four LoRA students on the replayed eval states (no new rollouts).

For each state with all four picks and two or more distinct candidate actions: how often each student picks the unique
most-proposed action, a singleton, a repeat of the previous executed action, a multi-call action, a navigation (URL / new
tab), go back, stop, or a scroll, each rate conditioned on that option being offered and not unanimous among the five.
`proposals` is the share of the five candidates in that option on the same states: what picking a proposal at random gives.

  student_behaviors.py  (CPU only; writes student-behaviors.json next to this file)
"""
from collections import defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0, "/gpfs/scrubbed/zixianma/openwebrl-runtime/selection-head-20261008/data-scale-20261023/source")
from openwebrl.selection_head import equivalent, parse_calls  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from student_live import groups  # noqa: E402
from student_outcomes import RD  # noqa: E402

HERE = Path(__file__).resolve().parent
S = ("luna-pre", "gpt55-pre", "luna-k3", "gpt55-k3")


def main():
    st = [json.loads(l) for l in open(RD / "states.jsonl")]
    pk = defaultdict(dict)
    for r in map(json.loads, open(RD / "picks.jsonl")):
        if r.get("student") and r["pick"] is not None:
            pk[r["i"]][r["student"]] = r["pick"]
    by = defaultdict(list)
    for i, p in pk.items():
        if len(p) == 4:
            by[st[i]["trace"]].append(i)
    feat = defaultdict(lambda: {s: [0, 0] for s in S}); base = defaultdict(float)
    for tr, ids in by.items():
        x = {s["turn"]: s for s in map(json.loads, open(tr))}
        for i in ids:
            t = st[i]["turn"]; c = [parse_calls(a.get("action") or "") for a in x[t]["candidates"]]
            g = groups(c)
            if len(g) < 2:
                continue
            sz = sorted((len(m) for _, m in g), reverse=True)
            prev = x.get(t - 1)
            pc = parse_calls(prev["candidates"][prev["selected_index"]]["action"] or "") if prev and prev["selected_index"] is not None else None
            has = lambda f: [k for k in range(5) if c[k] and f(c[k])]  # noqa: E731
            opts = {"go back": has(lambda q: any(z["name"] == "go_back" for z in q)),
                    "stop / answer": has(lambda q: any(z["name"] == "done" for z in q)),
                    "scroll only": has(lambda q: all(z["name"] == "scroll" for z in q)),
                    "open URL / new tab": has(lambda q: any(z["name"] in ("goto_url", "new_tab") for z in q)),
                    "multi-call action": has(lambda q: len(q) >= 2),
                    "repeat previous action": has(lambda q: pc is not None and equivalent(q, pc))}
            for s in S:
                j = pk[i][s]; mem = next(m for _, m in g if j in m)
                if sz[0] > sz[1]:
                    f = feat["unique most-proposed action"][s]; f[0] += len(mem) == sz[0]; f[1] += 1
                    base["unique most-proposed action"] += sz[0] / 20
                f = feat["singleton (1 of 5)"][s]; f[0] += len(mem) == 1; f[1] += 1
                base["singleton (1 of 5)"] += sum(len(m) == 1 for _, m in g) / 20
                for k, o in opts.items():
                    if o and len(o) < 5:
                        f = feat[k][s]; f[0] += j in o; f[1] += 1; base[k] += len(o) / 20
                    if k == "stop / answer" and o and len(o) < 5 and t >= 9:
                        f = feat["stop / answer, step >= 10"][s]; f[0] += j in o; f[1] += 1; base["stop / answer, step >= 10"] += len(o) / 20
    out = {k: {"proposals": round(100 * base[k] / max(v[S[0]][1], 1), 1), **{s: dict(rate=round(100 * a / max(b, 1), 1), n=b) for s, (a, b) in v.items()}}
           for k, v in feat.items()}
    (HERE / "student-behaviors.json").write_text(json.dumps(out, indent=1))
    print(f"{'':32s}{'proposals':>11s}" + "".join(f"{s:>11s}" for s in S) + "      n")
    for k, v in out.items():
        print(f"{k:32s}{v['proposals']:10.1f}%" + "".join(f"{v[s]['rate']:10.1f}%" for s in S) + f"  {v[S[0]]['n']:5d}")


if __name__ == "__main__":
    main()
