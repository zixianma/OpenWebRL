"""Rule-based selectors over the actor's N proposals, for the random and action-distribution (majority-vote) baselines
(2026-11-16). Used by deferjudge/xbench_defer_rules.py through the cached-head selector path; no model, no features.

random    every candidate scores U(0, 1): a uniform pick among the N proposals.
majority  candidates are grouped into the same action: same tool name and the same non-coordinate arguments, and for
          coordinate arguments (point_2d, start/end points) every point within RADIUS of the group's first member on the
          actor's 0-1000 normalized grid; score = group size + U(0, 0.5), so the largest group wins and ties between equal
          groups (and between members of the winning group) are broken uniformly at random. Unparseable actions form
          singleton groups.
typeprior candidates score by the teacher's pick rate for their action type on the Recipe 3 training panels (prior passed in;
          smoothed toward 0.2) + U(0, 1e-3), so the type the teacher favours most wins and ties are uniform; no state input.
"""
import json
import random
import re

RADIUS = 30.0
_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
_rng = random.SystemRandom()


def parse_action(action_text):
    """(name, args) from the action part of a proposal, or None."""
    m = _CALL.search(action_text or "")
    if not m:
        return None
    try:
        call = json.loads(m.group(1))
    except json.JSONDecodeError:
        return None
    return call.get("name"), call.get("arguments") or {}


def _points(args):
    return {k: v for k, v in args.items() if isinstance(v, list) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v)}


def same_action(a, b, radius=RADIUS):
    if a is None or b is None or a[0] != b[0]:
        return False
    pa, pb = _points(a[1]), _points(b[1])
    if set(pa) != set(pb):
        return False
    if any(max(abs(pa[k][0] - pb[k][0]), abs(pa[k][1] - pb[k][1])) > radius for k in pa):
        return False
    rest = lambda args, pts: {k: v for k, v in args.items() if k not in pts}
    norm = lambda d: json.dumps({k: (v.strip().lower() if isinstance(v, str) else v) for k, v in d.items()}, sort_keys=True)
    return norm(rest(a[1], pa)) == norm(rest(b[1], pb))


def groups(actions, radius=RADIUS):
    """Group index per candidate (first-member anchored)."""
    gid, anchors = [], []
    for a in actions:
        for g, anchor in enumerate(anchors):
            if same_action(anchor, a, radius):
                gid.append(g); break
        else:
            anchors.append(a); gid.append(len(anchors) - 1)
    return gid


def scores(rule, action_texts, prior=None):
    if rule == "random":
        return [_rng.random() for _ in action_texts]
    if rule == "majority":
        gid = groups([parse_action(t) for t in action_texts])
        size = [gid.count(g) for g in gid]
        return [float(s) + 0.5 * _rng.random() for s in size]
    if rule == "typeprior":
        rate = prior or {}
        names = [(parse_action(t) or ("unparsed", {}))[0] for t in action_texts]
        return [float(rate.get(n, 0.2)) + 1e-3 * _rng.random() for n in names]
    raise ValueError(f"unknown rule {rule!r}")
