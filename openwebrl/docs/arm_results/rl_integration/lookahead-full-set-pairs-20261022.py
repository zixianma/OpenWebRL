"""2026-10-22 copy of score_full_arbiter_k1.py: adds minus_uniform and post_minus_pre; otherwise unchanged."""
"""Main tables on all labelled states (736 grid + 435 new): every teacher condition and prior-work formulation, held-out
draws 3-7, valid draws only, under the o4-mini judge and the GPT-6 Luna re-judge. Views: usable informative states (some
valid outcome differs), all usable states, and the Piotr-task subset. Ties count as an even split."""
import json
import re
import sqlite3
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, "/gpfs/projects/krishna/zixianma/OpenWebRL"); sys.path.insert(0, "/gpfs/projects/krishna/zixianma/OpenWebRL/scripts")
from adapt_selection_head_outcomes import LEDGERS  # noqa: E402
from selection_outcome_experiments import cluster_boot, read  # noqa: E402
from openwebrl.eval.reward_online_mind2web import _parse_verdict  # noqa: E402

F = Path(__file__).resolve().parent; B = F.parent; G = B / "lookahead-repr-20261015"; N = B / "lookahead-new-20261016"
LIN = Path("/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-turn-bonus-preparation/branch-selector-scaleup-20261008/parallel-fresh64-preparation-20261010/piotr-lineage")
piotr = set(read(LIN / "task-membership.private.json")["piotr_training_task_counts"])
norm = lambda s: " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", s).casefold()))


def label_picks(roots):
    out = defaultdict(list)
    for root in roots:
        for f in (root / "judgments").glob("*.json"):
            for j in json.loads(f.read_text()):
                if j["valid"]:
                    out[j["state"]].append(j["candidate"])
    return out


picks = {"Pre-execution": label_picks([G / "s-pre", N / "s-pre"]), "Post-execution": label_picks([G / "s-post", N / "s-post"]),
         "Lookahead K=1": label_picks([G / "s-k1", N / "s-k1"]), "Lookahead K=3": label_picks([G / "s-k3", N / "s-k3b"])}
for m, name in (("arbiter", "WebArbiter formulation"), ("shepherd", "WebShepherd formulation"), ("procua", "PRO-CUA formulation")):
    picks[name] = {}
    for root in (B / "prm-formulations-20261016", N / "prm-formulations"):
        for f in (root / "results" / m).glob("*.json"):
            r = read(f); picks[name][r["state"]] = r["picks"]
picks["WebArbiter + lookahead K=1"] = {}
for f in (B / "prm-arbiter-k1-20261018" / "results" / "arbiter_k1").glob("*.json"):
    r = read(f)
    if r["picks"]:
        picks["WebArbiter + lookahead K=1"][r["state"]] = r["picks"]
luna = {}
for root in (B / "rejudge-luna-all-20261016", B / "rejudge-luna-rest2-20261017", N / "rejudge-luna"):
    for f in (root / "verdicts").glob("*.json"):
        v = read(f)
        if v.get("second_text"):
            luna[(v["state"], v["candidate"], v["repeat"])] = _parse_verdict(v["second_text"])
ents = []
for base, jd in (("lookahead-fast1k-20261013/k3", None), ("lookahead-v2-20261013/k3", None), ("lookahead-new-20261016/s-k3b", None)):
    for e in read(B / base / "plan.json")["states"]:
        j = B / base / "judgments" / f"{e['state']}.json"
        if j.exists() and read(j):
            ents.append(e)
slot, task, is_piotr, page, site_of = defaultdict(dict), {}, {}, {}, {}


def cap_site(states, site, share, seed=20261017):
    """Downsample one website to the given share of the result, round-robin across its tasks (seeded); others untouched."""
    import random
    others = [s for s in states if site_of[s] != site]
    target = int(round(share * len(others) / (1 - share)))
    by_task = defaultdict(list)
    for s in states:
        if site_of[s] == site:
            by_task[task[s]].append(s)
    rng = random.Random(seed)
    for v in by_task.values():
        rng.shuffle(v)
    order = sorted(by_task, key=lambda t: rng.random())
    kept = []
    while len(kept) < target and any(by_task.values()):
        for t in order:
            if by_task[t] and len(kept) < target:
                kept.append(by_task[t].pop())
    return sorted(others + kept)
for root in {str(p) for p in LEDGERS} | {e["root"] for e in ents}:
    f = Path(root) / "state.sqlite3"
    if f.exists():
        for (raw,) in sqlite3.connect(f"file:{f}?mode=ro", uri=True).execute("SELECT result_json FROM attempts WHERE status='finished'"):
            x = json.loads(raw)
            if x.get("mode") == "branch" and x.get("released"):
                v = bool(x["valid"])
                slot[x["state"]].setdefault((x["candidate"], x["repeat"]), (v, v and (x.get("reward") or 0) >= 1, bool(x.get("judge_http_attempts"))))
for e in ents:
    a = read(read(Path(e["root"]) / "states" / e["state"] / "anchor-reference.json")["path"])
    task[e["state"]] = a["task_id"]; is_piotr[e["state"]] = norm(a["task"] if isinstance(a["task"], str) else json.dumps(a["task"])) in piotr
    o = a["before_observation"]; url = o.get("active_tab_url") or ""; sp = o.get("selection_page") or {}
    blank = (url == "" or url.startswith("about:blank")) and not (sp.get("interactive_elements") or []) and not (sp.get("text") or "").strip()
    page[e["state"]] = "blank" if blank else "error" if url.startswith(("chrome-error", "chrome://")) else "page"
    site_of[e["state"]] = re.sub(r"^(www\d?|m)\.", "", (url.split("/")[2] if url.count("/") >= 2 else "").lower())
states = [e["state"] for e in ents if all(e["state"] in p for p in picks.values())]
out = {"labelled": len(ents), "with_all_conditions": len(states), "luna_coverage_missing": 0}


MISSING = defaultdict(int)


def held_out(s, judge):
    """Per-action held-out success; under Luna every judged continuation must have a Luna verdict (no o4-mini fallback)."""
    hold = []
    for i in range(5):
        cells = []
        for k in range(3, 8):
            if (i, k) in slot[s] and slot[s][i, k][0]:
                succ = slot[s][i, k][1]
                if judge == "luna" and slot[s][i, k][2]:
                    if luna.get((s, i, k)) is None:
                        MISSING[s] += 1
                        return None
                    succ = bool(luna[(s, i, k)])
                cells.append(succ)
        hold.append(float(np.mean(cells)) if cells else None)
    return None if None in hold else np.array(hold)


for judge in ("o4-mini", "luna"):
    per, informative = {}, set()
    for s in states:
        h = held_out(s, judge)
        if h is None:
            continue
        if any(s not in picks[m] for m in picks):
            continue
        per[s] = {m: float(np.mean([h[c] for c in picks[m][s]])) for m in picks} | {"Uniform": float(h.mean()), "Actor's first sample": float(h[0])}
        vals = [slot[s][i, k][1] for i in range(5) for k in range(8) if (i, k) in slot[s] and slot[s][i, k][0]]
        if len(set(vals)) > 1:
            informative.add(s)
    # rule fixed before seeing final results (2026-10-17): completely blank pages are environment artefacts, removed
    diverse = {r["state"] for r in json.loads((F / "action-diversity.json").read_text())["rows"] if r["effective"] >= 2}
    main = sorted(s for s in per if page[s] != "blank")
    # arXiv capped at 10% of the view (rule fixed 2026-10-17, before final results)
    views = {"main (blank pages removed)": main, "2+ different actions": [s for s in main if s in diverse],
             "without browser error pages": [s for s in main if page[s] != "error"], "arXiv capped at 10%": cap_site(main, "arxiv.org", 0.10),
             "all usable (incl. blank)": sorted(per)}
    out.setdefault("view_composition", {})[judge] = {k: dict(states=len(v), arxiv=sum(site_of[s] == "arxiv.org" for s in v)) for k, v in views.items()}
    out[judge] = {}
    for vname, st in views.items():
        g = [task[s] for s in st]
        out[judge][vname] = dict(states=len(st), tasks=len(set(g)),
                                 success={m: round(100 * np.mean([per[s][m] for s in st]), 2) for m in per[st[0]]},
                                 minus_k3={m: cluster_boot([per[s][m] - per[s]["Lookahead K=3"] for s in st], g) for m in per[st[0]] if m != "Lookahead K=3"},
                                 arbiter_k1_minus={m: cluster_boot([per[s]["WebArbiter + lookahead K=1"] - per[s][m] for s in st], g)
                                                   for m in ("WebArbiter formulation", "Lookahead K=1", "Lookahead K=3", "Post-execution", "Uniform")},
                                 minus_uniform={m: cluster_boot([per[s][m] - per[s]["Uniform"] for s in st], g) for m in per[st[0]] if m != "Uniform"},
                                 post_minus_pre=cluster_boot([per[s]["Post-execution"] - per[s]["Pre-execution"] for s in st], g))
out["luna_states_missing_verdicts"] = len(MISSING)
(F / "pairs-20261022.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if k in ("labelled", "with_all_conditions")}))
