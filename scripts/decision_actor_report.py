#!/usr/bin/env python3
"""Harness-iteration report for decision-actor runs (CPU, reads finished crossbench collections).

For each decision benchmark directory (``<root>/<decision slug>/<benchmark>``) it reports:
  * success: overall and valid-only, paired against a same-day control (``--control`` sft1 benchmark dir) on shared tasks;
  * projection coverage (projected_actor runs): share of steps whose first actor sample mapped to an option,
    the reason histogram for the rest, resample rescues and the escape (fallback) rate;
  * decision cost: options per step, element options per step, decision seconds, typing calls, browser steps.

  python scripts/decision_actor_report.py --dir R/decision-projected-sft-*/online_mind2web --control R/sft1/online_mind2web
"""
from __future__ import annotations

import argparse
from collections import Counter
import glob
import json
from pathlib import Path
import statistics


def records(bench_dir: Path) -> dict:
    plan = json.loads((bench_dir / "plan.json").read_text())
    out = {}
    for task_id in plan["task_ids"]:
        path = Path(plan["collection"]) / "records" / (plan["keys"][task_id] + ".json")
        if path.exists():
            out[task_id] = json.loads(path.read_text())
    return dict(plan=plan, rows=out)


def turns(bench_dir: Path) -> list[dict]:
    plan = json.loads((bench_dir / "plan.json").read_text())
    files = glob.glob(str(Path(plan["collection"]) / "attempts" / "*" / "*" / "selections" / "turns" / "*" / "turn-*.json"))
    return [json.loads(Path(f).read_text()) for f in sorted(files)]


def rate(rows):
    valid = [r for r in rows if r.get("valid")]
    wins = sum(r.get("reward") == 1 for r in valid)
    return dict(tasks=len(rows), valid=len(valid), successes=wins,
                overall=wins / len(rows) if rows else None, valid_only=wins / len(valid) if valid else None)


def coverage(steps):
    decided = [s for s in steps if s.get("decision_info", {}).get("attempts")]
    if not decided:
        return None
    first = Counter(s["decision_info"]["attempts"][0]["reason"] for s in decided)
    mapped_first = sum(s["decision_info"]["attempts"][0]["option_id"] is not None or
                       s["decision_info"]["attempts"][0]["reason"] in ("done", "write_after_type") for s in decided)
    rescued = sum(len(s["decision_info"]["attempts"]) > 1 and s["decision_info"].get("mapped") for s in decided)
    escaped = sum(not s["decision_info"].get("mapped", True) for s in decided)
    return dict(steps=len(decided), first_sample_mapped=mapped_first / len(decided), rescued_by_resample=rescued,
                escaped=escaped, escape_rate=escaped / len(decided), first_sample_reasons=dict(first.most_common()))


def cost(steps):
    decided = [s for s in steps if s.get("n_options")]
    secs = [s["decision_info"]["seconds"] for s in decided if (s.get("decision_info") or {}).get("seconds") is not None]
    return dict(browser_steps=len(steps), decision_steps=len(decided),
                options_per_step=statistics.mean(s["n_options"] for s in decided) if decided else None,
                elements_per_step=statistics.mean(s["n_elements"] for s in decided) if decided else None,
                decision_seconds_mean=statistics.mean(secs) if secs else None,
                typing_calls=sum(1 for s in steps if s.get("typing")),
                step_seconds_mean=statistics.mean(s["seconds"] for s in steps) if steps else None,
                action_kinds=dict(Counter((s.get("chosen") or {}).get("kind") or s.get("source") for s in steps)))


def paired(a: dict, b: dict):
    shared = sorted(set(a) & set(b))
    diff = [int(a[t].get("valid") and a[t].get("reward") == 1) - int(b[t].get("valid") and b[t].get("reward") == 1)
            for t in shared]
    return dict(shared_tasks=len(shared), overall_diff_pp=100 * statistics.mean(diff) if diff else None)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", nargs="+", required=True, help="decision benchmark directories")
    p.add_argument("--control", help="same-day control benchmark directory (e.g. sft1)")
    a = p.parse_args()
    control = records(Path(a.control)) if a.control else None
    report = {}
    for d in a.dir:
        bench = Path(d)
        rec = records(bench)
        steps = turns(bench)
        entry = dict(condition=rec["plan"]["condition"]["slug"], success=rate(list(rec["rows"].values())),
                     coverage=coverage(steps), cost=cost(steps))
        if control:
            entry["control"] = dict(condition=control["plan"]["condition"]["slug"],
                                    success=rate(list(control["rows"].values())), **paired(rec["rows"], control["rows"]))
        report[str(bench)] = entry
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
