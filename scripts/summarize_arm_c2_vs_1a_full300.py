#!/usr/bin/env python3
"""Combine fresh holdout-200 and existing fixed-100 C2/1A evaluations."""

import argparse
import collections
import json
from pathlib import Path
import statistics

from summarize_arm_c2_full300 import load_records, paired_report, percent, interval, rate_report
from summarize_arm_policy_behavior import actions


RUNS = Path("/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/runs")
LABELS = ("c2-update-000500", "1a-endpoint-000263")
FIXED_100 = {
    LABELS[0]: RUNS / "c2-full-282782-20260908T075414Z/evaluation/checkpoint-scaling-100/update-000500/online-mind2web",
    LABELS[1]: RUNS / "c2-ablation-1a/evaluation/checkpoint-scaling-100/endpoint-000263/online-mind2web",
}


def load_complete(directory: Path, task_ids: list[str]) -> dict:
    records = load_records(directory, task_ids)
    manifest = json.loads((directory / "manifest.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    if set(records) != set(task_ids) or manifest.get("task_ids") != task_ids:
        raise ValueError(f"Evaluation does not exactly match its frozen cohort: {directory}")
    if summary.get("scheduled") != len(task_ids) or summary.get("attempted") != len(task_ids):
        raise ValueError(f"Evaluation summary is incomplete: {directory}")
    return records


def behavior_from_records(records: dict, task_ids: list[str]) -> dict:
    rows = [records[task_id] for task_id in task_ids if task_id in records]
    valid = [row for row in rows if row.get("valid")]
    calls = collections.Counter()
    steps = []
    terminated = capped = primary_pairs = repeated_primary = 0
    for row in rows:
        turns = actions(row)
        names = [name for turn in turns for name in turn]
        calls.update(names)
        primaries = [turn[0] for turn in turns if turn]
        primary_pairs += max(0, len(primaries) - 1)
        repeated_primary += sum(left == right for left, right in zip(primaries, primaries[1:]))
        if row.get("total_steps") is not None:
            steps.append(row["total_steps"])
            terminated += "done" in names
            capped += row["total_steps"] >= 30
    successes = sum(row.get("reward") == 1 for row in valid)
    total_calls = sum(calls.values())
    return {
        "scheduled": len(task_ids),
        "attempted": len(rows),
        "valid": len(valid),
        "successes": successes,
        "mean_steps": statistics.mean(steps) if steps else None,
        "median_steps": statistics.median(steps) if steps else None,
        "termination_rate": terminated / len(steps) if steps else None,
        "step_cap_30_rate": capped / len(steps) if steps else None,
        "scroll_call_fraction": calls["scroll"] / total_calls if total_calls else None,
        "repeated_primary_action_rate": repeated_primary / primary_pairs if primary_pairs else None,
        "tool_call_counts": dict(calls.most_common()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", type=Path, required=True)
    parser.add_argument("--cohorts", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.cohorts.read_text())
    cohorts = payload["cohorts"]
    all_ids = cohorts["all_300"]["task_ids"]
    selected_ids = cohorts["checkpoint_selection_100"]["task_ids"]
    holdout_ids = cohorts["checkpoint_selection_holdout_200"]["task_ids"]
    if len(all_ids) != 300 or len(set(all_ids)) != 300:
        raise ValueError("The full cohort must contain 300 unique task IDs")
    if set(selected_ids) & set(holdout_ids) or set(selected_ids) | set(holdout_ids) != set(all_ids):
        raise ValueError("The fixed 100 and holdout 200 must partition all 300 tasks")

    fresh = {label: args.group / label / "online-mind2web" for label in LABELS}
    records = {}
    for label in LABELS:
        holdout_records = load_complete(fresh[label], holdout_ids)
        selected_records = load_complete(FIXED_100[label], selected_ids)
        records[label] = {**holdout_records, **selected_records}
        if len(records[label]) != 300:
            raise ValueError(f"Combined result for {label} does not contain 300 unique tasks")

    reports = {}
    for cohort_name, cohort in cohorts.items():
        task_ids = cohort["task_ids"]
        reports[cohort_name] = {
            "rates": {label: rate_report(rows, task_ids) for label, rows in records.items()},
            "paired_common_valid": paired_report(records[LABELS[0]], records[LABELS[1]], task_ids),
            "behavior": {label: behavior_from_records(rows, task_ids) for label, rows in records.items()},
        }
    report = {
        "cohort_manifest": str(args.cohorts.resolve()),
        "task_file_sha256": payload["task_file_sha256"],
        "fresh_holdout_200_runs": {label: str(path.resolve()) for label, path in fresh.items()},
        "existing_fixed_100_runs": {label: str(path.resolve()) for label, path in FIXED_100.items()},
        "cohorts": reports,
        "notes": [
            "The fresh 200-task holdout is the primary comparison because the fixed 100 informed checkpoint selection.",
            "Both policies run concurrently on the fresh holdout, controlling live-site timing within dispatch waves.",
            "The all-300 aggregate combines the fresh holdout with earlier fixed-100 evaluations collected at different times.",
            "McNemar comparisons include only tasks valid in both runs; overall rates retain every scheduled task.",
            "Unavailable-task retries, if any, must be reported separately.",
        ],
    }
    output_json = args.group / "comparison.json"
    output_json.write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# C2 update 500 vs 1A endpoint: Online-Mind2Web evaluation", ""]
    for cohort_name in ("checkpoint_selection_holdout_200", "all_300", "checkpoint_selection_100"):
        title = cohorts[cohort_name]["description"]
        rows = reports[cohort_name]
        lines += [
            f"## {title}",
            "",
            "| Policy | Overall (95% Wilson CI) | Valid-only (95% Wilson CI) | Unavailable / missing |",
            "| --- | ---: | ---: | ---: |",
        ]
        for label, row in rows["rates"].items():
            lines.append(
                f"| {label} | {row['successes']}/{row['scheduled']} = {percent(row['overall'])} "
                f"({interval(row['overall_wilson_95'])}) | {row['successes']}/{row['valid']} = "
                f"{percent(row['valid_only'])} ({interval(row['valid_wilson_95'])}) | "
                f"{row['unavailable']} / {row['missing']} |"
            )
        paired = rows["paired_common_valid"]
        lines += [
            "",
            f"1A has {paired['right_wins']} paired wins and {paired['right_losses']} paired losses "
            f"over {paired['common_valid']} common-valid tasks; exact McNemar "
            f"p={paired['mcnemar_exact_two_sided_p']:.4g}.",
            "",
        ]
    lines += ["## Interpretation constraints", "", *[f"- {note}" for note in report["notes"]], ""]
    (args.group / "comparison.md").write_text("\n".join(lines))
    print(args.group / "comparison.md")


if __name__ == "__main__":
    main()
