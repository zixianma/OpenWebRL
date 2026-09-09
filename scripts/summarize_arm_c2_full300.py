#!/usr/bin/env python3
"""Summarize the C2 update-500/update-700 full Online-Mind2Web evaluation."""

import argparse
import json
import math
from pathlib import Path

from summarize_arm_policy_behavior import behavior


def load_records(directory, task_ids):
    rows = [json.loads(path.read_text()) for path in sorted((directory / "results").glob("*.json"))]
    records = {row["task_id"]: row for row in rows}
    if len(records) != len(rows):
        raise ValueError(f"Duplicate task results in {directory}")
    unexpected = set(records) - set(task_ids)
    if unexpected:
        raise ValueError(f"Unexpected task IDs in {directory}: {sorted(unexpected)[:3]}")
    return records


def wilson(successes, total, z=1.959963984540054):
    if not total:
        return None
    rate = successes / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return [center - radius, center + radius]


def rate_report(records, task_ids):
    rows = [records[task_id] for task_id in task_ids if task_id in records]
    valid = [row for row in rows if row.get("valid")]
    successes = sum(row.get("reward") == 1 for row in valid)
    return {
        "scheduled": len(task_ids),
        "attempted": len(rows),
        "valid": len(valid),
        "unavailable": len(rows) - len(valid),
        "missing": len(task_ids) - len(rows),
        "successes": successes,
        "overall": successes / len(task_ids) if task_ids else None,
        "overall_wilson_95": wilson(successes, len(task_ids)),
        "valid_only": successes / len(valid) if valid else None,
        "valid_wilson_95": wilson(successes, len(valid)),
    }


def mcnemar_exact(wins, losses):
    discordant = wins + losses
    if not discordant:
        return 1.0
    smaller = min(wins, losses)
    return min(1.0, 2 * sum(math.comb(discordant, k) for k in range(smaller + 1)) / 2**discordant)


def paired_report(left, right, task_ids):
    common = [task_id for task_id in task_ids
              if task_id in left and task_id in right
              and left[task_id].get("valid") and right[task_id].get("valid")]
    right_wins = sum(left[t].get("reward") != 1 and right[t].get("reward") == 1 for t in common)
    right_losses = sum(left[t].get("reward") == 1 and right[t].get("reward") != 1 for t in common)
    return {
        "common_valid": len(common),
        "right_wins": right_wins,
        "right_losses": right_losses,
        "net_wins": right_wins - right_losses,
        "mcnemar_exact_two_sided_p": mcnemar_exact(right_wins, right_losses),
    }


def percent(value):
    return "n/a" if value is None else f"{100 * value:.1f}%"


def interval(value):
    return "n/a" if value is None else f"{100 * value[0]:.1f}–{100 * value[1]:.1f}%"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--cohorts", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.cohorts.read_text())
    cohorts = payload["cohorts"]
    all_ids = cohorts["all_300"]["task_ids"]
    if len(all_ids) != 300 or len(set(all_ids)) != 300:
        raise ValueError("The full cohort must contain 300 unique task IDs")
    if set(cohorts["checkpoint_selection_100"]["task_ids"]) & set(cohorts["checkpoint_selection_holdout_200"]["task_ids"]):
        raise ValueError("Selection and holdout strata overlap")
    if set(cohorts["checkpoint_selection_100"]["task_ids"]) | set(cohorts["checkpoint_selection_holdout_200"]["task_ids"]) != set(all_ids):
        raise ValueError("Selection and holdout strata do not partition all 300 tasks")

    group = args.run_root / "evaluation/checkpoint-full-300"
    directories = {
        "historical-base": args.baseline,
        "update-000500": group / "update-000500/online-mind2web",
        "update-000700": group / "update-000700/online-mind2web",
    }
    records = {label: load_records(directory, all_ids) for label, directory in directories.items()}
    reports = {}
    for cohort_name, cohort in cohorts.items():
        task_ids = cohort["task_ids"]
        reports[cohort_name] = {
            "rates": {label: rate_report(rows, task_ids) for label, rows in records.items()},
            "paired_common_valid": {
                "update-000500_vs_historical-base": paired_report(records["historical-base"], records["update-000500"], task_ids),
                "update-000700_vs_historical-base": paired_report(records["historical-base"], records["update-000700"], task_ids),
                "update-000700_vs_update-000500": paired_report(records["update-000500"], records["update-000700"], task_ids),
            },
            "behavior": {label: behavior(directory, task_ids) for label, directory in directories.items()},
        }
    report = {
        "cohort_manifest": str(args.cohorts.resolve()),
        "task_file_sha256": payload["task_file_sha256"],
        "runs": {label: str(path.resolve()) for label, path in directories.items()},
        "cohorts": reports,
        "notes": [
            "The 200-task holdout was not used to choose update 500 or update 700; it is the primary checkpoint-comparison stratum.",
            "The 100-task stratum repeats the checkpoint-selection cohort and is reported as a stability check, not independent confirmation.",
            "The base results are historical, while updates 500 and 700 are fresh matched runs; live-site drift limits comparisons to the base.",
            "McNemar comparisons include only tasks valid in both runs. Overall rates keep all scheduled tasks in the denominator.",
            "No unavailable-task replacement or multiple-comparison correction is applied.",
        ],
    }
    output_json = group / "comparison.json"
    output_json.write_text(json.dumps(report, indent=2) + "\n")

    lines = ["# C2 update 500 vs 700: full Online-Mind2Web evaluation", ""]
    for cohort_name in ("all_300", "checkpoint_selection_holdout_200", "checkpoint_selection_100"):
        title = cohorts[cohort_name]["description"]
        lines += [f"## {title}", "", "| Policy | Overall (95% Wilson CI) | Valid-only (95% Wilson CI) | Unavailable / missing |",
                  "| --- | ---: | ---: | ---: |"]
        for label, row in reports[cohort_name]["rates"].items():
            lines.append(f"| {label} | {row['successes']}/{row['scheduled']} = {percent(row['overall'])} "
                         f"({interval(row['overall_wilson_95'])}) | {row['successes']}/{row['valid']} = "
                         f"{percent(row['valid_only'])} ({interval(row['valid_wilson_95'])}) | "
                         f"{row['unavailable']} / {row['missing']} |")
        lines += ["", "Paired common-valid comparisons:", ""]
        for label, row in reports[cohort_name]["paired_common_valid"].items():
            lines.append(f"- `{label}`: {row['right_wins']} wins, {row['right_losses']} losses, "
                         f"{row['common_valid']} common-valid tasks, exact McNemar p={row['mcnemar_exact_two_sided_p']:.4g}.")
        lines.append("")
    lines += ["## Interpretation constraints", "", *[f"- {note}" for note in report["notes"]], "",
              f"Machine-readable report: `{output_json}`.", ""]
    output_md = group / "comparison.md"
    output_md.write_text("\n".join(lines))
    print(output_md)


if __name__ == "__main__":
    main()
