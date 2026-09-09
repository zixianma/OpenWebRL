#!/usr/bin/env python3
"""Report success and loop-collapse indicators from ARM policy evaluations."""

import argparse
import collections
import json
import math
from pathlib import Path
import re
import statistics


TOOL_NAME = re.compile(r'"name"\s*:\s*"([^"]+)"')


def load_results(directory):
    return [json.loads(path.read_text()) for path in sorted((directory / "results").glob("*.json"))]


def actions(record):
    messages = record.get("metadata", {}).get("messages", [])
    return [TOOL_NAME.findall(message.get("content", "")) for message in messages
            if message.get("role") == "assistant" and isinstance(message.get("content"), str)]


def behavior(directory):
    rows = load_results(directory)
    valid = [row for row in rows if row.get("valid")]
    successes = sum(row.get("reward") == 1 for row in valid)
    trajectories = []
    calls = collections.Counter()
    primary_pairs = repeated_primary = 0
    for row in rows:
        turns = actions(row)
        names = [name for turn in turns for name in turn]
        calls.update(names)
        primaries = [turn[0] for turn in turns if turn]
        primary_pairs += max(0, len(primaries) - 1)
        repeated_primary += sum(left == right for left, right in zip(primaries, primaries[1:]))
        if row.get("total_steps") is not None:
            trajectories.append({"steps": row["total_steps"], "calls": names})
    steps = [row["steps"] for row in trajectories]
    total_calls = sum(calls.values())
    terminated = sum("done" in row["calls"] for row in trajectories)
    capped = sum(row["steps"] >= 30 for row in trajectories)
    return {
        "directory": str(directory.resolve()),
        "attempted": len(rows),
        "valid": len(valid),
        "successes": successes,
        "success_rate_overall": successes / len(rows) if rows else None,
        "success_rate_valid": successes / len(valid) if valid else None,
        "trajectories_with_steps": len(trajectories),
        "mean_steps": statistics.mean(steps) if steps else None,
        "median_steps": statistics.median(steps) if steps else None,
        "termination_rate": terminated / len(trajectories) if trajectories else None,
        "step_cap_30_rate": capped / len(trajectories) if trajectories else None,
        "scroll_call_fraction": calls["scroll"] / total_calls if total_calls else None,
        "repeated_primary_action_rate": repeated_primary / primary_pairs if primary_pairs else None,
        "tool_call_counts": dict(calls.most_common()),
    }


def percent(value):
    return "n/a" if value is None else f"{100 * value:.1f}%"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=DIR")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    reports = {}
    for item in args.run:
        label, separator, value = item.partition("=")
        if not separator or not label:
            parser.error("--run must be LABEL=DIR")
        reports[label] = behavior(Path(value))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "runs": reports,
        "interpretation": [
            "Behavior metrics use saved assistant tool calls; unavailable tasks may lack trajectories.",
            "A success gain accompanied by longer trajectories, lower termination, more 30-step caps, or more scrolling may indicate the collapse reported by upstream actor distillation.",
        ],
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    markdown = args.output.with_suffix(".md")
    lines = [
        "# ARM actor behavior comparison", "",
        "| Run | Overall success | Valid-only success | Mean / median steps | Terminated | Hit 30 steps | Scroll calls | Repeated primary action |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, report in reports.items():
        lines.append(
            f"| {label} | {percent(report['success_rate_overall'])} "
            f"({report['successes']}/{report['attempted']}) | {percent(report['success_rate_valid'])} "
            f"({report['successes']}/{report['valid']}) | {report['mean_steps']:.2f} / "
            f"{report['median_steps']:.1f} | {percent(report['termination_rate'])} | "
            f"{percent(report['step_cap_30_rate'])} | {percent(report['scroll_call_fraction'])} | "
            f"{percent(report['repeated_primary_action_rate'])} |"
        )
    lines += ["", *payload["interpretation"], ""]
    markdown.write_text("\n".join(lines))
    print(markdown)


if __name__ == "__main__":
    main()
