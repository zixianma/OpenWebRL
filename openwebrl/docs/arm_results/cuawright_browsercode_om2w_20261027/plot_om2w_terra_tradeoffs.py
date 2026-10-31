#!/usr/bin/env python3
"""Render the combined Terra figure from the accompanying public aggregate.

Requires matplotlib. No API credentials, private trajectories, Slurm or original
campaign directories are needed. Run with --output-dir to export PNG/SVG/PDF/CSV.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/openwebrl-terra-tradeoff-matplotlib")

STYLES = {
    "serial_high": ("Sol single high", "s", "#08519C", True),
    "luna_single_high": ("Luna single high", "o", "#08519C", True),
    "parallel3_select_high": ("Luna N3 high", "o", "#A50F15", False),
    "parallel5_select_high": ("Luna N5 high", "o", "#A50F15", True),
    "luna_single_medium": ("Luna single medium", "o", "#4292C6", True),
    "luna_single_low": ("Luna single low", "o", "#9ECAE1", True),
    "luna_n5_medium": ("Luna N5 medium", "o", "#EF3B2C", True),
    "luna_n5_low": ("Luna N5 low", "o", "#FC9272", True),
}
HIGH_METHODS = tuple(STYLES)[:4]
METRICS = (
    ("mean_model_usd_per_task", "Model cost / task ($; log)", "log"),
    ("mean_browser_steps", "Browser commands / task", "linear"),
    ("mean_task_seconds", "Actor latency (s; log)", "log"),
)
PROTOCOL = {
    "model": "gpt-5.6-terra", "reasoning_effort": "medium",
    "max_completion_tokens": 4096, "seed": 42,
    "version": "gpt56-terra-terminal-success-v1",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_aggregate(path):
    data = json.loads(Path(path).read_text())
    require(data["schema"] == "terra-gpt-om2w-plot-aggregate-v1", "Unknown figure aggregate")
    require(all(type(data["judge_protocol"].get(k)) is type(v)
                and data["judge_protocol"][k] == v for k, v in PROTOCOL.items()),
            "Every point must use the exact Terra terminal protocol")
    points = data["points"]
    require(type(points) is list and len(points) == 12, "Twelve completed Terra series required")
    expected = {("browsercode", method) for method in STYLES}
    expected |= {("native", method) for method in HIGH_METHODS}
    actual = [(row["harness"], row["method"]) for row in points]
    require(set(actual) == expected and len(set(actual)) == len(points),
            "Exact BrowserCode and native method identities required; xhigh is excluded")
    for row in points:
        require(row["requested"] == 300, "Every cell must retain its 300-task planned roster")
        for key in ("graded_valid", "graded_invalid", "recorded", "eligible", "successes"):
            require(type(row[key]) is int and row[key] >= 0, "Exact nonnegative outcome counts required")
        require(0 < row["graded_valid"] <= row["eligible"] <= 300
                and row["graded_valid"] + row["graded_invalid"] == row["recorded"] == row["eligible"]
                and row["successes"] <= row["graded_valid"], "Incomplete or inconsistent Terra dispositions")
        require(row["original_valid"] + row["original_invalid_preserved"] == 300,
                "Original valid and invalid counts must retain the full planned roster")
        if row["cohort"] == "historical_terra_regrade":
            require(row["eligible"] + row["original_valid_missing_evidence"] == row["original_valid"],
                    "Historical evidence exclusions must remain explicit")
        if row["cohort"] == "recent_terra_effort_with_prescribed_retry":
            require(row["prescribed_retry_episodes"] == row["original_invalid_preserved"]
                    and row["graded_valid"] == row["original_valid"] + row["retry_recovered_valid"]
                    and row["graded_valid"] + row["resolved_invalid_tasks"] == 300,
                    "One prescribed retry per original invalid must retain unrecovered invalids")
        require(type(row["success_rate"]) in (int, float) and math.isfinite(row["success_rate"]),
                "Finite numeric success rate required")
        require(math.isclose(row["success_rate"], row["successes"] / row["graded_valid"],
                             abs_tol=1e-12, rel_tol=0), "Success must use the valid Terra denominator")
        for metric, _, _ in METRICS:
            require(type(row[metric]) in (int, float) and math.isfinite(row[metric]) and row[metric] > 0,
                    "Finite positive actor metrics required")
            observed = row[metric + "_observed_tasks"]
            require(type(observed) is int and 0 < observed <= 300,
                    "Metric-specific task coverage must remain explicit")
    return data


def write_coordinates(path, points):
    fields = list(dict.fromkeys(key for row in points for key, value in row.items()
                               if not isinstance(value, (dict, list))))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: value for key, value in row.items() if key in fields} for row in points)


def render(data, output_dir, stem="om2w_terra_all_tradeoffs"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import NullFormatter, StrMethodFormatter

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "svg.fonttype": "none", "svg.hashsalt": "openwebrl-terra-om2w-v1", "pdf.fonttype": 42,
                         "savefig.facecolor": "white"})
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.1), sharey="row")
    fig.subplots_adjust(left=0.065, right=0.735, top=0.90, bottom=0.16,
                        hspace=0.45, wspace=0.25)
    fig.suptitle("Online-Mind2Web · GPT-5.6 Terra terminal judge", x=0.065,
                 y=0.985, ha="left", fontsize=11, weight="bold")
    for index, harness in enumerate(("browsercode", "native")):
        rows = sorted((row for row in data["points"] if row["harness"] == harness),
                      key=lambda row: tuple(STYLES).index(row["method"]))
        ys = [100 * row["success_rate"] for row in rows]
        low = max(0, 5 * math.floor((min(ys) - 3) / 5))
        high = min(100, 5 * math.ceil((max(ys) + 3) / 5))
        position = axes[index, 0].get_position()
        title = "a) BrowserCode" if harness == "browsercode" else "b) Native CUAWright"
        fig.text(position.x0, position.y1 + 0.026, title, fontsize=9, weight="bold")
        for column, (metric, label, scale) in enumerate(METRICS):
            ax = axes[index, column]
            ax.set_xscale(scale)
            ax.set_ylim(low, high)
            ax.grid(False, which="both")
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(length=3, width=0.6)
            ax.set_xlabel(label, labelpad=4)
            ax.xaxis.set_minor_formatter(NullFormatter())
            if scale == "log":
                ax.xaxis.set_major_formatter(StrMethodFormatter("${x:g}" if column == 0 else "{x:g}"))
                if column == 0:
                    ax.set_xticks([0.002, 0.01, 0.05, 0.2] if harness == "browsercode" else [0.1, 0.3, 1])
                else:
                    ax.set_xticks([40, 100, 200] if harness == "browsercode" else [600, 900, 1200])
            if column == 0:
                ax.set_ylabel("Valid Terra success (%)", labelpad=4)
            for row in rows:
                _, marker, color, filled = STYLES[row["method"]]
                ax.scatter(row[metric], 100 * row["success_rate"], s=43,
                           marker=marker, edgecolors=color,
                           facecolors=color if filled else "white", linewidths=1.1, zorder=3)
        handles = []
        for row in rows:
            label, marker, color, filled = STYLES[row["method"]]
            handles.append(Line2D([], [], linestyle="none", marker=marker,
                                  markeredgecolor=color, markerfacecolor=color if filled else "white",
                                  markersize=5.8, label=f"{label} (n={row['graded_valid']})"))
        fig.legend(handles=handles, title="BrowserCode" if index == 0 else "Native CUAWright",
                   loc="upper left", bbox_to_anchor=(0.76, 0.925 if index == 0 else 0.47),
                   frameon=False, fontsize=7.5, title_fontsize=8,
                   handletextpad=0.5, labelspacing=0.5, borderaxespad=0)
    missing = sum(row.get("original_valid_missing_evidence", 0) for row in data["points"])
    retry = any(row.get("cohort") == "recent_terra_effort_with_prescribed_retry" for row in data["points"])
    footnote = ("n = valid Terra judgments; valid subsets differ. Native uses a selected endpoint-evidence subset: "
                f"{missing} original-valid runs ungraded.\n"
                "Model cost excludes judging. Actor limits, measurement scopes and live windows differ; descriptive comparison.")
    if retry:
        qualifier = "Medium/low provisional; " if data.get("provisional_recent_effort") else "Medium/low: "
        footnote += "\n" + qualifier + "one prescribed retry per original invalid; cumulative costs and observed task times."
    elif data.get("provisional_recent_effort"):
        footnote += "\nMedium/low results remain provisional."
    fig.text(0.065, 0.035, footnote, fontsize=7.5, ha="left", va="bottom")
    for extension in ("png", "svg", "pdf"):
        metadata = {"Date": None} if extension == "svg" else ({"CreationDate": None, "ModDate": None} if extension == "pdf" else {})
        fig.savefig(output_dir / f"{stem}.{extension}", dpi=300, bbox_inches="tight",
                    pad_inches=0.04, metadata=metadata)
        if extension == "svg":
            path = output_dir / f"{stem}.svg"
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)
    write_coordinates(output_dir / f"{stem}_coordinates.csv", data["points"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", type=Path,
                        default=Path(__file__).with_name("om2w_terra_all_aggregate.json"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    data = read_aggregate(args.aggregate)
    render(data, args.output_dir)
    print(json.dumps({"points": len(data["points"]), "axes": 6,
                      "judge": PROTOCOL["model"], "outputs": ["png", "svg", "pdf", "csv"]}))


if __name__ == "__main__":
    main()
