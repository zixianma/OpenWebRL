#!/usr/bin/env python3
"""Plot matched fresh Online-Mind2Web tasks under BrowserCode and CUAWright.
Requires matplotlib. Uses aggregate-only JSON; no private task data is needed.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/openwebrl-api-tradeoff-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, StrMethodFormatter
from matplotlib.lines import Line2D

METHODS = {
    "serial_high": ("Sol single", "#0072B2", "s"),
    "luna_single_high": ("Luna single", "#0072B2", "o"),
    "parallel3_select_high": ("Luna N3", "#E6B800", "o"),
    "parallel5_select_high": ("Luna N5", "#E87500", "o"),
}
LEGEND_LABELS = ("Sol single", "Luna single", "Luna N = 3", "Luna N = 5")


def read_rows(path):
    data = json.loads(path.read_text())
    n = data["shared_fresh_common_valid_tasks"]
    if type(n) is not int or n <= 0:
        raise ValueError("Positive exact shared task denominator required")
    series = {}
    for harness in ("browsercode", "native"):
        metrics = data[harness]["tradeoffs"]
        if set(metrics) != set(METHODS):
            raise ValueError("Exactly four methods required under each harness")
        rows = []
        for method in METHODS:
            r = metrics[method]
            if r["tasks"] != n or abs(r["success_rate"] - r["successes"] / n) > 1e-10:
                raise ValueError("Eight-cell shared cohort denominator mismatch")
            for key in ("mean_model_usd_per_task", "mean_task_seconds", "mean_browser_steps"):
                if not math.isfinite(r[key]) or r[key] <= 0:
                    raise ValueError("Finite positive tradeoff metrics required")
            rows.append(dict(r, method=method, mean_success_rate=r["success_rate"]))
        series[harness] = rows
    return data, series["browsercode"], series["native"]


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)



def draw(ax, means, metric, native):
    for series, filled in ((means, True), (native, False)):
        for row in series:
            label, color, marker = METHODS[row["method"]]
            ax.scatter(row[metric], 100 * row["mean_success_rate"], s=85,
                       marker=marker, facecolors=color if filled else "white",
                       edgecolors="white" if filled else color,
                       linewidths=0.9 if filled else 2.0, zorder=3)
    ax.set_ylabel("Success on common-valid tasks (%)")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    values = [100 * r["mean_success_rate"] for r in means + native]
    ax.set_ylim(max(0, min(values) - 3), min(100, max(values) + 4))
    if metric == "mean_model_usd_per_task":
        ax.set_xlabel("Mean model cost / task (USD; log scale)")
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(StrMethodFormatter("${x:g}"))
        ax.set_title("Performance vs. API cost", loc="left", pad=15)
    elif metric == "mean_task_seconds":
        ax.set_xlabel("Mean task latency (seconds; log scale)")
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
        ax.set_title("Performance vs. latency", loc="left", pad=15)
    else:
        ax.set_xlabel("Mean browser-command executions / task")
        ax.set_title("Performance vs. browser commands", loc="left", pad=15)
    xs = [r[metric] for r in means + native]
    if metric == "mean_browser_steps":
        ax.set_xlim(max(0, min(xs) - 2), max(xs) + 2)
    else:
        ax.set_xlim(min(xs) / 1.4, max(xs) * 1.5)
    ax.grid(True, color="#E2E6EA", linewidth=0.7, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#B7BEC7")


def legends(fig, x=0.835, y=0.70):
    methods = [Line2D([], [], linestyle="", marker=m, color=c, markersize=8, label=l)
               for l, c, m in METHODS.values()]
    fig.legend(handles=methods, title="Method (all high)", loc="upper left",
               bbox_to_anchor=(x, y), frameon=False, fontsize=10, title_fontsize=11,
               labelspacing=1.1)
    harnesses = [Line2D([], [], linestyle="", marker="o", color="#5A6470", markersize=8,
                        label="BrowserCode · OM2W"),
                 Line2D([], [], linestyle="", marker="o", markerfacecolor="white",
                        markeredgecolor="#5A6470", markeredgewidth=2, markersize=8,
                        label="Native CUAWright · OM2W\nAll 300 collected; same fresh tasks")]
    fig.legend(handles=harnesses, title="Harness / cohort", loc="upper left",
               bbox_to_anchor=(x, y - .29), frameon=False, fontsize=9, title_fontsize=11,
               labelspacing=1.2)



def save_figure(fig, path):
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.025)
    if path.suffix == ".svg":
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", type=Path, default=Path(__file__).with_name("om2w_harness_comparison_completed.json"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    comparison, means, native = read_rows(args.aggregate)
    data = json.loads(Path(__file__).with_name("native_cuawright_completed.json").read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "om2w_harness_tradeoff_coordinates.csv",
              [dict(harness=h, **r) for h, rows in (("BrowserCode", means), ("Native CUAWright", native)) for r in rows])
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                        "axes.titlesize": 15, "svg.fonttype": "none", "pdf.fonttype": 42,
                        "savefig.facecolor": "white"})
    footer = (f'OM2W only · same {comparison["shared_fresh_common_valid_tasks"]} fresh tasks valid in all eight cells; valid failures included. All four native methods: 300 outcomes each; collection complete.\n'
              'Native: 20,000 tokens / 100 decisions; BrowserCode: 12,288 / 30. Different collection dates and timing scopes; descriptive comparison. External judging excluded.\n'
              'One execution: a selected BrowserCode block or native browser-command log entry; either can contain several actions. This compares harness-level command volume, not standardized actions.\n'
              'Cost: native per-task all-attempt model charges, helpers and full unknown reserves; BrowserCode original proposal/selector receipts. Tables are reported separately.')
    for metric, suffix in (("mean_model_usd_per_task", "cost"),
                           ("mean_task_seconds", "latency"),
                           ("mean_browser_steps", "browser_steps")):
        fig, ax = plt.subplots(figsize=(11.5, 5.7))
        draw(ax, means, metric, native)
        fig.subplots_adjust(left=.09, right=.66, bottom=.33, top=.88)
        legends(fig, x=.68, y=.91)
        fig.text(.09, .025, footer.replace('; external judging excluded.', '; external judging excluded.\n'),
                 fontsize=7.3, color="#5A6470", linespacing=1.6)
        for ext in ("png", "svg", "pdf"):
            save_figure(fig, args.output_dir / f"om2w_harness_vs_{suffix}.{ext}")
        plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.1), sharey=True)
    for ax, metric in zip(axes, ("mean_model_usd_per_task", "mean_task_seconds", "mean_browser_steps")):
        draw(ax, means, metric, native)
        ax.title.set_fontsize(13)
    for ax in axes[1:]:
        ax.set_ylabel("")
    fig.suptitle("Online-Mind2Web · BrowserCode and native CUAWright", x=.05, ha="left", fontsize=17, y=.97)
    fig.subplots_adjust(left=.05, right=.80, bottom=.33, top=.85, wspace=.24)
    legends(fig, x=.815, y=.89)
    fig.text(.05, .025, footer, fontsize=8.5, color="#5A6470", linespacing=1.7)
    for ext in ("png", "svg", "pdf"):
        save_figure(fig, args.output_dir / f"om2w_harness_tradeoffs.{ext}")
    plt.close(fig)
    (args.output_dir / "om2w_harness_tradeoffs_provenance.json").write_text(json.dumps({
        "comparison_aggregate_sha256": hashlib.sha256(args.aggregate.read_bytes()).hexdigest(),
        "native_status_aggregate_sha256": hashlib.sha256(Path(__file__).with_name("native_cuawright_completed.json").read_bytes()).hexdigest(),
        "benchmark": "Online-Mind2Web only",
        "shared_fresh_common_valid_tasks": comparison["shared_fresh_common_valid_tasks"],
        "cohort": "Exact same fresh tasks valid in all four methods under both harnesses, including valid failures",
        "native_status": data["status"],
        "native_snapshot_utc": data["snapshot_utc"],
        "cost_scope": "Native: per-task all-attempt model USD, helpers/full unknown reserves. BrowserCode: original episode proposal/selector receipt USD. External judge excluded.",
        "latency_scope": "harness-specific native task seconds; external judge and saved-grade recovery excluded",
        "browser_steps_scope": "Browser-command executions: saved selected BrowserCode blocks or literal native browser-command log entries. Each may contain multiple actions; a harness-level volume proxy, not standardized action counts.",
        "comparability": "Different limits, collection windows and task-time boundaries; descriptive only",
        "uncertainty": "Point estimates; no uncertainty inferred from aggregate data"
    }, indent=2) + "\n")
    print(json.dumps({"browsercode": means, "native": native}, indent=2))

if __name__ == "__main__":
    main()
