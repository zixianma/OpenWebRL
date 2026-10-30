#!/usr/bin/env python3
"""Plot all 300 matched Online-Mind2Web tasks under BrowserCode and CUAWright.
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
from matplotlib.ticker import StrMethodFormatter, NullFormatter

METHODS = {
    "serial_high": ("Sol single", "#0072B2", "s"),
    "luna_single_high": ("Luna single", "#0072B2", "o"),
    "parallel3_select_high": ("Luna N3", "#E6B800", "o"),
    "parallel5_select_high": ("Luna N5", "#E87500", "o"),
}
LEGEND_LABELS = ("Sol single", "Luna single", "Luna N = 3", "Luna N = 5")


def read_rows(path):
    data = json.loads(path.read_text())
    n = data["tasks_per_cell"]
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
            if r["tasks"] != n or abs(r["success_rate"] - r["successes"] / r["valid"]) > 1e-10:
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



METRICS = (("mean_model_usd_per_task", "Model cost / task ($; log)", "cost"),
           ("mean_browser_steps", "Browser commands / task", "browser_steps"),
           ("mean_task_seconds", "Task latency (s; log)", "latency"))


def draw(ax, rows, metric, native):
    """Four observed points only; equal visibility and direct method labels."""
    ax.set_ylim(58, 81)
    ax.set_yticks([60, 70, 80])
    ax.grid(False, which="both")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=3, width=0.6, pad=2)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#555555")
        ax.spines[side].set_linewidth(0.6)
    if metric == "mean_model_usd_per_task":
        ax.set_xscale("log")
        ax.set_xlim((0.065, 2.1) if native else (0.006, 0.34))
        ax.set_xticks([0.1, 0.3, 1] if native else [0.01, 0.05, 0.2])
        ax.xaxis.set_major_formatter(StrMethodFormatter("${x:g}"))
    elif metric == "mean_browser_steps":
        ax.set_xlim((27.5, 39.5) if native else (12.8, 16.6))
        ax.set_xticks([28, 32, 36] if native else [13, 14, 15, 16])
    else:
        ax.set_xscale("log")
        ax.set_xlim((420, 1800) if native else (60, 310))
        ax.set_xticks([500, 900, 1300] if native else [90, 150, 220])
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    for row in rows:
        method = row["method"]
        label, color, marker = METHODS[method]
        x, y = row[metric], 100 * row["mean_success_rate"]
        ax.scatter(x, y, s=30, marker=marker, color=color,
                   edgecolor="white", linewidth=0.5, zorder=3)
        dx, dy, ha = 4, 6, "left"
        if method == "serial_high":
            dy = -9
            if metric == "mean_model_usd_per_task":
                dx, ha = -4, "right"
                if native:
                    dy = 7
        if method == "luna_single_high":
            dy = -9 if native else 6
        if method == "parallel3_select_high" and native:
            dy = -10
        if method == "parallel3_select_high" and metric == "mean_browser_steps" and native:
            dx, ha = -4, "right"
        if metric == "mean_task_seconds":
            if method in ("parallel3_select_high", "parallel5_select_high"):
                dx, ha = -4, "right"
            if method == "serial_high" and not native:
                dx, dy, ha = -4, 8, "right"
            if method == "luna_single_high":
                dy = -9
                if native:
                    dx, ha = -4, "right"
            if method == "parallel3_select_high" and native:
                dx, dy, ha = -4, 6, "right"
        ax.annotate(label, (x, y), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, va="center", fontsize=7, color=color)


def figure(means, native, metrics, output, stem):
    columns = len(metrics)
    fig, axes = plt.subplots(2, columns, figsize=(7 if columns == 3 else 3.35, 3.5),
                             sharey=True, squeeze=False)
    fig.subplots_adjust(left=0.075 if columns == 3 else 0.16, right=0.985,
                        top=0.84, bottom=0.13, hspace=0.78, wspace=0.18)
    for ri, (rows, title) in enumerate(((means, "a)  BrowserCode"),
                                       (native, "b)  Native CUAWright"))):
        pos = axes[ri, 0].get_position()
        fig.text(pos.x0, pos.y1 + 0.045, title, fontsize=8.5, weight="bold")
        for ci, (metric, label, _) in enumerate(metrics):
            draw(axes[ri, ci], rows, metric, native=bool(ri))
            axes[ri, ci].set_xlabel(label, labelpad=3)
            if ci == 0:
                axes[ri, ci].set_ylabel("Valid-only success (%)", labelpad=3)
    for ext in ("png", "svg", "pdf"):
        save_figure(fig, output / f"{stem}.{ext}")
    plt.close(fig)


def save_figure(fig, path):
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.025)
    if path.suffix == ".svg":
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", type=Path, default=Path(__file__).with_name("om2w_harness_comparison_valid_only.json"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    comparison, means, native = read_rows(args.aggregate)
    data = json.loads(Path(__file__).with_name("native_cuawright_completed.json").read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "om2w_harness_tradeoff_coordinates.csv",
              [dict(harness=h, **r) for h, rows in (("BrowserCode", means), ("Native CUAWright", native)) for r in rows])
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7,
                        "axes.labelsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                        "svg.fonttype": "none", "pdf.fonttype": 42,
                        "savefig.facecolor": "white"})
    for metric in METRICS:
        figure(means, native, [metric], args.output_dir, f"om2w_harness_vs_{metric[2]}")
    figure(means, native, METRICS, args.output_dir, "om2w_harness_tradeoffs")
    (args.output_dir / "om2w_harness_tradeoffs_provenance.json").write_text(json.dumps({
        "comparison_aggregate_sha256": hashlib.sha256(args.aggregate.read_bytes()).hexdigest(),
        "native_status_aggregate_sha256": hashlib.sha256(Path(__file__).with_name("native_cuawright_completed.json").read_bytes()).hexdigest(),
        "benchmark": "Online-Mind2Web only",
        "tasks_per_cell": comparison["tasks_per_cell"],
        "cohort": "300 attempted per cell; each success rate uses its own valid-outcome denominator; valid subsets differ",
        "native_status": "Full300 science/accounting verified; global cleanup uncertified",
        "native_snapshot_utc": data["snapshot_utc"],
        "metric_coverage": {h: {a: {k:v for k,v in x.items() if k.endswith("_observed_tasks") or "unknown" in k} for a,x in comparison[h]["tradeoffs"].items()} for h in ("browsercode","native")},
        "full300_source_numeric_review_sha256": "fb0b3156afb151c88e4b69453fb2c81d4feea7fb1d3898f56cf67a2a08488430",
        "cost_scope": "Native: per-task all-attempt model USD, helpers/full unknown reserves. BrowserCode: original episode proposal/selector receipt USD. External judge excluded.",
        "latency_scope": "harness-specific native task seconds; external judge and saved-grade recovery excluded",
        "browser_steps_scope": "Browser-command executions: saved selected BrowserCode blocks or literal native browser-command log entries. Each may contain multiple actions; a harness-level volume proxy, not standardized action counts.",
        "visualization": "Separate harness rows; directly labeled observed points with equal opacity; no connecting lines, frontier rings or grid lines. X-axis ranges differ by harness.",
        "comparability": "Different limits, collection windows and task-time boundaries; descriptive only",
        "uncertainty": "Point estimates; no uncertainty inferred from aggregate data"
    }, indent=2) + "\n")
    print(json.dumps({"browsercode": means, "native": native}, indent=2))

if __name__ == "__main__":
    main()
