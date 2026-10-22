#!/usr/bin/env python3
"""Plot equal-benchmark API tradeoffs from the adjacent public aggregate.

Usage: python3 plot_api_tradeoffs.py [--aggregate PATH] [--output-dir PATH]
Requires matplotlib. All plotted coordinates are unweighted means over the
Online-Mind2Web, WebVoyager and DeepShop common-valid efficiency cohorts.
Judge-only work is excluded from cost and task latency. No task payloads needed.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics

os.environ.setdefault("MPLCONFIGDIR", "/tmp/openwebrl-api-tradeoff-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, StrMethodFormatter

BENCHMARKS = ("Online-Mind2Web", "WebVoyager", "DeepShop")
METHODS = {
    "serial_high": ("Sol single high", "#0072B2", "o"),
    "luna_single_high": ("Luna single high", "#009E73", "s"),
    "parallel3_select_high": ("Luna N3 high + high", "#D55E00", "^"),
    "parallel5_select_high": ("Luna N5 high + high", "#CC79A7", "D"),
}


def read_rows(path):
    data = json.loads(path.read_text())
    rows = []
    for benchmark in BENCHMARKS:
        bench = data["benchmarks"][benchmark]
        cohort = bench["efficiency_common_valid_tasks"]
        if not isinstance(cohort, int) or cohort <= 0:
            raise ValueError(f"Invalid common-valid denominator: {benchmark}")
        for method, (label, _, _) in METHODS.items():
            metrics = bench["efficiency"].get(method)
            if metrics is None:
                raise ValueError(f"Incomplete aggregate: {benchmark} / {method}")
            success = metrics["success_rate"]
            if abs(success - metrics["successes"] / cohort) > 1e-10:
                raise ValueError(f"Success denominator mismatch: {benchmark} / {method}")
            rows.append(dict(benchmark=benchmark, method=method, label=label,
                common_valid_tasks=cohort, successes=metrics["successes"],
                success_rate=success,
                mean_model_usd_per_task=metrics["mean_model_usd_per_task"],
                mean_task_seconds=metrics["mean_task_seconds"],
                mean_browser_steps=metrics["mean_browser_steps"]))
    means = []
    for method, (label, _, _) in METHODS.items():
        selected = [row for row in rows if row["method"] == method]
        means.append(dict(method=method, label=label,
            mean_success_rate=statistics.mean(row["success_rate"] for row in selected),
            mean_model_usd_per_task=statistics.mean(row["mean_model_usd_per_task"] for row in selected),
            mean_task_seconds=statistics.mean(row["mean_task_seconds"] for row in selected),
            mean_browser_steps=statistics.mean(row["mean_browser_steps"] for row in selected)))
    return rows, means


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def draw(ax, means, metric):
    for row in means:
        method = row["method"]
        label, color, marker = METHODS[method]
        x, y = row[metric], 100 * row["mean_success_rate"]
        ax.scatter(x, y, s=115, color=color, marker=marker,
                   edgecolors="white", linewidths=0.9, zorder=3, label=label)
    ax.set_ylabel("Mean benchmark success (%)")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    values = [100 * row["mean_success_rate"] for row in means]
    span = max(values) - min(values)
    pad = max(3.0, span * 0.8)
    ax.set_ylim(max(0, min(values) - pad), min(100, max(values) + pad))
    ax.set_xlim(left=0, right=max(row[metric] for row in means) * 1.12)
    if metric == "mean_model_usd_per_task":
        ax.set_xlabel("Mean model cost per task (USD)")
        ax.xaxis.set_major_formatter(StrMethodFormatter("${x:.2f}"))
        ax.set_title("Performance vs. API cost", loc="left", pad=15)
    elif metric == "mean_task_seconds":
        ax.set_xlabel("Mean task latency (seconds)")
        ax.set_title("Performance vs. latency", loc="left", pad=15)
    else:
        ax.set_xlabel("Mean selected browser steps per task")
        ax.set_title("Performance vs. browser steps", loc="left", pad=15)
        xs = [row[metric] for row in means]
        ax.set_xlim(max(0, min(xs) - 1), max(xs) + 1)
    ax.grid(True, color="#E2E6EA", linewidth=0.7, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#B7BEC7")


def save_figure(fig, path):
    fig.savefig(path, dpi=240)
    if path.suffix == ".svg":
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", type=Path, default=Path(__file__).with_name("aggregate.json"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    rows, means = read_rows(args.aggregate)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "api_tradeoffs_by_benchmark.csv", rows)
    write_csv(args.output_dir / "api_tradeoffs_means.csv", means)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                        "axes.titlesize": 15, "svg.fonttype": "none",
                        "savefig.facecolor": "white"})
    footer = ("Equal weight: Online-Mind2Web, WebVoyager, DeepShop.\n"
              "Same four-method common-valid cohort within each benchmark; judge-only work excluded.")
    for metric, suffix in (("mean_model_usd_per_task", "cost"),
                           ("mean_task_seconds", "latency"),
                           ("mean_browser_steps", "browser_steps")):
        fig, ax = plt.subplots(figsize=(10.4, 5.6))
        draw(ax, means, metric)
        fig.text(0.12, 0.025, footer, fontsize=9, color="#5A6470", linespacing=1.5)
        fig.subplots_adjust(left=0.10, right=0.70, bottom=0.22, top=0.88)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.73, 0.56),
                   frameon=False, fontsize=10, labelspacing=1.4)
        for ext in ("png", "svg"):
            save_figure(fig, args.output_dir / f"api_performance_vs_{suffix}.{ext}")
        plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.6), sharey=True)
    draw(axes[0], means, "mean_model_usd_per_task")
    draw(axes[1], means, "mean_task_seconds")
    draw(axes[2], means, "mean_browser_steps")
    for ax in axes[1:]:
        ax.set_ylabel("")
    for ax in axes:
        ax.title.set_fontsize(13)
    fig.text(0.05, 0.025, footer, fontsize=9, color="#5A6470", linespacing=1.5)
    fig.subplots_adjust(left=0.05, right=0.82, bottom=0.22, top=0.88, wspace=0.22)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.835, 0.56),
               frameon=False, fontsize=10, labelspacing=1.4)
    for ext in ("png", "svg"):
        save_figure(fig, args.output_dir / f"api_performance_tradeoffs.{ext}")
    plt.close(fig)
    (args.output_dir / "api_tradeoffs_provenance.json").write_text(json.dumps({
        "aggregate_sha256": hashlib.sha256(args.aggregate.read_bytes()).hexdigest(),
        "benchmark_weighting": "unweighted arithmetic mean over three benchmarks",
        "cohort": "four-method common-valid within each benchmark",
        "cost_scope": "model USD/task, including valid failures; judge excluded",
        "latency_scope": "task seconds excluding judge-only work",
        "browser_steps_scope": "selected BrowserCode batches per task, averaged equally across benchmarks",
        "benchmark_common_valid_tasks": {r["benchmark"]: r["common_valid_tasks"] for r in rows},
        "uncertainty": "Point estimates only; no uncertainty inferred from aggregate data."
    }, indent=2) + "\n")
    print(json.dumps(means, indent=2))

if __name__ == "__main__":
    main()
