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
from matplotlib.ticker import MaxNLocator, StrMethodFormatter

BENCHMARKS = ("Online-Mind2Web", "WebVoyager", "DeepShop")
METHODS = {
    "serial_high": ("Sol single high", "#0072B2", "o"),
    "luna_single_high": ("Luna single high", "#009E73", "s"),
    "parallel3_select_high": ("Luna N3 high + high", "#D55E00", "^"),
    "parallel5_select_high": ("Luna N5 high + high", "#CC79A7", "D"),
}
LEGEND_LABELS = ("Sol single", "Luna single", "Luna N = 3", "Luna N = 5")


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
        # A hollow square keeps the nearly coincident single-call latency
        # points distinguishable at the final paper size.
        hollow = method == "luna_single_high"
        ax.scatter(x, y, s=32, facecolors="white" if hollow else color, marker=marker,
                   edgecolors=color if hollow else "white", linewidths=0.9 if hollow else 0.5,
                   zorder=3, label=label)
    ax.set_ylabel("Mean success (%)")
    values = [100 * row["mean_success_rate"] for row in means]
    span = max(values) - min(values)
    pad = max(0.6, span * 0.15)
    ax.set_ylim(max(0, min(values) - pad), min(100, max(values) + pad))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    xs = [row[metric] for row in means]
    xpad = (max(xs) - min(xs)) * 0.12
    ax.set_xlim(max(0, min(xs) - xpad), max(xs) + xpad)
    if metric == "mean_model_usd_per_task":
        ax.set_xlabel("Model cost / task ($)")
        ax.xaxis.set_major_formatter(StrMethodFormatter("${x:.2f}"))
    elif metric == "mean_task_seconds":
        ax.set_xlabel("Task latency (s)")
    else:
        ax.set_xlabel("Browser steps / task")
    ax.grid(False, which="both")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=3, width=0.6, pad=2)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#555555")
        ax.spines[side].set_linewidth(0.6)


def save_figure(fig, path):
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.025)
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
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                        "xtick.labelsize": 7, "ytick.labelsize": 7,
                        "axes.labelpad": 3, "svg.fonttype": "none", "pdf.fonttype": 42,
                        "savefig.facecolor": "white"})
    for metric, suffix in (("mean_model_usd_per_task", "cost"),
                           ("mean_task_seconds", "latency"),
                           ("mean_browser_steps", "browser_steps")):
        fig, ax = plt.subplots(figsize=(3.35, 2.15))
        draw(ax, means, metric)
        fig.subplots_adjust(left=0.16, right=0.98, bottom=0.20, top=0.77)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, LEGEND_LABELS, loc="upper center", bbox_to_anchor=(0.54, 1.01),
                   ncol=2, frameon=False, fontsize=7, handletextpad=0.4, columnspacing=1.0)
        for ext in ("png", "svg", "pdf"):
            save_figure(fig, args.output_dir / f"api_performance_vs_{suffix}.{ext}")
        plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(7, 1.75), sharey=True)
    draw(axes[0], means, "mean_model_usd_per_task")
    draw(axes[1], means, "mean_task_seconds")
    draw(axes[2], means, "mean_browser_steps")
    for ax in axes[1:]:
        ax.set_ylabel("")
    for letter, ax in zip("abc", axes):
        ax.set_xlabel(f"({letter}) {ax.get_xlabel()}")
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.23, top=0.80, wspace=0.16)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, LEGEND_LABELS, loc="upper center", bbox_to_anchor=(0.53, 1.01),
               ncol=4, frameon=False, fontsize=8, handletextpad=0.4, columnspacing=1.5)
    for ext in ("png", "svg", "pdf"):
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
