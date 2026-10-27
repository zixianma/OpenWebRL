#!/usr/bin/env python3
"""Plot separate BrowserCode and preliminary native CUAWright API tradeoffs.

Usage: python3 plot_api_tradeoffs.py [--aggregate PATH] [--output-dir PATH]
Requires matplotlib. BrowserCode coordinates are equal three-benchmark means;
native CUAWright coordinates use one matched fresh OM2W common-valid cohort.
Judge-only work is excluded from cost and task latency. No task payloads needed.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

os.environ.setdefault("MPLCONFIGDIR", "/tmp/openwebrl-api-tradeoff-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, StrMethodFormatter
from matplotlib.lines import Line2D

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


def native_rows(path):
    data = json.loads(path.read_text())
    cohort = data["fresh_common_valid_tasks"]
    if type(cohort) is not int or cohort <= 0 or set(data["tradeoffs"]) != set(METHODS):
        raise ValueError("Native cohort or method coverage invalid")
    rows = []
    for method in METHODS:
        row = dict(data["tradeoffs"][method], method=method)
        if row["tasks"] != cohort or abs(row["success_rate"] - row["successes"] / cohort) > 1e-10:
            raise ValueError("Native common-valid denominator mismatch")
        for metric in ("mean_model_usd_per_task", "mean_task_seconds", "mean_browser_steps"):
            if not math.isfinite(row[metric]) or row[metric] <= 0:
                raise ValueError("Native tradeoff metric must be finite and positive")
        rows.append(dict(row, mean_success_rate=row["success_rate"]))
    return data, rows


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
        ax.set_xlabel("Mean recorded browser batches / task")
        ax.set_title("Performance vs. browser batches", loc="left", pad=15)
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
                        label="BrowserCode · 3-benchmark mean"),
                 Line2D([], [], linestyle="", marker="o", markerfacecolor="white",
                        markeredgecolor="#5A6470", markeredgewidth=2, markersize=8,
                        label="Native CUAWright · OM2W only\nPreliminary; matched fresh tasks")]
    fig.legend(handles=harnesses, title="Harness / cohort", loc="upper left",
               bbox_to_anchor=(x, y - .29), frameon=False, fontsize=9, title_fontsize=11,
               labelspacing=1.2)


def status_table(fig, data, rect):
    ax = fig.add_axes(rect)
    ax.axis("off")
    rows = []
    for method, (label, _, _) in METHODS.items():
        r = data["latest_recorded"][method]
        f = data["matched_fresh_recorded"][method]
        rows.append([label.replace(" high + high", "").replace(" high", ""),
                     f'{r["recorded"]} / 300', str(r["successes"]),
                     f'{100 * r["success_rate"]:.1f}%', str(f["successes"]),
                     f'{100 * f["success_rate"]:.1f}%'])
    table = ax.table(cellText=rows,
                     colLabels=["Native CUAWright", "Recorded", "Successes", "Overall success", "Fresh successes",
                                f'Same {data["fresh_common_recorded_tasks"]} fresh tasks'],
                     cellLoc="center", loc="center", colWidths=[.20, .14, .13, .16, .16, .21])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.35)
    for (i, j), cell in table.get_celld().items():
        cell.set_edgecolor("#E2E6EA")
        cell.set_facecolor("#EFF3F7" if i == 0 else "white")
        if i == 0:
            cell.set_text_props(weight="bold")
    ax.set_title("Latest native outcomes · invalids count as failures (separate from scatter cohort)",
                 fontsize=11, loc="left", pad=7)

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
    data, native = native_rows(Path(__file__).with_name("native_cuawright_preliminary.json"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "api_tradeoffs_by_benchmark.csv", rows)
    write_csv(args.output_dir / "api_tradeoffs_means.csv", means)
    write_csv(args.output_dir / "api_tradeoffs_native_preliminary.csv", native)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                        "axes.titlesize": 15, "svg.fonttype": "none",
                        "savefig.facecolor": "white"})
    footer = (f'BrowserCode: equal OM2W / WebVoyager / DeepShop means; common-valid within each benchmark. '
              f'Native: same {data["fresh_common_valid_tasks"]} fresh OM2W tasks valid in all four methods.\n'
              'Model cost includes helpers, all attempts and conservative unknown reserves; external judging excluded. '
              'Harnesses, limits and collection windows differ; cross-harness gaps are descriptive.\n'
              f'Native snapshot: {data["snapshot_utc"]}; N5 incomplete. Overall table includes historical100 for three methods. '
              'Invalids, including two OOM outcomes, remain failures.')
    for metric, suffix in (("mean_model_usd_per_task", "cost"),
                           ("mean_task_seconds", "latency"),
                           ("mean_browser_steps", "browser_steps")):
        fig, ax = plt.subplots(figsize=(12, 7.5))
        draw(ax, means, metric, native)
        fig.subplots_adjust(left=.09, right=.66, bottom=.48, top=.90)
        legends(fig, x=.68, y=.91)
        status_table(fig, data, [.09, .17, .84, .20])
        fig.text(.09, .025, footer.replace('; external judging excluded.', '; external judging excluded.\n'),
                 fontsize=7.3, color="#5A6470", linespacing=1.6)
        for ext in ("png", "svg", "pdf"):
            save_figure(fig, args.output_dir / f"api_performance_vs_{suffix}.{ext}")
        plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(16, 7.2), sharey=True)
    for ax, metric in zip(axes, ("mean_model_usd_per_task", "mean_task_seconds", "mean_browser_steps")):
        draw(ax, means, metric, native)
        ax.title.set_fontsize(13)
    for ax in axes[1:]:
        ax.set_ylabel("")
    fig.suptitle("API performance tradeoffs · harnesses shown separately", x=.05, ha="left", fontsize=17, y=.97)
    fig.subplots_adjust(left=.05, right=.80, bottom=.47, top=.88, wspace=.24)
    legends(fig, x=.815, y=.89)
    status_table(fig, data, [.05, .16, .73, .21])
    fig.text(.05, .025, footer, fontsize=8.5, color="#5A6470", linespacing=1.7)
    for ext in ("png", "svg", "pdf"):
        save_figure(fig, args.output_dir / f"api_performance_tradeoffs.{ext}")
    plt.close(fig)
    (args.output_dir / "api_tradeoffs_provenance.json").write_text(json.dumps({
        "aggregate_sha256": hashlib.sha256(args.aggregate.read_bytes()).hexdigest(),
        "native_preliminary_aggregate_sha256": hashlib.sha256(Path(__file__).with_name("native_cuawright_preliminary.json").read_bytes()).hexdigest(),
        "browsercode": {"benchmark_weighting": "unweighted arithmetic mean over three benchmarks",
                        "cohort": "four-method common-valid within each benchmark",
                        "benchmark_common_valid_tasks": {r["benchmark"]: r["common_valid_tasks"] for r in rows}},
        "native": {"benchmark": "Online-Mind2Web only", "status": "preliminary, incomplete and not final audited",
                   "scatter_fresh_common_valid_tasks": data["fresh_common_valid_tasks"],
                   "snapshot_utc": data["snapshot_utc"]},
        "cost_scope": "all-attempt model USD/task including helpers and conservative unknown reserves; judge excluded",
        "latency_scope": "harness-specific native task seconds excluding external judge and saved-grade recovery",
        "browser_steps_scope": "recorded selected browser batches, not decisions or candidate proposals",
        "comparability": "Different benchmarks, limits, harnesses and collection windows; descriptive comparison only",
        "uncertainty": "Point estimates only; no uncertainty inferred from aggregate data."
    }, indent=2) + "\n")
    print(json.dumps({"browsercode": means, "native": native}, indent=2))

if __name__ == "__main__":
    main()
