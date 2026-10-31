#!/usr/bin/env python3
"""Plot Online-Mind2Web GPT tradeoffs, with a provisional BrowserCode effort cohort.
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
from matplotlib.lines import Line2D

METHODS = {
    "serial_high": ("Sol single", "#08519C", "s"),
    "luna_single_high": ("Luna single", "#08519C", "o"),
    "parallel3_select_high": ("Luna N3", "#A50F15", "^"),
    "parallel5_select_high": ("Luna N5", "#A50F15", "D"),
}
LEGEND_LABELS = ("Sol single", "Luna single", "Luna N = 3", "Luna N = 5")
EFFORT_METHODS = {
    "luna_single_medium": ("Single medium", "#4292C6", "o"),
    "luna_single_low": ("Single low", "#9ECAE1", "o"),
    "luna_n5_medium": ("N5 medium", "#EF3B2C", "D"),
    "luna_n5_low": ("N5 low", "#FC9272", "D"),
}
EXPECTED_EFFORT_ARMS = ("luna_single_xhigh", *EFFORT_METHODS)
EFFORT_AGGREGATE = Path(__file__).parent.parent / "luna_browsercode_effort_om2w300_20261030/aggregate.json"



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



def read_effort_rows(path):
    """Accept only complete recorded panels with explicit provisional status."""
    data = json.loads(path.read_text())
    if data.get("final_audit_complete") is not False or data.get("provisional") is not True:
        raise ValueError("The new effort cohort must retain its pending-audit qualification")
    indexed = {row["arm"]: row for row in data["aggregate_rows"]}
    if set(indexed) != set(EXPECTED_EFFORT_ARMS) or len(data["aggregate_rows"]) != len(indexed):
        raise ValueError("Exactly five distinct BrowserCode effort methods required")
    rows = []
    for arm in EXPECTED_EFFORT_ARMS:
        row = indexed[arm]
        if row["recorded"] != 300 or row["valid"] + row["invalid"] != 300:
            raise ValueError("Complete 300-disposition effort panels required")
        if not 0 <= row["successes"] <= row["valid"] or abs(row["success_rate"] - row["successes"] / row["valid"]) > 1e-10:
            raise ValueError("Effort success denominator mismatch")
        for key, _, _ in METRICS:
            if not math.isfinite(row[key]) or row[key] <= 0:
                raise ValueError("Finite positive effort metrics required")
            coverage = row[key + "_observed_tasks"]
            if type(coverage) is not int or not 0 < coverage <= 300:
                raise ValueError("Explicit observed-task coverage required")
        if arm in EFFORT_METHODS:
            rows.append(dict(row, method=arm, mean_success_rate=row["success_rate"]))
    return data, rows


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)



METRICS = (("mean_model_usd_per_task", "Model cost / task ($; log)", "cost"),
           ("mean_browser_steps", "Browser commands / task", "browser_steps"),
           ("mean_task_seconds", "Task latency (s; log)", "latency"))


def draw(ax, rows, metric, native):
    """Four observed points; method identities live in the side legend."""
    ax.set_ylim((68, 80) if native else (61, 72))
    ax.set_yticks([68, 72, 76, 80] if native else [61, 64, 68, 72])
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
        ax.scatter(x, y, s=42 if method == "parallel5_select_high" else 30,
                   marker=marker, color=color,
                   edgecolor="white", linewidth=0.5, zorder=3)


def draw_merged(ax, high, effort, metric):
    """Overlay numeric points without pooling judge cohorts or task denominators."""
    ax.set_ylim(30, 78)
    ax.set_yticks([30, 40, 50, 60, 70])
    ax.grid(False, which="both")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=3, width=0.6, pad=2)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#555555")
        ax.spines[side].set_linewidth(0.6)
    if metric == "mean_model_usd_per_task":
        ax.set_xscale("log")
        ax.set_xlim(0.0012, 0.4)
        ax.set_xticks([0.002, 0.01, 0.05, 0.2])
        ax.xaxis.set_major_formatter(StrMethodFormatter("${x:g}"))
    elif metric == "mean_browser_steps":
        ax.set_xlim(4.4, 19)
        ax.set_xticks([6, 10, 14, 18])
    else:
        ax.set_xscale("log")
        ax.set_xlim(27, 360)
        ax.set_xticks([40, 100, 250])
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    for recent, rows in ((False, high), (True, effort)):
        styles = EFFORT_METHODS if recent else METHODS
        for row in rows:
            method = row["method"]
            _, color, marker = styles[method]
            x, y = row[metric], 100 * row["mean_success_rate"]
            ax.scatter(x, y, s=40 if ("n5" in method or method == "parallel5_select_high") else 30,
                       marker=marker, facecolor="white" if recent else color,
                       edgecolor=color if recent else "white", linewidth=1.0 if recent else 0.5, zorder=3)


def legend_handles(rows, recent=False):
    styles = EFFORT_METHODS if recent else METHODS
    high_labels = {"serial_high": "Sol single high", "luna_single_high": "Luna single high",
                   "parallel3_select_high": "Luna N3 high", "parallel5_select_high": "Luna N5 high"}
    effort_labels = {"luna_single_medium": "Luna single medium",
                     "luna_single_low": "Luna single low", "luna_n5_medium": "Luna N5 medium + medium",
                     "luna_n5_low": "Luna N5 low + low"}
    labels = effort_labels if recent else high_labels
    return [Line2D([], [], linestyle="none", marker=styles[row["method"]][2], markersize=5,
                   markerfacecolor="white" if recent else styles[row["method"]][1],
                   markeredgecolor=styles[row["method"]][1], markeredgewidth=1 if recent else 0.5,
                   label=labels[row["method"]]) for row in rows]


def figure(means, native, metrics, output, stem, effort=()):
    columns = len(metrics)
    fig, axes = plt.subplots(2, columns,
                             figsize=(9.1 if columns == 3 else 5.4, 4.4),
                             gridspec_kw={"height_ratios": [1.35, 1]} if effort else {},
                             sharey="row", squeeze=False)
    fig.subplots_adjust(left=0.065 if columns == 3 else 0.12,
                        right=0.745 if columns == 3 else 0.60,
                        top=0.89, bottom=0.105, hspace=0.67, wspace=0.24)
    cohorts = [(means, "a)  BrowserCode", False), (native, "b)  Native CUAWright", True)]
    for ri, (rows, title, native_row) in enumerate(cohorts):
        pos = axes[ri, 0].get_position()
        fig.text(pos.x0, pos.y1 + 0.035, title, fontsize=8.5, weight="bold")
        for ci, (metric, label, _) in enumerate(metrics):
            if effort and not native_row:
                draw_merged(axes[ri, ci], means, effort, metric)
            else:
                draw(axes[ri, ci], rows, metric, native=native_row)
            axes[ri, ci].set_xlabel(label, labelpad=3)
            if ci == 0:
                axes[ri, ci].set_ylabel("Valid-only success (%)", labelpad=3)
    legend_x = 0.77 if columns == 3 else 0.65
    groups = [(means, False, "BrowserCode · historical (o4-mini)", 0.94)]
    if effort:
        groups.append((effort, True, "BrowserCode · Terra (provisional)", 0.68))
    groups.append((native, False, "Native CUAWright · high", 0.34))
    for rows, recent, title, top in groups:
        fig.legend(handles=legend_handles(rows, recent), title=title,
                   loc="upper left", bbox_to_anchor=(legend_x, top), frameon=False,
                   fontsize=7, title_fontsize=7.3, handletextpad=0.55,
                   labelspacing=0.5, borderaxespad=0)
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
    parser.add_argument("--effort-aggregate", type=Path, default=EFFORT_AGGREGATE)
    args = parser.parse_args()
    comparison, means, native = read_rows(args.aggregate)
    effort_data, effort = read_effort_rows(args.effort_aggregate)
    data = json.loads(Path(__file__).with_name("native_cuawright_completed.json").read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "om2w_harness_tradeoff_coordinates.csv",
              [dict(harness=h, **{key: value for key, value in r.items() if not isinstance(value, (dict, list))}) for h, rows in (("BrowserCode", means), ("BrowserCode effort (provisional)", effort), ("Native CUAWright", native)) for r in rows])
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7,
                        "axes.labelsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                        "svg.fonttype": "none", "pdf.fonttype": 42,
                        "savefig.facecolor": "white"})
    for metric in METRICS:
        figure(means, native, [metric], args.output_dir, f"om2w_harness_vs_{metric[2]}", effort=effort)
    figure(means, native, METRICS, args.output_dir, "om2w_harness_tradeoffs", effort=effort)
    (args.output_dir / "om2w_harness_tradeoffs_provenance.json").write_text(json.dumps({
        "comparison_aggregate_sha256": hashlib.sha256(args.aggregate.read_bytes()).hexdigest(),
        "native_status_aggregate_sha256": hashlib.sha256(Path(__file__).with_name("native_cuawright_completed.json").read_bytes()).hexdigest(),
        "effort_aggregate_sha256": hashlib.sha256(args.effort_aggregate.read_bytes()).hexdigest(),
        "effort_status": "Provisional; collection complete, final independent audit pending",
        "excluded_from_plot": ["luna_single_xhigh"],
        "color_encoding": {"single": {"high": "#08519C", "medium": "#4292C6", "low": "#9ECAE1"}, "candidate_plus_selection": {"high": "#A50F15", "medium": "#EF3B2C", "low": "#FC9272"}},
        "effort_final_audit_complete": effort_data["final_audit_complete"],
        "effort_judge": "gpt56-terra-terminal-success-v1",
        "effort_metric_coverage": {row["arm"]: {key: value for key, value in row.items() if key.endswith("_observed_tasks") or key in ("unknown_model_usage_calls", "derived_invalid_count")} for row in effort},
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
        "effort_cost_scope": "All-attempt model ledger charges plus full unknown reserves, including helpers/counters, divided by all300 dispositions; external judge excluded. This cost coordinate differs from valid-task efficiency.",
        "effort_latency_and_commands_scope": "All finite recorded measurements, including invalid outcomes; metric-specific coverage, no zero filling; actor-loop latency excludes setup, final judge-only capture, cleanup and external judge.",
        "visualization": "Merged BrowserCode row overlays four historical high-effort points (filled; o4-mini) and four recent medium/low effort points (open; Terra; provisional). Luna single xhigh is omitted from plotting; its source aggregate remains complete. Top y-axis30–78%, native68–80%; shared scales within each row. Method and cohort legends sit to the right, outside every data panel; no direct point labels, leader lines, method-connecting curves, frontiers or grid. Native coordinates unchanged. Blue encodes single-call policies and red encodes candidate-plus-selection policies; high is dark, medium intermediate and low light. Squares=Sol single, circles=Luna single, triangles=N3 and diamonds=N5.",
        "comparability": "Different judges, limits, collection windows, contention, efficiency coverage and task-time boundaries; descriptive only. Overlay does not imply paired or causal effort effects across cohorts; three-benchmark averages exclude the OM2W-only effort cohort.",
        "uncertainty": "Point estimates; no uncertainty inferred from aggregate data"
    }, indent=2) + "\n")
    print(json.dumps({"browsercode": means, "effort_provisional": effort, "native": native}, indent=2))

if __name__ == "__main__":
    main()
