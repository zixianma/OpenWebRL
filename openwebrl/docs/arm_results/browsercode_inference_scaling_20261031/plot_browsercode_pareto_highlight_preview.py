#!/usr/bin/env python3
"""Weighted Pareto highlight with completed cached and Sol settings.

Adds completed cached-context and Sol own-valid points to the original twelve
settings. Retains their coordinates, recomputes observed frontiers, and preserves
all other published figures. No model, browser or experiment calls.
"""
import importlib.util
import json
import math
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/openwebrl-sol-highlight-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/openwebrl-sol-highlight-cache")
from pathlib import Path
from tempfile import mkdtemp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# Shared paper typography, including bundled Albert Sans fonts.
import sys as _paper_sys
from pathlib import Path as _PaperPath
_paper_sys.path.insert(0, str(next(p for p in _PaperPath(__file__).resolve().parents
                                if (p / "paper_plot_style.py").is_file())))
import paper_plot_style as _paper_style


HERE = Path(__file__).resolve().parent
STEM = "browsercode_scaling_weighted_pareto_highlight_preview"
SOURCE = HERE / "plot_browsercode_pareto_previews.py"
spec = importlib.util.spec_from_file_location("base_pareto", SOURCE)
BASE = importlib.util.module_from_spec(spec)
spec.loader.exec_module(BASE)
LONG = {"luna_single_high_60steps", "luna_single_high_120steps", "parallel5_select_high_60steps", BASE.SOL60_METHOD}
FOCUS = "parallel5_select_high_60steps"
SIZE = 27
CACHED_SOURCE = HERE.parent / "browsercode_cached_selection_20261105" / "aggregate.json"
CACHED_RENDERER = HERE.parent / "browsercode_cached_selection_20261105" / "plot_browsercode_scaling14.py"
SOL_SOURCE = HERE / "sol_high120_xhigh30_aggregate.json"
SOL = {
    "sol_single_high120": ("Sol high · 120 steps", "s", "#08306b", True),
    "sol_single_xhigh30": ("Sol xhigh · 30 steps", "s", "#08306b", False),
}
SOL_PROTOCOL = dict(actor_model="gpt-6-sol", candidates=1, max_output_tokens=12288,
                    actor_deadline_seconds=900, harness="browsercode",
                    judge_model="gpt-5.6-terra", judge_effort="medium",
                    judge_max_output_tokens=4096, judge_seed=42)
CACHED = {
    "luna_n3_cached_high30": ("N=3/30 cached", "o", "#ef8a8a", True),
    "luna_n5_cached_high60": ("N=5/60 cached", "o", "#a50f15", True),
}


def cached_weighted_points():
    """Use only the six own-valid cached cells, never the repaired baselines."""
    spec = importlib.util.spec_from_file_location("cached_aggregate_validator", CACHED_RENDERER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = module.validate(json.loads(CACHED_SOURCE.read_text()))
    result = []
    for method in CACHED:
        cells = {p["benchmark"]: p for p in data["own_valid_points"] if p["method"] == method}
        assert set(cells) == set(BASE.SCALING.BENCHMARKS)
        counts = {b: dict(requested=p["planned"], native_saved=p["recorded"],
                         valid=p["valid"], invalid=p["invalid"], successes=p["successes"],
                         missing_native=0) for b, p in cells.items()}
        result.append(dict(
            method=method, benchmark_count=3,
            success_rate=math.fsum(n * cells[b]["success_rate"] for b, (_, n) in BASE.SCALING.BENCHMARKS.items()) / 1045,
            metrics={metric: math.fsum(n * cells[b]["metrics"][metric]["mean"]
                                      for b, (_, n) in BASE.SCALING.BENCHMARKS.items()) / 1045
                     for metric in BASE.METRICS},
            benchmark_counts=counts))
    return result


def completed_sol_points():
    """Accept only six complete, sanitized, native-qualified own-valid cells."""
    data = json.loads(SOL_SOURCE.read_text())
    assert set(data) == {"schema", "status", "methods", "protocols", "benchmarks",
                         "source_sha256", "comparison_caveats"}
    assert data["schema"] == "sol-high120-xhigh30-complete-own-valid-aggregate-v1"
    assert data["status"] == "VERIFIED_COMPLETE_WITH_PRESERVED_INVALIDS"
    assert data["methods"] == list(SOL)
    assert set(data["protocols"]) == set(SOL)
    assert set(data["source_sha256"]) == {"high120_native_endpoint", "xhigh30_native_endpoint", "private_projection_inputs"}
    assert all(type(h) is str and len(h) == 64 and all(c in "0123456789abcdef" for c in h)
               for h in data["source_sha256"].values())
    assert data["comparison_caveats"] == [
        "Own-valid cohorts and live collection windows differ; no paired, causal or significance claim is inferred.",
        "Fixed planned weights300/1045,595/1045,150/1045 apply to benchmark success and actor means; success is not pooled across valid tasks.",
        "Actor means include valid failures and exclude Terra, hosting and invalid-outcome spending; latency is measured actor-loop elapsed time."]
    assert set(data["benchmarks"]) == set(SOL)
    result = []
    for method in SOL:
        limit, effort = (120, "high") if method == "sol_single_high120" else (30, "xhigh")
        assert data["protocols"][method] == dict(SOL_PROTOCOL, reasoning_effort=effort,
                                               maximum_decisions_advertised=limit,
                                               maximum_decisions_enforced=limit)
        cells = data["benchmarks"][method]
        assert set(cells) == set(BASE.SCALING.BENCHMARKS)
        counts = {}
        for b, (_, planned) in BASE.SCALING.BENCHMARKS.items():
            c = cells[b]
            assert set(c) == {"planned", "recorded", "valid", "invalid", "successes", "success_rate", "metrics"}
            assert all(type(c[k]) is int and c[k] >= 0 for k in ("planned", "recorded", "valid", "invalid", "successes"))
            assert c["planned"] == c["recorded"] == planned and c["valid"] + c["invalid"] == planned
            assert 0 < c["valid"] and 0 <= c["successes"] <= c["valid"]
            assert type(c["success_rate"]) in (int, float) and math.isclose(c["success_rate"], c["successes"] / c["valid"], rel_tol=0, abs_tol=1e-12)
            assert set(c["metrics"]) == set(BASE.METRICS)
            for name, value in c["metrics"].items():
                assert set(value) == {"mean", "observed_tasks", "missing_valid_tasks"}
                assert type(value["observed_tasks"]) is int and value["observed_tasks"] == c["valid"]
                assert type(value["missing_valid_tasks"]) is int and value["missing_valid_tasks"] == 0
                assert type(value["mean"]) in (int, float) and math.isfinite(value["mean"]) and value["mean"] >= 0
                assert value["mean"] <= limit if name == "browser_commands" else value["mean"] > 0
            counts[b] = dict(requested=planned, native_saved=c["recorded"], valid=c["valid"],
                             invalid=c["invalid"], successes=c["successes"], missing_native=0)
        result.append(dict(method=method, benchmark_count=3,
            success_rate=math.fsum(n * cells[b]["success_rate"] for b, (_, n) in BASE.SCALING.BENCHMARKS.items()) / 1045,
            metrics={metric: math.fsum(n * cells[b]["metrics"][metric]["mean"]
                                      for b, (_, n) in BASE.SCALING.BENCHMARKS.items()) / 1045 for metric in BASE.METRICS},
            benchmark_counts=counts))
    return result


def compact_layout(fig):
    """Narrow this weighted highlight only; keep the paper font sizes."""
    fig.set_size_inches(6.8, 2.75)
    fig.subplots_adjust(left=.10, right=.985, top=.97, bottom=.60, wspace=.08)
    fig.axes[0].yaxis.set_label_coords(-.20, .30)
    previous = list(fig.legends)
    for legend in previous:
        legend.remove()
    for j, old in enumerate(previous[:2]):
        fig.legend(handles=old.get_lines(), labels=[t.get_text() for t in old.texts],
                   title=old.get_title().get_text(), loc="upper left",
                   bbox_to_anchor=(.10 if j == 0 else .565, 1.18 / fig.get_figheight()),
                   ncol=2, fontsize=8, title_fontsize=9, frameon=False,
                   handlelength=.8, handletextpad=.4, columnspacing=.7,
                   labelspacing=.3, borderaxespad=0, alignment="left")
    old = previous[2]
    fig.legend(handles=old.get_lines(), labels=[t.get_text() for t in old.texts],
               loc="lower left", bbox_to_anchor=(.10, .04 / fig.get_figheight()),
               frameon=False, fontsize=8, handlelength=1.8, handletextpad=.35, borderaxespad=0)
    for text in list(fig.texts):
        text.remove()
    footer = "GPT-6 Luna & Sol: high unless labeled xhigh.\nLuna baseline: 12,288 tokens · 30 steps. N=candidates."
    fig.text(.40, .025 / fig.get_figheight(), footer, fontsize=7.2, color="#555555", va="bottom")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # Place the two groups using their measured text widths and heights.
    # Extra model labels can change both dimensions without changing font sizes.
    group_boxes = [legend.get_window_extent(renderer) for legend in fig.legends[:2]]
    widths = [box.width / fig.dpi for box in group_boxes]
    legend_top = max(box.height / fig.dpi for box in group_boxes) + .32
    width = max(6.8, (sum(widths) + .15) / .90)
    height = max(3.0, legend_top + 1.50)
    fig.set_size_inches(width, height)
    fig.subplots_adjust(bottom=(legend_top + .45) / height)
    fig.legends[0].set_bbox_to_anchor((.08, legend_top / height))
    fig.legends[1].set_bbox_to_anchor((.08 + (widths[0] + .15) / width, legend_top / height))
    fig.legends[2].set_bbox_to_anchor((.08, .04 / height))
    fig.texts[0].set_position((.40, .025 / height))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [legend.get_window_extent(renderer) for legend in fig.legends]
    assert all(not a.overlaps(b) for i, a in enumerate(boxes) for b in boxes[i + 1:]), [box.bounds for box in boxes]
    footer_box = fig.texts[0].get_window_extent(renderer)
    assert all(not footer_box.overlaps(box) for box in boxes)
    labels = list(fig.texts)
    for legend in fig.legends:
        labels.extend([legend.get_title(), *legend.texts])
    for ax in fig.axes:
        labels.extend([ax.xaxis.label, ax.yaxis.label, ax.title, *ax.texts])
        for axis, limits in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            # Log axes retain off-range ticks whose text is never drawn.
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if min(limits) <= tick.get_loc() <= max(limits):
                    labels.extend([tick.label1, tick.label2])
    for text in labels:
        if text.get_visible() and text.get_text():
            box = text.get_window_extent(renderer)
            assert fig.bbox.contains(*box.p0) and fig.bbox.contains(*box.p1), text.get_text()


def build():
    raw = BASE.SCALING.validate(json.loads((HERE / "aggregate.json").read_text()))
    data = BASE.AVERAGE.average(raw, "task-count")
    assert data == json.loads((HERE / "browsercode_scaling_weighted_average.json").read_text())
    rows = [*data["points"], BASE.sol60_weighted_point(), *cached_weighted_points(), *completed_sol_points()]
    for styles in (BASE.SCALING.STYLES, BASE.AVERAGE.SCALING.STYLES):
        styles.update(CACHED)
        styles.update(SOL)
    # Extend this in-memory renderer only; other published previews stay frozen.
    BASE.SINGLE_LEGEND = tuple(p for p in BASE.SINGLE_LEGEND if p[0] not in SOL) + tuple((method, style[0]) for method, style in SOL.items())
    BASE.PAPER.LONG = set(BASE.PAPER.LONG) | {"sol_single_high120"}
    BASE.SELECTION_LEGEND = tuple(p for p in BASE.SELECTION_LEGEND if p[0] not in CACHED)
    BASE.SELECTION_LEGEND += tuple((method, style[0]) for method, style in CACHED.items())
    temporary = Path(mkdtemp(prefix="openwebrl-cached-highlight-", dir="/tmp"))
    fig = BASE.capture(BASE.AVERAGE.render, {**data, "points": rows}, temporary, "source_weighted")
    frontier_report = BASE.overlay(fig, [("weighted_own_valid", rows, metric)
                                        for metric in BASE.METRICS], method_count=16)
    BASE.grouped_bottom_legends(fig, weighted=True)
    label_to_method = {label: method for method, label in BASE.SINGLE_LEGEND + BASE.SELECTION_LEGEND}
    label_to_method[BASE.SOL60_STYLE[0]] = BASE.SOL60_METHOD
    BASE.PAPER.apply_style(fig, [[row["method"] for row in rows]] * len(fig.axes),
                          BASE.SCALING.STYLES, label_to_method, weighted=True)
    for ax in fig.axes:
        for row, point in zip(rows, ax.collections):
            if row["method"] in CACHED:
                point.set_facecolor(CACHED[row["method"]][2])
                point.set_edgecolor("#222222")
                point.set_linewidths([1.0])
    for legend in fig.legends[:2]:
        for line, text in zip(legend.get_lines(), legend.texts):
            if text.get_text() in {style[0] for style in CACHED.values()}:
                method = label_to_method[text.get_text()]
                line.set_markerfacecolor(CACHED[method][2])
                line.set_markeredgecolor("#222222")
                line.set_markeredgewidth(1.0)
    compact_layout(fig)
    return fig, frontier_report, rows


def main():
    fig, frontier_report, rows = build()
    report = dict(
        schema="browsercode-weighted-pareto-highlight-preview-v3",
        population="Original twelve own-valid settings plus two completed cached-context and two completed Sol own-valid settings; weights 300/1045, 595/1045, 150/1045. Historical baseline coordinates are unchanged; gap-repaired baseline rows are not substituted.",
        comparison_caveats=["Own-valid tasks and live collection windows differ; no paired or causal cache effect is inferred.",
                           "Cached selection uses full native context; cached N5/60 advertises 60 while historical N5/60 advertised 30 and enforced 60.",
                           "All four Sol settings use GPT-6 Sol; high/30, high/60, high/120 and xhigh/30 retain their stated effort and advertised/enforced decision limits.",
                           "Actor means exclude Terra, hosting and invalid-outcome spending. Latency is elapsed per task, not summed parallel call time.",
                           "Selected browser-code batches may contain multiple browser operations and are not browser resource or fee measurements."],
        points=rows,
        styles=dict(hollow="30-decision limit, including Sol xhigh/30", filled="Historical 60/120, Sol high/120 and both cached settings; cached N3 is solid light red; Sol xhigh/30 is hollow",
                    highlight=FOCUS, highlight_size_points_squared=SIZE,
                    baseline_size_points_squared=27, highlight_label="Luna ADM",
                    highlight_legend_label="Luna ADM; N=5, 60 steps",
                    layout="5.8 x 2.75 inches; wspace 0.08; two-column legend groups; wrapped footer; no gridlines",
                    typography="9 pt axes/ticks/direct labels; 8 pt legends; 9 pt legend titles; 7.2 pt footer",
                    selection_title="Best-of-N action decision scaling",
                    actor_family="GPT-6 (Luna and Sol)",
                    baseline_definition="Luna baseline: high effort, max output tokens = 12,288 per call, 30-step limit",
                    colors="Original colors retained; cached N3 solid light red and cached N5 solid dark red, both with black edges"),
        source_sha256={name: BASE.sha(HERE / name) for name in (
            "aggregate.json", "browsercode_scaling_weighted_average.json",
            "plot_browsercode_pareto_previews.py", "browsercode_pareto_paper_style.py",
            "../browsercode_sol_high60_20261104/aggregate.json",
            "../browsercode_cached_selection_20261105/aggregate.json",
            "../browsercode_cached_selection_20261105/plot_browsercode_scaling14.py",
            "sol_high120_xhigh30_aggregate.json")},
        renderer_sha256=BASE.sha(__file__), frontier=frontier_report)
    BASE.save(fig, STEM)
    report["output_sha256"] = {extension: BASE.sha(HERE / f"{STEM}.{extension}")
                               for extension in ("png", "svg", "pdf")}
    (HERE / f"{STEM}.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
