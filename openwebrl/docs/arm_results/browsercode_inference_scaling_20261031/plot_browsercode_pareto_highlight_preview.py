#!/usr/bin/env python3
"""Separate weighted Pareto preview: hollow 30-step, filled 60/120-step markers.

Enlarges and directly labels N5/60; preserves all input coordinates, axes, frontier
curves and published original figures. No model, browser or experiment calls.
"""
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
STEM = "browsercode_scaling_weighted_pareto_highlight_preview"
SOURCE = HERE / "plot_browsercode_pareto_previews.py"
spec = importlib.util.spec_from_file_location("base_pareto", SOURCE)
BASE = importlib.util.module_from_spec(spec)
spec.loader.exec_module(BASE)
LONG = {"luna_single_high_60steps", "luna_single_high_120steps", "parallel5_select_high_60steps", BASE.SOL60_METHOD}
FOCUS = "parallel5_select_high_60steps"
SIZE = 54


def build():
    figures = BASE.build()
    fig, _, frontier_report = figures[1]
    for other, _, _ in (figures[0], figures[2]):
        plt.close(other)
    data = json.loads((HERE / "browsercode_scaling_weighted_average.json").read_text())
    rows = [*data["points"], BASE.sol60_weighted_point()]
    for j, (ax, metric) in enumerate(zip(fig.axes, BASE.METRICS)):
        assert len(ax.collections) == len(rows) + 1
        for point, collection in zip(rows, ax.collections[:len(rows)]):
            method = point["method"]
            _, _, color, _ = BASE.SCALING.STYLES[method]
            expected = (BASE.coordinate(point, metric), 100 * point["success_rate"])
            assert np.allclose(collection.get_offsets()[0], expected)
            collection.set_facecolor(color if method in LONG else "white")
            collection.set_edgecolor(color)
            if method == FOCUS:
                collection.set_sizes([SIZE])
                collection.set_zorder(5)
                collection.set_linewidths([1.1])
                offset, align = ((8, 8), "left") if j == 1 else ((-7, 8), "right")
                ax.annotate("N=5, 60 steps", expected, xytext=offset,
                            textcoords="offset points", color=color, fontsize=7,
                            ha=align, va="bottom", zorder=6)
    label_to_method = {label: method for method, label in BASE.SINGLE_LEGEND + BASE.SELECTION_LEGEND}
    label_to_method[BASE.SOL60_STYLE[0]] = BASE.SOL60_METHOD
    for legend in fig.legends[:2]:
        for line, text in zip(legend.get_lines(), legend.texts):
            method = label_to_method[text.get_text()]
            color = BASE.SCALING.STYLES[method][2]
            line.set_markerfacecolor(color if method in LONG else "white")
            if method == FOCUS:
                line.set_markersize(np.sqrt(SIZE))
                line.set_markeredgewidth(1.1)
    return fig, frontier_report


def main():
    fig, frontier_report = build()
    report = dict(
        schema="browsercode-weighted-pareto-highlight-preview-v1",
        population="Original own-valid planned-task-weighted benchmark means; weights 300/1045, 595/1045, 150/1045",
        styles=dict(hollow="30-decision limit", filled="60/120-decision limit",
                    highlight=FOCUS, highlight_size_points_squared=SIZE,
                    baseline_size_points_squared=27, highlight_label="N=5, 60 steps",
                    colors="Original colors retained; no new halo"),
        source_sha256={name: BASE.sha(HERE / name) for name in (
            "aggregate.json", "browsercode_scaling_weighted_average.json",
            "plot_browsercode_pareto_previews.py",
            "../browsercode_sol_high60_20261104/aggregate.json")},
        renderer_sha256=BASE.sha(__file__), frontier=frontier_report)
    BASE.save(fig, STEM)
    report["output_sha256"] = {extension: BASE.sha(HERE / f"{STEM}.{extension}")
                               for extension in ("png", "svg", "pdf")}
    (HERE / f"{STEM}.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
