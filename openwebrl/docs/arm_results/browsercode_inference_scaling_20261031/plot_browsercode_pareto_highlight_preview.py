#!/usr/bin/env python3
"""Separate weighted Pareto preview: hollow 30-step, filled 60/120-step markers.

Uses standard-size dark-red filled markers and directly labels N5/60; preserves all input coordinates, axes, frontier
curves and published original figures. No model, browser or experiment calls.
"""
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
STEM = "browsercode_scaling_weighted_pareto_highlight_preview"
SOURCE = HERE / "plot_browsercode_pareto_previews.py"
spec = importlib.util.spec_from_file_location("base_pareto", SOURCE)
BASE = importlib.util.module_from_spec(spec)
spec.loader.exec_module(BASE)
LONG = {"luna_single_high_60steps", "luna_single_high_120steps", "parallel5_select_high_60steps", BASE.SOL60_METHOD}
FOCUS = "parallel5_select_high_60steps"
SIZE = 27


def build():
    figures = BASE.build()
    fig, _, frontier_report = figures[1]
    for other, _, _ in (figures[0], figures[2]):
        plt.close(other)
    data = json.loads((HERE / "browsercode_scaling_weighted_average.json").read_text())
    rows = [*data["points"], BASE.sol60_weighted_point()]
    label_to_method = {label: method for method, label in BASE.SINGLE_LEGEND + BASE.SELECTION_LEGEND}
    label_to_method[BASE.SOL60_STYLE[0]] = BASE.SOL60_METHOD
    BASE.PAPER.apply_style(fig, [[row["method"] for row in rows]] * len(fig.axes),
                          BASE.SCALING.STYLES, label_to_method, weighted=True)
    return fig, frontier_report


def main():
    fig, frontier_report = build()
    report = dict(
        schema="browsercode-weighted-pareto-highlight-preview-v1",
        population="Original own-valid planned-task-weighted benchmark means; weights 300/1045, 595/1045, 150/1045",
        styles=dict(hollow="30-decision limit", filled="60/120-decision limit",
                    highlight=FOCUS, highlight_size_points_squared=SIZE,
                    baseline_size_points_squared=27, highlight_label="Luna ADM",
                    highlight_legend_label="Luna ADM; N=5, 60 steps",
                    layout="Shorter panels; wspace 0.12; no gridlines",
                    typography="9 pt axes/ticks/direct labels; 8 pt legends; 9 pt legend titles; 7.2 pt footer",
                    selection_title="Best-of-N action decision scaling",
                    actor_family="GPT-6 (Luna and Sol)",
                    baseline_definition="Luna baseline: high effort, max output tokens = 12,288 per call, 30-step limit",
                    colors="Original colors retained; no new halo"),
        source_sha256={name: BASE.sha(HERE / name) for name in (
            "aggregate.json", "browsercode_scaling_weighted_average.json",
            "plot_browsercode_pareto_previews.py", "browsercode_pareto_paper_style.py",
            "../browsercode_sol_high60_20261104/aggregate.json")},
        renderer_sha256=BASE.sha(__file__), frontier=frontier_report)
    BASE.save(fig, STEM)
    report["output_sha256"] = {extension: BASE.sha(HERE / f"{STEM}.{extension}")
                               for extension in ("png", "svg", "pdf")}
    (HERE / f"{STEM}.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
