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


def compact_layout(fig):
    """Narrow this weighted highlight only; keep the paper font sizes."""
    fig.set_size_inches(5.8, 2.55)
    fig.subplots_adjust(left=.10, right=.985, top=.97, bottom=.57, wspace=.08)
    fig.axes[0].yaxis.set_label_coords(-.20, .30)
    previous = list(fig.legends)
    for legend in previous:
        legend.remove()
    for j, old in enumerate(previous[:2]):
        fig.legend(handles=old.get_lines(), labels=[t.get_text() for t in old.texts],
                   title=old.get_title().get_text(), loc="upper left",
                   bbox_to_anchor=(.10 if j == 0 else .565, 1.04 / fig.get_figheight()),
                   ncol=2, fontsize=8, title_fontsize=9, frameon=False,
                   handlelength=.8, handletextpad=.4, columnspacing=.7,
                   labelspacing=.3, borderaxespad=0, alignment="left")
    old = previous[2]
    fig.legend(handles=old.get_lines(), labels=[t.get_text() for t in old.texts],
               loc="lower left", bbox_to_anchor=(.10, .04 / fig.get_figheight()),
               frameon=False, fontsize=8, handlelength=1.8, handletextpad=.35, borderaxespad=0)
    for text in list(fig.texts):
        text.remove()
    footer = BASE.PAPER.FOOTER.replace("high effort · ", "high effort\n")
    fig.text(.40, .025 / fig.get_figheight(), footer, fontsize=7.2, color="#555555", va="bottom")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [legend.get_window_extent(renderer) for legend in fig.legends]
    assert all(not a.overlaps(b) for i, a in enumerate(boxes) for b in boxes[i + 1:])
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
    compact_layout(fig)
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
                    layout="5.8 x 2.55 inches; wspace 0.08; two-column legend groups; wrapped footer; no gridlines",
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
