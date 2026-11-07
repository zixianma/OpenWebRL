#!/usr/bin/env python3
"""Separate fourteen-setting Pareto preview with right legends and dotted caches.

Reuses the frozen highlight builder, retaining all own-valid coordinates, axis
ranges, colors, base marker sizes and observed frontiers. No experiment calls.
"""
import copy
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import Circle

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "plot_browsercode_pareto_highlight_preview.py"
spec = importlib.util.spec_from_file_location("original_highlight", SOURCE)
ORIGINAL = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ORIGINAL)
STEM = "browsercode_scaling_weighted_pareto_highlight_right_legend_preview"
DOTS = (0, (0.1, 1.3))
CACHE_EDGE = "#222222"


def cache_handle(color="white"):
    return Circle((0, 0), radius=1, facecolor=color, edgecolor=CACHE_EDGE,
                  linewidth=1.1, linestyle=DOTS, capstyle="round")


def legend_circle(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
    return Circle((width / 2 - xdescent, height / 2 - ydescent),
                  radius=min(width, height) * .46, capstyle="round")


def right_layout(fig):
    previous = list(fig.legends)
    assert len(previous) == 3
    for legend in previous:
        legend.remove()
    for text in list(fig.texts):
        text.remove()
    fig.set_size_inches(8.1, 2.65)
    fig.subplots_adjust(left=.072, right=.66, top=.96, bottom=.27, wspace=.12)
    fig.axes[0].yaxis.set_label_coords(-.25, .5)
    sidebar_x, top = .705, .96
    handlers = {Circle: HandlerPatch(patch_func=legend_circle)}
    cached_labels = {style[0]: style[2] for style in ORIGINAL.CACHED.values()}
    for j, old in enumerate(previous[:2]):
        handles, labels = [], [text.get_text() for text in old.texts]
        for line, label in zip(old.get_lines(), labels):
            handles.append(cache_handle(cached_labels[label]) if label in cached_labels
                           else copy.copy(line))
        title = old.get_title().get_text() if j == 0 else "Best-of-N action\ndecision scaling"
        legend = fig.legend(handles, labels, title=title, loc="upper left",
                            bbox_to_anchor=(sidebar_x, top), ncol=2, fontsize=8,
                            title_fontsize=9, frameon=False, handlelength=.9,
                            handletextpad=.4, columnspacing=.8, labelspacing=.35,
                            borderaxespad=0, alignment="left", handler_map=handlers)
        fig.canvas.draw()
        top = legend.get_window_extent(fig.canvas.get_renderer()).y0 / fig.bbox.height - .10 / fig.get_figheight()
    frontier = copy.copy(previous[2].get_lines()[0])
    fig.legend([frontier, cache_handle()], ["Observed frontier", "Dotted edge: cached"],
               loc="upper left", bbox_to_anchor=(sidebar_x, top), ncol=1,
               fontsize=8, frameon=False, handlelength=1.8, handletextpad=.4,
               labelspacing=.35, borderaxespad=0, handler_map=handlers)
    fig.text(.072, .025, ORIGINAL.BASE.PAPER.FOOTER.replace("high effort · ", "high effort\n"),
             fontsize=7.2, color="#555555", va="bottom")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legends = [legend.get_window_extent(renderer) for legend in fig.legends]
    assert all(not a.overlaps(b) for i, a in enumerate(legends) for b in legends[i + 1:])
    for box in legends:
        assert all(not box.overlaps(ax.get_tightbbox(renderer)) for ax in fig.axes)
        assert not box.overlaps(fig.texts[0].get_window_extent(renderer))
    labels = list(fig.texts)
    for legend in fig.legends:
        labels.extend([legend.get_title(), *legend.texts])
    for ax in fig.axes:
        labels.extend([ax.title, ax.xaxis.label, ax.yaxis.label, *ax.texts])
        for axis, limits in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if min(limits) <= tick.get_loc() <= max(limits):
                    labels.extend([tick.label1, tick.label2])
    for text in labels:
        if text.get_visible() and text.get_text():
            box = text.get_window_extent(renderer)
            assert fig.bbox.contains(*box.p0) and fig.bbox.contains(*box.p1), text.get_text()
    assert [len(legend.texts) for legend in fig.legends] == [8, 6, 2]


def build():
    fig, frontier, rows = ORIGINAL.build()
    assert len(rows) == 14
    for ax in fig.axes:
        for row, point in zip(rows, ax.collections[:len(rows)]):
            if row["method"] in ORIGINAL.CACHED:
                point.set_linestyle(DOTS)
                point.set_linewidths([1.1])
                point.set_capstyle("round")
    right_layout(fig)
    return fig, frontier, rows


def main():
    fig, frontier, rows = build()
    original_report = json.loads(SOURCE.with_name(ORIGINAL.STEM + ".json").read_text())
    assert rows == original_report["points"]
    assert frontier == original_report["frontier"]
    report = dict(schema="browsercode-weighted-pareto-highlight-right-legend-v1",
                  population=original_report["population"],
                  comparison_caveats=original_report["comparison_caveats"], points=rows,
                  frontier=frontier,
                  styles=dict(layout="8.1 x 2.65 inches; right-side legend groups; no gridlines",
                              cached="Filled original red circle, black dotted edge; matching method symbols and explicit key",
                              dots=dict(offset=DOTS[0], pattern=list(DOTS[1]), linewidth=1.1, capstyle="round"),
                              frontier="Original gray solid rings and dashed links, unchanged",
                              typography="Albert Sans; 9 pt axes/ticks/direct labels; 8 pt legends; 9 pt legend titles; 7.2 pt footer"),
                  source_sha256={**original_report["source_sha256"],
                                 SOURCE.name: ORIGINAL.BASE.sha(SOURCE),
                                 ORIGINAL.STEM + ".json": ORIGINAL.BASE.sha(SOURCE.with_name(ORIGINAL.STEM + ".json"))},
                  renderer_sha256=ORIGINAL.BASE.sha(__file__))
    ORIGINAL.BASE.save(fig, STEM)
    report["output_sha256"] = {ext: ORIGINAL.BASE.sha(HERE / f"{STEM}.{ext}")
                               for ext in ("png", "pdf", "svg")}
    (HERE / f"{STEM}.json").write_text(json.dumps(report, indent=2) + "\n")
    assert b"/FontFile2" in (HERE / f"{STEM}.pdf").read_bytes()
    plt.close(fig)


if __name__ == "__main__":
    main()
