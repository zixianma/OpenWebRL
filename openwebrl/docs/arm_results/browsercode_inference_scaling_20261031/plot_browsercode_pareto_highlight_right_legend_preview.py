#!/usr/bin/env python3
"""Separate sixteen-setting Pareto preview with right legends and shaded caches.

Reuses the frozen highlight builder, retaining all own-valid coordinates, axis
ranges, colors and observed frontiers. Cached markers use shaded circles. No experiment calls.
"""
import copy
import importlib.util
import json
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/openwebrl-sol-right-legend-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/openwebrl-sol-right-legend-cache")
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerPatch
from matplotlib.markers import MarkerStyle
from matplotlib.patches import Circle

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "plot_browsercode_pareto_highlight_preview.py"
spec = importlib.util.spec_from_file_location("original_highlight", SOURCE)
ORIGINAL = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ORIGINAL)
STEM = "browsercode_scaling_weighted_pareto_highlight_right_legend_preview"
CACHE_SIZE = 36
CACHE_HATCH = "////////"


def cache_handle(color="#8b8b8b"):
    return Circle((0, 0), radius=1, facecolor=color, edgecolor="white",
                  linewidth=0, hatch=CACHE_HATCH)


def legend_circle(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
    return Circle((width / 2 - xdescent, height / 2 - ydescent),
                  radius=min(width, height) * .46)


def right_layout(fig, previous=None, height=2.25):
    previous = list(fig.legends) if previous is None else previous
    assert len(previous) == 3
    for legend in list(fig.legends):
        legend.remove()
    for text in list(fig.texts):
        text.remove()
    fig.set_size_inches(8.8, height)
    fig.subplots_adjust(left=.065, right=.68, top=.95, bottom=.22, wspace=.10)
    for ax in fig.axes:
        ax.set_box_aspect(1)
    fig.axes[0].set_ylabel("")
    sidebar_x, top = .72, .95
    handlers = {Circle: HandlerPatch(patch_func=legend_circle)}
    cached_labels = {style[0]: style[2] for style in ORIGINAL.CACHED.values()}
    for j, old in enumerate(previous[:2]):
        handles, labels = [], [text.get_text() for text in old.texts]
        for line, label in zip(old.get_lines(), labels):
            handles.append(cache_handle(cached_labels[label]) if label in cached_labels
                           else copy.copy(line))
        if j == 0:
            index = labels.index("Sol high · 120 steps")
            handle, label = handles.pop(index), labels.pop(index)
            index = labels.index("Sol high · 60 steps") + 1
            handles.insert(index, handle)
            labels.insert(index, label)
        title = old.get_title().get_text() if j == 0 else "Best-of-N action decision scaling"
        legend = fig.legend(handles, labels, title=title, loc="upper left",
                            bbox_to_anchor=(sidebar_x, top), ncol=2, fontsize=8,
                            title_fontsize=9, frameon=False, handlelength=.9,
                            handletextpad=.4, columnspacing=.8, labelspacing=.10,
                            borderaxespad=0, alignment="left", handler_map=handlers)
        fig.canvas.draw()
        top = legend.get_window_extent(fig.canvas.get_renderer()).y0 / fig.bbox.height - .015 / fig.get_figheight()
    frontier = copy.copy(previous[2].get_lines()[0])
    encoding = fig.legend([frontier, cache_handle()], ["Observed frontier", "Cached context"],
                          loc="upper left", bbox_to_anchor=(sidebar_x, top), ncol=2,
                          fontsize=8, frameon=False, handlelength=1.2, handletextpad=.4,
                          columnspacing=.7, labelspacing=.10, borderaxespad=0,
                          handler_map=handlers)
    fig.canvas.draw()
    footer_top = encoding.get_window_extent(fig.canvas.get_renderer()).y0 / fig.bbox.height - .015 / fig.get_figheight()
    fig.text(sidebar_x, footer_top,
             "GPT-6 high unless xhigh. N=candidates.\nLuna baseline: 12,288 tokens · 30 steps.",
             fontsize=7.2, color="#555555", va="top")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    footer_box = fig.texts[0].get_window_extent(renderer)
    # Fit the added model row while retaining all original font sizes.
    if footer_box.y0 < 3:
        extra = (3 - footer_box.y0) / fig.dpi / .95 + .02
        assert height + extra < 3.5, "Sidebar unexpectedly exceeds compact figure height"
        return right_layout(fig, previous, height + extra)
    fig.text(.016, (.22 + .95) / 2, "Valid-only success (%)", rotation=90,
             fontsize=9, ha="center", va="center")
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
            assert fig.bbox.contains(*box.p0) and fig.bbox.contains(*box.p1), (text.get_text(), tuple(box.bounds), tuple(fig.bbox.bounds))
    for ax in fig.axes:
        box = ax.get_window_extent(renderer)
        assert abs(box.width - box.height) < .01, "Plotting area must be square"
    assert [len(legend.texts) for legend in fig.legends] == [10, 6, 2]


def build():
    fig, frontier, rows = ORIGINAL.build()
    matplotlib.rcParams["hatch.linewidth"] = .5
    assert len(rows) == 16
    circle = MarkerStyle("o")
    circle_path = circle.get_path().transformed(circle.get_transform())
    for ax in fig.axes:
        for row, point in zip(rows, ax.collections[:len(rows)]):
            if row["method"] in ORIGINAL.CACHED:
                point.set_paths([circle_path])
                point.set_sizes([CACHE_SIZE])
                point.set_edgecolors("white")
                point.set_linewidths([0])
                point.set_hatch(CACHE_HATCH)
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
                  styles=dict(layout=f"8.8 x {fig.get_figheight():.3f} inches; square panels; legends and footer on the right; no gridlines",
                              axes_box_aspect=1,
                              cached="Light/dark red circles with white diagonal shading and no outline; matching method symbols and a shaded gray circle key",
                              cached_marker=dict(marker="o", size_pt2=CACHE_SIZE, hatch=CACHE_HATCH, hatch_color="white", hatch_linewidth=.5, linewidth=0),
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
