#!/usr/bin/env python3
"""Separate sixteen-setting Pareto preview with right legends and shaded caches.

Reuses the frozen highlight builder, retaining all own-valid coordinates, axis
ranges and observed frontiers. Action decision scaling uses saturated purple accents and larger markers. Cached markers use shaded circles. No experiment calls.
"""
# Shared semantic palette; source measurements and geometry are unchanged.
from pathlib import Path as _PalettePath
import sys as _palette_sys
_palette_sys.path.insert(0, str(next(p for p in _PalettePath(__file__).resolve().parents
                                     if (p / "paper_plot_palette.py").is_file())))
from paper_plot_palette import COLORS as PALETTE, tint as palette_tint
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
SINGLE_GRAY = "#687580"
SOL_GREEN = PALETTE["ADM_4B"]
# Markers encode the intervention rather than relying on nearby gray shades.
SINGLE_MARKERS = {
    "luna_single_high": ("o", False),
    "luna_single_xhigh_12288": ("D", False),
    "luna_single_high_6144": ("v", False),
    "luna_single_high_24576": ("^", False),
    "luna_single_high_60steps": ("s", True),
    "luna_single_high_120steps": ("p", True),
    "serial_high": ("o", False),
    "sol_single_high60": ("s", True),
    "sol_single_high120": ("p", True),
    "sol_single_xhigh30": ("D", False),
}
SINGLE_SIZE = 32
SELECTION_PURPLE = "#8246E5"
SELECTION_SIZE = 42
FOCUS_SIZE = 49
CACHE_SIZE = 46
CACHE_HATCH = "////////"


def cache_handle(color=PALETTE["BASELINE_LIGHT"]):
    return Circle((0, 0), radius=1, facecolor=color, edgecolor="white",
                  linewidth=0, hatch=CACHE_HATCH)


def legend_circle(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
    return Circle((width / 2 - xdescent, height / 2 - ydescent),
                  radius=min(width, height) * .46)


def right_layout(fig, previous=None, height=2.4451461988304095):
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
            if j == 1 and label not in cached_labels:
                handles[-1].set_markersize(6.0)
                handles[-1].set_markeredgewidth(1.2)
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
             fontsize=7.2, color=PALETTE["MUTED"], va="top")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    footer_box = fig.texts[0].get_window_extent(renderer)
    # Fit the added model row while retaining all original font sizes.
    if footer_box.y0 < 3:
        extra = (3 - footer_box.y0) / fig.dpi / .95 + .02
        assert height + extra < 3.5, "Sidebar unexpectedly exceeds compact figure height"
        return right_layout(fig, previous, height + extra)
    fig.text(.016, (.22 + .95) / 2, "Success (%)", rotation=90,
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
    # Recolor this preview in memory; preserve sibling figures and all data.
    def purple_style(method, style):
        label, marker, color, filled = style
        fade = .18 if "parallel2" in method else .08 if "parallel3" in method or "n3_cached" in method else 0
        rgb = matplotlib.colors.to_rgb(SELECTION_PURPLE)
        color = matplotlib.colors.to_hex([v * (1 - fade) + fade for v in rgb])
        return label, marker, color, filled
    def gray_style(method, style):
        label, marker, color, filled = style
        if method == ORIGINAL.BASE.SOL60_METHOD:
            marker, filled = "s", True
        else:
            marker, filled = SINGLE_MARKERS[method]
        color = SOL_GREEN if method == ORIGINAL.BASE.SOL60_METHOD or method in ORIGINAL.SOL or method == "serial_high" else SINGLE_GRAY
        return label, marker, color, filled
    ORIGINAL.SOL = {method: gray_style(method, style) for method, style in ORIGINAL.SOL.items()}
    ORIGINAL.BASE.SOL60_STYLE = gray_style(ORIGINAL.BASE.SOL60_METHOD, ORIGINAL.BASE.SOL60_STYLE)
    for styles in (ORIGINAL.BASE.SCALING.STYLES, ORIGINAL.BASE.AVERAGE.SCALING.STYLES):
        for method in list(styles):
            if method.startswith("parallel"):
                styles[method] = purple_style(method, styles[method])
            elif method not in ORIGINAL.CACHED:
                styles[method] = gray_style(method, styles[method])
    ORIGINAL.CACHED = {method: purple_style(method, style)
                       for method, style in ORIGINAL.CACHED.items()}
    fig, frontier, rows = ORIGINAL.build()
    styles = {**ORIGINAL.BASE.SCALING.STYLES, **ORIGINAL.SOL,
              ORIGINAL.BASE.SOL60_METHOD: ORIGINAL.BASE.SOL60_STYLE}
    singles = {row["method"]: styles[row["method"]] for row in rows
               if not row["method"].startswith("parallel") and row["method"] not in ORIGINAL.CACHED}
    assert len(singles) == 10
    assert len({style[1:] for style in singles.values()}) == 10, "Single-call marker/color/fill must be unique"
    # The inherited paper formatter normalizes markers; reapply the explicit
    # single-call mapping to both points and handles after that formatter runs.
    labels_to_styles = {label: (marker, color, filled) for label, marker, color, filled in singles.values()}
    labels_to_styles.update({label: singles[method][1:] for method, label in ORIGINAL.BASE.SINGLE_LEGEND if method in singles})
    labels_to_styles[ORIGINAL.BASE.SOL60_STYLE[0]] = ORIGINAL.BASE.SOL60_STYLE[1:]
    for legend in fig.legends:
        for line, text in zip(legend.get_lines(), legend.texts):
            if text.get_text() in labels_to_styles:
                marker, color, filled = labels_to_styles[text.get_text()]
                line.set_marker(marker)
                line.set_markeredgecolor(color)
                line.set_markerfacecolor(color if filled else "white")
                line.set_markeredgewidth(1.0)
                line.set_markersize(5.5)
    matplotlib.rcParams["hatch.linewidth"] = .5
    assert len(rows) == 16
    circle = MarkerStyle("o")
    circle_path = circle.get_path().transformed(circle.get_transform())
    for ax in fig.axes:
        for row, point in zip(rows, ax.collections[:len(rows)]):
            if row["method"] in singles:
                _, marker, color, filled = singles[row["method"]]
                shape = MarkerStyle(marker)
                point.set_paths([shape.get_path().transformed(shape.get_transform())])
                point.set_facecolors(color if filled else "white")
                point.set_edgecolors(color)
                point.set_sizes([SINGLE_SIZE])
                point.set_linewidths([1.0])
            if row["method"].startswith("parallel") or row["method"] in ORIGINAL.CACHED:
                point.set_sizes([FOCUS_SIZE if row["method"] == ORIGINAL.FOCUS else SELECTION_SIZE])
                point.set_linewidths([1.2])
                point.set_zorder(5)
            if row["method"] in ORIGINAL.CACHED:
                point.set_paths([circle_path])
                point.set_sizes([CACHE_SIZE])
                point.set_edgecolors("white")
                point.set_linewidths([0])
                point.set_hatch(CACHE_HATCH)
        # Keep the observed-frontier rings visible around the enlarged points.
        emphasized = [point.get_offsets()[0] for row, point in zip(rows, ax.collections[:len(rows)])
                      if row["method"].startswith("parallel") or row["method"] in ORIGINAL.CACHED]
        rings = ax.collections[len(rows)]
        rings.set_sizes([90 if any(all(abs(float(a) - float(b)) < 1e-10 for a, b in zip(xy, p))
                                   for p in emphasized) else 57 for xy in rings.get_offsets()])
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
                              selection="Saturated purple action decision scaling; gray Luna controls and green Sol controls; slightly lighter purple for smaller candidate sets",
                              single_call_encoding="Baseline: circle; xhigh: diamond; 0.5x/2x tokens: down/up triangle; 60/120 steps: filled square/pentagon. Luna gray, Sol green.",
                              single_call_markers={key: dict(marker=style[0], filled=style[1]) for key, style in SINGLE_MARKERS.items()},
                              sol_color=SOL_GREEN, single_call_size_pt2=SINGLE_SIZE,
                              single_call_color=SINGLE_GRAY, selection_color=SELECTION_PURPLE, selection_size_pt2=SELECTION_SIZE, focus_size_pt2=FOCUS_SIZE,
                              cached="Saturated purple circles with white diagonal shading and no outline; matching method symbols and a shaded gray circle key",
                              cached_marker=dict(marker="o", size_pt2=CACHE_SIZE, hatch=CACHE_HATCH, hatch_color="white", hatch_linewidth=.5, linewidth=0),
                              frontier="Same observed frontier; gray rings enlarged around emphasized points, dashed links unchanged",
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
