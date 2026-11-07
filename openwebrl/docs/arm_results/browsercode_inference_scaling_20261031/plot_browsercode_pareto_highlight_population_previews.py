#!/usr/bin/env python3
"""Alternative success denominators for the frozen sixteen-setting comparison.

Reuses published counts, coordinates and paper styling. No experiment calls.
"""
import argparse
import copy
import importlib.util
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/openwebrl-population-previews-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/openwebrl-population-previews-cache")

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "plot_browsercode_pareto_highlight_right_legend_preview.py"
spec = importlib.util.spec_from_file_location("right_legend_template", SOURCE)
RIGHT = importlib.util.module_from_spec(spec)
spec.loader.exec_module(RIGHT)
ORIGINAL = RIGHT.ORIGINAL
BASE = ORIGINAL.BASE
plt = RIGHT.plt


def overall_points(source_rows):
    """Count recorded successes over the same 1,045 planned tasks per method."""
    rows = copy.deepcopy(source_rows)
    for row in rows:
        cells = row["benchmark_counts"]
        assert set(cells) == {"online_mind2web", "webvoyager", "deepshop"}
        for benchmark, planned in (("online_mind2web", 300), ("webvoyager", 595), ("deepshop", 150)):
            c = cells[benchmark]
            assert c["requested"] == planned
            assert c["native_saved"] + c["missing_native"] == planned
            assert c["valid"] + c["invalid"] == c["native_saved"]
            assert 0 <= c["successes"] <= c["valid"]
        total = {key: sum(c[key] for c in cells.values()) for key in
                 ("requested", "native_saved", "valid", "invalid", "successes", "missing_native")}
        assert total["requested"] == 1045
        row["source_valid_only_success_rate"] = row["success_rate"]
        row["success_rate"] = total["successes"] / total["requested"]
        row["overall_counts"] = total
    return rows


def build():
    fig, _, source_rows = ORIGINAL.build()
    source_report = json.loads((HERE / (RIGHT.STEM + ".json")).read_text())
    assert source_rows == source_report["points"]
    rows = overall_points(source_rows)
    low = max(0, 5 * math.floor((100 * min(r["success_rate"] for r in rows) - 1) / 5))
    high = min(100, 5 * math.ceil((100 * max(r["success_rate"] for r in rows) + 1) / 5))
    for ax, metric in zip(fig.axes, BASE.METRICS):
        for old_ring in list(ax.collections[len(rows):]):
            old_ring.remove()
        for line in list(ax.lines):
            if line.get_label() == "_observed_pareto_frontier":
                line.remove()
        for row, point in zip(rows, ax.collections):
            point.set_offsets([(row["metrics"][metric], 100 * row["success_rate"])])
        focus = next(r for r in rows if r["method"] == ORIGINAL.FOCUS)
        for text in ax.texts:
            if text.get_text() == "Luna ADM":
                text.xy = (focus["metrics"][metric], 100 * focus["success_rate"])
        ax.set_ylim(low, high)
    fig.legends[-1].remove()
    frontier = BASE.overlay(fig, [("overall_success", rows, metric)
                                  for metric in BASE.METRICS], method_count=16)
    RIGHT.matplotlib.rcParams["hatch.linewidth"] = .5
    circle = RIGHT.MarkerStyle("o")
    circle_path = circle.get_path().transformed(circle.get_transform())
    for ax in fig.axes:
        for row, point in zip(rows, ax.collections[:len(rows)]):
            if row["method"] in ORIGINAL.CACHED:
                point.set_paths([circle_path])
                point.set_sizes([RIGHT.CACHE_SIZE])
                point.set_edgecolors("white")
                point.set_linewidths([0])
                point.set_hatch(RIGHT.CACHE_HATCH)
    previous = list(fig.legends)
    height = 2.75
    for _ in range(4):
        RIGHT.right_layout(fig, previous=previous, height=height)
        for text in fig.texts:
            if text.get_text() == "Valid-only success (%)":
                text.set_text("Overall success (%)")
        footer = fig.texts[0]
        footer.set_text("Overall = successes / 1,045.\nInvalid/missing count as nonsuccess.\nX axes: valid-task actor means.\nGPT-6 high unless xhigh. N=candidates.")
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        box = footer.get_window_extent(renderer)
        if box.y0 >= 3:
            break
        height = fig.get_figheight() + (3 - box.y0) / fig.dpi / .95 + .02
    else:
        raise AssertionError("Population footer does not fit")
    for text in fig.texts:
        box = text.get_window_extent(renderer)
        assert fig.bbox.contains(*box.p0) and fig.bbox.contains(*box.p1), text.get_text()
    for legend in fig.legends:
        assert not legend.get_window_extent(renderer).overlaps(footer.get_window_extent(renderer))
    for original, row in zip(source_rows, rows):
        assert row["metrics"] == original["metrics"]
        assert row["benchmark_counts"] == original["benchmark_counts"]
    return fig, frontier, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", choices=("overall",), default="overall")
    parser.parse_args()
    fig, frontier, rows = build()
    stem = "browsercode_scaling_weighted_pareto_highlight_right_legend_overall_success_preview"
    BASE.save(fig, stem)
    report = dict(
        schema="browsercode-sixteen-setting-overall-success-pareto-v1",
        population="All 1,045 planned tasks per method: OM2W 300, WebVoyager 595, DeepShop 150.",
        success_definition="Sum of benchmark successes / 1,045; invalid and missing outcomes are nonsuccesses. Equivalent to planned-count weighting of benchmark overall success rates.",
        x_definition="Unchanged fixed planned-count-weighted means over each method's own valid tasks; these are not all-attempt cost or latency means.",
        comparison_caveats=[
            "Retains all original sixteen cohorts, including historical missing tasks; repaired baseline cohorts are not substituted.",
            "Task validity and collection windows differ; invalid outcomes include environment or infrastructure failures, not just actor failures.",
            "Descriptive observed frontiers, without a paired, causal or significance claim.",
            "Actor cost excludes Terra, browser hosting and invalid-outcome spending. Latency is per-task actor-loop elapsed time.",
            "Browser steps count selected nonempty BrowserCode batches, potentially containing multiple browser operations."],
        points=rows, frontier=frontier,
        dimensions_inches=list(fig.get_size_inches()),
        source_sha256={p.name: BASE.sha(p) for p in
                       (Path(__file__), SOURCE, HERE / (RIGHT.STEM + ".json"), HERE / "plot_browsercode_pareto_highlight_preview.py")},
        output_sha256={ext: BASE.sha(HERE / (stem + "." + ext)) for ext in ("png", "pdf", "svg")})
    (HERE / (stem + ".json")).write_text(json.dumps(report, indent=2) + "\n")
    assert b"/FontFile2" in (HERE / (stem + ".pdf")).read_bytes()
    plt.close(fig)


if __name__ == "__main__":
    main()
