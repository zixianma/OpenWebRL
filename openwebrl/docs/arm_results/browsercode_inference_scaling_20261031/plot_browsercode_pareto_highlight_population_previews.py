#!/usr/bin/env python3
"""Alternative success denominators for the frozen sixteen-setting comparison.

Reuses published counts, coordinates and paper styling. No experiment calls.
"""
import argparse
import copy
import hashlib
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


def common_points(source_rows):
    data = json.loads((HERE / "common16_aggregate.json").read_text())
    assert data["schema"] == "browsercode-sixteen-method-common-valid-aggregate-v1"
    assert data["methods"] == [r["method"] for r in source_rows]
    report_sha = BASE.sha(HERE / (RIGHT.STEM + ".json"))
    # The aggregate pins the pre-palette report. Permit only its identical numeric
    # points when style metadata/output hashes have changed; retain the original
    # source aggregate and its exact historical binding without rewriting it.
    if data["source_sha256"]["own_valid_report"] != report_sha:
        assert data["source_sha256"]["own_valid_report"] == "5e6ee9945e1d69f917d4997169af3a3da9ce80d9259924619b405e12fb518535"
        scientific_sha = hashlib.sha256(json.dumps(source_rows, sort_keys=True,
            separators=(",", ":")).encode()).hexdigest()
        assert scientific_sha == "8ecc393a3b4d5a31b0d377cc069dfe30e9dbef75d20c680202cbab4f5e56eff5"
    weights = {"online_mind2web": 300, "webvoyager": 595, "deepshop": 150}
    assert data["benchmark_weights"] == weights
    lookup = {(r["benchmark"], r["method"]): r for r in data["points"]}
    assert len(lookup) == len(data["points"]) == 48
    rows = copy.deepcopy(source_rows)
    for row in rows:
        row["source_valid_only_success_rate"] = row["success_rate"]
        row["source_benchmark_counts"] = row["benchmark_counts"]
        cells = {b: lookup[b, row["method"]] for b in weights}
        counts = {}
        for b, planned in weights.items():
            c, coverage = cells[b], data["benchmark_counts"][b]
            n = coverage["common_valid"]
            assert coverage["planned"] == planned and 0 < n <= planned
            assert coverage["excluded"] == planned - n
            assert c["common_valid"] == n and 0 <= c["successes"] <= n
            own = row["source_benchmark_counts"][b]
            assert n <= own["valid"] and c["successes"] <= own["successes"]
            assert n - c["successes"] <= own["valid"] - own["successes"]
            assert math.isclose(c["success_rate"], c["successes"] / n, abs_tol=1e-12)
            assert set(c["metrics"]) == set(BASE.METRICS)
            for metric in c["metrics"].values():
                assert metric["observed_tasks"] == n and metric["missing_valid_tasks"] == 0
                assert math.isfinite(metric["mean"]) and metric["mean"] >= 0
            counts[b] = dict(planned=planned, common_valid=n, excluded=planned - n,
                             successes=c["successes"])
        row["benchmark_counts"] = counts
        row["success_rate"] = math.fsum(weights[b] * cells[b]["success_rate"] for b in weights) / 1045
        row["metrics"] = {metric: math.fsum(weights[b] * cells[b]["metrics"][metric]["mean"]
                                           for b in weights) / 1045 for metric in BASE.METRICS}
    return rows, data


def build(population="overall"):
    fig, _, source_rows = ORIGINAL.build()
    source_report = json.loads((HERE / (RIGHT.STEM + ".json")).read_text())
    assert source_rows == source_report["points"]
    common = None
    if population == "overall":
        rows = overall_points(source_rows)
    else:
        rows, common = common_points(source_rows)
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
        if population != "overall":
            xs = [row["metrics"][metric] for row in rows]
            if metric == "actor_usd":
                assert min(xs) > 0
                ax.set_xlim(min(xs) / 1.15, max(xs) * 1.15)
            else:
                padding = max(.5, (max(xs) - min(xs)) * .08)
                ax.set_xlim(max(0, min(xs) - padding), max(xs) + padding)
    fig.legends[-1].remove()
    frontier = BASE.overlay(fig, [(population + "_success", rows, metric)
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
    height = 2.838128654970761
    for _ in range(4):
        RIGHT.right_layout(fig, previous=previous, height=height)
        for text in fig.texts:
            if text.get_text() in ("Valid-only success (%)", "Success (%)"):
                text.set_text("Overall success (%)" if population == "overall" else "Shared-task success (%)")
        footer = fig.texts[0]
        if population == "overall":
            note = "Overall = successes / 1,045.\nInvalid/missing count as nonsuccess.\nX axes: valid-task actor means.\nGPT-6 high unless xhigh. N=candidates."
        else:
            n = {b: c["common_valid"] for b, c in common["benchmark_counts"].items()}
            note = (f"Shared valid: OM2W {n['online_mind2web']}, WV {n['webvoyager']}, DS {n['deepshop']}.\n"
                    "Same tasks for success and all x axes.\nWeights: 300/595/150 planned tasks.\n"
                    "GPT-6 high unless xhigh. N=candidates.")
        footer.set_text(note)
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
        if population == "overall":
            assert row["metrics"] == original["metrics"]
            assert row["benchmark_counts"] == original["benchmark_counts"]
    return fig, frontier, rows, common


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", choices=("overall", "common-valid"), default="overall")
    args = parser.parse_args()
    fig, frontier, rows, common = build(args.population)
    suffix = "overall_success" if args.population == "overall" else "common_valid"
    stem = "browsercode_scaling_weighted_pareto_highlight_right_legend_" + suffix + "_preview"
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
    if common is not None:
        report.update(
            schema="browsercode-sixteen-setting-common-valid-pareto-v1",
            population="Same tasks valid across all sixteen methods, separately intersected within each benchmark.",
            success_definition="Successes / shared-valid count within each benchmark, then fixed planned-count weighting300/595/150; not pooled common-valid success.",
            x_definition="All three actor means are recomputed on exactly the same shared-valid tasks used for success; fixed planned benchmark weights300/595/150.",
            benchmark_counts=common["benchmark_counts"],
            comparison_caveats=common["comparison_caveats"])
        report["source_sha256"]["common16_aggregate.json"] = BASE.sha(HERE / "common16_aggregate.json")
    (HERE / (stem + ".json")).write_text(json.dumps(report, indent=2) + "\n")
    assert b"/FontFile2" in (HERE / (stem + ".pdf")).read_bytes()
    plt.close(fig)


if __name__ == "__main__":
    main()
