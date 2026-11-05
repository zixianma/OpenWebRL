#!/usr/bin/env python3
"""Render smoothed RL checkpoint success from public, aggregate-only counts.

Default: render the frozen adjacent-public JSON, with no network or experiment
calls. --refresh-aggregate rebuilds those counts from the existing audited
full-300 comparison builder. No trajectories or task identities are inputs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
DOCS = HERE.parents[2]
sys.path.insert(0, str(DOCS))
import paper_plot_style
from plot_rl_repetition_incidence import RUNS

OUTPUT = DOCS / "rl_results/rl_eval_learning_curves"
AGGREGATE = OUTPUT.with_suffix(".json")
WINDOW = 3
IDENTITIES = {"outcome48": ("outcome_only", "qcq7i4ug"),
              "outcome56": ("outcome56", "outcome-only-56groups-20261006"),
              "gate-b": ("gate_b", "arm-gate-b-309053")}


def smooth(points, window=WINDOW):
    """Pool counts over the latest at most three observed evaluations."""
    result = []
    for i, point in enumerate(points):
        selected = points[max(0, i - window + 1):i + 1]
        numerator = sum(p["successes"] for p in selected)
        denominator = sum(p["valid"] for p in selected)
        result.append(dict(iteration=point["iteration"],
                           percent=100 * numerator / denominator,
                           successes=numerator, valid=denominator,
                           window_iterations=[p["iteration"] for p in selected]))
    return result


def refresh():
    sys.path.insert(0, str(DOCS.parents[1] / "scripts"))
    from plot_arm_interactive import build_data
    published = build_data()
    sources = {"../../scripts/plot_arm_interactive.py":
               hashlib.sha256((DOCS.parents[1] / "scripts/plot_arm_interactive.py").read_bytes()).hexdigest()}
    methods = []
    for key, label, _, color, _ in RUNS:
        identity, lineage = IDENTITIES[key]
        source = next(s for s in published["series"] if s["id"] == identity)
        points = []
        for row in source["points"]:
            name = row["source"].split("/openwebrl/docs/", 1)[1]
            assert (DOCS / name).is_file()
            sources[name] = hashlib.sha256((DOCS / name).read_bytes()).hexdigest()
            assert row["tasks"] == 300 and 0 <= row["successes"] <= row["valid"] <= 300
            points.append(dict(iteration=row["iteration"], successes=row["successes"],
                               valid=row["valid"], planned=300,
                               percent=100 * row["successes"] / row["valid"], source=name))
        assert [p["iteration"] for p in points] == sorted({p["iteration"] for p in points})
        methods.append(dict(id=key, label=label, color=color, training_lineage=lineage,
                            points=points, smoothed=smooth(points)))
    data = dict(schema="rl-eval-learning-curves-v1", benchmark="Online-Mind2Web, full 300 tasks",
                protocol=published["protocol"],
                metric="100 * successes / valid episodes at each evaluated checkpoint; invalid episodes excluded",
                smoothing=dict(method="Trailing ratio of pooled success and valid counts", evaluations=WINDOW,
                               minimum_evaluations=1, location="Latest checkpoint in each window",
                               missing="Never imputed; no extrapolation before first or after last recorded evaluation",
                               interpretation="A descriptive trend, not another evaluation or a confidence interval"),
                comparability="One training lineage per method. Historical single evaluations at different collection dates and method-specific valid populations. Training iterations do not match Adam updates, rollout work or compute. Separate from the October 25 iteration-90 three-repeat cohorts.",
                methods=methods, source_sha256=sources)
    AGGREGATE.write_text(json.dumps(data, indent=2) + "\n")
    return data


def render(data):
    assert data["schema"] == "rl-eval-learning-curves-v1"
    assert [m["id"] for m in data["methods"]] == [r[0] for r in RUNS]
    plt.rcParams.update({"font.size": 7, "axes.labelsize": 7.5,
                         "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.hashsalt": "rl-eval-learning-curves"})
    fig, ax = plt.subplots(figsize=(3.35, 2.55))
    handles = []
    for method, reference in zip(data["methods"], RUNS):
        assert method["color"] == reference[3]
        points = method["points"]
        for p in points:
            assert p["planned"] == 300 and 0 <= p["successes"] <= p["valid"] <= 300
            assert abs(p["percent"] - 100 * p["successes"] / p["valid"]) < 1e-10
        assert method["smoothed"] == smooth(points)
        color = method["color"]
        ax.scatter([p["iteration"] for p in points], [p["percent"] for p in points],
                   color=color, s=12, alpha=0.35, edgecolors="none", zorder=2)
        ax.plot([p["iteration"] for p in method["smoothed"]],
                [p["percent"] for p in method["smoothed"]],
                color=color, linewidth=1.7, zorder=4 if method["id"] == "gate-b" else 3)
        handles.append(Line2D([], [], color=color, linewidth=1.7, label=method["label"]))
    ax.set_xlabel("Training iteration", labelpad=3)
    ax.set_ylabel("Valid-only success (%)", labelpad=3)
    ax.set_xlim(5, 105)
    ax.set_xticks([20, 40, 60, 80, 100])
    ax.set_ylim(25, 60)
    ax.set_yticks([30, 40, 50, 60])
    ax.grid(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#666666")
        ax.spines[side].set_linewidth(0.7)
    ax.tick_params(length=2.5, width=0.7, pad=2, color="#666666")
    legend = fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.55, 0.05),
                        frameon=False, ncol=1, fontsize=7, handlelength=1.7,
                        labelspacing=0.35, handletextpad=0.5)
    footer = fig.text(0.55, 0.015, "Trailing 3 evaluations · faint points: raw", ha="center",
                      va="bottom", fontsize=6.4, color="#626972")
    fig.subplots_adjust(left=0.16, right=0.98, top=0.96, bottom=0.37)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_box = legend.get_window_extent(renderer)
    assert not legend_box.overlaps(ax.get_tightbbox(renderer))
    assert not legend_box.overlaps(footer.get_window_extent(renderer))
    for text in fig.findobj(match=matplotlib.text.Text):
        if text.get_visible() and text.get_text():
            box = text.get_window_extent(renderer)
            assert fig.bbox.contains(*box.p0) and fig.bbox.contains(*box.p1), text.get_text()
    for ext in ("png", "pdf", "svg"):
        metadata = {"Date": None} if ext == "svg" else {"CreationDate": None, "ModDate": None} if ext == "pdf" else {}
        fig.savefig(OUTPUT.with_suffix("." + ext), dpi=300, metadata=metadata)
    svg = OUTPUT.with_suffix(".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    assert b"/FontFile2" in OUTPUT.with_suffix(".pdf").read_bytes()
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-aggregate", action="store_true")
    args = parser.parse_args()
    data = refresh() if args.refresh_aggregate else json.loads(AGGREGATE.read_text())
    render(data)
