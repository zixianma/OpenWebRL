#!/usr/bin/env python3
"""Vertical-bar preview of iteration-90 loop failures as task percentages.

Read the existing public aggregate and task-bootstrap confidence intervals.
The denominator includes all 300 planned OM2W tasks in each of three repeats;
intervals condition on these checkpoints and evaluation windows. No task records
are loaded and no uncertainty is recomputed.
"""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DOCS = HERE.parents[2]
SOURCE = HERE / "rl-action-loops-20261103.json"
INTERVALS = HERE / "rl-action-loops-iter90-ci-20261104.json"
RUNS = (("outcome48", "Outcome-only\n48 groups", "#9A9A9A"),
        ("outcome56", "Outcome-only\n56 groups", "#666666"),
        ("gate-b", "Outcome + ADM\nrewards (ours)", "#6250BE"))


def render():
    intervals = json.loads(INTERVALS.read_text())
    source = json.loads(SOURCE.read_text())
    assert intervals["aggregate_sha256"] == hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    denominator = intervals["uncertainty"]["tasks"]
    assert denominator == 300
    scale = 100 / denominator
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "svg.fonttype": "none",
                         "svg.hashsalt": "rl-action-loops-iter90-bars-preview"})
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    means, limits = [], []
    for key, _, _ in RUNS:
        row = intervals["methods"][key]
        counts = [r["loop_failures"] for r in source["iteration90"][key]]
        assert row["repeat_counts"] == counts
        assert np.isclose(row["mean_per300"], np.mean(counts))
        means.append(row["mean_per300"] * scale)
        limits.append(np.asarray(row["ci95_per300"]) * scale)
    means, limits = np.asarray(means), np.asarray(limits)
    errors = np.vstack((means - limits[:, 0], limits[:, 1] - means))
    bars = ax.bar(np.arange(3), means, width=0.56,
                  color=[color for _, _, color in RUNS],
                  yerr=errors, capsize=3,
                  error_kw={"ecolor": "#353535", "elinewidth": 1, "capthick": 1},
                  zorder=3)
    for j, (_, _, color) in enumerate(RUNS):
        ax.text(j, limits[j, 1] + 0.9, f"{means[j]:.1f}%", ha="center",
                va="bottom", fontsize=9, color=color, fontweight="medium")
    ax.set_xticks(np.arange(3), [label for _, label, _ in RUNS], fontsize=8)
    ax.get_xticklabels()[-1].set_color(RUNS[-1][2])
    ax.get_xticklabels()[-1].set_fontweight("medium")
    ax.set_ylabel("Repeated-action failures (%)", fontsize=9.5, labelpad=5)
    ax.yaxis.set_label_coords(-0.12, 0.42)
    ax.set_ylim(0, 27)
    ax.set_yticks([0, 5, 10, 15, 20, 25])
    ax.set_xlim(-0.48, 2.48)
    ax.tick_params(axis="x", length=0, pad=6)
    ax.tick_params(axis="y", labelsize=8.5, length=3, color="#666666")
    ax.grid(False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#666666")
        ax.spines[side].set_linewidth(0.7)
    fig.subplots_adjust(left=0.17, right=0.985, top=0.98, bottom=0.25)
    output = DOCS / "rl_results/rl_action_loops_iter90_bars_preview"
    fig.savefig(output.with_suffix(".png"), dpi=300)
    fig.savefig(output.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(output.with_suffix(".svg"), metadata={"Date": None})
    svg = output.with_suffix(".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    assert np.array_equal(np.asarray([bar.get_height() for bar in bars]), means)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    labels = [t.get_window_extent(renderer) for t in ax.get_xticklabels()]
    assert all(not a.overlaps(b) for a, b in zip(labels, labels[1:]))
    assert all(fig.bbox.contains(*box.p0) and fig.bbox.contains(*box.p1) for box in labels)
    assert b"/FontFile2" in output.with_suffix(".pdf").read_bytes()
    print(json.dumps({"denominator": denominator,
                      "percentages": {key: {"mean": float(means[j]), "ci95": limits[j].tolist()}
                                      for j, (key, _, _) in enumerate(RUNS)}}, indent=2))
    plt.close(fig)


if __name__ == "__main__":
    render()
