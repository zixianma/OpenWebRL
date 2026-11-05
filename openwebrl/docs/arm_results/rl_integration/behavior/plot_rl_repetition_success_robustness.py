#!/usr/bin/env python3
"""Combine success versus repetition and threshold robustness from public data."""
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from plot_rl_repetition_incidence import AGGREGATE as REPETITION, DOCS, RUNS, THRESHOLDS
from plot_rl_repetition_success import AGGREGATE as SUCCESS

OUTPUT = DOCS / "rl_results/rl_repetition_success_robustness"


def render():
    repetition = json.loads(REPETITION.read_text())
    success = json.loads(SUCCESS.read_text())
    assert success["repetition_aggregate_sha256"] == hashlib.sha256(REPETITION.read_bytes()).hexdigest()
    assert repetition["thresholds"] == list(THRESHOLDS) == [2, 3, 4, 5]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5,
                         "axes.titlesize": 10, "axes.labelsize": 10,
                         "xtick.labelsize": 9, "ytick.labelsize": 9,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
                         "svg.hashsalt": "rl-repetition-success-robustness"})
    fig, (left, right) = plt.subplots(1, 2, figsize=(6.8, 2.6))
    for key, _, _, color, _ in RUNS:
        row = success["methods"][key]
        x, y = row["repetition_incidence"], row["overall_success"]
        assert x["percent"] == repetition["methods"][key]["thresholds"]["3"]["percent"]
        left.errorbar(x["percent"], y["percent"],
                      yerr=[[y["percent"] - y["ci95_percent"][0]],
                            [y["ci95_percent"][1] - y["percent"]]],
                      fmt="o", color=color, markersize=5, capsize=2.5, elinewidth=1,
                      zorder=4 if key == "gate-b" else 3)
        rows = [repetition["methods"][key]["thresholds"][str(k)] for k in THRESHOLDS]
        values = np.asarray([r["percent"] for r in rows])
        ci = np.asarray([r["ci95_percent"] for r in rows])
        assert np.all(np.diff(values) <= 0)
        right.errorbar(THRESHOLDS, values,
                       yerr=np.vstack((values - ci[:, 0], ci[:, 1] - values)),
                       color=color, marker="o", markersize=5, linewidth=1.5,
                       capsize=2.5, elinewidth=0.9, zorder=4 if key == "gate-b" else 3)
    left_title = left.set_title("(a) Success vs. repetition (k=3)", loc="left", pad=9)
    left.set_xlabel("Repetition incidence (%)", labelpad=5)
    left.set_ylabel("Overall success (%)", labelpad=5)
    left.set_xlim(13, 47)
    left.set_xticks([15, 25, 35, 45])
    left.set_ylim(26, 49)
    left.set_yticks([30, 35, 40, 45])
    right_title = right.set_title("(b) Threshold robustness", loc="left", pad=9)
    right.set_xlabel("Minimum identical-action count, k", labelpad=5)
    right.set_ylabel("Repetition incidence (%)", labelpad=5)
    right.set_xlim(1.82, 5.18)
    right.set_xticks(THRESHOLDS)
    right.set_ylim(0, 70)
    right.set_yticks([0, 20, 40, 60])
    for ax in (left, right):
        ax.grid(False)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#666666")
            ax.spines[side].set_linewidth(0.7)
        ax.tick_params(length=3, color="#666666")
    handles = [Line2D([], [], color=color, marker="o", linewidth=1.5, markersize=5,
                      label=label) for _, label, _, color, _ in RUNS]
    legend = fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.53, 0.005),
                        frameon=False, ncol=3, fontsize=8.5, handlelength=1.4,
                        columnspacing=1.3, handletextpad=0.5)
    fig.subplots_adjust(left=0.085, right=0.99, top=0.85, bottom=0.29, wspace=0.32)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for text in fig.findobj(match=matplotlib.text.Text):
        if text.get_visible() and text.get_text():
            box = text.get_window_extent(renderer)
            assert fig.bbox.contains(*box.p0) and fig.bbox.contains(*box.p1), text.get_text()
    legend_box = legend.get_window_extent(renderer)
    assert all(not legend_box.overlaps(ax.xaxis.label.get_window_extent(renderer)) for ax in (left, right))
    assert not left_title.get_window_extent(renderer).overlaps(right_title.get_window_extent(renderer))
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=300)
    fig.savefig(OUTPUT.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(OUTPUT.with_suffix(".svg"), metadata={"Date": None})
    svg = OUTPUT.with_suffix(".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    assert b"/FontFile2" in OUTPUT.with_suffix(".pdf").read_bytes()
    plt.close(fig)


if __name__ == "__main__":
    render()
