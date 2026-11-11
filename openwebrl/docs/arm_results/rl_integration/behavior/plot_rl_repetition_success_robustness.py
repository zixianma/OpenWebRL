#!/usr/bin/env python3
"""Combine success versus repetition and threshold robustness from public data."""
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# Shared paper typography, including bundled Albert Sans fonts.
import sys as _paper_sys
from pathlib import Path as _PaperPath
_paper_sys.path.insert(0, str(next(p for p in _PaperPath(__file__).resolve().parents
                                if (p / "paper_plot_style.py").is_file())))
import paper_plot_style as _paper_style
from paper_plot_palette import COLORS as C

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
    plt.rcParams.update({"text.color": C["INK"], "axes.labelcolor": C["INK"],
                         "xtick.color": C["MUTED"], "ytick.color": C["MUTED"], "font.family": "Albert Sans", "font.size": 7,
                         "axes.titlesize": 7, "axes.labelsize": 7,
                         "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
                         "svg.hashsalt": "rl-repetition-success-robustness"})
    fig, (left, right) = plt.subplots(1, 2, figsize=(3.35, 2.25),
                                      gridspec_kw={"width_ratios": [0.78, 1]})
    for key, _, _, color, _ in RUNS:
        row = success["methods"][key]
        x, y = row["repetition_incidence"], row["valid_success"]
        assert x["percent"] == repetition["methods"][key]["thresholds"]["3"]["percent"]
        left.errorbar(x["percent"], y["percent"],
                      yerr=[[y["percent"] - y["ci95_percent"][0]],
                            [y["ci95_percent"][1] - y["percent"]]],
                      fmt="o", color=color, markersize=4, capsize=2, elinewidth=1,
                      zorder=4 if key == "gate-b" else 3)
        rows = [repetition["methods"][key]["thresholds"][str(k)] for k in THRESHOLDS]
        values = np.asarray([r["percent"] for r in rows])
        ci = np.asarray([r["ci95_percent"] for r in rows])
        assert np.all(np.diff(values) <= 0)
        right.errorbar(THRESHOLDS, values,
                       yerr=np.vstack((values - ci[:, 0], ci[:, 1] - values)),
                       color=color, marker="o", markersize=4, linewidth=1.2,
                       capsize=2, elinewidth=0.9, zorder=4 if key == "gate-b" else 3)
    left_title = left.set_title("(a) Iter 90 on OM2W", loc="left", pad=7)
    left.set_xlabel("Repetition % (K=3)", labelpad=3)
    left.set_ylabel("Success %", labelpad=3)
    left.set_xlim(13, 47)
    left.set_xticks([20, 30, 40])
    left.set_ylim(35, 60)
    left.set_yticks([40, 45, 50, 55])
    right_title = right.set_title("(b) Robustness", loc="left", pad=7)
    right.set_xlabel("Repetition threshold, K", labelpad=3)
    right.set_ylabel("Repetition (%)", labelpad=3)
    right.set_xlim(1.82, 5.18)
    right.set_xticks(THRESHOLDS)
    right.set_ylim(0, 70)
    right.set_yticks([0, 20, 40, 60])
    for ax in (left, right):
        ax.grid(False)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(C["BASELINE_DARK"])
            ax.spines[side].set_linewidth(0.7)
        ax.tick_params(length=2.5, pad=2, color=C["BASELINE_DARK"])
    handles = [Line2D([], [], color=color, marker="o", linewidth=1.2, markersize=4,
                      label=label) for _, label, _, color, _ in RUNS]
    legend = fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.54, 0.005),
                        frameon=False, ncol=1, fontsize=7, handlelength=1.4,
                        labelspacing=0.35, handletextpad=0.5)
    fig.subplots_adjust(left=0.15, right=0.98, top=0.86, bottom=0.43, wspace=0.65)
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
