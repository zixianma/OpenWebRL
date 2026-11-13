"""Merge published RL learning and repetition plots without changing their data.

CPU only. Reuses the learning series builder and audited repetition aggregates.
Writes rl_learning_compact_with_repetition.{png,pdf,svg,json}; preserves originals.
"""
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/openwebrl-rl-merged-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

HERE = Path(__file__).resolve().parent
DOCS = next(p for p in HERE.parents if (p / "paper_plot_style.py").is_file())
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(DOCS))
sys.path.insert(0, str(DOCS / "arm_results/rl_integration/behavior"))
import paper_plot_style  # noqa: E402,F401
from paper_plot_palette import COLORS as C  # noqa: E402
from rl_plot_layout import style_axis, LEGEND_LABELS, LEGEND_SIZE  # noqa: E402
import plot_rl_learning_compact as learning  # noqa: E402
import plot_rl_repetition_success_robustness as behavior  # noqa: E402

OUTPUT = HERE / "rl_learning_compact_with_repetition"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    plt.rcParams.update({"font.family": "Albert Sans", "font.size": 7,
                         "text.color": C["INK"], "pdf.fonttype": 42,
                         "svg.fonttype": "none", "svg.hashsalt": "rl-learning-repetition-merged"})
    points = {key: learning.series(key) for key in learning.METHODS}
    published = json.loads((HERE / "rl_learning_compact.json").read_text())
    assert points == published["points"], "Merged curves differ from published curves"
    success = json.loads(behavior.SUCCESS.read_text())
    repetition = json.loads(behavior.REPETITION.read_text())
    assert success["repetition_aggregate_sha256"] == sha(behavior.REPETITION)
    assert repetition["thresholds"] == list(behavior.THRESHOLDS) == [2, 3, 4, 5]
    fig, axes = plt.subplots(1, 4, figsize=(8.8, 2.10))
    fig.subplots_adjust(left=.05, right=.80, top=.84, bottom=.22, wspace=.35)
    for ax in axes:
        ax.set_box_aspect(1)
    for key, (_, color, ls) in learning.METHODS.items():
        tr, va = points[key]["training"], points[key]["validation"]
        width, z = (1.8, 4) if key == "gate_b" else (1.2, 3)
        dots = [i for i, iteration in enumerate(tr["iteration"]) if iteration % 10 == 0]
        axes[0].plot(tr["iteration"], tr["smoothed_percent"], color=color, linestyle=ls,
                     linewidth=width, marker="o", markevery=dots, markersize=2.4,
                     markeredgewidth=0, zorder=z)
        axes[1].scatter(va["iteration"], va["raw_percent"], color=color,
                        s=8, alpha=.25, edgecolors="none", zorder=2)
        axes[1].plot(va["iteration"], va["smoothed_percent"], color=color, linestyle=ls,
                     linewidth=width, marker="o", markersize=3, markeredgecolor="white",
                     markeredgewidth=.4, zorder=z)
        source_key = "gate-b" if key == "gate_b" else key
        x, y = (success["methods"][source_key][field] for field in ("repetition_incidence", "valid_success"))
        assert x["percent"] == repetition["methods"][source_key]["thresholds"]["3"]["percent"]
        axes[2].errorbar(x["percent"], y["percent"],
                         yerr=[[y["percent"] - y["ci95_percent"][0]], [y["ci95_percent"][1] - y["percent"]]],
                         fmt="o", color=color, markersize=4, capsize=2, elinewidth=1, zorder=z)
        rows = [repetition["methods"][source_key]["thresholds"][str(k)] for k in behavior.THRESHOLDS]
        values, ci = np.array([r["percent"] for r in rows]), np.array([r["ci95_percent"] for r in rows])
        assert np.all(np.diff(values) <= 0)
        axes[3].errorbar(behavior.THRESHOLDS, values,
                         yerr=np.vstack((values - ci[:, 0], ci[:, 1] - values)),
                         color=color, marker="o", markersize=4, linewidth=1.2,
                         linestyle=ls, capsize=2, elinewidth=.9, zorder=z)
    axes[0].set(xlim=(0, 94), xticks=[0, 30, 60, 90], ylim=(39, 61), yticks=[40, 50, 60],
                xlabel="Iteration", ylabel="Reward")
    axes[1].set(xlim=(17, 93), xticks=[20, 40, 60, 80, 90], ylim=(25, 60), yticks=[30, 40, 50, 60],
                xlabel="Iteration", ylabel="Success %")
    axes[2].set(xlim=(13, 47), xticks=[20, 30, 40], ylim=(35, 60), yticks=[40, 45, 50, 55],
                xlabel="Repetition % (K=3)", ylabel="Success %")
    axes[3].set(xlim=(1.82, 5.18), xticks=behavior.THRESHOLDS, ylim=(0, 70), yticks=[0, 20, 40, 60],
                xlabel="Repetition threshold, K", ylabel="Repetition (%)")
    for ax, title in zip(axes, ("a) Training Rewards", "b) OM2W across Iterations", "c) Iter 90 on OM2W", "d) Iter 90’s Repetition")):
        style_axis(ax, title)
    handles = [Line2D([], [], color=color, linestyle=ls, marker="o", linewidth=1.2, markersize=3.5)
               for _, color, ls in learning.METHODS.values()]
    legend = fig.legend(handles, LEGEND_LABELS, loc="center left", bbox_to_anchor=(.83, .53),
                        ncol=1, frameon=False, fontsize=LEGEND_SIZE, labelcolor=C["INK"],
                        handlelength=1.4, handletextpad=.5, labelspacing=1.0, borderaxespad=0)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_box = legend.get_window_extent(renderer)
    assert all(not legend_box.overlaps(ax.get_tightbbox(renderer)) for ax in axes)
    for ax in axes:
        bounds = ax.get_window_extent(renderer)
        assert abs(bounds.width - bounds.height) < .01
    for text in fig.findobj(match=matplotlib.text.Text):
        if text.get_visible() and text.get_text():
            box = text.get_window_extent(renderer)
            assert fig.bbox.contains(*box.p0) and fig.bbox.contains(*box.p1), text.get_text()
    tight = [ax.get_tightbbox(renderer) for ax in axes]
    assert all(not a.overlaps(b) for a, b in zip(tight, tight[1:])), "Adjacent panels overlap"
    for ext in ("png", "pdf", "svg"):
        metadata = {"Date": None} if ext == "svg" else {"CreationDate": None, "ModDate": None} if ext == "pdf" else {}
        path = OUTPUT.with_suffix("." + ext)
        fig.savefig(path, dpi=300, facecolor="white", metadata=metadata)
        if ext == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    report = dict(schema="rl-learning-repetition-merged-v1", layout="8.8 x 2.10 inches; four square panels; one vertical right legend",
                  learning_points=points, repetition_points={k: success["methods"][k] for k in ("outcome48", "outcome56", "gate-b")},
                  thresholds=behavior.THRESHOLDS, threshold_points=repetition["methods"],
                  learning_protocols={k: published[k] for k in ("training", "validation", "caveats")},
                  training_axis_note="Reward is the binary outcome reward displayed as a valid-only percentage; ADM bonus excluded.",
                  repetition_definition=repetition["definition"], repetition_uncertainty=repetition["uncertainty"],
                  success_uncertainty=success["uncertainty"],
                  sources={str(p.relative_to(DOCS)): sha(p) for p in
                           (learning.DATA_PATH, learning.VALIDATION_PATH, behavior.REPETITION, behavior.SUCCESS,
                            Path(learning.__file__), Path(behavior.__file__), DOCS / "rl_plot_layout.py", Path(__file__))},
                  output_sha256={ext: sha(OUTPUT.with_suffix("." + ext)) for ext in ("png", "pdf", "svg")})
    OUTPUT.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    assert b"/FontFile2" in OUTPUT.with_suffix(".pdf").read_bytes()
    plt.close(fig)


if __name__ == "__main__":
    main()
