"""Compact training and valid-only evaluation curves from published aggregates.

Training is a trailing-five mean of valid-only collected binary success rates.
Validation reproduces rl_eval_learning_curves.json, including its pooled trailing
three-evaluation smoother and raw dots. Both panels use the shared 2:1 layout
with a vertical legend on the right. Matched GPT-4.1 bars are omitted pending a complete shared-Terra comparison.
Use --overall to include invalid attempts in both denominators.
No training, browser or API calls.
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
DOCS = next(p for p in HERE.parents if (p / "paper_plot_style.py").is_file())
sys.path.insert(0, str(DOCS))
import paper_plot_style  # noqa: E402,F401
from paper_plot_palette import COLORS as C  # noqa: E402
from rl_plot_layout import make_axes, style_axis, add_legend, LABEL_SIZE  # noqa: E402

DATA_PATH = HERE / "arm_final_results_data.json"
DATA = json.loads(DATA_PATH.read_text())
VALIDATION_PATH = DOCS / "rl_results/rl_eval_learning_curves.json"
VALIDATION = json.loads(VALIDATION_PATH.read_text())
RUNS = {r["key"]: r for r in DATA["runs"]}
STEM = "rl_learning_compact"
WINDOW = 5
INK, MUTED = C["INK"], C["MUTED"]
METHODS = {
    "outcome48": ("Outcome only · 48 groups", C["BASELINE_LIGHT"], (0, (3, 2))),
    "outcome56": ("Outcome only · 56 groups", C["BASELINE_DARK"], "-"),
    "gate_b": ("Outcome + ADM (ours)", C["ADM_DEFAULT"], "-"),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def series(key, overall=False):
    run = RUNS[key]
    field = "training_success" if overall else "training_success_valid"
    xs = sorted(int(i) for i in run[field])
    raw = [100 * run[field][str(i)] for i in xs]
    ys = [sum(raw[max(0, i - WINDOW + 1):i + 1]) / min(i + 1, WINDOW)
          for i in range(len(raw))]
    source_key = "gate-b" if key == "gate_b" else key
    method = next(m for m in VALIDATION["methods"] if m["id"] == source_key)
    displayed = [p for p in method["points"] if 20 <= p["iteration"] <= 90]
    assert [p["iteration"] for p in displayed] == list(range(20, 91, 10))
    smoothed = method["smoothed"]
    for i, point in enumerate(smoothed):
        window = displayed[max(0, i - 2):i + 1]
        assert abs(point["percent"] - 100 * sum(p["successes"] for p in window)
                   / sum(p["valid"] for p in window)) < 1e-10
    if overall:
        displayed = [dict(p, percent=100 * p["successes"] / p["planned"]) for p in displayed]
        smoothed = []
        for i, point in enumerate(displayed):
            window = displayed[max(0, i - 2):i + 1]
            successes = sum(p["successes"] for p in window)
            planned = sum(p["planned"] for p in window)
            smoothed.append(dict(iteration=point["iteration"], percent=100 * successes / planned,
                                 successes=successes, planned=planned,
                                 window_iterations=[p["iteration"] for p in window]))
    return dict(training=dict(iteration=xs, raw_percent=raw, smoothed_percent=ys),
                validation=dict(iteration=[p["iteration"] for p in displayed],
                                raw_percent=[p["percent"] for p in displayed],
                                smoothed_percent=[p["percent"] for p in smoothed],
                                counts=displayed, windows=smoothed))



def build(overall=False):
    points = {key: series(key, overall=overall) for key in METHODS}
    fig, axes = make_axes()
    for ax, title in zip(axes, ("a) Training", "b) OM2W")):
        style_axis(ax, title)
    # Draw ours last without shifting measured checkpoint positions.
    order = [key for key in METHODS if key != "gate_b"] + ["gate_b"]
    for key in order:
        label, color, ls = METHODS[key]
        p = points[key]
        tr, va = p["training"], p["validation"]
        dots = [i for i, x in enumerate(tr["iteration"]) if x % 10 == 0]
        width = 1.8 if key == "gate_b" else 1.2
        axes[0].plot(tr["iteration"], tr["smoothed_percent"], color=color,
                     linestyle=ls, linewidth=width, marker="o", markevery=dots,
                     markersize=2.4, markeredgewidth=0, zorder=4 if key == "gate_b" else 3)
        axes[1].scatter(va["iteration"], va["raw_percent"], color=color,
                        s=8, alpha=.25, edgecolors="none", zorder=2)
        axes[1].plot(va["iteration"], va["smoothed_percent"], color=color,
                     linestyle=ls, linewidth=width, marker="o", markersize=3,
                     markeredgecolor="white", markeredgewidth=.4,
                     zorder=4 if key == "gate_b" else 3)
    for ax in axes[:2]:
        ax.set_xlim(0, 94)
        ax.set_xticks([0, 30, 60, 90])
        ax.set_xlabel("Iteration", fontsize=LABEL_SIZE)
    axes[0].set_ylim(15 if overall else 39, 61)
    axes[0].set_yticks([20, 40, 60] if overall else [40, 50, 60])
    axes[0].set_ylabel("Success %", fontsize=LABEL_SIZE)
    axes[1].set_xlim(17, 93)
    axes[1].set_xticks([20, 40, 60, 80, 90])
    axes[1].set_ylim(25, 45.5 if overall else 60)
    axes[1].set_yticks([25, 35, 45] if overall else [30, 40, 50, 60])
    axes[1].set_ylabel("Success %", fontsize=LABEL_SIZE)

    handles = [Line2D([], [], color=color, linestyle=ls,
                      linewidth=1.2, marker="o", markersize=3.5,
                      label=label) for key, (label, color, ls) in METHODS.items()]
    legend = add_legend(fig, handles)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    assert not any(legend.get_window_extent(renderer).overlaps(ax.get_tightbbox(renderer))
                   for ax in axes)
    texts = list(legend.texts)
    for ax in axes:
        texts.extend([ax.title, ax.xaxis.label, ax.yaxis.label, *ax.texts,
                      *ax.get_xticklabels(), *ax.get_yticklabels()])
    boxes = [(t.get_text(), t.get_window_extent(renderer)) for t in texts
             if t.get_visible() and t.get_text()]
    for label, bounds in boxes:
        assert fig.bbox.contains(*bounds.p0) and fig.bbox.contains(*bounds.p1), label
    assert not [(a, b) for i, (a, x) in enumerate(boxes)
                for b, y in boxes[i + 1:] if x.overlaps(y)]
    return fig, points


def main(overall=False):
    fig, points = build(overall=overall)
    stem = STEM + ("_overall" if overall else "")
    for ext in ("png", "pdf", "svg"):
        metadata = ({"Date": None} if ext == "svg" else
                    {"CreationDate": None, "ModDate": None} if ext == "pdf" else {})
        path = HERE / f"{stem}.{ext}"
        fig.savefig(path, dpi=300, facecolor="white", metadata=metadata)
        if ext == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    report = dict(schema="rl-learning-compact-v7",
                  population="All completed attempts; invalid attempts count as non-success" if overall else "Valid attempts only; invalid attempts excluded", source_sha256=sha(DATA_PATH),
                  validation_source=str(VALIDATION_PATH.relative_to(DOCS)),
                  validation_source_sha256=sha(VALIDATION_PATH),
                  palette_sha256=sha(DOCS / "paper_plot_palette.py"),
                  palette_colors_sha256=hashlib.sha256(json.dumps(C, sort_keys=True).encode()).hexdigest(),
                  renderer_sha256=sha(__file__), layout_source_sha256=sha(DOCS / "rl_plot_layout.py"),
                  layout="4.9 x 2.45 inches; 2:1 aspect; equal-width panels; vertical legend on the right", methods=METHODS,
                  training=dict(protocol=DATA["training_protocol"],
                                field="training_success" if overall else "training_success_valid",
                                metric="100 * successes / all completed collected trajectories" if overall else "100 * successes / valid collected trajectories",
                                includes_adm_bonus=False,
                                smoothing="Trailing 5 recorded iterations, arithmetic mean of per-iteration " + ("overall" if overall else "valid-only") + " success rates; not pooled counts.",
                                raw_overlay=False),
                  validation=dict(protocol=VALIDATION["protocol"],
                                  metric="100 * successes / planned episodes; invalid episodes count as non-success" if overall else VALIDATION["metric"],
                                  display=VALIDATION["display"],
                                  smoothing=dict(VALIDATION["smoothing"], method="Trailing ratio of pooled success and planned counts") if overall else VALIDATION["smoothing"],
                                  raw_overlay="Faint raw checkpoint dots; connected circles show the trailing three-evaluation pooled ratio."),
                  omitted_comparison="Iteration-90 48/56-group/ADM bars: the matched repeats are GPT-4.1 judged, and no complete shared-Terra counterpart is available. No new rejudging was launched.",
                  caveats=["Historical validation curves use mixed dates. Overall rates use all 300 tasks per checkpoint; valid-only rates use method-specific valid subsets.",
                           "Training trajectories are collected by each policy; valid-only collection success is not fixed-task held-out accuracy.",
                           "Failures in main batch and 40M+16F are excluded from the compact curves and retained in the results table and all-run archive."],
                  points=points,
                  output_sha256={ext: sha(HERE / f"{stem}.{ext}") for ext in ("png", "pdf", "svg")})
    (HERE / f"{stem}.json").write_text(json.dumps(report, indent=2) + "\n")
    assert b"/FontFile2" in (HERE / f"{stem}.pdf").read_bytes()
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overall", action="store_true", help="Plot success over all attempts, including invalid attempts in the denominator.")
    main(overall=parser.parse_args().overall)
