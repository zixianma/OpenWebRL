"""Compact training and valid-only evaluation curves from published aggregates.

Training is a trailing-five mean of valid-only collected binary success rates.
Validation reproduces rl_eval_learning_curves.json, including its pooled trailing
three-evaluation smoother and raw dots. Both axes retain their previous physical
size. Matched GPT-4.1 bars are omitted pending a complete shared-Terra comparison.
No training, browser or API calls.
"""
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

DATA_PATH = HERE / "arm_final_results_data.json"
DATA = json.loads(DATA_PATH.read_text())
VALIDATION_PATH = DOCS / "rl_results/rl_eval_learning_curves.json"
VALIDATION = json.loads(VALIDATION_PATH.read_text())
RUNS = {r["key"]: r for r in DATA["runs"]}
STEM = "rl_learning_compact"
WINDOW = 5
INK, MUTED = "#30343B", "#626972"
METHODS = {
    "outcome48": ("Outcome only · 48 groups", "#9AA0A6", (0, (3, 2))),
    "outcome56": ("Outcome only · 56 groups", "#555B63", "-"),
    "gate_b": ("Outcome + ADM (ours)", "#6250BE", "-"),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def series(key):
    run = RUNS[key]
    xs = sorted(int(i) for i in run["training_success_valid"])
    raw = [100 * run["training_success_valid"][str(i)] for i in xs]
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
    return dict(training=dict(iteration=xs, raw_percent=raw, smoothed_percent=ys),
                validation=dict(iteration=[p["iteration"] for p in displayed],
                                raw_percent=[p["percent"] for p in displayed],
                                smoothed_percent=[p["percent"] for p in smoothed],
                                counts=displayed, windows=smoothed))


def style(ax, title):
    ax.set_title(title, fontsize=9, loc="left", color=INK, pad=5)
    ax.grid(False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#A5ABB2")
        ax.spines[side].set_linewidth(.65)
    ax.tick_params(labelsize=8, colors=MUTED, width=.65, length=3)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)


def build():
    points = {key: series(key) for key in METHODS}
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.4),
                             gridspec_kw=dict(width_ratios=[1.45, 1.45, 1.05]))
    fig.subplots_adjust(left=.075, right=.985, bottom=.30, top=.88, wspace=.53)
    for ax, title in zip(axes, ("a) Training", "b) Validation")):
        style(ax, title)
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
        ax.set_xlabel("Iteration", fontsize=8.5)
    axes[0].set_ylim(39, 61)
    axes[0].set_yticks([40, 50, 60])
    axes[0].set_ylabel("Valid-only success (%)", fontsize=8.5)
    axes[1].set_xlim(17, 93)
    axes[1].set_xticks([20, 40, 60, 80, 90])
    axes[1].set_ylim(25, 60)
    axes[1].set_yticks([30, 40, 50, 60])
    axes[1].set_ylabel("Valid-only success (%)", fontsize=8.5)

    # Remove the GPT-4.1 judge comparison while preserving each curve axis in inches.
    positions = [ax.get_position().bounds for ax in axes[:2]]
    axes[2].remove()
    axes = axes[:2]
    old_width = fig.get_figwidth()
    fig.set_size_inches(4.9, 2.4, forward=True)
    for ax, (left, bottom, width, height) in zip(axes, positions):
        ax.set_position([left * old_width / 4.9, bottom,
                         width * old_width / 4.9, height])
    handles = [Line2D([], [], color=color, linestyle=ls,
                      linewidth=1.8 if key == "gate_b" else 1.2,
                      label=label) for key, (label, color, ls) in METHODS.items()]
    legend = fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.53, .005),
                        ncol=2, frameon=False, fontsize=7.5, handlelength=1.6,
                        handletextpad=.5, columnspacing=1.1, labelspacing=.25)
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


def main():
    fig, points = build()
    for ext in ("png", "pdf", "svg"):
        metadata = ({"Date": None} if ext == "svg" else
                    {"CreationDate": None, "ModDate": None} if ext == "pdf" else {})
        path = HERE / f"{STEM}.{ext}"
        fig.savefig(path, dpi=300, facecolor="white", metadata=metadata)
        if ext == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    report = dict(schema="rl-learning-compact-v6", source_sha256=sha(DATA_PATH),
                  validation_source=str(VALIDATION_PATH.relative_to(DOCS)),
                  validation_source_sha256=sha(VALIDATION_PATH),
                  renderer_sha256=sha(__file__), methods=METHODS,
                  training=dict(protocol=DATA["training_protocol"],
                                smoothing="Trailing 5 recorded iterations, arithmetic mean of per-iteration valid-only rates; not pooled counts.",
                                raw_overlay=False),
                  validation=dict(protocol=VALIDATION["protocol"],
                                  metric=VALIDATION["metric"],
                                  display=VALIDATION["display"],
                                  smoothing=VALIDATION["smoothing"],
                                  raw_overlay="Faint raw valid-only checkpoint dots; connected circles show the trailing three-evaluation pooled ratio."),
                  omitted_comparison="Iteration-90 48/56-group/ADM bars: the matched repeats are GPT-4.1 judged, and no complete shared-Terra counterpart is available. No new rejudging was launched.",
                  caveats=["Historical validation curves use mixed dates and method-specific valid subsets; invalid episodes are excluded from the valid-only rates.",
                           "Training trajectories are collected by each policy; valid-only collection success is not fixed-task held-out accuracy.",
                           "Failures in main batch and 40M+16F are excluded from the compact curves and retained in the results table and all-run archive."],
                  points=points,
                  output_sha256={ext: sha(HERE / f"{STEM}.{ext}") for ext in ("png", "pdf", "svg")})
    (HERE / f"{STEM}.json").write_text(json.dumps(report, indent=2) + "\n")
    assert b"/FontFile2" in (HERE / f"{STEM}.pdf").read_bytes()
    plt.close(fig)


if __name__ == "__main__":
    main()
