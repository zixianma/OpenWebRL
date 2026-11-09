"""Compact RL curves and the independently matched checkpoint comparison.

Reads only the published aggregate. Original all-run figures stay unchanged.
Training is a trailing-five mean of valid-only collected binary success rates;
validation is unsmoothed overall success on 300 tasks. Matched intervals are
the published paired task intervals from repeated evaluations of one checkpoint,
not uncertainty across training seeds. No training, browser or API calls.
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
RUNS = {r["key"]: r for r in DATA["runs"]}
STEM = "rl_learning_compact"
WINDOW = 5
INK, MUTED = "#30343B", "#626972"
METHODS = {
    "outcome48": ("Outcome only · 48 groups", "#9AA0A6", (0, (3, 2))),
    "outcome56": ("Outcome only · 56 groups", "#555B63", "-"),
    "gate_b": ("Outcome + ADM (ours)", "#6250BE", "-"),
    "g40m16f": ("40M + 16F", "#A99BE3", "-"),
    "all_failure": ("Failures in main batch", "#D8877E", "-"),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def series(key):
    run = RUNS[key]
    xs = sorted(int(i) for i in run["training_success_valid"])
    raw = [100 * run["training_success_valid"][str(i)] for i in xs]
    ys = [sum(raw[max(0, i - WINDOW + 1):i + 1]) / min(i + 1, WINDOW)
          for i in range(len(raw))]
    vx = sorted(int(i) for i in run["validation"])
    vy = [100 * run["validation"][str(i)]["successes"] / 300 for i in vx]
    return dict(training=dict(iteration=xs, raw_percent=raw, smoothed_percent=ys),
                validation=dict(iteration=vx, overall_percent=vy,
                                counts={str(i): run["validation"][str(i)] for i in vx}))


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
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.6),
                             gridspec_kw=dict(width_ratios=[1.45, 1.45, 1.05]))
    fig.subplots_adjust(left=.075, right=.985, bottom=.38, top=.88, wspace=.53)
    for ax, title in zip(axes, ("a) Training", "b) Validation", "c) Matched iter. 90")):
        style(ax, title)
    # Draw ours last, retaining the same actual data coordinates as the controls.
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
        axes[1].plot(va["iteration"], va["overall_percent"], color=color,
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
    axes[1].set_ylim(22, 45.5)
    axes[1].set_yticks([25, 35, 45])
    axes[1].set_ylabel("Overall success (%)", fontsize=8.5)

    matched = DATA["matched_iter90_local"]
    paired = {row["comparison"]: row for row in matched["paired"]}
    gains = []
    for y, key in ((1, "outcome48"), (0, "outcome56")):
        row = paired["Gate B − " + key]
        gain, (lo, hi) = row["diff_pp"], row["ci"]
        axes[2].errorbar(gain, y, xerr=[[gain - lo], [hi - gain]],
                         fmt="o", color=METHODS["gate_b"][1], markersize=4,
                         elinewidth=1.1, capsize=2.4, markeredgewidth=.7)
        axes[2].text(gain, y + .18, f"+{gain:.2f}", ha="center", va="bottom",
                     fontsize=8, color=METHODS["gate_b"][1])
        gains.append(dict(reference=key, difference_pp=gain, ci95_pp=[lo, hi],
                          holm_p=row["holm_p"]))
    axes[2].set_xlim(0, 11)
    axes[2].set_xticks([0, 5, 10])
    axes[2].set_ylim(-.5, 1.65)
    axes[2].set_yticks([1, 0], ["vs. 48 groups", "vs. 56 groups"], fontsize=7.5)
    axes[2].tick_params(axis="y", length=0, pad=4)
    axes[2].set_xlabel("Gain (pp)", fontsize=8.5)
    handles = [Line2D([], [], color=color, linestyle=ls,
                      linewidth=1.8 if key == "gate_b" else 1.2,
                      label=label) for key, (label, color, ls) in METHODS.items()]
    legend = fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.52, .015),
                        ncol=3, frameon=False, fontsize=8, handlelength=1.6,
                        handletextpad=.5, columnspacing=1.1, labelspacing=.55)
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
    return fig, points, gains


def main():
    fig, points, gains = build()
    for ext in ("png", "pdf", "svg"):
        metadata = ({"Date": None} if ext == "svg" else
                    {"CreationDate": None, "ModDate": None} if ext == "pdf" else {})
        path = HERE / f"{STEM}.{ext}"
        fig.savefig(path, dpi=300, facecolor="white", metadata=metadata)
        if ext == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    report = dict(schema="rl-learning-compact-v1", source_sha256=sha(DATA_PATH),
                  renderer_sha256=sha(__file__), methods=METHODS,
                  training=dict(protocol=DATA["training_protocol"],
                                smoothing="Trailing 5 recorded iterations, arithmetic mean of per-iteration valid-only rates; not pooled counts.",
                                raw_overlay=False),
                  validation=dict(protocol=DATA["validation_protocol"], smoothing="None"),
                  matched=dict(date=DATA["matched_iter90_local"]["date"],
                               protocol=DATA["matched_iter90_local"]["protocol"],
                               training_iteration=90, differences=gains,
                               uncertainty="Paired task intervals from evaluation repeats of one trained checkpoint, not training seeds."),
                  caveats=["Historical validation curves use mixed dates and method-specific valid subsets; overall rates use all 300 tasks.",
                           "Training trajectories are collected by each policy; valid-only collection success is not fixed-task held-out accuracy.",
                           "Failures in main batch replaces mixed groups within the 48-group batch and also uses D=5; ours uses D>=2 and separate failure loss.",
                           "Unshown ablations remain in the all-run figures; incomplete 48M+16F is not extrapolated."],
                  points=points,
                  output_sha256={ext: sha(HERE / f"{STEM}.{ext}") for ext in ("png", "pdf", "svg")})
    (HERE / f"{STEM}.json").write_text(json.dumps(report, indent=2) + "\n")
    assert b"/FontFile2" in (HERE / f"{STEM}.pdf").read_bytes()
    plt.close(fig)


if __name__ == "__main__":
    main()
