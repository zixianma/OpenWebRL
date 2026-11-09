"""Compact local RL curves and absolute stealth-browser success bars.

Reads only the published aggregate. Original all-run figures stay unchanged.
Training is a trailing-five mean of valid-only collected binary success rates;
validation shows raw overall success on 300 tasks. Stealth bars use
published pointwise task-cluster BCa intervals from three September evaluation repeats,
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
ABSOLUTE_CI_PATH = HERE.parent / "rl_integration/stealth-iteration90-five-method-overall-ci.json"
ABSOLUTE_CI = json.loads(ABSOLUTE_CI_PATH.read_text())
STEALTH_PROTOCOL_PATH = HERE.parent / "rl_integration/stealth-o4-t06-iteration90-three-repeat-summary.json"
STEALTH_PROTOCOL = json.loads(STEALTH_PROTOCOL_PATH.read_text())
RUNS = {r["key"]: r for r in DATA["runs"]}
STEM = "rl_learning_compact"
WINDOW = 5
INK, MUTED = "#30343B", "#626972"
METHODS = {
    "outcome48": ("Outcome only · 48 groups", "#9AA0A6", (0, (3, 2))),
    "outcome56": ("Outcome only · 56 groups", "#555B63", "-"),
    "gate_b": ("Outcome + ADM (ours)", "#6250BE", "-"),
}
BAR_METHODS = {
    "baseline": ("outcome48", "48\ngroups", "#9AA0A6"),
    "additive": ("additive", "Strict\ngate", "#A99BE3"),
    "gate-b": ("gate_b", "ADM", "#6250BE"),
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
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.4),
                             gridspec_kw=dict(width_ratios=[1.45, 1.45, 1.05]))
    fig.subplots_adjust(left=.075, right=.985, bottom=.30, top=.88, wspace=.53)
    for ax, title in zip(axes, ("a) Training", "b) Validation", "c) Stealth iter. 90")):
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

    absolute = []
    for x, (source_key, (key, label, color)) in enumerate(BAR_METHODS.items()):
        row = ABSOLUTE_CI["methods"][source_key]
        counts = row["per_repeat_successes"]
        assert row["repeats"] == 3 and row["tasks_per_repeat"] == 300
        mean = 100 * row["overall_mean"]
        lo, hi = [100 * v for v in row["overall_ci95"]]
        assert abs(mean - 100 * sum(counts) / 900) < 1e-10
        assert abs(mean - DATA["stealth_iter90"]["overall_mean_pct"][key]) < .005
        axes[2].bar(x, mean, bottom=0, width=.62, color=color,
                    alpha=.85, edgecolor="none", zorder=2)
        axes[2].errorbar(x, mean, yerr=[[mean - lo], [hi - mean]],
                         fmt="none", ecolor=MUTED, elinewidth=1,
                         capsize=2.2, capthick=.8, zorder=3)
        axes[2].text(x, hi + .8, f"{mean:.1f}", ha="center", va="bottom",
                     fontsize=7.5, color=color)
        absolute.append(dict(method=key, overall_percent=mean, ci95_percent=[lo, hi],
                             repeat_successes=counts, planned_episodes=900,
                             collection_dates_pacific=row["collection_dates_pacific"]))
    assert len({row["collection_dates_pacific"] for row in absolute}) == 1
    axes[2].set_xlim(-.6, 2.6)
    axes[2].set_xticks([0, 1, 2], [value[1] for value in BAR_METHODS.values()], fontsize=7.5)
    axes[2].set_ylim(0, 70)
    axes[2].set_yticks([0, 30, 60])
    axes[2].set_ylabel("Overall success (%)", fontsize=8.5)
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
    return fig, points, absolute


def main():
    fig, points, absolute = build()
    for ext in ("png", "pdf", "svg"):
        metadata = ({"Date": None} if ext == "svg" else
                    {"CreationDate": None, "ModDate": None} if ext == "pdf" else {})
        path = HERE / f"{STEM}.{ext}"
        fig.savefig(path, dpi=300, facecolor="white", metadata=metadata)
        if ext == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    report = dict(schema="rl-learning-compact-v4", source_sha256=sha(DATA_PATH),
                  absolute_interval_source=str(ABSOLUTE_CI_PATH.relative_to(DOCS)),
                  absolute_interval_source_sha256=sha(ABSOLUTE_CI_PATH),
                  stealth_protocol_source_sha256=sha(STEALTH_PROTOCOL_PATH),
                  renderer_sha256=sha(__file__), methods=METHODS, bar_methods=BAR_METHODS,
                  training=dict(protocol=DATA["training_protocol"],
                                smoothing="Trailing 5 recorded iterations, arithmetic mean of per-iteration valid-only rates; not pooled counts.",
                                raw_overlay=False),
                  validation=dict(protocol=DATA["validation_protocol"],
                                  smoothing="None; connected circles are raw measured checkpoint success rates."),
                  stealth=dict(collection_dates_pacific=absolute[0]["collection_dates_pacific"],
                               protocol=STEALTH_PROTOCOL["protocol"],
                               training_iteration=90, absolute_success=absolute,
                               display="Three vertical absolute overall-success bars from zero, for outcome48, strict-gate ADM, and relaxed-gate ADM.",
                               uncertainty=dict(method=ABSOLUTE_CI["interval_method"],
                                                tasks=300, repeats=3,
                                                draws=ABSOLUTE_CI["bootstrap_resamples"],
                                                seed=ABSOLUTE_CI["bootstrap_seed"],
                                                numerical_convention=ABSOLUTE_CI["numerical_convention"],
                                                scope="Conditional on these fixed checkpoints and evaluation windows; excludes training-seed and evaluation-window uncertainty.")),
                  caveats=["Historical validation curves use mixed dates and method-specific valid subsets; overall rates use all 300 tasks.",
                           "Training trajectories are collected by each policy; valid-only collection success is not fixed-task held-out accuracy.",
                           "Stealth bars use a different browser/judge/actor-sampling protocol from the local GPT-4.1/T0 validation curves.",
                           "No matching outcome56 stealth repeat cohort is available; the middle bar is the strict-gate additive ADM run.",
                           "The stealth ADM-versus-baseline gain is not statistically established (paired CI crosses zero; Holm p about .22).",
                           "Failures in main batch and 40M+16F are excluded from the compact curves and retained in the results table and all-run archive."],
                  points=points,
                  output_sha256={ext: sha(HERE / f"{STEM}.{ext}") for ext in ("png", "pdf", "svg")})
    (HERE / f"{STEM}.json").write_text(json.dumps(report, indent=2) + "\n")
    assert b"/FontFile2" in (HERE / f"{STEM}.pdf").read_bytes()
    plt.close(fig)


if __name__ == "__main__":
    main()
