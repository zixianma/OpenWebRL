"""Recipe 3 GPT-5.5 K = 3 4B LoRA (ADM-4B) training-data ablations, SFT 4B actor, November 4–6. Two figures:
  selection-recipe3-data-ablations-compact-20261105.*         equal-weight 3-benchmark mean of overall success
  selection-recipe3-data-ablations-per-benchmark-20261105.*   overall success on Online-Mind2Web, WebVoyager and DeepShop
Both subset families share one log axis of 5-action sets (panels) in the training data: fewer states (whole tasks until 252 / 1,001
states, all their sets) and fewer sets per state (1 / 4 per state, all 2,112 states); both end at the full data (2,112 states,
18,786 sets). The legend names each family's fixed setting; points are labelled only with the varied one (states, rounded in K, or mean
sets per state: the 4-per-state subset averages 3.5 because some states have fewer than 4 sets). Overall success = successes / tasks, 3 runs pooled per point; the
1-run subsets (500 states, 2 sets per state) are left out. Gray dashed: SFT 4B alone (3 runs, November 6).
Data: selection-recipe3-data-ablations-20261105.json (selection-recipe3-data-ablations-20261105-analysis.py); state and set counts
from the subset builder's summary (recipe3-ablations-20261104/lora-data/summary.json)."""
import json
import os
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", "/tmp/openwebrl-selection-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys as _paper_sys
from pathlib import Path as _PaperPath
_paper_sys.path.insert(0, str(next(p for p in _PaperPath(__file__).resolve().parents
                                if (p / "paper_plot_style.py").is_file())))
import paper_plot_style as _paper_style
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

HERE = Path(__file__).parent
d = json.loads((HERE / "selection-recipe3-data-ablations-20261105.json").read_text())
SETS = {"s250": 2234, "s1000": 8875, "pps1": 2112, "pps4": 7417, "full": 18786}
STATES = {"s250": 252, "s1000": 1001, "pps1": 2112, "pps4": 2112, "full": 2112}
SERIES = [("Fewer states (8.9 sets/state)", "#6250BE", "o", ["s250", "s1000", "full"], "states"),
          ("Fewer sets per state (2.1K states)", "#E07A1F", "D", ["pps1", "pps4", "full"], "sets")]
BENCH = [("online_mind2web", "Online-Mind2Web"), ("webvoyager", "WebVoyager"), ("deepshop", "DeepShop")]


def kfmt(n):
    return f"{n / 1000:.2g}K"


def name(k, var):
    r = SETS[k] / STATES[k]
    return f"{kfmt(STATES[k])} states" if var == "states" else f"{r:.2g} {'set' if r == 1 else 'sets'}/state"


def draw(ax, y, fs):
    vals = [y[k] for k in SETS] + [y["sft"]]
    for i, (_, color, marker, keys, var) in enumerate(SERIES):
        ax.plot([SETS[k] for k in keys], [y[k] for k in keys], color=color, linewidth=1.4, zorder=2)
        other = SERIES[1 - i][3]
        for j, k in enumerate(keys[:-1]):
            v = y[k]; up = v >= y[other[j]]
            ax.scatter(SETS[k], v, s=32, marker=marker, zorder=3, color=color)
            ax.annotate(f"{name(k, var)}\n{v:.1f}" if up else f"{v:.1f}\n{name(k, var)}", (SETS[k], v), textcoords="offset points",
                        xytext=(0, 6 if up else -6), ha="center", va="bottom" if up else "top", fontsize=fs, color=color, linespacing=1.0)
    ax.scatter(SETS["full"], y["full"], s=42, marker="s", zorder=4, color="#2b2b2b")
    ax.annotate(f"{y['full']:.1f}", (SETS["full"], y["full"]), textcoords="offset points", xytext=(0, -7),
                ha="center", va="top", fontsize=fs, color="#2b2b2b", linespacing=1.0)
    ax.axhline(y["sft"], color="#898781", linestyle="--", linewidth=0.9, zorder=1)
    ax.text(29000, y["sft"] - 0.3, f"SFT 4B alone {y['sft']:.1f}", fontsize=fs, color="#6b6963", va="top", ha="right")
    ax.set_xscale("log"); ax.set_xticks([2000, 4000, 8000, 16000]); ax.set_xticklabels(["2K", "4K", "8K", "16K"]); ax.minorticks_off()
    ax.set_xlim(1600, 30000); ax.set_ylim(min(vals) - 4.2, max(vals) + 3.6)
    ax.set_xlabel("5-action sets in training data")
    ax.yaxis.set_major_locator(MultipleLocator(2 if max(vals) - min(vals) < 12 else 4))
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.6, zorder=0)
    ax.tick_params(colors="#444444", length=3, width=0.6, pad=2)
    for s in ("top", "right"): ax.spines[s].set_visible(False)


def legend(fig, ncol=3):
    h = [Line2D([], [], color=c, marker=m, markersize=5, linewidth=1.4, label=l) for l, c, m, _, _ in SERIES]
    h += [Line2D([], [], color="#2b2b2b", marker="s", linestyle="none", markersize=5, label="Full data (2.1K states, 8.9 sets/state)")]
    fig.legend(handles=h, loc="lower center", ncol=ncol, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.02), handletextpad=0.4, columnspacing=1.2)


def save(fig, name):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(HERE / f"{name}.{ext}", dpi=220 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


fig, ax = plt.subplots(figsize=(4.2, 3.3))
draw(ax, {k: v["mean_overall"] for k, v in d["rates"].items()}, 7)
ax.set_ylabel("Mean overall success (%)")
legend(fig, ncol=2); fig.tight_layout(rect=(0, 0.11, 1, 1))
save(fig, "selection-recipe3-data-ablations-compact-20261105")

fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
for ax, (b, title) in zip(axes, BENCH):
    draw(ax, {k: v[b]["overall"] for k, v in d["rates"].items()}, 6.5)
    ax.set_title(title, fontsize=10); ax.set_ylabel("Overall success (%)")
legend(fig); fig.tight_layout(rect=(0, 0.07, 1, 1))
save(fig, "selection-recipe3-data-ablations-per-benchmark-20261105")
