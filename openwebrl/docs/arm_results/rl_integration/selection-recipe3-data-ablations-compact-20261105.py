"""Recipe 3 GPT-5.5 K = 3 4B LoRA (ADM-4B) training-data ablations in one compact figure, SFT 4B actor, November 4–6.
Both subset families on one axis of training examples (log): fewer states (whole tasks until 250 / 500 / 1,000 states, all their
panels) and fewer panels per state (1 / 2 / 4 per state, all 2,112 states); both end at the full data (32,974 examples).
y = equal-weight 3-benchmark mean of overall success (successes / tasks), 3 runs pooled per point; the 1-run subsets (500 states,
2 panels per state) are left out. Gray dashed: SFT 4B alone (3 runs, November 6).
Data: selection-recipe3-data-ablations-20261105.json (selection-recipe3-data-ablations-20261105-analysis.py)."""
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
NAME = "selection-recipe3-data-ablations-compact-20261105"
d = json.loads((HERE / "selection-recipe3-data-ablations-20261105.json").read_text())
runs = {k: len(v) for k, v in d["sources"].items()}
FULL = 32974
SERIES = [("Fewer states", "#6250BE", "o", [("s250", 3858, "252 states"), ("s1000", 15746, "1,001 states"), ("full", FULL, "")], 1),
          ("Fewer panels per state", "#E07A1F", "D", [("pps1", 3910, "1 panel/state"), ("pps4", 13532, "4 panels/state"), ("full", FULL, "")], -1)]
METRICS = [("mean_overall", "Mean overall success (%)")]

fig, ax0 = plt.subplots(figsize=(3.6, 2.9))
axes = [ax0]
for ax, (metric, ylabel) in zip(axes, METRICS):
    y = {k: v[metric] for k, v in d["rates"].items()}
    vals = [y[k] for _, _, _, pts, _ in SERIES for k, _, _ in pts] + [y["sft"]]
    for i, (label, color, marker, pts, side) in enumerate(SERIES):
        xs = [x for _, x, _ in pts]; ys = [y[k] for k, _, _ in pts]
        ax.plot(xs, ys, color=color, linewidth=1.4, zorder=2)
        for (k, x, name), v in zip(pts, ys):
            if k == "full":
                continue
            ax.scatter(x, v, s=30, marker=marker, zorder=3, facecolor="white" if runs[k] == 1 else color, edgecolor=color, linewidth=1.1)
            up = v >= y[[p[0] for p in SERIES[1 - i][3]][[p[0] for p in pts].index(k)]]
            ax.annotate(f"{name}\n{v:.1f}" if up else f"{v:.1f}\n{name}", (x, v), textcoords="offset points", xytext=(0, 7 if up else -7),
                        ha="center", va="bottom" if up else "top", fontsize=7.5, color=color, linespacing=1.0)
    ax.scatter(FULL, y["full"], s=42, marker="s", zorder=4, facecolor="#2b2b2b", edgecolor="#2b2b2b")
    ax.annotate(f"{y['full']:.1f}", (FULL, y["full"]), textcoords="offset points", xytext=(7, -3), ha="left", fontsize=8, color="#2b2b2b")
    ax.axhline(y["sft"], color="#898781", linestyle="--", linewidth=0.9, zorder=1)
    ax.text(2600, y["sft"] + 0.25, f"SFT 4B alone {y['sft']:.1f}", fontsize=7.5, color="#6b6963", va="bottom", ha="left")
    ax.set_xscale("log"); ax.set_xticks([4000, 8000, 16000, FULL]); ax.set_xticklabels(["4K", "8K", "16K", "33K\n(all)"]); ax.minorticks_off()
    ax.set_xlim(2500, FULL * 1.6); ax.set_ylim(min(vals) - 1.0, max(vals) + 2.6)
    ax.set_xlabel("Training examples"); ax.set_ylabel(ylabel)
    ax.yaxis.set_major_locator(MultipleLocator(2))
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.6, zorder=0)
    ax.tick_params(colors="#444444", length=3, width=0.6, pad=2)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
handles = [Line2D([], [], color=c, marker=m, markersize=5, linewidth=1.4, label=l) for l, c, m, _, _ in SERIES]
handles += [Line2D([], [], color="#2b2b2b", marker="s", linestyle="none", markersize=5, label="Full data"),
]
fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.02), handletextpad=0.4, columnspacing=1.2)
fig.tight_layout(rect=(0, 0.07, 1, 1))
for ext in ("png", "pdf", "svg"):
    fig.savefig(HERE / f"{NAME}.{ext}", dpi=220 if ext == "png" else None, bbox_inches="tight")
plt.close(fig)
