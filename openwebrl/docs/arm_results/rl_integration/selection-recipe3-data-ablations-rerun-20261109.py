"""Recipe 3 GPT-5.5 H = 4 4B LoRA data ablations: November 8-9 same-window rerun over the November 4-6 batch. Two figures:
  selection-recipe3-data-ablations-rerun-20261109.*                equal-weight 3-benchmark mean of overall success
  selection-recipe3-data-ablations-rerun-per-benchmark-20261109.*  overall success on Online-Mind2Web, WebVoyager and DeepShop
Same axes and encoding as selection-recipe3-data-ablations-compact-20261105.py (log axis of 5-action sets in the training data;
circles = fewer states, diamonds = fewer sets per state, square = full data). Solid: November 8-9 rerun, 3 runs per point, all
conditions in one window. Faded: the November 4-6 batch (subsets November 4-5, full data November 6), joined to the rerun of the
same LoRA by a thin gray line; only rerun points carry names (faded: circles at 2.2K / 8.9K sets = 252 / 1,001 states, diamonds at
2.1K / 7.4K = 1 / 3.5 sets per state). The 252-state subset was not rerun and appears faded only. Dashed: SFT 4B alone (3 runs, November 6; not
rerun). Data: selection-recipe3-data-ablations-rerun-20261109-analysis.json and selection-recipe3-data-ablations-20261105.json."""
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
import paper_plot_style as _paper_style  # noqa: F401
from paper_plot_palette import COLORS as C, tint
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

HERE = Path(__file__).parent
old = json.loads((HERE / "selection-recipe3-data-ablations-20261105.json").read_text())["rates"]
new = json.loads((HERE / "selection-recipe3-data-ablations-rerun-20261109-analysis.json").read_text())["rates"]
SETS = {"s250": 2234, "s1000": 8875, "pps1": 2112, "pps4": 7417, "full": 18786}
STATES = {"s250": 252, "s1000": 1001, "pps1": 2112, "pps4": 2112, "full": 2112}
MARK = {"s250": "o", "s1000": "o", "pps1": "D", "pps4": "D", "full": "s"}
SERIES = [("o", ["s250", "s1000", "full"], "states"), ("D", ["pps1", "pps4", "full"], "sets")]
BENCH = [("online_mind2web", "Online-Mind2Web"), ("webvoyager", "WebVoyager"), ("deepshop", "DeepShop")]
COL, FADE = C["ADM_4B"], tint("ADM_4B", 0.68)


def name(k):
    r = SETS[k] / STATES[k]
    return f"{STATES[k] / 1000:.2g}K states" if MARK[k] == "o" else f"{r:.2g} {'set' if r == 1 else 'sets'}/state"


def draw(ax, yo, yn, sft, fs):
    vals = list(yo.values()) + list(yn.values()) + [sft]
    for marker, keys, _ in SERIES:                                   # faded November 4-6 families
        ax.plot([SETS[k] for k in keys], [yo[k] for k in keys], color=FADE, linewidth=1.0, zorder=1)
        ks = [k for k in keys if k in yn]                            # solid rerun families
        if len(ks) > 1:
            ax.plot([SETS[k] for k in ks], [yn[k] for k in ks], color=COL, linewidth=1.4, zorder=2)
    for k in SETS:
        ax.scatter(SETS[k], yo[k], s=30, marker=MARK[k], color="white", edgecolors=FADE, linewidths=1.0, zorder=2)
        ax.annotate(f"{yo[k]:.1f}", (SETS[k], yo[k]), textcoords="offset points", xytext=(5, 0), ha="left", va="center",
                    fontsize=fs - 1, color=tint("MUTED", 0.3))
        if k in yn:
            ax.plot([SETS[k]] * 2, [yo[k], yn[k]], color=C["BORDER"], linewidth=0.8, zorder=1)
            ax.scatter(SETS[k], yn[k], s=40 if k == "full" else 32, marker=MARK[k], color=COL, zorder=4)
            up = yn[k] >= yo[k]
            lab = f"{yn[k]:.1f}" if k == "full" else f"{name(k)}\n{yn[k]:.1f}" if up else f"{yn[k]:.1f}\n{name(k)}"
            if MARK[k] == "o":                                       # right of the point: clear of the diamond at 7.4K sets
                ax.annotate(f"{name(k)}\n{yn[k]:.1f}" if up else f"{yn[k]:.1f}\n{name(k)}", (SETS[k], yn[k]), textcoords="offset points",
                            xytext=(7, 3 if up else -10), ha="left", va="bottom" if up else "top", fontsize=fs, color=COL, linespacing=1.0)
            else:
                ax.annotate(lab, (SETS[k], yn[k]), textcoords="offset points", xytext=(0, 6 if up else -6), ha="center",
                            va="bottom" if up else "top", fontsize=fs, color=COL, linespacing=1.0)
    ax.axhline(sft, color=C["BASELINE_LIGHT"], linestyle="--", linewidth=0.9, zorder=0)
    ax.text(29000, sft - 0.3, f"SFT 4B alone (Nov 6) {sft:.1f}", fontsize=fs, color=C["MUTED"], va="top", ha="right")
    ax.set_xscale("log"); ax.set_xticks([2000, 4000, 8000, 16000]); ax.set_xticklabels(["2K", "4K", "8K", "16K"]); ax.minorticks_off()
    ax.set_xlim(1600, 30000); ax.set_ylim(min(vals) - 4.2, max(vals) + 3.6)
    ax.set_xlabel("5-action sets in training data")
    ax.yaxis.set_major_locator(MultipleLocator(2 if max(vals) - min(vals) < 12 else 4))
    ax.grid(axis="y", color=C["BORDER"], linewidth=0.6, zorder=0)
    ax.tick_params(colors=C["MUTED"], length=3, width=0.6, pad=2)
    for s in ("top", "right"): ax.spines[s].set_visible(False)


def legend(fig, ncol):
    h = [Line2D([], [], color=COL, marker="o", markersize=5, linewidth=1.4, label="Fewer states (8.9 sets/state)"),
         Line2D([], [], color=COL, marker="D", markersize=5, linewidth=1.4, label="Fewer sets per state (2.1K states)"),
         Line2D([], [], color=COL, marker="s", linestyle="none", markersize=5, label="Full data"),
         Line2D([], [], color=COL, marker="o", linestyle="none", markersize=5, label="Nov 8–9 rerun, same window"),
         Line2D([], [], color=FADE, marker="o", markerfacecolor="white", linestyle="none", markersize=5, label="Nov 4–6 batch")]
    fig.legend(handles=h, loc="lower center", ncol=ncol, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.02), handletextpad=0.4, columnspacing=1.2)


def save(fig, stem):
    for ext in ("png", "pdf", "svg"):
        output = HERE / f"{stem}.{ext}"
        fig.savefig(output, dpi=220 if ext == "png" else None, bbox_inches="tight")
        if ext == "svg":
            output.write_text("\n".join(line.rstrip() for line in output.read_text().splitlines()) + "\n")
    plt.close(fig)


subs = list(SETS)
fig, ax = plt.subplots(figsize=(4.4, 3.5))
draw(ax, {k: old[k]["mean_overall"] for k in subs}, {k: v["mean_overall"] for k, v in new.items()}, old["sft"]["mean_overall"], 7)
ax.set_ylabel("Mean overall success (%)")
legend(fig, 2); fig.tight_layout(rect=(0, 0.16, 1, 1))
save(fig, "selection-recipe3-data-ablations-rerun-20261109")

fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
for ax, (b, title) in zip(axes, BENCH):
    draw(ax, {k: old[k][b]["overall"] for k in subs}, {k: v[b]["overall"] for k, v in new.items()}, old["sft"][b]["overall"], 6.5)
    ax.set_title(title, fontsize=10); ax.set_ylabel("Overall success (%)")
legend(fig, 5); fig.tight_layout(rect=(0, 0.08, 1, 1))
save(fig, "selection-recipe3-data-ablations-rerun-per-benchmark-20261109")
