#!/usr/bin/env python3
"""Label sources, one panel per evaluation setup: gain over one SFT 4B actor sample, shared y-axis.

(1) Offline, training states: the 1,380 SFT-action branch states the heads were trained on. Labels came from rollouts 1-3;
    each state's first five SFT proposals (duplicates kept) are scored on held-out rollouts 4-8.
(2) Offline, held-out states: 345 test states (86 tasks), same five proposals and rollouts 4-8.
    Teacher = the label's argmax among the five (ties averaged); head = the seed-0 head, the one run live
    (selection-label-same-state-20261027.json).
(3) Live: the seed-0 head, three-benchmark mean overall success (Online-Mind2Web 300 by gpt-5.6-terra, WebVoyager 595 and
    DeepShop 150 by GPT-4o) (selection-data-scale-20261023.json; post-execution from selection-label-stages-20261027.json).
Each panel's dashed line is one SFT sample on that panel's states or tasks; paired 95% bootstrap over tasks.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# Shared paper typography, including bundled Albert Sans fonts.
import sys as _paper_sys
from pathlib import Path as _PaperPath
_paper_sys.path.insert(0, str(next(p for p in _PaperPath(__file__).resolve().parents
                                if (p / "paper_plot_style.py").is_file())))
import paper_plot_style as _paper_style
from paper_plot_palette import COLORS as C, tint


def _paper_save(fig, path, **kwargs):
    """Keep raster and vector siblings synchronized with the same layout."""
    for suffix in (".png", ".pdf", ".svg"):
        fig.savefig(path.with_suffix(suffix), **kwargs)


def student_outline(source_color):
    return C["TRAIN"] if source_color == C["PRE_EXEC"] else source_color


from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
S = json.loads((HERE / "selection-label-same-state-20261027.json").read_text())
LIVE = json.loads((HERE / "selection-data-scale-20261023.json").read_text())["three_benchmarks"]
EXTRA = json.loads((HERE / "selection-label-stages-20261027.json").read_text()).get("live", {})
INK, MUTED = C["INK"], C["MUTED"]
ROWS = [  # tick label, colour, source key, live key
    ("Pre-exec", C["PRE_EXEC"], "teacher_pre", "Ours pre"),
    ("Lookahead\nH = 2", C["LOOKAHEAD_TEACHER"], "teacher_k1", "Ours K1"),
    ("Lookahead\nH = 4", C["LOOKAHEAD_TEACHER"], "teacher_k3", "Ours K3"),
    ("Oracle", C["ORACLE"], "outcome", "Ours outcome")]
tr, te = S["splits"]["train"], S["splits"]["test"]
PANELS = [
    (f"Offline · training states\n{tr['states']:,} states, {tr['tasks']} tasks; held-out rollouts 4–8", "train",
     S["train"]["sft_sample"]),
    (f"Offline · held-out test states\n{te['states']} states, {te['tasks']} tasks; held-out rollouts 4–8", "test",
     S["test"]["sft_sample"]),
    ("Live · 3 benchmarks\n1,045 tasks, mean overall success", "live", LIVE["SFT 4B"]["mean_overall"])]


def point(ax, x, v, color, marker, filled):
    d, lo, hi = v
    ax.errorbar(x, d, yerr=[[d - lo], [hi - d]], fmt=marker, ms=6.5, color=color, ecolor=C["BASELINE_DARK"], capsize=2, lw=0.9,
                zorder=3, markeredgewidth=1.7, markerfacecolor=C["ADM_STUDENT"] if filled else "white",
                markeredgecolor=student_outline(color) if filled else color)
    ax.text(x, hi + 0.2, f"{d:+.1f}".replace("-", "−"), ha="center", va="bottom", fontsize=6.5, color=INK)


fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), sharey=True, gridspec_kw=dict(width_ratios=[1, 1, 1]))
for ax, (title, split, base) in zip(axes, PANELS):
    ax.axhline(0, color=MUTED, lw=1, ls="--", zorder=0)
    for x, (_, color, src, live) in enumerate(ROWS):
        if split == "live":
            v = EXTRA.get(live) or LIVE[live]["minus_sft4b"]
            point(ax, x, v, color, "D", True)
        else:
            point(ax, x - 0.17, S[split][src]["teacher_minus_sft"], color, "o", False)
            point(ax, x + 0.17, S[split][src]["head_minus_sft"], color, "s", True)
    ax.set_title(f"{title}\nSFT 4B, one sample (dashed) = {base:.1f}%", fontsize=8.5, loc="left", color=INK)
    ax.set_xticks(range(len(ROWS)), [r[0] for r in ROWS], fontsize=7.8); ax.set_xlim(-0.6, len(ROWS) - 0.4)
    ax.grid(axis="y", alpha=.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
axes[0].set_ylim(-4.5, 10); axes[0].set_ylabel("Gain over one SFT 4B actor sample (pp)", fontsize=9)
handles = [Line2D([], [], marker=m, ls="", color=C["BASELINE_DARK"], ms=6.5, markeredgewidth=1.4,
                  markerfacecolor=C["ADM_STUDENT"] if f else "white",
                  markeredgecolor=C["TRAIN"] if f else C["BASELINE_DARK"]) for m, f in (("o", False), ("s", True), ("D", True))]
fig.legend(handles, ["Teacher's pick (label)", "Head trained on the labels (seed 0)", "Same head, live"],
           frameon=False, fontsize=8, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.06))
fig.text(0.01, 0.01, "Offline: each state's first five SFT proposals, scored on held-out rollouts 4–8 (labels used rollouts 1–3 of the "
         "training states; Oracle = best action on rollouts 1–3).\nBars: paired 95% intervals (bootstrap over tasks). Each panel's baseline is one SFT sample on "
         "its own states or tasks.", fontsize=6.5, color=MUTED)
fig.tight_layout(rect=(0, 0.12, 1, 1))
_paper_save(fig, HERE / "selection-label-setups-20261027.png", dpi=160); plt.close(fig)
