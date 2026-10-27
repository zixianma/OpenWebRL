#!/usr/bin/env python3
"""Label sources, one panel per evaluation setup: gain over one SFT 4B actor sample, shared y-axis.

(1) Offline, training states: the 1,380 SFT-action branch states the heads were trained on. Labels came from rollouts 0-2;
    each state's first five SFT proposals (duplicates kept) are scored on held-out rollouts 3-7.
(2) Offline, held-out states: 345 test states (86 tasks), same five proposals and rollouts 3-7.
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
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
S = json.loads((HERE / "selection-label-same-state-20261027.json").read_text())
LIVE = json.loads((HERE / "selection-data-scale-20261023.json").read_text())["three_benchmarks"]
EXTRA = json.loads((HERE / "selection-label-stages-20261027.json").read_text()).get("live", {})
INK, MUTED = "#222222", "#666666"
ROWS = [  # tick label, colour, source key, live key
    ("Outcome", "#1baf7a", "outcome", "Ours outcome"),
    ("Luna\n3 after", "#1f5fae", "teacher_k3", "Ours K3"),
    ("Luna\n1 after", "#93bdef", "teacher_k1", "Ours K1"),
    ("Luna\npost", "#eb6834", "teacher_post", "Ours post"),
    ("Luna\npre", "#7b4fd0", "teacher_pre", "Ours pre")]
tr, te = S["splits"]["train"], S["splits"]["test"]
PANELS = [
    (f"Offline · training states\n{tr['states']:,} states, {tr['tasks']} tasks; held-out rollouts 3–7", "train",
     S["train"]["sft_sample"]),
    (f"Offline · held-out test states\n{te['states']} states, {te['tasks']} tasks; held-out rollouts 3–7", "test",
     S["test"]["sft_sample"]),
    ("Live · 3 benchmarks\n1,045 tasks, mean overall success", "live", LIVE["SFT 4B"]["mean_overall"])]


def point(ax, x, v, color, marker, filled):
    d, lo, hi = v
    ax.errorbar(x, d, yerr=[[d - lo], [hi - d]], fmt=marker, ms=6.5, color=color, ecolor="#555555", capsize=2, lw=0.9,
                zorder=3, markeredgewidth=1.4, markerfacecolor=color if filled else "white", markeredgecolor=color)
    ax.text(x, hi + 0.2, f"{d:+.1f}".replace("-", "−"), ha="center", va="bottom", fontsize=6.5, color=INK)


fig, axes = plt.subplots(1, 3, figsize=(11, 4.3), sharey=True, gridspec_kw=dict(width_ratios=[1.25, 1.25, 1]))
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
handles = [Line2D([], [], marker=m, ls="", color="#555555", ms=6.5, markeredgewidth=1.4,
                  markerfacecolor="#555555" if f else "white") for m, f in (("o", False), ("s", True), ("D", True))]
fig.legend(handles, ["Teacher's pick (label)", "Head trained on the labels (seed 0)", "Same head, live"],
           frameon=False, fontsize=8, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.06))
fig.text(0.01, 0.01, "Offline: each state's first five SFT proposals, scored on rollouts 3–7 (labels used rollouts 0–2 of the "
         "training states). Bars: paired 95% intervals (bootstrap over tasks). Each panel's baseline is one SFT sample on "
         "its own states or tasks.", fontsize=6.5, color=MUTED)
fig.tight_layout(rect=(0, 0.12, 1, 1))
fig.savefig(HERE / "selection-label-setups-20261027.png", dpi=160); plt.close(fig)
