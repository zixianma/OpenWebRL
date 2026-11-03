#!/usr/bin/env python3
"""Labels from teacher to live (5-fold version of selection-label-stages-20261027): for each label source, gain over one
sample of the same SFT 4B actor at three stages.

(1) Teacher pick, offline: the label's argmax among each state's first five SFT proposals (duplicates kept, ties averaged),
    all 1,725 SFT-action branch states (433 tasks), held-out rollouts 4-8, valid draws only.
(2) Head trained on those labels, offline: task-level 5-fold, so every state is scored by heads trained on the other four
    folds (about 1,380 states; 3 seeds averaged per state). Same states and rollouts as (1).
    (1)-(2): task-clustered bootstrap, 10,000 resamples, seed 20261101 (selection-label-kfold-20261101.json).
(3) Head live: the seed-0 head trained on the original 1,380 training states, three-benchmark mean overall success
    (Online-Mind2Web 300 by gpt-5.6-terra, WebVoyager 595 and DeepShop 150 by GPT-4o) minus one SFT 4B sample, paired 95%
    bootstrap over tasks. Every head was run on the same three benchmarks in one same-day batch, 3 runs each with SFT alone (selection-label-live-3run-20261103.json, paired gain over SFT, per-task success averaged over runs).
Stages (1)-(2) share states; (3) uses different tasks, so compare orderings across stages, not sizes. The prior judge
formulations (WebArbiter, WebShepherd, PRO-CUA) were scored as teachers on a different 1,010-state set and are left out;
they remain in selection-label-stages-20261027.png.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
K = json.loads((HERE / "selection-label-kfold-20261101.json").read_text())["pooled"]
LIVE3 = json.loads((HERE / "selection-label-live-3run-20261103.json").read_text())  # same-day 3-run batch
LIVE3_KEY = {"teacher_pre": "pre", "teacher_post": "post", "teacher_k1": "h1", "teacher_k3": "h3",
             "teacher_k5": "h5", "outcome": "oracle"}
INK, MUTED = "#222222", "#666666"
ROWS = [  # tick label, colour, source key, live key
    ("Pre-exec", "#4B5563", "teacher_pre", "Ours pre"),
    ("Post-exec", "#D8877E", "teacher_post", "Ours post"),
    ("Lookahead\nH = 1", "#A99BE3", "teacher_k1", "Ours K1"),
    ("Lookahead\nH = 3", "#6250BE", "teacher_k3", "Ours K3"),
    ("Lookahead\nH = 5", "#33246E", "teacher_k5", "K5"),
    ("Oracle", "#ED7D00", "outcome", "Ours outcome")]
MARK = [(f"Teacher's pick, offline ({K['states']:,} states, 5-fold)", "o", False),
        ("Head trained on the labels, offline (out-of-fold, 3 seeds)", "s", False),
        ("Head, live (seed 0, 3 runs; mean of 3 benchmarks, 1,045 tasks)", "D", True)]
DX = (-0.24, 0, 0.24)


def values(src, live):
    lv = LIVE3["paired"][LIVE3_KEY[src] + " - sft"]["mean"]
    return [K[src]["teacher_minus_sft"], K[src]["head_minus_sft"], lv]


fig, ax = plt.subplots(figsize=(9.2, 4.2))
ax.axhline(0, color=MUTED, lw=1, ls="--", zorder=0)
for x, (_, color, src, live) in enumerate(ROWS):
    for (_, m, filled), dx, v in zip(MARK, DX, values(src, live)):
        if v is None:
            ax.text(x + dx, 0.4, "not run\nlive", ha="center", va="bottom", fontsize=6, color=MUTED)
            continue
        d, lo, hi = v
        ax.errorbar(x + dx, d, yerr=[[d - lo], [hi - d]], fmt=m, ms=6.5, color=color, ecolor="#555555", capsize=2,
                    lw=0.9, zorder=3, markeredgewidth=1.4, markerfacecolor=color if filled else "white",
                    markeredgecolor=color)
        ax.text(x + dx, hi + 0.25, f"{d:+.1f}".replace("-", "−"), ha="center", va="bottom", fontsize=6.3, color=INK)
ax.set_xticks(range(len(ROWS)), [r[0] for r in ROWS], fontsize=8); ax.set_xlim(-0.55, len(ROWS) - 0.45)
ax.set_ylim(-4, 10); ax.set_ylabel("Gain over one SFT 4B actor sample (pp)", fontsize=9)
ax.grid(axis="y", alpha=.25)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
handles = [Line2D([], [], marker=m, ls="", color="#555555", ms=6.5, markeredgewidth=1.4,
                  markerfacecolor="#555555" if f else "white") for _, m, f in MARK]
ax.legend(handles, [l for l, _, _ in MARK], frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.45, -0.17), ncol=1)
fig.text(0.01, 0.01, "Offline: first five SFT proposals per state, held-out rollouts 4–8 (labels used rollouts 1–3; Oracle = best "
         f"action on rollouts 1–3); one SFT sample = {K['sft_sample']:.1f}%. Live: one SFT sample = "
         f"{LIVE3['rates']['sft']['mean_overall']:.1f}%.\nBars: 95% intervals, task-clustered bootstrap (offline) or paired over tasks "
         "(live). Offline and live use different tasks: compare orderings, not sizes.", fontsize=6.5, color=MUTED)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(HERE / "selection-label-stages-kfold-20261101.png", dpi=160); plt.close(fig)
