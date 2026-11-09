#!/usr/bin/env python3
"""Label sources from teacher to live, scored on the same states (5-fold version of selection-label-same-state-20261027):
for each label source, gain over one sample of the SFT 4B actor at five points.

Offline: SFT-action branch states, each state's first five SFT proposals (duplicates kept) scored on held-out rollouts 4-8,
valid draws only. Teacher = the label's argmax among the five (ties averaged); head = the 22.6 M head, 3 seeds averaged per state.
  - Train states: the original 1,380 training states (347 tasks), scored by the heads trained on them (fold 0 of the 5-fold
    run; labels came from rollouts 1-3, so rollouts 4-8 are still held out). selection-label-train-states-kfold-20261101.json.
  - Held-out states: all 1,725 states (433 tasks), task-level 5-fold, each scored by heads trained on the other four folds.
    selection-label-kfold-20261101.json ("pooled").
Live: the seed-0 head, task-weighted overall success over the three benchmarks (Online-Mind2Web 300 by gpt-5.6-terra, WebVoyager 595 and
DeepShop 150 by GPT-4o) minus SFT 4B. Every head was run on the same three benchmarks in one same-day batch, 3 runs each with SFT alone (selection-label-live-3run-20261103.json "weighted": paired gain over SFT, each of the 1,045 tasks weighted equally, per-task success averaged over runs).
Each point is relative to one SFT sample on its own states/tasks; 95% intervals, task-clustered bootstrap (offline) or paired
over tasks (live).
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

from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
TR = json.loads((HERE / "selection-label-train-states-kfold-20261101.json").read_text())
KF = json.loads((HERE / "selection-label-kfold-20261101.json").read_text())["pooled"]
LIVE3 = json.loads((HERE / "selection-label-live-3run-20261103.json").read_text())  # same-day 3-run batch
LIVE3_KEY = {"teacher_pre": "pre", "teacher_post": "post", "teacher_k1": "h1", "teacher_k3": "h3",
             "teacher_k5": "h5", "outcome": "oracle"}
INK, MUTED, GRID = "#222222", "#666666", "#e6e6e6"
ROWS = [  # label, colour, source key, live key
    ("Pre-exec", "#4B5563", "teacher_pre", "Ours pre"),
    ("Lookahead\nH = 2", "#A99BE3", "teacher_k1", "Ours K1"),
    ("Lookahead\nH = 4", "#6250BE", "teacher_k3", "Ours K3"),
    ("Lookahead\nH = 6", "#33246E", "teacher_k5", "K5"),
    ("Oracle", "#ED7D00", "outcome", "Ours outcome")]
MARK = [  # legend, marker, filled, dx
    (f"Teacher, train states ({TR['states']:,})", "o", False, -0.32),
    ("Head, train states", "o", True, -0.18),
    (f"Teacher, held-out (5-fold, {KF['states']:,} states)", "s", False, -0.02),
    ("Head, held-out (out-of-fold)", "s", True, 0.12),
    ("Head, live (3 runs; task-weighted, 1,045 tasks)", "D", True, 0.32)]


def values(src, live):
    lv = LIVE3["weighted"][LIVE3_KEY[src]]["minus_sft"]
    t = TR["train"][src]
    return [t["teacher_minus_sft"], t["head_minus_sft"], KF[src]["teacher_minus_sft"], KF[src]["head_minus_sft"], lv]


fig, ax = plt.subplots(figsize=(10, 4.4))
ax.axhline(0, color=MUTED, lw=1, ls="--", zorder=0)
for x, (name, color, src, live) in enumerate(ROWS):
    ax.axvspan(x + 0.22, x + 0.42, color="#f3f3f3", zorder=0, lw=0)
    for (_, m, filled, dx), v in zip(MARK, values(src, live)):
        if v is None:
            ax.text(x + dx, 0.4, "not run live", ha="center", va="bottom", fontsize=6, color=MUTED, rotation=90)
            continue
        d, lo, hi = v
        ax.errorbar(x + dx, d, yerr=[[d - lo], [hi - d]], fmt=m, ms=6, color=color, ecolor="#555555", capsize=1.8,
                    lw=0.9, zorder=3, markeredgewidth=1.4, markerfacecolor=color if filled else "white",
                    markeredgecolor=color)
        ax.text(x + dx, hi + 0.2, f"{d:+.1f}".replace("-", "−"), ha="center", va="bottom", fontsize=5.8, color=INK)
for x in range(1, len(ROWS)):
    ax.axvline(x - 0.5, color=GRID, lw=1)
ax.set_xticks(range(len(ROWS)), [r[0] for r in ROWS], fontsize=8); ax.set_xlim(-0.55, len(ROWS) - 0.45)
ax.set_ylim(-4, 10); ax.set_ylabel("Gain over one SFT 4B actor sample (pp)", fontsize=9)
ax.grid(axis="y", alpha=.25)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
handles = [Line2D([], [], marker=m, ls="", color="#555555", ms=6, markeredgewidth=1.4,
                  markerfacecolor="#555555" if f else "white") for _, m, f, _ in MARK]
ax.legend(handles, [l for l, *_ in MARK], frameon=False, fontsize=7.3, loc="upper center", bbox_to_anchor=(0.5, -0.15),
          ncol=3, columnspacing=1.2)
fig.text(0.01, 0.01, "Hollow = teacher, solid = head; circle = train states, square = held-out (5-fold); live shaded. Offline: first "
         "five SFT proposals per state, held-out rollouts 4–8; Oracle = best action\non rollouts 1–3; offline heads average 3 seeds, "
         "the live head is seed 0. Bars: 95% intervals (task-clustered bootstrap offline, paired over tasks live). Each point is "
         f"relative to one SFT sample\non its own states or tasks (train {TR['train']['sft_sample']:.1f}%, held-out "
         f"{KF['sft_sample']:.1f}%, live task-weighted {LIVE3['weighted']['sft']['overall']:.1f}%).", fontsize=6.5, color=MUTED)
fig.tight_layout(rect=(0, 0.1, 1, 1))
fig.savefig(HERE / "selection-label-same-state-kfold-20261101.png", dpi=160); plt.close(fig)
