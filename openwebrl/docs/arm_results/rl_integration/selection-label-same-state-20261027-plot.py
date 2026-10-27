#!/usr/bin/env python3
"""Label sources from teacher to live, scored on the same states: for each label source, gain over one sample of the
SFT 4B actor at five points.

Offline (selection-label-same-state-20261027.json): SFT-action branch states, each state's first five SFT proposals
(duplicates kept) scored on held-out draws 3-7.
  - Teacher: the label's argmax among the five (ties averaged). Head: the 22.6 M head, mean of three seeds' picks.
  - Train states: the 1,380 states the heads were trained on (labels came from draws 0-2, so draws 3-7 are still held out).
  - Test states: 345 held-out states (86 tasks).
Live: the seed-0 head, three-benchmark mean overall success (Online-Mind2Web 300 by gpt-5.6-terra, WebVoyager 595 and
DeepShop 150 by GPT-4o) minus SFT 4B (selection-data-scale-20261023.json; post-execution from
selection-label-stages-20261027.json "live" when present).
Each point is relative to one SFT sample on its own states/tasks; paired 95% bootstrap over tasks.
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
INK, MUTED, GRID = "#222222", "#666666", "#e6e6e6"
ROWS = [  # label, colour, source key, live key
    ("Observed\noutcome", "#1baf7a", "outcome", "Ours outcome"),
    ("Luna,\n3 steps after", "#1f5fae", "teacher_k3", "Ours K3"),
    ("Luna,\n1 step after", "#93bdef", "teacher_k1", "Ours K1"),
    ("Luna,\npost-execution", "#eb6834", "teacher_post", "Ours post"),
    ("Luna,\npre-execution", "#7b4fd0", "teacher_pre", "Ours pre")]
tr, te = S["splits"]["train"], S["splits"]["test"]
MARK = [  # legend, marker, filled, dx
    (f"Teacher, train states ({tr['states']:,})", "o", False, -0.32),
    (f"Head, train states", "s", False, -0.18),
    (f"Teacher, held-out test states ({te['states']})", "o", True, -0.02),
    (f"Head, held-out test states", "s", True, 0.12),
    ("Head, live (mean of 3 benchmarks, 1,045 tasks)", "D", True, 0.32)]


def values(src, live):
    lv = EXTRA.get(live) or (LIVE[live]["minus_sft4b"] if live in LIVE else None)
    return [S["train"][src]["teacher_minus_sft"], S["train"][src]["head_minus_sft"],
            S["test"][src]["teacher_minus_sft"], S["test"][src]["head_minus_sft"], lv]


fig, ax = plt.subplots(figsize=(9.2, 4.4))
ax.axhline(0, color=MUTED, lw=1, ls="--", zorder=0)
ax.text(-0.5, -0.3, "one SFT sample", ha="left", va="top", fontsize=7.5, color=MUTED)
for x, (name, color, src, live) in enumerate(ROWS):
    ax.axvspan(x + 0.22, x + 0.42, color="#f3f3f3", zorder=0, lw=0)
    for (_, m, filled, dx), v in zip(MARK, values(src, live)):
        if v is None:
            ax.text(x + dx, 0.4, "pending", ha="center", va="bottom", fontsize=6, color=MUTED, rotation=90)
            continue
        d, lo, hi = v
        ax.errorbar(x + dx, d, yerr=[[d - lo], [hi - d]], fmt=m, ms=6, color=color, ecolor="#555555", capsize=1.8,
                    lw=0.9, zorder=3, markeredgewidth=1.4, markerfacecolor=color if filled else "white",
                    markeredgecolor=color)
        ax.text(x + dx, hi + 0.2, f"{d:+.1f}".replace("-", "−"), ha="center", va="bottom", fontsize=5.8, color=INK)
for x in range(1, len(ROWS)):
    ax.axvline(x - 0.5, color=GRID, lw=1)
ax.set_xticks(range(len(ROWS)), [r[0] for r in ROWS], fontsize=8); ax.set_xlim(-0.55, len(ROWS) - 0.45)
ax.set_ylim(-4.5, 10); ax.set_ylabel("Gain over one SFT 4B actor sample (pp)", fontsize=9)
ax.grid(axis="y", alpha=.25)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
handles = [Line2D([], [], marker=m, ls="", color="#555555", ms=6, markeredgewidth=1.4,
                  markerfacecolor="#555555" if f else "white") for _, m, f, _ in MARK]
ax.legend(handles, [l for l, *_ in MARK], frameon=False, fontsize=7.3, loc="upper center", bbox_to_anchor=(0.5, -0.15),
          ncol=3, columnspacing=1.2)
fig.text(0.01, 0.01, "Offline: first five SFT proposals per state, scored on held-out draws 3-7; head = mean of 3 seeds. "
         "Live (shaded): seed-0 head. Bars: paired 95% intervals (bootstrap over tasks).\nEach point is relative to one SFT "
         f"sample on its own states or tasks (train {S['train']['sft_sample']:.1f}%, test {S['test']['sft_sample']:.1f}%, "
         f"live 3-benchmark mean {LIVE['SFT 4B']['mean_overall']:.1f}%).", fontsize=6.5, color=MUTED)
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(HERE / "selection-label-same-state-20261027.png", dpi=160); plt.close(fig)
