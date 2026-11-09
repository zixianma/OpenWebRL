#!/usr/bin/env python3
"""Labels from teacher to live (5-fold version of selection-label-stages-20261027): for each label source, gain over one
sample of the same SFT 4B actor at three stages.

(1) Teacher pick, offline: the label's argmax among each state's first five SFT proposals (duplicates kept, ties averaged),
    all 1,725 SFT-action branch states (433 tasks), held-out rollouts 4-8, valid draws only.
(2) Head trained on those labels, offline: task-level 5-fold, so every state is scored by heads trained on the other four
    folds (about 1,380 states; 3 seeds averaged per state). Same states and rollouts as (1).
    (1)-(2): task-clustered bootstrap, 10,000 resamples, seed 20261101 (selection-label-kfold-20261101.json).
(3) Head live: the seed-0 head trained on the original 1,380 training states, task-weighted overall success over the three benchmarks
    (Online-Mind2Web 300 by gpt-5.6-terra, WebVoyager 595 and DeepShop 150 by GPT-4o) minus one SFT 4B sample, paired 95%
    bootstrap over tasks. Every head was run on the same three benchmarks in one same-day batch, 3 runs each with SFT alone (selection-label-live-3run-20261103.json "weighted": paired gain over SFT, each of the 1,045 tasks weighted equally, per-task success averaged over runs).
Stages (1)-(2) share states; (3) uses different tasks, so compare orderings across stages, not sizes. The prior judge
formulations (WebArbiter, WebShepherd, PRO-CUA) were scored as teachers on a different 1,010-state set and are left out;
they remain in selection-label-stages-20261027.png.
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
        target = path.with_suffix(suffix)
        fig.savefig(target, **kwargs)
        if suffix == ".svg":
            target.write_text("\n".join(line.rstrip() for line in target.read_text().splitlines()) + "\n")


def student_outline(source_color):
    return C["TRAIN"] if source_color == C["PRE_EXEC"] else source_color


from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
K = json.loads((HERE / "selection-label-kfold-20261101.json").read_text())["pooled"]
LIVE3 = json.loads((HERE / "selection-label-live-3run-20261103.json").read_text())  # same-day 3-run batch
LIVE3_KEY = {"teacher_pre": "pre", "teacher_post": "post", "teacher_k1": "h1", "teacher_k3": "h3",
             "teacher_k5": "h5", "outcome": "oracle"}
INK, MUTED = C["INK"], C["MUTED"]
ROWS = [  # tick label, colour, source key, live key
    ("Pre-exec", C["PRE_EXEC"], "teacher_pre", "Ours pre"),
    ("Lookahead\nH = 2", C["LOOKAHEAD_TEACHER"], "teacher_k1", "Ours K1"),
    ("Lookahead\nH = 4", C["LOOKAHEAD_TEACHER"], "teacher_k3", "Ours K3"),
    ("Lookahead\nH = 6", C["LOOKAHEAD_TEACHER"], "teacher_k5", "K5"),
    ("Oracle", C["ORACLE"], "outcome", "Ours outcome")]
MARK = [(f"Teacher's pick, offline ({K['states']:,} states, 5-fold)", "o", False),
        ("Head trained on the labels, offline (out-of-fold, 3 seeds)", "s", True),
        ("Head, live (seed 0, 3 runs; task-weighted, 1,045 tasks)", "D", True)]
DX = (-0.24, 0, 0.24)


def values(src, live):
    lv = LIVE3["weighted"][LIVE3_KEY[src]]["minus_sft"]
    return [K[src]["teacher_minus_sft"], K[src]["head_minus_sft"], lv]


fig, ax = plt.subplots(figsize=(9.2, 4.2))
ax.axhline(0, color=MUTED, lw=1, ls="--", zorder=0)
for x, (_, color, src, live) in enumerate(ROWS):
    for (_, m, filled), dx, v in zip(MARK, DX, values(src, live)):
        if v is None:
            ax.text(x + dx, 0.4, "not run\nlive", ha="center", va="bottom", fontsize=6, color=MUTED)
            continue
        d, lo, hi = v
        ax.errorbar(x + dx, d, yerr=[[d - lo], [hi - d]], fmt=m, ms=6.5, color=color, ecolor=C["BASELINE_DARK"], capsize=2,
                    lw=0.9, zorder=3, markeredgewidth=1.7, markerfacecolor=C["ADM_STUDENT"] if filled else "white",
                    markeredgecolor=student_outline(color) if filled else color)
        ax.text(x + dx, hi + 0.25, f"{d:+.1f}".replace("-", "−"), ha="center", va="bottom", fontsize=6.3, color=INK)
ax.set_xticks(range(len(ROWS)), [r[0] for r in ROWS], fontsize=8); ax.set_xlim(-0.55, len(ROWS) - 0.45)
ax.set_ylim(-4, 10); ax.set_ylabel("Gain over one SFT 4B actor sample (pp)", fontsize=9)
ax.grid(axis="y", alpha=.25)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
handles = [Line2D([], [], marker=m, ls="", color=C["BASELINE_DARK"], ms=6.5, markeredgewidth=1.4,
                  markerfacecolor=C["ADM_STUDENT"] if f else "white",
                  markeredgecolor=C["TRAIN"] if f else C["BASELINE_DARK"]) for _, m, f in MARK]
ax.legend(handles, [l for l, _, _ in MARK], frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.45, -0.17), ncol=1)
fig.text(0.01, 0.01, "Offline: first five SFT proposals per state, held-out rollouts 4–8 (labels used rollouts 1–3; Oracle = best "
         f"action on rollouts 1–3); one SFT sample = {K['sft_sample']:.1f}%. Live: one SFT sample = "
         f"{LIVE3['weighted']['sft']['overall']:.1f}%.\nBars: 95% intervals, task-clustered bootstrap (offline) or paired over tasks "
         "(live). Offline and live use different tasks: compare orderings, not sizes.", fontsize=6.5, color=MUTED)
fig.tight_layout(rect=(0, 0.07, 1, 1))
_paper_save(fig, HERE / "selection-label-stages-kfold-20261101.png", dpi=160); plt.close(fig)
