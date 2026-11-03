#!/usr/bin/env python3
"""Label sources, offline (task-level 5-fold) and live: gain over one SFT 4B actor sample.

(1) Offline, 5-fold: all 1,725 SFT-action branch states (433 tasks) split into five task-level folds (fold 0 = the
    original 345-state / 86-task test split). Each fold's heads (3 seeds, scores averaged per state) are trained on the
    other four folds' states, so every state is scored out-of-fold. Each state's first five SFT proposals (duplicates
    kept) are scored on held-out rollouts 4-8, valid draws only. Teacher = the label's argmax among the five (ties
    averaged). Task-clustered bootstrap, 10,000 resamples, seed 20261101 (selection-label-kfold-20261101.json).
(2) Live: the seed-0 head trained on the original 1,380 training states, three-benchmark mean overall success
    (Online-Mind2Web 300 by gpt-5.6-terra, WebVoyager 595 and DeepShop 150 by GPT-4o). Every head was run on the same three benchmarks in one same-day batch, 3 runs each with SFT alone (selection-label-live-3run-20261103.json, paired gain over SFT, per-task success averaged over runs).
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
K = json.loads((HERE / "selection-label-kfold-20261101.json").read_text())
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
P = K["pooled"]
PANELS = [
    (f"Offline · task-level 5-fold, out-of-fold heads\n{P['states']:,} states, {P['tasks']} tasks; held-out rollouts 4–8",
     "offline", P["sft_sample"]),
    ("Live · 3 benchmarks\n1,045 tasks, mean overall success", "live", LIVE3["rates"]["sft"]["mean_overall"])]


def point(ax, x, v, color, marker, filled):
    d, lo, hi = v
    ax.errorbar(x, d, yerr=[[d - lo], [hi - d]], fmt=marker, ms=6.5, color=color, ecolor="#555555", capsize=2, lw=0.9,
                zorder=3, markeredgewidth=1.4, markerfacecolor=color if filled else "white", markeredgecolor=color)
    ax.text(x, hi + 0.2, f"{d:+.1f}".replace("-", "−"), ha="center", va="bottom", fontsize=6.5, color=INK)


fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True, gridspec_kw=dict(width_ratios=[1.15, 1]))
for ax, (title, kind, base) in zip(axes, PANELS):
    ax.axhline(0, color=MUTED, lw=1, ls="--", zorder=0)
    for x, (_, color, src, live) in enumerate(ROWS):
        if kind == "live":
            if live is None:
                ax.text(x, 0.4, "not run\nlive", ha="center", va="bottom", fontsize=6.5, color=MUTED)
                continue
            point(ax, x, LIVE3["paired"][LIVE3_KEY[src] + " - sft"]["mean"], color, "D", True)
        else:
            point(ax, x - 0.17, P[src]["teacher_minus_sft"], color, "o", False)
            point(ax, x + 0.17, P[src]["head_minus_sft"], color, "s", True)
    ax.set_title(f"{title}\nSFT 4B, one sample (dashed) = {base:.1f}%", fontsize=8.5, loc="left", color=INK)
    ax.set_xticks(range(len(ROWS)), [r[0] for r in ROWS], fontsize=7.8); ax.set_xlim(-0.6, len(ROWS) - 0.4)
    ax.grid(axis="y", alpha=.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
axes[0].set_ylim(-4.5, 10); axes[0].set_ylabel("Gain over one SFT 4B actor sample (pp)", fontsize=9)
handles = [Line2D([], [], marker=m, ls="", color="#555555", ms=6.5, markeredgewidth=1.4,
                  markerfacecolor="#555555" if f else "white") for m, f in (("o", False), ("s", True), ("D", True))]
fig.legend(handles, ["Teacher's pick (label)", "Head trained on the labels (3 seeds, out-of-fold)", "Head, live (seed 0, 3 runs)"],
           frameon=False, fontsize=8, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.08))
fig.text(0.01, 0.01, "Offline: each state's first five SFT proposals, scored on held-out rollouts 4–8 (labels used rollouts 1–3; "
         "Oracle = best action on rollouts 1–3). Bars: 95% intervals,\ntask-clustered bootstrap (offline) or paired over "
         "tasks (live). Each panel's baseline is one SFT sample on its own states or tasks.", fontsize=6.5, color=MUTED)
fig.tight_layout(rect=(0, 0.14, 1, 1))
fig.savefig(HERE / "selection-label-setups-kfold-20261101.png", dpi=160); plt.close(fig)
