#!/usr/bin/env python3
"""Label sources, offline (task-level 5-fold) and live: gain over one SFT 4B actor sample.

(1) Offline, 5-fold: all 1,725 SFT-action branch states (433 tasks) split into five task-level folds (fold 0 = the
    original 345-state / 86-task test split). Each fold's heads (3 seeds, scores averaged per state) are trained on the
    other four folds' states, so every state is scored out-of-fold. Each state's first five SFT proposals (duplicates
    kept) are scored on held-out rollouts 4-8, valid draws only. Teacher = the label's argmax among the five (ties
    averaged). Task-clustered bootstrap, 10,000 resamples, seed 20261101 (selection-label-kfold-20261101.json).
(2) Live: the seed-0 head trained on the original 1,380 training states, task-weighted overall success over the three benchmarks
    (Online-Mind2Web 300 by gpt-5.6-terra, WebVoyager 595 and DeepShop 150 by GPT-4o). Every head was run on the same three benchmarks in one same-day batch, 3 runs each with SFT alone (selection-label-live-3run-20261103.json "weighted": paired gain over SFT, each of the 1,045 tasks weighted equally, per-task success averaged over runs).
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
from paper_plot_palette import COLORS as C


plt.rcParams.update({"font.family": "Albert Sans", "pdf.fonttype": 42,
                     "ps.fonttype": 42, "svg.fonttype": "none"})
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
K = json.loads((HERE / "selection-label-kfold-20261101.json").read_text())
LIVE3 = json.loads((HERE / "selection-label-live-3run-20261103.json").read_text())  # same-day 3-run batch
LIVE3_KEY = {"teacher_pre": "pre", "teacher_post": "post", "teacher_k1": "h1", "teacher_k3": "h3",
             "teacher_k5": "h5", "outcome": "oracle"}
INK, MUTED = C["INK"], C["MUTED"]
ROWS = [  # tick label, colour, source key, live key
    ("Pre-\nexec", C["PRE_EXEC"], "teacher_pre", "Ours pre"),
    ("H=1", C["LOOKAHEAD_TEACHER"], "teacher_post", "Ours post"),
    ("H=4", C["LOOKAHEAD_TEACHER"], "teacher_k3", "Ours K3"),
    ("H=6", C["LOOKAHEAD_TEACHER"], "teacher_k5", "K5"),
    ("Oracle", C["ORACLE"], "outcome", "Ours outcome")]
P = K["pooled"]
PANELS = [
    ("Offline: held-out rollout success",
     "offline", P["sft_sample"]),
    ("Online: overall success", "live", LIVE3["weighted"]["sft"]["overall"])]


def point(ax, x, v, color, marker, filled, outline=None):
    d, lo, hi = v
    ax.errorbar(x, d, yerr=[[d - lo], [hi - d]], fmt=marker, ms=4.1, color=color, ecolor=C["BASELINE_DARK"], capsize=1.5, lw=0.75,
                zorder=3, markeredgewidth=1.0,
                markerfacecolor=color if filled else "white",
                markeredgecolor=outline if outline is not None else color)
    ax.text(x, hi + 0.12, f"{d:+.1f}".replace("-", "−"), ha="center", va="bottom", fontsize=5.8, color=INK)


fig, axes = plt.subplots(1, 2, figsize=(3.35, 2.25), sharey=True)
for ax, (title, kind, base) in zip(axes, PANELS):
    ax.axhline(0, color=MUTED, lw=1, ls="--", zorder=0)
    for x, (_, color, src, live) in enumerate(ROWS):
        outline = C["TRAIN"] if src == "teacher_pre" else color
        if kind == "live":
            if live is None:
                ax.text(x, 0.4, "not run\nlive", ha="center", va="bottom", fontsize=6.5, color=MUTED)
                continue
            point(ax, x, LIVE3["weighted"][LIVE3_KEY[src]]["minus_sft"], C["ADM_STUDENT"], "o", True, outline=outline)
        else:
            point(ax, x - 0.17, P[src]["teacher_minus_sft"], color, "o", False)
            point(ax, x + 0.17, P[src]["head_minus_sft"], C["ADM_STUDENT"], "o", True, outline=outline)
    ax.set_title(title, fontsize=7, loc="left", color=INK, pad=6)
    ax.set_xticks(range(len(ROWS)), [r[0] for r in ROWS], fontsize=6.2); ax.set_xlim(-0.35, len(ROWS) - 0.65)
    # The shared label spans whichever horizons are displayed.
    horizon_ticks = [i for i, row in enumerate(ROWS) if row[0].startswith("H=")]
    ax.plot([horizon_ticks[0] - .35, horizon_ticks[-1] + .35], [-.19, -.19], transform=ax.get_xaxis_transform(),
            color=MUTED, lw=.5, clip_on=False)
    ax.text((horizon_ticks[0] + horizon_ticks[-1]) / 2, -.24, "Lookahead", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=6, color=INK)
    ax.tick_params(axis="y", labelsize=6.5)
    ax.tick_params(colors=INK)
    ax.grid(False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
axes[0].set_ylim(-3.5, 5.2)
axes[0].set_yticks([-2, 0, 2, 4])
axes[0].set_ylabel("Gain over SFT (pp)", fontsize=7, color=INK)
handles = [
    Line2D([], [], marker="o", ls="", color=C["LOOKAHEAD_TEACHER"], ms=4.1,
           markeredgewidth=1.0, markerfacecolor="white"),
    Line2D([], [], marker="o", ls="", color=C["LOOKAHEAD_TEACHER"], ms=4.1,
           markeredgewidth=1.0, markerfacecolor=C["ADM_STUDENT"]),
]
fig.legend(handles, ["Teacher ADM", "Student ADM"], frameon=False, fontsize=6.8,
           loc="lower center", ncol=2, bbox_to_anchor=(0.54, 0.005), columnspacing=1.8)
# Cohorts, baselines, seeds and interval definitions are in the document caption.
fig.subplots_adjust(left=0.13, right=0.985, top=0.81, bottom=0.28, wspace=0.10)
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
for artist in [*fig.legends, *[text for ax in axes for text in
                [ax.title, ax.xaxis.label, ax.yaxis.label, *ax.texts,
                 *ax.get_xticklabels(), *ax.get_yticklabels()]]]:
    if artist.get_visible() and (not hasattr(artist, "get_text") or artist.get_text()):
        bounds = artist.get_window_extent(renderer)
        assert fig.bbox.contains(*bounds.p0) and fig.bbox.contains(*bounds.p1), str(artist)
output = HERE / "selection-label-setups-kfold-20261101"
fig.savefig(output.with_suffix(".png"), dpi=300)
fig.savefig(output.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None})
fig.savefig(output.with_suffix(".svg"), metadata={"Date": None})
svg = output.with_suffix(".svg")
svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
plt.close(fig)
