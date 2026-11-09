#!/usr/bin/env python3
"""Labels from teacher to live: for each label source, gain over one sample of the same SFT 4B actor at three stages.

(1) Teacher pick, offline: Luna's (or the oracle's) pick on the 1,010 main-set branch states, scored on held-out rollouts 4-8,
    minus the actor's first sample (valid draws only; paired 95% task-clustered interval;
    lookahead-draw-split-20261013/curve_fixedfit_main.json).
(2) Head trained on those labels, offline: the 22.6 M head (seed 0) trained on the 1,380 branch states, fixed-five pick on
    the 348 held-out test states (86 tasks), held-out rollouts 4-8, minus one SFT sample (paired 95% interval;
    round1/<source>/fixed-five.json, copied in selection-label-stages-20261027.json).
(3) Same head, live: three-benchmark mean overall success (Online-Mind2Web 300 by gpt-5.6-terra, WebVoyager 595 and
    DeepShop 150 by GPT-4o) minus one SFT 4B sample (paired 95% bootstrap over tasks; selection-data-scale-20261023.json).
The three stages use different states and tasks, so compare orderings within a stage, not magnitudes across stages.
"Oracle" is the best action on rollouts 1-3 (observed outcome) at stage 1 and the head trained on those labels at stages 2-3. Prior
judge formulations (WebArbiter, WebShepherd, PRO-CUA, all Luna) exist only as teachers. Post-execution heads are filled in from
selection-label-stages-20261027.json when present.
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
TEACH = json.loads((HERE / "lookahead-draw-split-20261013/curve_fixedfit_main.json").read_text())["valid_only"]["3-7"]["pooled"]["minus_actor"]
STAGES = json.loads((HERE / "selection-label-stages-20261027.json").read_text())   # head offline (+ post-execution live)
LIVE = json.loads((HERE / "selection-data-scale-20261023.json").read_text())["three_benchmarks"]
INK, MUTED, GRID = C["INK"], C["MUTED"], C["BORDER"]
ROWS = [  # label, colour, teacher key, head-offline key, live key
    ("Pre-exec", C["PRE_EXEC"], "Before execution", "teacher_pre", "Ours pre"),
    ("Lookahead\nH = 2", C["LOOKAHEAD_TEACHER"], "Lookahead K=1", "teacher_k1", "Ours K1"),
    ("Lookahead\nH = 4", C["LOOKAHEAD_TEACHER"], "Lookahead K=3", "teacher_k3", "Ours K3"),
    ("Oracle", C["ORACLE"], "Best on observed draws", "outcome", "Ours outcome"),
    ("Web-\nArbiter", C["PRE_EXEC"], "WebArbiter formulation", None, None),
    ("Web-\nShepherd", C["PRE_EXEC"], "WebShepherd formulation", None, None),
    ("PRO-\nCUA", C["PRE_EXEC"], "PRO-CUA formulation", None, None)]
MARK = [("Teacher's pick, offline (held-out rollouts, 1,010 states)", "o", "hollow"),
        ("Head trained on the labels, offline (fixed five, 348 test states)", "s", "filled"),
        ("Same head, live (mean of 3 benchmarks, 1,045 tasks)", "D", "filled")]
DX = (-0.24, 0, 0.24)


def values(row):
    _, _, t, h, l = row
    head = STAGES["head_offline"].get(h) if h else None
    live = (STAGES.get("live", {}).get(l) or (LIVE[l]["minus_sft4b"] if l in LIVE else None)) if l else None
    return [TEACH[t], head, live]


fig, ax = plt.subplots(figsize=(9.2, 4.2))
ax.axhline(0, color=MUTED, lw=1, ls="--", zorder=0)
ax.text(4.55, -0.35, "one actor sample", ha="left", va="top", fontsize=7.5, color=MUTED)
for x, row in enumerate(ROWS):
    color = row[1]
    for (label, m, fill), dx, v in zip(MARK, DX, values(row)):
        if v is None:
            if row[3] is not None:   # a trainable label source whose head has not been run yet
                ax.text(x + dx, 0.4, "not\nrun", ha="center", va="bottom", fontsize=6, color=MUTED)
            continue
        d, lo, hi = v
        ax.errorbar(x + dx, d, yerr=[[d - lo], [hi - d]], fmt=m, ms=6.5, color=color, ecolor=C["BASELINE_DARK"], capsize=2,
                    lw=0.9, zorder=3, markeredgewidth=1.7,
                    markerfacecolor=C["ADM_STUDENT"] if fill == "filled" else "white",
                    markeredgecolor=student_outline(color) if fill == "filled" else color)
        ax.text(x + dx, hi + 0.25, f"{d:+.1f}".replace("-", "−"), ha="center", va="bottom", fontsize=6.3, color=INK)
ax.axvline(4.5, color=GRID, lw=1)
ax.text(6, 9.3, "prior judge formulations\n(teacher only)", ha="center", va="top", fontsize=7, color=MUTED)
ax.set_xticks(range(len(ROWS)), [r[0] for r in ROWS], fontsize=8); ax.set_xlim(-0.55, len(ROWS) - 0.45)
ax.set_ylim(-5, 10); ax.set_ylabel("Gain over one SFT 4B actor sample (pp)", fontsize=9)
ax.grid(axis="y", alpha=.25)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
handles = [Line2D([], [], marker=m, ls="", color=C["BASELINE_DARK"], ms=6.5, markeredgewidth=1.4,
                  markerfacecolor=C["ADM_STUDENT"] if f == "filled" else "white",
                  markeredgecolor=C["TRAIN"] if f == "filled" else C["BASELINE_DARK"]) for _, m, f in MARK]
ax.legend(handles, [l for l, _, _ in MARK], frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.45, -0.17), ncol=1)
fig.text(0.01, 0.01, "Bars: paired 95% intervals (bootstrap over tasks). Stages use different states and tasks: compare orderings "
         "within a stage, not magnitudes across stages.\nOne training seed per head. Oracle = best action on rollouts 1–3 (observed "
         "outcome); its head is trained on those labels. Rollouts numbered 1–8 as in the methods figure.", fontsize=6.5, color=MUTED)
fig.tight_layout(rect=(0, 0.05, 1, 1))
_paper_save(fig, HERE / "selection-label-stages-20261027.png", dpi=160); plt.close(fig)
