#!/usr/bin/env python3
"""Overview figure for ARM_SELECTION_HEAD.md: Piotr SelectionARM versus the selection head that reuses the actor's KV cache."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# Shared paper typography, including bundled Albert Sans fonts.
import sys as _paper_sys
from pathlib import Path as _PaperPath
_paper_docs = next(p / "openwebrl/docs" for p in _PaperPath(__file__).resolve().parents
                   if (p / "openwebrl/docs/paper_plot_style.py").is_file())
_paper_sys.path.insert(0, str(_paper_docs))
import paper_plot_style as _paper_style
from paper_plot_palette import COLORS as C, tint
from matplotlib.patches import FancyBboxPatch

INK, MUTED, SURFACE = C["INK"], C["MUTED"], C["BACKGROUND"]
ACTOR, EXTRA, CHEAP = (tint("BASELINE_LIGHT", .75), C["BASELINE_DARK"]), (tint("ADM_4B", .85), C["ADM_4B"]), (tint("ADM_22M", .85), C["ADM_22M"])

fig, ax = plt.subplots(figsize=(11, 3.5))
fig.patch.set_facecolor(SURFACE)
ax.set_xlim(0, 110); ax.set_ylim(0, 36); ax.axis("off")


def box(x, y, w, h, title, sub, colors):
    fill, edge = colors
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fill, ec=edge, lw=1.4))
    ax.text(x + w / 2, y + h - 2.6, title, ha="center", va="top", fontsize=10.5, weight="bold", color=INK)
    ax.text(x + w / 2, y + h - 6.4, sub, ha="center", va="top", fontsize=8.6, color=MUTED, linespacing=1.35)


def arrow(x0, x1, y):
    ax.annotate("", (x1, y), (x0, y), arrowprops=dict(arrowstyle="-|>", color=C["BASELINE_DARK"], lw=1.4, mutation_scale=14))


rows = [
    (21, "Piotr SelectionARM", EXTRA, "Separate 4B selector",
     "re-reads the screenshot and\nall five candidates, then\nwrites {\"selection\": k}",
     "selector adds ~58k tokens per episode\n= 36% of system compute; a 2nd model to serve"),
    (2, "Selection head (this work)", CHEAP, "Selection head",
     "small MLP on hidden states\ncaptured while the actor\ngenerates; no second pass",
     "selector adds ~375 tokens per episode\n(< 1% of compute); one model to serve"),
]
for y, label, colors, title, sub, cost in rows:
    ax.text(1, y + 13.2, label, fontsize=11, weight="bold", color=colors[1])
    box(1, y, 21, 11, "Browser state", "task, history,\nscreenshot", (tint("STATE", .60), C["STATE_BORDER"]))
    arrow(22.8, 26.2, y + 5.5)
    box(27, y, 23, 11, "SFT actor (4B)", "samples 5 candidate\nactions (shared prefix)", ACTOR)
    arrow(50.8, 54.2, y + 5.5)
    box(55, y, 25, 11, title, sub, colors)
    arrow(80.8, 84.2, y + 5.5)
    box(85, y, 13, 11, "Execute", "highest-\nscoring action", (tint("ACTION", .65), C["ACTION_INK"]))
    ax.text(100, y + 5.5, cost, fontsize=7.8, color=MUTED, va="center", ha="left")
ax.set_xlim(0, 128)
fig.tight_layout()
out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
for ext in ("png", "svg"):
    fig.savefig(out / f"selection_head_overview.{ext}", dpi=170, facecolor=SURFACE)
