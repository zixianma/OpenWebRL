#!/usr/bin/env python3
"""Simplified inference / training figure for the selection head (ARM_SELECTION_HEAD.md)."""
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
FROZEN, LEARNED, DATA, NEUTRAL = (tint("BASELINE_LIGHT", .75), C["BASELINE_DARK"]), (tint("ADM_22M", .85), C["ADM_22M"]), (tint("LOOKAHEAD_TEACHER", .85), C["LOOKAHEAD_TEACHER"]), ("white", C["BORDER"])

plt.rcParams["font.family"] = ["Albert Sans", "DejaVu Sans"]
fig, ax = plt.subplots(figsize=(11, 4.2))
fig.patch.set_facecolor(SURFACE)
ax.set_xlim(0, 124); ax.set_ylim(0, 41.5); ax.axis("off")


def box(x, y, w, h, title, sub, colors, dashed=False):
    fill, edge = colors
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fill, ec=edge, lw=1.4,
                                ls="--" if dashed else "-"))
    ax.text(x + w / 2, y + h - 2.4, title, ha="center", va="top", fontsize=10, weight="bold", color=INK)
    ax.text(x + w / 2, y + h - 6.0, sub, ha="center", va="top", fontsize=8.3, color=MUTED, linespacing=1.35)


def arrow(x0, x1, y):
    ax.annotate("", (x1, y), (x0, y), arrowprops=dict(arrowstyle="-|>", color=C["BASELINE_DARK"], lw=1.4, mutation_scale=14))


# A: inference
y = 26
ax.text(1, y + 14, "A  Inference, one browser step", fontsize=11, weight="bold", color=FROZEN[1])
box(1, y, 22, 11.5, "Frozen SFT actor", "prefills the prompt once,\nsamples 5 candidates", FROZEN)
arrow(23.8, 27.2, y + 5.75)
box(28, y, 28, 11.5, "Hidden states, captured", "layers 18 / 27 / 36, during generation:\nprompt end; reasoning mean,\naction mean, end token per candidate", FROZEN, dashed=True)
arrow(56.8, 60.2, y + 5.75)
box(61, y, 24, 11.5, "Selection head", "standardize → small MLP\nper candidate → 5 scores", LEARNED)
arrow(85.8, 89.2, y + 5.75)
box(90, y, 18, 11.5, "Execute", "highest-scoring\ncandidate", (tint("ACTION", .65), C["ACTION_INK"]))
ax.text(110, y + 5.75, "no second model,\nno extra pass", fontsize=8, color=MUTED, va="center")

# B: training
y = 3
ax.text(1, y + 14, "B  Training (the actor is never updated)", fontsize=11, weight="bold", color=LEARNED[1])
box(1, y, 22, 11.5, "Teacher-labelled panels", "Piotr release: GPT-5.5 picks\non 2,692 training states", DATA)
arrow(23.8, 27.2, y + 5.75)
box(28, y, 28, 11.5, "Same features, once", "frozen actor encodes each prompt\nand its 5 candidates; cached\n(39,675 panels, ~2 GPU-hours)", FROZEN, dashed=True)
arrow(56.8, 60.2, y + 5.75)
box(61, y, 24, 11.5, "Train the head only", "soft cross-entropy on the\nteacher's choice (≈3 s/epoch)", LEARNED)
arrow(85.8, 89.2, y + 5.75)
box(90, y, 18, 11.5, "Evaluate", "held-out outcomes;\nOnline-Mind2Web", NEUTRAL)
fig.tight_layout()
out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
for ext in ("png", "svg"):
    fig.savefig(out / f"selection_head_method.{ext}", dpi=170, facecolor=SURFACE)
