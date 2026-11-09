#!/usr/bin/env python3
"""Five selector designs: layer stacks (what runs, what is reused, what is trained), attention masks, and properties.

Used in ARM_SELECTION_HEAD.md (Kev-inspired variants).
"""
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
from matplotlib.patches import FancyBboxPatch, Rectangle

INK, MUTED, SURFACE, LINE = C["INK"], C["MUTED"], C["BACKGROUND"], C["BORDER"]
REUSED, EXTRA, TRAINED, EXTERNAL, HEAD, PIOTR = tint("BASELINE_LIGHT", .60), tint("BASELINE_DARK", .65), C["ADM_22M"], C["BASELINE_LIGHT"], C["ADM_22M"], C["BASELINE_DARK"]
W = 23                                              # column width
X0 = 15                                             # first column's left edge (row labels to its left)

plt.rcParams["font.family"] = ["Albert Sans", "DejaVu Sans"]
fig, ax = plt.subplots(figsize=(14.5, 11.5))
fig.patch.set_facecolor(SURFACE)
ax.axis("off")


def stack(x, y0, y1, color, label=None, hatch=None, edge=C["BASELINE_DARK"], w=6):
    ax.add_patch(Rectangle((x, y0), w, y1 - y0, fc=color, ec=edge, lw=1.0, hatch=hatch))
    if label:
        ax.text(x + w / 2, (y0 + y1) / 2, label, rotation=90, ha="center", va="center", fontsize=7.2, color=INK)


def chip(x, y, w, text, color):
    ax.add_patch(FancyBboxPatch((x, y), w, 2.8, boxstyle="round,pad=0.2,rounding_size=0.7", fc=color, ec="none"))
    ax.text(x + w / 2, y + 1.4, text, ha="center", va="center", fontsize=7.4, color="white", weight="bold")


def arrow(x0, y0, x1, y1, color=C["BASELINE_DARK"]):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=color, lw=1.1, mutation_scale=9))


def grid(cx, y, allow, labels, color):
    """Attention mask: rows are queries, columns keys; filled = may attend."""
    n, s = len(labels), 2.1
    x = cx - n * s / 2
    for j, lab in enumerate(labels):
        ax.text(x + j * s + s / 2, y + 0.6, lab, ha="center", fontsize=6.8, color=MUTED)
        ax.text(x - 0.5, y - j * s - s / 2, lab, ha="right", va="center", fontsize=6.8, color=MUTED)
    for i in range(n):
        for j in range(n):
            ax.add_patch(FancyBboxPatch((x + j * s + 0.15, y - (i + 1) * s + 0.15), s - 0.3, s - 0.3,
                                        boxstyle="round,pad=0,rounding_size=0.3", lw=0,
                                        fc=C["ATTENTION_ALLOWED"] if allow(i, j) else C["ATTENTION_MASKED"]))


cols = [("Piotr SelectionARM", "baseline", PIOTR), ("Selection head", "ours: frozen actor + small head", C["ADM_22M"]),
        ("Joint top layers", "Kev-inspired, inside the actor", TRAINED), ("Kev-style pass", "Kev's design on the actor's weights", TRAINED),
        ("Kev", "external reference", PIOTR)]
xs = [X0 + i * (W + 2) for i in range(5)]
for x, (title, sub, color) in zip(xs, cols):
    ax.text(x + W / 2, 112, title, ha="center", fontsize=12, weight="bold", color=INK)
    ax.text(x + W / 2, 109.6, sub, ha="center", fontsize=8.2, color=MUTED)

top, mid, bot = 106, 93, 84            # layer stacks span bot..top; mid = layer 28 boundary
# Piotr: actor unchanged + separate 4B selector
x = xs[0]
stack(x + 1, bot, top, REUSED, "actor")
stack(x + 8, bot, top, EXTERNAL, "separate 4B")
chip(x + 15, 94, 7.5, "generate", PIOTR)
# Selection head
x = xs[1]
stack(x + 1, bot, top, REUSED, "actor's generation pass")
for frac, lab in ((0.5, "18"), (0.75, "27"), (1.0, "36")):
    yy = bot + frac * (top - bot) - (0.5 if frac == 1.0 else 0)
    arrow(x + 7, yy, x + 11.5, 96.4, HEAD); ax.text(x + 7.3, yy + 0.3, lab, fontsize=6.5, color=MUTED)
chip(x + 11.5, 95, 10.5, "small MLP head", HEAD)
# Joint top layers
x = xs[2]
stack(x + 1, bot, mid, REUSED, "gen. pass 1–28")
stack(x + 8, mid, top, EXTRA, "29–36 again", hatch="///", edge=TRAINED)
arrow(x + 7, mid - 0.6, x + 8, mid + 0.6)
ax.text(x + 14.6, top - 1.5, "LoRA", fontsize=7, color=TRAINED, weight="bold")
chip(x + 14.6, 97, 7.6, "pointer", TRAINED); chip(x + 14.6, 93.4, 7.6, "+ head", HEAD)
# Kev-style pass
x = xs[3]
stack(x + 1, bot, top, REUSED, "gen. pass (unused)")
stack(x + 8, bot, top, EXTRA, "2nd full pass 1–36", hatch="///", edge=TRAINED)
ax.text(x + 14.6, top - 1.5, "LoRA, all\nlayers", fontsize=7, color=TRAINED, weight="bold", va="top")
chip(x + 14.6, 94, 7.6, "pointer", TRAINED)
# Kev
x = xs[4]
stack(x + 1, bot, top, REUSED, "actor")
stack(x + 8, bot, top, EXTERNAL, "separate 0.8–27B")
chip(x + 15, 94, 7.5, "pointer", PIOTR)

# attention masks
ax.text(1, 76, "Interaction", fontsize=9, weight="bold", color=INK, va="center")
ax.text(1, 73.2, "rows query,\ncolumns key;\nfilled = may\nattend", fontsize=6.8, color=MUTED, va="top")
C3, C4 = ["c1", "c2", "c3"], ["c1", "c2", "c3", "D"]
causal = lambda i, j: j <= i
alone = lambda i, j: i == j
decide = lambda i, j: i == j or i == 3
masks = [(C3, causal), (C3, alone), (C4, decide), (C4, decide), (C4, causal)]
notes = ["one causal sequence:\nlater see earlier", "none: each\ncandidate alone", "<decide> reads all,\ntop 8 layers only",
         "<decide> reads all,\nin every layer", "later options see\nearlier + <decide>"]
for x, (labels, allow), note, (_, _, color) in zip(xs, masks, notes, cols):
    grid(x + W / 2 + 0.6, 80, allow, labels, color)
    ax.text(x + W / 2, 69.3, note, ha="center", va="top", fontsize=7.6, color=INK, linespacing=1.25)

rows = [
    ("Reads", ["screenshot + task +\n5 candidate texts", "actor's prompt + screenshot;\nits own candidate states",
               "same as generation;\ncandidates' states from it", "prompt + screenshot +\ncandidates, re-encoded", "text-only page JSON +\ncandidate texts"]),
    ("Order effects", ["yes: picks candidate 1\non 62/124 states", "none", "none: shared positions", "none: shared positions",
                       "yes in Kev-27B;\nwe shuffle per state"]),
    ("Readout", ["generates\n{\"selection\": N}", "pooled states\n→ MLP per candidate", "pooled head\n+ gated pointer",
                 "pointer only:\nq(<decide>) · k(end)", "pointer\n+ fitted temperature"]),
    ("Trained", ["full 4B selector", "head only\n(actor frozen)", "LoRA top 8 +\npointer + head", "LoRA all 36 +\npointer + <decide>",
                 "Kev's own training\n(general decisions)"]),
    ("Extra compute\nper step", ["full 4B pass, ~3.8k tokens;\n36% of system FLOPs", "≈ 0: states come from\ngeneration (<1%)",
                                "≈ 8/36 of a pass over\ncandidate tokens", "≈ 1 full 4B pass: LoRA on\nthe prompt blocks cache reuse",
                                "1 full pass of a\nseparate model"]),
    ("Status", ["Luna 73.2%; held-out +2.5 pp;\nOM2W 40.0%; 6.76 s/decision", "Luna 61.7%; held-out +2.0 pp;\nOM2W 37.7%; 6.07 s/decision",
                "pointer alone:\nLuna 57.5%", "Luna 66.1%; held-out +1.1 pp;\nOM2W 37.7%; 8.74 s/decision", "not run on\nthese states"]),
]
ax.plot([0.5, xs[-1] + W], [64.6, 64.6], color=LINE, lw=0.8)
y = 60
for label, cells in rows:
    ax.text(1, y, label, fontsize=9, weight="bold", color=INK, va="center")
    for x, cell in zip(xs, cells):
        ax.text(x + W / 2, y, cell, fontsize=7.7, color=INK, va="center", ha="center", linespacing=1.25)
    ax.plot([0.5, xs[-1] + W], [y - 4.2, y - 4.2], color=LINE, lw=0.8)
    y -= 8.6
for i, (c, t) in enumerate([(REUSED, "actor's own generation pass"), (EXTRA, "extra computation"),
                            (TRAINED, "trained (hatched = LoRA)"), (EXTERNAL, "separate model")]):
    ax.add_patch(Rectangle((X0 + i * 30, 2.2), 2.4, 1.8, fc=c, ec=C["BASELINE_DARK"], lw=0.8))
    ax.text(X0 + 3.4 + i * 30, 3.1, t, fontsize=8, color=MUTED, va="center")
ax.set_xlim(0, xs[-1] + W + 1); ax.set_ylim(1, 115)
fig.tight_layout()
out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
for ext in ("png", "svg"):
    fig.savefig(out / f"selection_designs_layers.{ext}", dpi=170, facecolor=SURFACE)
