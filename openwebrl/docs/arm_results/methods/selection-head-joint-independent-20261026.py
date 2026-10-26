"""Standalone paper figure: joint versus independent candidate scoring.

Matches CandidateSetHead in openwebrl/selection_head.py. The transformer mask
uses all K=5 candidates plus the state. Allowed connections use one consistent
color in both masks. No training, API calls or result payloads are needed.
Writes adjacent 300-dpi PNG, editable SVG and PDF with embedded TrueType fonts.
"""
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/openwebrl-head-architecture-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

INK, MUTED, LINE = "#24272B", "#626972", "#C9CDD3"
JOINT, INDEPENDENT = "#168C9E", "#4B5563"
SHARED, SHARED_BG = "#737A83", "#F3F4F5"
ALLOWED_FILL, MASKED_FILL = "#81CCD7", "#F5F6F7"
K = 5
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7,
    "text.color": INK, "mathtext.fontset": "dejavusans",
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "savefig.facecolor": "white",
})
W, H = 7.0, 2.88
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set(xlim=(0, W), ylim=(0, H))
ax.axis("off")


def text(x, y, label, size=7, color=INK, weight="normal", ha="left", **kw):
    return ax.text(x, y, label, fontsize=size, color=color, weight=weight,
                   ha=ha, va="center", **kw)


def box(x, y, w, h, edge=LINE, fill="white", lw=0.65):
    patch = FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0,rounding_size=0.035",
        facecolor=fill, edgecolor=edge, linewidth=lw, zorder=2)
    ax.add_patch(patch)
    return patch


def arrow(x0, y0, x1, y1, color=MUTED):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
        mutation_scale=5, shrinkA=1, shrinkB=1, linewidth=0.65, color=color))


def attention_mask(joint):
    """True means permitted: rows query columns, including the state row."""
    return [[joint or row == col or (row > 0 and col == 0)
             for col in range(K + 1)] for row in range(K + 1)]


# Feature extraction/projection is shared; only the scoring interaction changes.
box(0.12, 2.51, 6.85, 0.25, edge=SHARED, fill=SHARED_BG, lw=0.95)
text(3.545, 2.635, r"Shared: state $s$ · candidates $\{c_i\}_{i=1}^{K}$ · $K=5$",
     color=SHARED, ha="center")
text(1.81, 2.29, "a)  Joint", size=8.5, color=JOINT, weight="bold", ha="center")
text(5.16, 2.29, "b)  Independent", size=8.5, color=INDEPENDENT, weight="bold", ha="center")

CELL, MATRIX_TOP = 0.19, 1.80
masks = {"joint": attention_mask(True), "independent": attention_mask(False)}
for x0, name, color in ((1.24, "joint", JOINT), (4.59, "independent", INDEPENDENT)):
    center = x0 + (K + 1) * CELL / 2
    text(center, 2.05, "Keys", color=MUTED, ha="center")
    text(x0 - 0.37, MATRIX_TOP - 3 * CELL, "Queries", color=MUTED,
         ha="center", rotation=90)
    names = [r"$s$"] + [rf"$c_{{{i}}}$" for i in range(1, K + 1)]
    for row, label in enumerate(names):
        label_color = INDEPENDENT if row == 0 else color
        text(x0 - 0.055, MATRIX_TOP - (row + 0.5) * CELL,
             label, color=label_color, ha="right")
        text(x0 + (row + 0.5) * CELL, MATRIX_TOP + 0.105,
             label, color=label_color, ha="center")
        for col in range(K + 1):
            fill = ALLOWED_FILL if masks[name][row][col] else MASKED_FILL
            ax.add_patch(Rectangle((x0 + col * CELL, MATRIX_TOP - (row + 1) * CELL),
                                   CELL, CELL, facecolor=fill, edgecolor="white", linewidth=0.8))
    arrow(center, 0.635, center, 0.565, color=color)
    box(center - 0.48, 0.32, 0.96, 0.24, edge=color)
    text(center, 0.44, r"$[z_1,\ldots,z_K]$", size=8, color=color, ha="center")

# Right-aligned legend; matrix cells encode connectivity, not scalar values.
legend_right = 6.97
renderer = fig.canvas.get_renderer()
for label, fill in reversed((("Allowed", ALLOWED_FILL), ("Masked", MASKED_FILL))):
    width = renderer.get_text_width_height_descent(
        label, FontProperties(size=7), ismath=False)[0] / fig.dpi
    marker_x = legend_right - width - 0.08
    ax.add_patch(Rectangle((marker_x - 0.03, 0.09), 0.06, 0.06,
                           facecolor=fill, edgecolor=LINE, linewidth=0.4))
    text(legend_right, 0.12, label, color=MUTED, ha="right")
    legend_right = marker_x - 0.18

for ext in ("png", "svg", "pdf"):
    path = Path(__file__).with_suffix(f".{ext}")
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.025)
    if ext == "svg":
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
plt.close(fig)
