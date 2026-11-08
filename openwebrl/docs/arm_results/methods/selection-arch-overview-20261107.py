"""Standalone paper figure: what each candidate's score can see, for the ADM head variants and the 4B decoder.

Left: CandidateSetHead (openwebrl/selection_head.py) on frozen actor features, interaction (joint / independent) x
scorer (two-layer transformer / depth-2 MLP). Transformer cells are attention connections. MLP flows concatenate
[c_i, s] independently and [c_i, s, c_i - mean of the others] jointly (the others enter only as a mean).
Right: the ADM-4B decoder, a LoRA on a separate OpenWebRL-4B-SFT copy that reads the selection prompt (task, history,
URL, screenshot) followed by the numbered candidates and generates {"selection": i} under JSON-schema decoding; its
causal attention lets each candidate read only the earlier ones, and the answer token reads everything.
The actor samples all five actions. Its encoded features feed the head, while the selection prompt and candidate text feed
the separate decoder. Either selector returns an index into those same actor proposals; only that action executes.
No training, API calls or result payloads are needed. Writes adjacent 300-dpi PNG, editable SVG and PDF.
"""
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/openwebrl-head-architecture-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# Shared paper typography, including bundled Albert Sans fonts.
import sys as _paper_sys
from pathlib import Path as _PaperPath
_paper_sys.path.insert(0, str(next(p for p in _PaperPath(__file__).resolve().parents
                                if (p / "paper_plot_style.py").is_file())))
import paper_plot_style as _paper_style  # noqa: F401

from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

INK, MUTED, LINE = "#24272B", "#626972", "#C9CDD3"
HEAD, DECODER = INK, "#4B5563"
SHARED, SHARED_BG = "#737A83", "#F3F4F5"
ALLOWED_FILL, MASKED_FILL = "#81CCD7", "#F5F6F7"
K = 5
plt.rcParams.update({
    "font.family": "Albert Sans", "font.size": 7,
    "text.color": INK, "mathtext.fontset": "custom",
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "savefig.facecolor": "white",
})
W, H = 7.0, 4.55
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set(xlim=(0, W), ylim=(-0.40, 4.15))
ax.axis("off")


def text(x, y, label, size=7, color=INK, weight="normal", ha="left", **kw):
    return ax.text(x, y, label, fontsize=size, color=color, weight=weight, ha=ha, va="center", **kw)


def box(x, y, w, h, edge=LINE, fill="white", lw=0.65):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.035",
                                facecolor=fill, edgecolor=edge, linewidth=lw, zorder=2))


def arrow(x0, y0, x1, y1, color=MUTED):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=5,
                                 shrinkA=1, shrinkB=1, linewidth=0.65, color=color))


def matrix(x0, top, rows, cols, cell, fill_of, row_color=lambda r: INK, col_color=lambda c: INK):
    """rows/cols: labels; fill_of(r, c) -> fill colour. Rows query (or score) columns."""
    for c, label in enumerate(cols):
        text(x0 + (c + 0.5) * cell, top + 0.075, label, color=col_color(c), ha="center")
    for r, label in enumerate(rows):
        text(x0 - 0.04, top - (r + 0.5) * cell, label, color=row_color(r), ha="right")
        for c in range(len(cols)):
            ax.add_patch(Rectangle((x0 + c * cell, top - (r + 1) * cell), cell, cell,
                                   facecolor=fill_of(r, c), edgecolor="white", linewidth=0.8))


STATE = [r"$s$"]
CANDS = [rf"$c_{{{i}}}$" for i in range(1, K + 1)]


def axes_labels(x0, top, n_rows, n_cols, cell, row_name, col_name, col_offset=.20):
    text(x0 + n_cols * cell / 2, top + col_offset, col_name, color=MUTED, ha="center")
    text(x0 - 0.30, top - n_rows * cell / 2, row_name, color=MUTED, ha="center", rotation=90)


# One actor proposes actions for either selector. Head tokens s/c_i are projected
# actor features, whereas the separate decoder reads prompt/action content.
box(0.10, 3.77, 1.10, 0.34, edge=SHARED, fill=SHARED_BG)
text(0.65, 3.94, "Context\n(task · history · image)", size=6.5,
     color=SHARED, ha="center", linespacing=1.05)
arrow(1.21, 3.94, 1.49, 3.94)
box(1.50, 3.77, 1.50, 0.34, edge=INK, fill=SHARED_BG, lw=0.95)
text(2.25, 3.94, "Frozen actor", size=8, weight="bold", ha="center")
arrow(3.01, 3.94, 4.51, 3.94)
text(3.76, 4.055, "Sample", size=6.5, color=MUTED, ha="center")
box(4.52, 3.77, 2.33, 0.34, edge=SHARED, fill=SHARED_BG)
text(5.685, 3.94, r"$K=5$ candidates: reasoning + action $a_t^i$", color=SHARED, ha="center")
arrow(5.685, 3.76, 5.685, 3.59, color=DECODER)
text(5.685, 3.47, r"Selection prompt $p$ + candidate text", color=DECODER, ha="center")
arrow(5.685, 3.35, 5.685, 3.20, color=DECODER)

# --- left: ADM head on frozen actor features ---------------------------------------------------------------------
arrow(2.25, 3.76, 2.25, 3.71)
box(0.55, 3.22, 3.55, 0.48, edge=SHARED, fill=SHARED_BG)
text(2.25, 3.60, "a)  ADM head · projected actor features", size=8,
     weight="bold", ha="center")
text(1.35, 3.37, r"$s$: context feature" + "\nlast prompt token", size=6.5,
     ha="center", linespacing=1.05)
text(3.05, 3.37, r"$c_i$: candidate feature" + "\nreasoning · action · end", size=6.5,
     ha="center", linespacing=1.05)
# The same state/candidate features feed both scoring architectures in each
# column. Route the buses outside the matrices, clear of their row labels.
for bus_x, center in ((0.65, 1.675), (2.55, 3.375)):
    ax.plot([bus_x, bus_x], [3.21, 1.40], color=SHARED, linewidth=.65, zorder=1)
    arrow(bus_x, 2.82, center, 2.75)
arrow(.65, 1.56, .75, 1.56)
for bus_x, inputs in ((.65, (1.09, 1.48)), (2.55, (3.10, 3.65))):
    ax.plot([bus_x, inputs[-1]], [1.40, 1.40], color=SHARED, linewidth=.65, zorder=1)
    for input_x in inputs:
        arrow(input_x, 1.40, input_x, 1.35)
CELL = 0.125
COLS = {"Joint": 1.30, "Independent": 3.00}
for name, x0 in COLS.items():
    text(x0 + 3 * CELL, 2.92, name, size=7.5, color=HEAD if name == "Joint" else DECODER, weight="bold", ha="center")
TTOP = 2.48
text(0.28, TTOP - (K + 1) * CELL / 2, "Transformer", size=7.5, weight="bold", ha="center", rotation=90)
text(0.28, 1.12, "MLP", size=7.5, weight="bold", ha="center", rotation=90)
for name, x0 in COLS.items():
    joint = name == "Joint"
    # transformer: attention over [s, c_1..c_K]; independent masks candidate-candidate links
    matrix(x0, TTOP, STATE + CANDS, STATE + CANDS, CELL,
           lambda r, c, j=joint: ALLOWED_FILL if j or r == c or (r > 0 and c == 0) else MASKED_FILL)
    axes_labels(x0, TTOP, K + 1, K + 1, CELL, "Queries", "Keys")

# MLP: show computation rather than token connectivity. Each column uses the
# same MLP weights for every candidate. The joint difference excludes c_i from
# the pooled mean and concatenates c_i and s as separate inputs as well.
text(1.00, 1.56, r"$\{c_j\}_{j\ne i}$", size=6.5, ha="center")
arrow(1.27, 1.56, 1.47, 1.56)
box(1.48, 1.47, .48, .18, edge=SHARED, fill=SHARED_BG)
text(1.72, 1.56, "Mean", size=6.5, ha="center")
arrow(1.97, 1.56, 2.10, 1.56)
text(2.24, 1.56, r"$\bar c_{-i}$", ha="center")
arrow(2.24, 1.49, 2.12, 1.36)
for center, inputs in (
        (1.675, ((1.09, r"$c_i$"), (1.48, r"$s$"), (2.12, r"$c_i-\bar c_{-i}$"))),
        (3.375, ((3.10, r"$c_i$"), (3.65, r"$s$")))):
    for input_x, label in inputs:
        text(input_x, 1.28, label, ha="center")
        arrow(input_x, 1.22, input_x, 1.19)
    box(center - .65, 1.00, 1.30, .18, edge=SHARED)
    text(center, 1.09, "Concat", size=6.5, ha="center")
    arrow(center, .99, center, .89)
    box(center - .50, .70, 1.00, .18, edge=SHARED)
    text(center, .79, r"Shared MLP $\rightarrow z_i$", size=6.5, ha="center")
arrow(1.675, .69, 2.03, .63)
arrow(3.375, .69, 2.55, .63)
box(1.45, 0.36, 1.60, 0.26, edge=HEAD)
text(2.25, 0.49, r"$z_i \rightarrow i^*=\arg\max_i z_i$", color=HEAD, ha="center")

# --- right: ADM-4B decoder ---------------------------------------------------------------------------------------
ax.plot([4.30, 4.30], [0.36, 3.18], color=LINE, linewidth=0.6)
text(5.68, 3.08, "b)  ADM-4B decoder", size=8.5, color=DECODER, weight="bold", ha="center")
text(5.68, 2.92, "separate 4B (LoRA)", size=7.5, color=DECODER, weight="bold", ha="center")
DCELL, DX, DTOP = 0.19, 5.02, 2.48
# Label each candidate-text block by its proposed action. The decoder receives
# full candidate reasoning/action text, never the head's projected c_i vectors.
TOKENS = [r"$p$"] + [rf"$a_t^{{{i}}}$" for i in range(1, K + 1)] + [r"$y$"]
matrix(DX, DTOP, TOKENS, TOKENS, DCELL, lambda r, c: ALLOWED_FILL if c <= r else MASKED_FILL)
axes_labels(DX, DTOP, len(TOKENS), len(TOKENS), DCELL, "Queries", "Keys", col_offset=.27)
mid = DX + len(TOKENS) * DCELL / 2
arrow(mid, DTOP - len(TOKENS) * DCELL - 0.04, mid, 0.66, color=DECODER)
box(mid - 0.80, 0.36, 1.60, 0.26, edge=DECODER)
text(mid, 0.49, r'$y \rightarrow$ {"selection": $i^*$}', color=DECODER, ha="center")

# Both alternatives execute one of the actor's original candidate actions.
arrow(2.25, 0.35, 3.15, -0.005, color=HEAD)
arrow(mid, 0.35, 4.15, -0.005, color=DECODER)
box(2.65, -0.28, 2.00, 0.26, edge=SHARED, fill=SHARED_BG)
text(3.65, -0.15, r"Execute actor action $a_t^{i^*}$", ha="center")

# Right-aligned legend; cells encode connectivity, not values.
legend_right = 6.95
renderer = fig.canvas.get_renderer()
for label, fill in reversed((("Allowed", ALLOWED_FILL), ("Masked", MASKED_FILL))):
    width = renderer.get_text_width_height_descent(label, FontProperties(size=7), ismath=False)[0] / fig.dpi
    marker_x = legend_right - width - 0.08
    ax.add_patch(Rectangle((marker_x - 0.03, -0.37), 0.06, 0.06, facecolor=fill, edgecolor=LINE, linewidth=0.4))
    text(legend_right, -0.34, label, color=MUTED, ha="right")
    legend_right = marker_x - 0.18

fig.canvas.draw()
renderer = fig.canvas.get_renderer()
for label in ax.texts:
    bounds = label.get_window_extent(renderer)
    assert fig.bbox.contains(*bounds.p0) and fig.bbox.contains(*bounds.p1), label.get_text()
for ext in ("png", "svg", "pdf"):
    path = Path(__file__).with_suffix(f".{ext}")
    metadata = {"Date": None} if ext == "svg" else {"CreationDate": None, "ModDate": None} if ext == "pdf" else {}
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.025, metadata=metadata)
    if ext == "svg":
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
assert b"/FontFile2" in Path(__file__).with_suffix(".pdf").read_bytes()
plt.close(fig)
