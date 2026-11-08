"""Standalone paper figure: what each candidate's score can see, for the ADM head variants and the 4B decoder.

Left: CandidateSetHead (openwebrl/selection_head.py) on frozen actor features, interaction (joint / independent) x
scorer (two-layer transformer / depth-2 MLP). Transformer cells are attention connections; MLP cells are the MLP's
inputs, [c_i, s] independently and [c_i, s, c_i - mean of the others] jointly (the others enter only as a mean).
Right: the ADM-4B decoder, a LoRA on a separate OpenWebRL-4B-SFT copy that reads the selection prompt (task, history,
URL, screenshot) followed by the numbered candidates and generates {"selection": i} under JSON-schema decoding; its
causal attention lets each candidate read only the earlier ones, and the answer token reads everything.
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
HEAD, DECODER = "#168C9E", "#4B5563"
SHARED, SHARED_BG = "#737A83", "#F3F4F5"
ALLOWED_FILL, POOLED_FILL, MASKED_FILL = "#81CCD7", "#CDEBEF", "#F5F6F7"
K = 5
plt.rcParams.update({
    "font.family": "Albert Sans", "font.size": 7,
    "text.color": INK, "mathtext.fontset": "custom",
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "savefig.facecolor": "white",
})
W, H = 7.0, 3.62
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set(xlim=(0, W), ylim=(0, H))
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

box(0.10, 3.30, 6.85, 0.24, edge=SHARED, fill=SHARED_BG, lw=0.95)
text(3.525, 3.42, r"Shared: state $s$ · $K=5$ candidates $\{c_i\}$ sampled by the frozen SFT actor · one pick executed",
     color=SHARED, ha="center")

# --- left: ADM head on frozen actor features ---------------------------------------------------------------------
text(2.15, 3.10, "a)  ADM head on frozen actor features", size=8.5, color=HEAD, weight="bold", ha="center")
CELL = 0.125
COLS = {"Joint": 1.22, "Independent": 2.92}
for name, x0 in COLS.items():
    text(x0 + 3 * CELL, 2.88, name, size=7.5, color=HEAD if name == "Joint" else DECODER, weight="bold", ha="center")
ROW_TOPS = {"Transformer\n(2 layers)": 2.62, "MLP\n(depth 2)": 1.70}
for (row, top), n_rows in zip(ROW_TOPS.items(), (K + 1, K)):
    text(0.36, top - n_rows * CELL / 2, row, size=7.5, color=INK, weight="bold", ha="center", rotation=90,
         multialignment="center")
cand_color = lambda r: INK
for name, x0 in COLS.items():
    joint = name == "Joint"
    # transformer: attention over [s, c_1..c_K]; independent masks candidate-candidate links
    matrix(x0, ROW_TOPS["Transformer\n(2 layers)"], STATE + CANDS, STATE + CANDS, CELL,
           lambda r, c, j=joint: ALLOWED_FILL if j or r == c or (r > 0 and c == 0) else MASKED_FILL)
    # MLP: the inputs of candidate i's score; jointly the others enter only through their mean
    matrix(x0, ROW_TOPS["MLP\n(depth 2)"], CANDS, STATE + CANDS, CELL,
           lambda r, c, j=joint: ALLOWED_FILL if c == 0 or c == r + 1 else (POOLED_FILL if j else MASKED_FILL))
text(COLS["Joint"] + 3 * CELL, 0.94, r"MLP input $[c_i,\ s,\ c_i-\overline{c}_{-i}]$", color=MUTED, ha="center")
text(COLS["Independent"] + 3 * CELL, 0.94, r"MLP input $[c_i,\ s]$", color=MUTED, ha="center")
box(0.62, 0.40, 3.22, 0.36, edge=HEAD)
text(2.23, 0.635, r"logit $z_i$ per candidate · softmax over $K$ · argmax", color=HEAD, ha="center")
text(2.23, 0.505, "reads actor hidden states (layers 18/27/36) · 22.6 M trained", color=MUTED, ha="center")

# --- right: ADM-4B decoder ---------------------------------------------------------------------------------------
ax.plot([4.22, 4.22], [0.40, 3.18], color=LINE, linewidth=0.6)
text(5.62, 3.10, "b)  ADM-4B decoder", size=8.5, color=DECODER, weight="bold", ha="center")
text(5.62, 2.88, "LoRA on a separate OpenWebRL-4B-SFT copy", size=7, color=MUTED, ha="center")
DCELL, DX, DTOP = 0.155, 5.08, 2.47
TOKENS = [r"$p$"] + CANDS + [r"$y$"]
matrix(DX, DTOP, TOKENS, TOKENS, DCELL, lambda r, c: ALLOWED_FILL if c <= r else MASKED_FILL)
text(DX - 0.36, DTOP - len(TOKENS) * DCELL / 2, "Queries", color=MUTED, ha="center", rotation=90)
text(DX + len(TOKENS) * DCELL / 2, DTOP + 0.19, "Keys", color=MUTED, ha="center")
mid = DX + len(TOKENS) * DCELL / 2
text(mid, 1.21, r"$p$: task, history, URL, screenshot", color=MUTED, ha="center")
text(mid, 1.08, r"$c_i$: numbered candidate (action + reasoning)", color=MUTED, ha="center")
text(mid, 0.95, r"causal: $c_i$ reads only earlier candidates", color=MUTED, ha="center")
arrow(mid, 0.89, mid, 0.78, color=DECODER)
box(mid - 1.00, 0.40, 2.00, 0.36, edge=DECODER)
text(mid, 0.635, r'$y$ generates {"selection": $i$} (JSON schema)', color=DECODER, ha="center")
text(mid, 0.505, "separate 4B pass over prompt + candidates", color=MUTED, ha="center")

# Right-aligned legend; cells encode connectivity, not values.
legend_right = 6.95
renderer = fig.canvas.get_renderer()
for label, fill in reversed((("Allowed", ALLOWED_FILL), ("Mean of the others only", POOLED_FILL), ("Masked", MASKED_FILL))):
    width = renderer.get_text_width_height_descent(label, FontProperties(size=7), ismath=False)[0] / fig.dpi
    marker_x = legend_right - width - 0.08
    ax.add_patch(Rectangle((marker_x - 0.03, 0.09), 0.06, 0.06, facecolor=fill, edgecolor=LINE, linewidth=0.4))
    text(legend_right, 0.12, label, color=MUTED, ha="right")
    legend_right = marker_x - 0.18

for ext in ("png", "svg", "pdf"):
    path = Path(__file__).with_suffix(f".{ext}")
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.025)
    if ext == "svg":
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
plt.close(fig)
