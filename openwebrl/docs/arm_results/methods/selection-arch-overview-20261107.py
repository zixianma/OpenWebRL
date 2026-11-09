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
SHARED, SHARED_BG = "#737A83", "white"
ALLOWED_FILL, MASKED_FILL = "#81CCD7", "#F5F6F7"
K = 5
plt.rcParams.update({
    "font.family": "Albert Sans", "font.size": 7,
    "text.color": INK, "mathtext.fontset": "custom",
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "savefig.facecolor": "white",
})
W, H = 6.25, 3.95
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set(xlim=(0, W), ylim=(-0.42, 3.53))
ax.axis("off")


def text(x, y, label, size=7, color=INK, weight="normal", ha="left", **kw):
    return ax.text(x, y, label, fontsize=size, color=color, weight=weight, ha=ha, va="center", **kw)


def box(x, y, w, h, edge=LINE, fill="white", lw=0.65):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.035",
                                facecolor=fill, edgecolor=edge, linewidth=lw, zorder=2))


def arrow(x0, y0, x1, y1, color=MUTED):
    patch = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=6,
                            shrinkA=0, shrinkB=0, linewidth=0.8, color=color, zorder=3)
    ax.add_patch(patch)
    return patch


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


# One actor proposes actions for either selector. Head tokens s/c_i are projected
# actor features, whereas the separate decoder reads prompt/action content.
box(.08, 3.21, 1.02, .28, edge=SHARED, fill=SHARED_BG)
text(.59, 3.35, "Context", color=SHARED, ha="center")
arrow(1.11, 3.35, 1.35, 3.35)
box(1.36, 3.21, 1.32, .28, edge=INK, fill=SHARED_BG, lw=.95)
text(2.02, 3.35, "Frozen actor", size=8, ha="center")
arrow(2.69, 3.35, 4.01, 3.35)
box(4.02, 3.21, 2.17, .28, edge=SHARED, fill=SHARED_BG)
text(5.105, 3.35, r"$K=5$ candidates: reasoning + $a_t^i$", color=SHARED, ha="center")
arrow(5.105, 3.20, 5.105, 3.10, color=DECODER)
text(5.105, 2.99, r"Prompt $p$ + candidates", color=DECODER, ha="center")
arrow(5.105, 2.90, 5.105, 2.80, color=DECODER)

# --- left: ADM head on frozen actor features ---------------------------------------------------------------------
actor_head_arrow = arrow(2.02, 3.20, 2.02, 3.01)
box(.50, 2.58, 3.28, .42, edge=SHARED, fill=SHARED_BG)
text(2.14, 2.91, "a)  ADM head", size=8,
     weight="medium", ha="center")
text(1.23, 2.73, r"$s$: context feature" + "\nlast prompt token", size=6.5,
     ha="center", linespacing=1.05)
text(2.97, 2.73, r"$c_i$: candidate feature" + "\nreasoning · action · end", size=6.5,
     ha="center", linespacing=1.05)
# Short shared-input connectors replace the long per-input routing buses.
for center in (1.285, 2.875):
    ax.plot([center, center], [2.57, 2.49], color=LINE, linewidth=.65)
CELL = .115
COLS = {"Joint": .94, "Independent": 2.53}
for name, x0 in COLS.items():
    text(x0 + 3 * CELL, 2.41, name, size=7.5, color=HEAD if name == "Joint" else DECODER, ha="center")
# Neutral row enclosures separate attention from MLP computation.
for y, height, fill in ((1.40, .92, "white"), (.33, .98, "white")):
    ax.add_patch(FancyBboxPatch((.06, y), 3.75, height,
        boxstyle="round,pad=0,rounding_size=.035", edgecolor=LINE,
        facecolor=fill, linewidth=.55, zorder=.5))
TTOP = 2.16
text(.22, 1.86, "Transformer", size=7.5, ha="center", rotation=90)
text(.22, .82, "MLP", size=7.5, ha="center", rotation=90)
for name, x0 in COLS.items():
    joint = name == "Joint"
    # transformer: attention over [s, c_1..c_K]; independent masks candidate-candidate links
    matrix(x0, TTOP, STATE + CANDS, STATE + CANDS, CELL,
           lambda r, c, j=joint: ALLOWED_FILL if j or r == c or (r > 0 and c == 0) else MASKED_FILL)

# MLP: show computation rather than token connectivity. Each column uses the
# same MLP weights for every candidate. The joint difference excludes c_i from
# the pooled mean and concatenates c_i and s as separate inputs as well.
box(.535, 1.05, 1.50, .18, edge=SHARED, fill=SHARED_BG)
text(1.285, 1.14, r"$\bar c_{-i}=\mathrm{mean}_{j\ne i}\,c_j$", size=6.5, ha="center")
arrow(1.285, 1.04, 1.285, .94)
for center, inputs in ((1.285, r"$[c_i, s, c_i-\bar c_{-i}]$"),
                       (2.875, r"$[c_i, s]$")):
    box(center - .65, .73, 1.30, .20, edge=SHARED)
    text(center, .83, inputs, ha="center")
    arrow(center, .72, center, .59)
    box(center - .50, .40, 1.00, .18, edge=SHARED)
    text(center, .49, r"MLP $\rightarrow z_i$", ha="center")
ax.plot([1.285, 1.285, 2.875, 2.875], [.39, .32, .32, .39], color=SHARED, linewidth=.65)
ax.plot([2.02, 2.02], [.32, .24], color=SHARED, linewidth=.65)
box(1.27, .02, 1.50, .22, edge=HEAD)
text(2.02, .13, r"$i^*=\arg\max_i z_i$", color=HEAD, ha="center")

# --- right: ADM-4B decoder ---------------------------------------------------------------------------------------
ax.plot([3.96, 3.96], [.02, 2.79], color=LINE, linewidth=.6)
text(5.105, 2.72, "b)  ADM-4B decoder", size=8.5, color=DECODER, weight="medium", ha="center")
DCELL, DX, DTOP = .17, 4.51, 2.16
# Label each candidate-text block by its proposed action. The decoder receives
# full candidate reasoning/action text, never the head's projected c_i vectors.
TOKENS = [r"$p$"] + [rf"$a_t^{{{i}}}$" for i in range(1, K + 1)] + [r"$y$"]
matrix(DX, DTOP, TOKENS, TOKENS, DCELL, lambda r, c: ALLOWED_FILL if c <= r else MASKED_FILL)
mid = DX + len(TOKENS) * DCELL / 2
arrow(mid, DTOP - len(TOKENS) * DCELL - .04, mid, .25, color=DECODER)
box(mid - .70, .02, 1.40, .22, edge=DECODER)
text(mid, .13, r"$y\rightarrow i^*$", color=DECODER, ha="center")

# Both alternatives execute one of the actor's original candidate actions.
arrow(2.02, .01, 3.02, -.10, color=HEAD)
arrow(mid, .01, 3.82, -.10, color=DECODER)
box(2.62, -.33, 1.60, .22, edge=SHARED, fill=SHARED_BG)
text(3.42, -.22, r"Execute $a_t^{i^*}$", ha="center")

# Right-aligned legend; cells encode connectivity, not values.
legend_right = 6.20
renderer = fig.canvas.get_renderer()
for label, fill in reversed((("Allowed", ALLOWED_FILL), ("Masked", MASKED_FILL))):
    width = renderer.get_text_width_height_descent(label, FontProperties(size=7), ismath=False)[0] / fig.dpi
    marker_x = legend_right - width - 0.08
    ax.add_patch(Rectangle((marker_x - 0.03, -.38), .06, .06, facecolor=fill, edgecolor=LINE, linewidth=.4))
    text(legend_right, -.35, label, color=MUTED, ha="right")
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
