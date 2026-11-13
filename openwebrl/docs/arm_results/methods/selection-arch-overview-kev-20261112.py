"""ADM architecture with the Kev-style selector: frozen actor, four feature-head
variants, decoder, and Kev-style decision pass (wider than the one-column 20261107 figure).

The head reads projected actor features s/c_i. Transformer rows visualize the
original K=5 joint/independent attention masks; MLP rows show their input vectors.
The decoder reads prompt/candidate text and generates the candidate index. All
selectors execute an original actor proposal. The Kev-style pass (openwebrl/selection_kev.py)
packs prompt, candidates and a <decide> token into one LoRA pass; each candidate sees the
prompt and itself, <decide> sees everything, and a pointer q(<decide>)·k(c_i end) scores
candidates. No experiments or API calls.
"""
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/openwebrl-head-architecture-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                           if (p / "paper_plot_style.py").is_file())))
import paper_plot_style  # noqa: E402,F401
from paper_plot_palette import COLORS as C, tint  # noqa: E402

INK, MUTED, LINE = C["INK"], C["MUTED"], C["BORDER"]
HEAD, DECODER = C["ADM_22M"], C["ADM_4B"]
# Kev-style selector: its own orange (oracle family, darkened for text/lines) so it is not confused with the decoder.
KEV, KEV_FILL = "#C97A22", tint("ORACLE", .80)
ICONS = Path(__file__).resolve().parents[2] / "assets/icons"
FROZEN, TRAINED = (plt.imread(ICONS / f) for f in ("twemoji-snowflake-2744.png", "twemoji-fire-1f525.png"))
ALLOWED, MASKED = C["ATTENTION_ALLOWED"], C["ATTENTION_MASKED"]
K = 5
W, H = 4.55, 1.95
plt.rcParams.update({"font.family": "Albert Sans", "font.size": 6,
                     "text.color": INK, "svg.fonttype": "none", "pdf.fonttype": 42,
                     "savefig.facecolor": "white"})
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set(xlim=(0, W), ylim=(0, H))
ax.axis("off")


def text(x, y, label, size=6, color=INK, ha="left", **kw):
    return ax.text(x, y, label, fontsize=size, color=color, ha=ha, va="center", **kw)


def box(x, y, w, h, edge=LINE, fill="white", lw=.55):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=.025",
                      facecolor=fill, edgecolor=edge, linewidth=lw, zorder=2)
    ax.add_patch(p)
    return p


def icon(x, y, img, size=.085):
    """Twemoji icon centred at (x, y), `size` inches square (axes units are inches)."""
    ax.imshow(img, extent=(x-size/2, x+size/2, y-size/2, y+size/2), aspect="auto", zorder=7, interpolation="lanczos")


def arrow(x0, y0, x1, y1, color=HEAD, head=4):
    p = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>,head_length=.30,head_width=.18",
                       mutation_scale=head, shrinkA=0, shrinkB=0,
                       color=color, linewidth=.8, zorder=5)
    ax.add_patch(p)
    return p


def matrix(x0, top, labels, cell, mask, answer=False, row_labels=True, out=None):
    for i, label in enumerate(labels):
        color = C["STATE_INK"] if i == 0 else (out or DECODER) if answer and i == K + 1 else C["ACTION_INK"]
        text(x0 + (i + .5)*cell, top+.055, label, size=5.6, color=color, ha="center")
        if row_labels:
            text(x0-.025, top-(i+.5)*cell, label, size=5.6, color=color, ha="right")
        for j in range(len(labels)):
            ax.add_patch(Rectangle((x0+j*cell, top-(i+1)*cell), cell, cell,
                                   facecolor=ALLOWED if mask[i][j] else MASKED,
                                   edgecolor="white", linewidth=.45, zorder=3))


# Shared actor sampling; both outgoing arrows remain explicit at printed size.
box(.025, 1.735, .53, .19, edge=C["STATE_BORDER"], fill=tint("STATE", .60))
text(.29, 1.83, "Context", ha="center")
arrow(.565, 1.83, .665, 1.83, color=MUTED)
box(.675, 1.735, .80, .19, edge=INK)
text(1.045, 1.83, r"Actor $\theta$", size=6.0, ha="center")
icon(1.26, 1.83, FROZEN)
arrow(1.485, 1.83, 1.65, 1.83, color=HEAD)
box(1.66, 1.735, 2.86, .19, edge=C["ACTION_INK"], fill=tint("ACTION", .65))
text(3.09, 1.83, r"$K=5$ responses $c_i$: thought + action", size=5.8, color=C["ACTION_INK"], ha="center")
arrow(1.075, 1.725, 1.075, 1.655, color=HEAD, head=4)
arrow(2.80, 1.725, 2.80, 1.655, color=DECODER, head=4)
# Feature head: attention and MLP are alternative scorers, each joint/independent.
box(.025, .325, 2.10, 1.32, edge=LINE, lw=.5)
text(1.075, 1.565, "a) ADM-23M", size=6.8, color="black", ha="center")
icon(1.43, 1.565, TRAINED)
text(1.075, 1.425, r"Actor features $s, c_i$", size=5.8, color=MUTED, ha="center")

CELL, HEAD_TOP, TOP = .074, 1.16, 1.105
labels = [r"$s$"]+[rf"$c_{{{i}}}$" for i in range(1, K+1)]
joint_mask = [[True]*(K+1) for _ in range(K+1)]
independent_mask = [[r == c or (r > 0 and c == 0) for c in range(K+1)] for r in range(K+1)]
assert sum(map(sum, joint_mask)) == 36
assert sum(map(sum, independent_mask)) == 11
for name, x0, mask in (("Joint", .365, joint_mask), ("Independent", 1.355, independent_mask)):
    text(x0+3*CELL, 1.30, name, size=6, color=HEAD, ha="center")
    matrix(x0, HEAD_TOP, labels, CELL, mask)
    center = x0 + 3*CELL
    arrow(center, .705, center, .630, color=HEAD, head=4)
    text(center, .585, r"Linear $\rightarrow z_i$", size=5.7, color=HEAD, ha="center")
text(.105, .94, "Transformer", size=5.8, color=HEAD, ha="center", rotation=90)
text(1.07, .530, "or", size=5.5, color=MUTED, ha="center")
text(.105, .445, "MLP", size=5.8, color=HEAD, ha="center", rotation=90)
# cbar_-i is the mean of OTHER candidates, defined in the caption. Each MLP
# shares parameters across candidates. Brackets denote vector concatenation.
for center, inputs in ((.617, r"$[c_i,s,\Delta c_i]$"), (1.607, r"$[c_i,s]$")):
    text(center, .505, inputs, size=5.8, ha="center")
    arrow(center, .460, center, .410, color=HEAD, head=4)
    text(center, .365, r"$z_i$", size=5.7, color=HEAD, ha="center")
# Every alternative explicitly ends in z_i; one panel-level arrow suffices.
arrow(1.11, .323, 1.11, .276, color=HEAD, head=4)
text(1.11, .235, r"$i^*=\arg\max_i z_i$", size=6, color=HEAD, ha="center")

# Decoder uses text rather than the head's projected feature vectors.
text(2.80, 1.565, "b) ADM-4B", size=6.8, color="black", ha="center")
# Matching theta symbols encode shared initialization without a long arrow.
box(2.44, 1.355, .29, .12, edge=LINE)
text(2.545, 1.415, r"$\theta$", size=6.4, ha="center")
icon(2.655, 1.415, FROZEN, .075)
text(2.81, 1.415, "+", size=7, color=DECODER, ha="center")
box(2.90, 1.355, .35, .12, edge=DECODER, fill=tint("ADM_4B", .93))
text(3.025, 1.415, r"$\Delta\theta$", size=6.4, color=DECODER, ha="center")
icon(3.175, 1.415, TRAINED, .075)
DX, DCELL = 2.48, .109
TOKENS = [r"$p$"]+[rf"$c_{{{i}}}$" for i in range(1,K+1)]+[r"$y$"]
causal_mask = [[c <= r for c in range(K+2)] for r in range(K+2)]
assert sum(map(sum, causal_mask)) == 28
matrix(DX, TOP, TOKENS, DCELL, causal_mask, answer=True)
mid = DX + len(TOKENS)*DCELL/2
arrow(mid, .330, mid, .280, color=DECODER, head=4)
text(mid, .235, r"$y\rightarrow i^*$", size=6.3, color=DECODER, ha="center")

# Kev-style pass: the same weights theta plus a LoRA, re-encoding prompt and
# candidates in one packed sequence; a <decide> token points at a candidate.
KC = 4.025
arrow(KC, 1.725, KC, 1.655, color=KEV, head=4)
text(KC, 1.565, "c) Kev-style ADM", size=6.8, color="black", ha="center")
box(KC-.36, 1.355, .29, .12, edge=LINE)
text(KC-.255, 1.415, r"$\theta$", size=6.4, ha="center")
icon(KC-.145, 1.415, FROZEN, .075)
text(KC-.01, 1.415, "+", size=7, color=KEV, ha="center")
box(KC+.08, 1.355, .35, .12, edge=KEV, fill=KEV_FILL)
text(KC+.205, 1.415, r"$\Delta\theta$", size=6.4, color=KEV, ha="center")
icon(KC+.355, 1.415, TRAINED, .075)
# Kev-A (the reported variant): LoRA on every token. d is the learned <decide> token.
text(KC-.04, 1.29, r"$d = \langle$decide$\rangle$", size=5.4, color=KEV, ha="center")
icon(KC+.235, 1.29, TRAINED, .07)
KX = KC - 3.5*DCELL + .02
KTOK = [r"$p$"]+[rf"$c_{{{i}}}$" for i in range(1,K+1)]+[r"$d$"]
# p causal; each c_i sees p and itself; <decide> d sees p, every c_i and itself.
kev_mask = [[c == 0 or c == r or r == K+1 for c in range(K+2)] for r in range(K+2)]
assert sum(map(sum, kev_mask)) == 18
matrix(KX, TOP, KTOK, DCELL, kev_mask, answer=True, out=KEV)
kmid = KX + len(KTOK)*DCELL/2
arrow(kmid, .330, kmid, .280, color=KEV, head=4)
text(kmid, .235, r"$q(d)\!\cdot\!k(c_i)\rightarrow z_i\rightarrow i^*$", size=5.4, color=KEV, ha="center")

EXC = 2.40
# One original action executes, whichever alternative selector is used.
ax.plot([1.11, 1.11, EXC], [.185, .17, .17], color=HEAD, linewidth=.65)
ax.plot([mid, mid, EXC], [.185, .17, .17], color=DECODER, linewidth=.65)
ax.plot([kmid, kmid, mid], [.185, .17, .17], color=KEV, linewidth=.65)
arrow(EXC, .17, EXC, .137, color=C["ACTION_INK"], head=3)
box(EXC-.565, .015, 1.13, .12, edge=C["ACTION_INK"], fill=tint("ACTION", .65))
text(EXC, .075, r"Execute $a_t^{i^*}$", size=6, color=C["ACTION_INK"], ha="center")
# Define the joint MLP's candidate-relative feature without crowding the row.
text(.055, .160, r"$\Delta c_i = c_i - \bar c_{-i}$", size=5.5, color=MUTED)
text(.055, .065, r"$\bar c_{-i}$: mean of others", size=5.2, color=MUTED)
# Icon legend.
icon(.86, .075, FROZEN, .07); text(.91, .075, "Frozen", size=5.6, color=MUTED)
icon(1.27, .075, TRAINED, .07); text(1.32, .075, "Trained", size=5.6, color=MUTED)
# Compact legend, right aligned. Rows query columns in all matrices.
for x, fill, label in ((3.55, ALLOWED, "Allowed"), (4.03, MASKED, "Masked")):
    ax.add_patch(Rectangle((x, .052), .045, .045, facecolor=fill, edgecolor=LINE, linewidth=.3))
    text(x+.065, .075, label, size=5.6, color=MUTED)

fig.canvas.draw()
renderer = fig.canvas.get_renderer()
for label in ax.texts:
    bounds = label.get_window_extent(renderer)
    assert fig.bbox.contains(*bounds.p0) and fig.bbox.contains(*bounds.p1), label.get_text()
for ext in ("png", "svg", "pdf"):
    path = Path(__file__).with_suffix(f".{ext}")
    metadata = {"Date": None} if ext == "svg" else {"CreationDate": None, "ModDate": None} if ext == "pdf" else {}
    fig.savefig(path, dpi=300, metadata=metadata)
    if ext == "svg":
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines())+"\n")
assert b"/FontFile2" in Path(__file__).with_suffix(".pdf").read_bytes()
plt.close(fig)
