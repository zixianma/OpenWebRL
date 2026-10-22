"""Selection-head variants of the 2026-10-21 ablation (openwebrl/selection_head.py): interaction (joint / independent) x
scorer (2-layer transformer / depth-2 MLP), and the training loss (soft cross-entropy / per-candidate BCE)."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

INK, MUTED, GRID = "#2b2a27", "#6b6962", "#c3c2b7"
STATE, CAND, ON, OFF = "#6250d6", "#2a78d6", "#2a78d6", "#f1f0ec"
plt.rcParams.update({"font.size": 10, "text.color": INK})
fig = plt.figure(figsize=(13, 6.2))


def box(ax, x, y, w, h, text, fc="#ffffff", ec=GRID, size=9.5, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.02", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, weight=weight)


def attention(ax, x0, y0, cell, joint):
    names = ["s", "c1", "c2", "c3"]
    for i, q in enumerate(names):          # rows: query token
        ax.text(x0 - 0.02, y0 - (i + 0.5) * cell, q, ha="right", va="center", fontsize=8.5, color=STATE if q == "s" else CAND)
        ax.text(x0 + (i + 0.5) * cell, y0 + 0.02, q, ha="center", va="bottom", fontsize=8.5, color=STATE if q == "s" else CAND)
        for j in range(4):
            allowed = joint or i == j or (i > 0 and j == 0)
            ax.add_patch(Rectangle((x0 + j * cell, y0 - (i + 1) * cell), cell, cell, fc=ON if allowed else OFF, ec="white", lw=1.5,
                                   alpha=0.85 if allowed else 1))
    ax.text(x0 + 2 * cell, y0 - 4 * cell - 0.035, ("full bidirectional attention, no position embeddings:\norder-free (no causal variant)" if joint else "candidate attends to state and itself;\nstate to itself (row attends to column)"), ha="center", va="top", fontsize=8, color=MUTED)


ax = fig.add_axes([0, 0, 0.66, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.text(0.02, 0.965, "Scorer × interaction", fontsize=12, weight="bold")
ax.text(0.02, 0.925, "Inputs: per candidate cᵢ, the actor's hidden states (layers 18/27/36: reasoning mean, action mean, end token); "
        "state s, the prompt's last token.", fontsize=8.5, color=MUTED)
ax.text(0.37, 0.87, "Joint: a candidate's score sees the others", ha="center", fontsize=10.5, weight="bold")
ax.text(0.80, 0.87, "Independent: each candidate alone", ha="center", fontsize=10.5, weight="bold")
ax.text(0.075, 0.66, "2-layer\ntransformer\nover\n[s, c1 … c5]", ha="center", va="center", fontsize=10, weight="bold")
ax.text(0.075, 0.24, "Depth-2\nMLP\n(width-\nmatched)", ha="center", va="center", fontsize=10, weight="bold")
attention(ax, 0.30, 0.82, 0.045, True)
attention(ax, 0.73, 0.82, 0.045, False)
for x in (0.17, 0.60):
    ax.plot([x, x + 0.4], [0.45, 0.45], color=GRID, lw=1)
box(ax, 0.19, 0.27, 0.36, 0.1, "MLP( [ cᵢ ,  s ,  cᵢ − mean of the other c ] )", size=10)
box(ax, 0.64, 0.27, 0.32, 0.1, "MLP( [ cᵢ ,  s ] )", size=10)
ax.text(0.37, 0.22, "the difference term is its only view of the others", ha="center", fontsize=8.5, color=MUTED)
ax.text(0.80, 0.22, "no other-candidate term", ha="center", fontsize=8.5, color=MUTED)
ax.text(0.37, 0.12, "→ one logit zᵢ per candidate (linear readout)", ha="center", fontsize=9, color=MUTED)
ax.text(0.80, 0.12, "→ one logit zᵢ per candidate", ha="center", fontsize=9, color=MUTED)
ax.text(0.02, 0.025, "Default head = joint transformer + soft CE (22.6 M parameters; MLPs width-matched to it). "
        "Pointer variant: a learned <decide> token scores q·k / √d against each candidate.", fontsize=8.5, color=MUTED)

# losses: worked example
lx = fig.add_axes([0.68, 0, 0.32, 1]); lx.set_xlim(0, 1); lx.set_ylim(0, 1); lx.axis("off")
lx.text(0.02, 0.965, "Loss: a worked example", fontsize=12, weight="bold")
lx.text(0.02, 0.925, "Teacher votes 3 : 1 for c1 over c2; both losses see the same logits z", fontsize=8.5, color=MUTED)
import math
t = [0.75, 0.25, 0, 0, 0]; z = [2.0, 1.0, 0.5, 0.0, -0.5]
e = [math.exp(v) for v in z]; p = [v / sum(e) for v in e]
sig = [1 / (1 + math.exp(-v)) for v in z]
bce = [-(ti * math.log(si) + (1 - ti) * math.log(1 - si)) for ti, si in zip(t, sig)]
ce = -sum(ti * math.log(pi) for ti, pi in zip(t, p) if ti)
xs = [0.30 + k * 0.14 for k in range(5)]
def row(y, label, vals, fmt="{:.2f}", color=INK, weight="normal"):
    lx.text(0.04, y, label, fontsize=9, va="center", color=color, weight=weight)
    for x, v in zip(xs, vals):
        lx.text(x, y, fmt.format(v) if isinstance(v, float) else v, ha="center", va="center", fontsize=9, color=color, weight=weight)
row(0.865, "", [f"c{k + 1}" for k in range(5)], color=CAND, weight="bold")
row(0.825, "target tᵢ", t)
row(0.785, "logit zᵢ", z, "{:+.1f}")
box(lx, 0.02, 0.43, 0.96, 0.31, "", ec=CAND)
lx.text(0.5, 0.71, "Soft cross-entropy (default)", ha="center", fontsize=10.5, weight="bold")
row(0.655, "softmax pᵢ", p)
lx.text(0.5, 0.60, "pᵢ sum to 1: candidates compete", ha="center", fontsize=8.5, color=MUTED)
lx.text(0.5, 0.505, "loss = −Σᵢ tᵢ log pᵢ\n= −(0.75 log %.2f + 0.25 log %.2f) = %.2f" % (p[0], p[1], ce), ha="center", va="center", fontsize=9.5)
box(lx, 0.02, 0.05, 0.96, 0.34, "", ec=GRID)
lx.text(0.5, 0.36, "Per-candidate BCE", ha="center", fontsize=10.5, weight="bold")
row(0.305, "σ(zᵢ)", sig)
row(0.255, "BCE(σ(zᵢ), tᵢ)", bce)
lx.text(0.5, 0.20, "σ values sum to %.2f: each is fit alone" % sum(sig), ha="center", fontsize=8.5, color=MUTED)
lx.text(0.5, 0.11, "loss = mean over candidates = %.2f\n(c3–c5 are pushed toward 0 one by one)" % (sum(bce) / 5), ha="center", va="center", fontsize=9.5)
fig.savefig(Path(__file__).with_suffix(".png"), dpi=160, bbox_inches="tight", facecolor="white")
