"""Paper figure for offline selector evaluation (October 21 ablation).

The outcomes and teacher picks are illustrative, preserved from the original
figure. Draws 0--2 supply outcome labels; teachers see only draw 0, and all
pickers are scored on held-out draws 3--7. Invalid draws are excluded from the
rate denominator. No experiment artifacts or API calls are needed to render.

Usage: python3 selection-offline-eval-20261022.py
Writes adjacent PNG (300 dpi), SVG (editable text), and PDF (embedded fonts).
"""
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/openwebrl-offline-eval-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

INK, MUTED, LINE = "#24272B", "#626972", "#C9CDD3"
PRE, POST, LOOK, OUTCOME, HELD = "#B84071", "#087F83", "#6250BE", "#AD750C", "#2164A5"
SUCCESS = "#31785A"
LABEL_BG, HELD_BG = "#FCF5E6", "#EDF3FA"
# Same 5 x 8 outcomes and picks as the original worked example.
OUTCOMES = (
    (1, 0, 1, 1, 0, 1, 1, 0),
    (0, 0, 1, 0, 0, 0, 1, 0),
    (1, 1, 1, 1, 0, 1, 1, 1),
    (0, 0, 0, 0, 0, 1, 0, 0),
    (0, 1, 0, 1, 0, 0, 1, 0),
)
PICKS = {"Pre-execution": 0, "Post-execution": 0, "Lookahead K = 3": 2,
         "Observed outcome": 2, "Oracle": 2}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7,
    "text.color": INK, "mathtext.fontset": "dejavusans",
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "savefig.facecolor": "white",
})
W, H = 7.0, 3.95
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set(xlim=(0, W), ylim=(0, H))
ax.axis("off")


def text(x, y, label, size=7, color=INK, weight="normal", ha="left", va="center", **kw):
    return ax.text(x, y, label, fontsize=size, color=color, weight=weight,
                   ha=ha, va=va, **kw)


def box(x, y, w, h, edge=LINE, fill="white", lw=0.65, radius=0.035):
    patch = FancyBboxPatch((x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=fill, edgecolor=edge, linewidth=lw, zorder=2)
    ax.add_patch(patch)
    return patch


def arrow(x0, y0, x1, y1, color=MUTED, lw=0.65):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
        mutation_scale=5, shrinkA=1, shrinkB=1, linewidth=lw, color=color))


def rule(x0, x1, y, color=LINE, lw=0.6):
    ax.plot([x0, x1], [y, y], color=color, linewidth=lw, zorder=1)


# (a) Each outcome dot summarizes a complete independently replayed rollout.
text(0.12, 3.74, "(a)  Branch and collect", size=8.5, weight="bold")
text(0.12, 3.51, r"Each draw: replay $s_t$ → action → frozen actor → judge", color=MUTED)
xs = [1.48 + 0.17 * d for d in range(8)]
ys = [2.77 - 0.32 * i for i in range(5)]
y_low, y_high = ys[-1] - 0.15, ys[0] + 0.15
ax.add_patch(Rectangle((xs[0] - 0.085, y_low), 3 * 0.17, y_high - y_low,
                       facecolor=LABEL_BG, edgecolor="none", zorder=0))
ax.add_patch(Rectangle((xs[3] - 0.085, y_low), 5 * 0.17, y_high - y_low,
                       facecolor=HELD_BG, edgecolor="none", zorder=0))
# The draw-0 column is the only continuation evidence available to teachers.
ax.add_patch(Rectangle((xs[0] - 0.074, y_low), 0.148, y_high - y_low,
                       facecolor="none", edgecolor=LOOK, linewidth=0.7, zorder=1))
for first, last, title, color in ((0, 2, "Label pool", OUTCOME), (3, 7, "Held out", HELD)):
    rule(xs[first] - 0.08, xs[last] + 0.08, 3.18, color, 1.1)
    text((xs[first] + xs[last]) / 2, 3.31, title, color=color, weight="bold", ha="center")
    for d in range(first, last + 1):
        text(xs[d], 3.01, str(d), color=MUTED, ha="center")
text(1.02, 3.01, "Action", color=MUTED, ha="center")
text(3.06, 3.31, "Label", color=OUTCOME, weight="bold", ha="center")
text(3.06, 3.01, "0–2", color=MUTED, ha="center")
text(3.58, 3.31, "Score", color=HELD, weight="bold", ha="center")
text(3.58, 3.01, "3–7", color=MUTED, ha="center")
mid = sum(ys) / len(ys)
box(0.13, mid - 0.23, 0.43, 0.46, edge=INK)
text(0.345, mid, r"$s_t$", size=9, ha="center")
text(0.345, mid - 0.33, "Task +", color=MUTED, ha="center")
text(0.345, mid - 0.46, "history", color=MUTED, ha="center")
label_rates = [sum(out[:3]) / 3 for out in OUTCOMES]
held_rates = [sum(out[3:]) / 5 for out in OUTCOMES]
for i, (y, outcomes) in enumerate(zip(ys, OUTCOMES)):
    arrow(0.57, mid, 0.80, y, color=LINE)
    box(0.81, y - 0.10, 0.41, 0.20)
    text(1.015, y, rf"$a_{i + 1}$", size=8, ha="center")
    arrow(1.23, y, 1.36, y, color=LINE)
    for d, success in enumerate(outcomes):
        ax.add_patch(Circle((xs[d], y), 0.028,
            facecolor=SUCCESS if success else "white",
            edgecolor=SUCCESS if success else "#88919C", linewidth=0.65))
    text(3.06, y, f"{sum(outcomes[:3])}/3", color=OUTCOME, ha="center",
         weight="bold" if label_rates[i] == max(label_rates) else "normal")
    text(3.58, y, f"{100 * held_rates[i]:.0f}%", color=HELD, weight="bold", ha="center")
ax.add_patch(Circle((0.17, 1.17), 0.025, facecolor=SUCCESS, edgecolor=SUCCESS, linewidth=0.6))
text(0.24, 1.17, "success", color=MUTED)
ax.add_patch(Circle((0.86, 1.17), 0.025, facecolor="white", edgecolor="#88919C", linewidth=0.6))
text(0.93, 1.17, "failure", color=MUTED)
text(1.58, 1.17, r"Outcome label: $a_3$ (best on 0–2)", color=OUTCOME)
text(0.12, 0.91, "Train a head: 1,380 training states · soft CE", weight="bold")
for x, w, label in ((0.13, 1.13, "Frozen SFT-4B\nfeatures"),
                     (1.49, 1.12, "Selection head\n22.6 M parameters"),
                     (2.84, 0.94, "Argmax action\nsame scoring")):
    box(x, 0.35, w, 0.38)
    text(x + w / 2, 0.54, label, ha="center", linespacing=1.35)
arrow(1.27, 0.54, 1.48, 0.54)
arrow(2.62, 0.54, 2.83, 0.54)

# (b) One shared timeline replaces forty miniature browser trajectories.
ax.plot([3.90, 3.90], [0.33, 3.82], color=LINE, linewidth=0.6)
text(4.03, 3.74, "(b)  Evidence used for labels", size=8.5, weight="bold")
text(4.03, 3.51, r"Common input: task, history, $s_t$, actions", color=MUTED)
future_x = [5.35 + 0.43 * k for k in range(4)]
for k, x in enumerate(future_x):
    text(x, 3.28, rf"$s_{{t+{k + 1}}}$", size=7, color=MUTED, ha="center")
for y, label, color, visible in ((3.06, "Pre-execution", PRE, 0),
                                  (2.81, "Post-execution", POST, 1),
                                  (2.56, "Lookahead K = 3", LOOK, 4)):
    text(4.03, y, label, color=color)
    if not visible:
        text((future_x[0] + future_x[-1]) / 2, y, "No future states", color=MUTED, ha="center")
        continue
    rule(future_x[0], future_x[-1], y, color=LINE)
    for k, x in enumerate(future_x):
        box(x - 0.08, y - 0.045, 0.16, 0.09,
            edge=color if k < visible else LINE,
            fill=color if k < visible else "white", radius=0.012)
text(4.03, 2.32, r"Draw 0 only; K = 1 ends at $s_{t+2}$.", color=MUTED)
text(4.03, 2.16, "Final answers are hidden from lookahead.", color=MUTED)

# (c) Every number below is derived from the matrix, not a measured result.
text(4.03, 1.92, "(c)  Score on held-out draws 3–7", size=8.5, weight="bold")
for x, title, align in ((4.03, "Picker", "left"), (6.04, "Pick", "center"), (6.83, "Score", "right")):
    text(x, 1.70, title, color=MUTED, ha=align)
rule(4.03, 6.86, 1.59)
rows = [("Uniform", MUTED, "all 5", sum(held_rates) / 5),
        ("SFT sample", MUTED, "all 5", sum(held_rates) / 5)]
colors = {"Pre-execution": PRE, "Post-execution": POST, "Lookahead K = 3": LOOK,
          "Observed outcome": OUTCOME, "Oracle": INK}
for label, pick in PICKS.items():
    rows.append((label, colors[label], rf"$a_{pick + 1}$", held_rates[pick]))
for i, (label, color, pick, score) in enumerate(rows):
    y = 1.45 - 0.17 * i
    text(4.03, y, label, color=color)
    text(6.04, y, pick, ha="center")
    text(6.83, y, f"{100 * score:.0f}%", color=HELD, weight="bold", ha="right")
rule(4.03, 6.86, 0.31)
text(0.12, 0.12, "Illustrative example · valid draws only. Test mean: 348 states from 86 unseen tasks.", color=MUTED)

for ext in ("png", "svg", "pdf"):
    path = Path(__file__).with_suffix(f".{ext}")
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.025)
    if ext == "svg":
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
plt.close(fig)
