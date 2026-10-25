"""Paper figure for offline selector evaluation (October 21 ablation).

The outcomes and teacher picks are illustrative, preserved from the original
figure. Rollouts 0--2 supply outcome labels; teachers see only rollout 0, and all
pickers are scored on held-out rollouts 3--7. Invalid rollouts are excluded from the
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
PRE, POST, LOOK, OUTCOME, HELD = "#4B5563", "#008AC4", "#6250BE", "#AD750C", "#2164A5"
SUCCESS, FAILURE = "#31785A", "#C53A36"
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
W, H = 7.0, 4.92
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



def state_card(x, y, step, color, width=0.40, height=0.21, candidate="i"):
    # State symbols stand for the saved screenshot/observation and tool feedback.
    box(x - width / 2, y - height / 2, width, height, edge=color, fill="white")
    ax.add_patch(Rectangle((x - width / 2, y + height * 0.29), width, height * 0.21,
                           facecolor=color, edgecolor="none", alpha=0.15, zorder=2))
    text(x, y - 0.006, rf"$s_{{t+{step}}}^{{{candidate}}}$", ha="center", color=color)


def percent(rate):
    """Compact display, rounded to at most one decimal; calculations stay exact."""
    return f"{100 * rate:.1f}".rstrip("0").rstrip(".") + "%"


label_rates = [sum(out[:3]) / 3 for out in OUTCOMES]
held_rates = [sum(out[3:]) / 5 for out in OUTCOMES]

# (a) The exact original 5 x 8 outcomes; labels never use the blue columns.
text(0.12, 4.71, "(a)  Eight rollouts per action", size=8.5, weight="bold")
xs = [3.76 + 0.22 * d for d in range(8)]
trajectory_x = [1.76 + 0.48 * k for k in range(4)]
ys = [4.04 - 0.22 * i for i in range(5)]
y_low, y_high = ys[-1] - 0.10, ys[0] + 0.10
ax.add_patch(Rectangle((xs[0] - 0.11, y_low), 3 * 0.22, y_high - y_low,
                       facecolor=LABEL_BG, edgecolor="none", zorder=0))
ax.add_patch(Rectangle((xs[3] - 0.11, y_low), 5 * 0.22, y_high - y_low,
                       facecolor=HELD_BG, edgecolor="none", zorder=0))
for first, last, title, color in ((0, 2, "Labels", OUTCOME), (3, 7, "Held out", HELD)):
    rule(xs[first] - 0.105, xs[last] + 0.105, 4.36, color, 1.1)
    text((xs[first] + xs[last]) / 2, 4.48, title, color=color, ha="center")
    for d in range(first, last + 1):
        text(xs[d], 4.26, str(d), color=LOOK if d == 0 else MUTED,
             weight="bold" if d == 0 else "normal", ha="center")
text(1.105, 4.26, "Actions / picks", color=MUTED, ha="center")
text(6.00, 4.48, "Rate", color=OUTCOME, ha="center")
text(6.00, 4.26, "0–2", color=MUTED, ha="center")
text(6.74, 4.48, "Score", color=HELD, ha="center")
text(6.74, 4.26, "3–7", color=HELD, ha="center")
mid = sum(ys) / len(ys)
box(0.13, mid - 0.14, 0.43, 0.28, edge=INK)
text(0.345, mid, r"$s_t$", size=9, ha="center")
# Rollout 0's saved state sequence is shown once per candidate. Other rollouts
# retain their terminal outcomes in the matrix; their trajectories are omitted.
text(2.63, 4.48, "Rollout 0 (expanded)", color=LOOK, ha="center")
text(4.53, 4.71, "Rollout outcomes", color=MUTED, ha="center")
for k in range(1, 4):
    text((trajectory_x[k - 1] + trajectory_x[k]) / 2, 4.26,
         rf"$a_{{t+{k}}}^{{i}}$", color=MUTED, ha="center")
# Match the picked candidates to the source colors in panel (b), directly.
pick_colors = {}
for label, color in (("Pre-execution", PRE), ("Post-execution", POST),
                     ("Lookahead K = 3", LOOK), ("Observed outcome", OUTCOME)):
    pick_colors.setdefault(PICKS[label], []).append(color)
ax.add_patch(Rectangle((5.70, y_low), 0.60, y_high - y_low,
                       facecolor=LABEL_BG, edgecolor="none", zorder=0))
ax.add_patch(Rectangle((6.49, y_low), 0.48, y_high - y_low,
                       facecolor=HELD_BG, edgecolor="none", zorder=0))
for i, (y, outcomes) in enumerate(zip(ys, OUTCOMES)):
    # One enclosure makes the prefix/outcome association explicit for rollout 0.
    capsule = box(1.50, y - 0.10, xs[0] + 0.11 - 1.50, 0.20,
                  edge=LOOK, fill="#F6F3FC", lw=0.8, radius=0.045)
    capsule.set_zorder(0.5)
    arrow(0.57, mid, 0.77, y, color=LINE)
    box(0.79, y - 0.10, 0.63, 0.20)
    text(0.925, y, rf"$a_t^{{{i + 1}}}$", size=8, ha="center")
    for j, color in enumerate(pick_colors.get(i, ())):
        ax.add_patch(Circle((1.15 + 0.13 * j, y), 0.027,
                           facecolor=color, edgecolor="white", linewidth=0.4, zorder=3))
    arrow(1.44, y, 1.56, y, color=LOOK)
    for k, x in enumerate(trajectory_x):
        state_card(x, y, k + 1, POST if k == 0 else LOOK,
                   width=0.36, height=0.19, candidate=i + 1)
        if k:
            arrow(trajectory_x[k - 1] + 0.185, y, x - 0.185, y, color=LOOK)
    text(3.48, y, "…", color=MUTED, ha="center")
    arrow(3.57, y, 3.64, y, color=LINE)
    for d, success in enumerate(outcomes):
        ax.add_patch(Circle((xs[d], y), 0.025,
            facecolor=SUCCESS if success else FAILURE,
            edgecolor=SUCCESS if success else FAILURE, linewidth=0.65, zorder=3))
    text(6.00, y, percent(label_rates[i]), color=OUTCOME, ha="center",
         weight="bold" if label_rates[i] == max(label_rates) else "normal")
    text(6.74, y, f"{100 * held_rates[i]:.0f}%", color=HELD, weight="bold", ha="center")
for x, fill, label in ((0.17, SUCCESS, "Success"), (0.91, FAILURE, "Failure")):
    ax.add_patch(Circle((x, 2.98), 0.025, facecolor=fill,
                       edgecolor=fill, linewidth=0.6))
    text(x + 0.08, 2.98, label, color=MUTED)



# Source labels and fills sit on the data itself; no separate routing boxes.
text(1.65, 2.98, "Rollout 0 expanded; 1–7 summarized.", color=MUTED)

# Explicit method legend for candidate pick markers and teacher lanes.
METHOD_COLORS = (("Pre-execution", PRE), ("Post-execution", POST),
                 ("Lookahead K = 3", LOOK), ("Oracle (observed rollouts)", OUTCOME))
text(0.12, 2.74, "Methods", color=MUTED)
for x, (label, color) in zip((0.79, 2.10, 3.57, 5.25), METHOD_COLORS):
    ax.add_patch(Circle((x, 2.74), 0.032, facecolor=color,
                       edgecolor="white", linewidth=0.4, zorder=3))
    text(x + 0.09, 2.74, label, color=color)

# (b) Aligned input -> teacher -> chosen action lanes. All teachers compare
# all five candidates; the indexed trajectory below is supplied for EACH one.
text(0.12, 2.50, "(b)  Selection", size=8.5, weight="bold")
text(0.12, 2.28, r"Shared: task, history, $s_t$, $a_t^{1},\ldots,a_t^{5}$", color=MUTED)
text(4.66, 2.28, "Teacher: GPT-6 Luna · no verdicts", color=MUTED)
text(0.12, 2.03, "Method", color=MUTED)
text(1.43, 2.03, r"Rollout 0 input (per action)", color=LOOK)
text(4.88, 2.03, "Selector", color=MUTED, ha="center")
text(5.85, 2.03, "Pick", color=MUTED, ha="center")
text(6.74, 2.03, "Score", color=HELD, ha="center")
text(6.74, 1.87, "3–7", color=HELD, ha="center")
ax.add_patch(Rectangle((6.49, 0.20), 0.48, 1.63,
                       facecolor=HELD_BG, edgecolor="none", zorder=0))
future_x = [2.10 + 0.65 * k for k in range(4)]


lane_rows = ((1.68, "Pre-execution", PRE, 0),
             (1.30, "Post-execution", POST, 1),
             (0.92, "Lookahead K = 3", LOOK, 4))
for y, label, color, n_states in lane_rows:
    text(0.12, y, label, color=color)
    if n_states == 0:
        text(2.83, y, "Shared input only", color=MUTED, ha="center")
    else:
        box(1.43, y - 0.10, 0.34, 0.20, edge=color)
        text(1.60, y, r"$a_t^{i}$", size=8, color=color, ha="center")
        arrow(1.79, y, 1.88, y, color=color)
        for k, x in enumerate(future_x[:n_states]):
            state_card(x, y, k + 1, color)
            if k:
                arrow(future_x[k - 1] + 0.21, y, x - 0.21, y, color=color)
                text((future_x[k - 1] + x) / 2, y + 0.14,
                     rf"$a_{{t+{k}}}^{{i}}$", color=MUTED, ha="center")
    arrow(future_x[n_states - 1] + 0.21 if n_states else 4.29,
          y, 4.53, y, color=color)
    box(4.55, y - 0.13, 0.66, 0.26, edge=color)
    text(4.88, y, "Teacher", color=color, ha="center")
    arrow(5.23, y, 5.48, y, color=color)
    pick = PICKS[label]
    box(5.50, y - 0.13, 0.70, 0.26, edge=color)
    text(5.85, y, rf"$a_t^{{{pick + 1}}}$", size=9, color=color, ha="center")
    arrow(6.23, y, 6.51, y, color=HELD)
    text(6.74, y, f"{100 * held_rates[pick]:.0f}%", color=HELD, weight="bold", ha="center")
text(1.43, 0.67, "Subscript: time · superscript: candidate", color=MUTED)
# Outcome-derived labels are a separate operation, not a Luna response.
y = 0.36
text(0.12, y, "Oracle (0–2)", color=OUTCOME)
ax.add_patch(Rectangle((1.40, y - 0.12), 2.86, 0.24,
                       facecolor=LABEL_BG, edgecolor="none", zorder=0))
text(1.43, y, "0–2:  [" + ", ".join(percent(rate) for rate in label_rates) + "]", color=OUTCOME)
arrow(4.29, y, 4.53, y, color=OUTCOME)
box(4.55, y - 0.13, 0.66, 0.26, edge=OUTCOME)
text(4.88, y, "Argmax", color=OUTCOME, ha="center")
arrow(5.23, y, 5.48, y, color=OUTCOME)
box(5.50, y - 0.13, 0.70, 0.26, edge=OUTCOME)
text(5.85, y, r"$a_t^{3}$", size=9, color=OUTCOME, ha="center")
arrow(6.23, y, 6.51, y, color=HELD)
text(6.74, y, "80%", color=HELD, weight="bold", ha="center")



text(0.12, 0.10, "Illustrative example · valid rollouts only", color=MUTED)

for ext in ("png", "svg", "pdf"):
    path = Path(__file__).with_suffix(f".{ext}")
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.025)
    if ext == "svg":
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
plt.close(fig)
