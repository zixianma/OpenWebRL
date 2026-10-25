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
W, H = 7.0, 4.55
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


label_rates = [sum(out[:3]) / 3 for out in OUTCOMES]
held_rates = [sum(out[3:]) / 5 for out in OUTCOMES]

# (a) The exact original 5 x 8 outcomes; labels never use the blue columns.
text(0.12, 4.35, "(a)  Collect eight continuation draws per action", size=8.5, weight="bold")
xs = [1.48 + 0.17 * d for d in range(8)]
ys = [3.70 - 0.19 * i for i in range(5)]
y_low, y_high = ys[-1] - 0.09, ys[0] + 0.09
ax.add_patch(Rectangle((xs[0] - 0.085, y_low), 3 * 0.17, y_high - y_low,
                       facecolor=LABEL_BG, edgecolor="none", zorder=0))
ax.add_patch(Rectangle((xs[3] - 0.085, y_low), 5 * 0.17, y_high - y_low,
                       facecolor=HELD_BG, edgecolor="none", zorder=0))
ax.add_patch(Rectangle((xs[0] - 0.074, y_low), 0.148, y_high - y_low,
                       facecolor="none", edgecolor=LOOK, linewidth=0.7, zorder=1))
for first, last, title, color in ((0, 2, "Label pool", OUTCOME), (3, 7, "Held out", HELD)):
    rule(xs[first] - 0.08, xs[last] + 0.08, 4.02, color, 1.1)
    text((xs[first] + xs[last]) / 2, 4.14, title, color=color, ha="center")
    for d in range(first, last + 1):
        text(xs[d], 3.92, str(d), color=MUTED, ha="center")
text(1.015, 3.92, "Action", color=MUTED, ha="center")
text(3.06, 4.14, "Label", color=OUTCOME, ha="center")
text(3.06, 3.92, "0–2", color=MUTED, ha="center")
text(3.58, 4.14, "Score", color=HELD, ha="center")
text(3.58, 3.92, "3–7", color=MUTED, ha="center")
mid = sum(ys) / len(ys)
box(0.13, mid - 0.14, 0.43, 0.28, edge=INK)
text(0.345, mid, r"$s_t$", size=9, ha="center")
for i, (y, outcomes) in enumerate(zip(ys, OUTCOMES)):
    arrow(0.57, mid, 0.80, y, color=LINE)
    box(0.81, y - 0.075, 0.41, 0.15)
    text(1.015, y, rf"$a_{i + 1}$", size=8, ha="center")
    arrow(1.23, y, 1.36, y, color=LINE)
    for d, success in enumerate(outcomes):
        ax.add_patch(Circle((xs[d], y), 0.025,
            facecolor=SUCCESS if success else "white",
            edgecolor=SUCCESS if success else "#88919C", linewidth=0.65))
    text(3.06, y, f"{sum(outcomes[:3])}/3", color=OUTCOME, ha="center",
         weight="bold" if label_rates[i] == max(label_rates) else "normal")
    text(3.58, y, f"{100 * held_rates[i]:.0f}%", color=HELD, weight="bold", ha="center")
for x, fill, label in ((0.17, SUCCESS, "success"), (0.91, "white", "failure")):
    ax.add_patch(Circle((x, 2.76), 0.025, facecolor=fill,
                       edgecolor=SUCCESS if fill != "white" else "#88919C", linewidth=0.6))
    text(x + 0.08, 2.76, label, color=MUTED)



# Three outputs of the collected records, with explicit connectors into (b).
for y, color, label in (
    (3.72, LOOK, "Draw 0: states, actions, tool feedback"),
    (3.32, OUTCOME, "Draws 0–2: per-action success rates"),
    (2.92, HELD, r"Draws 3–7: held-out success rates $q_i$"),
):
    box(3.97, y - 0.14, 2.65, 0.28, edge=color)
    text(5.295, y, label, color=color, ha="center")


def route(points, color):
    # Reserve arrowheads for the endpoint; intermediate bends are plain lines.
    ax.plot([p[0] for p in points[:-1]], [p[1] for p in points[:-1]],
            color=color, linewidth=0.75, zorder=1)
    arrow(*points[-2], *points[-1], color=color, lw=0.75)


route([(3.96, 3.72), (3.83, 3.72), (3.83, 2.65), (2.80, 2.65), (2.80, 2.57)], LOOK)
route([(3.96, 3.32), (3.91, 3.32), (3.91, 2.65), (5.18, 2.65), (5.18, 2.57)], OUTCOME)
route([(6.64, 2.92), (6.92, 2.92), (6.92, 2.15), (6.74, 2.15), (6.74, 2.10)], HELD)
text(2.80, 2.50, "Draw 0 → post / lookahead", color=LOOK, ha="center")
text(5.18, 2.50, "Draws 0–2 → outcome label", color=OUTCOME, ha="center")
# (b) Aligned input -> teacher -> chosen action lanes. All teachers compare
# all five candidates; the indexed trajectory below is supplied for EACH one.
text(0.12, 2.50, "(b)  Choose and score", size=8.5, weight="bold")
text(0.12, 2.28, r"Shared input: task + history + current state $s_t$ + all five candidate texts $a_1,\ldots,a_5$", color=MUTED)
text(4.66, 2.28, "Luna sees no outcome verdicts", color=MUTED)
text(0.12, 2.03, "Label source", color=MUTED)
text(1.43, 2.03, r"Additional input for each candidate $a_i$", color=MUTED)
text(4.88, 2.03, "Operation", color=MUTED, ha="center")
text(5.85, 2.03, "Output: choice", color=MUTED, ha="center")
text(6.74, 2.03, r"Score $q_i$", color=HELD, ha="center")
text(6.74, 1.87, "from (a)", color=HELD, ha="center")
future_x = [2.10 + 0.65 * k for k in range(4)]

def state_card(x, y, step, color):
    # Every state is a screenshot/observation, rather than an outcome verdict.
    box(x - 0.20, y - 0.105, 0.40, 0.21, edge=color, fill="white")
    ax.add_patch(Rectangle((x - 0.20, y + 0.061), 0.40, 0.044,
                           facecolor=color, edgecolor="none", alpha=0.15, zorder=2))
    text(x, y - 0.014, rf"$s_{{t+{step}}}$", ha="center", color=color)


lane_rows = ((1.68, "Pre-execution", PRE, 0),
             (1.30, "Post-execution", POST, 1),
             (0.92, "Lookahead K = 3", LOOK, 4))
for y, label, color, n_states in lane_rows:
    text(0.12, y, label, color=color)
    if n_states == 0:
        text(2.83, y, "Shared input only; no execution", color=MUTED, ha="center")
    else:
        box(1.43, y - 0.10, 0.34, 0.20, edge=color)
        text(1.60, y, r"$a_i$", size=8, color=color, ha="center")
        arrow(1.79, y, 1.88, y, color=color)
        for k, x in enumerate(future_x[:n_states]):
            state_card(x, y, k + 1, color)
            if k:
                arrow(future_x[k - 1] + 0.21, y, x - 0.21, y, color=color)
                text((future_x[k - 1] + x) / 2, y + 0.14,
                     rf"$u_{k}$", color=MUTED, ha="center")
    arrow(future_x[n_states - 1] + 0.21 if n_states else 4.29,
          y, 4.53, y, color=color)
    box(4.55, y - 0.13, 0.66, 0.26, edge=color)
    text(4.88, y, "GPT-6 Luna", color=color, ha="center")
    arrow(5.23, y, 5.48, y, color=color)
    pick = PICKS[label]
    box(5.50, y - 0.13, 0.70, 0.26, edge=color)
    text(5.85, y, rf"$a_{pick + 1}$", size=9, color=color, ha="center")
    arrow(6.23, y, 6.51, y, color=HELD)
    text(6.74, y, f"{100 * held_rates[pick]:.0f}%", color=HELD, weight="bold", ha="center")
text(1.43, 0.67, r"Draw 0 only; $u_1,u_2,u_3$: actor steps. K = 1 stops at $s_{t+2}$.", color=MUTED)
# Outcome-derived labels are a separate operation, not a Luna response.
y = 0.36
text(0.12, y, "Observed outcome", color=OUTCOME)
text(1.43, y, "Draws 0–2:  [2/3, 1/3, 3/3, 0/3, 1/3]", color=OUTCOME)
arrow(4.29, y, 4.53, y, color=OUTCOME)
box(4.55, y - 0.13, 0.66, 0.26, edge=OUTCOME)
text(4.88, y, "Argmax rate", color=OUTCOME, ha="center")
arrow(5.23, y, 5.48, y, color=OUTCOME)
box(5.50, y - 0.13, 0.70, 0.26, edge=OUTCOME)
text(5.85, y, r"$a_3$", size=9, color=OUTCOME, ha="center")
arrow(6.23, y, 6.51, y, color=HELD)
text(6.74, y, "80%", color=HELD, weight="bold", ha="center")



text(0.12, 0.10, "Illustrative values · valid draws only. Teacher inputs omit outcome verdicts; draws 3–7 are reserved for scoring.", color=MUTED)

for ext in ("png", "svg", "pdf"):
    path = Path(__file__).with_suffix(f".{ext}")
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.025)
    if ext == "svg":
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
plt.close(fig)
