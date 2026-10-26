"""Paper figure for offline selector evaluation (October 21 ablation).

The outcomes and teacher picks are illustrative, preserved from the original
figure. Displayed rollouts 1--3 supply outcome labels; teachers see only rollout 1, and all
pickers are scored on held-out rollouts 4--8. Invalid rollouts are excluded from the
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
from matplotlib.font_manager import FontProperties

INK, MUTED, LINE = "#24272B", "#626972", "#C9CDD3"
PRE, POST, LOOK, OUTCOME, HELD = "#4B5563", "#D8877E", "#6250BE", "#ED7D00", "#2164A5"
ROLLOUT, ROLLOUT_BG = "#168C9E", "#EAF7FA"
SHARED, SHARED_BG = "#737A83", "#F3F4F5"
SUCCESS, FAILURE = "#31785A", "#C53A36"
LABEL_BG, HELD_BG = "#FFF0DD", "#EDF3FA"
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
W, H = 7.0, 4.22
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
text(0.12, 4.06, "a)  Data collection", size=8.5, weight="bold")
xs = [3.76 + 0.22 * d for d in range(8)]
trajectory_x = [1.76 + 0.48 * k for k in range(4)]
ys = [3.43 - 0.21 * i for i in range(5)]
text(4.53, 4.06, r"$R^{i,j}\in\{0,1\}$", size=7, color=MUTED, ha="center")
y_low, y_high = ys[-1] - 0.10, ys[0] + 0.10
ax.add_patch(Rectangle((xs[0] - 0.11, y_low), 3 * 0.22, y_high - y_low,
                       facecolor=LABEL_BG, edgecolor="none", zorder=0))
ax.add_patch(Rectangle((xs[3] - 0.11, y_low), 5 * 0.22, y_high - y_low,
                       facecolor=HELD_BG, edgecolor="none", zorder=0))
for first, last, title, color in ((0, 2, "Train labels", OUTCOME), (3, 7, "Held-out eval", HELD)):
    rule(xs[first] - 0.105, xs[last] + 0.105, 3.74, color, 1.1)
    text((xs[first] + xs[last]) / 2, 3.85, title, color=color, ha="center")
    for d in range(first, last + 1):
        text(xs[d], 3.65, str(d + 1), color=ROLLOUT if d == 0 else MUTED,
             weight="bold" if d == 0 else "normal", ha="center")
text(xs[0] - 0.17, 3.65, r"$j=$", color=MUTED, ha="right")
text(1.105, 3.65, r"$K=5$ actions", color=MUTED, ha="center")
text(6.00, 3.85, "Train %", color=OUTCOME, ha="center")
text(6.00, 3.65, "1–3", color=OUTCOME, ha="center")
text(6.74, 3.85, "Eval %", color=HELD, ha="center")
text(6.74, 3.65, "4–8", color=HELD, ha="center")
mid = sum(ys) / len(ys)
box(0.13, mid - 0.14, 0.43, 0.28, edge=INK)
text(0.345, mid, r"$s_t$", size=9, ha="center")
# Spell out rollout 1's states for candidates 1 and 5; ellipses omit 2--4.
# Every candidate and all eight terminal outcomes remain in the matrix.
text(2.63, 3.85, "Rollout 1", color=ROLLOUT, weight="bold", ha="center")
for k in range(1, 4):
    text((trajectory_x[k - 1] + trajectory_x[k]) / 2, 3.65,
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
# One cyan enclosure groups rollout 1's prefixes and terminal outcomes
# across all candidates. State colors retain their evidence-method association.
rollout_box = box(1.50, y_low - 0.02, xs[0] + 0.11 - 1.50,
                  y_high - y_low + 0.04, edge=ROLLOUT, fill=ROLLOUT_BG,
                  lw=1.05, radius=0.045)
rollout_box.set_zorder(0.5)
for i, (y, outcomes) in enumerate(zip(ys, OUTCOMES)):
    arrow(0.57, mid, 0.77, y, color=LINE)
    box(0.79, y - 0.10, 0.63, 0.20)
    text(0.925, y, rf"$a_t^{{{i + 1}}}$", size=8, ha="center")
    for j, color in enumerate(pick_colors.get(i, ())):
        ax.add_patch(Circle((1.15 + 0.13 * j, y), 0.027,
                           facecolor=color, edgecolor="white", linewidth=0.4, zorder=3))
    arrow(1.44, y, 1.56, y, color=POST)
    if i in (0, 4):
        for k, x in enumerate(trajectory_x):
            state_card(x, y, k + 1, POST if k == 0 else LOOK,
                       width=0.36, height=0.19, candidate=i + 1)
            if k:
                arrow(trajectory_x[k - 1] + 0.185, y, x - 0.185, y, color=LOOK)
        text(3.48, y, "…", color=MUTED, ha="center")
    else:
        text(2.47, y, r"$\cdots$", size=11, color=MUTED, ha="center")
    arrow(3.57, y, 3.64, y, color=LINE)
    for d, success in enumerate(outcomes):
        ax.add_patch(Circle((xs[d], y), 0.025,
            facecolor=SUCCESS if success else FAILURE,
            edgecolor=SUCCESS if success else FAILURE, linewidth=0.65, zorder=3))
    text(6.00, y, percent(label_rates[i]), color=OUTCOME, ha="center",
         weight="bold" if label_rates[i] == max(label_rates) else "normal")
    text(6.74, y, f"{100 * held_rates[i]:.0f}%", color=HELD, weight="bold", ha="center")
# One right-aligned legend combines status and method keys.
METHOD_COLORS = (("Pre-exec", PRE), ("Post-exec", POST),
                 ("Lookahead H = 3", LOOK), ("Oracle (1–3)", OUTCOME))
LEGEND_ITEMS = (("Success", SUCCESS), ("Failure", FAILURE), ("|", MUTED)) + METHOD_COLORS
renderer = fig.canvas.get_renderer()
legend_right = 6.97
for label, color in reversed(LEGEND_ITEMS):
    if label == "|":
        text(legend_right + 0.035, 2.32, "|", color=MUTED, ha="right")
        legend_right -= 0.05
        continue
    label_width = renderer.get_text_width_height_descent(
        label, FontProperties(size=7), ismath=False)[0] / fig.dpi
    marker_x = legend_right - label_width - 0.08
    ax.add_patch(Circle((marker_x, 2.32), 0.027, facecolor=color,
                       edgecolor="white", linewidth=0.4, zorder=3))
    text(legend_right, 2.32, label, color=color, ha="right")
    legend_right = marker_x - 0.19

# (b) Aligned input -> teacher -> chosen action lanes. All teachers compare
# all five candidates; the indexed trajectory below is supplied for EACH one.
text(0.12, 2.10, "b)  Teacher Labeling Methods", size=8.5, weight="bold")
# A neutral gray context box fans into all three teacher inputs. Junction dots
# explicitly join the shared context and each complete candidate/evidence set.
box(0.12, 1.745, 4.23, 0.27, edge=SHARED, fill=SHARED_BG, lw=0.95)
text(2.235, 1.88, r"Shared: task · history · $s_t$ · $K=5$ actions", color=SHARED, ha="center")
shared_x = 4.68
ax.plot([4.35, shared_x, shared_x], [1.88, 1.88, 0.63],
        color=SHARED, linewidth=0.95, zorder=1)
text(1.43, 1.62, "Candidates + rollout 1", color=MUTED)
text(6.035, 1.62, "Pick", color=MUTED, ha="center")
text(6.74, 1.62, "Eval %", color=HELD, ha="center")
text(6.74, 1.46, "4–8", color=HELD, ha="center")
ax.add_patch(Rectangle((6.49, 0.07), 0.48, 1.38,
                       facecolor=HELD_BG, edgecolor="none", zorder=0))
future_x = [2.25 + 0.55 * k for k in range(4)]


lane_rows = ((1.31, "Pre-execution", PRE, 0),
             (0.97, "Post-execution", POST, 1),
             (0.63, "Lookahead K = 3", LOOK, 4))
for y, label, color, n_states in lane_rows:
    display_label = {"Pre-execution": "Pre-exec", "Post-execution": "Post-exec",
                     "Lookahead K = 3": "Lookahead H = 3"}[label]
    text(0.12, y, display_label, color=color)
    if n_states == 0:
        text(2.54, y, r"$\{a_t^{i}\}_{i=1}^{K}$", size=9, color=color, ha="center")
        input_end = 4.29
    else:
        # The entire candidate/evidence pair is inside an i=1..K set (K=5). All five
        # pairs are supplied together in ONE teacher prompt, not separate calls.
        text(1.45, y, r"$\{$", size=14, color=color, ha="center")
        box(1.63, y - 0.10, 0.34, 0.20, edge=color)
        text(1.80, y, r"$a_t^{i}$", size=8, color=color, ha="center")
        arrow(1.99, y, 2.05, y, color=color)
        for k, x in enumerate(future_x[:n_states]):
            state_card(x, y, k + 1, color, width=0.36)
            if k:
                arrow(future_x[k - 1] + 0.185, y, x - 0.185, y, color=color)
                text((future_x[k - 1] + x) / 2, y + 0.14,
                     rf"$a_{{t+{k}}}^{{i}}$", color=MUTED, ha="center")
        last_x = future_x[n_states - 1]
        text(last_x + 0.23, y, r"$\}_{i=1}^{K}$", size=14, color=color)
        input_end = last_x + 0.66
    rule(input_end, shared_x, y, color=color)
    ax.add_patch(Circle((shared_x, y), 0.019, facecolor=SHARED,
                        edgecolor="white", linewidth=0.3, zorder=3))
    arrow(shared_x, y, 4.80, y, color=color)
    box(4.82, y - 0.13, 0.66, 0.26, edge=color)
    text(5.15, y, "Teacher", color=color, ha="center")
    arrow(5.50, y, 5.73, y, color=color)
    pick = PICKS[label]
    box(5.75, y - 0.13, 0.57, 0.26, edge=color)
    text(6.035, y, rf"$a_t^{{{pick + 1}}}$", size=9, color=color, ha="center")
    arrow(6.34, y, 6.51, y, color=HELD)
    text(6.74, y, f"{100 * held_rates[pick]:.0f}%", color=HELD, weight="bold", ha="center")
# Outcome-derived labels are a separate operation, not a Luna response.
y = 0.21
text(0.12, y, "Oracle", color=OUTCOME)
ax.add_patch(Rectangle((1.40, y - 0.12), 2.86, 0.24,
                       facecolor=LABEL_BG, edgecolor="none", zorder=0))
text(1.43, y, "[" + ", ".join(percent(rate) for rate in label_rates) + "]", color=OUTCOME)
arrow(4.29, y, 4.80, y, color=OUTCOME)
box(4.82, y - 0.13, 0.66, 0.26, edge=OUTCOME)
text(5.15, y, "Argmax", color=OUTCOME, ha="center")
arrow(5.50, y, 5.73, y, color=OUTCOME)
box(5.75, y - 0.13, 0.57, 0.26, edge=OUTCOME)
text(6.035, y, r"$a_t^{3}$", size=9, color=OUTCOME, ha="center")
arrow(6.34, y, 6.51, y, color=HELD)
text(6.74, y, "80%", color=HELD, weight="bold", ha="center")




for ext in ("png", "svg", "pdf"):
    path = Path(__file__).with_suffix(f".{ext}")
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.025)
    if ext == "svg":
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
plt.close(fig)
