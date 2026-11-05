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
PRE, POST, LOOK, OUTCOME, HELD = "#4B5563", "#D8877E", "#6250BE", "#397FA3", "#173F78"
ORACLE = OUTCOME
ROLLOUT, ROLLOUT_BG = "#111111", "#F4F5F6"
SHARED, SHARED_BG = "#737A83", "#F3F4F5"
SUCCESS, FAILURE = "#31785A", "#8D949B"
LABEL_BG, HELD_BG = "#EDF6FA", "#EBF0F8"
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
# Half the previous nominal height; panels now share one horizontal row.
W, H = 7.6, 2.11
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set(xlim=(0, W), ylim=(0, H))
ax.axis("off")


def text(x, y, label, size=7, color=INK, weight="normal", ha="left", va="center", **kw):
    return ax.text(x, y, label, fontsize=size, color=color, weight=weight,
                   ha=ha, va=va, **kw)


def box(x, y, w, h, edge=LINE, fill="white", lw=0.65, radius=0.025):
    patch = FancyBboxPatch((x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=fill, edgecolor=edge, linewidth=lw, zorder=2)
    ax.add_patch(patch)
    return patch


def arrow(x0, y0, x1, y1, color=MUTED, lw=0.65):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
        mutation_scale=4, shrinkA=0.4, shrinkB=0.4, linewidth=lw, color=color))


def rule(x0, x1, y, color=LINE, lw=0.6):
    ax.plot([x0, x1], [y, y], color=color, linewidth=lw, zorder=1)


def state_card(x, y, step, color, candidate):
    width, height = 0.28, 0.18
    box(x - width / 2, y - height / 2, width, height, edge=color)
    ax.add_patch(Rectangle((x - width / 2, y + height * 0.29), width, height * 0.21,
                           facecolor=color, edgecolor="none", alpha=0.15, zorder=2))
    text(x, y - 0.006, rf"$s_{{t+{step}}}^{{{candidate}}}$", color=color, ha="center")


def percent(rate):
    # Rates are scaled to percent; the column headers carry the percent sign.
    return f"{100 * rate:.1f}".rstrip("0").rstrip(".")


def outcome_marker(x, y, success, radius=0.018):
    color = SUCCESS if success else FAILURE
    ax.add_patch(Circle((x, y), radius, facecolor=color if success else "white",
                       edgecolor=color, linewidth=0.5 if success else 0.75, zorder=3))


def pick_check(x, y, color, width=0.040):
    """Vector check mark: distinct from the circular terminal-outcome symbols."""
    ax.plot([x - width * .45, x - width * .10, x + width * .50],
            [y - width * .02, y - width * .40, y + width * .44],
            color=color, linewidth=.85, solid_capstyle="round", zorder=4)


label_rates = [sum(out[:3]) / 3 for out in OUTCOMES]
held_rates = [sum(out[3:]) / 5 for out in OUTCOMES]
assert [PICKS[k] for k in ("Pre-execution", "Post-execution", "Lookahead K = 3", "Oracle")] == [0, 0, 2, 2]

# (a) Preserve all five candidates and all forty original binary outcomes.
text(0.06, 1.98, "a)  Data collection", size=8.5, weight="bold")
outcome_header = text(1.52, 0.245, r"Rollout success $R^{i,j}$", size=6.5, color=MUTED)
outcome_marker(2.60, 0.245, True, radius=0.020)
success_key = text(2.65, 0.245, "Success", size=6.5, color=SUCCESS)
outcome_marker(3.13, 0.245, False, radius=0.020)
failure_key = text(3.18, 0.245, "Failure", size=6.5, color=MUTED)
rollout_pitch = 0.11
xs = [2.24 + rollout_pitch * j for j in range(8)]
trajectory_x = [0.88 + 0.32 * k for k in range(4)]
ys = [1.40 - 0.225 * i for i in range(5)]
y_low, y_high = ys[-1] - 0.10, ys[0] + 0.10
for first, last, title, color, fill in (
        (0, 2, "Train", OUTCOME, LABEL_BG), (3, 7, "Eval", HELD, HELD_BG)):
    ax.add_patch(Rectangle((xs[first] - rollout_pitch / 2, y_low), (last - first + 1) * rollout_pitch,
                           y_high - y_low, facecolor=fill, edgecolor="none", zorder=0))
    rule(xs[first] - rollout_pitch / 2 + .001, xs[last] + rollout_pitch / 2 - .001, 1.69, color, 1.0)
    text((xs[first] + xs[last]) / 2, 1.79, title, color=color, ha="center")
    for j in range(first, last + 1):
        text(xs[j], 1.60, str(j + 1), size=6.5, color=ROLLOUT if j == 0 else MUTED,
             weight="bold" if j == 0 else "normal", ha="center")
text(xs[0] - 0.095, 1.60, r"$j=$", size=6.5, color=MUTED, ha="right")
text(0.49, 1.60, r"$K=5$ actions", size=6.5, color=MUTED, ha="center")
text(1.48, 1.79, "Rollout 1", color=ROLLOUT, weight="bold", ha="center")
for k in range(1, 4):
    text((trajectory_x[k - 1] + trajectory_x[k]) / 2, 1.60,
         rf"$a_{{t+{k}}}^{{i}}$", size=6.5, color=MUTED, ha="center")
for x, title, split, color, fill, width in (
        (3.27, "Train %", "1–3", OUTCOME, LABEL_BG, 0.34),
        (3.65, "Eval %", "4–8", HELD, HELD_BG, 0.28)):
    text(x, 1.79, title, size=6.5, color=color, ha="center")
    text(x, 1.60, split, size=6.5, color=color, ha="center")
    ax.add_patch(Rectangle((x - width / 2, y_low), width, y_high - y_low,
                           facecolor=fill, edgecolor="none", zorder=0))
mid = sum(ys) / len(ys)
box(0.03, mid - 0.12, 0.24, 0.24, edge=INK)
text(0.15, mid, r"$s_t$", size=9, ha="center")
rollout_box = box(0.72, y_low - 0.02, xs[0] + rollout_pitch / 2 - 0.72,
                  y_high - y_low + 0.04, edge=ROLLOUT, fill=ROLLOUT_BG, lw=1.0)
rollout_box.set_zorder(0.5)
pick_colors = {}
for label, color in (("Pre-execution", PRE), ("Post-execution", POST),
                     ("Lookahead K = 3", LOOK), ("Observed outcome", ORACLE)):
    pick_colors.setdefault(PICKS[label], []).append(color)
for i, (y, outcomes) in enumerate(zip(ys, OUTCOMES)):
    arrow(0.28, mid, 0.345, y, color=LINE)
    box(0.35, y - 0.09, 0.34, 0.18)
    text(0.465, y, rf"$a_t^{{{i + 1}}}$", size=7.5, ha="center")
    for j, color in enumerate(pick_colors.get(i, ())):
        pick_check(0.603 + 0.055 * j, y, color)
    arrow(0.695, y, 0.736, y, color=POST)
    if i in (0, 4):
        for k, x in enumerate(trajectory_x):
            state_card(x, y, k + 1, POST if k == 0 else LOOK, i + 1)
            if k:
                arrow(trajectory_x[k - 1] + 0.142, y, x - 0.142, y, color=LOOK)
        text(2.08, y, "…", color=MUTED, ha="center")
    else:
        text(1.48, y, r"$\cdots$", size=10, color=MUTED, ha="center")
    for j, success in enumerate(outcomes):
        outcome_marker(xs[j], y, success)
    text(3.27, y, percent(label_rates[i]), color=OUTCOME, ha="center",
         weight="bold" if label_rates[i] == max(label_rates) else "normal")
    text(3.65, y, f"{100 * held_rates[i]:.0f}", color=HELD, weight="bold", ha="center")

# (b) Shared context explicitly feeds each joint candidate-comparison call.
ax.plot([3.86, 3.86], [0.26, 1.91], color=LINE, linewidth=0.6)
method_x, evidence_x = 3.96, 5.02
teacher_x, teacher_w = 5.75, 0.74
pick_x, pick_w, eval_x = 6.60, 0.45, 7.27
text(method_x, 1.98, "b)  Teacher Labeling Methods", size=8.5, weight="bold")
box(method_x, 1.67, 3.58, 0.24, edge=SHARED, fill=SHARED_BG, lw=0.9)
shared_label = text(5.75, 1.79,
     r"Shared: task · history · $s_t$ · $\{a_t^{i}\}_{i=1}^{K}$",
     color=SHARED, ha="center")
text(evidence_x, 1.55, "Rollout 1 evidence", size=6.5, color=ROLLOUT, ha="center")
pick_header = text(pick_x + pick_w / 2, 1.55, "Pick", size=6.5, color=MUTED, ha="center")
eval_header = text(eval_x, 1.55, "Eval %", size=6.5, color=HELD, ha="center")
shared_x = 5.64
ax.plot([shared_x, shared_x], [1.67, 0.66], color=SHARED, linewidth=0.9, zorder=1)
ax.add_patch(Rectangle((7.14, 0.20), 0.32, 1.23,
                       facecolor=HELD_BG, edgecolor="none", zorder=0))
lanes = ((1.30, "Pre-execution", "Pre-exec", PRE, "—"),
         (1.01, "Post-execution", "Post-exec", POST, r"$\{s_{t+1}^{i}\}_{i=1}^{K}$"),
         (0.66, "Lookahead K = 3", "Lookahead\nH = 3", LOOK,
          r"$\{s_{t+1:t+4}^{i}\}_{i=1}^{K}$" + "\n" + r"$\{a_{t+1:t+3}^{i}\}_{i=1}^{K}$"))
for y, key, label, color, evidence in lanes:
    text(method_x, y, label, color=color, linespacing=1.0)
    text(evidence_x, y, evidence, color=color, ha="center", linespacing=0.95)
    rule(5.50, shared_x, y, color=color)
    ax.add_patch(Circle((shared_x, y), 0.015, facecolor=SHARED,
                        edgecolor="white", linewidth=0.3, zorder=3))
    arrow(shared_x + 0.02, y, teacher_x - 0.01, y, color=color)
    box(teacher_x, y - 0.115, teacher_w, 0.23, edge=color)
    text(teacher_x + teacher_w / 2, y, "Teacher", color=color, ha="center")
    arrow(teacher_x + teacher_w + 0.02, y, pick_x - 0.015, y, color=color)
    box(pick_x, y - 0.115, pick_w, 0.23, edge=color)
    pick = PICKS[key]
    text(pick_x + pick_w / 2, y, rf"$a_t^{{{pick + 1}}}$", size=8, color=color, ha="center")
    arrow(pick_x + pick_w + 0.015, y, 7.15, y, color=HELD)
    text(eval_x, y, f"{100 * held_rates[pick]:.0f}", color=HELD,
         weight="bold", ha="center")
# Lighter-blue rate labels refer to the complete five-action Train % column in (a).
y = 0.32
text(method_x, y, "Oracle", color=ORACLE)
box(4.52, y - 0.105, 1.00, 0.21, edge=OUTCOME, fill=LABEL_BG, lw=0.5)
text(evidence_x, y, "Train %", color=OUTCOME, ha="center")
arrow(5.54, y, teacher_x - 0.01, y, color=ORACLE)
box(teacher_x, y - 0.115, teacher_w, 0.23, edge=ORACLE)
text(teacher_x + teacher_w / 2, y, "Argmax", color=ORACLE, ha="center")
arrow(teacher_x + teacher_w + 0.02, y, pick_x - 0.015, y, color=ORACLE)
box(pick_x, y - 0.115, pick_w, 0.23, edge=ORACLE)
text(pick_x + pick_w / 2, y, r"$a_t^{3}$", size=8, color=ORACLE, ha="center")
arrow(pick_x + pick_w + 0.015, y, 7.15, y, color=HELD)
text(eval_x, y, "80", color=HELD, weight="bold", ha="center")

# Keep the method key right aligned; outcome status is keyed beside its R label.
items = (("Pre-exec", PRE), ("Post-exec", POST), ("Lookahead H = 3", LOOK), ("Oracle", ORACLE))
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
legend_right = W - 0.03
for label, color in reversed(items):
    width = renderer.get_text_width_height_descent(label, FontProperties(size=7), ismath=False)[0] / fig.dpi
    marker_x = legend_right - width - 0.08
    pick_check(marker_x, 0.085, color, width=0.060)
    text(legend_right, 0.085, label, color=color, ha="right")
    legend_right = marker_x - 0.19

fig.canvas.draw()
renderer = fig.canvas.get_renderer()
assert eval_header.get_window_extent(renderer).x0 - pick_header.get_window_extent(renderer).x1 >= 3
assert outcome_header.get_window_extent(renderer).x1 < 2.57 * fig.dpi
assert success_key.get_window_extent(renderer).x1 < 3.10 * fig.dpi
assert failure_key.get_window_extent(renderer).x1 < 3.86 * fig.dpi
for label in ax.texts:
    bounds = label.get_window_extent(renderer)
    assert fig.bbox.contains(*bounds.p0) and fig.bbox.contains(*bounds.p1), label.get_text()
for ext in ("png", "svg", "pdf"):
    path = Path(__file__).with_suffix(f".{ext}")
    metadata = {"Date": None} if ext == "svg" else {"CreationDate": None, "ModDate": None} if ext == "pdf" else {}
    fig.savefig(path, dpi=300, metadata=metadata)
    if ext == "svg":
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
assert b"/FontFile2" in Path(__file__).with_suffix(".pdf").read_bytes()
plt.close(fig)
