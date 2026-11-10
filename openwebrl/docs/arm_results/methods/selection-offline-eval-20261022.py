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
# Shared paper typography, including bundled Albert Sans fonts.
import sys as _paper_sys
from pathlib import Path as _PaperPath
_paper_sys.path.insert(0, str(next(p for p in _PaperPath(__file__).resolve().parents
                                if (p / "paper_plot_style.py").is_file())))
import paper_plot_style as _paper_style
from paper_plot_palette import COLORS as C, tint

from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

INK, MUTED, LINE = C["INK"], C["MUTED"], C["BORDER"]
PRE, IMMEDIATE, LOOK, OUTCOME, HELD = (C["PRE_EXEC"], C["LOOKAHEAD_EVIDENCE"], C["LOOKAHEAD_TEACHER"], C["TRAIN"], C["EVAL"])
ORACLE = C["ORACLE"]
ROLLOUT, ROLLOUT_BG = C["INK"], tint("BASELINE_LIGHT", .88)
SHARED, SHARED_BG = C["BASELINE_DARK"], tint("BASELINE_LIGHT", .92)
SUCCESS, FAILURE = C["SUCCESS"], C["FAILURE"]
LABEL_BG, HELD_BG = tint("TRAIN", .35), tint("EVAL", .35)
# Same 5 x 8 outcomes and picks as the original worked example.
OUTCOMES = (
    (1, 0, 1, 1, 0, 1, 1, 0),
    (0, 0, 1, 0, 0, 0, 1, 0),
    (1, 1, 1, 1, 0, 1, 1, 1),
    (0, 0, 0, 0, 0, 1, 0, 0),
    (0, 1, 0, 1, 0, 0, 1, 0),
)
PICKS = {"Pre-execution": 0, "Post-execution": 0, "Lookahead H = 4": 2,
         "Observed outcome": 2, "Oracle": 2}

plt.rcParams.update({
    "font.family": "Albert Sans", "font.size": 7,
    "text.color": INK, "mathtext.fontset": "custom",
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


def arrow(x0, y0, x1, y1, color=MUTED, lw=0.65, head=4):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
        mutation_scale=head, shrinkA=0.4, shrinkB=0.4, linewidth=lw, color=color))


def rule(x0, x1, y, color=LINE, lw=0.6):
    ax.plot([x0, x1], [y, y], color=color, linewidth=lw, zorder=1)


def state_card(x, y, step, color, candidate):
    width, height = 0.28, 0.18
    box(x - width / 2, y - height / 2, width, height, edge=color,
        fill=tint("STATE", .60), lw=1.6)
    ax.add_patch(Rectangle((x - width / 2, y + height * 0.29), width, height * 0.21,
                           facecolor=C["STATE"], edgecolor="none", alpha=0.65, zorder=2))
    text(x, y - 0.006, rf"$s_{{t+{step}}}^{{{candidate}}}$", color=C["STATE_INK"], ha="center")


def percent(rate):
    # Rates are scaled to percent; the column headers carry the percent sign.
    return f"{100 * rate:.1f}".rstrip("0").rstrip(".")


def outcome_marker(x, y, success, radius=0.018):
    color = SUCCESS if success else FAILURE
    ax.add_patch(Circle((x, y), radius, facecolor=color if success else "white",
                       edgecolor=color, linewidth=0.5 if success else 0.75, zorder=3))


label_rates = [sum(out[:3]) / 3 for out in OUTCOMES]
held_rates = [sum(out[3:]) / 5 for out in OUTCOMES]
assert [PICKS[k] for k in ("Pre-execution", "Post-execution", "Lookahead H = 4", "Oracle")] == [0, 0, 2, 2]

# (a) Preserve all five candidates and all forty original binary outcomes.
text(0.06, 1.98, "a)  Rollout Collection", size=8.5, weight="bold", color=C["ACTION_INK"])
outcome_header = text(1.65, 0.245, r"Rollout success $R^{i,j}\in\{0,1\}$",
                      color=MUTED, ha="center")
# Outcome status key only; selected-action check marks remain omitted.
for x, success, label in ((2.95, True, "Success"), (3.73, False, "Failure")):
    outcome_marker(x, 0.245, success, radius=0.020)
    text(x + 0.055, 0.245, label, color=MUTED, size=6.5)
rollout_pitch = 0.11
xs = [2.79 + rollout_pitch * j for j in range(8)]
trajectory_x = [1.10 + 0.41 * k for k in range(4)]
ys = [1.40 - 0.225 * i for i in range(5)]
y_low, y_high = ys[-1] - 0.10, ys[0] + 0.10
for first, last, title, color, fill in (
        (0, 2, "Train", OUTCOME, LABEL_BG), (3, 7, "Eval", HELD, HELD_BG)):
    ax.add_patch(Rectangle((xs[first] - rollout_pitch / 2, y_low), (last - first + 1) * rollout_pitch,
                           y_high - y_low, facecolor=fill, edgecolor="none", zorder=0))
    rule(xs[first] - rollout_pitch / 2 + .001, xs[last] + rollout_pitch / 2 - .001, 1.69, color, 1.0)
    text((xs[first] + xs[last]) / 2, 1.79, title, color=INK, ha="center")
    for j in range(first, last + 1):
        text(xs[j], 1.60, str(j + 1), size=6.5, color=ROLLOUT if j == 0 else MUTED,
             weight="bold" if j == 0 else "normal", ha="center")
text(xs[0] - 0.095, 1.60, r"$j=$", size=6.5, color=MUTED, ha="right")
text(0.64, 1.60, r"$K=5$ actions", size=6.5, color=MUTED, ha="center")
text(1.73, 1.79, "Rollout 1", color=ROLLOUT, weight="bold", ha="center")
for k in range(1, 4):
    text((trajectory_x[k - 1] + trajectory_x[k]) / 2, 1.60,
         rf"$a_{{t+{k}}}^{{i}}$", size=6.5, color=C["ACTION_INK"], ha="center")
for x, title, split, color, fill, width in (
        (3.82, "Train %", "1–3", OUTCOME, LABEL_BG, 0.34),
        (4.20, "Eval %", "4–8", HELD, HELD_BG, 0.28)):
    text(x, 1.79, title, size=6.5, color=INK, ha="center")
    text(x, 1.60, split, size=6.5, color=INK, ha="center")
    ax.add_patch(Rectangle((x - width / 2, y_low), width, y_high - y_low,
                           facecolor=fill, edgecolor="none", zorder=0))
mid = sum(ys) / len(ys)
box(0.03, mid - 0.12, 0.24, 0.24, edge=C["STATE_BORDER"], fill=tint("STATE", .60))
text(0.15, mid, r"$s_t$", size=9, color=C["STATE_INK"], ha="center")
for i, (y, outcomes) in enumerate(zip(ys, OUTCOMES)):
    arrow(0.28, mid, 0.465, y, color=C["ACTION_INK"], lw=.75, head=5)
    box(0.47, y - 0.09, 0.34, 0.18, edge=C["ACTION_INK"], fill=tint("ACTION", .65), lw=.85)
    text(0.64, y, rf"$a_t^{{{i + 1}}}$", size=7.5, color=C["ACTION_INK"], ha="center")
    arrow(0.815, y, 0.950, y, color=C["ACTION_INK"], lw=.75, head=5)
    if i in (0, 4):
        for k, x in enumerate(trajectory_x):
            state_card(x, y, k + 1, IMMEDIATE if k == 0 else LOOK, i + 1)
            if k:
                arrow(trajectory_x[k - 1] + 0.142, y, x - 0.142, y,
                      color=C["ACTION_INK"], lw=.75, head=5)
        arrow(trajectory_x[-1] + .145, y, 2.57, y, color=C["ACTION_INK"])
        text(2.63, y, "…", color=MUTED, ha="center")
    else:
        text(1.73, y, r"$\cdots$", size=10, color=MUTED, ha="center")
    for j, success in enumerate(outcomes):
        outcome_marker(xs[j], y, success)
    text(3.82, y, percent(label_rates[i]), color=INK, ha="center")
    text(4.20, y, f"{100 * held_rates[i]:.0f}", color=INK, ha="center")

# (b) Shared context explicitly feeds each joint candidate-comparison call.
ax.plot([4.42, 4.42], [0.26, 1.91], color=LINE, linewidth=0.6)
method_x, evidence_x = 4.52, 5.55
teacher_x, teacher_w = 6.20, 0.54
pick_x, pick_w, eval_x = 6.86, 0.32, 7.40
text(method_x, 1.98, "b)  Action Decision Labeling", size=8.5, weight="bold", color=C["ACTION_INK"])
box(method_x, 1.67, 3.02, 0.24, edge=SHARED, fill=SHARED_BG, lw=0.9)
shared_label = text(6.03, 1.79,
     r"Shared: task · history · $s_t$ · $\{a_t^{i}\}_{i=1}^{K}$",
     color=SHARED, ha="center")
text(evidence_x, 1.55, "Rollout 1 evidence", size=6.5, color=ROLLOUT, ha="center")
pick_header = text(pick_x + pick_w / 2, 1.55, "Pick", size=6.5, color=MUTED, ha="center")
eval_header = text(eval_x, 1.55, "Eval %", size=6.5, color=INK, ha="center")
shared_x = 6.08
ax.plot([shared_x, shared_x], [1.67, 0.83], color=SHARED, linewidth=0.9, zorder=1)
ax.add_patch(Rectangle((7.26, 0.20), 0.28, 1.23,
                       facecolor=HELD_BG, edgecolor="none", zorder=0))
# H counts the candidate action; the legacy K=3 source now displays H=4.
# The immediate-only (H=1) teacher result is retained in source tables, not plotted.
lanes = ((1.27, "Pre-execution", "Pre-execution", PRE, "—"),
         (0.83, "Lookahead H = 4", "Lookahead\nH = 4", LOOK,
          r"$\{s_{t+1:t+4}^{i}\}_{i=1}^{K}$" + "\n" + r"$\{a_{t+1:t+3}^{i}\}_{i=1}^{K}$"))
for y, key, label, color, evidence in lanes:
    is_lookahead = key.startswith("Lookahead")
    text(method_x, y, label, color=color, weight="bold", linespacing=1.0)
    text(evidence_x, y, evidence, color=color,
         weight="bold" if is_lookahead else "normal", ha="center", linespacing=0.95)
    rule(5.99, shared_x, y, color=color)
    ax.add_patch(Circle((shared_x, y), 0.015, facecolor=SHARED,
                        edgecolor="white", linewidth=0.3, zorder=3))
    arrow(shared_x + 0.02, y, teacher_x - 0.01, y, color=color)
    box(teacher_x, y - 0.115, teacher_w, 0.23, edge=color)
    text(teacher_x + teacher_w / 2, y, "Teacher", color=color, weight="bold", ha="center")
    arrow(teacher_x + teacher_w + 0.02, y, pick_x - 0.015, y, color=color)
    box(pick_x, y - 0.115, pick_w, 0.23, edge=C["ACTION_INK"], fill=tint("ACTION", .65), lw=.85)
    pick = PICKS[key]
    text(pick_x + pick_w / 2, y, rf"$a_t^{{{pick + 1}}}$", size=8, color=C["ACTION_INK"], ha="center")
    arrow(pick_x + pick_w + 0.015, y, 7.28, y, color=HELD)
    text(eval_x, y, f"{100 * held_rates[pick]:.0f}", color=INK, ha="center")
# The yellow input box refers to the complete five-action Train % column in (a).
y = 0.32
text(method_x, y, "Oracle", color=INK)
box(5.15, y - 0.105, 0.80, 0.21, edge=OUTCOME, fill=LABEL_BG, lw=0.5)
text(evidence_x, y, "Train %", color=INK, ha="center")
arrow(5.97, y, teacher_x - 0.01, y, color=ORACLE)
box(teacher_x, y - 0.115, teacher_w, 0.23, edge=ORACLE)
text(teacher_x + teacher_w / 2, y, "Argmax", color=INK, ha="center")
arrow(teacher_x + teacher_w + 0.02, y, pick_x - 0.015, y, color=ORACLE)
box(pick_x, y - 0.115, pick_w, 0.23, edge=C["ACTION_INK"], fill=tint("ACTION", .65), lw=.85)
text(pick_x + pick_w / 2, y, r"$a_t^{3}$", size=8, color=C["ACTION_INK"], ha="center")
arrow(pick_x + pick_w + 0.015, y, 7.28, y, color=HELD)
text(eval_x, y, "80", color=INK, ha="center")

fig.canvas.draw()
renderer = fig.canvas.get_renderer()
assert eval_header.get_window_extent(renderer).x0 - pick_header.get_window_extent(renderer).x1 >= 3
assert outcome_header.get_window_extent(renderer).x1 < 4.42 * fig.dpi
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
