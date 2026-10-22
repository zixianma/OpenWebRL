"""Offline evaluation on SFT-actor branching states (scripts/train_head_round.py: targets, held_rate; 2026-10-21 ablation).
(a) one branching state: 2-5 SFT-proposed actions, each executed in a replayed browser and continued 8 times by the frozen SFT
actor to a judged outcome; (b) training labels; (c) training the head and scoring every picker on held-out draws."""
from pathlib import Path
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

INK, MUTED, LINE = "#26251f", "#6b6962", "#b9b7ad"
WIN, FAIL = "#1a9e6e", "#c9c6bc"
TEACH, PRE, OUTC, HELD, STU = "#6250d6", "#d6457e", "#d48a00", "#2a78d6", "#e05a24"
OUTC_BG, HELD_BG = "#fdf3df", "#e6effb"
plt.rcParams.update({"font.size": 9, "text.color": INK, "font.family": "DejaVu Sans"})
fig = plt.figure(figsize=(16, 8.6))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 16); ax.set_ylim(0, 8.6); ax.axis("off")


def rbox(x, y, w, h, ec=LINE, fc="white", lw=1.0, r=0.08):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.0,rounding_size={r}", fc=fc, ec=ec, lw=lw))


def arrow(x0, y0, x1, y1, color=MUTED, lw=1.1, style="-|>", ms=10):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=ms, color=color, lw=lw, shrinkA=0, shrinkB=0))


def screen(x, y, w, h, ec=LINE, lw=1.0, label=None, label_color=INK):
    rbox(x, y, w, h, ec=ec, lw=lw, r=0.04)
    ax.add_patch(Rectangle((x, y + h - 0.09), w, 0.09, fc="#efeee9", ec="none"))
    for k in range(3):
        ax.add_patch(Rectangle((x + 0.08, y + h - 0.25 - k * 0.12), w * (0.75 - 0.2 * k), 0.05, fc="#dedcd5", ec="none"))
    if label:
        ax.text(x + w / 2, y - 0.1, label, ha="center", va="top", fontsize=8.5, color=label_color)


# ---------------- (a) one branching state ----------------
ax.text(0.15, 8.35, "(a)", fontsize=12, weight="bold")
ax.text(0.6, 8.35, "One branching state and its continuations", fontsize=11.5, weight="bold")
# history
screen(0.25, 4.15, 0.55, 0.45, label="t−2")
screen(1.0, 4.15, 0.55, 0.45, label="t−1")
arrow(0.82, 4.37, 0.98, 4.37)
screen(1.8, 3.95, 0.95, 0.85, ec=INK, lw=1.6)
arrow(1.57, 4.37, 1.78, 4.37)
ax.text(2.27, 3.82, "state sₜ", ha="center", va="top", fontsize=9.5, weight="bold")
ax.text(1.45, 3.45, "task + screenshot + history\nfrom an SFT-4B episode", ha="center", va="top", fontsize=8, color=MUTED)

actions = ['click "Search"', 'type "red shoes"', "scroll down", 'click "Deals"']
outcomes = [[1, 0, 1, 1, 0, 1, 1, 0], [0, 0, 1, 0, 0, 0, 1, 0], [1, 1, 1, 1, 0, 1, 1, 1], [0, 0, 0, 0, 0, 1, 0, 0]]
rng = random.Random(7)
row_h, gap = 0.155, 0.42
group_h = 8 * row_h
tops = [7.75 - i * (group_h + gap) for i in range(4)]
ax_x, ax_w, sx, roll_x0, roll_x1, end_x = 2.95, 1.1, 4.45, 5.95, 8.55, 8.75
ax.text(ax_x + ax_w / 2, 8.0, "proposed actions", ha="center", fontsize=8.5, color=MUTED)
ax.text(sx + 0.3, 8.0, "after action", ha="center", fontsize=8.5, color=MUTED)
ax.text(roll_x0 - 0.3, 8.0, "draws 0–7: SFT actor continues sₜ₊₂, sₜ₊₃, …", ha="left", fontsize=8.5, color=MUTED)
# what the pre-execution teacher sees: history, sₜ and the action texts
ax.add_patch(FancyBboxPatch((0.12, 1.25), ax_x + ax_w + 0.08 - 0.12, 6.65, boxstyle="round,pad=0,rounding_size=0.12", fc="none", ec=PRE, lw=1.4, ls=(0, (4, 3))))
ax.text(0.3, 1.45, "pre-execution teacher sees\nonly what is inside this box", fontsize=8.5, color=PRE, weight="bold", va="bottom")
ax.text(end_x + 0.05, 8.0, "success", ha="center", fontsize=8.5, color=MUTED)
for i, (name, outs) in enumerate(zip(actions, outcomes)):
    top = tops[i]; mid = top - group_h / 2
    # edge from state to action
    arrow(2.77, 4.37, ax_x - 0.02, mid, color=LINE, lw=1.0, ms=8)
    rbox(ax_x, mid - 0.2, ax_w, 0.4, ec=INK, lw=1.0)
    ax.text(ax_x + ax_w / 2, mid, f"a{i + 1}", ha="center", va="center", fontsize=9, weight="bold")
    ax.text(ax_x + ax_w / 2, mid - 0.3, name, ha="center", va="top", fontsize=7.6, color=MUTED)
    # executing the action gives the next state s_{t+1} (draw 0's screenshot shown; seen by the lookahead teacher)
    arrow(ax_x + ax_w + 0.02, mid, sx - 0.02, mid, color=INK, lw=1.0, ms=8)
    screen(sx, mid - 0.27, 0.6, 0.54, ec=TEACH, lw=1.6)
    ax.text(sx + 0.3, mid - 0.36, f"sₜ₊₁ after a{i + 1}", ha="center", va="top", fontsize=7.8, color=TEACH, weight="bold")
    # draw bands
    ax.add_patch(Rectangle((roll_x0 - 0.32, top - 3 * row_h), roll_x1 - roll_x0 + 0.75, 3 * row_h, fc=OUTC_BG, ec="none", zorder=0))
    ax.add_patch(Rectangle((roll_x0 - 0.32, top - 8 * row_h), roll_x1 - roll_x0 + 0.75, 5 * row_h, fc=HELD_BG, ec="none", zorder=0))
    for d, o in enumerate(outs):
        y = top - (d + 0.5) * row_h
        ax.plot([sx + 0.62, roll_x0 - 0.3], [mid, y], color=LINE, lw=0.5, zorder=1)
        ax.text(roll_x0 - 0.17, y, str(d), ha="center", va="center", fontsize=6.5, color=MUTED)
        n = rng.randint(4, 12)
        xs = [roll_x0 + (roll_x1 - roll_x0) * k / 12 for k in range(n)]
        ax.plot([xs[0], xs[-1]], [y, y], color="#8d8b83", lw=0.7, zorder=2)
        ax.scatter(xs, [y] * n, s=5, color="#8d8b83", zorder=3)
        if d == 0:     # lookahead window seen by the teacher
            ax.add_patch(Rectangle((xs[0] - 0.07, y - 0.065), xs[2] - xs[0] + 0.14, 0.13, fc="none", ec=TEACH, lw=1.4, zorder=4))
            if i == 0:
                ax.text((xs[0] + xs[2]) / 2, y + 0.1, "K = 3", ha="center", va="bottom", fontsize=7.5, color=TEACH, weight="bold")
        ax.text(end_x, y, "✓" if o else "✗", ha="center", va="center", fontsize=8, color=WIN if o else "#9a978d", weight="bold")
    a, h = sum(outs[:3]), sum(outs[3:])
    ax.text(end_x + 0.3, top - 1.5 * row_h, f"{a}/3", va="center", fontsize=8, color=OUTC, weight="bold")
    ax.text(end_x + 0.3, top - 5.5 * row_h, f"{h}/5 = {20 * h}%", va="center", fontsize=8, color=HELD, weight="bold")
# legend for (a)
ly = 0.55
ax.add_patch(Rectangle((0.3, ly - 0.08), 0.3, 0.16, fc="none", ec=PRE, lw=1.4, ls=(0, (3, 2))))
ax.text(0.7, ly, "pre-execution view", va="center", fontsize=8.5)
ax.add_patch(Rectangle((2.2, ly - 0.07), 0.3, 0.14, fc="none", ec=TEACH, lw=1.4))
ax.text(2.6, ly, "lookahead adds sₜ₊₁ and the next K steps of draw 0", va="center", fontsize=8.5)
ax.add_patch(Rectangle((5.75, ly - 0.08), 0.3, 0.16, fc=OUTC_BG, ec=OUTC, lw=0.6))
ax.text(6.15, ly, "draws 0–2: outcome labels", va="center", fontsize=8.5)
ax.add_patch(Rectangle((7.95, ly - 0.08), 0.3, 0.16, fc=HELD_BG, ec=HELD, lw=0.6))
ax.text(8.35, ly, "draws 3–7: eval only", va="center", fontsize=8.5)
ax.text(0.3, 0.02, "Each draw re-opens sₜ in a fresh replayed browser, executes the action, and lets the frozen SFT actor finish the task; "
        "o4-mini judges success.\nsₜ₊₁ shown is draw 0's; it can differ slightly between draws. Invalid draws are dropped. Example values.", fontsize=8, color=MUTED)

# ---------------- (b) labels ----------------
bx = 10.15
ax.plot([9.85, 9.85], [0.4, 8.3], color="#e3e1da", lw=1)
ax.text(bx, 8.35, "(b)", fontsize=12, weight="bold")
ax.text(bx + 0.45, 8.35, "Training labels (per training state)", fontsize=11.5, weight="bold")
labels = [("Pre-execution (dashed box)", PRE, "GPT-6 Luna sees the history, sₜ and the action\ntexts; nothing is executed"),
          ("Lookahead K = 1 / 3 (purple)", TEACH, "Luna also sees each action's sₜ₊₁ and the next K\nactor steps of draw 0 (final answer hidden)"),
          ("Observed outcome", OUTC, "best success on draws 0–2\n(here a3, 3/3)")]
for j, (title, color, text) in enumerate(labels):
    y = 7.55 - j * 0.85
    ax.add_patch(Rectangle((bx, y - 0.32), 0.06, 0.7, fc=color, ec="none"))
    ax.text(bx + 0.18, y + 0.22, title, fontsize=9.5, weight="bold", color=color)
    ax.text(bx + 0.18, y + 0.02, text, fontsize=8.3, va="top")
ax.text(bx + 0.18, 5.15, "Each gives a soft target over the actions (vote shares; ties split evenly).\nNo label uses draws 3–7.", fontsize=8.3, va="center", color=MUTED)

# ---------------- (c) train and score ----------------
ax.text(bx, 4.65, "(c)", fontsize=12, weight="bold")
ax.text(bx + 0.45, 4.65, "Train the head, score every picker", fontsize=11.5, weight="bold")
rbox(bx, 3.2, 2.55, 1.1, ec=STU, lw=1.3)
ax.text(bx + 0.12, 4.08, "Selection head (22.6 M)", fontsize=9, weight="bold", color=STU)
ax.text(bx + 0.12, 3.85, "frozen SFT-4B features of sₜ\nand each action → one score\neach; soft CE on (b), 1,380\ntrain states", fontsize=7.9, va="top")
rbox(bx + 2.85, 3.2, 2.85, 1.1, ec=HELD, lw=1.3)
ax.text(bx + 2.97, 4.08, "348 test states (86 unseen tasks)", fontsize=9, weight="bold", color=HELD)
ax.text(bx + 2.97, 3.85, "pick = top-scoring action\n(teachers: their own pick)\nscore = its success on draws 3–7", fontsize=7.9, va="top")
arrow(bx + 2.57, 3.75, bx + 2.83, 3.75, color=STU)
ax.text(bx, 2.75, "Example at left", fontsize=9, weight="bold")
rows = [("pick a3", "80%"), ("uniform (mean of a1–a4)", "45%"), ("SFT sample (actor's own frequencies)", "weighted mean"), ("oracle (best on draws 3–7)", "80%")]
for k, (lab, val) in enumerate(rows):
    y = 2.45 - k * 0.28
    ax.text(bx + 0.1, y, lab, fontsize=8.5)
    ax.text(bx + 4.2, y, val, fontsize=8.5, ha="right", color=HELD, weight="bold")
ax.text(bx, 1.0, "Average per state, then over states. Distinct view: one candidate per action\n(as at left); natural view: 5 SFT proposals at the "
        "actor's own frequencies.", fontsize=8, color=MUTED, va="top")
fig.savefig(Path(__file__).with_suffix(".png"), dpi=170, bbox_inches="tight", facecolor="white")
