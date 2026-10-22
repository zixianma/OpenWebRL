"""How labels are made and how teachers and selection heads are scored offline on SFT-actor branching states
(scripts/train_head_round.py: targets, held_rate; 2026-10-21 ablation)."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

INK, MUTED, GRID = "#2b2a27", "#6b6962", "#c3c2b7"
WIN, LOSE = "#1baf7a", "#e8e6df"
TEACH, OUTC, HELD, STU = "#6250d6", "#eda100", "#2a78d6", "#eb6834"
plt.rcParams.update({"font.size": 10, "text.color": INK})
fig = plt.figure(figsize=(15, 8.4))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 15); ax.set_ylim(0, 8.4); ax.axis("off")


def box(x, y, w, h, ec=GRID, fc="white", lw=1.3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=ec, lw=lw))


def arrow(x0, y0, x1, y1, color=MUTED):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14, color=color, lw=1.6))


def step(n, x, y, title):
    ax.add_patch(plt.Circle((x + 0.18, y), 0.17, color=INK))
    ax.text(x + 0.18, y, str(n), color="white", ha="center", va="center", fontsize=10, weight="bold")
    ax.text(x + 0.45, y, title, va="center", fontsize=12, weight="bold")


# 1. data
step(1, 0.2, 8.0, "Branching state: every action gets 8 continuations")
box(0.2, 1.0, 6.3, 6.6)
ax.text(0.45, 7.3, "State s: task + screenshot + history at one step of an SFT-4B episode", fontsize=9.5)
ax.text(0.45, 7.0, "2–5 distinct actions the SFT actor proposed there", fontsize=9, color=MUTED)
draws = [[1, 0, 1, 1, 0, 1, 1, 0], [0, 0, 1, 0, 0, 0, 1, 0], [1, 1, 1, 1, 0, 1, 1, 1], [0, 0, 0, 0, 0, 1, 0, 0]]
x0, cw, top = 1.55, 0.5, 6.3
for k in range(8):
    ax.text(x0 + k * cw + cw / 2, top + 0.25, str(k), ha="center", fontsize=9, color=MUTED)
ax.text(x0 - 0.15, top + 0.25, "draw", ha="right", fontsize=9, color=MUTED)
for i, row in enumerate(draws):
    y = top - (i + 1) * 0.6
    ax.text(x0 - 0.15, y + 0.22, f"action a{i + 1}", ha="right", va="center", fontsize=9.5)
    for k, v in enumerate(row):
        ax.add_patch(Rectangle((x0 + k * cw + 0.04, y), cw - 0.08, 0.44, fc=WIN if v else LOSE, ec="white"))
yb = top - 4 * 0.6 - 0.15
def bracket(k0, k1, color, label, dy):
    xa, xb = x0 + k0 * cw + 0.04, x0 + (k1 + 1) * cw - 0.04
    ax.plot([xa, xa, xb, xb], [yb - dy + 0.12, yb - dy, yb - dy, yb - dy + 0.12], color=color, lw=2)
    ax.text((xa + xb) / 2, yb - dy - 0.12, label, ha="center", va="top", fontsize=9, color=color, weight="bold")
bracket(0, 0, TEACH, "teacher\nevidence", 0.0)
bracket(0, 2, OUTC, "outcome label", 0.75)
bracket(3, 7, HELD, "held-out scoring (draws 3–7)", 0.0)
ax.add_patch(Rectangle((x0 + 0.02, top - 4 * 0.6 - 0.02), cw - 0.04, 4 * 0.6 + 0.04, fc="none", ec=TEACH, lw=2))
ax.text(0.45, 1.75, "Each draw executes the action in a replayed browser, then the frozen SFT actor\n"
        "continues to the end; o4-mini judges task success (green). Invalid draws are dropped.", fontsize=8.8, color=MUTED)
ax.text(0.45, 1.2, "1,738 states, 437 tasks; split by task: 1,390 train / 348 test (86 tasks).", fontsize=9.2)

# 2. labels
step(2, 6.9, 8.0, "Labels on training states")
ax.text(7.35, 7.68, "soft targets over the actions", fontsize=9, color=MUTED)
items = [("Lookahead K = 1 / K = 3", TEACH, "GPT-6 Luna sees s, every action, and the next K\nsteps of draw 0 (screenshots; final answer hidden);\nvote share over 2 randomized-order calls"),
         ("Pre-execution", TEACH, "GPT-6 Luna sees s and every action only;\nvote share over 2 calls"),
         ("Observed outcome", OUTC, "action(s) with the best success on draws 0–2;\nties split evenly.  Example: a3 (3 of 3) → target 1 on a3")]
for j, (title, color, text) in enumerate(items):
    y = 6.15 - j * 1.6
    box(6.9, y, 4.0, 1.35, ec=color, lw=1.8)
    ax.text(7.05, y + 1.1, title, fontsize=10, weight="bold", color=color)
    ax.text(7.05, y + 0.88, text, fontsize=8.6, va="top")
ax.text(7.05, 1.2, "No label ever uses draws 3–7.", fontsize=9.5, weight="bold", color=HELD)
arrow(6.5, 5.2, 6.88, 5.2)

# 3. student
step(3, 11.3, 8.0, "Student and scoring")
box(11.3, 5.6, 3.5, 2.0, ec=STU, lw=1.8)
ax.text(11.45, 7.32, "Selection head (student)", fontsize=10, weight="bold", color=STU)
ax.text(11.45, 7.05, "frozen SFT-4B hidden states of s and\neach action (layers 18/27/36)\n→ 22.6 M head → one score per action\n"
        "trained with soft CE on the 1,380\ntraining states' labels", fontsize=8.6, va="top")
arrow(10.9, 6.3, 11.28, 6.3)
box(11.3, 1.0, 3.5, 4.25, ec=HELD, lw=1.8)
ax.text(11.45, 4.97, "Score on the 348 test states", fontsize=10, weight="bold", color=HELD)
ax.text(11.45, 4.72, "pick = the head's top-scoring action\n(or the teacher's own pick)\n→ score = that action's success\n    on its valid draws 3–7", fontsize=8.8, va="top")
ax.text(11.45, 3.6, "Example (grid at left): pick a3 → 4/5 = 80%\nuniform = mean(60, 20, 80, 20) = 45%\noracle = best on draws 3–7 = 80%", fontsize=8.8, va="top", color=HELD)
ax.text(11.45, 2.55, "average per state, then over states\n(each state counts once)\nreferences on the same draws: uniform\n(mean over actions), SFT sample (actor's\nown frequencies), oracle (best action)", fontsize=8.3, va="top", color=MUTED)
arrow(13.05, 5.58, 13.05, 5.27, STU)
ax.text(0.2, 0.45, "Views: distinct-action panels (one candidate per action, as above) and natural panels (5 SFT proposals drawn at the actor's own "
        "frequencies, duplicates kept, 20 seeded panels per state). Teachers are scored the same way on the 1,037-state labelling study.",
        fontsize=8.8, color=MUTED)
fig.savefig(Path(__file__).with_suffix(".png"), dpi=150, bbox_inches="tight", facecolor="white")
