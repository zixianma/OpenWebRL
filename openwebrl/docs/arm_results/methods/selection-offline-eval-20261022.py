"""Offline evaluation on SFT-actor branching states (scripts/train_head_round.py: targets, held_rate; 2026-10-21 ablation).
(a) one branching state on a timeline: 2-5 SFT-proposed actions, each executed in a replayed browser and continued 8 times by the
frozen SFT actor to a judged outcome; brackets show what each teacher reads; (b) training labels; (c) training the head and scoring
every picker on held-out draws."""
from pathlib import Path
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

INK, MUTED, LINE, GLYPH = "#26251f", "#6b6962", "#b9b7ad", "#a9a69c"
WIN = "#1a9e6e"
TEACH, PRE, OUTC, HELD, STU = "#6250d6", "#d6457e", "#d48a00", "#2a78d6", "#e05a24"
OUTC_BG, HELD_BG, TEACH_BG = "#fdf3df", "#e6effb", "#e4e0fa"
plt.rcParams.update({"font.size": 9, "text.color": INK, "font.family": "DejaVu Sans"})
W, H = 16, 9.6
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")


def rbox(x, y, w, h, ec=LINE, fc="white", lw=1.0, r=0.08, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.0,rounding_size={r}", fc=fc, ec=ec, lw=lw, zorder=z))


def arrow(x0, y0, x1, y1, color=MUTED, lw=1.1, ms=10):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw, shrinkA=0, shrinkB=0, zorder=2))


def screen(cx, cy, w, h, ec=GLYPH, lw=0.9, z=3):
    """The one glyph used for every state: a browser screenshot (title bar + content lines), centred at (cx, cy)."""
    x, y = cx - w / 2, cy - h / 2
    rbox(x, y, w, h, ec=ec, lw=lw, r=min(w, h) * 0.12, z=z)
    ax.add_patch(Rectangle((x, y + h * 0.78), w, h * 0.22, fc="#e9e7e0", ec="none", zorder=z + 0.1))
    for k, f in enumerate((0.7, 0.5)):
        ax.add_patch(Rectangle((x + w * 0.14, y + h * (0.52 - 0.24 * k)), w * f, h * 0.1, fc="#d6d3ca", ec="none", zorder=z + 0.1))


def bracket(x0, x1, y, color, text):
    ax.plot([x0, x0, x1, x1], [y - 0.1, y, y, y - 0.1], color=color, lw=1.6, solid_capstyle="round", zorder=5)
    ax.text(x0 + 0.05, y + 0.07, text, color=color, fontsize=8.8, weight="bold", va="bottom")


# ---------------- (a) one branching state ----------------
ax.text(0.15, 9.35, "(a)", fontsize=12, weight="bold")
ax.text(0.6, 9.35, "One branching state and its continuations", fontsize=11.5, weight="bold")

# timeline columns
hist_x, st_x, act_x, act_w = (0.45, 1.05), 1.95, 2.65, 1.2
cols = [4.75 + 0.55 * k for k in range(5)]          # s_{t+1} .. s_{t+5}
dots_x, end_x, cnt_x = cols[-1] + 0.5, cols[-1] + 0.95, cols[-1] + 1.25
gw, gh = 0.3, 0.15                                  # rollout glyph
row_h, gap = 0.18, 0.24
group_h = 8 * row_h
tops = [7.8 - i * (group_h + gap) for i in range(4)]
y_head = 8.0
for x, lab in [(hist_x[0], "sₜ₋₂"), (hist_x[1], "sₜ₋₁"), (st_x, "sₜ"), (act_x + act_w / 2, "action")] + \
              [(c, f"sₜ₊{'₁₂₃₄₅'[k]}") for k, c in enumerate(cols)] + [(dots_x, "…"), (end_x, "judge")]:
    ax.text(x, y_head, lab, ha="center", va="bottom", fontsize=9, color=MUTED)
ax.text(cols[0] - 0.33, y_head, "draw", ha="right", va="bottom", fontsize=7.5, color=MUTED)

# what each teacher reads: brackets over the timeline columns
bracket(hist_x[0] - 0.3, act_x + act_w, 8.35, PRE, "pre-execution teacher reads")
bracket(hist_x[0] - 0.3, cols[3] + gw / 2 + 0.05, 8.72, TEACH, "lookahead K = 3 teacher reads (draw 0 only)")

# history and current state, vertically centred
mid_all = (tops[0] + tops[-1] - group_h) / 2
screen(hist_x[0], mid_all, 0.42, 0.32)
screen(hist_x[1], mid_all, 0.42, 0.32)
arrow(hist_x[0] + 0.22, mid_all, hist_x[1] - 0.22, mid_all, ms=7, lw=0.9)
screen(st_x, mid_all, 0.62, 0.46, ec=INK, lw=1.4)
arrow(hist_x[1] + 0.22, mid_all, st_x - 0.32, mid_all, ms=7, lw=0.9)
ax.text(st_x, mid_all - 0.33, "task + history", ha="center", va="top", fontsize=7.8, color=MUTED)

actions = ['click "Search"', 'type "red shoes"', "scroll down", 'click "Deals"']
outcomes = [[1, 0, 1, 1, 0, 1, 1, 0], [0, 0, 1, 0, 0, 0, 1, 0], [1, 1, 1, 1, 0, 1, 1, 1], [0, 0, 0, 0, 0, 1, 0, 0]]
rng = random.Random(7)
for i, (name, outs) in enumerate(zip(actions, outcomes)):
    top = tops[i]; mid = top - group_h / 2
    ax.plot([st_x + 0.32, act_x], [mid_all, mid], color=LINE, lw=1.0, zorder=1)
    rbox(act_x, mid - 0.25, act_w, 0.5, ec=INK, lw=1.0)
    ax.text(act_x + act_w / 2, mid + 0.08, f"a{i + 1}", ha="center", va="center", fontsize=9, weight="bold")
    ax.text(act_x + act_w / 2, mid - 0.12, name, ha="center", va="center", fontsize=7.4, color=MUTED)
    band_x0, band_x1 = cols[0] - 0.42, end_x + 0.18
    ax.add_patch(Rectangle((band_x0, top - 3 * row_h), band_x1 - band_x0, 3 * row_h, fc=OUTC_BG, ec="none", zorder=0))
    ax.add_patch(Rectangle((band_x0, top - 8 * row_h), band_x1 - band_x0, 5 * row_h, fc=HELD_BG, ec="none", zorder=0))
    # lookahead window on draw 0: s_{t+1} .. s_{t+4}
    rbox(cols[0] - gw / 2 - 0.05, top - row_h + 0.01, cols[3] - cols[0] + gw + 0.1, row_h - 0.02, ec=TEACH, fc=TEACH_BG, lw=1.3, r=0.04, z=1)
    for d, o in enumerate(outs):
        y = top - (d + 0.5) * row_h
        ax.plot([act_x + act_w, cols[0] - gw / 2 - 0.02], [mid, y], color=LINE, lw=0.5, zorder=1)
        ax.text(cols[0] - 0.3, y, str(d), ha="right", va="center", fontsize=6.5, color=MUTED)
        n = rng.randint(2, 9)                           # states reached after the action, s_{t+1} .. s_{t+n}
        shown = min(n, len(cols))
        ax.plot([cols[0], cols[shown - 1]], [y, y], color=GLYPH, lw=0.7, zorder=2)
        for k in range(shown):
            screen(cols[k], y, gw, gh, lw=0.7)
        if n > len(cols):
            ax.text(dots_x, y, "⋯", ha="center", va="center", fontsize=8, color=MUTED)
        ax.plot([cols[shown - 1] + gw / 2 if n <= len(cols) else dots_x + 0.15, end_x - 0.12], [y, y], color=GLYPH, lw=0.6, ls=(0, (1, 2)), zorder=2)
        ax.text(end_x, y, "✓" if o else "✗", ha="center", va="center", fontsize=8.5, color=WIN if o else "#9a978d", weight="bold")
    a, h = sum(outs[:3]), sum(outs[3:])
    ax.text(cnt_x, top - 1.5 * row_h, f"{a}/3", va="center", fontsize=8.5, color=OUTC, weight="bold")
    ax.text(cnt_x, top - 5.5 * row_h, f"{h}/5 = {20 * h}%", va="center", fontsize=8.5, color=HELD, weight="bold")

# keys for the shaded regions
ky = 0.62
for x, fc, ec, text in [(0.3, TEACH_BG, TEACH, "lookahead window (draw 0)"), (3.05, OUTC_BG, OUTC, "draws 0–2: outcome labels"),
                        (5.65, HELD_BG, HELD, "draws 3–7: evaluation only")]:
    ax.add_patch(Rectangle((x, ky - 0.08), 0.3, 0.16, fc=fc, ec=ec, lw=0.8))
    ax.text(x + 0.4, ky, text, va="center", fontsize=8.5)
ax.text(0.3, 0.22, "Each draw re-opens sₜ in a fresh replayed browser, executes the action and lets the frozen SFT actor finish the task; "
        "o4-mini judges success.\nInvalid draws are dropped. K = 1 reads draw 0 up to sₜ₊₂. Example values.", fontsize=8, color=MUTED, va="center")

# ---------------- (b) labels ----------------
bx, dy = 10.15, 1.0
ax.plot([9.85, 9.85], [0.4, 9.3], color="#e3e1da", lw=1)
ax.text(bx, 8.35 + dy, "(b)", fontsize=12, weight="bold")
ax.text(bx + 0.45, 8.35 + dy, "Training labels (per training state)", fontsize=11.5, weight="bold")
labels = [("Pre-execution", PRE, "GPT-6 Luna reads the history, sₜ and the action\ntexts; nothing is executed"),
          ("Lookahead K = 1 / 3", TEACH, "Luna also reads draw 0 of every action from sₜ₊₁\nto sₜ₊₂ / sₜ₊₄ (final answer hidden)"),
          ("Observed outcome", OUTC, "best success on draws 0–2\n(here a3, 3/3)")]
for j, (title, color, text) in enumerate(labels):
    y = 7.55 + dy - j * 0.85
    ax.add_patch(Rectangle((bx, y - 0.32), 0.06, 0.7, fc=color, ec="none"))
    ax.text(bx + 0.18, y + 0.22, title, fontsize=9.5, weight="bold", color=color)
    ax.text(bx + 0.18, y + 0.02, text, fontsize=8.3, va="top")
ax.text(bx + 0.18, 5.15 + dy, "Each gives a soft target over the actions (vote shares; ties split evenly).\nNo label uses draws 3–7.", fontsize=8.3, va="center", color=MUTED)

# ---------------- (c) train and score ----------------
dy = 0.8
ax.text(bx, 4.65 + dy, "(c)", fontsize=12, weight="bold")
ax.text(bx + 0.45, 4.65 + dy, "Train the head, score every picker", fontsize=11.5, weight="bold")
rbox(bx, 3.2 + dy, 2.55, 1.1, ec=STU, lw=1.3)
ax.text(bx + 0.12, 4.08 + dy, "Selection head (22.6 M)", fontsize=9, weight="bold", color=STU)
ax.text(bx + 0.12, 3.85 + dy, "frozen SFT-4B features of sₜ\nand each action → one score\neach; soft CE on (b), 1,380\ntrain states", fontsize=7.9, va="top")
rbox(bx + 2.85, 3.2 + dy, 2.85, 1.1, ec=HELD, lw=1.3)
ax.text(bx + 2.97, 4.08 + dy, "348 test states (86 unseen tasks)", fontsize=9, weight="bold", color=HELD)
ax.text(bx + 2.97, 3.85 + dy, "pick = top-scoring action\n(teachers: their own pick)\nscore = its success on draws 3–7", fontsize=7.9, va="top")
arrow(bx + 2.57, 3.75 + dy, bx + 2.83, 3.75 + dy, color=STU)
ax.text(bx, 2.75 + dy, "Example at left", fontsize=9, weight="bold")
rows = [("pick a3", "80%"), ("uniform (mean of a1–a4)", "45%"), ("SFT sample (actor's own frequencies)", "weighted mean"), ("oracle (best on draws 3–7)", "80%")]
for k, (lab, val) in enumerate(rows):
    y = 2.45 + dy - k * 0.28
    ax.text(bx + 0.1, y, lab, fontsize=8.5)
    ax.text(bx + 4.2, y, val, fontsize=8.5, ha="right", color=HELD, weight="bold")
ax.text(bx, 1.0 + dy, "Average per state, then over states. Distinct view: one candidate per action\n(as at left); natural view: 5 SFT proposals at the "
        "actor's own frequencies.", fontsize=8, color=MUTED, va="top")
fig.savefig(Path(__file__).with_suffix(".png"), dpi=170, bbox_inches="tight", facecolor="white")
