#!/usr/bin/env python3
"""Label ablation: live success of the four heads trained on our 1,380 SFT-actor branch states, which differ only in their
training labels, against one SFT 4B sample (reads selection-data-scale-20261023.json, the terra-judged three-benchmark table).

selection-label-ablation-vs-sft-20261027.png: dot plot of overall success on the three-benchmark mean, each head labelled
with its gain over SFT 4B. Error bars are the paired 95% bootstrap interval of that gain (tasks resampled within each benchmark), drawn around the head's rate, so a bar clearing the dashed
SFT line is a significant gain. Heads: joint transformer, soft CE, batch 64, last of 20 epochs, seed 0; SFT-4B actor, five
candidates, T 1.0 / top-p 0.9, 30 steps; heads run 2026-10-21, SFT 4B 2026-10-18. Online-Mind2Web judged by
gpt-5.6-terra (re-judged from the saved requests), WebVoyager and DeepShop by GPT-4o. Overall counts invalid episodes as
failures. Neither training-seed variance nor run-date drift (DeepShop) is in the bars.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
T = json.loads((HERE / "selection-data-scale-20261023.json").read_text())["three_benchmarks"]
INK, MUTED, GRID, BLUE, GRAY = "#222222", "#666666", "#e6e6e6", "#2a78d6", "#444444"
ROWS = [("SFT 4B\n(1 sample)", "SFT 4B"), ("Observed\noutcome", "Ours outcome"), ("Luna,\n3 steps after", "Ours K3"),
        ("Luna,\n1 step after", "Ours K1"), ("Luna,\nbefore execution", "Ours pre")]


# dot plot (position, not bar length, encodes the value, so the cropped axis is honest); SFT 4B is the dashed reference
# line only, so every plotted point carries an interval. Colours follow the lookahead label figure
# (lookahead-draw-split-20261013/style.py): observed outcome green, lookahead K = 3 / K = 1 dark / light blue, pre-execution purple.
COLORS = {"Ours outcome": "#1baf7a", "Ours K3": "#1f5fae", "Ours K1": "#93bdef", "Ours pre": "#7b4fd0"}
HEADS = [r for r in ROWS if r[1] != "SFT 4B"]
fig, ax = plt.subplots(figsize=(5.8, 3.6))
sft = T["SFT 4B"]["mean_overall"]
ax.axhline(sft, color=MUTED, lw=1, ls="--", zorder=0)
ax.text(-0.45, sft - 0.25, f"SFT 4B, one sample: {sft:.1f}", ha="left", va="top", fontsize=7.5, color=MUTED)
for x, (_, n) in enumerate(HEADS):
    y = T[n]["mean_overall"]; d, lo, hi = T[n]["minus_sft4b"]
    ax.errorbar(x, y, yerr=[[d - lo], [hi - d]], fmt="o", ms=8, color=COLORS[n], ecolor="#333333", capsize=3, lw=1,
                markeredgecolor="white", markeredgewidth=1, zorder=3)
    ax.text(x, y + hi - d + .3, f"{y:.1f}\n({d:+.1f})".replace("-", "\u2212"), ha="center", va="bottom", fontsize=8)
ax.set_xticks(range(len(HEADS)), [r for r, _ in HEADS], fontsize=8); ax.set_xlim(-0.5, len(HEADS) - 0.5)
ax.set_ylabel("Overall success, mean of 3 benchmarks (%)", fontsize=9)
ax.set_ylim(32, 48); ax.set_yticks(range(32, 49, 2))
ax.grid(axis="y", alpha=.25)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
fig.text(0.01, 0.01, "Values: success (gain over SFT 4B, pp). Bars: paired 95% bootstrap of the gain, drawn around the head's "
         "rate.\nOnline-Mind2Web 300 (gpt-5.6-terra), WebVoyager 595, DeepShop 150 (GPT-4o), equal weight.\n"
         "One training seed per head; heads 10-21, SFT 4B 10-18.", fontsize=6.5, color=MUTED)
fig.tight_layout(rect=(0, 0.12, 1, 1))
fig.savefig(HERE / "selection-label-ablation-vs-sft-20261027.png", dpi=160); plt.close(fig)
