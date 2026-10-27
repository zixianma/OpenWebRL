#!/usr/bin/env python3
"""Label ablation: live success of the four heads trained on our 1,380 SFT-actor branch states, which differ only in their
training labels, against one SFT 4B sample (reads selection-data-scale-20261023.json, the terra-judged three-benchmark table).

selection-label-ablation-vs-sft-20261027.png: absolute overall success on the three-benchmark mean, each head labelled
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


fig, ax = plt.subplots(figsize=(6.6, 3.8), dpi=200)
xs = list(range(len(ROWS)))
sft = T["SFT 4B"]["mean_overall"]
ax.axhline(sft, color=MUTED, lw=1, ls="--")
for x, (_, n) in zip(xs, ROWS):
    y = T[n]["mean_overall"]; d, lo, hi = T[n]["minus_sft4b"]
    if n == "SFT 4B":
        ax.plot(x, y, "o", color=GRAY, ms=6, markeredgecolor="white", markeredgewidth=1)
        ax.annotate(f"{y:.1f}%", (x, y), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8, color=INK)
        continue
    ax.errorbar(x, y, yerr=[[d - lo], [hi - d]], fmt="o", color=BLUE, ms=6, capsize=3, elinewidth=1.2,
                markeredgecolor="white", markeredgewidth=1)
    ax.annotate(f"{d:+.1f}".replace("-", "\u2212"), (x, y + hi - d), xytext=(0, 3), textcoords="offset points",
                ha="center", fontsize=8, color=INK)
ax.set_xticks(xs); ax.set_xticklabels([r for r, _ in ROWS], fontsize=8.5); ax.set_xlim(-0.5, len(ROWS) - 0.5)
ax.set_ylabel("Overall success, mean of 3 benchmarks, %", fontsize=9); ax.set_ylim(top=ax.get_ylim()[1] + 0.8)
ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.set_title("Training labels for the selection head (1,380 branch states)", fontsize=10)
fig.text(0.01, 0.01, "Labels: gain over SFT 4B, pp. Bars: paired 95% bootstrap of that gain around the head's rate; dashed: SFT 4B.\nOnline-Mind2Web 300 (gpt-5.6-terra), WebVoyager 595, DeepShop 150 (GPT-4o). "
         "Equal weight.\nOne training seed per head; heads 10-21, SFT 4B 10-18.", fontsize=6.5, color=MUTED)
fig.tight_layout(rect=(0, 0.1, 1, 1))
fig.savefig(HERE / "selection-label-ablation-vs-sft-20261027.png"); plt.close(fig)
