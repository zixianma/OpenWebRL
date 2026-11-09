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
# Shared paper typography, including bundled Albert Sans fonts.
import sys as _paper_sys
from pathlib import Path as _PaperPath
_paper_sys.path.insert(0, str(next(p for p in _PaperPath(__file__).resolve().parents
                                if (p / "paper_plot_style.py").is_file())))
import paper_plot_style as _paper_style
from paper_plot_palette import COLORS as C, tint


def _paper_save(fig, path, **kwargs):
    """Keep raster and vector siblings synchronized with the same layout."""
    for suffix in (".png", ".pdf", ".svg"):
        fig.savefig(path.with_suffix(suffix), **kwargs)



HERE = Path(__file__).resolve().parent
T = json.loads((HERE / "selection-data-scale-20261023.json").read_text())["three_benchmarks"]
INK, MUTED, GRID, BLUE, GRAY = C["INK"], C["MUTED"], C["BORDER"], C["ADM_22M"], C["MUTED"]
ROWS = [("SFT 4B\n(1 sample)", "SFT 4B"), ("Observed\noutcome", "Ours outcome"), ("Lookahead\nH = 4", "Ours K3"),
        ("Lookahead\nH = 2", "Ours K1"), ("Luna,\nbefore execution", "Ours pre")]


# Purple fill identifies the learned 22M ADM; the thick warm outline identifies its label source.
COLORS = {"Ours outcome": C["ORACLE"], "Ours K3": C["LOOKAHEAD_TEACHER"],
          "Ours K1": C["LOOKAHEAD_TEACHER"], "Ours pre": C["TRAIN"]}
HEADS = [r for r in ROWS if r[1] != "SFT 4B"]
fig, ax = plt.subplots(figsize=(5.8, 3.6))
sft = T["SFT 4B"]["mean_overall"]
ax.axhline(sft, color=MUTED, lw=1, ls="--", zorder=0)
ax.text(-0.45, sft - 0.5, f"SFT 4B, one sample: {sft:.1f}", ha="left", va="top", fontsize=7.5, color=MUTED)
for x, (_, n) in enumerate(HEADS):
    y = T[n]["mean_overall"]; d, lo, hi = T[n]["minus_sft4b"]
    ax.errorbar(x, y, yerr=[[d - lo], [hi - d]], fmt="o", ms=8, color=C["ADM_STUDENT"], ecolor=C["INK"], capsize=3, lw=1,
                markerfacecolor=C["ADM_STUDENT"], markeredgecolor=COLORS[n], markeredgewidth=1.7, zorder=3)
    ax.text(x, y + hi - d + .3, f"{y:.1f}\n({d:+.1f})".replace("-", "\u2212"), ha="center", va="bottom", fontsize=8)
ax.set_xticks(range(len(HEADS)), [r for r, _ in HEADS], fontsize=8); ax.set_xlim(-0.5, len(HEADS) - 0.5)
ax.set_ylabel("Overall success, mean of 3 benchmarks (%)", fontsize=9)
ax.set_ylim(25, 50); ax.set_yticks(range(25, 51, 5))  # wide range: intervals read at the scale of the rates
ax.grid(axis="y", alpha=.25)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
fig.text(0.01, 0.01, "Values: success (gain over SFT 4B, pp). Bars: paired 95% bootstrap of the gain, drawn around the head's "
         "rate.\nOnline-Mind2Web 300 (gpt-5.6-terra), WebVoyager 595, DeepShop 150 (GPT-4o), equal weight.\n"
         "One training seed per head; heads 10-21, SFT 4B 10-18.", fontsize=6.5, color=MUTED)
fig.tight_layout(rect=(0, 0.12, 1, 1))
_paper_save(fig, HERE / "selection-label-ablation-vs-sft-20261027.png", dpi=160); plt.close(fig)
