#!/usr/bin/env python3
"""Teacher-data-scaling figures against SFT 4B (reads selection-data-scale-20261023.json).

Absolute overall success on the y-axis; each head is labelled with its gain over one SFT 4B sample on the same tasks, and
its error bar is the paired 95% bootstrap interval of that gain drawn around the head's rate (tasks resampled within each
benchmark for the mean), so a bar clearing the dashed SFT line is a significant gain.
(1) selection-data-scale-paired-20261026.png: Piotr-data heads (seed 0; 254 / 501 / 1,011 / 2,587 states, every panel;
    one-panel head excluded), one panel per benchmark.
(2) selection-heads-vs-sft-20261026.png: the data-scaling heads only (254 / 501 / 1,011 / 2,587 states, every panel; 1,393
    states with one panel each, hollow) on the three-benchmark mean.
Neither interval includes training-seed variance (one seed per head) or run-date drift (DeepShop).
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
T = json.loads((HERE / "selection-data-scale-20261023.json").read_text())["three_benchmarks"]
INK, MUTED, GRID = "#222222", "#666666", "#e6e6e6"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#444444"


def style(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)


def gain(ax, x, y, d, lo, hi, color, marker="o", fs=8, hollow=False):
    ax.errorbar(x, y, yerr=[[d - lo], [hi - d]], fmt=marker, color=color, ms=6, capsize=3, elinewidth=1,
                markeredgecolor=color if hollow else "white", markerfacecolor="white" if hollow else color,
                markeredgewidth=1.6 if hollow else 1.2, zorder=4)
    ax.annotate(f"{d:+.1f}".replace("-", "−"), (x, y + hi - d), xytext=(0, 3), textcoords="offset points",
                ha="center", fontsize=fs, color=INK)


# ---- (1) per benchmark, absolute
RUNS = [("Piotr 250", 254), ("Piotr 500", 501), ("Piotr 1000", 1011), ("Piotr all 2587", 2587)]
xs = [x for _, x in RUNS]
PANELS = [("Online-Mind2Web (300 tasks)", "online_mind2web", BLUE), ("WebVoyager (595 tasks)", "webvoyager", ORANGE),
          ("DeepShop (150 tasks)", "deepshop", AQUA)]
fig, axes = plt.subplots(1, 3, figsize=(11, 3.9), dpi=150)
for ax, (title, b, color) in zip(axes, PANELS):
    sft = T["SFT 4B"][b]["overall"]
    ys = [T[n][b]["overall"] for n, _ in RUNS]
    ax.axhline(sft, color=MUTED, lw=1, ls="--")
    ax.plot(xs, ys, color=color, lw=2, zorder=3)
    for x, y, (n, _) in zip(xs, ys, RUNS):
        gain(ax, x, y, *T[n]["minus_sft4b_by_benchmark"][b], color)
    ax.set_xscale("log"); ax.set_xticks(xs); ax.set_xticklabels(["254", "501", "1,011", "2,587"], fontsize=8)
    ax.minorticks_off(); ax.set_xlim(180, 3600); ax.set_title(f"{title}\nSFT 4B {sft:.1f}% (dashed)", fontsize=9.5); style(ax)
    ax.set_xlabel("Piotr training states (every panel)", fontsize=8.5)
axes[0].set_ylabel("Overall success, %")
fig.suptitle("Selection heads trained on Piotr's panels: success per benchmark, labelled with the gain over one SFT 4B sample",
             fontsize=11)
fig.text(0.01, 0.01, "Bars: paired 95% bootstrap of the gain over SFT 4B on the same tasks, drawn around the head's rate; dashed: "
         "SFT 4B. One training seed per point;\nheads run 10-23/24, SFT 4B 10-18 (DeepShop drifts between dates). "
         "Online-Mind2Web judged by gpt-5.6-terra, the others by GPT-4o.", fontsize=7.5, color=MUTED)
fig.tight_layout(rect=(0, 0.08, 1, 0.95))
fig.savefig(HERE / "selection-data-scale-paired-20261026.png"); plt.close(fig)

# ---- (2) data-scaling heads only (label-ablation heads have their own figure,
# selection-label-ablation-vs-sft-20261027.png), three-benchmark mean, absolute; wide y range (25-50%) so the intervals
# read at the scale of the success rates
ROWS = [("254 states", "Piotr 250"), ("501 states", "Piotr 500"), ("1,011 states", "Piotr 1000"),
        ("2,587 states\n(all)", "Piotr all 2587"), ("1,393 states,\none panel each", "Piotr 1380 one-panel")]
fig, ax = plt.subplots(figsize=(5.8, 3.6))
sft = T["SFT 4B"]["mean_overall"]
ax.axhline(sft, color=MUTED, lw=1, ls="--", zorder=0)
ax.text(-0.45, sft - 0.5, f"SFT 4B, one sample: {sft:.1f}", ha="left", va="top", fontsize=7.5, color=MUTED)
for x, (_, n) in enumerate(ROWS):
    y = T[n]["mean_overall"]; d, lo, hi = T[n]["minus_sft4b"]; hollow = "one-panel" in n
    ax.errorbar(x, y, yerr=[[d - lo], [hi - d]], fmt="o", ms=8, color=BLUE, ecolor="#333333", capsize=3, lw=1, zorder=3,
                markerfacecolor="white" if hollow else BLUE, markeredgecolor=BLUE if hollow else "white",
                markeredgewidth=1.6 if hollow else 1)
    ax.text(x, y + hi - d + .4, f"{y:.1f}\n({d:+.1f})".replace("-", "\u2212"), ha="center", va="bottom", fontsize=8)
ax.axvline(3.5, color=GRID, lw=1)
ax.set_xticks(range(len(ROWS)), [r for r, _ in ROWS], fontsize=8); ax.set_xlim(-0.5, len(ROWS) - 0.5)
ax.set_ylabel("Overall success, mean of 3 benchmarks (%)", fontsize=9)
ax.set_ylim(25, 50); ax.set_yticks(range(25, 51, 5)); style(ax)
fig.text(0.01, 0.01, "Heads trained on Piotr's GPT-5.5 panels (every panel unless noted), seed 0. Values: success (gain over SFT 4B, "
         "pp).\nBars: paired 95% bootstrap of the gain, drawn around the head's rate. Online-Mind2Web 300 (gpt-5.6-terra),\n"
         "WebVoyager 595, DeepShop 150 (GPT-4o), equal weight. Heads 10-23/24, SFT 4B 10-18 (DeepShop drifts).",
         fontsize=6.5, color=MUTED)
fig.tight_layout(rect=(0, 0.13, 1, 1))
fig.savefig(HERE / "selection-heads-vs-sft-20261026.png", dpi=160); plt.close(fig)
