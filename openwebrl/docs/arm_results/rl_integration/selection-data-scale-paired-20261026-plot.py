#!/usr/bin/env python3
"""Paired-difference versions of the teacher-data-scaling figures (reads selection-data-scale-20261023.json).

(1) selection-data-scale-paired-20261026.png: Piotr-data heads (seed 0; 254 / 501 / 1,011 / 2,587 states; one-panel
    head excluded), success minus SFT 4B one sample on the same tasks, one panel per benchmark plus the three-benchmark
    and Online-Mind2Web + WebVoyager means. Error bars: paired 95% bootstrap over tasks (within each benchmark for means).
(2) selection-heads-vs-sft-20261026.png: every head in the table, gain over SFT 4B on the three-benchmark mean and on the
    Online-Mind2Web + WebVoyager mean, same paired intervals.
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


def style(ax):
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)


# ---- (1) scaling curves as paired differences
RUNS = [("Piotr 250", 254), ("Piotr 500", 501), ("Piotr 1000", 1011), ("Piotr all 2587", 2587)]
xs = [x for _, x in RUNS]
PANELS = [("Online-Mind2Web (300 tasks)", lambda r: r["minus_sft4b_by_benchmark"]["online_mind2web"], BLUE),
          ("WebVoyager (595 tasks)", lambda r: r["minus_sft4b_by_benchmark"]["webvoyager"], ORANGE),
          ("DeepShop (150 tasks)", lambda r: r["minus_sft4b_by_benchmark"]["deepshop"], AQUA),
          ("Mean of three benchmarks", lambda r: r["minus_sft4b"], GRAY),
          ("Mean of Online-Mind2Web + WebVoyager", lambda r: r["minus_sft4b_om2w_wv"], GRAY)]
fig, axes = plt.subplots(1, 5, figsize=(15, 3.9), dpi=150, sharey=True)
for ax, (title, get, color) in zip(axes, PANELS):
    v = [get(T[n]) for n, _ in RUNS]
    y = [p for p, _, _ in v]; lo = [p - a for p, a, _ in v]; hi = [b - p for p, _, b in v]
    ax.axhline(0, color=MUTED, lw=1, ls="--")
    ax.errorbar(xs, y, yerr=[lo, hi], color=color, lw=2, marker="o", ms=6, capsize=3, elinewidth=1,
                markeredgecolor="white", markeredgewidth=1.2)
    for x, p in zip(xs, y):
        ax.annotate(f"{p:+.1f}", (x, p), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8, color=INK)
    ax.set_xscale("log"); ax.set_xticks(xs); ax.set_xticklabels(["254", "501", "1,011", "2,587"], fontsize=8)
    ax.minorticks_off(); ax.set_xlim(180, 3600); ax.set_title(title, fontsize=9.5); style(ax)
    ax.set_xlabel("Piotr training states", fontsize=8.5)
axes[0].set_ylabel("Success − SFT 4B, pp")
axes[0].annotate("SFT 4B", (190, 0), xytext=(0, 3), textcoords="offset points", fontsize=7.5, color=MUTED)
fig.suptitle("Selection heads trained on Piotr's panels: gain over one SFT 4B sample, paired on the same tasks", fontsize=11)
fig.text(0.01, 0.01, "Error bars: paired 95% bootstrap over tasks (resampled within each benchmark for means). "
         "One training seed per point; DeepShop drifts between run dates. Neither is in the bars.", fontsize=7.5, color=MUTED)
fig.tight_layout(rect=(0, 0.04, 1, 0.95))
fig.savefig(HERE / "selection-data-scale-paired-20261026.png"); plt.close(fig)

# ---- (2) every head: gain over SFT 4B, dot plot
ROWS = [("Piotr's panels, 254 states", "Piotr 250"), ("Piotr's panels, 501 states", "Piotr 500"),
        ("Piotr's panels, 1,011 states", "Piotr 1000"), ("Piotr's panels, 2,587 states", "Piotr all 2587"),
        ("Piotr's panels, 1,393 states, one panel each", "Piotr 1380 one-panel"),
        ("Earlier Piotr-data head (2,692 states)", "Piotr-data head (listwise, old)"), ("Piotr SelectionARM", "Piotr ARM"),
        ("Ours, K = 3 labels", "Ours K3"), ("Ours, observed outcome", "Ours outcome"), ("Ours, K = 1 labels", "Ours K1"),
        ("Ours, pre-execution labels", "Ours pre")]
SER = [("Mean of three benchmarks", "minus_sft4b", BLUE, -0.17, "o"),
       ("Mean of Online-Mind2Web + WebVoyager", "minus_sft4b_om2w_wv", ORANGE, 0.17, "D")]
fig, ax = plt.subplots(figsize=(8.6, 5.6), dpi=150)
ys = list(range(len(ROWS)))[::-1]
for label, key, color, dy, m in SER:
    v = [T[n][key] for _, n in ROWS]
    ax.errorbar([p for p, _, _ in v], [y + dy for y in ys], xerr=[[p - a for p, a, _ in v], [b - p for p, _, b in v]],
                fmt=m, color=color, ms=6, capsize=2.5, elinewidth=1.2, markeredgecolor="white", markeredgewidth=1, label=label)
ax.axvline(0, color=MUTED, lw=1, ls="--")
for y in (ys[4] - 0.5, ys[6] - 0.5): ax.axhline(y, color=GRID, lw=1)
ax.set_yticks(ys); ax.set_yticklabels([r for r, _ in ROWS], fontsize=8.5)
ax.set_xlabel("Success − SFT 4B one sample, pp (paired 95% bootstrap)")
ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.annotate("SFT 4B", (0, ys[0] + 0.55), xytext=(3, 0), textcoords="offset points", fontsize=7.5, color=MUTED)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.4, -0.1), ncol=2, fontsize=8.5)
ax.set_title("Live gain of each selection head over one SFT 4B sample", fontsize=11)
fig.text(0.01, 0.01, "Tasks: Online-Mind2Web 300, WebVoyager 595, DeepShop 150. One training seed per head; run dates differ "
         "(DeepShop drifts).", fontsize=7.5, color=MUTED)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(HERE / "selection-heads-vs-sft-20261026.png"); plt.close(fig)
