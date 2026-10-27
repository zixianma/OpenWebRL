#!/usr/bin/env python3
"""Paired-difference scaling plot for the one-panel state scaling (reads selection-state-scale-1panel-20261027.json).

selection-state-scale-1panel-paired-20261027.png: heads trained on one GPT-5.5 panel per state (seed 0; 254 / 501 / 1,011 /
2,587 states, plus the earlier 1,393-state one-panel head, drawn hollow), success minus SFT 4B one sample on the same tasks,
one panel per benchmark plus the three-benchmark and Online-Mind2Web + WebVoyager means. Error bars: paired 95% bootstrap
over tasks (within each benchmark for means). Neither training-seed variance (one seed per head) nor run-date drift is in
the bars.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
J = json.loads((HERE / "selection-state-scale-1panel-20261027.json").read_text())
P = {p["a"]: p for p in J["paired"] if p["b"] == "sft1"}
INK, MUTED, GRID = "#222222", "#666666", "#e6e6e6"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#444444"
RUNS = [("n250", 254), ("n500", 501), ("n1000", 1011), ("n1393", 1393), ("nall", 2587)]
xs = [x for _, x in RUNS]
PANELS = [("Online-Mind2Web (300 tasks)", "online_mind2web", BLUE), ("WebVoyager (595 tasks)", "webvoyager", ORANGE),
          ("DeepShop (150 tasks)", "deepshop", AQUA), ("Mean of three benchmarks", "mean", GRAY),
          ("Mean of Online-Mind2Web + WebVoyager", "om2w_wv", GRAY)]
fig, axes = plt.subplots(1, 5, figsize=(15, 3.9), dpi=150, sharey=True)
for ax, (title, key, color) in zip(axes, PANELS):
    v = [P[k][key] for k, _ in RUNS]
    y = [p for p, _, _ in v]; lo = [p - a for p, a, _ in v]; hi = [b - p for p, _, b in v]
    ax.axhline(0, color=MUTED, lw=1, ls="--")
    ax.errorbar(xs, y, yerr=[lo, hi], color=color, lw=2, marker="o", ms=6, capsize=3, elinewidth=1,
                markeredgecolor="white", markeredgewidth=1.2)
    ax.plot([1393], [y[3]], "o", ms=6, color="white", markeredgecolor=color, markeredgewidth=1.6, zorder=5)
    for x, p, (_, _, b) in zip(xs, y, v):  # value above the upper cap, so it never sits on a bar
        ax.annotate(f"{p:+.1f}".replace("-", "\u2212"), (x, b), xytext=(0, 3), textcoords="offset points", ha="center",
                    fontsize=7.5, color=INK)
    ax.set_xscale("log"); ax.set_xticks(xs)
    ax.set_xticklabels(["254", "501", "1,011", "1,393", "2,587"], fontsize=7.5)
    for t, ha in zip(ax.get_xticklabels(), ("center", "center", "right", "left", "center")): t.set_ha(ha)
    ax.minorticks_off(); ax.set_xlim(180, 3600); ax.set_title(title, fontsize=9.5)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.set_xlabel("Piotr training states, 1 panel each", fontsize=8.5)
axes[0].set_ylabel("Success − SFT 4B, pp")
axes[1].annotate("SFT 4B", (190, 0), xytext=(2, -10), textcoords="offset points", ha="left", fontsize=7.5, color=MUTED)
fig.suptitle("Selection heads trained on one GPT-5.5 panel per Piotr state: gain over one SFT 4B sample, paired on the same tasks",
             fontsize=11)
fig.text(0.01, 0.01, "Error bars: paired 95% bootstrap over tasks (resampled within each benchmark for means). One training seed "
         "per point. Hollow: earlier one-panel head (OM2W/WV run 10-23/24, DeepShop 10-26); others 10-26/27. "
         "DeepShop vs same-night SFT 4B.", fontsize=7.5, color=MUTED)
fig.tight_layout(rect=(0, 0.04, 1, 0.95))
fig.savefig(HERE / "selection-state-scale-1panel-paired-20261027.png"); plt.close(fig)
