#!/usr/bin/env python3
"""One-panel state scaling against SFT 4B (reads selection-state-scale-1panel-20261027.json).

Absolute overall success on the y-axis; each head (seed 0; 254 / 501 / 1,011 / 2,587 states, plus the earlier 1,393-state
one-panel head, drawn hollow) is labelled with its gain over one SFT 4B sample on the same tasks, and its error bar is the
paired 95% bootstrap interval of that gain drawn around the head's rate (tasks resampled within each benchmark for the
mean), so a bar clearing the dashed SFT line is a significant gain. Neither training-seed variance nor run-date drift is in
the bars.
(1) selection-state-scale-1panel-paired-20261027.png: one panel per benchmark.
(2) selection-state-scale-1panel-mean-20261027.png: the three-benchmark mean.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
J = json.loads((HERE / "selection-state-scale-1panel-20261027.json").read_text())
P = {p["a"]: p for p in J["paired"] if p["b"] == "sft1"}
R = {r["key"]: r for r in J["rows"]}
INK, MUTED, GRID = "#222222", "#666666", "#e6e6e6"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#444444"
RUNS = [("n250", 254), ("n500", 501), ("n1000", 1011), ("n1393", 1393), ("nall", 2587)]
xs = [x for _, x in RUNS]
TICKS = ["254", "501", "1,011", "1,393", "2,587"]


def rate(k, key):
    return R[k]["mean_overall"] if key == "mean" else R[k][key]["overall"]


def draw(ax, key, color, fs=7.5):
    sft = rate("sft1", key)
    ys = [rate(k, key) for k, _ in RUNS]
    ax.axhline(sft, color=MUTED, lw=1, ls="--")
    ax.plot(xs, ys, color=color, lw=2, zorder=3)
    for x, y, (k, _) in zip(xs, ys, RUNS):
        d, lo, hi = P[k][key]; hollow = k == "n1393"
        ax.errorbar(x, y, yerr=[[d - lo], [hi - d]], fmt="o", color=color, ms=6, capsize=3, elinewidth=1, zorder=4,
                    markerfacecolor="white" if hollow else color, markeredgecolor=color if hollow else "white",
                    markeredgewidth=1.6 if hollow else 1.2)
        ax.annotate(f"{d:+.1f}".replace("-", "−"), (x, y + hi - d), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=fs, color=INK)
    ax.set_xscale("log"); ax.set_xticks(xs); ax.set_xticklabels(TICKS, fontsize=fs)
    for t, ha in zip(ax.get_xticklabels(), ("center", "center", "right", "left", "center")): t.set_ha(ha)
    ax.minorticks_off(); ax.set_xlim(180, 3600)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.set_xlabel("Piotr training states, 1 panel each", fontsize=8.5)
    return sft


PANELS = [("Online-Mind2Web (300 tasks)", "online_mind2web", BLUE), ("WebVoyager (595 tasks)", "webvoyager", ORANGE),
          ("DeepShop (150 tasks)", "deepshop", AQUA)]
fig, axes = plt.subplots(1, 3, figsize=(11, 3.9), dpi=150)
for ax, (title, key, color) in zip(axes, PANELS):
    sft = draw(ax, key, color); ax.set_title(f"{title}\nSFT 4B {sft:.1f}% (dashed)", fontsize=9.5)
axes[0].set_ylabel("Overall success, %")
fig.suptitle("Selection heads trained on one GPT-5.5 panel per Piotr state: success per benchmark, labelled with the gain "
             "over one SFT 4B sample", fontsize=10.5)
fig.text(0.01, 0.01, "Bars: paired 95% bootstrap of the gain over SFT 4B on the same tasks, drawn around the head's rate. One "
         "training seed per point.\nHollow: earlier one-panel head (OM2W/WV run 10-23/24, DeepShop 10-26); others 10-26/27. "
         "DeepShop vs same-night SFT 4B (10-27).", fontsize=7.5, color=MUTED)
fig.tight_layout(rect=(0, 0.08, 1, 0.95))
fig.savefig(HERE / "selection-state-scale-1panel-paired-20261027.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(5.2, 3.9), dpi=200)
sft = draw(ax, "mean", GRAY, fs=8.5)
ax.set_ylim(top=ax.get_ylim()[1] + 0.6)
ax.set_ylabel("Overall success, mean of 3 benchmarks, %", fontsize=9)
ax.set_title(f"Selection head, one GPT-5.5 panel per state: success vs\ntraining states, labelled with gain over SFT 4B ({sft:.1f}%, dashed)",
             fontsize=10)
fig.text(0.01, 0.01, "Online-Mind2Web 300, WebVoyager 595, DeepShop 150, equal weight. Bars: paired 95%\nbootstrap of the gain, "
         "around the head's rate; one training seed per point. Hollow: earlier\none-panel head (run 10-23/24, DeepShop 10-26).",
         fontsize=6.5, color=MUTED)
fig.tight_layout(rect=(0, 0.09, 1, 1))
fig.savefig(HERE / "selection-state-scale-1panel-mean-20261027.png"); plt.close(fig)
