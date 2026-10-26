#!/usr/bin/env python3
"""Live success of the Piotr-data heads vs number of training states, one curve per benchmark.

Reads selection-data-scale-20261023.json (written by selection-data-scale-20261023-analysis.py). Heads trained on
254 / 501 / 1,011 / 2,587 of Piotr's states (nested subsets of whole tasks, seed 0; the one-panel-per-state head is
excluded). Overall success (invalid = failure); error bars are 95% Wilson intervals per run. Dashed lines: SFT 4B, one
sample (run 10-18). Online-Mind2Web judged by gpt-5.6-terra, WebVoyager and DeepShop by GPT-4o.
"""
import json, math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
T = json.loads((HERE / "selection-data-scale-20261023.json").read_text())["three_benchmarks"]
RUNS = [("Piotr 250", 254), ("Piotr 500", 501), ("Piotr 1000", 1011), ("Piotr all 2587", 2587)]
BENCH = [("online_mind2web", "Online-Mind2Web (300)", "#2a78d6", 0.96, 5), ("webvoyager", "WebVoyager (595)", "#eb6834", 1.0, 0),
         ("deepshop", "DeepShop (150)", "#1baf7a", 1.04, -5)]   # x dodge (log-scale factor), end-label y offset (pt)


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=160)
xs = [x for _, x in RUNS]
for key, label, color, dodge, dy in BENCH:
    ys, lo, hi = [], [], []
    for name, _ in RUNS:
        r = T[name][key]; y = 100 * r["success"] / r["tasks"]; a, b = wilson(r["success"], r["tasks"])
        ys.append(y); lo.append(y - a); hi.append(b - y)
    ax.errorbar([x * dodge for x in xs], ys, yerr=[lo, hi], color=color, lw=2, marker="o", ms=7, capsize=3, elinewidth=1,
                markeredgecolor="white", markeredgewidth=1.5, label=label)
    ax.annotate(f"{ys[-1]:.1f}%", (xs[-1] * dodge, ys[-1]), xytext=(8, dy), textcoords="offset points", va="center",
                fontsize=9, color="#333333")
    s = T["SFT 4B"][key]; sft = 100 * s["success"] / s["tasks"]
    ax.axhline(sft, color=color, lw=1, ls="--", alpha=0.8)
    ax.annotate(f"SFT 4B {sft:.1f}%", (xs[0], sft), xytext=(-4, -10), textcoords="offset points", ha="right",
                fontsize=8, color="#555555")
ax.set_xscale("log"); ax.set_xticks(xs); ax.set_xticklabels([f"{x:,}" for x in xs]); ax.minorticks_off()
ax.set_xlim(150, 4200)
ax.set_xlabel("Piotr training states (log scale)"); ax.set_ylabel("Overall success (%)")
ax.set_title("Selection head trained on Piotr's panels: live success vs. training states", fontsize=11)
ax.grid(axis="y", color="#e6e6e6", lw=0.8); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=9)
fig.text(0.01, 0.005, "Error bars: 95% Wilson interval per run; points offset horizontally for legibility.\nDashed: SFT 4B, one sample (10-18). DeepShop drifts between run dates.",
         fontsize=7, color="#666666")
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(HERE / "selection-data-scale-benchmarks-20261026.png")
