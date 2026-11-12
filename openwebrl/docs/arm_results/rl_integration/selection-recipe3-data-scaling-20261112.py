"""Recipe 3 GPT-5.5 H = 4 4B LoRA decoder: every data-scaling LoRA (November 8-12, 3 runs each) as the difference in 3-benchmark
mean overall success from SFT 4B alone (1 sample), versus 5-action panels in the training data. Labels: states x panels per state,
epochs; * = SFT from a different window (the November 8-9 and 10 windows ran no SFT; November 11 SFT used). Filled: trained >= ~480
optimizer steps; hollow: < ~250 steps (these collapse to answering candidate 2). Circles: 1 panel per state; diamonds: all panels
of fewer states or 4 panels per state; squares: full data in each SFT window. Bars: paired 95% bootstrap intervals.
Data: selection-recipe3-data-scaling-vs-sft-20261112.json."""
import json
import os
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", "/tmp/openwebrl-selection-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys as _s
_s.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "paper_plot_style.py").is_file())))
import paper_plot_style as _style  # noqa: F401
from paper_plot_palette import COLORS as C
from matplotlib.lines import Line2D

HERE = Path(__file__).parent
P = json.loads((HERE / "selection-recipe3-data-scaling-vs-sft-20261112.json").read_text())
COL = C["ADM_4B"]
fig, ax = plt.subplots(figsize=(5.2, 3.6))
ax.axhline(0, color=C["MUTED"], linewidth=0.9, linestyle="--", zorder=1)
ax.text(38000, -0.35, "SFT 4B alone (1 sample)", fontsize=6.5, color=C["MUTED"], ha="right", va="top")
OFF = {"s1000-pps1": 0.96, "s1000-pps1-e4": 1.04, "s250-pps1": 0.9, "s250-pps1-e4": 1.0, "s250-pps1-e16": 1.1,
       "s250": 1.04, "pps1": 0.96, "full-1111": 0.96, "full-1112": 1.04}
LAB = {"s1000-pps1": (6, 3, "left"), "s1000-pps1-e4": (6, 0, "left"), "s250-pps1": (-6, 6, "right"),
       "s250-pps1-e4": (6, 0, "left"), "s250-pps1-e16": (6, 0, "left"), "s250": (6, -3, "left"),
       "pps1": (-6, 3, "right"), "pps4": (-6, 3, "right"), "s1000": (-6, -3, "right"), "full-1111": (6, -4, "left"), "full-1112": (6, 4, "left")}
for k, r in P.items():
    pt, lo, hi = r["minus_sft_mean_overall_pp"]; x = r["panels"] * OFF.get(k, 1.0); ok = r["steps"] >= 450
    full = k.startswith("full")
    ax.errorbar(x, pt, yerr=[[pt - lo], [hi - pt]], fmt="s" if full else "o" if r["one_panel_per_state"] else "D",
                markersize=5.5 if full else 6 if r["one_panel_per_state"] else 5.5, capsize=2, linewidth=0.9,
                color=COL, markerfacecolor=COL if ok else "white", markeredgecolor=COL, zorder=3)
    dx, dy, ha = LAB[k]
    lab = r["label"].replace("full data", "full") + ("" if r["same_window"] else "*")
    ax.annotate(lab, (x, pt), textcoords="offset points", xytext=(dx, dy), ha=ha, va="center", fontsize=6, color=COL)
ax.set_xscale("log"); ax.set_xticks([250, 500, 1000, 2000, 4000, 8000, 16000]); ax.set_xticklabels(["250", "500", "1K", "2K", "4K", "8K", "16K"])
ax.minorticks_off(); ax.set_xlim(95, 40000); ax.set_ylim(-2.5, 15)
ax.set_xlabel("5-action panels in training data"); ax.set_ylabel("Mean overall − SFT 4B alone (pp)")
ax.grid(axis="y", color=C["BORDER"], linewidth=0.6, zorder=0)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
h = [Line2D([], [], color=COL, marker="o", linestyle="none", markersize=5, label="≥ ~480 optimizer steps"),
     Line2D([], [], color=COL, marker="o", markerfacecolor="white", linestyle="none", markersize=5, label="< ~250 steps (collapsed)"),
     Line2D([], [], color=C["MUTED"], marker="o", linestyle="none", markersize=5, label="1 panel per state"),
     Line2D([], [], color=C["MUTED"], marker="D", linestyle="none", markersize=4.5, label="other subsets"),
     Line2D([], [], color=C["MUTED"], marker="s", linestyle="none", markersize=4.5, label="full data")]
fig.legend(handles=h, loc="lower center", ncol=5, fontsize=6, frameon=False, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.06, 1, 1))
for ext in ("png", "pdf", "svg"):
    out = HERE / f"selection-recipe3-data-scaling-20261112.{ext}"
    fig.savefig(out, dpi=220 if ext == "png" else None, bbox_inches="tight")
    if ext == "svg": out.write_text("\n".join(l.rstrip() for l in out.read_text().splitlines()) + "\n")
