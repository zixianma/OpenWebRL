"""Recipe 3 GPT-5.5 H = 4 4B LoRA decoder: all same-window data-scaling results (November 8-12) as the difference in 3-benchmark
mean overall success from the full-data LoRA evaluated in the same window, versus 5-action panels in the training data.
Labels: states × panels per state, epochs. Filled: trained >= ~480 optimizer steps; hollow: < ~250 steps (these collapse to answering candidate 2). Circles: 1 panel per
state; diamonds: all panels of fewer states or 4 panels per state. Bars: paired 95% bootstrap intervals. Gray band: SFT 4B alone
(no selector) relative to the full-data LoRA in the November 11 and 12 windows. The 1,001-panel 4-epoch point's reference is the
November 11 afternoon window (same day). Data: the adjacent analysis JSONs listed in POINTS."""
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
J = lambda f: json.loads((HERE / f).read_text())
rr = J("selection-recipe3-data-ablations-rerun-20261109-analysis.json")["rates"]
s250 = J("selection-recipe3-data-s250-20261110.json")["conditions"]
b2k = J("selection-recipe3-data-below2k-20261111.json")["conditions"]
ev = J("selection-evening-20261111.json")["step_matched"]["conditions"]
st = J("selection-recipe3-data-s250-steps-20261112.json")["same_window"]["conditions"]
# (label, panels, diff [pt, lo, hi], steps, one panel per state)
POINTS = [
    ("4 / state", 7417, rr["pps4"]["diff_mean_overall_vs_full_pp"], 1700, False),
    ("1,000 states", 8875, rr["s1000"]["diff_mean_overall_vs_full_pp"], 2000, False),
    ("250 states", 2234, s250["arm5-d6c492ad4c40"]["diff_mean_overall_pp"], 482, False),
    ("1 / state", 2112, rr["pps1"]["diff_mean_overall_vs_full_pp"], 489, True),
    ("1K × 1, 2 ep", 1001, b2k["arm5-5d57cf0e0de4"]["diff_mean_overall_pp"], 238, True),
    ("1K × 1, 4 ep", 1001, ev["s1000-pps1, 4 epochs"]["minus full: mean_overall_pp"], 476, True),
    ("250 × 1, 2 ep", 252, b2k["arm5-c5ec9b4862f6"]["diff_mean_overall_pp"], 62, True),
    ("250 × 1, 4 ep", 252, st["s250-pps1, 4 epochs"]["minus full: mean_overall_pp"], 124, True),
    ("250 × 1, 16 ep", 252, st["s250-pps1, 16 epochs"]["minus full: mean_overall_pp"], 495, True),
]
sft = [b2k["sft1"]["diff_mean_overall_pp"][0], st["SFT 4B alone"]["minus full: mean_overall_pp"][0]]
COL = C["ADM_4B"]
fig, ax = plt.subplots(figsize=(5.2, 3.6))
ax.axhspan(min(sft), max(sft), color=C["BORDER"], alpha=0.6, zorder=0, linewidth=0)
ax.text(19000, min(sft) - 0.3, "SFT 4B alone", fontsize=6.5, color=C["MUTED"], ha="right", va="top")
ax.axhline(0, color=COL, linewidth=0.9, linestyle="--", zorder=1)
ax.text(21000, 3.9, "dashed: full data (18.8K panels)", fontsize=6.5, color=COL, ha="right", va="top")
OFF = {"1K × 1, 2 ep": 0.96, "1K × 1, 4 ep": 1.04, "250 × 1, 2 ep": 0.9, "250 × 1, 4 ep": 1.0,
       "250 × 1, 16 ep": 1.1, "250 states": 1.04, "1 / state": 0.96}
LAB = {"1K × 1, 2 ep": (-6, -3, "right"), "1K × 1, 4 ep": (6, -9, "left"), "250 × 1, 2 ep": (-6, 3, "right"),
       "250 × 1, 4 ep": (6, -8, "left"), "250 × 1, 16 ep": (6, 4, "left"), "250 states": (6, 4, "left"),
       "1 / state": (-6, 4, "right"), "4 / state": (-6, 5, "right"), "1,000 states": (6, -9, "left")}
for lab, x, d, steps, one in POINTS:
    pt, lo, hi = d; x *= OFF.get(lab, 1.0); ok = steps >= 450
    ax.errorbar(x, pt, yerr=[[pt - lo], [hi - pt]], fmt="o" if one else "D", markersize=6 if one else 5.5, capsize=2, linewidth=0.9,
                color=COL, markerfacecolor=COL if ok else "white", markeredgecolor=COL, zorder=3)
    dx, dy, ha = LAB[lab]
    ax.annotate(f"{lab}\n{pt:+.1f}", (x, pt), textcoords="offset points", xytext=(dx, dy), ha=ha, va="center", fontsize=6, color=COL, linespacing=1.0)
ax.set_xscale("log"); ax.set_xticks([250, 500, 1000, 2000, 4000, 8000, 16000]); ax.set_xticklabels(["250", "500", "1K", "2K", "4K", "8K", "16K"])
ax.minorticks_off(); ax.set_xlim(95, 22000); ax.set_ylim(-13.5, 4.5)
ax.set_xlabel("5-action panels in training data"); ax.set_ylabel("Mean overall − full-data LoRA, same window (pp)")
ax.grid(axis="y", color=C["BORDER"], linewidth=0.6, zorder=0)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
h = [Line2D([], [], color=COL, marker="o", linestyle="none", markersize=5, label="≥ ~480 optimizer steps"),
     Line2D([], [], color=COL, marker="o", markerfacecolor="white", linestyle="none", markersize=5, label="< ~250 steps (collapsed)"),
     Line2D([], [], color=C["MUTED"], marker="o", linestyle="none", markersize=5, label="1 panel per state"),
     Line2D([], [], color=C["MUTED"], marker="D", linestyle="none", markersize=4.5, label="other subsets")]
fig.legend(handles=h, loc="lower center", ncol=4, fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.06, 1, 1))
for ext in ("png", "pdf", "svg"):
    out = HERE / f"selection-recipe3-data-scaling-20261112.{ext}"
    fig.savefig(out, dpi=220 if ext == "png" else None, bbox_inches="tight")
    if ext == "svg": out.write_text("\n".join(l.rstrip() for l in out.read_text().splitlines()) + "\n")
