"""Selection-head model scale, every eval run pooled (November 11-12). Left: 3-benchmark mean overall success minus SFT 4B alone
(pooled runs of both; bars: paired 95% bootstrap intervals). Right: mean selection time per decision step, with the 4B LoRA
decoder for reference. Training seed 0 for every head. Data: selection-head-scale-pooled-20261112.json and
selection-head-scale-cost-20261112.json."""
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

HERE = Path(__file__).parent
d = json.loads((HERE / "selection-head-scale-pooled-20261112.json").read_text())["conditions"]
cost = json.loads((HERE / "selection-head-scale-cost-20261112.json").read_text())
H = [k for k in d if k != "sft1"]
X = [d[k]["params"] / 1e6 for k in H]
COL = C["ADM_22M"]
TICKS = ([4, 10, 25, 100, 300, 1800], ["4M", "10M", "25M", "100M", "300M", "1.8B"])
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.1))
pts = [d[k]["minus_sft_mean_overall_pp"] for k in H]
a1.plot(X, [p[0] for p in pts], color=COL, linewidth=1.2, zorder=2)
for k, x, (pt, lo, hi) in zip(H, X, pts):
    a1.errorbar(x, pt, yerr=[[pt - lo], [hi - pt]], fmt="o", color=COL, markersize=5.5, capsize=2, linewidth=0.9, zorder=3,
                markerfacecolor=COL if k != "head5-91e7a5a67417" else "white")
a1.axhline(0, color=C["MUTED"], linestyle="--", linewidth=0.9, zorder=0)
a1.text(2500, 0.3, "SFT 4B alone (1 sample)", fontsize=6.5, color=C["MUTED"], ha="right", va="bottom")
a1.set_ylim(-1.5, 12.5); a1.set_ylabel("Mean overall − SFT 4B alone (pp)")
a1.text(22.56, d["head5-91e7a5a67417"]["minus_sft_mean_overall_pp"][1] - 0.4, "default", fontsize=6.5, color=COL, ha="center", va="top")
ms = [1e3 * cost[k]["selection_seconds_mean"] for k in H]
a2.plot(X, ms, color=COL, linewidth=1.2, marker="o", markersize=5.5, zorder=3)
dec = 1e3 * cost["lora-full"]["selection_seconds_mean"]
a2.axhline(dec, color=C["ADM_4B"], linestyle="--", linewidth=0.9, zorder=1)
a2.text(3.2, dec * 1.12, "4B LoRA decoder", fontsize=6.5, color=C["ADM_4B"], ha="left", va="bottom")
a2.set_yscale("log"); a2.set_yticks([10, 30, 100, 300, 1000]); a2.set_yticklabels(["10", "30", "100", "300", "1,000"]); a2.set_ylim(8, 1500)
a2.set_ylabel("Selection time per step, mean (ms)")
for ax in (a1, a2):
    ax.set_xscale("log"); ax.set_xticks(TICKS[0]); ax.set_xticklabels(TICKS[1]); ax.minorticks_off(); ax.set_xlim(3, 2800)
    ax.set_xlabel("Head parameters"); ax.grid(axis="y", color=C["BORDER"], linewidth=0.6, zorder=0)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
fig.tight_layout(w_pad=2.5)
for ext in ("png", "pdf", "svg"):
    out = HERE / f"selection-head-scale-pooled-20261112.{ext}"
    fig.savefig(out, dpi=220 if ext == "png" else None, bbox_inches="tight")
    if ext == "svg": out.write_text("\n".join(l.rstrip() for l in out.read_text().splitlines()) + "\n")
