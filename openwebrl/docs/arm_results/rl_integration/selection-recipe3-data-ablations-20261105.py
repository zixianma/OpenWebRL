"""Recipe 3 GPT-5.5 K = 3 4B LoRA (ADM-4B) training-data ablations, SFT 4B actor, November 4–5. Two figures:
  selection-recipe3-data-states-20261105.*  fewer states: whole tasks until 250 / 500 / 1,000 states, all their panels
  selection-recipe3-data-panels-20261105.*  fewer panels: 1 / 2 / 4 panels sampled per state, all 2,112 states
Both end at the full data (2,112 states, 8.9 panels per state). y = equal-weight 3-benchmark mean of valid-only success, runs pooled
(3 runs; s500 and pps2 1 run, drawn hollow). Gray dashed line: SFT 4B alone (3 runs, November 6 same-window reference).
Data: selection-recipe3-data-ablations-20261105.json (selection-recipe3-data-ablations-20261105-analysis.py)."""
import json
import os
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", "/tmp/openwebrl-selection-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys as _paper_sys
from pathlib import Path as _PaperPath
_paper_sys.path.insert(0, str(next(p for p in _PaperPath(__file__).resolve().parents
                                if (p / "paper_plot_style.py").is_file())))
import paper_plot_style as _paper_style
from paper_plot_palette import COLORS as C, tint
from matplotlib.ticker import MultipleLocator

HERE = Path(__file__).parent
d = json.loads((HERE / "selection-recipe3-data-ablations-20261105.json").read_text())
runs = {k: len(v) for k, v in d["sources"].items()}
y = {k: v["mean_valid_only"] for k, v in d["rates"].items()}
ADM_GPT55 = C["ADM_4B"]
FIGS = [("selection-recipe3-data-states-20261105", "Training states", [("s250", 252), ("s500", 500), ("s1000", 1001), ("full", 2112)],
         ["252", "500", "1,001", "2,112"]),
        ("selection-recipe3-data-panels-20261105", "Panels per state", [("pps1", 1), ("pps2", 2), ("pps4", 4), ("full", 18786 / 2112)],
         ["1", "2", "4", "8.9 (all)"])]
lo = min(y["sft"], *(y[k] for _, _, pts, _ in FIGS for k, _ in pts)) - 1.5
hi = max(*(y[k] for _, _, pts, _ in FIGS for k, _ in pts)) + 1.5
for name, xlabel, pts, ticks in FIGS:
    fig, ax = plt.subplots(figsize=(3.1, 2.35))
    xs = [x for _, x in pts]; ys = [y[k] for k, _ in pts]
    ax.plot(xs, ys, color=ADM_GPT55, linewidth=1.4, zorder=2)
    for (k, x), v in zip(pts, ys):
        one = runs[k] == 1
        ax.scatter(x, v, s=30, marker="D", zorder=3, facecolor="white" if one else ADM_GPT55, edgecolor=ADM_GPT55, linewidth=1.1)
        ax.annotate(f"{v:.1f}", (x, v), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7, color=C["MUTED"])
    for key, color, ls, label in (("sft", C["BASELINE_LIGHT"], "--", "SFT 4B"),):
        ax.axhline(y[key], color=color, linestyle=ls, linewidth=0.9, zorder=1)
        ax.text(xs[-1] * 1.3, y[key] + 0.2, label, fontsize=7, color=color, va="bottom", ha="right")
    ax.set_xscale("log"); ax.set_xticks(xs); ax.set_xticklabels(ticks); ax.minorticks_off()
    ax.set_xlim(xs[0] / 1.35, xs[-1] * 1.35); ax.set_ylim(lo, hi)
    ax.set_xlabel(xlabel); ax.set_ylabel("Mean valid-only success (%)")
    ax.yaxis.set_major_locator(MultipleLocator(2))
    ax.grid(axis="y", color=C["BORDER"], linewidth=0.6, zorder=0)
    ax.tick_params(colors=C["MUTED"], length=3, width=0.6, pad=2)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        output = HERE / f"{name}.{ext}"
        fig.savefig(output, dpi=220 if ext == "png" else None, bbox_inches="tight")
        if ext == "svg":
            output.write_text("\n".join(line.rstrip() for line in output.read_text().splitlines()) + "\n")
    plt.close(fig)
