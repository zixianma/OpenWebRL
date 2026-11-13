"""Selection-head model scale, offline: final-epoch agreement with the GPT-5.5 teacher's pick on 1,727 held-out validation
panels versus head parameters. Small points: each training seed (seeds 0-2 for 4.4M and 22.6M, seed 0 for the rest), with Wilson
95% intervals over panels; line: mean over seeds. Chance (uniform pick of 5) is 20%. Data: selection-head-scale-agreement-20261112.json."""
import json
import os
from collections import defaultdict
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
h = json.loads((HERE / "selection-head-scale-agreement-20261112.json").read_text())["heads"]
COL = C["ADM_22M"]
by = defaultdict(list)
for r in h.values(): by[r["params"]].append(r)
xs = sorted(by)
fig, ax = plt.subplots(figsize=(4.4, 3.1))
ax.plot([x / 1e6 for x in xs], [100 * sum(r["final_agreement"] for r in by[x]) / len(by[x]) for x in xs], color=COL, linewidth=1.2, zorder=2)
JIT = {0: 1.0, 1: 0.88, 2: 1.13}
for x in xs:
    for r in by[x]:
        p, (lo, hi) = 100 * r["final_agreement"], r["final_agreement_ci95"]
        ax.errorbar(x / 1e6 * (JIT[r["seed"]] if len(by[x]) > 1 else 1), p, yerr=[[p - 100 * lo], [100 * hi - p]], fmt="o", markersize=4.5,
                    color=COL, markerfacecolor=COL if r["seed"] == 0 else "white", capsize=1.5, linewidth=0.8, zorder=3)
ax.set_xscale("log"); ax.set_xticks([4, 10, 25, 100, 300, 1800]); ax.set_xticklabels(["4M", "10M", "25M", "100M", "300M", "1.8B"]); ax.minorticks_off()
ax.set_xlim(3, 2800); ax.set_ylim(40, 50)
ax.set_xlabel("Head parameters"); ax.set_ylabel("Agreement with teacher pick, held out (%)")
ax.grid(axis="y", color=C["BORDER"], linewidth=0.6, zorder=0)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], color=COL, marker="o", linestyle="none", markersize=4.5, label="training seed 0"),
                   Line2D([], [], color=COL, marker="o", markerfacecolor="white", linestyle="none", markersize=4.5, label="seeds 1, 2"),
                   Line2D([], [], color=COL, linewidth=1.2, label="mean over seeds")], fontsize=6.5, frameon=False, loc="upper right", ncol=3)
fig.tight_layout()
for ext in ("png", "pdf", "svg"):
    out = HERE / f"selection-head-scale-agreement-20261112.{ext}"
    fig.savefig(out, dpi=220 if ext == "png" else None, bbox_inches="tight")
    if ext == "svg": out.write_text("\n".join(l.rstrip() for l in out.read_text().splitlines()) + "\n")
