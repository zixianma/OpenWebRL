"""Pre-execution vs lookahead labels across selection-head sizes (November 14-15, one window, 3 runs each): 3-benchmark mean overall
success minus SFT 4B alone (same window, 3 runs) versus head parameters, one line per label type. Bars: paired 95% bootstrap
intervals. Data: selection-head-pre-vs-lookahead-20261115.json."""
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
d = json.loads((HERE / "selection-head-pre-vs-lookahead-20261115.json").read_text())
sizes = d["sizes"]
X = [s["params"] / 1e6 for s in sizes]
fig, ax = plt.subplots(figsize=(4.6, 3.2))
for lab, col, mk, off in (("lookahead", C["LOOKAHEAD_TEACHER"], "o", 0.93), ("pre", C["PRE_EXEC"], "s", 1.075)):
    pts = [d["conditions"][f"{s['key']}-{lab}"]["minus_sft"]["mean_overall_pp"] for s in sizes]
    xs = [x * off for x in X]
    ax.plot(xs, [p[0] for p in pts], color=col, linewidth=1.2, zorder=2)
    for x, (pt, lo, hi) in zip(xs, pts):
        ax.errorbar(x, pt, yerr=[[pt - lo], [hi - pt]], fmt=mk, color=col, markersize=5.5, capsize=2, linewidth=0.9, zorder=3)
ax.axhline(0, color=C["MUTED"], linestyle="--", linewidth=0.9, zorder=0)
ax.text(2600, 0.3, "SFT 4B alone (1 sample)", fontsize=6.5, color=C["MUTED"], ha="right", va="bottom")
ax.set_xscale("log"); ax.set_xticks([0.023, 4.4, 22.6, 1770]); ax.set_xticklabels(["23K\nlinear", "4.4M", "22.6M", "1.8B"]); ax.minorticks_off()
ax.set_xlim(0.012, 3500); ax.set_ylim(-1.5, 14)
ax.set_xlabel("Head parameters"); ax.set_ylabel("Mean overall − SFT 4B alone (pp)")
ax.grid(axis="y", color=C["BORDER"], linewidth=0.6, zorder=0)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax.legend(handles=[Line2D([], [], color=C["LOOKAHEAD_TEACHER"], marker="o", linewidth=1.2, markersize=5, label="lookahead H = 4 labels"),
                   Line2D([], [], color=C["PRE_EXEC"], marker="s", linewidth=1.2, markersize=5, label="pre-execution labels")],
          fontsize=6.5, frameon=False, loc="upper right")
fig.tight_layout()
for ext in ("png", "pdf", "svg"):
    out = HERE / f"selection-head-pre-vs-lookahead-20261115.{ext}"
    fig.savefig(out, dpi=220 if ext == "png" else None, bbox_inches="tight")
    if ext == "svg": out.write_text("\n".join(l.rstrip() for l in out.read_text().splitlines()) + "\n")
