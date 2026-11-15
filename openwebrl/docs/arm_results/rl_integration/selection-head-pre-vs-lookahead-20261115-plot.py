"""Pre-execution vs lookahead labels across selection-head sizes (November 14-15, one window, 3 runs each): 3-benchmark mean overall
success minus SFT 4B alone (same window, 3 runs) versus head parameters, one line per label type. Bars: paired 95% bootstrap
intervals of each head - SFT. Right: the paired difference lookahead - pre at each size and pooled over sizes (paired bootstrap
over tasks, the test of the label effect; marginal intervals on the left can overlap while this one excludes 0).
Data: selection-head-pre-vs-lookahead-20261115.json."""
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
X = list(range(len(sizes)))   # evenly spaced, ordered by size
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.6, 3.1), gridspec_kw=dict(width_ratios=[1.15, 1]))
for lab, col, mk, off in (("lookahead", C["LOOKAHEAD_TEACHER"], "o", -0.09), ("pre", C["PRE_EXEC"], "s", 0.09)):
    pts = [d["conditions"][f"{s['key']}-{lab}"]["minus_sft"]["mean_overall_pp"] for s in sizes]
    xs = [x + off for x in X]
    ax.plot(xs, [p[0] for p in pts], color=col, linewidth=1.2, zorder=2)
    for x, (pt, lo, hi) in zip(xs, pts):
        ax.errorbar(x, pt, yerr=[[pt - lo], [hi - pt]], fmt=mk, color=col, markersize=5.5, capsize=2, linewidth=0.9, zorder=3)
ax.axhline(0, color=C["MUTED"], linestyle="--", linewidth=0.9, zorder=0)
ax.text(3.4, 0.3, "SFT 4B alone (1 sample)", fontsize=6.5, color=C["MUTED"], ha="right", va="bottom")
ax.set_xticks(X); ax.set_xticklabels(["23K", "4.4M", "22.6M", "1.8B"])
ax.set_xlim(-0.5, 3.5); ax.set_ylim(-1.5, 14)
ax.set_xlabel("Head parameters"); ax.set_ylabel("Mean overall − SFT 4B alone (%)")
ax.grid(axis="y", color=C["BORDER"], linewidth=0.6, zorder=0)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax.legend(handles=[Line2D([], [], color=C["LOOKAHEAD_TEACHER"], marker="o", linewidth=1.2, markersize=5, label="lookahead H = 4 labels"),
                   Line2D([], [], color=C["PRE_EXEC"], marker="s", linewidth=1.2, markersize=5, label="pre-execution labels")],
          fontsize=6.5, frameon=False, loc="upper right")
names = [s["key"] for s in sizes]
lab2 = ["23K", "4.4M", "22.6M", "1.8B", "pooled"]
diffs = [d["contrasts"][f"{k}: pre - lookahead"]["mean_overall_pp"] for k in names] + [d["contrasts"]["all sizes: pre - lookahead"]["mean_overall_pp"]]
diffs = [(-pt, -hi, -lo) for pt, lo, hi in diffs]   # lookahead - pre
for i, (pt, lo, hi) in enumerate(diffs):
    ax2.errorbar(i, pt, yerr=[[pt - lo], [hi - pt]], fmt="D" if i == 4 else "o", color=C["INK"], markersize=5.5, capsize=2.5, linewidth=1.0, zorder=3)
ax2.axhline(0, color=C["MUTED"], linestyle="--", linewidth=0.9, zorder=0)
ax2.axvline(3.5, color=C["BORDER"], linewidth=0.8, linestyle=":", zorder=1)
ax2.set_xticks(range(5)); ax2.set_xticklabels(lab2); ax2.set_xlim(-0.6, 4.6); ax2.set_ylim(-1.5, 9)
ax2.set_ylabel("Lookahead − pre-execution (%)"); ax2.set_xlabel("Head parameters")
ax2.grid(axis="y", color=C["BORDER"], linewidth=0.6, zorder=0)
for sp in ("top", "right"): ax2.spines[sp].set_visible(False)
fig.tight_layout(w_pad=2.5)
for ext in ("png", "pdf", "svg"):
    out = HERE / f"selection-head-pre-vs-lookahead-20261115.{ext}"
    fig.savefig(out, dpi=220 if ext == "png" else None, bbox_inches="tight")
    if ext == "svg": out.write_text("\n".join(l.rstrip() for l in out.read_text().splitlines()) + "\n")
