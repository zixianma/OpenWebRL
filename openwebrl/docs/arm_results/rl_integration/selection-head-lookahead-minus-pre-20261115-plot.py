"""Lookahead vs pre-execution labels across selection-head sizes (November 14-15, one window, 3 runs each): the paired difference in
3-benchmark mean overall success, lookahead H = 4 labels minus pre-execution labels (both GPT-5.5, same 18,781 panels), at each
head size (23K linear probe, 4.4M, 22.6M, 1.8B; training seed 0) and pooled over the four sizes. Bars: paired 95% bootstrap
intervals over tasks. Data: selection-head-pre-vs-lookahead-20261115.json."""
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
d = json.loads((HERE / "selection-head-pre-vs-lookahead-20261115.json").read_text())
keys = [s["key"] for s in d["sizes"]]
labels = ["23K", "4.4M", "22.6M", "1.8B", "pooled"]
diffs = [d["contrasts"][f"{k}: pre - lookahead"]["mean_overall_pp"] for k in keys] + [d["contrasts"]["all sizes: pre - lookahead"]["mean_overall_pp"]]
diffs = [(-pt, -hi, -lo) for pt, lo, hi in diffs]   # lookahead - pre
COL = C["ADM_MAIN"]
fig, ax = plt.subplots(figsize=(3.4, 2.9))
for i, (pt, lo, hi) in enumerate(diffs):
    ax.errorbar(i, pt, yerr=[[pt - lo], [hi - pt]], fmt="D" if i == 4 else "o", color=COL, markersize=6, capsize=3, linewidth=1.2, zorder=3)
ax.axhline(0, color=C["MUTED"], linestyle="--", linewidth=0.9, zorder=0)
ax.axvline(3.5, color=C["BORDER"], linewidth=0.8, linestyle=":", zorder=1)
ax.set_xticks(range(5)); ax.set_xticklabels(labels); ax.set_xlim(-0.6, 4.6); ax.set_ylim(-1, 9)
ax.set_ylabel("Lookahead − pre-execution (%)"); ax.set_xlabel("Head parameters")
ax.grid(axis="y", color=C["BORDER"], linewidth=0.6, zorder=0)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
fig.tight_layout()
for ext in ("png", "pdf", "svg"):
    out = HERE / f"selection-head-lookahead-minus-pre-20261115.{ext}"
    fig.savefig(out, dpi=220 if ext == "png" else None, bbox_inches="tight")
    if ext == "svg": out.write_text("\n".join(l.rstrip() for l in out.read_text().splitlines()) + "\n")
