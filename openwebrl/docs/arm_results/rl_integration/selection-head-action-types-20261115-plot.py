"""Action types chosen by lookahead- vs pre-execution-label heads (November 14-15 window, 3 runs x 1,045 tasks per condition):
share of decision steps by the chosen action's tool name, one panel per tool and one bar per method. Bars: all five candidates
offered to the heads (the actor's proposals), SFT 4B alone (its single sample), then each head by size for lookahead H = 4 labels
and for pre-execution labels. Data: selection-head-action-types-20261115.json."""
import collections
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
from matplotlib.patches import Patch

HERE = Path(__file__).parent
d = json.loads((HERE / "selection-head-action-types-20261115.json").read_text())
TOOLS = [("click", "click"), ("scroll", "scroll"), ("go_back", "go back"), ("done", "done"), ("new_tab", "new tab"), ("goto_url", "go to URL")]
SIZES = [("probe", "23K"), ("d128", "4.4M"), ("d512", "22.6M"), ("d4096", "1.8B")]
share = lambda c, t: 100 * c.get(t, 0) / sum(c.values())
offered = sum((collections.Counter(d[k]["offered"]) for k in d if k != "sft"), collections.Counter())
# (label, counts, face, edge, hatch)
BARS = [("offered", offered, "white", C["MUTED"]), ("SFT", collections.Counter(d["sft"]["picked"]), C["INK"], C["INK"])]
BARS += [(s, collections.Counter(d[f"{k}-lookahead"]["picked"]), C["LOOKAHEAD_TEACHER"], C["LOOKAHEAD_TEACHER"]) for k, s in SIZES]
BARS += [(s, collections.Counter(d[f"{k}-pre"]["picked"]), C["PRE_EXEC"], C["PRE_EXEC"]) for k, s in SIZES]
XS = [0, 1, 2.6, 3.6, 4.6, 5.6, 7.2, 8.2, 9.2, 10.2]
fig, axes = plt.subplots(2, 3, figsize=(7.6, 4.6))
for ax, (tool, title) in zip(axes.flat, TOOLS):
    vals = [share(c, tool) for _, c, _, _ in BARS]
    ax.bar(XS, vals, width=0.82, color=[b[2] for b in BARS], edgecolor=[b[3] for b in BARS], linewidth=1.0, zorder=3)
    ax.set_title(title, fontsize=8, pad=3)
    ax.set_xticks(XS); ax.set_xticklabels([b[0] for b in BARS], rotation=90, fontsize=5.8)
    ax.set_xlim(-0.7, 10.9); ax.set_ylim(0, max(vals) * 1.18)
    ax.grid(axis="y", color=C["BORDER"], linewidth=0.6, zorder=0)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", labelsize=6.5)
for ax in axes[:, 0]: ax.set_ylabel("% of decision steps", fontsize=7)
fig.legend(handles=[Patch(facecolor="white", edgecolor=C["MUTED"], label="all 5 candidates offered"),
                    Patch(facecolor=C["INK"], label="SFT 4B alone (1 sample)"),
                    Patch(facecolor=C["LOOKAHEAD_TEACHER"], label="heads, lookahead H = 4 labels (by size)"),
                    Patch(facecolor=C["PRE_EXEC"], label="heads, pre-execution labels (by size)")],
           loc="lower center", ncol=2, fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.07, 1, 1), h_pad=1.2, w_pad=1.0)
for ext in ("png", "pdf", "svg"):
    out = HERE / f"selection-head-action-types-20261115.{ext}"
    fig.savefig(out, dpi=220 if ext == "png" else None, bbox_inches="tight")
    if ext == "svg": out.write_text("\n".join(l.rstrip() for l in out.read_text().splitlines()) + "\n")
