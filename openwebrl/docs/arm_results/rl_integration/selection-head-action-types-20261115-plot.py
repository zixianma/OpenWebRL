"""Action types chosen by lookahead- vs pre-execution-label heads (November 14-15 window, 3 runs x 1,045 tasks per condition):
share of decision steps by the chosen action's tool name. Markers: heads pooled over the four sizes (23K probe, 4.4M, 22.6M, 1.8B)
per label type; thin bars: range over the four sizes. Gray ring: all five candidates offered to the heads (the actor's proposals);
black diamond: SFT 4B alone (its single sample). Rarer tools (press_keys, drag, switch_tab, close_tab, unparsed) are grouped as "other".
Data: selection-head-action-types-20261115.json."""
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
from matplotlib.lines import Line2D

HERE = Path(__file__).parent
d = json.loads((HERE / "selection-head-action-types-20261115.json").read_text())
TYPES = ["click", "scroll", "done", "go_back", "new_tab", "goto_url", "write", "wait", "other"]
LAB = {"go_back": "go back", "new_tab": "new tab", "goto_url": "go to URL"}


def shares(c):
    c = collections.Counter(c); t = sum(c.values()); g = collections.Counter()
    for k, v in c.items(): g[k if k in TYPES else "other"] += v
    return {k: 100 * g[k] / t for k in TYPES}


L = [k for k in d if k.endswith("-lookahead")]; P = [k for k in d if k.endswith("-pre")]
pool = lambda keys, f: sum((collections.Counter(d[k][f]) for k in keys), collections.Counter())
offered, sft = shares(pool(L + P, "offered")), shares(d["sft"]["picked"])
groups = [("lookahead H = 4 heads", L, C["LOOKAHEAD_TEACHER"], "o", -0.13), ("pre-execution heads", P, C["PRE_EXEC"], "s", 0.13)]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 3.4), gridspec_kw=dict(width_ratios=[1, 3.2]))
for ax, types, xmax in ((a1, TYPES[:1], 70), (a2, TYPES[1:], 26)):
    ys = {t: i for i, t in enumerate(types)}
    for t in types:
        y = ys[t]
        ax.plot(offered[t], y, "o", markersize=7.5, markerfacecolor="none", markeredgecolor=C["MUTED"], markeredgewidth=1.2, zorder=2)
        ax.plot(sft[t], y, "D", markersize=4.5, color=C["INK"], zorder=3)
        for name, keys, col, mk, dy in groups:
            per = [shares(d[k]["picked"])[t] for k in keys]
            ax.plot([min(per), max(per)], [y + dy] * 2, color=col, linewidth=1.0, zorder=3)
            ax.plot(shares(pool(keys, "picked"))[t], y + dy, mk, markersize=5.5, color=col, zorder=4)
    ax.set_yticks(range(len(types))); ax.set_yticklabels([LAB.get(t, t) for t in types]); ax.invert_yaxis()
    ax.set_xlim(0 if xmax < 50 else 50, xmax); ax.grid(axis="x", color=C["BORDER"], linewidth=0.6, zorder=0)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
a1.set_ylim(0.5, -0.5)
a2.set_xlabel("Share of decision steps (%)"); a1.set_xlabel("(%)")
h = [Line2D([], [], color=C["LOOKAHEAD_TEACHER"], marker="o", linewidth=1.0, markersize=5, label="lookahead H = 4 heads (range: 4 sizes)"),
     Line2D([], [], color=C["PRE_EXEC"], marker="s", linewidth=1.0, markersize=5, label="pre-execution heads (range: 4 sizes)"),
     Line2D([], [], color=C["INK"], marker="D", linestyle="none", markersize=4.5, label="SFT 4B alone (1 sample)"),
     Line2D([], [], color=C["MUTED"], marker="o", markerfacecolor="none", linestyle="none", markersize=7, label="all 5 candidates offered")]
fig.legend(handles=h, loc="lower center", ncol=2, fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.1, 1, 1), w_pad=2.0)
for ext in ("png", "pdf", "svg"):
    out = HERE / f"selection-head-action-types-20261115.{ext}"
    fig.savefig(out, dpi=220 if ext == "png" else None, bbox_inches="tight")
    if ext == "svg": out.write_text("\n".join(l.rstrip() for l in out.read_text().splitlines()) + "\n")
