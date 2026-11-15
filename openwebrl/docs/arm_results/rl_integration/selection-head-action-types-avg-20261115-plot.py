"""Action types chosen by lookahead- vs pre-execution-label heads, averaged over head sizes (November 14-15 window): share of
decision steps by the chosen action's tool, one panel per tool. Bars: all five candidates offered (the actor's proposals), SFT 4B
alone (its single sample), and the four heads of each label type pooled (23K probe, 4.4M, 22.6M, 1.8B; 3 runs x 1,045 tasks each).
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

HERE = Path(__file__).parent
d = json.loads((HERE / "selection-head-action-types-20261115.json").read_text())
TOOLS = [("click", "click"), ("scroll", "scroll"), ("go_back", "go back"), ("done", "done"), ("new_tab", "new tab"), ("goto_url", "go to URL")]
pool = lambda keys, f: sum((collections.Counter(d[k][f]) for k in keys), collections.Counter())
share = lambda c, t: 100 * c.get(t, 0) / sum(c.values())
L = [k for k in d if k.endswith("-lookahead")]; P = [k for k in d if k.endswith("-pre")]
BARS = [("offered", pool(L + P, "offered"), "white", C["MUTED"]), ("SFT alone", collections.Counter(d["sft"]["picked"]), C["INK"], C["INK"]),
        ("lookahead", pool(L, "picked"), C["LOOKAHEAD_TEACHER"], C["LOOKAHEAD_TEACHER"]), ("pre-exec.", pool(P, "picked"), C["PRE_EXEC"], C["PRE_EXEC"])]
fig, axes = plt.subplots(2, 3, figsize=(6.4, 4.2))
for ax, (tool, title) in zip(axes.flat, TOOLS):
    vals = [share(c, tool) for _, c, _, _ in BARS]
    ax.bar(range(4), vals, width=0.72, color=[b[2] for b in BARS], edgecolor=[b[3] for b in BARS], linewidth=1.0, zorder=3)
    for x, v in enumerate(vals):
        ax.text(x, v + max(vals) * 0.02, f"{v:.1f}", ha="center", va="bottom", fontsize=6, color=C["INK"])
    ax.set_title(title, fontsize=8, pad=3)
    ax.set_xticks(range(4)); ax.set_xticklabels([b[0] for b in BARS], rotation=35, ha="right", fontsize=6.5)
    ax.set_ylim(0, max(vals) * 1.2); ax.grid(axis="y", color=C["BORDER"], linewidth=0.6, zorder=0)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", labelsize=6.5)
for ax in axes[:, 0]: ax.set_ylabel("% of decision steps", fontsize=7)
fig.tight_layout(h_pad=1.0, w_pad=1.0)
for ext in ("png", "pdf", "svg"):
    out = HERE / f"selection-head-action-types-avg-20261115.{ext}"
    fig.savefig(out, dpi=220 if ext == "png" else None, bbox_inches="tight")
    if ext == "svg": out.write_text("\n".join(l.rstrip() for l in out.read_text().splitlines()) + "\n")
