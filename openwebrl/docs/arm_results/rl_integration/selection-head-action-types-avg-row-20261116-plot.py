"""Single-row full-width version; the two-row original is preserved.

Action types chosen by lookahead- vs pre-execution-label heads, averaged over head sizes (November 14-15 window): share of
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
from paper_plot_palette import COLORS as C, tint

HERE = Path(__file__).parent
d = json.loads((HERE / "selection-head-action-types-20261115.json").read_text())
TOOLS = [("click", "click"), ("scroll", "scroll"), ("go_back", "go back"), ("done", "done"), ("new_tab", "new tab"), ("goto_url", "go to URL")]
pool = lambda keys, f: sum((collections.Counter(d[k][f]) for k in keys), collections.Counter())
share = lambda c, t: 100 * c.get(t, 0) / sum(c.values())
L = [k for k in d if k.endswith("-lookahead")]; P = [k for k in d if k.endswith("-pre")]
BARS = [("offered", pool(L + P, "offered"), "white", "black"), ("SFT alone", collections.Counter(d["sft"]["picked"]), "black", "black"),
        ("lookahead", pool(L, "picked"), C["LOOKAHEAD_TEACHER"], C["LOOKAHEAD_TEACHER"]), ("pre-exec.", pool(P, "picked"), C["PRE_EXEC"], C["PRE_EXEC"])]
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, StrMethodFormatter

STEM = "selection-head-action-types-avg-row-20261116"
fig, axes = plt.subplots(1, len(TOOLS), figsize=(7.2, 1.75))
values = {}
for ax, (tool, title) in zip(axes, TOOLS):
    vals = [share(c, tool) for _, c, _, _ in BARS]
    values[tool] = vals
    ax.bar(range(4), vals, width=.72, color=[b[2] for b in BARS],
           edgecolor=[b[3] for b in BARS], linewidth=.8, zorder=3)
    for x, v in enumerate(vals):
        ax.text(x, v + max(vals)*.025, f"{v:.1f}", ha="center", va="bottom",
                fontsize=6, color=C["INK"])
    ax.set_title(title[0].upper() + title[1:], fontsize=7.5, pad=4)
    ax.set_xticks([])
    ax.set_xlim(-.65, 3.65)
    ax.set_ylim(0, max(vals)*1.22)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    ax.tick_params(axis="y", labelsize=6.5, length=2.5, pad=2)
    ax.grid(False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
axes[0].set_ylabel("Decision steps (%)", fontsize=7)
labels = ["Offered actions", "SFT actor", "Lookahead H=4", "Pre-execution"]
handles = [Patch(facecolor=fill, edgecolor=edge, linewidth=.8) for _,_,fill,edge in BARS]
legend = fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
                    fontsize=7, bbox_to_anchor=(.52, .005), handlelength=1.0,
                    handletextpad=.5, columnspacing=1.8)
fig.subplots_adjust(left=.065, right=.99, bottom=.24, top=.83, wspace=.38)
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
for ax in axes:
    assert not legend.get_window_extent(renderer).overlaps(ax.get_tightbbox(renderer))
    for text in [ax.title,ax.yaxis.label,*ax.texts,*ax.get_yticklabels()]:
        if text.get_visible() and text.get_text():
            box=text.get_window_extent(renderer)
            # Major locators can place unused ticks outside the visible range.
            if text in ax.get_yticklabels() and not ax.get_ylim()[0] <= text.get_position()[1] <= ax.get_ylim()[1]:
                continue
            assert fig.bbox.contains(*box.p0) and fig.bbox.contains(*box.p1), text.get_text()
for ext in ("png","pdf","svg"):
    path = HERE / f"{STEM}.{ext}"
    metadata = {"Date":None} if ext=="svg" else {"CreationDate":None,"ModDate":None} if ext=="pdf" else {}
    fig.savefig(path,dpi=300,metadata=metadata)
    if ext=="svg":
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines())+"\n")
(HERE / f"{STEM}.json").write_text(json.dumps(dict(
    source="selection-head-action-types-20261115.json", groups=labels,
    group_event_counts=[sum(c.values()) for _,c,_,_ in BARS],
    action_shares_percent=values,
    definition="Same pooled event counts and percentages as the original two-row average figure; unshown action categories remain in each denominator.",
    note="Each tool panel retains its own y-axis range. Counts pool decision steps across tasks, runs and heads; benchmarks and head sizes are not equally weighted.",
    dimensions_inches=[7.2,1.75]),indent=2)+"\n")
plt.close(fig)
