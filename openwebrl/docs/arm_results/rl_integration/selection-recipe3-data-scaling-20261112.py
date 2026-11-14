"""Separate training duration from data quantity; preserve all source estimates and CIs.
Lines in panel a join runs with the same training subset, across evaluation windows;
they are visual guides rather than matched-window causal estimates. Panel b keeps
only runs with >=450 optimizer steps (observed minimum 476). Caption explains
cross-window SFT references and bootstrap uncertainty.
"""
import json
import os
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", "/tmp/openwebrl-selection-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "paper_plot_style.py").is_file())))
import paper_plot_style as _style
from paper_plot_palette import COLORS as C
HERE = Path(__file__).resolve().parent
SOURCE = HERE / "selection-recipe3-data-scaling-vs-sft-20261112.json"
P = json.loads(SOURCE.read_text())
PURPLE, LIGHT = "#8246E5", "#B69AE8"
plt.rcParams.update({"font.family": "Albert Sans", "font.size": 8,
                     "axes.labelsize": 8.5, "axes.titlesize": 8.5,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "svg.fonttype": "none", "pdf.fonttype": 42})
fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.55), sharey=True)
seen = set()
def point(ax, key, x, *, color=PURPLE, marker="o", filled=True):
    r = P[key]
    mean, lo, hi = r["minus_sft_mean_overall_pp"]
    ax.errorbar(x, mean, yerr=[[mean-lo], [hi-mean]], fmt=marker,
                markersize=4.9, capsize=2, elinewidth=.85, capthick=.85,
                color=color, markerfacecolor=color if filled else "white",
                markeredgecolor=color, markeredgewidth=1.1, zorder=3)
    seen.add(key)
    return mean
for ax in axes:
    ax.axhline(0, color=C["MUTED"], linewidth=.8, linestyle=(0,(3,3)), zorder=0)
    ax.grid(False)
    ax.set_ylim(-2.5, 15)
    ax.set_yticks([0, 5, 10, 15])
    ax.tick_params(length=3, width=.6, pad=2, colors=C["MUTED"])
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(C["MUTED"])
        ax.spines[spine].set_linewidth(.6)
a, b = axes
a.set_title("a) More training, same data", loc="left", pad=7)
for keys, color in [(["s250-pps1", "s250-pps1-e4", "s250-pps1-e16"], PURPLE),
                    (["s1000-pps1", "s1000-pps1-e4"], LIGHT)]:
    a.plot([P[k]["steps"] for k in keys], [P[k]["minus_sft_mean_overall_pp"][0] for k in keys],
           color=color, linewidth=.95, zorder=1)
    for key in keys:
        point(a, key, P[key]["steps"], color=color, filled=P[key]["steps"]>=450)
a.text(510, 10.8, "252 panels", color=PURPLE, fontsize=7.5, va="center")
a.text(510, 5.8, "1,001 panels", color=PURPLE, fontsize=7.5, va="center")
a.set_xlim(20, 690)
a.set_xticks([0, 250, 500])
a.set_xlabel("Optimizer steps")
a.set_ylabel("Mean success gain (pp)")
a.text(45, 13.9, "Hollow: ≤250 steps\nFilled: ≈480+ steps", fontsize=7, color=C["MUTED"], va="top")
b.set_title("b) Data scaling after ≈480+ steps", loc="left", pad=7)
keys = [k for k,r in P.items() if r["steps"] >= 450]
offsets = {"pps1": .91, "s250": 1.09, "full-1111": .95, "full-1112": 1.05}
for key in keys:
    r=P[key]; full=key.startswith("full")
    x=r["panels"]*offsets.get(key,1)
    y=point(b,key,x,color=C["MUTED"] if full else PURPLE,marker="s" if full else "o")
    if not r["same_window"]:
        b.annotate("*",(x,y),xytext=(5,2),textcoords="offset points",
                   fontsize=7,color=C["MUTED"],ha="left",va="bottom")
b.text(18786,14.55,"Full data",color=C["MUTED"],fontsize=7,ha="center",va="center")
b.set_xscale("log")
b.set_xticks([252,1001,4000,18786],["252","1K","4K","18.8K"])
b.minorticks_off();b.set_xlim(175,42000)
b.set_xlabel("Training panels (log)")
b.text(210, 1.1, "SFT 4B", color=C["MUTED"], fontsize=7)
fig.subplots_adjust(left=.085,right=.985,bottom=.21,top=.86,wspace=.20)
assert seen == set(P), (seen,set(P))
fig.canvas.draw()
# Check rendered labels against figure bounds; no scientific values are changed.
renderer=fig.canvas.get_renderer()
for ax in axes:
    for text in [ax.title,ax.xaxis.label,ax.yaxis.label,*ax.texts]:
        if text.get_visible() and text.get_text():
            bounds=text.get_window_extent(renderer)
            assert fig.bbox.contains(*bounds.p0) and fig.bbox.contains(*bounds.p1), text.get_text()
for ext in ("png","pdf","svg"):
    out=HERE / f"selection-recipe3-data-scaling-20261112.{ext}"
    fig.savefig(out,dpi=300,bbox_inches="tight",pad_inches=.025)
    if ext=="svg":
        out.write_text("\n".join(line.rstrip() for line in out.read_text().splitlines())+"\n")
plt.close(fig)
