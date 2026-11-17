#!/usr/bin/env python3
"""Paper comparison at 4.4M, 22.6M and 1.8B: gains over SFT and paired
lookahead minus pre-execution differences, with task-paired 95% intervals.
Three runs per condition in the November 14–15 window, with contemporaneous
SFT controls. The figure shows per-size estimates only. Valid-only success (successes / valid episodes,
equal-weight 3-benchmark mean; 2026-11-16 user request), so episodes whose site failed to load (e.g. Wolfram Alpha) do not count.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys as _paper_sys
_paper_sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "paper_plot_style.py").is_file())))
import paper_plot_style as _paper_style  # noqa: F401
from paper_plot_palette import COLORS as C
from matplotlib.lines import Line2D

plt.rcParams.update({"font.family": "Albert Sans", "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
HERE = Path(__file__).resolve().parent
D = json.loads((HERE / "selection-head-pre-vs-lookahead-20261115.json").read_text())
INK, MUTED = C["INK"], C["MUTED"]
KEYS = ["d128", "d512", "d4096"]
SIZES = ["4M", "23M", "1.8B"]
LOOK, PRE = C["LOOKAHEAD_TEACHER"], C["PRE_EXEC"]
PURPLES = [C["ADM_MAIN"]] * len(KEYS)  # one purple for the paired label effect


def point(ax, x, v, color, marker="o", label_dy=0.15, label_dx=0, label=True, ms=5.0, bold=False):
    d, lo, hi = v
    ax.errorbar(x, d, yerr=[[d - lo], [hi - d]], fmt=marker, ms=ms, color=color, ecolor=C["BASELINE_DARK"], capsize=2, lw=1.05,
                zorder=3, markeredgewidth=1.0)
    if label:
        ax.text(x + label_dx, hi + label_dy, f"{d:+.1f}".replace("-", "−"), ha="center", va="bottom", fontsize=6.5 if bold else 6.3, color=INK, fontweight="bold" if bold else "normal")


fig, (a1, a2) = plt.subplots(1, 2, figsize=(3.5, 2.25), gridspec_kw=dict(width_ratios=[1, 1], wspace=0.38))
for x, k in enumerate(KEYS):
    point(a1, x - 0.21, D["conditions"][f"{k}-pre"]["minus_sft"]["mean_valid_only_pp"], PRE, label_dx=-.05)
    point(a1, x + 0.21, D["conditions"][f"{k}-lookahead"]["minus_sft"]["mean_valid_only_pp"], LOOK, label_dx=.05)
diffs = [D["contrasts"][f"{k}: pre - lookahead"]["mean_valid_only_pp"] for k in KEYS]
for x, (pt, lo, hi) in enumerate(diffs):
    point(a2, x, (-pt, -hi, -lo), PURPLES[x])
for ax, title, ticks in ((a1, "Online: mean success", SIZES), (a2, "Lookahead benefit", SIZES)):
    ax.axhline(0, color=MUTED, lw=1, ls="--", zorder=0)
    ax.set_title(title, fontsize=7.0, loc="left", color=INK, pad=6)
    ax.set_xticks(range(len(ticks)), ticks, fontsize=6.7)
    ax.set_xlim(-0.5, len(ticks) - 0.5)
    ax.tick_params(axis="y", labelsize=7); ax.tick_params(colors=INK); ax.grid(False)
    for side in ("top", "right"): ax.spines[side].set_visible(False)
a1.set_ylim(-1, 16); a1.set_yticks([0, 4, 8, 12, 16]); a1.set_ylabel("Gain over SFT (%)", fontsize=8, color=INK)
a2.set_ylim(-1, 9.5); a2.set_yticks([0, 2, 4, 6, 8]); a2.set_ylabel("Gain over pre-exec (%)", fontsize=8, color=INK)
a1.text(.98, .12, "SFT = 37.6%", transform=a1.transAxes,
        fontsize=6.2, color=MUTED, ha="right", va="bottom",
        bbox=dict(facecolor="white", edgecolor="none", pad=.4))
a1.set_xlabel("ADM-Head size", fontsize=7.2, color=INK); a2.set_xlabel("ADM-Head size", fontsize=7.2, color=INK)
handles = [Line2D([], [], marker="o", ls="", color=PRE, ms=5.0), Line2D([], [], marker="o", ls="", color=LOOK, ms=5.0),
           Line2D([], [], marker="o", ls="", color=PURPLES[1], ms=5.0)]
fig.legend(handles, ["Pre-exec", "Lookahead H=4", "Lookahead − pre-exec"], frameon=False, fontsize=6.5,
           loc="lower center", ncol=3, bbox_to_anchor=(0.52, 0.005), columnspacing=1.2, handletextpad=0.3)
fig.subplots_adjust(left=0.15, right=0.985, top=0.81, bottom=0.29)
fig.canvas.draw()
r = fig.canvas.get_renderer()
for artist in [*fig.legends, *[t for ax in (a1, a2) for t in [ax.title, ax.xaxis.label, ax.yaxis.label, *ax.texts, *ax.get_xticklabels(), *ax.get_yticklabels()]]]:
    if artist.get_visible() and (not hasattr(artist, "get_text") or artist.get_text()):
        b = artist.get_window_extent(r)
        assert fig.bbox.contains(*b.p0) and fig.bbox.contains(*b.p1), str(artist)
out = HERE / "selection-head-pre-vs-lookahead-paper-4m-1p8b-20261115"
fig.savefig(out.with_suffix(".png"), dpi=300)
fig.savefig(out.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None})
fig.savefig(out.with_suffix(".svg"), metadata={"Date": None})
svg = out.with_suffix(".svg"); svg.write_text("\n".join(l.rstrip() for l in svg.read_text().splitlines()) + "\n")
