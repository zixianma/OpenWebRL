#!/usr/bin/env python3
"""Variant with sizes 4.4M, 23M and 1.8B (23K left out) of the paper version of selection-head-pre-vs-lookahead-20261115 in the style of selection-label-setups-with-head-sizes-20261115 (column
width, Albert Sans, gray interval bars, value labels, shared legend). Left: each student ADM's online 3-benchmark mean overall
success minus SFT 4B alone, lookahead H = 4 vs pre-execution labels (GPT-5.5, full data), at three head sizes (23K, 22.6M, 1.8B;
the 4.4M head is dropped because its seed-0 lookahead head is a lucky seed). Right: the paired difference lookahead -
pre-execution per size and pooled over the three sizes (selection-head-pre-vs-lookahead-3sizes-4m-1p8b-20261115.json). One window (November 14-15), 3 runs each; paired bootstrap over tasks.
Data: selection-head-pre-vs-lookahead-20261115.json."""
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
D3 = json.loads((HERE / "selection-head-pre-vs-lookahead-3sizes-4m-1p8b-20261115.json").read_text())
LOOK, PRE = C["LOOKAHEAD_TEACHER"], C["PRE_EXEC"]
PURPLES = ["#C3AFF6", C["ADM_MAIN"], "#6A4BC2", "#4B3191"]   # light -> dark with size; pooled darkest


def point(ax, x, v, color, marker="o", label_dy=0.15, label=True, ms=4.1, bold=False):
    d, lo, hi = v
    ax.errorbar(x, d, yerr=[[d - lo], [hi - d]], fmt=marker, ms=ms, color=color, ecolor=C["BASELINE_DARK"], capsize=1.5, lw=0.75,
                zorder=3, markeredgewidth=1.0)
    if label:
        ax.text(x, hi + label_dy, f"{d:+.1f}".replace("-", "−"), ha="center", va="bottom", fontsize=6.2 if bold else 5.6, color=INK, fontweight="bold" if bold else "normal")


fig, (a1, a2) = plt.subplots(1, 2, figsize=(3.7, 2.3), gridspec_kw=dict(width_ratios=[1, 1.1], wspace=0.40))
for x, k in enumerate(KEYS):
    point(a1, x - 0.15, D["conditions"][f"{k}-lookahead"]["minus_sft"]["mean_overall_pp"], LOOK, label=False)
    point(a1, x + 0.15, D["conditions"][f"{k}-pre"]["minus_sft"]["mean_overall_pp"], PRE, label=False)
diffs = [D["contrasts"][f"{k}: pre - lookahead"]["mean_overall_pp"] for k in KEYS] + [D3["three sizes: pre - lookahead"]["mean_overall_pp"]]
for x, (pt, lo, hi) in enumerate(diffs):
    pooled = x == len(KEYS)
    point(a2, x, (-pt, -hi, -lo), PURPLES[x], "D" if pooled else "o", ms=4.8 if pooled else 4.1, bold=pooled)
a2.axvline(len(KEYS) - 0.5, color=C["BORDER"], lw=0.8, ls=":", zorder=1)   # marks the pooled estimate off from the per-size points
for ax, title, ticks in ((a1, "", SIZES), (a2, "", [*SIZES, "Pooled"])):
    ax.axhline(0, color=MUTED, lw=1, ls="--", zorder=0)
    ax.set_title(title, fontsize=7, loc="left", color=INK, pad=6)
    ax.set_xticks(range(len(ticks)), ticks, fontsize=5.9)
    ax.set_xlim(-0.5, len(ticks) - 0.5)
    ax.tick_params(axis="y", labelsize=6.5); ax.tick_params(colors=INK); ax.grid(False)
    for side in ("top", "right"): ax.spines[side].set_visible(False)
a2.get_xticklabels()[-1].set_fontweight("bold")
fig.suptitle("Online: 3-benchmark mean success", fontsize=7, color=INK, y=0.955)   # one title for both panels
a1.set_ylim(-1, 14); a1.set_yticks([0, 4, 8, 12]); a1.set_ylabel("Success − SFT (%)", fontsize=7, color=INK)
a2.set_ylim(-1, 9.5); a2.set_yticks([0, 2, 4, 6, 8]); a2.set_ylabel("Lookahead − pre-exec (%)", fontsize=7, color=INK)
a1.set_xlabel("Student ADM size", fontsize=6.5, color=INK); a2.set_xlabel("Student ADM size", fontsize=6.5, color=INK)
handles = [Line2D([], [], marker="o", ls="", color=LOOK, ms=4.1), Line2D([], [], marker="o", ls="", color=PRE, ms=4.1),
           Line2D([], [], marker="o", ls="", color=PURPLES[1], ms=4.1)]
fig.legend(handles, ["Lookahead H=4", "Pre-exec", "Lookahead − pre-exec"], frameon=False, fontsize=6.4,
           loc="lower center", ncol=3, bbox_to_anchor=(0.53, 0.0), columnspacing=1.2, handletextpad=0.3)
fig.subplots_adjust(left=0.12, right=0.985, top=0.84, bottom=0.30)
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
