"""Recipe 3 GPT-5.5 H = 4 4B LoRA data ablations: November 8-9 same-window rerun over the November 4-6 batch. Two figures:
  selection-recipe3-data-ablations-rerun-20261109.*                equal-weight 3-benchmark mean of overall success
  selection-recipe3-data-ablations-rerun-per-benchmark-20261109.*  overall success on Online-Mind2Web, WebVoyager and DeepShop
  ...-rerun-clean-20261109.* / ...-rerun-clean-per-benchmark-20261109.*  the same without the faded November 4-6 batch
Same axes and encoding as selection-recipe3-data-ablations-compact-20261105.py (log axis of 5-action sets in the training data;
diamonds = ADM-4B; solid lines = fewer states, dashed lines = fewer sets per state; larger diamond = full data). Solid: November 8-9 rerun, 3 runs per point, all
conditions in one window. Faded: the November 4-6 batch (subsets November 4-5, full data November 6), joined to the rerun of the
same LoRA by a thin gray line; only rerun points carry names (faded: points at 2.2K / 8.9K sets = 252 / 1,001 states, points at
2.1K / 7.4K = 1 / 3.5 sets per state). The 252-state subset was rerun on November 10-11 in its own window with a full-data reference
(45.6% mean overall there vs 46.1% on November 8-9); its solid point is that window's raw value. Dashed: SFT 4B alone (3 runs, November 6;
not rerun). Data: selection-recipe3-data-ablations-rerun-20261109-analysis.json, selection-recipe3-data-s250-20261110.json and
selection-recipe3-data-ablations-20261105.json."""
import json
import os
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", "/tmp/openwebrl-selection-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys as _paper_sys
from pathlib import Path as _PaperPath
_paper_sys.path.insert(0, str(next(p for p in _PaperPath(__file__).resolve().parents
                                if (p / "paper_plot_style.py").is_file())))
import paper_plot_style as _paper_style  # noqa: F401
from paper_plot_palette import COLORS as C, tint
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

HERE = Path(__file__).parent
old = json.loads((HERE / "selection-recipe3-data-ablations-20261105.json").read_text())["rates"]
new = json.loads((HERE / "selection-recipe3-data-ablations-rerun-20261109-analysis.json").read_text())["rates"]
_s250 = json.loads((HERE / "selection-recipe3-data-s250-20261110.json").read_text())["conditions"]["arm5-d6c492ad4c40"]
new["s250"] = {"mean_overall": 100 * _s250["mean_overall"], **{b: {"overall": 100 * _s250[b]["overall"]} for b in
               ("online_mind2web", "webvoyager", "deepshop")}}                # November 10-11 window
SETS = {"s250": 2234, "s1000": 8875, "pps1": 2112, "pps4": 7417, "full": 18786}
STATES = {"s250": 252, "s1000": 1001, "pps1": 2112, "pps4": 2112, "full": 2112}
MARK = {"s250": "o", "s1000": "o", "pps1": "D", "pps4": "D", "full": "s"}
SERIES = [("o", ["s250", "s1000", "full"], "states"), ("D", ["pps1", "pps4", "full"], "sets")]
BENCH = [("online_mind2web", "Online-Mind2Web"), ("webvoyager", "WebVoyager"), ("deepshop", "DeepShop")]
COL, FADE = C["ADM_4B"], tint("ADM_4B", 0.68)


def name(k):
    r = SETS[k] / STATES[k]
    if k == "s250":
        return "0.25K states†"
    return f"{STATES[k] / 1000:.2g}K states" if MARK[k] == "o" else f"{r:.2g} {'set' if r == 1 else 'sets'}/state"


def draw(ax, yo, yn, sft, fs, faded=True):
    if not faded:                                                    # clean view: rerun points only
        yo = dict(yn)
    vals = list(yo.values()) + list(yn.values()) + [sft]
    for marker, keys, family in SERIES:                                   # faded November 4-6 families
        if faded:
            ax.plot([SETS[k] for k in keys], [yo[k] for k in keys], color=FADE, linestyle="-" if family == "states" else "--", linewidth=1.0, zorder=1)
        ks = [k for k in keys if k in yn]                            # solid rerun families
        if len(ks) > 1:
            ax.plot([SETS[k] for k in ks], [yn[k] for k in ks], color=COL, linestyle="-" if family == "states" else "--", linewidth=1.4, zorder=2)
    for k in SETS:
        if faded:
            ax.scatter(SETS[k], yo[k], s=30, marker="D", color="white", edgecolors=FADE, linewidths=1.0, zorder=2)
        if faded: ax.annotate(f"{yo[k]:.1f}", (SETS[k], yo[k]), textcoords="offset points", xytext=(5, 0), ha="left",
                    va="center", fontsize=fs - 1, color=tint("MUTED", 0.3))
        if k in yn:
            if faded:
                ax.plot([SETS[k]] * 2, [yo[k], yn[k]], color=C["BORDER"], linewidth=0.8, zorder=1)
            ax.scatter(SETS[k], yn[k], s=40 if k == "full" else 32, marker="D", color=COL, zorder=4)
            up = yn[k] >= yo[k]
            if not faded and k in ("pps4", "s1000"):                  # clean view: the ~8K-set pair splits above / below
                up = yn[k] >= yn["s1000" if k == "pps4" else "pps4"]
            lab = f"{yn[k]:.1f}" if k == "full" else f"{name(k)}\n{yn[k]:.1f}" if up else f"{yn[k]:.1f}\n{name(k)}"
            if k in ("pps1", "s250"):                                 # ~2K-set pair (6% apart): both label on the left, higher above, lower below
                hi = yn[k] >= yn["s250" if k == "pps1" else "pps1"]
                ax.annotate(f"{name(k)}\n{yn[k]:.1f}" if hi else f"{yn[k]:.1f}\n{name(k)}", (SETS[k], yn[k]), textcoords="offset points",
                            xytext=(-8, 6 if hi else -6), ha="right",
                            va="bottom" if hi else "top", fontsize=fs, color=COL, linespacing=1.0)
            elif MARK[k] == "o":                                     # right of the point: clear of the diamond at 7.4K sets
                ax.annotate(f"{name(k)}\n{yn[k]:.1f}" if up else f"{yn[k]:.1f}\n{name(k)}", (SETS[k], yn[k]), textcoords="offset points",
                            xytext=(7, 3 if up else -10), ha="left", va="bottom" if up else "top", fontsize=fs, color=COL, linespacing=1.0)
            else:
                ax.annotate(lab, (SETS[k], yn[k]), textcoords="offset points", xytext=(0, 6 if up else -6), ha="center",
                            va="bottom" if up else "top", fontsize=fs, color=COL, linespacing=1.0)
    ax.axhline(sft, color=C["BASELINE_LIGHT"], linestyle="--", linewidth=0.9, zorder=0)
    ax.text(29000, sft - 0.3, f"SFT 4B alone (Nov 6) {sft:.1f}", fontsize=fs, color=C["MUTED"], va="top", ha="right")
    ax.set_xscale("log"); ax.set_xticks([2000, 4000, 8000, 16000]); ax.set_xticklabels(["2K", "4K", "8K", "16K"]); ax.minorticks_off()
    ax.set_xlim(1050, 30000); ax.set_ylim(min(vals) - 4.2, max(vals) + 3.6)
    ax.set_xlabel("5-action sets in training data")
    ax.yaxis.set_major_locator(MultipleLocator(2 if max(vals) - min(vals) < 12 else 4))
    ax.grid(axis="y", color=C["BORDER"], linewidth=0.6, zorder=0)
    ax.tick_params(colors=C["MUTED"], length=3, width=0.6, pad=2)
    for s in ("top", "right"): ax.spines[s].set_visible(False)


def legend(fig, ncol, faded=True):
    h = [Line2D([], [], color=COL, marker="D", markersize=5, linewidth=1.4, label="Fewer states (8.9 sets/state)"),
         Line2D([], [], color=COL, marker="D", markersize=5, linewidth=1.4, linestyle="--", label="Fewer sets per state (2.1K states)"),
         Line2D([], [], color=COL, marker="D", linestyle="none", markersize=5, label="Full data"),
         Line2D([], [], color=COL, marker="D", linestyle="none", markersize=5, label="Rerun, Nov 8–9 († Nov 10–11)"),
         Line2D([], [], color=FADE, marker="D", markerfacecolor="white", linestyle="none", markersize=5, label="Nov 4–6 batch")]
    if not faded:
        h = h[:3] + [Line2D([], [], color="none", label="Nov 8–9, 3 runs († Nov 10–11)")]
    fig.legend(handles=h, loc="lower center", ncol=ncol, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.02), handletextpad=0.4, columnspacing=1.2, handlelength=2.8)


def save(fig, stem):
    for ext in ("png", "pdf", "svg"):
        output = HERE / f"{stem}.{ext}"
        fig.savefig(output, dpi=220 if ext == "png" else None, bbox_inches="tight")
        if ext == "svg":
            output.write_text("\n".join(line.rstrip() for line in output.read_text().splitlines()) + "\n")
    plt.close(fig)


subs = list(SETS)
for faded, tag in ((True, ""), (False, "-clean")):
    fig, ax = plt.subplots(figsize=(4.4, 3.5))
    draw(ax, {k: old[k]["mean_overall"] for k in subs}, {k: v["mean_overall"] for k, v in new.items()}, old["sft"]["mean_overall"], 7, faded)
    ax.set_ylabel("Mean overall success (%)")
    legend(fig, 2, faded); fig.tight_layout(rect=(0, 0.16, 1, 1))
    save(fig, f"selection-recipe3-data-ablations-rerun{tag}-20261109")

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    for ax, (b, title) in zip(axes, BENCH):
        draw(ax, {k: old[k][b]["overall"] for k in subs}, {k: v[b]["overall"] for k, v in new.items()}, old["sft"][b]["overall"], 6.5, faded)
        ax.set_title(title, fontsize=10); ax.set_ylabel("Overall success (%)")
    legend(fig, 5 if faded else 4, faded); fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, f"selection-recipe3-data-ablations-rerun{tag}-per-benchmark-20261109")
