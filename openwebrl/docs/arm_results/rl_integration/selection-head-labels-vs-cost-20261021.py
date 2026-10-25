"""Overall success vs parameters, PFLOP per episode and model time per episode (2026-10-21/22). Top row: mean over Online-Mind2Web,
WebVoyager and DeepShop, each benchmark weighted equally (success and per-episode cost alike). Bottom row: Online-Mind2Web 300 only, adding the N = 3 selector runs; lines
join each selector's N = 3 (triangle) and N = 5 runs.
Data: selection-head-sft-actions-full-20261021.json (heads, Piotr, SFT 4B) and selection-n3-sft8b-full-20261021.json
(SFT 8B, N = 3 runs)."""
import json
import os
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", "/tmp/openwebrl-selection-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

HERE = Path(__file__).parent
full = json.loads((HERE / "selection-head-sft-actions-full-20261021.json").read_text())["rows"]
new = json.loads((HERE / "selection-n3-sft8b-full-20261021.json").read_text())["rows"]
HEAD, POINT = 4.44e9 + 22.6e6, 4.44e9 + 16.8e6
methods = [  # label, (source, id), params, color, marker, hollow
    ("SFT 4B, 1 sample", ("full", "SFT1"), 4.44e9, "#898781", "s", False),
    ("SFT 8B, 1 sample", ("new", "sft8b"), 8.77e9, "#4a4844", "s", False),
    ("Piotr SelectionARM", ("full", "Piotr"), 8.88e9, "#eb6834", "D", False),
    ("Piotr SelectionARM, N = 3", ("new", "piotr3"), 8.88e9, "#eb6834", "v", False),
    ("Head, Piotr's data", ("full", "Head, Piotr's data (GPT-5.5)"), HEAD, "#2a78d6", "o", False),
    ("Head, Piotr's data, N = 3", ("new", "head3-755d4cd49137"), HEAD, "#2a78d6", "v", False),
    ("Pointwise head, Piotr's data, N = 3", ("new", "head3-26c57559ccb4"), POINT, "#2a78d6", "v", True),
    ("Head, outcome labels", ("full", "Head, Observed outcome, 1,380 SFT states"), HEAD, "#1baf7a", "o", False),
    ("Head, K3 labels", ("full", "Head, Lookahead K3, 1,380 SFT states"), HEAD, "#6250d6", "o", False),
    ("Head, K1 labels", ("full", "Head, Lookahead K1, 1,380 SFT states"), HEAD, "#eda100", "o", False),
    ("Head, pre-execution labels", ("full", "Head, Pre-execution, 1,380 SFT states"), HEAD, "#e87ba4", "o", False),
]
links = [("Piotr SelectionARM", "Piotr SelectionARM, N = 3", "#eb6834"), ("Head, Piotr's data", "Head, Piotr's data, N = 3", "#2a78d6")]


BENCH = ("online_mind2web", "webvoyager", "deepshop")


def one(src, bench):
    s, key = src
    rows = full if s == "full" else new
    r = next((r for r in rows if r.get("condition" if s == "full" else "key") == key and r["benchmark"] == bench), None)
    return r and dict(overall=r["overall"], pflop=r["pflop_per_step"] * r["steps_per_episode"],
                      seconds=r["decision_seconds_mean"] * r["decisions"] / r["tasks"])


def row(src, bench):
    if bench != "mean":
        return one(src, bench)
    rs = [one(src, b) for b in BENCH]
    return None if None in rs else {k: sum(r[k] for r in rs) / 3 for k in rs[0]}


def x_of(r, p, k):
    return p / 1e9 if k == "params" else r[k]


cols = [("params", "Actor + selector (B)"), ("pflop", "PFLOP / episode"), ("seconds", "Model time / episode (s)")]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.labelpad": 3, "axes.spines.top": False, "axes.spines.right": False,
                     "svg.fonttype": "none", "pdf.fonttype": 42, "savefig.facecolor": "white"})
fig, axes = plt.subplots(2, 3, figsize=(7, 3.35), sharey="row")
for ax_row, bench, ylabel in ((axes[0], "mean", "3-benchmark mean\nSuccess (%)"),
                              (axes[1], "online_mind2web", "Online-Mind2Web\nSuccess (%)")):
    pts = {m[0]: (row(m[1], bench), m[2]) for m in methods}
    for ax, (k, xlabel) in zip(ax_row, cols):
        for a, b, color in links:
            if pts[a][0] and pts[b][0]:
                ax.plot([x_of(*pts[a], k), x_of(*pts[b], k)], [pts[a][0]["overall"], pts[b][0]["overall"]], color=color,
                        linewidth=1.2, alpha=0.55, zorder=2)
        for label, src, p, color, marker, hollow in methods:
            r = pts[label][0]
            if r is None:
                continue
            ax.scatter(x_of(r, p, k), r["overall"], s=32 if marker == "v" else 24, marker=marker, zorder=4 if label.startswith("SFT 4B") else 3,
                       facecolor="white" if hollow else color, edgecolor=color if hollow else "white", linewidth=1.0 if hollow else 0.6,
                       label=label)
        ax.grid(False, which="both")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        ax.tick_params(colors="#444444", length=3, width=0.6, pad=2)
        for spine in ax.spines.values():
            spine.set_color("#555555")
            spine.set_linewidth(0.6)
    ax_row[0].set_ylabel(ylabel); ax_row[0].set_xlim(3.5, 9.5)
    ax_row[0].set_xticks([4.44, 6.5, 8.88], ["4.44", "6.5", "8.88"])
axes[0, 0].set_ylim(36.6, 47.4)
axes[1, 0].set_ylim(31.8, 45.8)
for letter, ax, (_, xlabel) in zip("abc", axes[1], cols):
    ax.set_xlabel(f"({letter}) {xlabel}")
handles, labels = axes[1, 0].get_legend_handles_labels()
short_labels = ["SFT 4B", "SFT 8B", "Piotr, N = 5", "Piotr, N = 3",
                "Head: Piotr, N = 5", "Head: Piotr, N = 3", "Pointwise: Piotr, N = 3",
                "Head: outcome", "Head: K = 3", "Head: K = 1", "Head: pre-execution"]
fig.legend(handles, short_labels, loc="upper center", ncol=4, frameon=False,
           bbox_to_anchor=(0.52, 1.005), fontsize=7,
           handletextpad=0.35, columnspacing=1.2, labelspacing=0.5)
fig.subplots_adjust(left=0.095, right=0.985, bottom=0.12, top=0.79, hspace=0.22, wspace=0.15)
for ext in ("png", "svg", "pdf"):
    path = HERE / f"selection-head-labels-vs-cost-20261021.{ext}"
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.025)
    if ext == "svg":
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
plt.close(fig)
