"""Overall success vs parameters, PFLOP per episode and model time per episode (2026-10-21/22). Top row: all 1,045 tasks
(Online-Mind2Web + WebVoyager + DeepShop). Bottom row: Online-Mind2Web 300 only, adding the N = 3 selector runs; lines
join each selector's N = 3 (triangle) and N = 5 runs.
Data: selection-head-sft-actions-full-20261021.json (heads, Piotr, SFT 4B) and selection-n3-sft8b-full-20261021.json
(SFT 8B, N = 3 runs)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def row(src, bench):
    s, key = src
    rows = full if s == "full" else new
    return next((r for r in rows if r.get("condition" if s == "full" else "key") == key and r["benchmark"] == bench), None)


def x_of(r, p, k):
    return {"params": p / 1e9, "pflop": r["pflop_per_step"] * r["steps_per_episode"],
            "seconds": r["decision_seconds_mean"] * r["decisions"] / r["tasks"]}[k]


cols = [("params", "Parameters, actor + selector (B)"), ("pflop", "PFLOP per episode"), ("seconds", "Model time per episode (s)")]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(2, 3, figsize=(13, 8.6), sharey="row")
for ax_row, bench, ylabel in ((axes[0], "all", "All 1,045 tasks: overall success (%)"),
                              (axes[1], "online_mind2web", "Online-Mind2Web 300: overall success (%)")):
    pts = {m[0]: (row(m[1], bench), m[2]) for m in methods}
    for ax, (k, xlabel) in zip(ax_row, cols):
        for a, b, color in links:
            if pts[a][0] and pts[b][0]:
                ax.plot([x_of(*pts[a], k), x_of(*pts[b], k)], [pts[a][0]["overall"], pts[b][0]["overall"]], color=color,
                        linewidth=2, alpha=0.45, zorder=2)
        for label, src, p, color, marker, hollow in methods:
            r = pts[label][0]
            if r is None:
                continue
            ax.scatter(x_of(r, p, k), r["overall"], s=95 if marker == "v" else 70, marker=marker, zorder=4 if label.startswith("SFT 4B") else 3,
                       facecolor="white" if hollow else color, edgecolor=color if hollow else "white", linewidth=1.6 if hollow else 2,
                       label=label)
        ax.set_xlabel(xlabel)
        ax.grid(True, color="#e6e5e0", linewidth=0.8); ax.set_axisbelow(True)
        ax.tick_params(colors="#55534d"); [s.set_color("#c3c2b7") for s in ax.spines.values()]
    ax_row[0].set_ylabel(ylabel); ax_row[0].set_xlim(3.5, 9.5)
axes[0, 0].set_ylim(39.3, 48.8)
axes[1, 0].set_ylim(31.8, 45.8)
for ax, y in ((axes[0, 0], 39.6), (axes[1, 0], 32.2)):
    ax.annotate("SFT 4B 4.44 B; every head 4.46 B", (4.6, y), fontsize=8.5, color="#55534d")
axes[1, 1].annotate("SFT 4B and 8B both 36.3%", (0.27, 35.3), fontsize=8.5, color="#55534d")
handles, labels = axes[1, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01))
fig.suptitle("Selection methods, T 1.0 / top-p 0.9, 30 steps (N = 5 candidates unless marked N = 3)", y=0.995)
fig.tight_layout(rect=(0, 0.09, 1, 0.98))
fig.savefig(HERE / "selection-head-labels-vs-cost-20261021.png", dpi=160, bbox_inches="tight")
