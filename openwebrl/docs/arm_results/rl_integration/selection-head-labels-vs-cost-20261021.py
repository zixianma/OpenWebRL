"""Overall success on all 1,045 tasks vs parameters, PFLOP per episode and model time per episode (2026-10-21).
Data: selection-head-sft-actions-full-20261021.json (rows with benchmark == 'all')."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
rows = {r["condition"]: r for r in json.loads((HERE / "selection-head-sft-actions-full-20261021.json").read_text())["rows"]
        if r["benchmark"] == "all"}
HEAD_P = 4.44e9 + 22.6e6
methods = [  # label, condition, params, color, marker
    ("SFT 4B, 1 sample", "SFT1", 4.44e9, "#898781", "s"),
    ("Piotr SelectionARM", "Piotr", 8.88e9, "#eb6834", "D"),
    ("Head, Piotr's data", "Head, Piotr's data (GPT-5.5)", HEAD_P, "#2a78d6", "o"),
    ("Head, outcome labels", "Head, Observed outcome, 1,380 SFT states", HEAD_P, "#1baf7a", "o"),
    ("Head, K3 labels", "Head, Lookahead K3, 1,380 SFT states", HEAD_P, "#6250d6", "o"),
    ("Head, K1 labels", "Head, Lookahead K1, 1,380 SFT states", HEAD_P, "#eda100", "o"),
    ("Head, pre-execution labels", "Head, Pre-execution, 1,380 SFT states", HEAD_P, "#e87ba4", "o"),
]
def x_of(r, p, k):
    return {"params": p / 1e9, "pflop": r["pflop_per_step"] * r["steps_per_episode"],
            "seconds": r["decision_seconds_mean"] * r["decisions"] / r["tasks"]}[k]
panels = [("params", "Parameters, actor + selector (B)"), ("pflop", "PFLOP per episode"),
          ("seconds", "Model time per episode (s)")]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharey=True)
for ax, (k, xlabel) in zip(axes, panels):
    for label, cond, p, color, marker in methods:
        r = rows[cond]
        ax.scatter(x_of(r, p, k), r["overall"], s=70, color=color, marker=marker, edgecolor="white", linewidth=2, zorder=4 if cond == "SFT1" else 3,
                   label=label)
    ax.set_xlabel(xlabel)
    ax.grid(True, color="#e6e5e0", linewidth=0.8); ax.set_axisbelow(True)
    ax.tick_params(colors="#55534d"); [s.set_color("#c3c2b7") for s in ax.spines.values()]
axes[0].set_ylabel("Overall success, 1,045 tasks (%)")
axes[0].set_xlim(3.5, 9.5)
axes[0].set_ylim(39.3, 48.8)
axes[0].annotate("SFT 4.44 B; every head 4.46 B", (4.6, 39.6), fontsize=8.5, color="#55534d")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.suptitle("Selection methods on Online-Mind2Web + WebVoyager + DeepShop (T 1.0 / top-p 0.9, 30 steps)", y=0.99)
fig.tight_layout(rect=(0, 0.1, 1, 0.97))
fig.savefig(HERE / "selection-head-labels-vs-cost-20261021.png", dpi=160, bbox_inches="tight")
