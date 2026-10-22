"""Online-Mind2Web (300 tasks): overall success vs PFLOP per episode and model time per episode for N = 5 and N = 3
candidates per decision (N = 5 circles/diamond, N = 3 triangles), plus SFT 4B and 8B single samples (2026-10-21). Lines join the same selector at N = 3 and N = 5.
Data: selection-n3-sft8b-full-20261021.json (benchmark == 'online_mind2web')."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
rows = {r["key"]: r for r in json.loads((HERE / "selection-n3-sft8b-full-20261021.json").read_text())["rows"]
        if r["benchmark"] == "online_mind2web"}
methods = [  # label, key, color, marker
    ("SFT 4B, 1 sample", "sft1", "#898781", "s"), ("SFT 8B, 1 sample", "sft8b", "#4a4844", "s"),
    ("Piotr SelectionARM, N = 5", "piotr5", "#eb6834", "D"), ("Piotr SelectionARM, N = 3", "piotr3", "#eb6834", "v"),
    ("Joint head, Piotr's data, N = 5", "head5-755d4cd49137", "#2a78d6", "o"),
    ("Joint head, Piotr's data, N = 3", "head3-755d4cd49137", "#2a78d6", "v"),
    ("Pointwise head, Piotr's data, N = 3", "head3-26c57559ccb4", "#1baf7a", "v"),
]
links = [("piotr3", "piotr5", "#eb6834"), ("head3-755d4cd49137", "head5-755d4cd49137", "#2a78d6")]
panels = [("pflop_per_episode", "PFLOP per episode"), ("model_seconds_per_episode", "Model time per episode (s)")]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
for ax, (k, xlabel) in zip(axes, panels):
    for a, b, color in links:
        ax.plot([rows[a][k], rows[b][k]], [rows[a]["overall"], rows[b]["overall"]], color=color, linewidth=2, alpha=0.5, zorder=2)
    for label, key, color, marker in methods:
        ax.scatter(rows[key][k], rows[key]["overall"], s=80, color=color, marker=marker, edgecolor="white", linewidth=2, zorder=3,
                   label=label)
    ax.set_xlabel(xlabel)
    ax.grid(True, color="#e6e5e0", linewidth=0.8); ax.set_axisbelow(True)
    ax.tick_params(colors="#55534d"); [s.set_color("#c3c2b7") for s in ax.spines.values()]
axes[0].set_ylabel("Overall success, Online-Mind2Web 300 (%)")
axes[0].set_ylim(35.3, 45.6)
axes[0].annotate("SFT 4B and 8B both 36.3%", (rows["sft1"]["pflop_per_episode"], 35.6), fontsize=8.5, color="#55534d")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.04))
fig.suptitle("Candidates per decision on Online-Mind2Web (T 1.0 / top-p 0.9, 30 steps)", y=0.99)
fig.tight_layout(rect=(0, 0.14, 1, 0.97))
fig.savefig(HERE / "selection-n3-om2w-vs-cost-20261021.png", dpi=160, bbox_inches="tight")
