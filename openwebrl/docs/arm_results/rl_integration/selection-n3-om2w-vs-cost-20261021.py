"""Online-Mind2Web (300 tasks): overall success vs PFLOP per episode and model time per episode for N = 5 and N = 3
candidates per decision (N = 5 circles/diamond, N = 3 triangles), plus SFT 4B and 8B single samples (2026-10-21). Lines join the same selector at N = 3 and N = 5.
Data: selection-n3-sft8b-full-20261021.json (benchmark == 'online_mind2web')."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# Shared paper typography, including bundled Albert Sans fonts.
import sys as _paper_sys
from pathlib import Path as _PaperPath
_paper_sys.path.insert(0, str(next(p for p in _PaperPath(__file__).resolve().parents
                                if (p / "paper_plot_style.py").is_file())))
import paper_plot_style as _paper_style
from paper_plot_palette import COLORS as C, tint


def _paper_save(fig, path, **kwargs):
    """Keep raster and vector siblings synchronized with the same layout."""
    for suffix in (".png", ".pdf", ".svg"):
        target = path.with_suffix(suffix)
        fig.savefig(target, **kwargs)
        if suffix == ".svg":
            target.write_text("\n".join(line.rstrip() for line in target.read_text().splitlines()) + "\n")



HERE = Path(__file__).parent
rows = {r["key"]: r for r in json.loads((HERE / "selection-n3-sft8b-full-20261021.json").read_text())["rows"]
        if r["benchmark"] == "online_mind2web"}
methods = [  # label, key, color, marker
    ("SFT 4B, 1 sample", "sft1", C["BASELINE_LIGHT"], "s"), ("SFT 8B, 1 sample", "sft8b", C["BASELINE_DARK"], "s"),
    ("Piotr SelectionARM, N = 5", "piotr5", C["ADM_4B"], "D"), ("Piotr SelectionARM, N = 3", "piotr3", C["ADM_4B"], "D"),
    ("Joint head, Piotr's data, N = 5", "head5-755d4cd49137", C["ADM_22M"], "o"),
    ("Joint head, Piotr's data, N = 3", "head3-755d4cd49137", C["ADM_22M"], "v"),
    ("Pointwise head, Piotr's data, N = 3", "head3-26c57559ccb4", C["ADM_22M"], "s"),
]
links = [("piotr3", "piotr5", C["ADM_4B"]), ("head3-755d4cd49137", "head5-755d4cd49137", C["ADM_22M"])]
panels = [("pflop_per_episode", "PFLOP per episode"), ("model_seconds_per_episode", "Model time per episode (s)")]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
for ax, (k, xlabel) in zip(axes, panels):
    for a, b, color in links:
        ax.plot([rows[a][k], rows[b][k]], [rows[a]["overall"], rows[b]["overall"]], color=color, linewidth=2, alpha=0.5, zorder=2)
    for label, key, color, marker in methods:
        ax.scatter(rows[key][k], rows[key]["overall"], s=80, facecolor="white" if key == "piotr3" else color, marker=marker, edgecolor=color if key == "piotr3" else "white", linewidth=2, zorder=3,
                   label=label)
    ax.set_xlabel(xlabel)
    ax.grid(True, color=C["BORDER"], linewidth=0.8); ax.set_axisbelow(True)
    ax.tick_params(colors=C["MUTED"]); [s.set_color(C["BORDER"]) for s in ax.spines.values()]
axes[0].set_ylabel("Overall success, Online-Mind2Web 300 (%)")
axes[0].set_ylim(35.3, 45.6)
axes[0].annotate("SFT 4B and 8B both 36.3%", (rows["sft1"]["pflop_per_episode"], 35.6), fontsize=8.5, color=C["MUTED"])
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.04))
fig.suptitle("Candidates per decision on Online-Mind2Web (T 1.0 / top-p 0.9, 30 steps)", y=0.99)
fig.tight_layout(rect=(0, 0.14, 1, 0.97))
_paper_save(fig, HERE / "selection-n3-om2w-vs-cost-20261021.png", dpi=160, bbox_inches="tight")
