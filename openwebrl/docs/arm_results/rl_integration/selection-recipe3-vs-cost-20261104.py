"""Recipe 3 update of selection-head-labels-vs-cost-20261021: 3-benchmark mean overall success (3 runs pooled) vs actor +
selector parameters, PFLOP per episode and model time per episode, for the Recipe 3 selectors (SFT 4B actor) and the
same-day SFT 4B run (November 2–4). Colors follow the API plots (Okabe-Ito blue / vermilion). Benchmarks weighted equally for success and cost.
Data: selection-recipe3-vs-cost-20261104.json (selection-recipe3-vs-cost-20261104-analysis.py)."""
import json
import os
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", "/tmp/openwebrl-selection-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

HERE = Path(__file__).parent
rows = json.loads((HERE / "selection-recipe3-vs-cost-20261104.json").read_text())["rows"]
HEAD, LORA, SFT = 4.44e9 + 22.6e6, 8.88e9, 4.44e9
methods = [  # key, legend label, params, color, marker, hollow
    ("sft1", "SFT 4B", SFT, "#898781", "s", False),
    ("lora-gpt55-k3", "LoRA: GPT-5.5 K = 3 ¹", LORA, "#0072B2", "D", False),
    ("lora-luna-k3", "LoRA: Luna K = 3", LORA, "#D55E00", "D", False),
    ("lora-gpt55-pre", "LoRA: GPT-5.5 pre ¹", LORA, "#0072B2", "D", True),
    ("lora-luna-pre", "LoRA: Luna pre", LORA, "#D55E00", "D", True),
    ("head-gpt55-k3", "Head: GPT-5.5 K = 3", HEAD, "#0072B2", "o", False),
    ("head-luna-k3", "Head: Luna K = 3", HEAD, "#D55E00", "o", False),
    ("head-gpt55-pre", "Head: GPT-5.5 pre", HEAD, "#0072B2", "o", True),
    ("head-luna-pre", "Head: Luna pre", HEAD, "#D55E00", "o", True),
]
BENCH = ("online_mind2web", "webvoyager", "deepshop")


def mean(key):
    rs = [next(r for r in rows if r["key"] == key and r["benchmark"] == b) for b in BENCH]
    return dict(overall=sum(r["overall"] for r in rs) / 3, pflop=sum(r["pflop_per_episode"] for r in rs) / 3,
                seconds=sum(r["model_seconds_per_episode"] for r in rs) / 3)


cols = [("params", "Actor + selector (B)"), ("pflop", "PFLOP / episode"), ("seconds", "Model time / episode (s)")]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.labelpad": 3, "axes.spines.top": False, "axes.spines.right": False,
                     "svg.fonttype": "none", "pdf.fonttype": 42, "savefig.facecolor": "white"})
fig, axes = plt.subplots(1, 3, figsize=(7, 2.15), sharey="row")
for ax, (k, xlabel) in zip(axes, cols):
    for key, label, p, color, marker, hollow in methods:
        r = mean(key)
        x = p / 1e9 if k == "params" else r[k]
        ax.scatter(x, r["overall"], s=26 if marker == "D" else 24, marker=marker, zorder=4 if key == "sft1" else 3,
                   facecolor="white" if hollow else color, edgecolor=color if hollow else "white",
                   linewidth=1.0 if hollow else 0.6, label=label)
    ax.grid(False, which="both")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    ax.tick_params(colors="#444444", length=3, width=0.6, pad=2)
    for spine in ax.spines.values():
        spine.set_color("#555555"); spine.set_linewidth(0.6)
axes[0].set_ylabel("3-benchmark mean\nSuccess (%)")
axes[0].set_xlim(3.5, 9.5); axes[0].set_xticks([4.44, 6.5, 8.88], ["4.44", "6.5", "8.88"])
for letter, ax, (_, xlabel) in zip("abc", axes, cols):
    ax.set_xlabel(f"({letter}) {xlabel}")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.52, 1.02), fontsize=7,
           handletextpad=0.35, columnspacing=1.2, labelspacing=0.5)
fig.subplots_adjust(left=0.095, right=0.985, bottom=0.2, top=0.66, wspace=0.15)
for ext in ("png", "svg", "pdf"):
    path = HERE / f"selection-recipe3-vs-cost-20261104.{ext}"
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.025)
    if ext == "svg":
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
plt.close(fig)
