"""Recipe 3 cost plots (SFT 4B actor, November 2–4, with the same-day SFT 4B run; plus the RL 8B actor alone and with the GPT-5.5
LoRAs, November 4–5, down triangles): valid-only success (successes / valid episodes, 3 runs pooled; mean = equal-weight mean of the three) vs actor +
selector parameters, PFLOP per episode and model time per episode. Renders two figures:
  selection-recipe3-vs-cost-20261104.*               3-benchmark equal-weight mean (success and cost alike)
  selection-recipe3-vs-cost-per-benchmark-20261104.*  one row per benchmark
  selection-recipe3-vs-cost[-per-benchmark]-sft-actors-20261104.*  the same without the RL checkpoints (SFT actors only)
Observed Pareto frontiers as in the API plots (plot_browsercode_pareto_previews.py): dashed straight segments join the
observed nondominated points of each panel (lower x and higher success; dominance needs at least one strict gain), and thin
rings mark frontier points. No smoothing or statistical superiority is implied. Colors follow the API plots (Okabe-Ito blue /
vermilion). Pre-execution-label conditions are omitted from the plots (still in the JSON). Data: selection-recipe3-vs-cost-20261104.json (selection-recipe3-vs-cost-20261104-analysis.py)."""
import json
import os
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", "/tmp/openwebrl-selection-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# Shared paper typography, including bundled Albert Sans fonts.
import sys as _paper_sys
from pathlib import Path as _PaperPath
_paper_sys.path.insert(0, str(next(p for p in _PaperPath(__file__).resolve().parents
                                if (p / "paper_plot_style.py").is_file())))
import paper_plot_style as _paper_style

from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

HERE = Path(__file__).parent
rows = json.loads((HERE / "selection-recipe3-vs-cost-20261104.json").read_text())["rows"]
HEAD, LORA, SFT = 4.44e9 + 22.6e6, 8.88e9, 4.44e9
FRONTIER = "#596675"
methods = [  # key, legend label, params, color, marker, hollow
    ("sft1", "SFT 4B", SFT, "#898781", "s", False),
    ("sft8b", "SFT 8B", 8.77e9, "#4a4844", "s", False),
    ("rl4b", "RL 4B ²", SFT, "#4a4844", "^", False),
    ("rl8b", "RL 8B ³", 8.77e9, "#4a4844", "v", False),
    ("rl8b-lora-gpt55-k3", "RL 8B + ADM-4B (GPT-5.5) ³", 8.77e9 + SFT, "#0072B2", "v", False),
    ("lora-gpt55-k3", "ADM-4B (GPT-5.5) ¹", LORA, "#0072B2", "D", False),
    ("lora-luna-k3", "ADM-4B (GPT-6 Luna)", LORA, "#D55E00", "D", False),
    ("head-gpt55-k3", "ADM-22M (GPT-5.5)", HEAD, "#0072B2", "o", False),
    ("head-luna-k3", "ADM-22M (GPT-6 Luna)", HEAD, "#D55E00", "o", False),
]
BENCH = ("online_mind2web", "webvoyager", "deepshop")
NAMES = {"mean": "3-benchmark mean", "online_mind2web": "OM2W", "webvoyager": "WebVoyager", "deepshop": "DeepShop"}
cols = [("params", "Actor + selector (B)"), ("pflop", "PFLOP / episode"), ("seconds", "Model time / episode (s)")]


def point(key, bench):
    rs = [next(r for r in rows if r["key"] == key and r["benchmark"] == b) for b in (BENCH if bench == "mean" else (bench,))]
    return dict(success=sum(r["valid_only"] for r in rs) / len(rs), pflop=sum(r["pflop_per_episode"] for r in rs) / len(rs),
                seconds=sum(r["model_seconds_per_episode"] for r in rs) / len(rs))


def frontier(pts):
    """Lower x and higher success; dominance needs at least one strict gain."""
    keep = [(x, y) for x, y in pts if not any(a <= x and b >= y and (a < x or b > y) for a, b in pts)]
    return sorted(set(keep))


def panel(ax, bench, k, methods):
    pts = []
    for key, label, p, color, marker, hollow in methods:
        r = point(key, bench)
        x = p / 1e9 if k == "params" else r[k]
        pts.append((x, r["success"]))
        ax.scatter(x, r["success"], s=24, marker=marker, zorder=4 if key in ("sft1", "sft8b", "rl4b", "rl8b") else 3,
                   facecolor="white" if hollow else color, edgecolor=color if hollow else "white",
                   linewidth=1.0 if hollow else 0.6, label=label)
    fr = frontier(pts)
    if len(fr) > 1:
        ax.plot([x for x, _ in fr], [y for _, y in fr], linestyle="--", color=FRONTIER, linewidth=0.95, zorder=1.8)
    ax.scatter([x for x, _ in fr], [y for _, y in fr], s=57, facecolors="none", edgecolors=FRONTIER, linewidths=0.6, zorder=2.4)
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.6, zorder=0)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
    ax.tick_params(colors="#444444", length=3, width=0.6, pad=2)
    for spine in ax.spines.values():
        spine.set_color("#555555"); spine.set_linewidth(0.6)
    if k == "params":
        wide = any(m[2] > 9e9 for m in methods)
        ax.set_xlim(3.5, 14.2 if wide else 9.5)
        ax.set_xticks(*([[4.44, 8.88, 13.2], ["4.44", "8.88", "13.2"]] if wide else [[4.44, 8.88], ["4.44", "8.88"]]))


def legend(fig, axes, x0):
    handles, labels = axes.flat[0].get_legend_handles_labels()
    handles.append(Line2D([], [], linestyle="--", color=FRONTIER, linewidth=0.95, marker="o", markerfacecolor="white",
                          markeredgecolor=FRONTIER, markeredgewidth=0.6, markersize=4.4))
    labels.append("Observed frontier")
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(x0, 0.5), frameon=False, fontsize=6.6,
               handletextpad=0.35, labelspacing=0.45, borderaxespad=0)


def render(benches, name, height, methods=methods):
    compact_sft_mean = name == "selection-recipe3-vs-cost-sft-actors-20261104"
    if compact_sft_mean:
        methods = [(key, label.replace(" ¹", ""), *style) for key, label, *style in methods]
    plt.rcParams.update({"font.family": "Albert Sans", "font.size": 7.5, "xtick.labelsize": 6.8, "ytick.labelsize": 6.8,
                         "axes.labelpad": 2, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "pdf.fonttype": 42, "savefig.facecolor": "white"})
    fig, axes = plt.subplots(len(benches), 3, figsize=(7, height), sharey="row", squeeze=False)
    for i, (ax_row, bench) in enumerate(zip(axes, benches)):
        for ax, (k, xlabel) in zip(ax_row, cols):
            panel(ax, bench, k, methods)
            if compact_sft_mean:
                for collection in ax.collections:
                    collection.set_clip_on(False)
            if i == 0:
                title = {"params": "System parameters (B)", "seconds": "Model latency (s)"}.get(k, xlabel) if compact_sft_mean else xlabel
                ax.set_title(title, fontsize=7.5, pad=4)
        ax_row[0].set_ylabel("Valid-only\nsuccess (%)" if compact_sft_mean else
                             (f"{NAMES[bench]}\nvalid-only (%)" if len(benches) > 1 else f"{NAMES[bench]}\nvalid-only success (%)"))
    legend(fig, axes, 0.80)
    fig.subplots_adjust(left=0.085, right=0.785, bottom=0.08 if len(benches) > 1 else 0.14,
                        top=0.93 if len(benches) > 1 else 0.85, hspace=0.32, wspace=0.17)
    if compact_sft_mean:
        fig.subplots_adjust(bottom=0.19, top=0.80)
    for ext in ("png", "svg", "pdf"):
        path = HERE / f"{name}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.025)
        if ext == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)
    return {bench: {k: frontier([(p / 1e9 if k == "params" else point(key, bench)[k], point(key, bench)["success"])
                                 for key, _, p, *_ in methods]) for k, _ in cols} for bench in benches}


if __name__ == "__main__":
    print(render(("mean",), "selection-recipe3-vs-cost-20261104", 1.9))
    render(BENCH, "selection-recipe3-vs-cost-per-benchmark-20261104", 3.9)
    sft = [m for m in methods if not m[0].startswith("rl")]
    print(render(("mean",), "selection-recipe3-vs-cost-sft-actors-20261104", 1.30, sft))
    render(BENCH, "selection-recipe3-vs-cost-per-benchmark-sft-actors-20261104", 3.9, sft)
