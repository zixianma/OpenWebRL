"""Recipe 3 cost plots (SFT 4B actor, November 2–4, with the same-day SFT 4B run; plus the RL 8B actor alone and with the GPT-5.5
LoRAs, November 4–5, down triangles): valid-only success (successes / valid episodes, 3 runs pooled; mean = equal-weight mean of the three) vs actor +
selector parameters, PFLOP per episode and model time per episode. Renders five figures:
  selection-recipe3-vs-cost-20261104.*               3-benchmark equal-weight mean (success and cost alike)
  selection-recipe3-vs-cost-per-benchmark-20261104.*  one row per benchmark
  selection-recipe3-vs-cost[-per-benchmark]-sft-actors-20261104.*  the SFT actors plus the user-provided ADM-distilled actor (4.44B parameters assumed)
  selection-recipe3-vs-cost-rl-actors-20261104.*  compact mean view of the RL 4B / 8B checkpoints alone and with selectors only
Observed Pareto frontiers as in the API plots (plot_browsercode_pareto_previews.py): dashed straight segments join the
observed nondominated points of each panel (lower x and higher success; dominance needs at least one strict gain), and thin
rings mark frontier points. No smoothing or statistical superiority is implied. ADM-22M uses light purple. ADM-4B uses diamonds: green/teal for GPT-5.5 and salmon for Luna labels. ADM-22M teacher variants use circle/square markers; RL baselines retain up/down triangles, with hollow markers for ADM-22M. SFT-only views use filled ADM-23M circles and larger filled ADM-4B diamonds, with bright purple for GPT-5.5 and salmon for Luna labels. Pre-execution-label conditions are omitted from the plots (still in the JSON). Data: selection-recipe3-vs-cost-20261104.json (selection-recipe3-vs-cost-20261104-analysis.py)."""
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
from paper_plot_palette import COLORS as C, tint

from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

HERE = Path(__file__).parent
rows = json.loads((HERE / "selection-recipe3-vs-cost-20261104.json").read_text())["rows"]
HEAD, LORA, SFT = 4.44e9 + 22.6e6, 8.88e9, 4.44e9
DISTILLED = json.loads((HERE / "selection-adm-distilled-actor-20261110.json").read_text())
FRONTIER = C["FRONTIER"]
methods = [  # key, legend label, params, color, marker, hollow
    ("sft1", "SFT 4B", SFT, C["BASELINE_LIGHT"], "s", False),
    ("sft8b", "SFT 8B", 8.77e9, C["BASELINE_DARK"], "s", False),
    ("rl4b", "RL 4B ²", SFT, C["BASELINE_DARK"], "^", False),
    ("rl4b-lora-gpt55-k3", "RL 4B + ADM-4B (GPT-5.5) ²", LORA, C["ADM_4B"], "D", False),
    ("rl4b-head-gpt55-k3", "RL 4B + ADM-22M (GPT-5.5) ²", HEAD, C["ADM_22M"], "^", True),
    ("rl8b", "RL 8B ³", 8.77e9, C["BASELINE_DARK"], "v", False),
    ("rl8b-lora-gpt55-k3", "RL 8B + ADM-4B (GPT-5.5) ³", 8.77e9 + SFT, tint("ADM_4B", .30), "D", False),
    ("lora-gpt55-k3", "ADM-4B (GPT-5.5) ¹", LORA, C["ADM_4B"], "D", False),
    ("lora-luna-k3", "ADM-4B (GPT-6 Luna)", LORA, C["ADM_4B_LUNA"], "D", False),
    ("head-gpt55-k3", "ADM-22M (GPT-5.5)", HEAD, C["ADM_22M"], "o", False),
    ("head-luna-k3", "ADM-22M (GPT-6 Luna)", HEAD, C["ADM_22M"], "s", False),
]
BENCH = ("online_mind2web", "webvoyager", "deepshop")
NAMES = {"mean": "3-benchmark mean", "online_mind2web": "OM2W", "webvoyager": "WebVoyager", "deepshop": "DeepShop"}
cols = [("params", "Actor + selector (B)"), ("pflop", "PFLOP / episode"), ("seconds", "Model time / episode (s)")]


def point(key, bench, success_metric="valid_only"):
    rs = ([DISTILLED["benchmarks"][b] for b in (BENCH if bench == "mean" else (bench,))]
          if key == "adm-distilled" else
          [next(r for r in rows if r["key"] == key and r["benchmark"] == b) for b in (BENCH if bench == "mean" else (bench,))])
    return dict(success=sum(r[success_metric] for r in rs) / len(rs), pflop=sum(r["pflop_per_episode"] for r in rs) / len(rs),
                seconds=sum(r["model_seconds_per_episode"] for r in rs) / len(rs))


def frontier(pts):
    """Lower x and higher success; dominance needs at least one strict gain."""
    keep = [(x, y) for x, y in pts if not any(a <= x and b >= y and (a < x or b > y) for a, b in pts)]
    return sorted(set(keep))


def panel(ax, bench, k, methods, *, circle_encoding=False, success_metric="valid_only"):
    pts = []
    for key, label, p, color, marker, hollow in methods:
        r = point(key, bench, success_metric)
        x = p / 1e9 if k == "params" else r[k]
        pts.append((x, r["success"]))
        size = 44 if circle_encoding and "lora" in key else 24
        ax.scatter(x, r["success"], s=size, marker=marker, zorder=4 if key in ("sft1", "sft8b", "rl4b", "rl8b") else 3,
                   facecolor="white" if hollow else color, edgecolor=color if hollow else "white",
                   linewidth=1.25 if hollow and circle_encoding else 1.0 if hollow else 0.6, label=label)
    fr = frontier(pts)
    emphasize = circle_encoding and success_metric == "overall"
    if len(fr) > 1:
        ax.plot([x for x, _ in fr], [y for _, y in fr], linestyle="--", color=FRONTIER,
                linewidth=1.4 if emphasize else .95, zorder=1.8)
    ax.scatter([x for x, _ in fr], [y for _, y in fr], s=80 if emphasize else 57,
               facecolors="none", edgecolors=FRONTIER, linewidths=1.2 if emphasize else .6,
               zorder=4.5 if emphasize else 2.4)
    if emphasize and bench == "mean":
        best = max(fr, key=lambda xy: xy[1])
        ax.annotate("Pareto-optimal", best, xytext=(9, -2), textcoords="offset points",
                    fontsize=6.3, color=C["MUTED"], ha="left", va="top", zorder=5)
    ax.grid(axis="y", color=C["BORDER"], linewidth=0.6, zorder=0)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
    ax.tick_params(colors=C["MUTED"], length=3, width=0.6, pad=2)
    for spine in ax.spines.values():
        spine.set_color(C["MUTED"]); spine.set_linewidth(0.6)
    if k == "params":
        wide = any(m[2] > 9e9 for m in methods)
        ax.set_xlim(3.5, 14.2 if wide else 9.5)
        ax.set_xticks(*([[4.44, 8.88, 13.2], ["4.44", "8.88", "13.2"]] if wide else [[4.44, 8.88], ["4.44", "8.88"]]))


def legend(fig, axes, *, sft_only):
    handles, labels = axes.flat[0].get_legend_handles_labels()
    handles.append(Line2D([], [], linestyle="--", color=FRONTIER, linewidth=0.95, marker="o", markerfacecolor="white",
                          markeredgecolor=FRONTIER, markeredgewidth=0.6, markersize=4.4))
    labels.append("Observed frontier")
    return fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.54, 0.01),
                      ncol=4 if sft_only else 3, frameon=False, fontsize=6.6,
                      handlelength=1.5, handletextpad=0.35, columnspacing=1.2,
                      labelspacing=0.45, borderaxespad=0)


def render(benches, name, height, methods=methods, *, success_metric="valid_only"):
    assert success_metric in ("valid_only", "overall")
    metric_label = "Overall" if success_metric == "overall" else "Valid-only"
    canonical_name = name.replace("-overall-", "-")
    circle_encoding = "sft-actors" in name
    if circle_encoding:
        styled = []
        for key, label, params, color, marker, hollow in methods:
            if key.startswith(("head-", "lora-")):
                color = "#8246E5" if "gpt55" in key else C["LOOKAHEAD_TEACHER"]
                marker, hollow = ("D" if key.startswith("lora-") else "o"), False
                label = label.replace("ADM-22M", "ADM-23M")
            styled.append((key, label, params, color, marker, hollow))
        methods = [*styled, ("adm-distilled", "ADM-distilled actor", DISTILLED["system_parameters"], C["ADM_4B"], "s", False)]
    compact_sft_mean = canonical_name in ("selection-recipe3-vs-cost-sft-actors-20261104", "selection-recipe3-vs-cost-rl-actors-20261104")
    if compact_sft_mean:
        methods = [(key, label.replace(" ¹", "").replace(" ²", "").replace(" ³", ""), *style) for key, label, *style in methods]
    plt.rcParams.update({"font.family": "Albert Sans", "font.size": 7.5, "xtick.labelsize": 6.8, "ytick.labelsize": 6.8,
                         "text.color": C["INK"], "axes.labelcolor": C["INK"], "axes.titlecolor": C["INK"],
                         "axes.labelpad": 2, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "pdf.fonttype": 42, "savefig.facecolor": "white"})
    fig, axes = plt.subplots(len(benches), 3, figsize=(7, height), sharey="row", squeeze=False)
    for i, (ax_row, bench) in enumerate(zip(axes, benches)):
        for ax, (k, xlabel) in zip(ax_row, cols):
            panel(ax, bench, k, methods, circle_encoding=circle_encoding, success_metric=success_metric)
            # Keep frontier rings whole at the edge of short panels.
            for collection in ax.collections:
                collection.set_clip_on(False)
            if i == 0:
                title = {"params": "System parameters (B)", "seconds": "Model latency (s)"}.get(k, xlabel) if compact_sft_mean else xlabel
                ax.set_title(title, fontsize=7.5, pad=4)
        ax_row[0].set_ylabel(f"{metric_label}\nsuccess (%)" if compact_sft_mean else
                             (f"{NAMES[bench]}\n{metric_label.lower()} (%)" if len(benches) > 1 else f"{NAMES[bench]}\n{metric_label.lower()} success (%)"))
    bottom_legend = legend(fig, axes, sft_only=all(not m[0].startswith("rl") for m in methods))
    fig.canvas.draw()
    # Reserve the measured legend height plus room for ticks and a clear gap.
    legend_height = bottom_legend.get_window_extent(fig.canvas.get_renderer()).height / fig.dpi
    fig.subplots_adjust(left=0.085, right=0.99,
                        bottom=(legend_height + 0.26) / height,
                        top=1 - 0.18 / height, hspace=0.32, wspace=0.17)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_bounds = bottom_legend.get_window_extent(renderer)
    assert fig.bbox.contains(*legend_bounds.p0) and fig.bbox.contains(*legend_bounds.p1)
    for ax in axes.flat:
        assert not legend_bounds.overlaps(ax.get_tightbbox(renderer)), "Legend overlaps panel labels"
    labels = [bottom_legend.get_title(), *bottom_legend.texts]
    for ax in axes.flat:
        labels.extend([ax.title, ax.xaxis.label, ax.yaxis.label, *ax.texts])
        for axis, limits in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            # Locators retain off-range tick objects which are never drawn.
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if min(limits) <= tick.get_loc() <= max(limits):
                    labels.extend([tick.label1, tick.label2])
    for label in labels:
        if label.get_visible() and label.get_text():
            bounds = label.get_window_extent(renderer)
            assert fig.bbox.contains(*bounds.p0) and fig.bbox.contains(*bounds.p1), label.get_text()
    for ext in ("png", "svg", "pdf"):
        path = HERE / f"{name}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.025)
        if ext == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)
    return {bench: {k: frontier([(p / 1e9 if k == "params" else point(key, bench, success_metric)[k], point(key, bench, success_metric)["success"])
                                 for key, _, p, *_ in methods]) for k, _ in cols} for bench in benches}


if __name__ == "__main__":
    print(render(("mean",), "selection-recipe3-vs-cost-20261104", 2.20))
    render(BENCH, "selection-recipe3-vs-cost-per-benchmark-20261104", 4.35)
    sft = [m for m in methods if not m[0].startswith("rl")]
    print(render(("mean",), "selection-recipe3-vs-cost-sft-actors-20261104", 1.60, sft))
    render(BENCH, "selection-recipe3-vs-cost-per-benchmark-sft-actors-20261104", 4.35, sft)
    render(("mean",), "selection-recipe3-vs-cost-sft-actors-overall-20261104", 1.60, sft, success_metric="overall")
    render(BENCH, "selection-recipe3-vs-cost-per-benchmark-sft-actors-overall-20261104", 4.35, sft, success_metric="overall")
    rl = [m for m in methods if m[0].startswith("rl")]
    print(render(("mean",), "selection-recipe3-vs-cost-rl-actors-20261104", 1.60, rl))
