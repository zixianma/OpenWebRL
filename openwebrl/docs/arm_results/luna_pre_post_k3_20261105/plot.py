#!/usr/bin/env python3
"""Pre-exec vs post-exec vs K = 3 lookahead teacher picks (GPT-6 Luna, same 1,727 branch states): four panels from stats.json.
(a) agreement between the sources' picks (call-matched) and, on the diagonal, each source with itself across its two calls;
(b) the same pairwise agreement by the number of distinct candidate actions, with chance (1 / actions);
(c) action type of the picks (both calls, all states) against the candidates offered;
(d) held-out success (draws 3-7, valid only) of each teacher's pick minus one SFT sample on the fixed five, and pairwise
    differences on the states where two teachers' picks differ; task-clustered 95% intervals (analysis.py).
"""
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
import paper_plot_style as _paper_style  # noqa: E402,F401

HERE = Path(__file__).resolve().parent
S = json.loads((HERE / "stats.json").read_text())
INK, MUTED, GRID = "#222222", "#666666", "#e6e6e6"
SRC = [("pre", "Pre-exec", "#4B5563"), ("post", "Post-exec", "#D8877E"), ("k3", "Lookahead K = 3", "#6250BE")]
ORACLE = "#397FA3"


def style(ax):
    ax.tick_params(colors="#444444", length=3, width=0.6, pad=2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#555555"); ax.spines[side].set_linewidth(0.6)


def title(ax, s, text):
    ax.set_title(text, fontsize=7.2, color=INK, pad=4, loc="left")
    ax.annotate(s, (0, 1), xycoords="axes fraction", xytext=(-4, 4), textcoords="offset points", fontsize=8.5,
                fontweight="bold", color=INK, ha="right", va="bottom")


def main():
    plt.rcParams.update({"font.family": "Albert Sans", "font.size": 7.2, "xtick.labelsize": 6.6, "ytick.labelsize": 6.6,
                         "axes.labelpad": 2, "svg.fonttype": "none", "pdf.fonttype": 42, "savefig.facecolor": "white"})
    fig = plt.figure(figsize=(7, 4.9))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.25], height_ratios=[1, 1.1], wspace=0.62, hspace=0.62,
                          left=0.09, right=0.985, top=0.93, bottom=0.07)

    # (a) agreement matrix
    ax = fig.add_subplot(gs[0, 0]); A = S["agreement"]["all"]
    keys = [k for k, *_ in SRC]; names = [n.replace("Lookahead ", "Lookahead\n") for _, n, _ in SRC]
    M = [[A.get(f"{a}~{b}", A.get(f"{b}~{a}")) for b in keys] for a in keys]
    im = ax.imshow(M, cmap="Purples", vmin=30, vmax=95)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{M[i][j]:.0f}%", ha="center", va="center", fontsize=7.4,
                    color="white" if M[i][j] > 72 else INK, fontweight="bold" if i == j else "normal")
    ax.set_xticks(range(3), names); ax.set_yticks(range(3), names)
    ax.tick_params(length=0, pad=3)
    for sp in ax.spines.values():
        sp.set_visible(False)
    title(ax, "a", f"Same pick (%), {S['states']:,} states\ndiagonal: same teacher, two calls; chance {A['chance pair']:.0f}%")

    # (b) agreement by number of distinct actions
    ax = fig.add_subplot(gs[0, 1]); style(ax); by = S["agreement"]["by_actions"]; n = sorted(by, key=int)
    x = [int(k) for k in n]
    for pair, label, c, m in (("post~k3", "Post-exec ~ K = 3", "#9A6FB0", "o"), ("pre~post", "Pre-exec ~ Post-exec", "#B98A86", "s"),
                              ("pre~k3", "Pre-exec ~ K = 3", "#4B5563", "D")):
        ax.plot(x, [by[k][pair] for k in n], marker=m, ms=3.6, lw=1.1, color=c, label=label)
    ax.plot(x, [by[k]["chance pair"] for k in n], ls="--", lw=0.9, color=MUTED, label="Chance (1 / actions)")
    ax.fill_between(x, [min(by[k][f"{s}~{s}"] for s in keys) for k in n], [max(by[k][f"{s}~{s}"] for s in keys) for k in n],
                    color="#cfcfe6", alpha=0.6, lw=0, label="Same teacher, two calls")
    ax.set_xticks(x, [f"{k}\n({by[k]['states']:,})" for k in n]); ax.set_xlabel("Distinct candidate actions (states)")
    ax.set_ylabel("Same pick (%)"); ax.set_ylim(10, 122); ax.set_yticks(range(20, 101, 20)); ax.spines["left"].set_bounds(10, 100); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=6.0, loc="upper left", handlelength=1.6, labelspacing=0.25, columnspacing=1.0, ncol=2, borderaxespad=0.1)
    title(ax, "b", "Post-exec agrees more with K = 3 than with pre-exec")

    # (c) action types
    ax = fig.add_subplot(gs[1, 0]); style(ax); T = S["action_types"]["all"]
    types = ["click", "scroll", "type", "back", "stop / answer", "navigate (URL)"]
    h = 0.2; y0 = range(len(types))
    for i, (k, name, c) in enumerate(SRC):
        ax.barh([y + (i - 1) * h for y in y0], [T[k][t] for t in types], height=h, color=c, label=name)
    ax.scatter([T["offered"][t] for t in types], [y + 1.75 * h for y in y0], marker="|", s=34, color=INK, lw=1.0,
               label="Offered candidates", zorder=3)
    ax.set_yticks(list(y0), types); ax.invert_yaxis(); ax.set_xlabel("Share of picks (%)")
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=6.2, loc="lower right", handlelength=1.2, labelspacing=0.3)
    title(ax, "c", "Pre-exec over-picks scroll")

    # (d) outcomes
    ax = fig.add_subplot(gs[1, 1]); style(ax); O = S["outcomes"]; H = O["head_to_head"]
    rows = [(f"{n} − SFT sample", O[k]["minus_sft"], c) for k, n, c in SRC] + [("Oracle − SFT sample", O["oracle"]["minus_sft"], ORACLE)]
    rows += [(f"Post − Pre  ({H['post-pre']['states']} states)", H["post-pre"]["diff"], "#B98A86"),
             (f"K = 3 − Post  ({H['k3-post']['states']} states)", H["k3-post"]["diff"], "#9A6FB0"),
             (f"K = 3 − Pre  ({H['k3-pre']['states']} states)", H["k3-pre"]["diff"], "#6250BE")]
    ys = [0, 1, 2, 3, 5.0, 6.0, 7.0]
    for y, (label, (m, lo, hi), c) in zip(ys, rows):
        ax.plot([lo, hi], [y, y], color=c, lw=1.4, solid_capstyle="round")
        ax.scatter([m], [y], s=22, color=c, zorder=3, edgecolor="white", linewidth=0.5)
        ax.text(hi + 0.12, y, f"{m:+.1f}", va="center", fontsize=6.4, color=INK)
    ax.axvline(0, color=MUTED, lw=0.7)
    ax.set_yticks(ys, [r[0] for r in rows]); ax.set_xlim(-1.2, 5.0)
    ax.axhline(3.9, color=GRID, lw=0.8)
    ax.set_ylim(7.5, -0.9)
    ax.text(4.95, -0.55, f"all {O['states']:,} states", ha="right", va="center", fontsize=6.2, color=MUTED)
    ax.text(4.95, 4.35, "states where the two picks differ", ha="right", va="center", fontsize=6.2, color=MUTED)
    ax.set_xlabel("Held-out success of the pick, pp (draws 4–8, valid only; 95% CI)")
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    title(ax, "d", "Most of the gain comes from the immediate result")

    for ext in ("png", "svg", "pdf"):
        path = HERE / f"pre-post-k3-picks.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.03)
        if ext == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")


if __name__ == "__main__":
    main()
