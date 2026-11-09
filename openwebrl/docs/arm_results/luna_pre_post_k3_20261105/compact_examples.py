#!/usr/bin/env python3
"""Compact two-example figure (paper, about 1/3 page): (a) the screen right after an action already separates the
candidates (Netflix help page); (b) only the actor's next steps do (arXiv search box vs going back).

One column grid for both examples: current screen (what every teacher sees) | candidate action | screen right after the
action | the actor's next two actions and their screens (lookahead H = 4) | teacher picks and
held-out success. Success = successful share of held-out continuations (draws 4-8, valid only); an action the actor
proposed k times (merged duplicates) has 5k held-out continuations. Same saved screenshots as render_examples.py.

  compact_examples.py   (CPU only; writes examples-compact.{png,pdf} next to this file)
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_examples import (R, SRC, INK, MUTED, GOOD, BAD, read, img, attempt_for, lookahead,  # noqa: E402
                             held, picks, calls)

from paper_plot_palette import COLORS as C, tint

HERE = Path(__file__).resolve().parent
COLOR = {k: c for k, _, c in SRC}; NAME = {k: n for k, n, _ in SRC}
NAME["pre"], NAME["k3"] = "Pre-execution", "Lookahead H = 4"
BAND, BANDEDGE = tint("LOOKAHEAD_EVIDENCE", 0.84), C["LOOKAHEAD_EVIDENCE"]
GOODC, BADC = C["SUCCESS"], C["FAILURE"]
EXAMPLES = [
    dict(letter="a", title="The screen right after the action decides",
         task="Find out what payment methods Netflix accepts.", site="help.netflix.com",
         sid="13d42f1f406049a59cfd502f38a90b09", cur=(0.0, 0.3, 0.56, 0.97), thumb=(0.0, 0.0, 0.64, 0.6),
         decide=[0],
         cands=[(1, "click “Netflix\npayment options”", ["answer page opens", "scroll down", "scroll down"]),
                (0, "scroll down", ["more billing text", "scroll up", "click: site can't be reached"])]),
    dict(letter="b", title="Only the next steps decide",
         task="Find the latest research papers on machine learning and deep learning …", site="arxiv.org",
         sid="72dbd347d60817b95b0e603b", cur=(0.0, 0.0, 0.86, 1.0), thumb=(0.0, 0.0, 0.64, 0.6),
         decide=[1, 2],
         cands=[(1, "click “search”", ["search box opens", "type “machine learning”", "Enter: 353,132 results"]),
                (0, "go back", ["arXiv home page", "click: same listing again", "go back: loop"])]),
]


def state(sid):
    for line in open(R / "actions.jsonl"):
        if sid in line[:80]:
            x = json.loads(line)
            if x["state_id"] == sid:
                return x
    raise KeyError(sid)


def crop(im, box):
    W, H = im.size; l, t, r, b = box
    return im.crop((int(l * W), int(t * H), int(r * W), int(b * H)))


def pic(fig, rect, im, edge=C["BORDER"], lw=0.6):
    ax = fig.add_axes(rect); ax.imshow(im, aspect="auto"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(edge); sp.set_linewidth(lw)
    return ax


def chip(fig, x, y, w, h, text, color, fs=6.0):
    fig.add_artist(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.008", fc=color, ec="none",
                                  transform=fig.transFigure, alpha=0.95))
    fig.text(x + w / 2, y + h / 2, text, fontsize=fs, color=INK, fontweight="bold", ha="center", va="center",
             linespacing=1.05)


def arrow(fig, x0, x1, y, color=C["BASELINE_DARK"]):
    fig.add_artist(FancyArrowPatch((x0, y), (x1, y), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=7,
                                   lw=0.9, color=color))


def main():
    plt.rcParams.update({"font.family": "Albert Sans", "savefig.facecolor": "white", "pdf.fonttype": 42})
    fig = plt.figure(figsize=(7.0, 3.15))
    xc, wc = 0.0, 0.215                 # current screen
    xa, wa = 0.229, 0.112              # action chip
    xt, wt, gt = 0.365, 0.143, 0.008   # three screens: right after the action, next action 1, next action 2
    xp = xt + 3 * wt + 2 * gt + 0.03   # picks / success (after a "…" marking further lookahead)
    th, rowgap = 0.13, 0.06
    heads = [(xc + wc / 2, "Current screen"), (xa + wa / 2, "Action"), (xt + wt / 2, "Lookahead H = 1"),
             (xt + 1.5 * wt + 1.5 * gt + wt / 2, "Lookahead H = 4"), (xp + 0.055, "Picked by")]
    for x, t in heads:
        fig.text(x, 0.995, t, fontsize=6.4, color=INK, ha="center", va="top", fontweight="bold", linespacing=1.1)
    for x0, x1 in ((xt, xt + wt), (xt + wt + gt, xt + 3 * wt + 2 * gt)):
        fig.add_artist(plt.Line2D([x0, x1], [0.955, 0.955], color=C["BORDER"], lw=0.6, transform=fig.transFigure))
    y = 0.895
    for ex in EXAMPLES:
        x = state(ex["sid"]); root = Path(x["root"]); p = picks(x)
        anc = read(read(root / "states" / ex["sid"] / "anchor-reference.json")["path"]); depth = anc["depth"]
        block_h = 2 * th + rowgap
        fig.text(xc, y + 0.008, ex["letter"], fontsize=8.6, fontweight="bold", color=INK, va="bottom")
        fig.text(xc + 0.022, y + 0.008, f"Task: “{ex['task']}” ({ex['site']}, step {depth + 1})",
                 fontsize=6.1, color=MUTED, va="bottom")
        cur = img(x["image"]); W, H = cur.size; l, t, r, b = ex["cur"]
        c0, c1 = min(ex["decide"]), max(ex["decide"])       # highlight the evidence that decides this example
        bx0 = xt + c0 * (wt + gt) - 0.006; bx1 = xt + c1 * (wt + gt) + wt + 0.006
        fig.add_artist(FancyBboxPatch((bx0, y - block_h - 0.045), bx1 - bx0, block_h + 0.05,
                                      boxstyle="round,pad=0,rounding_size=0.01", fc=BAND, ec=BANDEDGE, lw=0.8,
                                      transform=fig.transFigure, zorder=-5))
        fig.text(bx1 - 0.006, y - block_h - 0.04, "decides", fontsize=6.2, color=INK, fontweight="bold", ha="right", va="bottom")
        ax = pic(fig, [xc, y - block_h, wc, block_h], crop(cur, ex["cur"]))
        for k, (i, label, caps) in enumerate(ex["cands"]):
            col = C["ACTION"]
            c = calls(anc["candidates"][x["actions"][i]["teacher_exec"]]["output"][0])[0]
            pt = (c.get("arguments") or {}).get("point_2d")
            if c.get("name") == "click" and pt:            # box the click target on the current screen
                px, py = (pt[0] / 1000 - l) * W, (pt[1] / 1000 - t) * H
                ax.add_patch(Rectangle((px - 0.075 * W, py - 0.03 * H), 0.15 * W, 0.06 * H, fill=False, ec=col, lw=1.6))
            if c.get("name") == "scroll":
                ax.annotate("", (0.93, 0.08), (0.93, 0.32), xycoords="axes fraction",
                            arrowprops=dict(arrowstyle="-|>", color=col, lw=1.6, mutation_scale=9))
            ry = y - (k + 1) * th - k * rowgap
            chip(fig, xa, ry + th * 0.18, wa, th * 0.64, label, col)
            arrow(fig, xa + wa + 0.002, xt - 0.004, ry + th / 2, col)
            att, imm = attempt_for(root, ex["sid"], x["actions"][i]["teacher_exec"])
            shots = [img(imm["observation"]["screenshot"]["path"])]
            shots += [img(s["shot"]) for s in lookahead(att, depth)[:2] if s["shot"]]
            for j in range(3):
                rx = xt + j * (wt + gt)
                if j < len(shots):
                    key = j in ex["decide"]
                    a_ = pic(fig, [rx, ry, wt, th], crop(shots[j], ex["thumb"]), edge=C["LOOKAHEAD_EVIDENCE"] if key else C["BORDER"],
                             lw=1.3 if key else 0.5)
                    if not key:                    # fade evidence that does not decide this example
                        a_.add_patch(Rectangle((0, 0), 1, 1, transform=a_.transAxes, fc="white", alpha=0.55, ec="none"))
                    fig.text(rx + 0.003, ry - 0.006, caps[j] if j == 0 else f"{j}. {caps[j]}", fontsize=5.8,
                             color=INK if key else C["MUTED"], va="top", fontweight="bold" if key else "normal")
                elif j == 1:
                    fig.text(rx + wt + gt / 2, ry + th / 2, "not needed: the screen right after\nthe action already differs",
                             fontsize=5.8, color=MUTED, ha="center", va="center", style="italic")
            fig.text(xt + 3 * wt + 2 * gt + 0.004, ry + th / 2, "…", fontsize=10, color=C["MUTED"], va="center")
            bx = xp
            for kk in [kk for kk in ("pre", "k3") if p[kk].count(i) == 2]:
                t_ = fig.text(bx, ry + th * 0.7, NAME[kk], fontsize=5.8, color="white" if kk == "pre" else INK, fontweight="bold", va="center",
                              bbox=dict(boxstyle="round,pad=0.22,rounding_size=0.4", fc=COLOR[kk], ec="none"))
                fig.canvas.draw(); bx = t_.get_window_extent().transformed(fig.transFigure.inverted()).x1 + 0.006
            s, n = held(x["actions"][i])
            fig.text(xp, ry + th * 0.25, f"{100 * s / n:.0f}% success", fontsize=6.6, fontweight="bold",
                     color=GOOD if s / n > 0.5 else BAD, va="center")
        y = y - block_h - 0.125
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"examples-compact.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.02)


if __name__ == "__main__":
    main()
