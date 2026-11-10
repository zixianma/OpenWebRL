#!/usr/bin/env python3
"""Compact two-example figure (paper, about 1/3 page): (a) the screen right after an action already separates the
candidates (Netflix help page); (b) only the actor's next steps do (arXiv search box vs going back).

One column grid for both examples: current screen (what every teacher sees) | candidate action | screen right after the
action | the actor's next two actions and their screens (lookahead H = 4) | teacher picks and
held-out success. Success = successful share of held-out continuations (draws 4-8, valid only); an action the actor
proposed k times (merged duplicates) has 5k held-out continuations. Same saved screenshots as render_examples.py.
Indices 1/2 label the two displayed candidate rows, independently within each example;
they do not replace the saved candidate indices. Salmon frames mark the evidence
that separates the choices. Only two later actions are shown; the ellipsis retains
the remaining H=4 continuation.

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
COLOR = {k: c for k, _, c in SRC}
NAME = {"pre": "Pre-execution", "k3": "Lookahead H=4"}
BAND, BANDEDGE = tint("LOOKAHEAD_EVIDENCE", 0.92), C["LOOKAHEAD_EVIDENCE"]
GOODC, BADC = C["SUCCESS"], C["FAILURE"]
EXAMPLES = [
    dict(letter="a", title="The screen right after the action decides",
         task="Find Netflix's accepted payment methods.", site="help.netflix.com",
         sid="13d42f1f406049a59cfd502f38a90b09", cur=(0.0, 0.3, 0.56, 0.97), thumb=(0.0, 0.0, 0.64, 0.6),
         decide=[0],
         cands=[(1, "Click “Netflix\npayment options”", ["Answer page opens", "Scroll down", "Scroll down"]),
                (0, "Scroll down", ["More billing text", "Scroll up", "Site unavailable"])]),
    dict(letter="b", title="Only the next steps decide",
         task="Find recent machine learning and deep learning papers.", site="arxiv.org",
         sid="72dbd347d60817b95b0e603b", cur=(0.0, 0.0, 0.86, 1.0), thumb=(0.0, 0.0, 0.64, 0.6),
         decide=[1, 2],
         cands=[(1, "Click search", ["Search box opens", "Type “machine\nlearning”", "353,132 results"]),
                (0, "Go back", ["arXiv home page", "Same listing again", "Back again: loop"])]),
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


def chip(fig, x, y, w, h, action_id, label, color, fs=6.0):
    fig.add_artist(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.008",
                                  fc=tint("ACTION", .72), ec=color, lw=.85, transform=fig.transFigure))
    fig.text(x + w / 2, y + h * .76, rf"$a_t^{{{action_id}}}$", fontsize=8, color=color,
             fontweight="bold", ha="center", va="center")
    fig.text(x + w / 2, y + h * .23, label, fontsize=fs, color=INK, ha="center", va="center",
             linespacing=1.05)


def arrow(fig, x0, x1, y, color=C["BASELINE_DARK"]):
    fig.add_artist(FancyArrowPatch((x0, y), (x1, y), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=7,
                                   lw=0.9, color=color))


def main():
    plt.rcParams.update({"font.family": "Albert Sans", "savefig.facecolor": "white", "pdf.fonttype": 42})
    fig = plt.figure(figsize=(7.2, 3.6))
    xc, wc = 0.02, 0.185               # current screen
    xa, wa = 0.227, 0.108              # action chip
    xt, wt, gt = 0.365, 0.127, 0.016   # three screens: candidate result and two later states
    xp = 0.815                         # picks / success; generous label column
    th, rowgap = 0.115, 0.07
    heads = [(xc + wc / 2, r"$s_t$"), (xa + wa / 2, r"$a_t^i$"),
             (xt + (3 * wt + 2 * gt) / 2, "Lookahead (H = 4)"),
             (xp + 0.07, "Teacher choice")]
    for x, t in heads:
        fig.text(x, .950 if t.startswith("$") else .993, t, fontsize=7.5,
                 color=INK, ha="center", va="top", fontweight="bold", linespacing=1.1)
    for j in range(3):
        label = rf"$s_{{t+{j + 1}}}^i$"
        fig.text(xt + j * (wt + gt) + wt / 2, .950, label,
                 fontsize=7.5, color=MUTED, ha="center", va="top")
    y = 0.855
    for ex in EXAMPLES:
        x = state(ex["sid"]); root = Path(x["root"]); p = picks(x)
        anc = read(read(root / "states" / ex["sid"] / "anchor-reference.json")["path"]); depth = anc["depth"]
        block_h = 2 * th + rowgap
        fig.text(xc, y + 0.008, ex["letter"] + ")", fontsize=8.2, fontweight="bold", color=INK, va="bottom")
        fig.text(xc + 0.026, y + 0.008, f"Task: {ex['task']}  ({ex['site']}, step {depth + 1})",
                 fontsize=6.3, color=MUTED, va="bottom")
        cur = img(x["image"]); W, H = cur.size; l, t, r, b = ex["cur"]
        c0, c1 = min(ex["decide"]), max(ex["decide"])       # highlight the evidence that decides this example
        bx0 = xt + c0 * (wt + gt) - 0.006; bx1 = xt + c1 * (wt + gt) + wt + 0.006
        fig.add_artist(FancyBboxPatch((bx0, y - block_h - 0.045), bx1 - bx0, block_h + 0.05,
                                      boxstyle="round,pad=0,rounding_size=0.01", fc=BAND, ec=BANDEDGE, lw=0.8,
                                      transform=fig.transFigure, zorder=-5))
        ax = pic(fig, [xc, y - block_h, wc, block_h], crop(cur, ex["cur"]))
        for k, (i, label, caps) in enumerate(ex["cands"]):
            action_id = str(k + 1)
            col = C["ACTION_INK"] if k == 0 else C["ACTION"]
            c = calls(anc["candidates"][x["actions"][i]["teacher_exec"]]["output"][0])[0]
            pt = (c.get("arguments") or {}).get("point_2d")
            if c.get("name") == "click" and pt:            # box the click target on the current screen
                px, py = (pt[0] / 1000 - l) * W, (pt[1] / 1000 - t) * H
                ax.add_patch(Rectangle((px - 0.075 * W, py - 0.03 * H), 0.15 * W, 0.06 * H, fill=False, ec=col, lw=1.6))
                crop_h, crop_w = ax.images[0].get_array().shape[:2]
                badge_x = min(max(px + .075 * W, .06 * crop_w), .98 * crop_w)
                badge_y = min(max(py - .03 * H, .12 * crop_h), .95 * crop_h)
                ax.text(badge_x, badge_y, action_id, ha="right", va="bottom", clip_on=True,
                        color="white", fontsize=6.5, weight="bold",
                        bbox=dict(boxstyle="round,pad=.15", fc=col, ec="white", lw=.5))
            if c.get("name") == "scroll":
                ax.annotate("", (0.93, 0.08), (0.93, 0.32), xycoords="axes fraction",
                            arrowprops=dict(arrowstyle="-|>", color=col, lw=1.6, mutation_scale=9))
                ax.text(.93, .36, action_id, transform=ax.transAxes, ha="center", va="bottom",
                        color="white", fontsize=6.5, weight="bold",
                        bbox=dict(boxstyle="round,pad=.15", fc=col, ec="white", lw=.5))
            ry = y - (k + 1) * th - k * rowgap
            chip(fig, xa, ry + th * 0.02, wa, th * 0.96, action_id, label, col)
            arrow(fig, xa + wa + 0.002, xt - 0.004, ry + th / 2, col)
            att, imm = attempt_for(root, ex["sid"], x["actions"][i]["teacher_exec"])
            shots = [img(imm["observation"]["screenshot"]["path"])]
            shots += [img(s["shot"]) for s in lookahead(att, depth)[:2] if s["shot"]]
            for j in range(3):
                rx = xt + j * (wt + gt)
                if j < len(shots):
                    key = j in ex["decide"]
                    a_ = pic(fig, [rx, ry, wt, th], crop(shots[j], ex["thumb"]), edge=BANDEDGE if key else C["BORDER"],
                             lw=1.3 if key else 0.5)
                    if not key:                    # fade evidence that does not decide this example
                        a_.add_patch(Rectangle((0, 0), 1, 1, transform=a_.transAxes, fc="white", alpha=0.25, ec="none"))
                    if key:
                        fig.text(rx + 0.003, ry - 0.006, caps[j], fontsize=6.0,
                                 color=INK, va="top", fontweight="bold")
                elif j == 1:
                    fig.text(rx + wt + gt / 2, ry + th / 2, "not needed: the screen right after\nthe action already differs",
                             fontsize=5.8, color=MUTED, ha="center", va="center", style="italic")
            fig.text(xt + 3 * wt + 2 * gt + 0.004, ry + th / 2, "…", fontsize=10, color=C["MUTED"], va="center")
            bx = xp
            for kk in [kk for kk in ("pre", "k3") if p[kk].count(i) == 2]:
                t_ = fig.text(bx, ry + th * 0.7, NAME[kk] + ": " + rf"$a_t^{{{action_id}}}$", fontsize=7.2, color="white" if kk == "pre" else INK, fontweight="bold", va="center",
                              bbox=dict(boxstyle="round,pad=0.22,rounding_size=0.4", fc=COLOR[kk], ec="none"))
                fig.canvas.draw(); bx = t_.get_window_extent().transformed(fig.transFigure.inverted()).x1 + 0.006
            s, n = held(x["actions"][i])
            fig.text(xp, ry + th * 0.25, f"{100 * s / n:.0f}% success", fontsize=6.6, fontweight="bold",
                     color=GOOD if s / n > 0.5 else BAD, va="center")
        y = y - block_h - 0.15
    fig.text(xc, .025, "Success: successful valid held-out continuations (rollouts 4–8).",
             fontsize=5.8, color=MUTED, va="bottom")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for i, label in enumerate(fig.texts):
        bounds = label.get_window_extent(renderer)
        assert fig.bbox.contains(*bounds.p0) and fig.bbox.contains(*bounds.p1), label.get_text()
        for other in fig.texts[:i]:
            assert not bounds.overlaps(other.get_window_extent(renderer)), (label.get_text(), other.get_text())
    for ext in ("png", "pdf"):
        metadata = {"CreationDate": None, "ModDate": None} if ext == "pdf" else {}
        fig.savefig(HERE / f"examples-compact.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.09, metadata=metadata)
    plt.close(fig)


if __name__ == "__main__":
    main()
