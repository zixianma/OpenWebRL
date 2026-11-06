#!/usr/bin/env python3
"""Compact two-example figure (paper, about 1/3 page): (a) the immediate post-execution result fixes a pre-execution guess
(Netflix help page); (b) three lookahead steps reveal what one result cannot (arXiv search box vs going back).
Same saved screenshots, picks and held-out success (draws 4-8, valid only) as render_examples.py.

  compact_examples.py   (CPU only; writes examples-compact.{png,pdf} next to this file)
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_examples import (R, SRC, CAND, INK, MUTED, GOOD, BAD, read, img, attempt_for, lookahead,  # noqa: E402
                             held, picks, calls)

HERE = Path(__file__).resolve().parent
NETFLIX, ARXIV = "13d42f1f406049a59cfd502f38a90b09", "72dbd347d60817b95b0e603b"
COLOR = {k: c for k, _, c in SRC}; NAME = {k: n for k, n, _ in SRC}


def crop(im, top=0.0, bottom=0.62, left=0.0, right=1.0):
    W, H = im.size
    return im.crop((int(left * W), int(top * H), int(right * W), int(bottom * H)))


def thumb(fig, rect, im, title, sub=None, edge="#bbbbbb", lw=0.5):
    ax = fig.add_axes(rect); ax.imshow(im, aspect="auto"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(edge); sp.set_linewidth(lw)
    ax.set_title(title, fontsize=6.2, color=INK, pad=2, loc="left")
    if sub:
        ax.text(0, -0.05, sub, transform=ax.transAxes, fontsize=5.8, color=MUTED, va="top")
    return ax


def badges(fig, x, y, keys, gap=0.006):
    for k in keys:
        t = fig.text(x, y, NAME[k], fontsize=5.6, color="white", fontweight="bold", va="center",
                     bbox=dict(boxstyle="round,pad=0.22,rounding_size=0.4", fc=COLOR[k], ec="none"))
        fig.canvas.draw(); x = t.get_window_extent().transformed(fig.transFigure.inverted()).x1 + gap
    return x


def outcome(fig, x, y, s, n, good):
    fig.text(x, y, f"{s}/{n} succeed", fontsize=6.4, fontweight="bold", color=GOOD if good else BAD, va="center")


def marker(ax, W, H, pt, i, crop_box):
    l, t, r, b = crop_box
    px, py = pt[0] / 1000 * W - l * W, pt[1] / 1000 * H - t * H
    ax.scatter([px], [py], s=46, color=CAND[i], edgecolor="white", linewidth=0.8, zorder=3)
    ax.text(px, py, str(i + 1), fontsize=5.2, color="white", ha="center", va="center", fontweight="bold", zorder=4)


def state(sid):
    for line in open(R / "actions.jsonl"):
        if sid in line[:80]:
            x = json.loads(line)
            if x["state_id"] == sid:
                return x
    raise KeyError(sid)


def main():
    plt.rcParams.update({"font.family": "Albert Sans", "savefig.facecolor": "white", "pdf.fonttype": 42})
    fig = plt.figure(figsize=(7.0, 4.0))
    L = 0.005

    def head(y, letter, title, task):
        fig.text(L, y, letter, fontsize=9, fontweight="bold", color=INK, va="top")
        fig.text(L + 0.022, y, title, fontsize=7.4, fontweight="bold", color=INK, va="top")
        fig.text(L + 0.022, y - 0.04, task, fontsize=6.2, color=MUTED, va="top")

    def group(x0, x1, y, text):
        fig.add_artist(plt.Line2D([x0, x1], [y - 0.012, y - 0.012], color="#999999", lw=0.6, transform=fig.transFigure))
        fig.text((x0 + x1) / 2, y, text, fontsize=5.9, color=MUTED, ha="center", va="bottom")

    # (a) Netflix: post-exec sees that the link opens the payment-options article; pre-exec scrolls.
    x = state(NETFLIX); root = Path(x["root"]); p = picks(x)
    anc = read(read(root / "states" / NETFLIX / "anchor-reference.json")["path"])
    head(0.995, "a", "Post-execution evidence fixes a guess",
         "Task: “Find out what payment methods Netflix accepts.”  (help.netflix.com, step 4)")
    before = img(x["image"]); W, H = before.size; box = (0.0, 0.3, 0.52, 0.9)
    tw, th, y0 = 0.29, 0.30, 0.555
    ax = thumb(fig, [L, y0, tw, th], crop(before, box[1], box[3], box[0], box[2]), "Current screen (all teachers see)")
    for i, a in enumerate(x["actions"]):
        c = calls(anc["candidates"][a["teacher_exec"]]["output"][0])[0]
        pt = (c.get("arguments") or {}).get("point_2d")
        if pt:
            marker(ax, W, H, pt, i, box)
            if c.get("name") == "scroll":
                l, t = box[0], box[1]; px, py = pt[0] / 1000 * W - l * W, pt[1] / 1000 * H - t * H
                ax.annotate("", (px, py + 0.12 * H), (px, py + 0.03 * H), arrowprops=dict(arrowstyle="-|>", color=CAND[i], lw=1.2))
    xs = [0.345, 0.68]
    group(xs[0], xs[1] + tw, y0 + th + 0.045, "Immediate result of each candidate (post-exec and K = 3 see this; pre-exec does not)")
    for j, (i, cap) in enumerate(((0, "1  scroll down: more billing text"), (1, "2  click “Netflix payment options”: answer page"))):
        _, imm = attempt_for(root, NETFLIX, x["actions"][i]["teacher_exec"])
        s, n = held(x["actions"][i])
        thumb(fig, [xs[j], y0, tw, th], crop(img(imm["observation"]["screenshot"]["path"]), 0.0, 0.6, 0.0, 0.52), cap, edge=CAND[i], lw=1.1)
        e = badges(fig, xs[j], y0 - 0.04, [k for k in ("pre", "post", "k3") if p[k].count(i) == 2])
        outcome(fig, e + 0.01, y0 - 0.04, s, n, s / n > 0.5)

    # (b) arXiv: one step after "click search" shows only an open box; the next steps reach results. Going back loops.
    x = state(ARXIV); root = Path(x["root"]); p = picks(x)
    anc = read(read(root / "states" / ARXIV / "anchor-reference.json")["path"]); depth = anc["depth"]
    head(0.445, "b", "Lookahead steps reveal what one result cannot",
         "Task: “Find the latest research papers on machine learning and deep learning …”  (arxiv.org, step 7)")
    rows = [(1, ["search box opens", "+1  type “machine learning”", "+2  Enter: 353,132 results"]),
            (0, ["arXiv home page", "+1  click: same listing again", "+2  go back: loop"])]
    bw, bh, x0, gap = 0.265, 0.125, 0.16, 0.012
    group(x0, x0 + bw, 0.33, "Immediate result (post-exec, K = 3)")
    group(x0 + bw + gap, x0 + 3 * bw + 2 * gap, 0.33, "Next two actor steps (K = 3 only)")
    for r, (i, caps) in enumerate(rows):
        att, imm = attempt_for(root, ARXIV, x["actions"][i]["teacher_exec"])
        steps = lookahead(att, depth)
        shots = [img(imm["observation"]["screenshot"]["path"])] + [img(s["shot"]) for s in steps[:2]]
        y = 0.16 - r * 0.165
        s, n = held(x["actions"][i])
        fig.text(L, y + bh * 0.82, "2  click “search”" if i == 1 else "1  go back", fontsize=6.4, color=INK, va="center", fontweight="bold")
        badges(fig, L, y + bh * 0.45, [k for k in ("pre", "post", "k3") if p[k].count(i) == 2])
        fig.text(L, y + bh * 0.1, f"{s}/{n} succeed", fontsize=6.4, fontweight="bold", color=GOOD if s / n > 0.5 else BAD, va="center")
        for c, (im, cap) in enumerate(zip(shots, caps)):
            thumb(fig, [x0 + c * (bw + gap), y, bw, bh], crop(im, 0.0, 0.62), cap,
                  edge=CAND[i] if c == 0 else "#bbbbbb", lw=1.1 if c == 0 else 0.5)
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"examples-compact.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.02)


if __name__ == "__main__":
    main()
