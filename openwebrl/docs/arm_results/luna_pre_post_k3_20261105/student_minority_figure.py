#!/usr/bin/env python3
"""Compact figure: two eval states where a minority proposal picked by students is the sensible move (M2, M5).

Each row: the shared screen with the minority pick and the most-proposed action marked, the next screen after the minority
pick (from one matched live episode), and short excerpts of each action's own reasoning with the students that picked it.
No success counts: the final judge verdicts on these tasks are unreliable (see the rollouts page), so the figure argues only
from the decisions.

  student_minority_figure.py  (CPU only; writes minority-examples.png to $OUT_DIR or next to this file)
"""
import json
import os
from pathlib import Path
import textwrap

from PIL import Image, ImageDraw
import sys
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "paper_plot_palette.py").is_file())))
from paper_plot_palette import COLORS as C, tint

from student_outcome_examples import F, INK, K3, MUTED, PRE, RD, calls, fit, images, png

HERE = Path(__file__).resolve().parent
MIN, MAJ = C["ACTION"], tint("ACTION", 0.4)
WHO = {"gpt55-pre": ("Pre GPT-5.5", PRE), "luna-pre": ("Pre Luna", PRE), "gpt55-k3": ("Lookahead GPT-5.5", K3),
       "luna-k3": ("Lookahead Luna", K3)}
EX = [  # state, title, minority pick, majority action, minority excerpt, majority excerpt, (episode root, run) for the next screen
    (2809, "(a) Hugging Face, step 6: find a sentiment-analysis model last updated in March 2025",
     'click the "Files and versions" tab', "go back to the search results",
     'I notice there\'s a "Files and versions" tab in the navigation menu. This tab likely contains version history, '
     "including the last update date.",
     "Since I can't find the last update date on this model page, and the previous scrolls didn't reveal it … The most "
     "logical next step is to go back to the search results and try a different sentiment analysis model.",
     ("p16-lora-lunapre-20261103", "run1")),
    (7700, "(b) arXiv home, step 1: count yesterday's 'autonomous vehicles' papers in Electrical Engineering and Systems Science",
     "scroll down the subject list", "click the site search icon",
     "I can see sections for Physics, Mathematics, Computer Science, and Quantitative Biology … there might be more sections "
     "below the current view. Let me scroll down first to see if there are more categories.",
     "In arXiv, electrical engineering and systems science is typically in the cs.CV … or cs.RO … I should click on that "
     'search icon to open the search interface where I can enter the keyword "autonomous vehicles".',
     ("sameday-lora-20261104", "run1")),
]
W = 2000


def mark(im, a, col, tag):
    d = ImageDraw.Draw(im); X, Y = im.size
    for c in calls(a):
        g = c.get("arguments") or {}
        if c.get("name") == "click" and g.get("point_2d"):
            x, y = g["point_2d"][0] * X / 1000, g["point_2d"][1] * Y / 1000
            d.rectangle([x - 70, y - 26, x + 70, y + 26], outline=col, width=9)
            d.rectangle([x + 70, y - 26, x + 110, y + 18], fill=col); d.text((x + 80, y - 26), tag, fill=INK, font=F(36, True))
        elif c.get("name") == "scroll":
            t = f"{tag}  scroll down"; w = d.textlength(t, font=F(44, True)) + 40
            d.rectangle([X - 20 - w, Y - 90, X - 20, Y - 20], fill=col); d.text((X - w, Y - 82), t, fill=INK, font=F(44, True))
        elif c.get("name") == "go_back":
            t = f"{tag}  go back"; w = d.textlength(t, font=F(44, True)) + 40
            d.rectangle([20, 20, 20 + w, 90], fill=col); d.text((40, 28), t, fill=INK, font=F(44, True))
    return im


def badges(d, x, y, who, wmax):
    bx = x
    for s in who:
        t, c = WHO[s]; w = d.textlength(t, font=F(26, True)) + 18
        if bx + w > wmax:
            bx, y = x, y + 40
        d.rounded_rectangle([bx, y, bx + w, y + 34], 8, fill=C["ADM_4B"], outline=c, width=3); d.text((bx + 9, y + 2), t, fill="white", font=F(26, True)); bx += w + 10
    return y + 46


def para(d, x, y, text, width, font, fill):
    for ln in textwrap.wrap(text, width):
        d.text((x, y), ln, fill=fill, font=font); y += font.size + 10
    return y


def row(i, title, pick_lbl, maj_lbl, pick_q, maj_q, ep):
    rows = {r["i"]: r for r in map(json.loads, open(RD / "candidate-outcomes.jsonl")) if r["i"] == i}; r = rows[i]
    st = [json.loads(l) for l in open(RD / "states.jsonl")][i]
    c = {x["turn"]: x for x in map(json.loads, open(r["trace"]))}[r["turn"]]["candidates"]
    maj = max(r["groups"], key=lambda g: g["size"])
    pick = next(g for g in r["groups"] if g is not maj and g["picked_by"])
    cur = png(images(st["trace"]), r["turn"])
    mark(cur, c[maj["members"][0]]["action"], MAJ, "B"); mark(cur, c[pick["members"][0]]["action"], MIN, "A")
    e = next(e for e in pick["episodes"] if (e["root"], e["run"]) == ep)
    nxt = png(images(e["trace"]), r["turn"] + 1)
    canvas = Image.new("RGB", (W, 2200), "white"); d = ImageDraw.Draw(canvas)
    y = para(d, 30, 20, title, 95, F(40, True), INK) + 10
    sw = 820; a = fit(cur, sw); b = fit(nxt, 520)
    canvas.paste(a, (30, y)); d.rectangle([30, y, 30 + sw, y + a.size[1]], outline=C["STATE_BORDER"], width=2)
    d.text((30, y + a.size[1] + 8), "Current screen (A = minority pick, B = most-proposed action)", fill=MUTED, font=F(26))
    bx = 30 + sw + 30
    canvas.paste(b, (bx, y)); d.rectangle([bx, y, bx + 520, y + b.size[1]], outline=C["STATE_BORDER"], width=2)
    for k, ln in enumerate(textwrap.wrap("Next screen after A, from one live episode", 26)):
        d.text((bx, y + b.size[1] + 8 + 34 * k), ln, fill=MUTED, font=F(26))
    tx = bx + 520 + 40; tw = W - tx - 30; yy = y
    for tag, col, lbl, g, q in (("A", MIN, pick_lbl, pick, pick_q), ("B", MAJ, maj_lbl, maj, maj_q)):
        d.rectangle([tx, yy, tx + 44, yy + 44], fill=col); d.text((tx + 10, yy), tag, fill=INK, font=F(36, True))
        yy = para(d, tx + 58, yy, lbl, 21, F(34, True), INK)
        d.text((tx + 58, yy), f"{g['size']} of 5 proposals · picked by", fill=MUTED, font=F(26)); yy += 40
        yy = badges(d, tx + 58, yy, g["picked_by"], W - 30)
        yy = para(d, tx, yy, f"“{q}”", 38, F(26), INK) + 26
    h = max(y + a.size[1] + 50, yy) + 20
    return canvas.crop((0, 0, W, h))


def main():
    parts = [row(*x) for x in EX]
    out = Image.new("RGB", (W, sum(p.size[1] for p in parts) + 30 * (len(parts) - 1)), "white"); y = 0
    d = ImageDraw.Draw(out)
    for k, p in enumerate(parts):
        out.paste(p, (0, y)); y += p.size[1]
        if k < len(parts) - 1:
            d.line([30, y + 14, W - 30, y + 14], fill=C["BORDER"], width=3); y += 30
    out.thumbnail((1600, 10000))
    out.save(Path(os.environ.get("OUT_DIR", HERE)) / "minority-examples.png", optimize=True)


if __name__ == "__main__":
    main()
