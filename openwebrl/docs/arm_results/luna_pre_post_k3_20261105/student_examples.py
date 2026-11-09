#!/usr/bin/env python3
"""Compact two-example figure: what the trained students (ADM-4B LoRAs, Recipe 3) pick on the same live eval state.

Both examples are replayed eval states (student_replay.py) on which the two pre-execution students (GPT-5.5 and Luna labels)
pick one action and the two H = 4 lookahead students pick another. Left: the state's screenshot as logged live; right: the
five actor proposals grouped into distinct actions (with how many of the five proposed each) and who picked which.
(a) leaving a scroll loop; (b) answering before the calculator inputs match the task. The rate line under each example is
the same-state rate over all replayed states of that kind (student-replay.json).

  student_examples.py  (CPU only; writes student-examples.{png,pdf} next to this file)
"""
import base64
import glob
import io
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "paper_plot_style.py").is_file())))
import paper_plot_style as _paper_style  # noqa: E402,F401
from paper_plot_palette import COLORS as C, tint

HERE = Path(__file__).resolve().parent
RD = Path("/gpfs/scrubbed/zixianma/openwebrl-runtime/selection-head-20261008/label-kfold-20261101/pre-post-k3/student-replay")
INK, MUTED = C["INK"], C["MUTED"]
PRE, K3 = C["TRAIN"], C["LOOKAHEAD_TEACHER"]
R = json.loads((HERE / "student-replay.json").read_text())
A, ST = R["after_scrolls"], R["behaviour"]
EXAMPLES = [
    dict(letter="a", i=614, crop=(0.0, 0.03, 0.62, 0.85), site="huggingface.co, a quantized Qwen3-8B model card",
         title="Leaving a scroll loop",
         history="last 4 steps: scroll down, scroll up, scroll down, scroll up",
         groups=[("scroll down", 4, "pre"), ("go back", 1, "k3")], boxes=[],
         rate=f"Over {A['states']} replayed states after 2+ scrolls with “go back” offered: go back "
              f"{A['luna-pre']['back']:.0f}% / {A['gpt55-pre']['back']:.0f}% (pre: Luna / GPT-5.5) vs "
              f"{A['luna-k3']['back']:.0f}% / {A['gpt55-k3']['back']:.0f}% (H = 4)"),
    dict(letter="b", i=4528, crop=(0.04, 0.0, 0.84, 0.70), site="nerdwallet.com 401(k) calculator",
         title="Answering before the inputs match the task",
         history="the page still shows defaults (starting balance \\$100,000; contributions 16% of income)",
         groups=[("set starting balance to \\$0", 2, "pre"), ("answer “\\$2,496,238”", 3, "k3")],
         boxes=[((0.06, 0.20, 0.28, 0.245), PRE), ((0.47, 0.075, 0.67, 0.125), K3)],
         rate=f"When only 1–2 of the 5 proposals stop, the student stops: "
              f"Luna {ST['luna-pre']['stop_given_minority_offer']:.0f}% to {ST['luna-k3']['stop_given_minority_offer']:.0f}%, "
              f"GPT-5.5 {ST['gpt55-pre']['stop_given_minority_offer']:.0f}% to {ST['gpt55-k3']['stop_given_minority_offer']:.0f}% "
              f"(pre to H = 4)"),
]


def shot(i):
    st = [json.loads(l) for l in open(RD / "states.jsonl")][i]
    att = Path(st["trace"]).parent.parent
    sample = json.loads(Path(glob.glob(str(att / "samples" / "turn" / "*" / "*.json"))[0]).read_text())
    task = json.loads((att / "result.json").read_text())["intent"]
    png = base64.b64decode(sample["images"][st["turn"]][0].split(",", 1)[1])
    return Image.open(io.BytesIO(png)).convert("RGB"), task, st


def badge(fig, x, y, text, color, fs=5.9):
    t = fig.text(x, y, text, fontsize=fs, color="white", fontweight="bold", va="center",
                 bbox=dict(boxstyle="round,pad=0.25,rounding_size=0.4", fc=C["ADM_4B"], ec=color, lw=1.7))
    fig.canvas.draw()
    return t.get_window_extent().transformed(fig.transFigure.inverted()).x1 + 0.008


def main():
    plt.rcParams.update({"font.family": "Albert Sans", "savefig.facecolor": "white", "pdf.fonttype": 42})
    fig = plt.figure(figsize=(7.0, 3.25))
    xs, ws, xr = 0.0, 0.30, 0.325
    y, h = 0.93, 0.36
    for ex in EXAMPLES:
        im, task, st = shot(ex["i"]); W, H = im.size; l, t, r, b = ex["crop"]
        fig.text(xs, y + 0.012, ex["letter"], fontsize=8.6, fontweight="bold", color=INK, va="bottom")
        fig.text(xs + 0.022, y + 0.012, ex["title"], fontsize=7.0, fontweight="bold", color=INK, va="bottom")
        ax = fig.add_axes([xs, y - h, ws, h])
        ax.imshow(im.crop((int(l * W), int(t * H), int(r * W), int(b * H))), aspect="auto"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(C["STATE_BORDER"]); sp.set_linewidth(0.6)
        cw, ch = (r - l) * W, (b - t) * H
        for (x0, y0, x1, y1), c in ex["boxes"]:
            ax.add_patch(Rectangle(((x0 - l) * W, (y0 - t) * H), (x1 - x0) * W, (y1 - y0) * H, fill=False, ec=c, lw=1.6))
        bench = {"online_mind2web": "Online-Mind2Web", "webvoyager": "WebVoyager", "deepshop": "DeepShop"}[st["bench"]]
        short = task if len(task) <= 118 else task[:115].rsplit(" ", 1)[0] + " …"
        fig.text(xr, y - 0.015, f"Task: “{short}”", fontsize=6.1, color=INK, va="top", wrap=True)
        fig.text(xr, y - 0.075, f"{bench}, step {st['turn'] + 1}; {ex['site']}; {ex['history']}", fontsize=5.9,
                 color=MUTED, va="top")
        fig.text(xr, y - 0.13, "Five actor proposals, grouped", fontsize=6.0, color=MUTED, va="top", style="italic")
        for k, (label, n, who) in enumerate(ex["groups"]):
            yy = y - 0.185 - k * 0.07; c = PRE if who == "pre" else K3
            fig.add_artist(FancyBboxPatch((xr, yy - 0.024), 0.235, 0.048, boxstyle="round,pad=0,rounding_size=0.008",
                                          fc=tint("ACTION", 0.72), ec=C["ACTION_INK"], lw=1.0, transform=fig.transFigure))
            fig.text(xr + 0.008, yy, label, fontsize=6.4, color=INK, va="center", fontweight="bold")
            fig.text(xr + 0.227, yy, f"{n} of 5", fontsize=6.0, color=MUTED, va="center", ha="right")
            bx = xr + 0.25
            for j in ("GPT-5.5", "Luna"):
                bx = badge(fig, bx, yy, f"{'Pre-exec' if who == 'pre' else 'H = 4'} ({j})", c)
        fig.text(xr, y - h + 0.005, ex["rate"], fontsize=5.8, color=MUTED, va="bottom", style="italic")
        y -= h + 0.105
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"student-examples.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.02)


if __name__ == "__main__":
    main()
