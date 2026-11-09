#!/usr/bin/env python3
"""Review sheets: eval states where the pre and H = 4 students split and both picks were played out live (student_outcomes.py).

Each sheet: the task, the current screen with the pre pick (yellow source outline) and the H = 4 pick (salmon source outline), and how many matched
episodes succeeded after each. Then, per side, one real matched episode whose outcome matches that side's majority: the
next three screens with the action taken on each (purple box) and the judged outcome. W = the H = 4 pick did better,
L = worse. Pick labels are written by hand; later actions are summarised from their tool calls.

  student_outcome_examples.py  (CPU only; writes outcome-review-*.jpg next to this file)
"""
import base64
import glob
import io
import json
from pathlib import Path
import textwrap
from urllib.parse import urlparse

from PIL import Image, ImageDraw, ImageFont
import sys
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "paper_plot_palette.py").is_file())))
from paper_plot_palette import COLORS as C, tint

RD = Path("/gpfs/scrubbed/zixianma/openwebrl-runtime/selection-head-20261008/label-kfold-20261101/pre-post-k3/student-replay")
HERE = Path(__file__).resolve().parent
EX = [  # name, state, site, pre pick, H = 4 pick
    ("W1", 2586, "Hugging Face", 'click "Documentation" on the 429 page', "go back"),
    ("W2", 7262, "arXiv", "type into the search box", 'open the "gr-qc" listing'),
    ("W3", 5324, "Allrecipes", "go to google.com", "new tab: bbcgoodfood.com"),
    ("L1", 2809, "Hugging Face", 'click "Files and versions"', "go back"),
    ("L2", 677, "Apple", "scroll down", "go back"),
    ("L3", 6795, "Apple", "click the search icon", 'click "Entertainment"'),
    ("alt-W", 3610, "Hugging Face", "search on the 429 page", "go back"),
    ("alt-L", 3000, "arXiv", 'click "General information"', "scroll down"),
]
FONT_DIR = next(p for p in Path(__file__).resolve().parents if (p / "paper_plot_style.py").is_file()) / "assets/fonts/albert-sans"
FONT = str(FONT_DIR / "AlbertSans{}.ttf")
F = lambda n, b=False: ImageFont.truetype(FONT.format("-Bold" if b else "-Regular"), n)  # noqa: E731
INK, MUTED, PRE, K3, NEXT = C["INK"], C["MUTED"], C["TRAIN"], C["LOOKAHEAD_TEACHER"], C["ACTION_INK"]
GOOD, BAD = C["SUCCESS"], C["FAILURE"]
W = 1800


def images(trace):
    att = Path(trace).parent.parent
    return json.loads(Path(glob.glob(str(att / "samples" / "turn" / "*" / "*.json"))[0]).read_text())["images"]


def png(ims, t):
    if t >= len(ims) or not ims[t]:
        return None
    return Image.open(io.BytesIO(base64.b64decode(ims[t][0].split(",", 1)[1]))).convert("RGB")


def calls(a):
    out = []
    for x in a.replace("<|im_end|>", "").split("<tool_call>"):
        x = x.replace("</tool_call>", "").strip()
        try:
            out.append(json.loads(x))
        except ValueError:
            pass
    return out


def say(a):
    """Short readable label for an executed response."""
    parts = []
    for c in calls(a):
        n, g = c.get("name"), c.get("arguments") or {}
        if n == "click":
            parts.append("click")
        elif n == "write":
            parts.append(f'type "{g.get("message", "")[:28]}"')
        elif n == "press_keys":
            parts.append("+".join(g.get("keys", [])))
        elif n == "scroll":
            parts.append(f'scroll {g.get("direction", "")}')
        elif n == "go_back":
            parts.append("go back")
        elif n == "goto_url":
            parts.append("open " + urlparse(g.get("url", "")).netloc.replace("www.", ""))
        elif n == "done":
            parts.append('answer: "' + textwrap.shorten(g.get("response", ""), 60, placeholder=" …") + '"')
        elif n in ("new_tab", "switch_tab", "close_tab"):
            parts.append(n.replace("_", " "))
        else:
            parts.append(str(n))
    return ", ".join(dict.fromkeys(parts)) or "(no action)"


def mark(im, a, col, w=8):
    d = ImageDraw.Draw(im); X0, Y0 = im.size
    for c in calls(a):
        p = (c.get("arguments") or {}).get("point_2d")
        if c.get("name") == "click" and p:
            x, y = p[0] * X0 / 1000, p[1] * Y0 / 1000
            d.rectangle([x - 60, y - 26, x + 60, y + 26], outline=col, width=w)
    return im


def fit(im, w):
    return im.resize((w, int(im.size[1] * w / im.size[0])))


def main():
    st = [json.loads(l) for l in open(RD / "states.jsonl")]
    pk = {}
    for r in map(json.loads, open(RD / "picks.jsonl")):
        if r.get("student"):
            pk.setdefault(r["i"], {})[r["student"]] = r["pick"]
    rows = {r["i"]: r for r in map(json.loads, open(RD / "split-outcomes.jsonl"))}
    for name, i, site, pre_label, k3_label in EX:
        s, r = st[i], rows[i]; j, turn = r["judge"], s["turn"]
        steps = {x["turn"]: x for x in map(json.loads, open(s["trace"]))}
        task = json.loads((Path(s["trace"]).parent.parent / "result.json").read_text())["intent"]
        act = {side: steps[turn]["candidates"][pk[i][f"{j}-{side}"]]["action"] for side in ("pre", "k3")}
        cur = mark(mark(png(images(s["trace"]), turn), act["pre"], PRE), act["k3"], K3)
        canvas = Image.new("RGB", (W, 4000), "white"); d = ImageDraw.Draw(canvas)
        who = "GPT-5.5" if j == "gpt55" else "Luna"
        d.text((24, 18), f"{name}  ·  {site}, step {turn + 1}  ·  {who} students", fill=INK, font=F(44, True))
        y = 84
        for ln in textwrap.wrap(f"Task: {task}", 72)[:3]:
            d.text((24, y), ln, fill=INK, font=F(34)); y += 46
        y += 14
        cw = 980; c = fit(cur, cw); canvas.paste(c, (24, y))
        d.rectangle([24, y, 24 + cw, y + c.size[1]], outline=C["STATE_BORDER"], width=2)
        x0 = 24 + cw + 40; yy = y + 10
        for side, label, col in (("pre", pre_label, PRE), ("k3", k3_label, K3)):
            k, n = r[side]["success"], r[side]["n"]
            d.text((x0, yy), "Pre-exec students pick" if side == "pre" else "Lookahead H = 4 students pick", fill=C["ADM_4B"], font=F(32, True))
            yy += 46
            for ln in textwrap.wrap(label, 30):
                d.text((x0, yy), ln, fill=INK, font=F(38, True)); yy += 50
            d.text((x0, yy), f"{k} of {n} episodes succeed", fill=GOOD if k / n > 0.5 else BAD, font=F(36, True))
            yy += 90
        y += max(c.size[1], yy - y) + 40
        for side, col in (("pre", PRE), ("k3", K3)):
            eps = r[side]["episodes"]; good = r[side]["success"] / r[side]["n"] > 0.5
            e = next((e for e in eps if (e["reward"] == 1) == good), eps[0])
            ims = images(e["trace"]); es = {x["turn"]: x for x in map(json.loads, open(e["trace"]))}
            ok = e["reward"] == 1
            d.line([24, y, W - 24, y], fill=C["BORDER"], width=2); y += 16
            d.text((24, y), f"After the {'pre-exec' if side == 'pre' else 'lookahead H = 4'} pick, one episode:", fill=C["ADM_4B"], font=F(34, True))
            d.text((780, y), "SUCCESS" if ok else "FAILURE", fill=GOOD if ok else BAD, font=F(34, True))
            y += 56
            tw, gap, hmax = (W - 48 - 2 * 24) // 3, 24, 0
            for k in range(3):
                t = turn + 1 + k; p = png(ims, t)
                if p is None:
                    continue
                a = es[t]["candidates"][es[t]["selected_index"]]["action"] if t in es else ""
                p = fit(mark(p, a, NEXT, 10), tw); x = 24 + k * (tw + gap)
                canvas.paste(p, (x, y)); d.rectangle([x, y, x + tw, y + p.size[1]], outline=C["FUTURE_STATE_BORDER"], width=2)
                cap = textwrap.wrap(f"then: {say(a)}" if a else "(episode ended)", 30)[:3]
                for q, ln in enumerate(cap):
                    d.text((x, y + p.size[1] + 10 + 40 * q), ln, fill=INK, font=F(30))
                hmax = max(hmax, p.size[1] + 20 + 40 * len(cap))
            y += hmax + 10
            last = es[max(es)]
            if max(es) > turn + 3:
                d.text((24, y), f"… {max(es) - turn - 3} more steps, then {say(last['candidates'][last['selected_index']]['action'])}",
                       fill=MUTED, font=F(28)); y += 50
            y += 20
        out = canvas.crop((0, 0, W, y)); out.thumbnail((1400, 10000))
        out.save(HERE / f"outcome-review-{name}-{i}.jpg", quality=85, optimize=True)
        print(name, i, flush=True)


if __name__ == "__main__":
    main()
