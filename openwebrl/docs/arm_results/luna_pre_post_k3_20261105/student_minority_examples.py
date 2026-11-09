#!/usr/bin/env python3
"""Review sheets: all five candidates on eval states where a student picked a minority action that did better live.

From student_candidate_outcomes.py (live eval episodes only): states with a unique most-proposed action, where at least one
student picked a less-proposed action that is not a near-copy of the majority action (same calls with clicks within 30/1000,
same scroll direction, same text), the majority action was played out in >= 2 valid episodes, and the picked action's
success rate beats the majority action's by >= 50 points. M sheets: the pick was played out in >= 2 episodes; S sheets: in
one episode only. Each sheet: the task; the current screen with every click candidate boxed and
lettered; every distinct candidate action with how many of the five proposals it got, which students picked it, and how
many matched episodes succeeded after it; then one real episode after the minority pick and one after the majority action.

  student_minority_examples.py  (CPU only; writes minority-review-*.jpg next to this file)
"""
import json
import os
import sys
from pathlib import Path
import re
import textwrap

from PIL import Image, ImageDraw
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "paper_plot_palette.py").is_file())))
from paper_plot_palette import COLORS as C, tint

# Identical frozen parser, kept local so qualitative plotting does not require torch.
TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)

def parse_calls(response):
    """Executable tool calls after the thinking block, or None if unparseable."""
    text = response.rsplit("</think>", 1)[-1].replace("<|im_end|>", "")
    blocks = TOOL_CALL.findall(text)
    if not blocks or len(blocks) != text.count("<tool_call>"):
        return None
    try:
        calls = [json.loads(b) for b in blocks]
    except json.JSONDecodeError:
        return None
    if any(not isinstance(c, dict) or not isinstance(c.get("name"), str)
           or not isinstance(c.get("arguments", {}), dict) for c in calls):
        return None
    return [{"name": c["name"], "arguments": c.get("arguments", {})} for c in calls]
from student_outcome_examples import BAD, F, GOOD, INK, K3, MUTED, PRE, RD, W, calls, fit, images, png, say

HERE = Path(__file__).resolve().parent
COLS = [tint("ACTION", f) for f in (0, 0.12, 0.24, 0.36, 0.48)]
NEXT = C["ACTION_INK"]
WHO = {"gpt55-pre": ("Pre GPT-5.5", PRE), "luna-pre": ("Pre Luna", PRE), "gpt55-k3": ("H = 4 GPT-5.5", K3),
       "luna-k3": ("H = 4 Luna", K3)}
SITE = lambda task: task.split("/", 1)[1].split("--")[0]  # noqa: E731


def label(c):
    """Readable label for one candidate: say(), with the click target named from the actor's thought when it is quoted."""
    s = say(c["action"])
    m = re.findall(r'click(?:ing)? (?:on )?(?:the )?["“]([^"”]{1,40})["”]', c.get("thought") or "", re.I)
    if not m:
        m = re.findall(r"click(?:ing)? (?:on )?(?:the )?((?:[\w'-]+ ){0,3}(?:button|icon|link|tab|box|field|option|dropdown|menu))",
                       c.get("thought") or "", re.I)
        m = [x for x in m if not x.lower().startswith(("on ", "it ", "that ", "this "))]
        if m:
            m = [f"the {m[-1]}"]
            if s.startswith("click"):
                return "click " + m[-1] + s[5:]
    if m and s.startswith("click"):
        s = f'click "{m[-1]}"' + s[5:]
    return s


def box(im, a, col, tag, w=8, taken=None):
    d = ImageDraw.Draw(im); X0, Y0 = im.size; taken = [] if taken is None else taken
    for c in calls(a):
        p = (c.get("arguments") or {}).get("point_2d")
        if c.get("name") == "click" and p:
            x, y = p[0] * X0 / 1000, p[1] * Y0 / 1000
            d.rectangle([x - 50, y - 22, x + 50, y + 22], outline=col, width=w)
            if tag:
                ty = y - 22
                while any(abs(ty - v) < 38 and abs(x - u) < 40 for u, v in taken):
                    ty += 38
                taken.append((x, ty))
                d.rectangle([x + 50, ty, x + 84, ty + 36], fill=col)
                d.text((x + 58, ty), tag, fill=INK, font=F(30, True))
    return im


def near(a, b):
    """Near-copy: same tool calls, clicks within 30/1000, same scroll direction, same text / URL / answer."""
    if a is None or b is None or [c["name"] for c in a] != [c["name"] for c in b]:
        return False
    for x, y in zip(a, b):
        ga, gb = x.get("arguments") or {}, y.get("arguments") or {}
        if x["name"] == "click" and max(abs(p - q) for p, q in zip(ga.get("point_2d", [0, 0]), gb.get("point_2d", [0, 0]))) > 30:
            return False
        if (x["name"] == "scroll" and ga.get("direction") != gb.get("direction")) or (x["name"] in ("write", "goto_url", "done") and ga != gb):
            return False
    return True


def chosen():
    out = []
    for r in map(json.loads, open(RD / "candidate-outcomes.jsonl")):
        gs = r["groups"]; sz = sorted((g["size"] for g in gs), reverse=True)
        if sz[0] == sz[1]:
            continue
        maj = max(gs, key=lambda g: g["size"])
        if maj["n"] < 2:
            continue
        c = {x["turn"]: x for x in map(json.loads, open(r["trace"]))}[r["turn"]]["candidates"]
        rep = lambda g: parse_calls(c[g["members"][0]]["action"] or "")  # noqa: E731
        best = [g for g in gs if g is not maj and g["picked_by"] and g["n"] and not near(rep(g), rep(maj))
                and g["success"] / g["n"] - maj["success"] / maj["n"] >= 0.5]
        if best:
            out.append((r, max(best, key=lambda g: (g["success"] / g["n"], g["n"])), maj))
    return sorted(out, key=lambda x: (x[1]["n"] < 2, x[0]["i"]))


def episode(d, canvas, y, g, title, col, turn, pick):
    good = g["success"] / g["n"] >= 0.5 if pick else g["success"] / g["n"] > 0.5
    e = next((e for e in g["episodes"] if (e["reward"] == 1) == good), g["episodes"][0])
    ims = images(e["trace"]); es = {x["turn"]: x for x in map(json.loads, open(e["trace"]))}; ok = e["reward"] == 1
    d.line([24, y, W - 24, y], fill=C["BORDER"], width=2); y += 16
    d.text((24, y), title, fill=col, font=F(34, True))
    d.text((W - 260, y), "SUCCESS" if ok else "FAILURE", fill=GOOD if ok else BAD, font=F(34, True)); y += 56
    tw, gap, hmax = (W - 48 - 2 * 24) // 3, 24, 0
    for k in range(3):
        t = turn + 1 + k; p = png(ims, t)
        if p is None:
            continue
        a = es[t]["candidates"][es[t]["selected_index"]]["action"] if t in es else ""
        p = fit(box(p, a, NEXT, None, 10), tw); x = 24 + k * (tw + gap)
        canvas.paste(p, (x, y)); d.rectangle([x, y, x + tw, y + p.size[1]], outline=C["STATE_BORDER"], width=2)
        cap = textwrap.wrap(f"then: {say(a)}" if a else "(episode ended)", 30)[:3]
        for q, ln in enumerate(cap):
            d.text((x, y + p.size[1] + 10 + 40 * q), ln, fill=INK, font=F(30))
        hmax = max(hmax, p.size[1] + 20 + 40 * len(cap))
    y += hmax + 10
    last = es[max(es)]
    if max(es) > turn + 3:
        d.text((24, y), f"… {max(es) - turn - 3} more steps, then {say(last['candidates'][last['selected_index']]['action'])}",
               fill=MUTED, font=F(28)); y += 50
    return y + 20


def main():
    st = [json.loads(l) for l in open(RD / "states.jsonl")]
    cnt = {"M": 0, "S": 0}
    for r, pick, maj in chosen():
        i, turn = r["i"], r["turn"]; ser = "M" if pick["n"] >= 2 else "S"; cnt[ser] += 1; name = f"{ser}{cnt[ser]}"
        steps = {x["turn"]: x for x in map(json.loads, open(r["trace"]))}; cands = steps[turn]["candidates"]
        task = json.loads((Path(r["trace"]).parent.parent / "result.json").read_text())["intent"]
        gs = sorted(r["groups"], key=lambda g: -g["size"]); tag = {id(g): "ABCDE"[k] for k, g in enumerate(gs)}
        cur = png(images(st[i]["trace"]), turn); taken = []
        for k, g in enumerate(gs):
            box(cur, cands[g["members"][0]]["action"], COLS[k], tag[id(g)], taken=taken)
        canvas = Image.new("RGB", (W, 5000), "white"); d = ImageDraw.Draw(canvas)
        d.text((24, 18), f"{name}  ·  {SITE(r['task'])}, step {turn + 1}  ·  state {i}", fill=INK, font=F(44, True))
        y = 84
        for ln in textwrap.wrap(f"Task: {task}", 72)[:3]:
            d.text((24, y), ln, fill=INK, font=F(34)); y += 46
        y += 14; cw = 980; c = fit(cur, cw); canvas.paste(c, (24, y))
        d.rectangle([24, y, 24 + cw, y + c.size[1]], outline=C["STATE_BORDER"], width=2)
        x0, yy = 24 + cw + 30, y
        d.text((x0, yy), "All five candidates, grouped", fill=MUTED, font=F(30, True)); yy += 50
        for k, g in enumerate(gs):
            col = COLS[k]
            d.rectangle([x0, yy, x0 + 40, yy + 40], fill=col); d.text((x0 + 9, yy + 1), tag[id(g)], fill=INK, font=F(32, True))
            lines = textwrap.wrap(label(cands[g["members"][0]]), 26)[:2]
            for q, ln in enumerate(lines):
                d.text((x0 + 54, yy + 44 * q), ln, fill=INK, font=F(34, True))
            yy += 44 * len(lines) + 2
            d.text((x0 + 54, yy), f"{g['size']} of 5 proposals" + ("  (majority)" if g is maj else ""), fill=MUTED, font=F(28)); yy += 38
            if g["picked_by"]:
                bx = x0 + 54
                for s in g["picked_by"]:
                    t, cc = WHO[s]; wdt = d.textlength(t, font=F(24, True)) + 16
                    if bx + wdt > W - 10:
                        bx = x0 + 54; yy += 38
                    d.rounded_rectangle([bx, yy, bx + wdt, yy + 32], 8, fill=C["ADM_4B"], outline=cc, width=3); d.text((bx + 8, yy + 2), t, fill="white", font=F(24, True))
                    bx += wdt + 8
                yy += 40
            if g["n"]:
                ok = g["success"] / g["n"] > 0.5
                d.text((x0 + 54, yy), f"{g['success']} of {g['n']} episodes succeed", fill=GOOD if ok else BAD, font=F(30, True))
            else:
                d.text((x0 + 54, yy), "never played out", fill=MUTED, font=F(28))
            yy += 58
        y += max(c.size[1], yy - y) + 30
        pt, mt = tag[id(pick)], tag[id(maj)]
        y = episode(d, canvas, y, pick, f"After {pt} (minority, picked by {len(pick['picked_by'])} student{'s' * (len(pick['picked_by']) > 1)}), one episode:",
                    COLS[gs.index(pick)], turn, True)
        y = episode(d, canvas, y, maj, f"After {mt} (majority action), one episode:", COLS[gs.index(maj)], turn, False)
        out = canvas.crop((0, 0, W, y)); out.thumbnail((1400, 10000))
        out.save(Path(os.environ.get("OUT_DIR", HERE)) / f"minority-review-{name}-{i}.jpg", quality=85, optimize=True)
        print(name, i, r["task"], flush=True)


if __name__ == "__main__":
    main()
