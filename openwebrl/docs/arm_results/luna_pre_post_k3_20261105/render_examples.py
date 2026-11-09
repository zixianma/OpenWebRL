#!/usr/bin/env python3
"""Render qualitative pre-exec / lookahead H = 4 examples from the saved branch states (real screenshots).

For each chosen state: the screen the teachers judge (with each candidate's click / scroll point), a list of all candidates
with each teacher's pick and held-out success (draws 4-8, valid only), and, for the two contested candidates, what
the lookahead H = 4 teacher saw: the immediate result of rollout 1 and the actor's next
three steps after it. Teachers' picks are firm: both calls of each teacher agree (select_examples.py).

  render_examples.py STATE:GOOD:BAD:SLUG [...]     (CPU only; writes example-<SLUG>.jpg next to this file)
"""
import base64
import io
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image
# Shared paper typography, including bundled Albert Sans fonts.
import sys as _paper_sys
from pathlib import Path as _PaperPath
_paper_sys.path.insert(0, str(next(p for p in _PaperPath(__file__).resolve().parents
                                if (p / "paper_plot_style.py").is_file())))
import paper_plot_style as _paper_style  # noqa: E402,F401
from paper_plot_palette import COLORS as C, tint

R = Path("/gpfs/scrubbed/zixianma/openwebrl-runtime/selection-head-20261008/sft-actions-20261021")
HERE = Path(__file__).resolve().parent
SRC = [("pre", "Pre", C["PRE_EXEC"]), ("k3", "H = 4", C["LOOKAHEAD_TEACHER"])]
CAND = [tint("ACTION", f) for f in (0, 0.12, 0.24, 0.36, 0.48)]
INK, MUTED, GOOD, BAD = C["INK"], C["MUTED"], C["SUCCESS"], C["FAILURE"]
CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
DATA_URL = re.compile(r"data:image/[a-z]+;base64,[A-Za-z0-9+/=]+")


def read(p):
    return json.loads(Path(p).read_text())


def calls(text):
    out = []
    for m in CALL.findall(text or ""):
        try:
            out.append(json.loads(m))
        except json.JSONDecodeError:
            pass
    return out


def describe(c, key=None):
    """Short human description of one tool call (uses the a11y element name from the action key when given)."""
    n, a = c.get("name", "?"), c.get("arguments", {})
    if n == "click":
        return f"click “{key}”" if isinstance(key, str) else "click"
    if n == "scroll":
        return f"scroll {a.get('direction', '')}"
    if n == "write":
        return f"type “{a.get('message') or a.get('text') or a.get('content') or ''}”"
    if n == "press_keys":
        return "press " + "+".join(a.get("keys", []))
    if n == "goto_url":
        return f"go to {a.get('url', '')}"
    if n == "done":
        return "stop / answer"
    return n.replace("_", " ")


def short(s, n):
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 1] + "…"


def action_label(action, cand_text):
    key = json.loads(action["key"])
    cs = calls(cand_text)
    parts = []
    for i, c in enumerate(cs):
        k = key[i] if i < len(key) else None
        parts.append(describe(c, k[2] if k and len(k) > 2 and k[0] == "click" and k[1] == "element" and isinstance(k[2], str) else None))
    return ", then ".join(parts) or short(action["key"], 40), cs


def img(path_or_url):
    if str(path_or_url).startswith("data:"):
        return Image.open(io.BytesIO(base64.b64decode(str(path_or_url).split(",", 1)[1]))).convert("RGB")
    return Image.open(path_or_url).convert("RGB")


def attempt_for(root, sid, cand, repeat=0):
    for d in read(Path(root) / "states" / sid / "release.json")["attempt_directories"]:
        f = Path(d) / "immediate.json"
        if f.exists():
            x = read(f)
            if int(x["candidate"]) == cand and int(x["repeat"]) == repeat:
                return Path(d), x
    raise KeyError((sid, cand))


def lookahead(attempt, depth, k=3):
    files = list((attempt / "rollouts").rglob("*.json"))
    r = read(files[0])
    segs = r["llm_input_texts"].split("<|im_start|>assistant\n")[1:]
    turns = segs[:-1] + [segs[-1] + r["llm_response"]] if segs else [r["llm_response"]]
    images = [m.group(0) if (m := DATA_URL.search(str(x))) else None for x in r["images"]]
    steps = []
    for j in range(1, k + 1):
        t = depth + j
        if t >= len(turns):
            break
        cs = calls(turns[t].split("<|im_end|>", 1)[0])
        steps.append(dict(label=", then ".join(describe(c) for c in cs) or "(no call)",
                          shot=images[t + 1] if t + 1 < len(images) else None))
    return steps


def held(action):
    cells = [c for d in range(3, 8) for c in action["outcomes"][str(d)] if c and c[0]]
    return sum(c[1] for c in cells), len(cells)


def picks(x):
    where = {a["teacher_exec"]: i for i, a in enumerate(x["actions"])}
    return {src: [where[j["candidate"]] for j in read(R / "labels" / src / "judgments" / f"{x['state_id']}.json") if j.get("valid")]
            for src, *_ in SRC}


def pill(ax, x, y, text, color, fs=6.0):
    t = ax.text(x, y, text, fontsize=fs, color="white" if color == C["PRE_EXEC"] else INK, ha="left", va="center", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25,rounding_size=0.5", fc=color, ec="none"))
    return t


def render(sid, good, bad, slug, recs):
    x = recs[sid]; root = Path(x["root"])
    anc = read(read(root / "states" / sid / "anchor-reference.json")["path"])
    depth = anc["depth"]; p = picks(x)
    before = img(x["image"]); W, H = before.size
    labels = []
    for i, a in enumerate(x["actions"]):
        lab, cs = action_label(a, anc["candidates"][a["teacher_exec"]]["output"][0]); labels.append((lab, cs))
    rows = []
    for which, i in (("better", good), ("worse", bad)):
        att, imm = attempt_for(root, sid, x["actions"][i]["teacher_exec"])
        rows.append((which, i, img(imm["observation"]["screenshot"]["path"]), lookahead(att, depth)))

    fig = plt.figure(figsize=(7.2, 5.15))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.55, 1, 1], hspace=0.38, wspace=0.06, left=0.105, right=0.995, top=0.9, bottom=0.03)
    fig.text(0.005, 0.975, f"Task: {short(x['task'], 120)}", fontsize=7.4, color=INK, va="top", fontweight="bold")
    fig.text(0.005, 0.945, f"{short(anc['before_observation'].get('active_tab_url', x['url']), 80)}  ·  step {depth + 1}",
             fontsize=6.4, color=MUTED, va="top")
    # (1) the screen being judged, with candidate points
    ax = fig.add_subplot(gs[0, :2]); ax.imshow(before); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(C["STATE_BORDER"]); sp.set_linewidth(0.5)
    ax.set_title("Screen the teachers judge (pre-exec sees only this)", fontsize=6.6, color=INK, pad=3, loc="left")
    for i, (_, cs) in enumerate(labels):
        for c in cs[:1]:
            pt = (c.get("arguments") or {}).get("point_2d")
            if pt and len(pt) == 2:
                px, py = pt[0] / 1000 * W, pt[1] / 1000 * H
                dx = (i - (len(labels) - 1) / 2) * 0.012 * W if c.get("name") == "scroll" else 0
                ax.scatter([px + dx], [py], s=70, color=CAND[i], edgecolor="white", linewidth=0.9, zorder=3)
                ax.text(px + dx, py, str(i + 1), fontsize=5.6, color=INK, ha="center", va="center", fontweight="bold", zorder=4)
                if c.get("name") == "scroll":
                    d = -1 if (c.get("arguments") or {}).get("direction") == "up" else 1
                    ax.annotate("", (px + dx, py + d * 0.09 * H), (px + dx, py + d * 0.025 * H),
                                arrowprops=dict(arrowstyle="-|>", color=CAND[i], lw=1.2), zorder=3)
    # (2) candidate list
    ax = fig.add_subplot(gs[0, 2:]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0, 1.0, "Candidates", fontsize=6.6, color=INK, va="top")
    ax.text(0.98, 1.0, "held-out success", fontsize=6.0, color=MUTED, va="top", ha="right")
    n = len(labels); step = min(0.19, 0.88 / n)
    for i, (lab, _) in enumerate(labels):
        y = 0.86 - i * step
        ax.scatter([0.025], [y], s=70, color=CAND[i], edgecolor="white", linewidth=0.9)
        ax.text(0.025, y, str(i + 1), fontsize=5.6, color=INK, ha="center", va="center", fontweight="bold")
        ax.text(0.065, y + 0.035, short(lab, 52), fontsize=6.2, color=INK, va="center")
        s, m = held(x["actions"][i]); col = GOOD if i == good else BAD if i == bad else MUTED
        ax.text(0.98, y + 0.035, f"{s}/{m}", fontsize=6.6, color=col, va="center", ha="right", fontweight="bold")
        xx = 0.065
        for src, name, color in SRC:
            if p[src].count(i):
                t = pill(ax, xx, y - 0.035, name + (" ×2" if p[src].count(i) == 2 else ""), color, fs=5.4)
                fig.canvas.draw(); bb = t.get_window_extent().transformed(ax.transData.inverted()); xx = bb.x1 + 0.02
    # (3) what lookahead H = 4 additionally sees, for the two contested candidates
    for r, (which, i, imm, steps) in enumerate(rows, start=1):
        shots = [("After candidate action", imm)] + [(f"+{j + 1}: {short(s['label'], 34)}", img(s["shot"]) if s["shot"] else None)
                                                                  for j, s in enumerate(steps)]
        for c in range(4):
            ax = fig.add_subplot(gs[r, c]); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color(C["FUTURE_STATE_BORDER"]); sp.set_linewidth(1.2 if c == 0 else 0.5)
            if c < len(shots) and shots[c][1] is not None:
                ax.imshow(shots[c][1]); ax.set_title(shots[c][0], fontsize=5.6, color=INK if c == 0 else MUTED, pad=2, loc="left")
            elif c < len(shots):
                ax.set_title(shots[c][0], fontsize=5.6, color=MUTED, pad=2, loc="left")
                ax.text(0.5, 0.5, "episode ends here\n(no screenshot)", fontsize=6, color=MUTED, ha="center", va="center", transform=ax.transAxes)
            else:
                ax.axis("off"); ax.text(0.5, 0.5, "no later step recorded", fontsize=6, color=MUTED, ha="center", va="center", transform=ax.transAxes)
            if c == 0:
                s, m = held(x["actions"][i])
                ax.text(-0.06, 0.5, f"Cand. {i + 1}\n{s}/{m} succeed", transform=ax.transAxes, fontsize=6.2, ha="right", va="center",
                        color=GOOD if which == "better" else BAD, fontweight="bold")
        if r == 1:
            fig.text(0.6, gs[1, 0].get_position(fig).y1 + 0.035, "Lookahead H = 4 includes the next three actor steps", fontsize=6.0, color=MUTED)
    buf = io.BytesIO(); fig.savefig(buf, dpi=220, format="png", facecolor="white"); plt.close(fig)
    Image.open(buf).convert("RGB").save(HERE / f"example-{slug}.jpg", quality=84, optimize=True)
    return dict(state=sid, slug=slug, task=x["task"].strip(), depth=depth, labels=[l for l, _ in labels],
                held=[held(a) for a in x["actions"]], picks=p, good=good, bad=bad,
                steps={i: [s["label"] for s in st] for _, i, _, st in rows})


def main():
    plt.rcParams.update({"font.family": "Albert Sans", "savefig.facecolor": "white"})
    want = [a.split(":") for a in sys.argv[1:]]
    ids = {w[0] for w in want}; recs = {}
    for line in open(R / "actions.jsonl"):
        if any(i in line[:80] for i in ids):
            x = json.loads(line)
            if x["state_id"] in ids:
                recs[x["state_id"]] = x
    meta = [render(sid, int(g), int(b), slug, recs) for sid, g, b, slug in want]
    print(json.dumps(meta, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
