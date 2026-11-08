#!/usr/bin/env python3
"""Review sheets: eval states where the pre and K = 3 students split and both picks were played out live (student_outcomes.py).

Each sheet: the task, earlier actions, the current screen with the pre pick (grey box) and the K = 3 pick (purple box), and
how many matched episodes succeeded after each. Then, per side, one real matched episode (a success and a failure when
both exist): the next three screens with the action taken on each (orange box), the final answer and the judged outcome.
W = the K = 3 pick did better, L = worse.

  student_outcome_examples.py  (CPU only; writes outcome-review-*.jpg next to this file)
"""
import json, glob, base64, io, sys, textwrap
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
RD = '/gpfs/scrubbed/zixianma/openwebrl-runtime/selection-head-20261008/label-kfold-20261101/pre-post-k3/student-replay/'
HERE = Path(__file__).resolve().parent
EX = {"W1": 2586, "W2": 7262, "W3": 5324, "L1": 2809, "L2": 677, "L3": 6795, "alt-W": 3610, "alt-L": 3000}
F = lambda s: ImageFont.truetype("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf", s) if Path("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf").exists() else ImageFont.load_default()
st = [json.loads(l) for l in open(RD + 'states.jsonl')]
pk = {}
for r in map(json.loads, open(RD + 'picks.jsonl')):
    if r.get('student'): pk.setdefault(r['i'], {})[r['student']] = r['pick']
rows = {r['i']: r for r in map(json.loads, open(RD + 'split-outcomes.jsonl'))}

def shots(trace):
    att = Path(trace).parent.parent
    return json.loads(Path(glob.glob(str(att / 'samples' / 'turn' / '*' / '*.json'))[0]).read_text())['images']

def png(images, t):
    if t >= len(images) or not images[t]: return None
    return Image.open(io.BytesIO(base64.b64decode(images[t][0].split(',', 1)[1]))).convert('RGB')

def act(a):
    a = a.replace('<tool_call>', '').replace('</tool_call>', '').replace('<|im_end|>', '').split('\n')
    return ' ; '.join(x.strip() for x in a if x.strip())

def mark(im, a, col):
    d = ImageDraw.Draw(im); W, H = im.size
    for line in act(a).split(' ; '):
        try: c = json.loads(line)
        except Exception: continue
        p = (c.get('arguments') or {}).get('point_2d')
        if p: X, Y = p[0] * W / 1000, p[1] * H / 1000; d.rectangle([X - 40, Y - 18, X + 40, Y + 18], outline=col, width=6)
    return im

TW, TH = 760, 470
for name, i in EX.items():
    s = st[i]; r = rows[i]; j = r['judge']; turn = s['turn']
    steps = {x['turn']: x for x in map(json.loads, open(s['trace']))}; x = steps[turn]
    task = json.loads((Path(s['trace']).parent.parent / 'result.json').read_text())['intent']
    cur = png(shots(s['trace']), turn)
    A = {side: x['candidates'][pk[i][f'{j}-{side}']]['action'] for side in ('pre', 'k3')}
    cur = mark(mark(cur, A['pre'], (60, 60, 60)), A['k3'], (98, 80, 190))
    lines = []
    for side in ('pre', 'k3'):
        eps = r[side]['episodes']
        pick = [e for e in eps if e['reward'] == 1][:1] + [e for e in eps if e['reward'] != 1][:1]
        for e in pick: lines.append((side, e))
    canvas = Image.new("RGB", (TW * 3 + 40, 260 + 2 * TH + len(lines) * (TH + 130)), "white"); d = ImageDraw.Draw(canvas)
    hist = [act(steps[t]['candidates'][steps[t]['selected_index']]['action'])[:70] for t in range(turn)][-4:]
    d.text((10, 8), f"{name}  (state {i}, {r['task']}, step {turn + 1}, students: {'GPT-5.5' if j == 'gpt55' else 'Luna'})", fill='black', font=F(26))
    for k, t in enumerate(textwrap.wrap('Task: ' + task, 150)[:2]): d.text((10, 45 + 30 * k), t, fill='black', font=F(22))
    d.text((10, 110), 'Previous actions: ' + ' | '.join(hist), fill=(90, 90, 90), font=F(18))
    d.text((10, 140), f"PRE pick (grey box):  {act(A['pre'])[:150]}   -> {r['pre']['success']}/{r['pre']['n']} episodes succeed", fill=(60, 60, 60), font=F(20))
    d.text((10, 170), f"K3 pick (purple box): {act(A['k3'])[:150]}   -> {r['k3']['success']}/{r['k3']['n']} episodes succeed", fill=(98, 80, 190), font=F(20))
    c = cur.copy(); c.thumbnail((TW * 2, TH * 2)); canvas.paste(c, (10, 210)); y = 220 + c.size[1]
    for side, e in lines:
        im = shots(e['trace']); es = {x['turn']: x for x in map(json.loads, open(e['trace']))}
        col = (60, 60, 60) if side == 'pre' else (98, 80, 190)
        d.text((10, y + 5), f"{side.upper()} branch, one matched episode ({e['root']} {e['run']} {e['slug']}): {'SUCCESS' if e['reward'] == 1 else 'FAILURE'}, {len(es)} steps total", fill=(27, 127, 59) if e['reward'] == 1 else (179, 38, 30), font=F(22))
        for k in range(3):
            t = turn + 1 + k; p = png(im, t)
            if p is None: continue
            if t in es: p = mark(p, es[t]['candidates'][es[t]['selected_index']]['action'], (230, 120, 0))
            p.thumbnail((TW - 10, TH)); canvas.paste(p, (10 + k * TW, y + 40))
            cap = act(es[t]['candidates'][es[t]['selected_index']]['action'])[:95] if t in es else '(end)'
            for q, ln in enumerate(textwrap.wrap(f"after +{k + 1}: next action {cap}", 70)[:2]): d.text((10 + k * TW, y + 45 + TH + 22 * q), ln, fill='black', font=F(17))
        last = es[max(es)]; fin = act(last['candidates'][last['selected_index']]['action'])
        d.text((10, y + TH + 92), 'final action: ' + fin[:200], fill=(90, 90, 90), font=F(16))
        y += TH + 120
    out = canvas.crop((0, 0, canvas.size[0], y + 10)); out.thumbnail((1600, 10000))
    out.save(HERE / f'outcome-review-{name}-{i}.jpg', quality=82, optimize=True)
    print(name, i, 'done', flush=True)
