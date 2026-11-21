"""Matched GPT-5.5 K=3 pilot analysis (2026-11-19). All picks are compared on the executed target (distinct action) they
choose, so duplicate proposals of one action count as the same pick. Labels compared on the pilot sample:
  new  = GPT-5.5 K=3 in the pre format (this pilot, repeat 0; repeat 1 on the first 200 for test-retest)
  old  = existing GPT-5.5 K=3 (effort high, index-form schema prompt, no rationale)  labels/<b>/k3/judgments
  pre  = GPT-5.5 pre-execution (pre format, rationale)                                labels/<b>/pre
  lk3p, lpre = Luna K=3 (pre format) and Luna pre                                      labels-luna/<b>/{k3p,pre}
Agreement = share of panels where two labels pick the same target; chance = expected agreement of two independent uniform
picks over the panel's distinct targets. 95% intervals: percentile bootstrap over states (10,000, seed 20261119)."""
import json, random, re, sys, collections
from pathlib import Path
import numpy as np
H = Path(__file__).resolve().parent; P16 = H.parent
rows = [json.loads(l) for l in open(H/'sample.jsonl')]


def tool(text):
    m = re.search(r'"name":\s*"([^"]+)"', text or ''); return m.group(1) if m else '?'


def get(r):
    b, stem = r['batch'], f"{r['state']}-p{r['panel']:02d}"
    out = {}
    for rep in (0, 1):
        f = H/'k3p'/f'{stem}-r{rep}.json'
        if f.exists():
            out[f'new{rep}'] = json.loads(f.read_text())['executed']
    out['old'] = json.loads((P16/'labels'/b/'k3'/'judgments'/f'{stem}.json').read_text())[0]['candidate']
    out['pre'] = json.loads((P16/'labels'/b/'pre'/f'{stem}.json').read_text())['executed']
    for k, sub in (('lk3p', 'k3p'), ('lpre', 'pre')):
        f = P16/'labels-luna'/b/sub/f'{stem}.json'
        if f.exists():
            out[k] = json.loads(f.read_text())['executed']
    return out


data = []
for r in rows:
    g = get(r)
    if 'new0' not in g:
        continue
    tgt = sorted(set(r['executed'])); ntext = {e: r['texts'][r['executed'].index(e)] for e in tgt}
    data.append(dict(state=r['state'], labels=g, chance=1 / len(tgt), types={e: tool(t.split('</think>')[-1]) for e, t in ntext.items()}))
rng = np.random.default_rng(20261119)
states = sorted({d['state'] for d in data}); sidx = {s: i for i, s in enumerate(states)}
boots = [rng.integers(0, len(states), len(states)) for _ in range(10000)]


def agree(a, b):
    xs = [(sidx[d['state']], float(d['labels'][a] == d['labels'][b]), d['chance']) for d in data if a in d['labels'] and b in d['labels']]
    if not xs:
        return None
    s = np.array([x[0] for x in xs]); v = np.array([x[1] for x in xs]); c = np.array([x[2] for x in xs])
    per = collections.defaultdict(list)
    for si, vi in zip(s, v): per[si].append(vi)
    sums = np.zeros(len(states)); cnt = np.zeros(len(states))
    for si, vs in per.items(): sums[si] = sum(vs); cnt[si] = len(vs)
    bs = [sums[b].sum() / max(cnt[b].sum(), 1) for b in boots]
    return dict(n=len(xs), agree=100 * v.mean(), ci=[100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5)], chance=100 * c.mean())


pairs = [('new0', 'new1', 'test-retest: new vs new (same panel, 2nd call)'), ('new0', 'old', 'new (matched) vs old K=3'),
         ('new0', 'pre', 'new K=3 vs GPT-5.5 pre'), ('old', 'pre', 'old K=3 vs GPT-5.5 pre'),
         ('new0', 'lk3p', 'new vs Luna K=3 (matched)'), ('old', 'lk3p', 'old vs Luna K=3 (matched)'), ('lk3p', 'lpre', 'Luna K=3 vs Luna pre')]
res = dict(n_panels=len(data), n_states=len(states), pairs={})
for a, b, name in pairs:
    x = agree(a, b)
    if x:
        res['pairs'][name] = x; print(f"{name:48s} n={x['n']:4d} agree {x['agree']:.1f}% [{x['ci'][0]:.1f}, {x['ci'][1]:.1f}]  chance {x['chance']:.1f}%")
mix = {}
for k in ('new0', 'old', 'pre', 'lk3p', 'lpre'):
    c = collections.Counter(d['types'][d['labels'][k]] for d in data if k in d['labels']); n = sum(c.values())
    mix[k] = {t: round(100 * v / n, 1) for t, v in c.most_common(6)}
    print(f"pick type mix {k:5s}", mix[k])
res['type_mix'] = mix
led = json.loads((H/'ledger.json').read_text()); res['ledger'] = led
print('ledger', led)
(H/'analysis.json').write_text(json.dumps(res, indent=1))
