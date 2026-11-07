#!/usr/bin/env python3
"""Recipe 3 GPT-5.5 K=3 selection-head architecture/loss ablations (9 heads, 1 run each, rolled out 2026-11-06 judge-deferred and
judged afterwards with the native judges) vs the default head (joint transformer, soft-CE; same window, 1 run) and same-window SFT 4B
(eval-ctrl, 3 runs, 2026-11-06). SFT actor, N=5 head5, T1.0/p0.9, 30 steps, all tasks (OM2W 300, WV 595, DS 150). Valid-only =
successes/valid; mean overall = unweighted mean of benchmark overall rates. Paired bootstrap over tasks (10,000, seed 20261029),
benchmarks resampled independently. Task-weighted: 300/595/150 of 1,045."""
import json, sys
from pathlib import Path
import numpy as np
W = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/recipe3-ablations-20261104')
BENCH = ('online_mind2web', 'webvoyager', 'deepshop'); N = dict(online_mind2web=300, webvoyager=595, deepshop=150)
SLUG = {}
for line in (W/'eval-arch'/'jobs.txt').read_text().splitlines():
    p = line.split(); SLUG[p[2]] = Path(p[5]).name
SRC = {'sft': [W/'eval-ctrl'/f'run{i}'/'{b}'/'sft1' for i in (1, 2, 3)]} | {k: [W/'eval-arch'/'run1'/'{b}'/v] for k, v in SLUG.items()}
DEF = 'joint-transformer-softce-default'
rng = np.random.default_rng(20261029)
def load(d, b):
    d = Path(str(d).format(b=b))/b; keys = json.loads((d/'plan.json').read_text())['keys']; out = {}
    for t, k in keys.items():
        r = json.loads((d/'collection'/'records'/f'{k}.json').read_text())
        assert not str(r.get('judge_text')).startswith(('DEFERRED', 'All judge API retry')), (d, t, 'unjudged record')
        v = str(r.get('valid')) == 'True'; out[t] = (v, v and float(r.get('reward') or 0) >= 1)
    assert len(out) == N[b], (d, len(out)); return out
data = {c: [{b: load(d, b) for b in BENCH} for d in ds] for c, ds in SRC.items()}
tasks = {b: sorted(data['sft'][0][b]) for b in BENCH}
def vec(rs, b): return np.array([np.mean([r[b][t][1] for r in rs]) for t in tasks[b]], float)
def boot(xs):
    est = np.mean([x.mean() for x in xs]); bs = np.mean([x[rng.integers(0, len(x), (10000, len(x)))].mean(1) for x in xs], 0)
    return [round(100*est, 1), round(100*np.percentile(bs, 2.5), 1), round(100*np.percentile(bs, 97.5), 1)]
def rate(rs):
    r = {}
    for b in BENCH:
        sc = sum(x[1] for d in rs for x in d[b].values()); va = sum(x[0] for d in rs for x in d[b].values()); n = sum(len(d[b]) for d in rs)
        r[b] = dict(successes=int(sc), valid=int(va), episodes=n, overall=round(100*sc/n, 1), valid_only=round(100*sc/va, 1))
    r['mean_overall'] = round(float(np.mean([r[b]['overall'] for b in BENCH])), 1)
    r['mean_valid_only'] = round(float(np.mean([r[b]['valid_only'] for b in BENCH])), 1); return r
Wt = {b: N[b] / sum(N.values()) for b in BENCH}; wrng = np.random.default_rng(20261029)
def wboot(a, c):
    xs = {b: vec(a, b) - vec(c, b) for b in BENCH}; est = sum(Wt[b] * xs[b].mean() for b in BENCH)
    bs = sum(Wt[b] * xs[b][wrng.integers(0, len(xs[b]), (10000, len(xs[b])))].mean(1) for b in BENCH)
    return [round(100*est, 1), round(100*np.percentile(bs, 2.5), 1), round(100*np.percentile(bs, 97.5), 1)]
rates = {c: rate(v) for c, v in data.items()}
paired = {f'{c} - {r}': {**{b: boot([vec(data[c], b) - vec(data[r], b)]) for b in BENCH},
                         'mean': boot([vec(data[c], b) - vec(data[r], b) for b in BENCH])}
          for c in SLUG for r in ('sft', DEF) if c != r}
weighted = {c: dict(minus_sft=wboot(data[c], data['sft']), minus_default=None if c == DEF else wboot(data[c], data[DEF])) for c in SLUG}
for c in ['sft'] + sorted(SLUG, key=lambda c: -rates[c]['mean_overall']):
    r = rates[c]; print(f'{c:<34}', *(f"{r[b]['valid_only']:>5}" for b in BENCH), f"{r['mean_overall']:>5}", f"{r['mean_valid_only']:>5}",
                        '' if c == 'sft' else paired[f'{c} - sft']['mean'], '' if c in ('sft', DEF) else paired[f'{c} - {DEF}']['mean'])
if len(sys.argv) > 1: Path(sys.argv[1]).write_text(json.dumps(dict(note=__doc__, sources={k: [str(x) for x in v] for k, v in SRC.items()},
                                                                   rates=rates, paired=paired, weights=Wt, weighted=weighted), indent=1))
