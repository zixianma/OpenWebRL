#!/usr/bin/env python3
"""Live 3-run comparison of selection heads trained on each label source (pre, post, lookahead H1/H3/H5, Oracle) vs SFT alone.
head-k5-20261101 run1/rerun2/rerun3; run1 of SFT/H3/H5 is 2026-11-01/02, everything else 2026-11-02/03. SFT actor, 1280x1000,
N=5, T1.0/p0.9, 30 steps, all tasks (OM2W 300, WV 595, DS 150). Per-task success averaged over 3 runs. Valid-only = successes/valid;
mean overall = unweighted mean of benchmark overall rates. Paired bootstrap over tasks (10,000, seed 20261029), benchmarks
resampled independently. Task-weighted: each of the 1,045 tasks counts once (weights 300/595/150 of 1,045)."""
import json, sys
from pathlib import Path
import numpy as np
R = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations/head-k5-20261101'); BENCH = ('online_mind2web', 'webvoyager', 'deepshop')
RUNS = ('run1', 'rerun2', 'rerun3'); N = dict(online_mind2web=300, webvoyager=595, deepshop=150)
SLUG = {'sft': 'sft1', 'pre': 'head5-2af75856eeed', 'post': 'head5-73757f12d4e2', 'h1': 'head5-fd5bf58b5fa6',
        'h3': 'head5-11d5eff8ffe1', 'h5': 'head5-ccf9e214b15a', 'oracle': 'head5-7586978ccf07'}
rng = np.random.default_rng(20261029)
def load(run, slug, b):
    d = R/run/b/slug/b; keys = json.loads((d/'plan.json').read_text())['keys']; out = {}
    for t, k in keys.items():
        r = json.loads((d/'collection'/'records'/f'{k}.json').read_text()); v = str(r.get('valid')) == 'True'
        out[t] = (v, v and float(r.get('reward') or 0) >= 1)
    assert len(out) == N[b], (run, slug, b, len(out)); return out
data = {c: [{b: load(r, s, b) for b in BENCH} for r in RUNS] for c, s in SLUG.items()}
tasks = {b: sorted(data['sft'][0][b]) for b in BENCH}
for c in data:
    for d in data[c]:
        for b in BENCH: assert sorted(d[b]) == tasks[b]
def vec(rs, b): return np.array([np.mean([r[b][t][1] for r in rs]) for t in tasks[b]], float)
def boot(xs):
    est = np.mean([x.mean() for x in xs]); bs = np.mean([x[rng.integers(0, len(x), (10000, len(x)))].mean(1) for x in xs], 0)
    return [round(100*est, 1), round(100*np.percentile(bs, 2.5), 1), round(100*np.percentile(bs, 97.5), 1)]
def diff(a, c):
    d = {b: boot([vec(a, b) - vec(c, b)]) for b in BENCH}; d['mean'] = boot([vec(a, b) - vec(c, b) for b in BENCH]); return d
def rate(rs):
    r = {}
    for b in BENCH:
        sc = sum(x[1] for d in rs for x in d[b].values()); va = sum(x[0] for d in rs for x in d[b].values()); n = sum(len(d[b]) for d in rs)
        r[b] = dict(successes=sc, valid=va, episodes=n, overall=round(100*sc/n, 1), valid_only=round(100*sc/va, 1))
    r['mean_overall'] = round(np.mean([r[b]['overall'] for b in BENCH]), 1); r['mean_valid_only'] = round(np.mean([r[b]['valid_only'] for b in BENCH]), 1)
    return r
rates = {c: rate(v) for c, v in data.items()}
per_run = {c: {r: rate([d]) for r, d in zip(RUNS, v)} for c, v in data.items()}
for c, r in sorted(rates.items(), key=lambda kv: -kv[1]['mean_overall']):
    print(f"{c:<7}", *(f"{r[b]['valid_only']:>5}" for b in BENCH), r['mean_overall'], r['mean_valid_only'], [r[b]['valid'] for b in BENCH])
paired = {f'{c} - sft': diff(data[c], data['sft']) for c in SLUG if c != 'sft'} | {f'{c} - h3': diff(data[c], data['h3']) for c in SLUG if c not in ('sft', 'h3')}
for k, v in paired.items(): print(f'{k:<13} mean', v['mean'], '| OM2W', v['online_mind2web'], 'WV', v['webvoyager'], 'DS', v['deepshop'])
print('per-run DS valid-only', {c: [per_run[c][r]['deepshop']['valid_only'] for r in RUNS] for c in SLUG})
# Task-weighted (300/595/150 of 1,045): rates pooled over all tasks; paired bootstrap of the weighted difference, fresh rng, same seed.
W = {b: N[b] / sum(N.values()) for b in BENCH}
wrng = np.random.default_rng(20261029)
def wboot(a, c):
    xs = {b: vec(a, b) - vec(c, b) for b in BENCH}
    est = sum(W[b] * xs[b].mean() for b in BENCH)
    bs = sum(W[b] * xs[b][wrng.integers(0, len(xs[b]), (10000, len(xs[b])))].mean(1) for b in BENCH)
    return [round(100*est, 1), round(100*np.percentile(bs, 2.5), 1), round(100*np.percentile(bs, 97.5), 1)]
def wrate(rs):
    sc = sum(x[1] for d in rs for b in BENCH for x in d[b].values()); va = sum(x[0] for d in rs for b in BENCH for x in d[b].values())
    n = sum(len(d[b]) for d in rs for b in BENCH)
    return dict(overall=round(100*sc/n, 1), valid_only=round(100*sc/va, 1))
weighted = {c: dict(**wrate(data[c]), minus_sft=None if c == 'sft' else wboot(data[c], data['sft']),
                    minus_h3=None if c in ('sft', 'h3') else wboot(data[c], data['h3'])) for c in SLUG}
for c, r in sorted(weighted.items(), key=lambda kv: -kv[1]['overall']): print('weighted', f'{c:<7}', r)
if len(sys.argv) > 1: Path(sys.argv[1]).write_text(json.dumps(dict(note=__doc__, slugs=SLUG, rates=rates, per_run=per_run, paired=paired,
                                                                   weights=W, weighted=weighted), indent=1))
