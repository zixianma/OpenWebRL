#!/usr/bin/env python3
"""Paired task bootstrap (10,000, seed 20261029; benchmarks resampled independently): RL-state head − SFT-trained head,
both on the RL actor at 4,096 tokens, 10-30. Mean overall success; invalid = failure."""
import json, glob, numpy as np
R = '/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations/rlactor-4096-20261030/rl'; B = ('online_mind2web', 'webvoyager', 'deepshop')
def load(slug, b):
    d = f'{R}/{b}/{slug}/{b}'; keys = json.load(open(d + '/plan.json'))['keys']; o = {}
    for t, k in keys.items():
        r = json.load(open(f'{d}/collection/records/{k}.json')); o[t] = float(str(r['valid']) == 'True' and float(r.get('reward') or 0) >= 1)
    return o
rng = np.random.default_rng(20261029); dif = []; per = {}
for b in B:
    a, c = load('head5-d954817f5c0e', b), load('head5x-11d5eff8ffe1', b); ts = sorted(a)
    x = np.array([a[t] - c[t] for t in ts]); dif.append(x)
    bs = x[rng.integers(0, len(x), (10000, len(x)))].mean(1); per[b] = [round(100 * x.mean(), 1), round(100 * np.percentile(bs, 2.5), 1), round(100 * np.percentile(bs, 97.5), 1)]
bs = np.mean([x[rng.integers(0, len(x), (10000, len(x)))].mean(1) for x in dif], 0)
per['mean'] = [round(100 * np.mean([x.mean() for x in dif]), 1), round(100 * np.percentile(bs, 2.5), 1), round(100 * np.percentile(bs, 97.5), 1)]
print(json.dumps(per)); json.dump(per, open(R[:-3] + '/analysis/paired.json', 'w'))
