#!/usr/bin/env python3
"""Paired task bootstrap (10,000, seed 20261029 per pair; benchmarks resampled independently), all runs at 4,096 tokens in
the 10-30 window. Mean overall success; invalid = failure. Pairs: each RL-actor selector − RL alone, RL-state head − SFT-trained
head and − SelectionARM, SelectionARM − SFT-trained head, RL alone − SFT 4B alone."""
import json, numpy as np
E = '/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations/rlactor-4096-20261030'; B = ('online_mind2web', 'webvoyager', 'deepshop')
RUN = {'rl_alone': 'rl/sft1', 'rl_arm': 'rl/piotr5', 'rl_h1380x': 'rl/head5x-11d5eff8ffe1', 'rl_hrl': 'rl/head5-d954817f5c0e', 'sft_alone': 'sft/sft1'}
PAIRS = [('rl_hrl', 'rl_h1380x'), ('rl_arm', 'rl_alone'), ('rl_h1380x', 'rl_alone'), ('rl_hrl', 'rl_alone'), ('rl_hrl', 'rl_arm'),
         ('rl_arm', 'rl_h1380x'), ('rl_alone', 'sft_alone')]
def load(run, b):
    root, slug = RUN[run].split('/'); d = f'{E}/{root}/{b}/{slug}/{b}'; keys = json.load(open(d + '/plan.json'))['keys']; o = {}
    for t, k in keys.items():
        r = json.load(open(f'{d}/collection/records/{k}.json')); o[t] = float(str(r['valid']) == 'True' and float(r.get('reward') or 0) >= 1)
    return o
out = {}
for x_, y_ in PAIRS:
    rng = np.random.default_rng(20261029); dif = []; per = {}
    for b in B:
        a, c = load(x_, b), load(y_, b); ts = sorted(a); assert sorted(c) == ts
        x = np.array([a[t] - c[t] for t in ts]); dif.append(x)
        bs = x[rng.integers(0, len(x), (10000, len(x)))].mean(1); per[b] = [round(100 * x.mean(), 1), round(100 * np.percentile(bs, 2.5), 1), round(100 * np.percentile(bs, 97.5), 1)]
    bs = np.mean([x[rng.integers(0, len(x), (10000, len(x)))].mean(1) for x in dif], 0)
    per['mean'] = [round(100 * np.mean([x.mean() for x in dif]), 1), round(100 * np.percentile(bs, 2.5), 1), round(100 * np.percentile(bs, 97.5), 1)]
    out[f'{x_} - {y_}'] = per; print(f'{x_} - {y_}', json.dumps(per))
json.dump(out, open(E + '/analysis/paired.json', 'w'), indent=1)
