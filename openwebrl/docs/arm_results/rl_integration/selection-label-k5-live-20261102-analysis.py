#!/usr/bin/env python3
"""Live H = 5 lookahead-label head vs H = 3 head vs SFT alone, same day (head-k5-20261101/run1, 2026-11-01/02), SFT actor,
1280x1000, N=5, T1.0/p0.9, 30 steps, all tasks (OM2W 300, WV 595, DS 150). One run each. Valid-only = successes / valid;
mean overall = unweighted mean of benchmark overall rates. Paired bootstrap over tasks (10,000, seed 20261029), benchmarks
resampled independently. Context: H = 3 head and SFT pooled over the 3 runs of sameday-20261029 (cross-day)."""
import json, sys
from pathlib import Path
import numpy as np
R = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations'); BENCH = ('online_mind2web', 'webvoyager', 'deepshop')
SLUG = {'sft': 'sft1', 'h3': 'head5-11d5eff8ffe1', 'h5': 'head5-ccf9e214b15a'}
tasks = {b: list(json.loads((R/'crossbench-full-t10-20261018'/b/'sft1'/b/'plan.json').read_text())['keys']) for b in BENCH}
rng = np.random.default_rng(20261029)
def load(root, slug, b):
    d = R/root/b/slug/b; keys = json.loads((d/'plan.json').read_text())['keys']; out = {}
    for t in tasks[b]:
        r = json.loads((d/'collection'/'records'/f'{keys[t]}.json').read_text()); v = str(r.get('valid')) == 'True'
        out[t] = (v, v and float(r.get('reward') or 0) >= 1)
    return out
today = {c: [{b: load('head-k5-20261101/run1', s, b) for b in BENCH}] for c, s in SLUG.items()}
oct29 = {c: [{b: load(rt, SLUG[c], b) for b in BENCH} for rt in ('sameday-20261029', 'sameday-20261029/rerun2', 'sameday-20261029/rerun3')] for c in ('sft', 'h3')}
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
rates = {f'today/{c}': rate(v) for c, v in today.items()} | {f'oct29_3run/{c}': rate(v) for c, v in oct29.items()}
for k, r in rates.items(): print(f"{k:<16}", *(f"{r[b]['valid_only']:>5}" for b in BENCH), r['mean_overall'], r['mean_valid_only'], [r[b]['valid'] for b in BENCH])
paired = {'h5 - sft': diff(today['h5'], today['sft']), 'h3 - sft': diff(today['h3'], today['sft']), 'h5 - h3': diff(today['h5'], today['h3']),
          'h3 today - h3 oct29 (cross-day)': diff(today['h3'], oct29['h3']), 'sft today - sft oct29 (cross-day)': diff(today['sft'], oct29['sft'])}
for k, v in paired.items(): print(k, 'mean', v['mean'], '| OM2W', v['online_mind2web'], 'WV', v['webvoyager'], 'DS', v['deepshop'])
if len(sys.argv) > 1: Path(sys.argv[1]).write_text(json.dumps(dict(note=__doc__, slugs=SLUG, rates=rates, paired=paired), indent=1))
