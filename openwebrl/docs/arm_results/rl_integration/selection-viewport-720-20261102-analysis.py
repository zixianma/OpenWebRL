#!/usr/bin/env python3
"""SFT actor at 1280x720 (sft-720-20261101, 3 runs) vs the same conditions at 1280x1000 (sameday-20261029, 3 runs):
SFT alone, Recipe 2 2.6K head (head5-5ca25c38f7ad), Recipe 2 2.6K 4B LoRA (arm5-03edd8c36a91). N=5, T1.0/p0.9, 30 steps,
all tasks. Pooled valid-only = successes / valid episodes over 3 runs; mean overall = unweighted mean of benchmark overall rates.
Paired bootstrap over tasks on per-task success averaged over runs (10,000, seed 20261029), benchmarks resampled independently."""
import json, sys
from pathlib import Path
import numpy as np
R = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations')
BENCH = ('online_mind2web', 'webvoyager', 'deepshop')
SLUG = {'sft': 'sft1', 'head': 'head5-5ca25c38f7ad', 'lora': 'arm5-03edd8c36a91'}
ROOTS = {'720': ['sft-720-20261101/run1', 'sft-720-20261101/rerun2', 'sft-720-20261101/rerun3'],
         '1000': ['sameday-20261029', 'sameday-20261029/rerun2', 'sameday-20261029/rerun3']}
tasks = {b: list(json.loads((R/'crossbench-full-t10-20261018'/b/'sft1'/b/'plan.json').read_text())['keys']) for b in BENCH}
rng = np.random.default_rng(20261029)
def load(root, slug, b):
    d = R/root/b/slug/b; keys = json.loads((d/'plan.json').read_text())['keys']; out = {}
    for t in tasks[b]:
        r = json.loads((d/'collection'/'records'/f'{keys[t]}.json').read_text())
        v = str(r.get('valid')) == 'True'; out[t] = (v, v and float(r.get('reward') or 0) >= 1)
    return out
runs = {(vp, c): [{b: load(rt, s, b) for b in BENCH} for rt in ROOTS[vp]] for vp in ROOTS for c, s in SLUG.items()}
def vec(rs, b): return np.array([np.mean([r[b][t][1] for r in rs]) for t in tasks[b]], float)
def boot(xs):
    est = np.mean([x.mean() for x in xs]); bs = np.mean([x[rng.integers(0, len(x), (10000, len(x)))].mean(1) for x in xs], 0)
    return [round(100*est, 1), round(100*np.percentile(bs, 2.5), 1), round(100*np.percentile(bs, 97.5), 1)]
def diff(a, c):
    d = {b: boot([vec(a, b) - vec(c, b)]) for b in BENCH}; d['mean'] = boot([vec(a, b) - vec(c, b) for b in BENCH]); return d
rates = {}
for k, rs in runs.items():
    r = {}
    for b in BENCH:
        sc = sum(x[1] for d in rs for x in d[b].values()); va = sum(x[0] for d in rs for x in d[b].values()); n = sum(len(d[b]) for d in rs)
        r[b] = dict(successes=sc, valid=va, episodes=n, overall=round(100*sc/n, 1), valid_only=round(100*sc/va, 1))
    r['mean_overall'] = round(np.mean([r[b]['overall'] for b in BENCH]), 1); r['mean_valid_only'] = round(np.mean([r[b]['valid_only'] for b in BENCH]), 1)
    rates['/'.join(k)] = r
    print(f"{k[0]:>4} {k[1]:<5}", *(f"{r[b]['valid_only']:>5}" for b in BENCH), r['mean_overall'], r['mean_valid_only'], 'valid', [r[b]['valid'] for b in BENCH])
paired = {}
for vp in ROOTS:
    for c in ('head', 'lora'):
        paired[f'{vp}: {c} - sft'] = diff(runs[(vp, c)], runs[(vp, 'sft')])
    paired[f'{vp}: lora - head'] = diff(runs[(vp, 'lora')], runs[(vp, 'head')])
for c in SLUG:
    paired[f'{c}: 720 - 1000 (cross-day)'] = diff(runs[('720', c)], runs[('1000', c)])
paired['gain over sft: 720 - 1000 (head)'] = None
for k, v in paired.items():
    if v: print(k, 'mean', v['mean'], '| OM2W', v['online_mind2web'][0], 'WV', v['webvoyager'][0], 'DS', v['deepshop'][0])
# difference-in-differences: (sel - sft)@720 - (sel - sft)@1000, per-task
def did(c):
    xs = [(vec(runs[('720', c)], b) - vec(runs[('720', 'sft')], b)) - (vec(runs[('1000', c)], b) - vec(runs[('1000', 'sft')], b)) for b in BENCH]
    return boot(xs)
paired.pop('gain over sft: 720 - 1000 (head)')
paired['did_head'] = did('head'); paired['did_lora'] = did('lora')
print('gain-over-SFT change (720 - 1000): head', paired['did_head'], 'lora', paired['did_lora'])
if len(sys.argv) > 1: Path(sys.argv[1]).write_text(json.dumps(dict(note=__doc__, slugs=SLUG, roots=ROOTS, rates=rates, paired=paired), indent=1))
