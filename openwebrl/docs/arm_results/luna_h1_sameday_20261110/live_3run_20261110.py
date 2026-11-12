#!/usr/bin/env python3
"""Luna p16 4B decoder LoRAs, live same-night eval (2026-11-10 22:49 runs 1(-2), top-up 2026-11-11 ~00:50 runs 2-3): pre (no evidence),
H = 1 (immediate result + screenshot) and H = 4 (+3 actor continuation steps) labels, 3 runs each. SFT actor, N=5, T1.0/p0.9,
all tasks (OM2W 300, WV 595, DS 150), judge-deferred then rejudged (gpt-4o WV/DS, Terra OM2W). Per-task success averaged over a
condition's runs. Valid-only = successes/valid; mean = unweighted mean of benchmark rates. Paired bootstrap over tasks (10,000,
seed 20261029), benchmarks resampled independently. H = 4 runs 2-3 and pre run 3 OM2W/DS ran on eval sources edited at 00:59
(re-prepared after a source-drift start failure); H = 4 run 3 OM2W resumed at 94/300 after a GPU fault."""
import json, sys
from pathlib import Path
import numpy as np
E = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations/luna-h1-sameday-20261110')
import os; BENCH = tuple(os.environ.get('BENCH', 'online_mind2web,webvoyager,deepshop').split(',')); N = dict(online_mind2web=300, webvoyager=595, deepshop=150)
SEL = {'pre': 'arm5-3b148a583f85', 'h1': 'arm5-5de0d929af3b', 'h4': 'arm5-2ed62dd7c4dc'}
SRC = {c: [E/c/f'run{i}'/'{b}'/h for i in (1, 2, 3)] for c, h in SEL.items()}
SRC['h1_runs12'] = SRC['h1'][:2]; SRC['pre_run1'] = SRC['pre'][:1]; SRC['h4_run1'] = SRC['h4'][:1]   # the originally planned 4 runs
rng = np.random.default_rng(20261029)
def load(d, b):
    d = Path(str(d).format(b=b))/b; keys = json.loads((d/'plan.json').read_text())['keys']; out = {}
    for t, k in keys.items():
        r = json.loads((d/'collection'/'records'/f'{k}.json').read_text())
        assert not str(r.get('judge_text')).startswith(('DEFERRED', 'All judge API retry')), (d, t, 'unjudged record')
        v = str(r.get('valid')) == 'True'
        out[t] = (v, v and float(r.get('reward') or 0) >= 1)
    assert len(out) == N[b], (d, len(out)); return out
cache = {}
def get(d, b): return cache.setdefault((str(d), b), load(d, b))
data = {c: [{b: get(d, b) for b in BENCH} for d in ds] for c, ds in SRC.items()}
tasks = {b: sorted(data['pre'][0][b]) for b in BENCH}
for c in data:
    for d in data[c]:
        for b in BENCH: assert sorted(d[b]) == tasks[b], (c, b)
def vec(rs, b): return np.array([np.mean([r[b][t][1] for r in rs]) for t in tasks[b]], float)
def boot(xs):
    est = np.mean([x.mean() for x in xs]); bs = np.mean([x[rng.integers(0, len(x), (10000, len(x)))].mean(1) for x in xs], 0)
    return [round(100*est, 1), round(100*np.percentile(bs, 2.5), 1), round(100*np.percentile(bs, 97.5), 1)]
def rate(rs):
    r = {}
    for b in BENCH:
        sc = sum(x[1] for d in rs for x in d[b].values()); va = sum(x[0] for d in rs for x in d[b].values()); n = sum(len(d[b]) for d in rs)
        r[b] = dict(successes=sc, valid=va, episodes=n, overall=round(100*sc/n, 1), valid_only=round(100*sc/va, 1))
    r['mean_overall'] = round(np.mean([r[b]['overall'] for b in BENCH]), 1); r['mean_valid_only'] = round(np.mean([r[b]['valid_only'] for b in BENCH]), 1)
    return r
rates = {c: rate(v) for c, v in data.items()}
per_run = {f'{c}/run{i+1}': rate([d]) for c in SEL for i, d in enumerate(data[c])}
paired = {f'{a} - {r}': {**{b: boot([vec(data[a], b) - vec(data[r], b)]) for b in BENCH},
                         'mean': boot([vec(data[a], b) - vec(data[r], b) for b in BENCH])}
          for a, r in (('h1', 'pre'), ('h4', 'pre'), ('h1', 'h4'), ('h1_runs12', 'pre_run1'), ('h4_run1', 'pre_run1'))}
for c in list(SEL) + ['h1_runs12', 'pre_run1', 'h4_run1']:
    r = rates[c]; print(f'{c:<10}', *(f"{r[b]['valid_only']:>5}" for b in BENCH), 'ov', r['mean_overall'], 'vo', r['mean_valid_only'])
for k, r in per_run.items(): print(f'{k:<10}', *(f"{r[b]['valid_only']:>5}" for b in BENCH), 'ov', r['mean_overall'], 'vo', r['mean_valid_only'])
for k, v in paired.items(): print(f'{k:<20}', 'mean', v['mean'], '| OM2W', v['online_mind2web'], 'WV', v['webvoyager'], 'DS', v.get('deepshop'))
if len(sys.argv) > 1: Path(sys.argv[1]).write_text(json.dumps(dict(note=__doc__, sources={k: [str(x) for x in v] for k, v in SRC.items()},
                                                                   rates=rates, per_run=per_run, paired=paired), indent=1))
