#!/usr/bin/env python3
"""Recipe 3 GPT-5.5 K=3 selection heads, second round (2026-11-07): 2 x 2 of attention (joint / independent) x scorer (transformer /
MLP), all soft CE, 2 live runs each. Run 1 of joint-MLP, independent-transformer and independent-MLP is 2026-11-06; their run 2 and both
runs of the retrained default (joint transformer soft CE, source-arch/train_arch.sbatch, byte-identical to heads/full-seed0-repro) are
2026-11-07. Also: the original default head (heads-all2/k3-best/seed0, run 1 only, 11-06) and same-window SFT 4B (eval-ctrl, 3 runs,
11-06). All rolled out judge-deferred and judged afterwards with the native judges (OM2W gpt-5.6-terra, WV/DS gpt-4o). SFT actor, N=5,
T1.0/p0.9, 30 steps, OM2W 300 / WV 595 / DS 150 tasks. Per-task success averaged over runs; mean overall = unweighted mean of the
benchmark overall rates. Paired bootstrap over tasks (10,000, seed 20261029), benchmarks resampled independently."""
import json, sys
from pathlib import Path
import numpy as np
W = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/recipe3-ablations-20261104'); A = W/'eval-arch'
BENCH = ('online_mind2web', 'webvoyager', 'deepshop'); N = dict(online_mind2web=300, webvoyager=595, deepshop=150)
SLUG = {}
for line in (A/'jobs.txt').read_text().splitlines():
    p = line.split(); SLUG[p[2]] = Path(p[5]).name
HEADS = {'joint-transformer': 'joint-transformer-softce', 'joint-mlp': 'joint-mlp-softce',
         'independent-transformer': 'independent-transformer-softce', 'independent-mlp': 'independent-mlp-softce'}
SRC = {'sft': [W/'eval-ctrl'/f'run{i}'/'{b}'/'sft1' for i in (1, 2, 3)],
       'original-default': [A/'run1'/'{b}'/SLUG['joint-transformer-softce-default']]}
SRC |= {h: [A/r/'{b}'/SLUG[l] for r in ('run1', 'run2')] for h, l in HEADS.items()}
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
def contrast(plus, minus):     # mean of `plus` conditions - mean of `minus` conditions, per-task, paired
    f = lambda cs, b: np.mean([vec(data[c], b) for c in cs], 0)
    return {**{b: boot([f(plus, b) - f(minus, b)]) for b in BENCH}, 'mean': boot([f(plus, b) - f(minus, b) for b in BENCH])}
def rate(rs):
    r = {}
    for b in BENCH:
        sc = sum(x[1] for d in rs for x in d[b].values()); va = sum(x[0] for d in rs for x in d[b].values()); n = sum(len(d[b]) for d in rs)
        r[b] = dict(successes=int(sc), valid=int(va), episodes=n, overall=round(100*sc/n, 1), valid_only=round(100*sc/va, 1))
    r['mean_overall'] = round(float(np.mean([r[b]['overall'] for b in BENCH])), 1)
    r['mean_valid_only'] = round(float(np.mean([r[b]['valid_only'] for b in BENCH])), 1); return r
rates = {c: rate(v) for c, v in data.items()}
per_run = {c: [rate([d])['mean_overall'] for d in v] for c, v in data.items()}
J, I, T, M = ['joint-transformer', 'joint-mlp'], ['independent-transformer', 'independent-mlp'], ['joint-transformer', 'independent-transformer'], ['joint-mlp', 'independent-mlp']
effects = {'mlp - transformer': contrast(M, T), 'joint - independent': contrast(J, I),
           'interaction (jm - jt) - (im - it)': contrast(['joint-mlp', 'independent-transformer'], ['joint-transformer', 'independent-mlp']),
           'retrained default - original default': contrast(['joint-transformer'], ['original-default'])}
effects |= {f'{h} - sft': contrast([h], ['sft']) for h in HEADS} | {f'{h} - joint-transformer': contrast([h], ['joint-transformer']) for h in HEADS if h != 'joint-transformer'}
for c in ['sft', 'original-default'] + sorted(HEADS, key=lambda c: -rates[c]['mean_overall']):
    r = rates[c]; print(f'{c:<24}', *(f"{r[b]['valid_only']:>5}" for b in BENCH), f"{r['mean_overall']:>5}", f"{r['mean_valid_only']:>5}", 'runs', per_run[c])
for k, v in effects.items(): print(f'{k:<38}', 'mean', v['mean'], '| OM2W', v['online_mind2web'], 'WV', v['webvoyager'], 'DS', v['deepshop'])
if len(sys.argv) > 1: Path(sys.argv[1]).write_text(json.dumps(dict(note=__doc__, sources={k: [str(x) for x in v] for k, v in SRC.items()},
                                                                   rates=rates, per_run=per_run, effects=effects), indent=1))
