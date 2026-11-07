#!/usr/bin/env python3
"""Recipe 3 (GPT-5.5 K=3) 4B decoder LoRA data ablations, live 3 runs for pps1/pps4/s250/s1000 and 1 run for s500/pps2 (2026-11-04/05) vs same-window 3-run references
(eval-ctrl, 2026-11-06; judged late via deferjudge/ after the API key died at 11:26; earlier SFT 11-03 / full 11-04 kept for drift). SFT actor, N=5, T1.0/p0.9,
30 steps, all tasks (OM2W 300, WV 595, DS 150). Per-task success averaged over a condition's runs. Valid-only = successes/valid;
mean overall = unweighted mean of benchmark overall rates. Paired bootstrap over tasks (10,000, seed 20261029), benchmarks resampled
independently. Task-weighted: weights 300/595/150 of 1,045. """
import json, sys
from pathlib import Path
import numpy as np
E = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations'); A = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/recipe3-ablations-20261104/eval')
BENCH = ('online_mind2web', 'webvoyager', 'deepshop'); N = dict(online_mind2web=300, webvoyager=595, deepshop=150)
C = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/recipe3-ablations-20261104/eval-ctrl')   # same-window controls, 2026-11-06
REF = {'sft': [C/f'run{i}'/'{b}'/'sft1' for i in (1, 2, 3)],
       'full': [C/f'run{i}'/'{b}'/'arm5-9fadc8dc4655' for i in (1, 2, 3)],
       'sft_1103': [E/'controls-sameday-20261103'/f'run{i}'/'{b}'/'sft1' for i in (1, 2, 3)],
       'full_1104': [E/'sameday-lora-20261104'/f'run{i}'/'{b}'/'arm5-9fadc8dc4655' for i in (1, 2, 3)]}
ABL = {'s250': 'arm5-d6c492ad4c40', 's500': 'arm5-125542b9f216', 's1000': 'arm5-fc760b23e227',
       'pps1': None, 'pps2': 'arm5-4ac334f69f45', 'pps4': 'arm5-9117fa5a37bb'}
for line in (A/'jobs.txt').read_text().splitlines():
    p = line.split()
    if p[1] == 'run1' and p[3] == 'webvoyager': ABL[p[2]] = Path(p[5]).name
RERUN = ('pps1', 'pps4', 's250', 's1000')   # runs 2-3 approved 2026-11-04 ~21:00; s500 and pps2 stay 1 run
SRC = REF | {k: [A/r/'{b}'/v for r in (('run1', 'run2', 'run3') if k in RERUN else ('run1',))] for k, v in ABL.items()}
rng = np.random.default_rng(20261029)
def load(d, b):
    d = Path(str(d).format(b=b))/b; keys = json.loads((d/'plan.json').read_text())['keys']; out = {}
    for t, k in keys.items():
        r = json.loads((d/'collection'/'records'/f'{k}.json').read_text())
        assert not str(r.get('judge_text')).startswith(('DEFERRED', 'All judge API retry')), (d, t, 'unjudged record')
        v = str(r.get('valid')) == 'True'
        out[t] = (v, v and float(r.get('reward') or 0) >= 1)
    assert len(out) == N[b], (d, len(out)); return out
data = {c: [{b: load(d, b) for b in BENCH} for d in ds] for c, ds in SRC.items()}
tasks = {b: sorted(data['sft'][0][b]) for b in BENCH}
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
W = {b: N[b] / sum(N.values()) for b in BENCH}; wrng = np.random.default_rng(20261029)
def wboot(a, c):
    xs = {b: vec(a, b) - vec(c, b) for b in BENCH}; est = sum(W[b] * xs[b].mean() for b in BENCH)
    bs = sum(W[b] * xs[b][wrng.integers(0, len(xs[b]), (10000, len(xs[b])))].mean(1) for b in BENCH)
    return [round(100*est, 1), round(100*np.percentile(bs, 2.5), 1), round(100*np.percentile(bs, 97.5), 1)]
def wrate(rs):
    sc = sum(x[1] for d in rs for b in BENCH for x in d[b].values()); va = sum(x[0] for d in rs for b in BENCH for x in d[b].values())
    n = sum(len(d[b]) for d in rs for b in BENCH); return dict(overall=round(100*sc/n, 1), valid_only=round(100*sc/va, 1))
rates = {c: rate(v) for c, v in data.items()}
paired = {f'{c} - {r}': {**{b: boot([vec(data[c], b) - vec(data[r], b)]) for b in BENCH},
                         'mean': boot([vec(data[c], b) - vec(data[r], b) for b in BENCH])}
          for c in list(ABL) + ['full', 'sft_1103', 'full_1104'] for r in ('sft', 'full') if c != r}
weighted = {c: dict(**wrate(data[c]), minus_sft=None if c == 'sft' else wboot(data[c], data['sft']),
                    minus_full=None if c in ('sft', 'full') else wboot(data[c], data['full'])) for c in SRC}
for c in ['sft', 'sft_1103', 'full', 'full_1104'] + list(ABL):
    r = rates[c]; print(f'{c:<6}', *(f"{r[b]['valid_only']:>5}" for b in BENCH), r['mean_overall'], r['mean_valid_only'], 'W', weighted[c])
for k, v in paired.items(): print(f'{k:<13}', 'mean', v['mean'], '| OM2W', v['online_mind2web'], 'WV', v['webvoyager'], 'DS', v['deepshop'])
if len(sys.argv) > 1: Path(sys.argv[1]).write_text(json.dumps(dict(note=__doc__, sources={k: [str(x) for x in v] for k, v in SRC.items()},
                                                                   rates=rates, paired=paired, weights=W, weighted=weighted), indent=1))
