#!/usr/bin/env python3
"""Live comparison of the Luna-K3 Piotr-style-panel head (lunak3-panels-20261027, head5-cafea4f265cf) against the
Piotr-data head nall/seed0 (data-scale 10-23 and same-day rerun 10-25), the 1,380-state one-panel K3 head
(sft-actions-full 10-21) and SFT 4B (crossbench 10-18; DeepShop also same-day 10-27). SFT-4B actor, N=5, T1.0/p0.9,
30 steps, every task of OM2W (300), WebVoyager (595), DeepShop (150). OM2W judge gpt-5.6-terra (older runs re-judged,
rejudge-terra-20261023); WebVoyager/DeepShop GPT-4o. Overall counts invalid as failure; valid-only pairs tasks valid in
both arms. Mean = unweighted mean of the three benchmark rates. Paired 95% bootstrap over tasks (10,000, seed 20261027),
benchmarks resampled independently for the mean."""
import collections, json
from pathlib import Path
import numpy as np

R = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations')
RJ = R/'rejudge-terra-20261023'/'out'
BENCH = ('online_mind2web', 'webvoyager', 'deepshop')
COND = {  # name: (round, slug, rejudge label for OM2W or None)
    'Luna-K3 panels (new)': ('lunak3-panels-20261027', 'head5-cafea4f265cf', None),
    'Piotr data 10-23': ('data-scale-20261023', 'head5-fc589b57c9e0', None),
    'Piotr data 10-25': ('piotr-head-sameday-20261025', 'head5-fc589b57c9e0', None),
    'K3 1,380 states 1-panel': ('sft-actions-full-20261021', 'head5-11d5eff8ffe1', 'sft-actions-full-20261021:head5-11d5eff8ffe1'),
    'SFT 4B': ('crossbench-full-t10-20261018', 'sft1', 'crossbench-full-t10-20261018:sft1'),
}
rng = np.random.default_rng(20261027)
tasks = {b: list(json.loads((R/'crossbench-full-t10-20261018'/b/'sft1'/b/'plan.json').read_text())['keys']) for b in BENCH}
judges = collections.Counter()


def load(rnd, slug, b, rj):
    d = R/rnd/b/slug/b
    keys = json.loads((d/'plan.json').read_text())['keys']
    out = {}
    for t in tasks[b]:
        r = json.loads((d/'collection'/'records'/f'{keys[t]}.json').read_text())
        valid = str(r.get('valid')) == 'True'
        succ = valid and float(r.get('reward') or 0) >= 1
        jm = r.get('judge_model')
        if rj and b == 'online_mind2web' and valid:
            vp = RJ/rj/'verdicts'/f"{r['key']}.json"   # no file: episode never reached the judge, reward stays 0 (rejudge rule)
            if vp.exists():
                v = json.loads(vp.read_text()); jm = v['returned_model']
                valid = v['verdict'] is not None; succ = valid and float(v['verdict']) >= 1
            else:
                jm = 'not judged (reward 0)'
        if valid:
            judges[(rnd, slug, b, jm)] += 1
        out[t] = (valid, succ)
    return out


data = {(c, b): load(*COND[c][:2], b, COND[c][2]) for c in COND for b in BENCH}
data[('SFT 4B (DS same-day)', 'deepshop')] = load('lunak3-panels-20261027', 'sft1', 'deepshop', None)
for k, v in sorted(judges.items()):
    print('judge', k, v)


def boot(per_bench):  # per_bench: list of arrays (one per benchmark) -> mean of means
    est = np.mean([x.mean() for x in per_bench])
    bs = np.mean([x[rng.integers(0, len(x), (10000, len(x)))].mean(1) for x in per_bench], 0)
    return [round(100 * est, 1), round(100 * np.percentile(bs, 2.5), 1), round(100 * np.percentile(bs, 97.5), 1)]


rows = []
for (c, b), v in data.items():
    n = len(v); val = sum(x[0] for x in v.values()); s = sum(x[1] for x in v.values())
    rows.append(dict(condition=c, benchmark=b, tasks=n, valid=val, successes=s, overall=round(100*s/n, 1), valid_only=round(100*s/val, 1)))
for c in COND:
    ov = [r['overall'] for r in rows if r['condition'] == c]; vo = [r['valid_only'] for r in rows if r['condition'] == c]
    rows.append(dict(condition=c, benchmark='mean', overall=round(np.mean(ov), 1), valid_only=round(np.mean(vo), 1)))
for r in rows:
    print(f"{r['condition']:<26} {r['benchmark']:<16} {r.get('successes','')!s:>4} {r.get('valid','')!s:>4} {r.get('tasks','')!s:>4} {r['overall']:>6} {r['valid_only']:>6}")

new = 'Luna-K3 panels (new)'
pairs = []
for base in ('Piotr data 10-23', 'Piotr data 10-25', 'K3 1,380 states 1-panel', 'SFT 4B'):
    rec = dict(a=new, b=base)
    for b in list(BENCH) + ['mean']:
        bs = BENCH if b == 'mean' else (b,)
        ov = [np.array([data[new, bb][t][1] - data[base, bb][t][1] for t in tasks[bb]], float) for bb in bs]
        vo = [np.array([data[new, bb][t][1] - data[base, bb][t][1] for t in tasks[bb] if data[new, bb][t][0] and data[base, bb][t][0]], float) for bb in bs]
        rec[b] = dict(overall=boot(ov), valid_only=boot(vo), valid_tasks=sum(len(x) for x in vo))
    pairs.append(rec); print(json.dumps(rec))
ds = [np.array([data[new, 'deepshop'][t][1] - data['SFT 4B (DS same-day)', 'deepshop'][t][1] for t in tasks['deepshop']], float)]
pairs.append(dict(a=new, b='SFT 4B (DS same-day)', deepshop=dict(overall=boot(ds)))); print(json.dumps(pairs[-1]))
out = Path(__file__).resolve().parent/'results.json'
out.write_text(json.dumps(dict(note=__doc__, judges=[[*k, v] for k, v in judges.items()], rows=rows, paired=pairs), indent=1))
