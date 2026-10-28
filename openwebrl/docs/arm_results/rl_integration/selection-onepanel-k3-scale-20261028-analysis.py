#!/usr/bin/env python3
"""Live state scaling of the one-panel-per-state Luna K=3 head on our branching states (onepanel-k3-scale-20261027):
263 / 501 / 1,003 / 2,612 states (whole-task prefixes, recipe of the 1,380-state head: distinct view, vote-share target,
batch 64, last of 20 epochs, seed 0) vs the 1,380-state head, SFT 4B, the Luna-K3 Piotr-style-panel head and the
Piotr-data head. SFT-4B actor, N=5, T1.0/p0.9, 30 steps, all tasks of OM2W (300), WebVoyager (595), DeepShop (150).
OM2W judge gpt-5.6-terra (older runs re-judged, rejudge-terra-20261023); WebVoyager/DeepShop GPT-4o. DeepShop of the
1,380-state head and SFT 4B are same-day reruns (10-27/28); their earlier DeepShop runs are reported as separate rows.
Overall counts invalid as failure; mean = unweighted mean of the three benchmark rates. Paired 95% bootstrap over
tasks (10,000, seed 20261028), benchmarks resampled independently for the mean."""
import collections, json
from pathlib import Path
import numpy as np

R = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations')
RJ = R/'rejudge-terra-20261023'/'out'
OP = 'onepanel-k3-scale-20261027'
BENCH = ('online_mind2web', 'webvoyager', 'deepshop')
SRC = {  # name: {bench: (round, slug, rejudge label or None)}
    '263 states': {b: (OP, 'head5-4b497ab0a73c', None) for b in BENCH},
    '501 states': {b: (OP, 'head5-35255d70d9f1', None) for b in BENCH},
    '1,003 states': {b: (OP, 'head5-9ab27026faef', None) for b in BENCH},
    '1,380 states': {'online_mind2web': ('sft-actions-full-20261021', 'head5-11d5eff8ffe1', 'sft-actions-full-20261021:head5-11d5eff8ffe1'),
                     'webvoyager': ('sft-actions-full-20261021', 'head5-11d5eff8ffe1', None), 'deepshop': (OP, 'head5-11d5eff8ffe1', None)},
    '2,612 states': {b: (OP, 'head5-5ca25c38f7ad', None) for b in BENCH},
    'Luna-K3 Piotr-style panels': {b: ('lunak3-panels-20261027', 'head5-cafea4f265cf', None) for b in BENCH},
    'Piotr data 10-23': {b: ('data-scale-20261023', 'head5-fc589b57c9e0', None) for b in BENCH},
    'SFT 4B': {'online_mind2web': ('crossbench-full-t10-20261018', 'sft1', 'crossbench-full-t10-20261018:sft1'),
               'webvoyager': ('crossbench-full-t10-20261018', 'sft1', None), 'deepshop': (OP, 'sft1', None)},
}
EXTRA = {'1,380 states, DeepShop 10-21': ('sft-actions-full-20261021', 'head5-11d5eff8ffe1', None),
         'SFT 4B, DeepShop 10-18': ('crossbench-full-t10-20261018', 'sft1', None),
         'SFT 4B, DeepShop 10-27 (lunak3 round)': ('lunak3-panels-20261027', 'sft1', None)}
rng = np.random.default_rng(20261028)
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


data = {(c, b): load(*src[b][:2], b, src[b][2]) for c, src in SRC.items() for b in BENCH}
extra = {c: load(*s[:2], 'deepshop', s[2]) for c, s in EXTRA.items()}
for k, v in sorted(judges.items()):
    print('judge', k, v)


def boot(per_bench):
    est = np.mean([x.mean() for x in per_bench])
    bs = np.mean([x[rng.integers(0, len(x), (10000, len(x)))].mean(1) for x in per_bench], 0)
    return [round(100 * est, 1), round(100 * np.percentile(bs, 2.5), 1), round(100 * np.percentile(bs, 97.5), 1)]


def row(c, b, v):
    n = len(v); val = sum(x[0] for x in v.values()); s = sum(x[1] for x in v.values())
    return dict(condition=c, benchmark=b, tasks=n, valid=val, successes=s, overall=round(100*s/n, 1), valid_only=round(100*s/val, 1))


rows = [row(c, b, v) for (c, b), v in data.items()] + [row(c, 'deepshop', v) for c, v in extra.items()]
for c in SRC:
    ov = [r['overall'] for r in rows if r['condition'] == c]; vo = [r['valid_only'] for r in rows if r['condition'] == c]
    rows.append(dict(condition=c, benchmark='mean', overall=round(np.mean(ov), 1), valid_only=round(np.mean(vo), 1)))
for r in rows:
    print(f"{r['condition']:<38} {r['benchmark']:<16} {r.get('successes','')!s:>4} {r.get('valid','')!s:>4} {r.get('tasks','')!s:>4} {r['overall']:>6} {r['valid_only']:>6}")

pairs = []
for a, base in [(h, '1,380 states') for h in ('263 states', '501 states', '1,003 states', '2,612 states')] + \
               [(h, 'SFT 4B') for h in ('263 states', '501 states', '1,003 states', '1,380 states', '2,612 states')] + \
               [('2,612 states', '263 states'), ('2,612 states', 'Luna-K3 Piotr-style panels'), ('2,612 states', 'Piotr data 10-23')]:
    rec = dict(a=a, b=base)
    for b in list(BENCH) + ['mean']:
        bs = BENCH if b == 'mean' else (b,)
        ov = [np.array([data[a, bb][t][1] - data[base, bb][t][1] for t in tasks[bb]], float) for bb in bs]
        vo = [np.array([data[a, bb][t][1] - data[base, bb][t][1] for t in tasks[bb] if data[a, bb][t][0] and data[base, bb][t][0]], float) for bb in bs]
        rec[b] = dict(overall=boot(ov), valid_only=boot(vo), valid_tasks=sum(len(x) for x in vo))
    pairs.append(rec); print(a, '-', base, {b: rec[b]['overall'] for b in list(BENCH) + ['mean']}, 'mean valid-only', rec['mean']['valid_only'])
d = [np.array([data['1,380 states', 'deepshop'][t][1] - extra['1,380 states, DeepShop 10-21'][t][1] for t in tasks['deepshop']], float)]
pairs.append(dict(a='1,380 states DeepShop 10-28 rerun', b='same head 10-21', deepshop=dict(overall=boot(d)))); print(pairs[-1])
out = Path(__file__).resolve().parent/'results-onepanel.json'
out.write_text(json.dumps(dict(note=__doc__, judges=[[*k, v] for k, v in judges.items()], rows=rows, paired=pairs), indent=1))
