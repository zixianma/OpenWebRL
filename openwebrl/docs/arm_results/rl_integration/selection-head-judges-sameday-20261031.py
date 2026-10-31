#!/usr/bin/env python3
"""Same-day (2026-10-31) comparison: GPT-5.5 / Sol K=3 heads (run 1 = judge-models-20261030, runs 2-3 here), Luna 2,612 head,
Piotr-data head, Piotr SelectionARM (runs 1-3 here). Success = valid and reward == 1. Per benchmark, only complete runs are used
and per-task success is averaged over them. Paired bootstrap over tasks (10,000 draws, seed 0) on equal and task-weighted means."""
import json, glob, os, collections, time
import numpy as np
E = '/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations'; S = E + '/sameday-20261031'
BENCH = ('online_mind2web', 'webvoyager', 'deepshop'); N = dict(online_mind2web=300, webvoyager=595, deepshop=150)
C = {'Sol': ('head5-ed0dd34e4875', [E + '/judge-models-20261030', S + '/run2', S + '/run3']),
     'GPT-5.5': ('head5-f1defdcdfc85', [E + '/judge-models-20261030', S + '/run2', S + '/run3']),
     'Luna 2612': ('head5-5ca25c38f7ad', [S + '/run1', S + '/run2', S + '/run3']),
     'Piotr-data head': ('head5-755d4cd49137', [S + '/run1', S + '/run2', S + '/run3']),
     'Piotr SelectionARM': ('piotr5', [S + '/run1', S + '/run2', S + '/run3'])}
data, used, judges, dates, bad = {}, {}, collections.defaultdict(set), collections.Counter(), []
for n, (slug, roots) in C.items():
    for b in BENCH:
        runs = []
        for root in roots:
            fs = glob.glob(f'{root}/{b}/{slug}/{b}/collection/records/*.json')
            if len(fs) != N[b]: continue
            r = {}
            for f in fs:
                x = json.load(open(f)); r[x['task_id']] = (int(bool(x['valid'] and x['reward'] == 1)), int(bool(x['valid'])))
                judges[b].add(x['judge_model']); dates[time.strftime('%m-%d', time.localtime(os.path.getmtime(f)))] += 1
            runs.append(r)
        data[n, b] = runs; used[n, b] = len(runs)
tasks = {b: sorted(data['Sol', b][0]) for b in BENCH}
for (n, b), runs in data.items():
    for r in runs: assert sorted(r) == tasks[b], (n, b)
W = {b: N[b] / sum(N.values()) for b in BENCH}
vec = {k: np.mean([[r[t][0] for t in tasks[k[1]]] for r in v], 0) for k, v in data.items()}
rng = np.random.default_rng(0); idx = {b: rng.integers(0, N[b], (10000, N[b])) for b in BENCH}
def rate(n):
    o = {}
    for b in BENCH:
        s = sum(sum(v[0] for v in r.values()) for r in data[n, b]); v = sum(sum(v[1] for v in r.values()) for r in data[n, b])
        o[b] = dict(runs=used[n, b], successes=s, valid=v, episodes=N[b] * used[n, b], valid_only=round(100 * s / v, 1))
    ov = {b: o[b]['successes'] / o[b]['episodes'] for b in BENCH}; vo = {b: o[b]['successes'] / o[b]['valid'] for b in BENCH}
    o['mean_overall'] = round(100 * np.mean(list(ov.values())), 1); o['mean_valid_only'] = round(100 * np.mean(list(vo.values())), 1)
    o['tw_overall'] = round(100 * sum(W[b] * ov[b] for b in BENCH), 1)
    o['tw_valid_only'] = round(100 * sum(o[b]['successes'] for b in BENCH) / sum(o[b]['valid'] for b in BENCH), 1)
    return o
def diff(a, c, w):
    d = {b: vec[a, b] - vec[c, b] for b in BENCH}; s = sum(w[b] * d[b][idx[b]].mean(1) for b in BENCH) * 100
    return [round(float(sum(w[b] * d[b].mean() for b in BENCH) * 100), 1), round(float(np.percentile(s, 2.5)), 1), round(float(np.percentile(s, 97.5)), 1)]
EQ = {b: 1 / 3 for b in BENCH}
out = dict(judges={b: sorted(v) for b, v in judges.items()}, record_dates=dict(dates), rates={n: rate(n) for n in C},
           diffs={ref: {n: dict(equal=diff(n, ref, EQ), task_weighted=diff(n, ref, W)) for n in C if n != ref}
                  for ref in ('Luna 2612', 'Piotr-data head', 'Piotr SelectionARM')})
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sameday31.json'), 'w'), indent=1)
print(json.dumps(out, indent=1))
