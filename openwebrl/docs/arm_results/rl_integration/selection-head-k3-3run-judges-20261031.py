#!/usr/bin/env python3
"""K=3 arch heads pooled over 3 runs (run 1 = 2026-10-29, runs 2-3 = 2026-10-30, OM2W/DS resumed 10-31 after the OpenAI 401
outage), plus the one-run judge-label heads (GPT-5.5 / GPT-6.1 Sol high K=3 labels, 2,612 one-panel states, run 10-31).
Success = valid and reward == 1 over scheduled tasks; per task, success is averaged over a condition's runs. Mean = average of
the three benchmarks' rates. Paired bootstrap over tasks within each benchmark (10,000 draws, seed 0). Output k3_pooled.json."""
import json, glob, os
import numpy as np
E = '/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations'
S = E + '/sameday-20261029'
BENCH = ('online_mind2web', 'webvoyager', 'deepshop')
RUNS = {'run1': S, 'run2': S + '/rerun2', 'run3': S + '/rerun3'}
C = {'joint-transformer-softce': 'head5-11d5eff8ffe1', 'joint-transformer-bce': 'head5-4a92d9e12332',
     'independent-mlp-bce': 'head5-15ff2a7e9bce', 'causal-transformer-softce': 'head5-f76a5cad2863',
     'independent-transformer-softce': 'head5-040311428ed1', 'independent-transformer-bce': 'head5-6e2111c2c1e5',
     'piotr-data-head': 'head5-755d4cd49137', 'luna-2612-head': 'head5-5ca25c38f7ad', 'sft-no-selector': 'sft1'}
J = {'gpt55-2612-head': 'head5-f1defdcdfc85', 'sol-2612-head': 'head5-ed0dd34e4875'}


def load(root, slug, b):
    rows, val, judges = {}, {}, set()
    for f in glob.glob(f'{root}/{b}/{slug}/{b}/collection/records/*.json'):
        r = json.load(open(f))
        rows[r['task_id']] = int(bool(r['valid'] and r['reward'] == 1)); val[r['task_id']] = int(bool(r['valid']))
        judges.add(r['judge_model'])
    return rows, val, judges


data, judges = {}, {}
for name, slug in C.items():
    for run, root in RUNS.items():
        for b in BENCH:
            data[name, run, b] = load(root, slug, b)[:2]; judges.setdefault(b, set()).update(load(root, slug, b)[2])
for name, slug in J.items():
    for b in BENCH:
        data[name, 'run1', b] = load(E + '/judge-models-20261030', slug, b)[:2]
        judges.setdefault(b, set()).update(load(E + '/judge-models-20261030', slug, b)[2])
tasks = {b: sorted(data['sft-no-selector', 'run1', b][0]) for b in BENCH}
for k, (rows, _) in data.items():
    assert sorted(rows) == tasks[k[2]], (k, len(rows), len(tasks[k[2]]))

rng = np.random.default_rng(0)
idx = {b: rng.integers(0, len(tasks[b]), (10000, len(tasks[b]))) for b in BENCH}


def runs_of(name, which):
    return ['run1'] if name in J else which


def vec(name, which):  # per-task success averaged over runs
    return {b: np.mean([[data[name, r, b][0][t] for t in tasks[b]] for r in runs_of(name, which)], 0) for b in BENCH}


def summary(name, which):
    rs = runs_of(name, which); out = {}
    for b in BENCH:
        s = sum(sum(data[name, r, b][0].values()) for r in rs); v = sum(sum(data[name, r, b][1].values()) for r in rs)
        out[b] = dict(successes=s, valid=v, episodes=len(tasks[b]) * len(rs), valid_only=round(100 * s / v, 2))
    out['mean_overall'] = round(float(np.mean([out[b]['successes'] / out[b]['episodes'] for b in BENCH])) * 100, 2)
    out['mean_valid_only'] = round(float(np.mean([out[b]['successes'] / out[b]['valid'] for b in BENCH])) * 100, 2)
    out['runs'] = len(rs)
    return out


def diff(a, b_, which, bench=BENCH):
    x, y = vec(a, which), vec(b_, which); d = {k: x[k] - y[k] for k in bench}
    s = np.mean([d[k][idx[k]].mean(1) for k in bench], 0) * 100
    return [round(float(np.mean([d[k].mean() for k in bench]) * 100), 2), round(float(np.percentile(s, 2.5)), 2),
            round(float(np.percentile(s, 97.5)), 2)]


out = dict(judges={b: sorted(v) for b, v in judges.items()})
# record-write dates per condition, run and benchmark (file mtime, local time) to expose cross-day comparisons
import time, collections
out['record_dates'] = {f'{n}|{r}|{b}': dict(collections.Counter(time.strftime('%m-%d', time.localtime(os.path.getmtime(f)))
    for f in glob.glob(f'{(E + "/judge-models-20261030") if n in J else RUNS[r]}/{b}/{(J if n in J else C)[n]}/{b}/collection/records/*.json')))
    for n in list(C) + list(J) for r in (['run1'] if n in J else RUNS) for b in BENCH}
for label, which in (('runs123', ['run1', 'run2', 'run3']), ('runs23', ['run2', 'run3'])):
    names = list(C) + list(J)
    out[label] = dict(
        rates={n: summary(n, which) for n in names},
        vs_sft={n: diff(n, 'sft-no-selector', which) for n in names if n != 'sft-no-selector'},
        vs_default={n: diff(n, 'joint-transformer-softce', which) for n in names if n != 'joint-transformer-softce'},
        vs_piotr={n: diff(n, 'piotr-data-head', which) for n in names if n != 'piotr-data-head'},
        vs_luna2612={n: diff(n, 'luna-2612-head', which) for n in names if n != 'luna-2612-head'},
        # DeepShop drifts by date: reference reruns ran 10-29, arch reruns 10-30/31, judge-label heads 10-31
        om2w_wv={ref: {n: diff(n, ref, which, BENCH[:2]) for n in names if n != ref}
                  for ref in ('sft-no-selector', 'joint-transformer-softce', 'piotr-data-head', 'luna-2612-head')},
        om2w_wv_mean={n: round(float(np.mean([summary(n, which)[b]['successes'] / summary(n, which)[b]['episodes'] for b in BENCH[:2]])) * 100, 2) for n in names},
        deepshop={ref: {n: diff(n, ref, which, BENCH[2:]) for n in names if n != ref}
                  for ref in ('joint-transformer-softce', 'luna-2612-head')})
out['runs1_only'] = dict(rates={n: summary(n, ['run1']) for n in C},
    vs_default={n: diff(n, 'joint-transformer-softce', ['run1']) for n in C if n != 'joint-transformer-softce'})
# judge-label heads against same-window references is not possible (one run, 10-31); also report vs each reference's
# individual runs to show run-to-run spread
out['ref_run_spread'] = {n: {r: summary_r for r, summary_r in ((r, (lambda rr: round(float(np.mean(
    [sum(data[n, rr, b][0].values()) / len(tasks[b]) for b in BENCH])) * 100, 2))(r)) for r in RUNS)}
    for n in ('joint-transformer-softce', 'piotr-data-head', 'luna-2612-head', 'sft-no-selector')}
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'k3_pooled.json'), 'w'), indent=1)
print(json.dumps(out, indent=1))
