#!/usr/bin/env python3
"""K=3 selection-head ablations (2026-10-29): same-day cohort, success = valid and reward == 1 over scheduled tasks.
Mean = average of OM2W (gpt-5.6-terra), WebVoyager, DeepShop (GPT-4o) rates. Paired bootstrap over tasks within each
benchmark (10,000 draws, seed 0) for differences of the 3-benchmark mean. Output k3_ablation.json."""
import json, glob, os
import numpy as np
E = '/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations/sameday-20261029'
BENCH = ('online_mind2web', 'webvoyager', 'deepshop')
C = {'joint-transformer-softce': 'head5-11d5eff8ffe1', 'joint-transformer-bce': 'head5-4a92d9e12332',
     'independent-transformer-softce': 'head5-040311428ed1', 'independent-transformer-bce': 'head5-6e2111c2c1e5',
     'joint-mlp-softce': 'head5-440292874f27', 'joint-mlp-bce': 'head5-fbac2672a438',
     'independent-mlp-softce': 'head5-aba71a4d9890', 'independent-mlp-bce': 'head5-15ff2a7e9bce',
     'piotr-data-head': 'head5-755d4cd49137', 'sft-no-selector': 'sft1'}
win, judges = {}, {}
for name, slug in C.items():
    for b in BENCH:
        rows = {}
        for f in glob.glob(f'{E}/{b}/{slug}/{b}/collection/records/*.json'):
            r = json.load(open(f)); rows[r['task_id']] = int(bool(r['valid'] and r['reward'] == 1))
            judges.setdefault((name, b), set()).add(r['judge_model'])
        win[name, b] = rows
tasks = {b: sorted(win['sft-no-selector', b]) for b in BENCH}
for (name, b), rows in win.items():
    assert sorted(rows) == tasks[b], (name, b, len(rows), len(tasks[b]))
M = {name: {b: np.array([win[name, b][t] for t in tasks[b]]) for b in BENCH} for name in C}
rng = np.random.default_rng(0)
idx = {b: rng.integers(0, len(tasks[b]), (10000, len(tasks[b]))) for b in BENCH}
def boot(x):  # x: {b: per-task values}
    return np.mean([x[b][idx[b]].mean(1) for b in BENCH], 0)
def ci(a, b):
    d = {k: M[a][k] - M[b][k] for k in BENCH}; s = boot(d) * 100
    return [round(float(np.mean([d[k].mean() for k in BENCH]) * 100), 2), round(float(np.percentile(s, 2.5)), 2), round(float(np.percentile(s, 97.5)), 2)]
out = dict(judges={f'{k[0]}|{k[1]}': sorted(v) for k, v in judges.items()},
           rates={n: {b: [int(M[n][b].sum()), len(tasks[b])] for b in BENCH} | {'mean': round(float(np.mean([M[n][b].mean() for b in BENCH]) * 100), 2)} for n in C},
           vs_sft={n: ci(n, 'sft-no-selector') for n in C if n != 'sft-no-selector'},
           vs_joint_tf_softce={n: ci(n, 'joint-transformer-softce') for n in C if n != 'joint-transformer-softce'},
           contrasts={})
avg = lambda names: {b: np.mean([M[n][b] for n in names], 0) for b in BENCH}
def con(label, A, B):
    a, b_ = avg(A), avg(B); d = {k: a[k] - b_[k] for k in BENCH}; s = boot(d) * 100
    out['contrasts'][label] = [round(float(np.mean([d[k].mean() for k in BENCH]) * 100), 2), round(float(np.percentile(s, 2.5)), 2), round(float(np.percentile(s, 97.5)), 2)]
tf = ['joint-transformer-softce', 'joint-transformer-bce', 'independent-transformer-softce', 'independent-transformer-bce']
ml = ['joint-mlp-softce', 'joint-mlp-bce', 'independent-mlp-softce', 'independent-mlp-bce']
allc = tf + ml
con('joint - independent (all 8)', [n for n in allc if n.startswith('joint')], [n for n in allc if n.startswith('indep')])
con('softce - bce (all 8)', [n for n in allc if n.endswith('softce')], [n for n in allc if n.endswith('bce')])
con('transformer - mlp (all 8)', tf, ml)
con('joint - independent (transformer)', tf[:2], tf[2:])
con('softce - bce (transformer)', tf[0::2], tf[1::2])
json.dump(out, open(os.path.join(os.path.dirname(__file__), 'k3_ablation.json'), 'w'), indent=1)
print(json.dumps(out, indent=1))
