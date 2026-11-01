#!/usr/bin/env python3
"""RL checkpoint actor, 4,096 tokens: 3 runs each (run 1 = rl/ on 10-30 for RL alone / SelectionARM and 10-31 for the heads;
runs 2-3 = rerun2/, rerun3/ on 10-31) of RL alone (sft1), SelectionARM (piotr5), Recipe 1 head and Recipe 2 2.6K head scoring the
RL actor's own hidden states (head5, SFT-trained, no retraining). Pooled = per-task mean success over the 3 runs (invalid =
failure); valid-only per benchmark = pooled successes / pooled valid episodes; mean = unweighted mean of the three benchmarks.
Paired 95% bootstrap over tasks on pooled per-task success (10,000, fresh seed 20261029 per pair, benchmarks resampled
independently). Incomplete runs are reported and skipped."""
import json, os
import numpy as np
E = '/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations/rlactor-4096-20261030'; B = ('online_mind2web', 'webvoyager', 'deepshop')
COND = {'rl_alone': 'sft1', 'rl_arm': 'piotr5', 'rl_hpiotr': 'head5-755d4cd49137', 'rl_h2612': 'head5-5ca25c38f7ad', 'rl_lora2612': 'arm5-03edd8c36a91'}
ROOTS = ['rl', 'rerun2/rl', 'rerun3/rl']
PAIRS = [('rl_arm', 'rl_alone'), ('rl_h2612', 'rl_alone'), ('rl_hpiotr', 'rl_alone'), ('rl_h2612', 'rl_arm'), ('rl_hpiotr', 'rl_arm'),
         ('rl_h2612', 'rl_hpiotr'), ('rl_lora2612', 'rl_alone'), ('rl_lora2612', 'rl_arm'), ('rl_lora2612', 'rl_h2612')]
def load(root, slug, b):
    d = f'{E}/{root}/{b}/{slug}/{b}'; keys = json.load(open(d + '/plan.json'))['keys']; o = {}
    for t, k in keys.items():
        f = f'{d}/collection/records/{k}.json'
        if not os.path.exists(f): return None
        r = json.load(open(f)); v = str(r['valid']) == 'True'; o[t] = (v, v and float(r.get('reward') or 0) >= 1)
    return o
runs = {}
for n, slug in COND.items():
    runs[n] = []
    for root in ROOTS:
        d = {b: load(root, slug, b) for b in B}
        if all(d.values()): runs[n].append(d)
        else: print('incomplete:', root, n, [b for b in B if d[b] is None])
tasks = {b: sorted(runs['rl_alone'][0][b]) for b in B}
out = {'runs_used': {n: len(r) for n, r in runs.items()}, 'rates': {}, 'per_run_mean_overall': {}, 'paired_pooled_pp': {}}
for n, rs in runs.items():
    o = {}
    for b in B:
        succ = sum(r[b][t][1] for r in rs for t in tasks[b]); val = sum(r[b][t][0] for r in rs for t in tasks[b]); tot = len(rs) * len(tasks[b])
        o[b] = dict(episodes=tot, valid=val, succ=succ, overall=round(100 * succ / tot, 1), valid_only=round(100 * succ / val, 1))
    o['mean_overall'] = round(np.mean([o[b]['overall'] for b in B]), 1); o['mean_valid_only'] = round(np.mean([o[b]['valid_only'] for b in B]), 1)
    out['rates'][n] = o
    out['per_run_mean_overall'][n] = [round(100 * np.mean([np.mean([r[b][t][1] for t in tasks[b]]) for b in B]), 1) for r in rs]
def vec(n, b): return np.array([np.mean([r[b][t][1] for r in runs[n]]) for t in tasks[b]], float)
for x_, y_ in PAIRS:
    rng = np.random.default_rng(20261029); dif = []; per = {}
    for b in B:
        x = vec(x_, b) - vec(y_, b); dif.append(x)
        bs = x[rng.integers(0, len(x), (10000, len(x)))].mean(1); per[b] = [round(100 * x.mean(), 1), round(100 * np.percentile(bs, 2.5), 1), round(100 * np.percentile(bs, 97.5), 1)]
    bs = np.mean([x[rng.integers(0, len(x), (10000, len(x)))].mean(1) for x in dif], 0)
    per['mean'] = [round(100 * np.mean([x.mean() for x in dif]), 1), round(100 * np.percentile(bs, 2.5), 1), round(100 * np.percentile(bs, 97.5), 1)]
    out['paired_pooled_pp'][f'{x_} - {y_}'] = per
json.dump(out, open(E + '/analysis/three_run.json', 'w'), indent=1)
for n, o in out['rates'].items():
    print(n, out['runs_used'][n], [o[b]['valid_only'] for b in B], o['mean_overall'], o['mean_valid_only'], 'per-run', out['per_run_mean_overall'][n])
for k, v in out['paired_pooled_pp'].items(): print(k, json.dumps(v))
