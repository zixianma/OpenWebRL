#!/usr/bin/env python3
"""Live results of the 10-27/28 selection-head experiments, organized by question:
(1) Piotr's data vs our branching data, (2) decoder LoRA selector vs 22.6 M head on the same data, (3) data scale
1,380 vs 2,612 of our states, (4) training device (CPU vs GPU) and rerun noise of identical weights.
SFT-4B actor, N=5, T1.0/p0.9, 30 steps, all tasks of Online-Mind2Web (300), WebVoyager (595), DeepShop (150).
OM2W judge gpt-5.6-terra (10-18 and 10-21 runs re-judged, rejudge-terra-20261023); WebVoyager/DeepShop GPT-4o.
Overall counts invalid as failure; mean = unweighted mean of the three benchmark rates. A condition with several runs
(copies or reruns) is pooled per task (mean success over runs). Paired 95% bootstrap over tasks (10,000, seed 20261028),
benchmarks resampled independently for the mean."""
import collections, json, sys
from pathlib import Path
import numpy as np

R = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations')
RJ = R/'rejudge-terra-20261023'/'out'
BENCH = ('online_mind2web', 'webvoyager', 'deepshop')
CB, OP = 'crossbench-full-t10-20261018', 'onepanel-k3-scale-20261027'


def same(rnd, slug, rj=False):
    return {b: (rnd, slug, f'{rnd}:{slug}' if rj and b == 'online_mind2web' else None) for b in BENCH}


RUNS = {  # run name: {bench: (round, slug, rejudge label)}
    'piotr_arm_1018': same(CB, 'piotr5', True),
    'piotr_head_orig_1018': same(CB, 'head5-755d4cd49137', True),
    'piotr_head_orig_1025': same('piotr-head-sameday-20261025', 'head5-755d4cd49137'),
    'piotr_head_gpu_1028': same('device-repro-20261028', 'head5-755d4cd49137'),      # = original weights
    'piotr_head_cpu_1028': same('device-repro-20261028', 'head5-1c9a7d38a567'),
    'h2612_cpu_1027': same(OP, 'head5-5ca25c38f7ad'),
    'h2612_cpu_1028': same('device-2612-20261028', 'head5-5ca25c38f7ad'),             # = 10-27 weights
    'h2612_gpu_1028': same('device-2612-20261028', 'head5-d15fea171bb7'),
    'h1380': {'online_mind2web': ('sft-actions-full-20261021', 'head5-11d5eff8ffe1', 'sft-actions-full-20261021:head5-11d5eff8ffe1'),
              'webvoyager': ('sft-actions-full-20261021', 'head5-11d5eff8ffe1', None),
              'deepshop': (OP, 'head5-11d5eff8ffe1', None)},
    'h1380_ds_1021': {'deepshop': ('sft-actions-full-20261021', 'head5-11d5eff8ffe1', None)},
    'lora2612_1028': same('lora-selectors-20261028', 'arm5-03edd8c36a91'),
    'lora1380_1028': same('lora-selectors-20261028', 'arm5-8dc8c40b7cfb'),
    'sft4b': {'online_mind2web': (CB, 'sft1', f'{CB}:sft1'), 'webvoyager': (CB, 'sft1', None), 'deepshop': (OP, 'sft1', None)},
}
rng = np.random.default_rng(20261028)
tasks = {b: list(json.loads((R/CB/b/'sft1'/b/'plan.json').read_text())['keys']) for b in BENCH}
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
        if rj and valid:
            vp = RJ/rj/'verdicts'/f"{r['key']}.json"   # no file: never reached the judge, reward stays 0 (rejudge rule)
            if vp.exists():
                v = json.loads(vp.read_text()); jm = v['returned_model']
                valid = v['verdict'] is not None; succ = valid and float(v['verdict']) >= 1
            else:
                jm = 'not judged (reward 0)'
        if valid:
            judges[(rnd, slug, b, jm)] += 1
        out[t] = (valid, succ)
    return out


data = {(n, b): load(*s[b][:2], b, s[b][2]) for n, s in RUNS.items() for b in s}
rows = []
for n, s in RUNS.items():
    for b in s:
        v = data[n, b]; k = len(v); val = sum(x[0] for x in v.values()); sc = sum(x[1] for x in v.values())
        rows.append(dict(run=n, benchmark=b, date_dir=s[b][0], tasks=k, valid=val, successes=sc,
                         overall=round(100 * sc / k, 1), valid_only=round(100 * sc / val, 1)))
    if len(s) == 3:
        rows.append(dict(run=n, benchmark='mean', overall=round(np.mean([r['overall'] for r in rows if r['run'] == n and r['benchmark'] in BENCH]), 1)))


def vec(runs, b):
    return np.array([np.mean([data[r, b][t][1] for r in runs]) for t in tasks[b]], float)


def boot(per_bench):
    est = np.mean([x.mean() for x in per_bench])
    bs = np.mean([x[rng.integers(0, len(x), (10000, len(x)))].mean(1) for x in per_bench], 0)
    return [round(100 * est, 1), round(100 * np.percentile(bs, 2.5), 1), round(100 * np.percentile(bs, 97.5), 1)]


PIOTR_HEAD = ('piotr_head_cpu_1028', 'piotr_head_gpu_1028'); H2612 = ('h2612_cpu_1028', 'h2612_gpu_1028')
COMPARE = {
    '1_recipe': [('Piotr-data head (2 copies) - 2,612 head (2 copies), same day', PIOTR_HEAD, H2612),
                 ('Piotr-data head (2 copies) - 1,380 head', PIOTR_HEAD, ('h1380',)),
                 ('Piotr SelectionARM (10-18) - LoRA 2,612 (10-28)', ('piotr_arm_1018',), ('lora2612_1028',)),
                 ('Piotr SelectionARM (10-18) - LoRA 1,380 (10-28)', ('piotr_arm_1018',), ('lora1380_1028',))],
    '2_lora_vs_head': [('LoRA 2,612 - 2,612 head (2 copies), same day', ('lora2612_1028',), H2612),
                       ('LoRA 1,380 - 1,380 head', ('lora1380_1028',), ('h1380',)),
                       ('Piotr SelectionARM - Piotr-data head, same day 10-18', ('piotr_arm_1018',), ('piotr_head_orig_1018',))],
    '3_scale': [('2,612 head (3 runs) - 1,380 head', ('h2612_cpu_1027',) + H2612, ('h1380',)),
                ('LoRA 2,612 - LoRA 1,380, same day', ('lora2612_1028',), ('lora1380_1028',))],
    '4_device': [('Piotr-data head: GPU copy - CPU copy, same day', ('piotr_head_gpu_1028',), ('piotr_head_cpu_1028',)),
                 ('2,612 head: GPU copy - CPU copy, same day', ('h2612_gpu_1028',), ('h2612_cpu_1028',)),
                 ('Same weights, Piotr-data head: 10-28 - 10-25', ('piotr_head_gpu_1028',), ('piotr_head_orig_1025',)),
                 ('Same weights, Piotr-data head: 10-28 - 10-18', ('piotr_head_gpu_1028',), ('piotr_head_orig_1018',)),
                 ('Same weights, 2,612 head: 10-28 - 10-27', ('h2612_cpu_1028',), ('h2612_cpu_1027',))],
}
paired = {}
for q, items in COMPARE.items():
    paired[q] = []
    for name, a, c in items:
        rec = dict(name=name, a=a, b=c)
        for b in BENCH:
            rec[b] = boot([vec(a, b) - vec(c, b)])
        rec['mean'] = boot([vec(a, b) - vec(c, b) for b in BENCH])
        paired[q].append(rec); print(name, {k: rec[k] for k in (*BENCH, 'mean')})
pooled = {}
for name, runs in {'piotr_head_2copies_1028': PIOTR_HEAD, 'h2612_2copies_1028': H2612,
                   'h2612_3runs': ('h2612_cpu_1027',) + H2612}.items():
    r = {b: round(100 * vec(runs, b).mean(), 1) for b in BENCH}; r['mean'] = round(np.mean(list(r.values())), 1)
    pooled[name] = r
for r in rows:
    print(f"{r['run']:<22} {r['benchmark']:<16} {r.get('successes','')!s:>4} {r.get('valid','')!s:>4} {r['overall']:>6} {r.get('valid_only','')}")
print(pooled)
out = Path(sys.argv[1])
out.write_text(json.dumps(dict(note=__doc__, runs={n: {b: list(v) for b, v in s.items()} for n, s in RUNS.items()},
                               judges=[[*k, v] for k, v in sorted(judges.items())], rows=rows, pooled=pooled,
                               paired=paired), indent=1))
