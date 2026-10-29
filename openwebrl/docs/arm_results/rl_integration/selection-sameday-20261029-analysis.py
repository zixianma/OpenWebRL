#!/usr/bin/env python3
"""Same-day cohort 2026-10-29 (option B): every core selector of ARM_SELECTION_HEAD#selection-head-recipe-model-scale-device-20261028
rerun in one window, plus the 22K Piotr-style-panel head when complete. SFT-4B actor, N=5, T1.0/p0.9, 30 steps, all tasks of
Online-Mind2Web (300, gpt-5.6-terra), WebVoyager (595, GPT-4o), DeepShop (150, GPT-4o). Overall counts invalid as failure; mean =
unweighted mean of the three rates. Paired 95% bootstrap over tasks (10,000, seed 20261029), benchmarks resampled independently.
Also: the same weights today vs their earlier runs (rerun drift), and earlier+today pooled per task."""
import json, sys
from pathlib import Path
import numpy as np

R = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations'); RJ = R/'rejudge-terra-20261023'/'out'
BENCH = ('online_mind2web', 'webvoyager', 'deepshop'); SD = 'sameday-20261029'
CB, OP = 'crossbench-full-t10-20261018', 'onepanel-k3-scale-20261027'
TODAY = {'piotr_head': 'head5-755d4cd49137', 'h2612': 'head5-5ca25c38f7ad', 'h1380': 'head5-11d5eff8ffe1',
         'h22k': 'head5-cafea4f265cf', 'lora2612': 'arm5-03edd8c36a91', 'lora1380': 'arm5-8dc8c40b7cfb',
         'piotr_arm': 'piotr5', 'sft4b': 'sft1'}
def same(rnd, slug, rj=False):
    return {b: (rnd, slug, f'{rnd}:{slug}' if rj and b == 'online_mind2web' else None) for b in BENCH}
EARLIER = {  # name: list of earlier runs of the same weights
    'piotr_head': [same(CB, TODAY['piotr_head'], True), same('piotr-head-sameday-20261025', TODAY['piotr_head']),
                   same('device-repro-20261028', TODAY['piotr_head'])],
    'h2612': [same(OP, TODAY['h2612']), same('device-2612-20261028', TODAY['h2612'])],
    'h1380': [{'online_mind2web': ('sft-actions-full-20261021', TODAY['h1380'], f"sft-actions-full-20261021:{TODAY['h1380']}"),
               'webvoyager': ('sft-actions-full-20261021', TODAY['h1380'], None), 'deepshop': (OP, TODAY['h1380'], None)}],
    'h22k': [same('lunak3-panels-20261027', TODAY['h22k'])],
    'lora2612': [same('lora-selectors-20261028', TODAY['lora2612'])],
    'lora1380': [same('lora-selectors-20261028', TODAY['lora1380'])],
    'piotr_arm': [same(CB, 'piotr5', True)],
    'sft4b': [{'online_mind2web': (CB, 'sft1', f'{CB}:sft1'), 'webvoyager': (CB, 'sft1', None), 'deepshop': (OP, 'sft1', None)}],
}
rng = np.random.default_rng(20261029)
tasks = {b: list(json.loads((R/CB/b/'sft1'/b/'plan.json').read_text())['keys']) for b in BENCH}


def load(rnd, slug, b, rj=None):
    d = R/rnd/b/slug/b
    if not d.exists():
        return None
    keys = json.loads((d/'plan.json').read_text())['keys']; out = {}
    for t in tasks[b]:
        f = d/'collection'/'records'/f'{keys[t]}.json'
        if not f.exists():
            return None                       # incomplete run
        r = json.loads(f.read_text()); valid = str(r.get('valid')) == 'True'; succ = valid and float(r.get('reward') or 0) >= 1
        if rj and valid:
            vp = RJ/rj/'verdicts'/f"{r['key']}.json"
            if vp.exists():
                v = json.loads(vp.read_text()); valid = v['verdict'] is not None; succ = valid and float(v['verdict']) >= 1
            else:
                succ = False
        out[t] = (valid, succ)
    return out


today = {}
for n, slug in TODAY.items():
    d = {b: load(SD, slug, b) for b in BENCH}
    if all(d.values()):
        today[n] = d
    else:
        print('incomplete today:', n, [b for b in BENCH if not d[b]])
earlier = {n: [{b: load(*s[b][:2], b, s[b][2]) for b in BENCH} for s in runs] for n, runs in EARLIER.items() if n in today}


def vec(runs, b):  # runs: list of {bench: {task: (valid, succ)}}; per-task mean success
    return np.array([np.mean([r[b][t][1] for r in runs]) for t in tasks[b]], float)


def boot(per_bench):
    est = np.mean([x.mean() for x in per_bench])
    bs = np.mean([x[rng.integers(0, len(x), (10000, len(x)))].mean(1) for x in per_bench], 0)
    return [round(100 * est, 1), round(100 * np.percentile(bs, 2.5), 1), round(100 * np.percentile(bs, 97.5), 1)]


def diff(a, c):
    rec = {b: boot([vec(a, b) - vec(c, b)]) for b in BENCH}; rec['mean'] = boot([vec(a, b) - vec(c, b) for b in BENCH]); return rec


rates = {}
for n, d in today.items():
    r = {b: dict(successes=sum(x[1] for x in d[b].values()), valid=sum(x[0] for x in d[b].values()), tasks=len(d[b]),
                 overall=round(100 * sum(x[1] for x in d[b].values()) / len(d[b]), 1)) for b in BENCH}
    r['mean'] = round(np.mean([r[b]['overall'] for b in BENCH]), 1); rates[n] = r
    print(f"{n:<11}", *(f"{r[b]['overall']:>5}" for b in BENCH), r['mean'])
COMPARE = {
    '1_recipe': [('piotr_head', 'h2612'), ('piotr_head', 'h1380'), ('piotr_head', 'h22k'), ('piotr_arm', 'lora2612'), ('piotr_arm', 'lora1380')],
    '2_lora_vs_head': [('lora2612', 'h2612'), ('lora1380', 'h1380'), ('piotr_arm', 'piotr_head')],
    '3_scale': [('h2612', 'h1380'), ('lora2612', 'lora1380')],
    '4_lookahead_panels': [('h22k', 'h2612'), ('h22k', 'h1380')],
    'vs_sft': [(n, 'sft4b') for n in TODAY if n != 'sft4b'],
}
paired = {q: [dict(a=a, b=c, today=diff([today[a]], [today[c]]),
                   pooled_all_runs=diff([today[a]] + earlier[a], [today[c]] + earlier[c]))
              for a, c in items if a in today and c in today] for q, items in COMPARE.items()}
for q, items in paired.items():
    for x in items:
        print(q, x['a'], '-', x['b'], 'today', x['today']['mean'], 'DS', x['today']['deepshop'], '| pooled', x['pooled_all_runs']['mean'])
drift = {n: [dict(earlier_run=EARLIER[n][i]['online_mind2web'][0], today_minus_earlier=diff([today[n]], [e]))
             for i, e in enumerate(earlier[n])] for n in today}
for n, xs in drift.items():
    for x in xs:
        print('drift', n, x['earlier_run'], x['today_minus_earlier']['mean'], 'DS', x['today_minus_earlier']['deepshop'])
Path(sys.argv[1]).write_text(json.dumps(dict(note=__doc__, today_slugs=TODAY, earlier_runs=EARLIER, rates=rates, paired=paired,
                                             drift=drift), indent=1))
