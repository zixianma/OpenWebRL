#!/usr/bin/env python3
"""Same-day cohort 2026-10-29/30; every selector run 3 times in that window (roots ., rerun2/, rerun3/; pooled per task); LoRA variants o1/p22k once: every core selector of ARM_SELECTION_HEAD#selection-head-recipe-model-scale-device-20261028
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


RERUNS = {'piotr_head', 'h1380', 'piotr_arm', 'h2612', 'h22k', 'lora2612', 'lora1380', 'sft4b'}  # 3 runs on 10-29 (roots rerun2/, rerun3/); incomplete runs skipped
TODAY.update({'lora_o1': 'arm5-28f57864be1f', 'lora_p22k': 'arm5-67a86473ef0d'})
today, today_runs = {}, {}
for n, slug in TODAY.items():
    runs = []
    for rnd in [SD] + ([f'{SD}/rerun2', f'{SD}/rerun3'] if n in RERUNS else []):
        d = {b: load(rnd, slug, b) for b in BENCH}
        if all(d.values()):
            runs.append(d)
        else:
            print('incomplete:', rnd, n, [b for b in BENCH if not d[b]])
    if runs:
        today[n] = runs[0]; today_runs[n] = runs
earlier = {n: [] for n in today}
earlier.update({n: [{b: load(*s[b][:2], b, s[b][2]) for b in BENCH} for s in runs] for n, runs in EARLIER.items() if n in today})


def vec(runs, b):  # runs: list of {bench: {task: (valid, succ)}}; per-task mean success
    return np.array([np.mean([r[b][t][1] for r in runs]) for t in tasks[b]], float)


def boot(per_bench):
    est = np.mean([x.mean() for x in per_bench])
    bs = np.mean([x[rng.integers(0, len(x), (10000, len(x)))].mean(1) for x in per_bench], 0)
    return [round(100 * est, 1), round(100 * np.percentile(bs, 2.5), 1), round(100 * np.percentile(bs, 97.5), 1)]


def diff(a, c):
    rec = {b: boot([vec(a, b) - vec(c, b)]) for b in BENCH}; rec['mean'] = boot([vec(a, b) - vec(c, b) for b in BENCH]); return rec


rates = {}
for n, runs in today_runs.items():
    r = {}
    for b in BENCH:
        sc = sum(x[1] for d in runs for x in d[b].values()); va = sum(x[0] for d in runs for x in d[b].values()); k = sum(len(d[b]) for d in runs)
        r[b] = dict(runs=len(runs), successes=sc, valid=va, episodes=k, overall=round(100 * sc / k, 1), valid_only=round(100 * sc / va, 1))
    r['mean_overall'] = round(np.mean([r[b]['overall'] for b in BENCH]), 1); r['mean_valid_only'] = round(np.mean([r[b]['valid_only'] for b in BENCH]), 1)
    r['per_run_mean_overall'] = [round(np.mean([100 * sum(x[1] for x in d[b].values()) / len(d[b]) for b in BENCH]), 1) for d in runs]
    rates[n] = r
    print(f"{n:<11} runs={len(runs)}", *(f"{r[b]['valid_only']:>5}" for b in BENCH), r['mean_overall'], r['mean_valid_only'], r['per_run_mean_overall'])
COMPARE = {
    '1_recipe': [('piotr_head', 'h2612'), ('piotr_head', 'h1380'), ('piotr_head', 'h22k'), ('piotr_arm', 'lora2612'), ('piotr_arm', 'lora1380')],
    '2_lora_vs_head': [('lora2612', 'h2612'), ('lora1380', 'h1380'), ('piotr_arm', 'piotr_head')],
    '3_scale': [('h2612', 'h1380'), ('lora2612', 'lora1380')],
    '4_lookahead_panels': [('h22k', 'h2612'), ('h22k', 'h1380')],
    'vs_sft': [(n, 'sft4b') for n in TODAY if n != 'sft4b'],
    'lora_variants': [('lora_o1', 'lora2612'), ('lora_p22k', 'lora2612'), ('lora_o1', 'piotr_arm'), ('lora_p22k', 'piotr_arm')],
}
paired = {q: [dict(a=a, b=c, runs=[len(today_runs[a]), len(today_runs[c])], today=diff(today_runs[a], today_runs[c]),
                   pooled_all_runs=diff(today_runs[a] + earlier[a], today_runs[c] + earlier[c]))
              for a, c in items if a in today and c in today] for q, items in COMPARE.items()}
for q, items in paired.items():
    for x in items:
        print(q, x['a'], '-', x['b'], 'today', x['today']['mean'], 'DS', x['today']['deepshop'], '| pooled', x['pooled_all_runs']['mean'])
drift = {n: [dict(earlier_run=EARLIER[n][i]['online_mind2web'][0], today_minus_earlier=diff([today[n]], [e]))
             for i, e in enumerate(earlier[n])] for n in today if n in EARLIER}
drift['within_today'] = {n: [dict(run=k + 2, minus_run1=diff([today_runs[n][k + 1]], [today_runs[n][0]])) for k in range(len(today_runs[n]) - 1)]
                         for n in RERUNS if n in today_runs}
for n, xs in drift.items():
    if n == 'within_today':
        print('within today', {k: [x['minus_run1']['mean'] for x in v] for k, v in xs.items()}); continue
    for x in xs:
        print('drift', n, x['earlier_run'], x['today_minus_earlier']['mean'], 'DS', x['today_minus_earlier']['deepshop'])
Path(sys.argv[1]).write_text(json.dumps(dict(note=__doc__, today_slugs=TODAY, earlier_runs=EARLIER, rates=rates, paired=paired,
                                             drift=drift), indent=1))
