#!/usr/bin/env python3
"""Teacher-data scaling of the joint head on Online-Mind2Web (gpt-5.6-terra judge) plus the o4-mini -> terra re-judge.

Live: data-scale-20261023 runs (judged by terra at collection). References: earlier runs re-judged by replaying each
committed judge request with gpt-5.6-terra (scripts/rejudge_crossbench_om2w.py). Overall = invalid counted as failure;
paired bootstrap over the 300 tasks (10,000 resamples, seed 0). Offline: fixed-five.json / scores.json of each head.
WebVoyager / DeepShop (GPT-4o judge, unchanged) read directly; "mean" weights each benchmark 1/3 and is bootstrapped with
tasks resampled within each benchmark; "OM2W+WV" weights those two 1/2 each. Writes selection-data-scale-20261023.json next to this script.
"""
import json, glob, random, sys
from pathlib import Path
sys.path.insert(0, '/gpfs/projects/krishna/zixianma/OpenWebRL/scripts')
from rejudge_crossbench_om2w import committed_requests
EV = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations')
RJ = EV / 'rejudge-terra-20261023/out'
def live(slug):
    out = {}
    for f in (EV / 'data-scale-20261023/online_mind2web' / slug / 'online_mind2web/collection/records').glob('*.json'):
        r = json.loads(f.read_text()); out[r['task_id'] if 'task_id' in r else r['key']] = r['reward'] if r['valid'] else None
    return out
def ref(label, root):
    out = {}
    for key, (rec, req) in committed_requests(Path(root)).items():
        tid = rec.get('task_id', key)
        if not rec['valid']: out[tid] = None
        elif req is None: out[tid] = 0
        else: out[tid] = json.loads((RJ / label / 'verdicts' / f'{key}.json').read_text())['verdict']
    return out
runs = dict(l.strip().split('=', 1) for l in open(EV / 'rejudge-terra-20261023/runs.txt') if '=' in l)
R = {
 'SFT 4B': ref('crossbench-full-t10-20261018:sft1', runs['crossbench-full-t10-20261018:sft1']),
 'Piotr ARM': ref('crossbench-full-t10-20261018:piotr5', runs['crossbench-full-t10-20261018:piotr5']),
 'Piotr-data head (listwise, old)': ref('crossbench-full-t10-20261018:head5-755d4cd49137', runs['crossbench-full-t10-20261018:head5-755d4cd49137']),
 'Ours outcome': ref('sft-actions-full-20261021:head5-7586978ccf07', runs['sft-actions-full-20261021:head5-7586978ccf07']),
 'Ours K3': ref('sft-actions-full-20261021:head5-11d5eff8ffe1', runs['sft-actions-full-20261021:head5-11d5eff8ffe1']),
 'Ours K1': ref('sft-actions-full-20261021:head5-fd5bf58b5fa6', runs['sft-actions-full-20261021:head5-fd5bf58b5fa6']),
 'Ours pre': ref('sft-actions-full-20261021:head5-2af75856eeed', runs['sft-actions-full-20261021:head5-2af75856eeed']),
 'Piotr 250': live('head5-280d633b0ca8'), 'Piotr 500': live('head5-605054a42c08'),
 'Piotr 1000': live('head5-18b1c8afcbbb'), 'Piotr all 2587': live('head5-fc589b57c9e0'),
 'Piotr 1380 one-panel': live('head5-878e69f9264f'),
}
keys = set(R['SFT 4B'])
for k, v in R.items(): assert set(v) == keys, (k, len(set(v) ^ keys))
keys = sorted(keys)
def ov(v, ks): return [1 if v[k] == 1 else 0 for k in ks]
def boot(a, b, n=10000):
    rng = random.Random(0); d = [x - y for x, y in zip(a, b)]; N = len(d)
    s = sorted(sum(d[rng.randrange(N)] for _ in range(N)) / N for _ in range(n))
    return round(100*sum(d)/N, 1), round(100*s[int(.025*n)], 1), round(100*s[int(.975*n)], 1)
OUT = dict(online_mind2web={}, rejudge={}, offline={})
print(f"{'name':34s} valid succ overall valid_only | vs SFT [CI] | vs Piotr ARM [CI]")
for k, v in R.items():
    val = sum(x is not None for x in v.values()); suc = sum(x == 1 for x in v.values())
    a = ov(v, keys)
    OUT['online_mind2web'][k] = dict(valid=val, success=suc, overall=round(100*suc/300, 2), valid_only=round(100*suc/val, 2),
        minus_sft4b=boot(a, ov(R['SFT 4B'], keys)), minus_piotr=boot(a, ov(R['Piotr ARM'], keys)))
    print(f"{k:34s} {val:4d} {suc:4d} {100*suc/300:6.2f} {100*suc/val:6.2f} | {boot(a, ov(R['SFT 4B'], keys))} | {boot(a, ov(R['Piotr ARM'], keys))}")
for f in sorted(RJ.glob('*/summary.json')):
    s = json.loads(f.read_text()); OUT['rejudge'][s['label']] = s
H = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/selection-head-20261008/data-scale-20261023/heads')
for n in ('n250', 'n500', 'n1000', 'nall', 'n1380-1panel'):
    s = json.loads((H / n / 'seed0/scores.json').read_text()); f = json.loads((H / n / 'seed0/fixed-five.json').read_text())
    OUT['offline'][n] = dict(train=s['extra'], recipe=dict(batch_size=s['batch_size'], epochs=s['epochs'], last_epoch=s.get('last_epoch', False)),
                            validation=s['extra_validation'], fixed_five=f, distinct_head_minus_uniform=s['head_minus_uniform'])
ledger = json.loads((RJ / 'ledger.json').read_text()); OUT['rejudge_cost'] = ledger
(Path(__file__).resolve().parent / 'selection-data-scale-20261023.json').write_text(json.dumps(OUT, indent=1))

# ---- WebVoyager + DeepShop and the three-benchmark mean ----
def records(root):
    out = {}
    for f in (Path(root) / 'collection/records').glob('*.json'):
        r = json.loads(f.read_text()); out[r.get('task_id', r['key'])] = r['reward'] if r['valid'] else None
    return out
SLUG = {'SFT 4B': 'sft1', 'Piotr ARM': 'piotr5', 'Piotr-data head (listwise, old)': 'head5-755d4cd49137',
        'Ours outcome': 'head5-7586978ccf07', 'Ours K3': 'head5-11d5eff8ffe1', 'Ours K1': 'head5-fd5bf58b5fa6', 'Ours pre': 'head5-2af75856eeed',
        'Piotr 250': 'head5-280d633b0ca8', 'Piotr 500': 'head5-605054a42c08', 'Piotr 1000': 'head5-18b1c8afcbbb',
        'Piotr all 2587': 'head5-fc589b57c9e0', 'Piotr 1380 one-panel': 'head5-878e69f9264f'}
def wvds_root(name, b):
    s = SLUG[name]
    base = {'SFT 4B': 'crossbench-full-t10-20261018', 'Piotr ARM': 'crossbench-full-t10-20261018',
            'Piotr-data head (listwise, old)': 'crossbench-full-t10-20261018'}.get(name, 'sft-actions-full-20261021' if name.startswith('Ours') else 'data-scale-20261023')
    return EV / base / b / s / b
ALL = {n: {'online_mind2web': R[n]} for n in R}
for n in R:
    for b in ('webvoyager', 'deepshop'):
        ALL[n][b] = records(wvds_root(n, b))
BK = {b: sorted(ALL['SFT 4B'][b]) for b in ('online_mind2web', 'webvoyager', 'deepshop')}
for n in ALL:
    for b in BK: assert set(ALL[n][b]) == set(BK[b]), (n, b)
def vec(n, b): return [1 if ALL[n][b][k] == 1 else 0 for k in BK[b]]
def boot_mean(a, b_, n=10000, benches=None):
    rng = random.Random(0); diffs = {bb: [x - y for x, y in zip(vec(a, bb), vec(b_, bb))] for bb in benches or BK}
    point = sum(sum(d) / len(d) for d in diffs.values()) / len(diffs)
    s = sorted(sum(sum(d[rng.randrange(len(d))] for _ in range(len(d))) / len(d) for d in diffs.values()) / len(diffs) for _ in range(n))
    return round(100 * point, 1), round(100 * s[int(.025 * n)], 1), round(100 * s[int(.975 * n)], 1)
OUT['three_benchmarks'] = {}
print(f"\n{'name':34s}  OM2W   WV     DS   | mean ov  mean vo | -SFT [CI] | -Piotr [CI]")
for n in ALL:
    row = {}
    for b in BK:
        v = ALL[n][b]; val = sum(x is not None for x in v.values()); suc = sum(x == 1 for x in v.values())
        row[b] = dict(tasks=len(v), valid=val, success=suc, overall=round(100 * suc / len(v), 2), valid_only=round(100 * suc / val, 2))
    row['mean_overall'] = round(sum(row[b]['overall'] for b in BK) / 3, 2); row['mean_valid_only'] = round(sum(row[b]['valid_only'] for b in BK) / 3, 2)
    row['minus_sft4b'] = boot_mean(n, 'SFT 4B'); row['minus_piotr'] = boot_mean(n, 'Piotr ARM')
    # Online-Mind2Web + WebVoyager only (each 1/2): leaves out DeepShop, which drifted between run dates
    row['minus_sft4b_om2w_wv'] = boot_mean(n, 'SFT 4B', benches=('online_mind2web', 'webvoyager'))
    OUT['three_benchmarks'][n] = row
    print(f"{n:34s} {row['online_mind2web']['overall']:5.1f} {row['webvoyager']['overall']:5.1f} {row['deepshop']['overall']:5.1f} | {row['mean_overall']:6.2f} {row['mean_valid_only']:6.2f} | {row['minus_sft4b']} | {row['minus_piotr']} | OM2W+WV -SFT {row['minus_sft4b_om2w_wv']}")
PAIRS = [('Piotr all 2587', 'Piotr 250'), ('Piotr 1000', 'Piotr 250'), ('Piotr all 2587', 'Piotr-data head (listwise, old)'),
         ('Piotr 1380 one-panel', 'Ours K3'), ('Piotr 1380 one-panel', 'Ours outcome'), ('Piotr 1380 one-panel', 'Piotr all 2587')]
OUT['three_benchmark_pairs'] = {f'{a} - {b}': boot_mean(a, b) for a, b in PAIRS}
for k, v in OUT['three_benchmark_pairs'].items(): print(k, v)
(Path(__file__).resolve().parent / 'selection-data-scale-20261023.json').write_text(json.dumps(OUT, indent=1))
