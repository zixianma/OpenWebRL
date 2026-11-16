"""Site-exclusion sensitivity analysis of two SFT evaluation windows.

Exclude all 46 Wolfram Alpha tasks by source-site metadata in EVERY condition,
not by outcomes. Use the same remaining task IDs across conditions and windows.
Keep invalid episodes as failures. Average three runs per task, then benchmarks
equally; bootstrap tasks independently within benchmarks (10,000, seed 20261029).
Absolute-rate intervals and paired within-window contrasts are recomputed.
The across-window SFT contrast is descriptive, not a controlled time effect.
Reads saved records only; exports aggregate statistics without task payloads.
"""
from pathlib import Path
from urllib.parse import urlparse
import json
import numpy as np
HERE = Path(__file__).resolve().parent
NEWROOT = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/recipe3-ablations-20261104/eval-pre')
OLDROOT = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations/head-k5-20261101')
NEW = json.loads((HERE / 'selection-head-pre-vs-lookahead-20261115.json').read_text())
OLD = json.loads((HERE / 'selection-label-live-3run-20261103.json').read_text())
BENCH = ('online_mind2web', 'webvoyager', 'deepshop')
source = NEWROOT / 'run1/webvoyager/sft1/webvoyager/tasks-source.jsonl'
excluded = {r['id'] for line in source.read_text().splitlines() if (r := json.loads(line)) and urlparse(r['web']).hostname in ('www.wolframalpha.com', 'wolframalpha.com')}
assert len(excluded) == 46
rng = np.random.default_rng(20261029)
result = dict(method=__doc__, excluded_site='Wolfram Alpha', excluded_tasks=46,
              tasks_by_benchmark={}, conditions={}, earlier_conditions={}, contrasts={}, window_sft_change={})
values, samples = {}, {}
for bench in BENCH:
    data = {}
    for window, root, runs, slugs in (
        ('current', NEWROOT, ('run1','run2','run3'), {k:v['slug'] for k,v in NEW['conditions'].items()}),
        ('earlier', OLDROOT, ('run1','rerun2','rerun3'), OLD['slugs'])):
        for key, slug in slugs.items():
            rows = []
            for run in runs:
                records = {}
                for p in (root / run / bench / slug / bench / 'collection/records').glob('*.json'):
                    r = json.loads(p.read_text()); judge = str(r.get('judge_text') or '')
                    assert not judge.startswith('DEFERRED') and judge.strip() != 'All judge API retry attempts exhausted.', p
                    valid = str(r['valid']) == 'True'
                    records[r['task_id']] = (valid, valid and float(r.get('reward') or 0) >= 1)
                assert records
                frozen = NEW['conditions'][key][bench] if window == 'current' else OLD['per_run'][key][run][bench]
                assert len(records) == (frozen['tasks'] if window == 'current' else frozen['episodes'])
                rows.append(records)
            # Verify the original full-population summary before exclusions.
            frozen = NEW['conditions'][key][bench] if window == 'current' else OLD['rates'][key][bench]
            assert sum(v[1] for row in rows for v in row.values()) == frozen['successes']
            assert sum(v[0] for row in rows for v in row.values()) == frozen['valid']
            data[window, key] = rows
    populations = [set(row) for rows in data.values() for row in rows]
    assert all(p == populations[0] for p in populations)
    tasks = sorted(populations[0] - (excluded if bench == 'webvoyager' else set()))
    assert len(tasks) == {'online_mind2web':300,'webvoyager':549,'deepshop':150}[bench]
    result['tasks_by_benchmark'][bench] = len(tasks)
    indices = rng.integers(0, len(tasks), (10000, len(tasks)))
    for (window, key), rows in data.items():
        v = np.array([np.mean([row[t][1] for row in rows]) for t in tasks])
        values[window,key,bench] = v
        samples[window,key,bench] = v[indices].mean(1)
        dst = result['conditions'] if window == 'current' else result['earlier_conditions']
        c = dst.setdefault(key, {'runs':3})
        c[bench] = dict(tasks=len(tasks), episodes=3*len(tasks),
                        successes=sum(row[t][1] for row in rows for t in tasks),
                        valid=sum(row[t][0] for row in rows for t in tasks), overall=float(v.mean()))

def summary(window, key):
    point = 100 * np.mean([values[window,key,b].mean() for b in BENCH])
    boot = 100 * np.mean([samples[window,key,b] for b in BENCH], axis=0)
    return [float(point), *map(float,np.percentile(boot,[2.5,97.5]))]

def contrast(a, b, benches=BENCH):
    point = 100*np.mean([values[a+(t,)].mean()-values[b+(t,)].mean() for t in benches])
    boot = 100*np.mean([samples[a+(t,)]-samples[b+(t,)] for t in benches],axis=0)
    return [float(point), *map(float,np.percentile(boot,[2.5,97.5]))]
for window, field in (('current','conditions'),('earlier','earlier_conditions')):
    for key, c in result[field].items():
        c['mean_overall_percent'] = summary(window,key)
        c['mean_overall'] = c['mean_overall_percent'][0]/100
        if key != 'sft':
            c['minus_sft'] = {'mean_overall_pp':contrast((window,key),(window,'sft'))}
for key in ('probe','d128','d512','d4096'):
    result['contrasts'][key+': pre - lookahead'] = {'mean_overall_pp':contrast(('current',key+'-pre'),('current',key+'-lookahead'))}
result['window_sft_change']['mean_overall_pp'] = contrast(('current','sft'),('earlier','sft'))
for b in BENCH:
    result['window_sft_change'][b] = contrast(('current','sft'),('earlier','sft'),(b,))
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
for window, field in (('current','conditions'),('earlier','earlier_conditions')):
    print(window,'SFT',result[field]['sft']['mean_overall_percent'],
          {b:round(100*result[field]['sft'][b]['overall'],2) for b in BENCH})
for key,c in result['contrasts'].items(): print(key,c['mean_overall_pp'])
print('Matched-window counts:',result['tasks_by_benchmark'])
