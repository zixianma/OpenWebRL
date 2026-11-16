"""Absolute mean-overall 95% CIs for the frozen November 14–15 head comparison.

Bootstrap tasks independently within each benchmark, keeping each task's three
runs together; give each benchmark one third weight. Same 10,000 resamples,
seed and common-task population as selection-head-pre-vs-lookahead-20261115.py.
Intervals quantify task uncertainty conditional on the observed runs and heads.
No calls or experiments; reads saved judged episodes and exports aggregates only.
"""
from pathlib import Path
import json
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/recipe3-ablations-20261104/eval-pre')
SOURCE = json.loads((HERE / 'selection-head-pre-vs-lookahead-20261115.json').read_text())
BENCH = ('online_mind2web', 'webvoyager', 'deepshop')
RUNS = sorted(ROOT.glob('run*'))
assert len(RUNS) == 3
B, SEED = 10000, 20261029
rng = np.random.default_rng(SEED)
points, samples = {}, {}
for bench in BENCH:
    data = {}
    for key, condition in SOURCE['conditions'].items():
        rows = []
        for run in RUNS:
            records = {}
            folder = run / bench / condition['slug'] / bench / 'collection/records'
            for path in folder.glob('*.json'):
                row = json.loads(path.read_text())
                judge = str(row.get('judge_text') or '')
                assert not judge.startswith('DEFERRED') and judge.strip() != 'All judge API retry attempts exhausted.', path
                valid = str(row['valid']) == 'True'
                records[row['task_id']] = (valid, valid and float(row.get('reward') or 0) >= 1)
            assert records, folder
            rows.append(records)
        data[key] = rows
    tasks = sorted(set.intersection(*(set(run) for rows in data.values() for run in rows)))
    indices = rng.integers(0, len(tasks), (B, len(tasks)))
    for key, rows in data.items():
        values = np.array([np.mean([run[t][1] for run in rows]) for t in tasks])
        frozen = SOURCE['conditions'][key][bench]
        assert len(tasks) == frozen['tasks']
        assert sum(run[t][1] for run in rows for t in tasks) == frozen['successes']
        assert np.isclose(values.mean(), frozen['overall'])
        points[key, bench] = values.mean()
        samples[key, bench] = values[indices].mean(axis=1)
result = {'method': __doc__, 'resamples': B, 'seed': SEED, 'conditions': {}}
for key, condition in SOURCE['conditions'].items():
    point = 100 * np.mean([points[key, b] for b in BENCH])
    bootstrap = 100 * np.mean([samples[key, b] for b in BENCH], axis=0)
    assert np.isclose(point, 100 * condition['mean_overall'])
    result['conditions'][key] = {'mean_overall_percent': [float(point), *map(float, np.percentile(bootstrap, [2.5, 97.5]))]}
# Also verify the existing paired intervals exactly use this bootstrap population.
for key, condition in SOURCE['conditions'].items():
    if key == 'sft':
        continue
    delta = 100 * np.mean([samples[key, b] - samples['sft', b] for b in BENCH], axis=0)
    assert np.allclose(np.percentile(delta, [2.5, 97.5]), condition['minus_sft']['mean_overall_pp'][1:])
Path(__file__).with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
print('Verified frozen counts, means and paired CIs; computed absolute CIs for', len(result['conditions']), 'conditions.')
