#!/usr/bin/env python3
"""Bounded, single-process integrity and shortcut audit; no models or API calls.

Frozen manifests are never modified. The length-only control is specified here,
fit on train only, and evaluated once without choosing settings on future data.
"""
import collections
import hashlib
import json
import os
from pathlib import Path
import resource
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
ROOT = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-turn-bonus-preparation/outcome-reward-20260927')
DEST = ROOT / 'readiness-20260928'
PUBLIC = Path(__file__).resolve().parents[1] / 'openwebrl/docs/arm_results/rl_integration/outcome-reward-readiness.json'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def sigmoid(x):
    import numpy as np
    return 1 / (1 + np.exp(-np.clip(x, -40, 40)))


def features(rows):
    import numpy as np
    # Metadata only; no terminal reason, reward, judge text, or post-action state.
    return np.asarray([[1., *np.log1p([r[k] for k in (
        'turn_count', 'supervised_response_tokens', 'context_tokens', 'maximum_context_tokens')])]
                       for r in rows], dtype=np.float64)


def fit_control(x, y):
    import numpy as np
    # Fixed L2 coefficient; Newton updates on five parameters, no search.
    penalty = np.diag([0., .01, .01, .01, .01])
    w = np.zeros(x.shape[1])
    for _ in range(30):
        p = sigmoid(x @ w)
        gradient = x.T @ (p - y) / len(y) + penalty @ w
        hessian = (x.T * (p * (1 - p))) @ x / len(y) + penalty + np.eye(x.shape[1]) * 1e-8
        delta = np.linalg.solve(hessian, gradient)
        w -= delta
        if np.max(np.abs(delta)) < 1e-8:
            break
    return w


def auc(y, score):
    import numpy as np
    positive, negative = score[y == 1], score[y == 0]
    if not len(positive) or not len(negative):
        return None
    return float(np.mean((positive[:, None] > negative) + .5 * (positive[:, None] == negative)))


def metrics(rows, p):
    import numpy as np
    y = np.array([r['terminal_success'] for r in rows]); p = np.clip(p, 1e-9, 1-1e-9)
    groups = collections.defaultdict(list)
    for i, r in enumerate(rows):
        groups[r['task_group']].append(i)
    within = [auc(y[idx], p[idx]) for idx in groups.values()]
    within = [v for v in within if v is not None]
    return dict(trajectories=len(rows), tasks=len(groups), success_rate=float(y.mean()),
        brier=float(np.mean((p-y)**2)), log_loss=float(np.mean(-y*np.log(p)-(1-y)*np.log(1-p))),
        auroc=auc(y, p), within_task_pair_accuracy_macro=float(np.mean(within)) if within else None,
        mixed_tasks=len(within))


def main():
    import numpy as np
    resource.setrlimit(resource.RLIMIT_CPU, (180, 185))
    resource.setrlimit(resource.RLIMIT_AS, (2*1024**3, 2*1024**3))
    started = time.monotonic(); DEST.mkdir(exist_ok=True)
    manifest = json.loads((ROOT/'manifest.json').read_text())
    rows = {}
    for split, expected in manifest['data_sha256'].items():
        path = ROOT/f'{split}.jsonl'
        if digest(path) != expected:
            raise ValueError('Frozen manifest checksum mismatch: '+split)
        rows[split] = [json.loads(line) for line in path.read_text().splitlines()]
    image_hashes, source_hashes = {}, {}
    for rs in rows.values():
        for row in rs:
            for path, expected, target in [(row['source_group'], row['source_sha256'], source_hashes),
                    *[(im['path'], im['sha256'], image_hashes) for im in row['image_references']]]:
                if path in target and target[path] != expected:
                    raise ValueError('Conflicting declared hash')
                target[path] = expected
    sources = {**source_hashes, **manifest['launch_provenance_sha256']}
    cache_path = DEST/'verified-images.json'
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    counts = collections.Counter(); failures = []
    for kind, entries in [('source_and_launch', sources), ('image', image_hashes)]:
        for filename, expected in entries.items():
            if time.monotonic()-started > 240:
                cache_path.write_text(json.dumps(cache)+'\n')
                raise TimeoutError('Bounded 240-second wall time reached; verified image cache saved')
            path = Path(filename); st = path.stat()
            identity = [st.st_size, st.st_mtime_ns, st.st_ctime_ns, expected]
            if kind == 'image' and cache.get(filename) == identity:
                counts['cached_images'] += 1
            elif digest(path) != expected:
                failures.append(dict(kind=kind, path=filename))
            elif kind == 'image':
                cache[filename] = identity
            counts[kind+'_files'] += 1; counts[kind+'_bytes'] += st.st_size
        cache_path.write_text(json.dumps(cache)+'\n')
    xtrain = features(rows['train']); means=xtrain[:,1:].mean(0); scales=xtrain[:,1:].std(0)
    if np.any(scales == 0): raise ValueError('Constant control feature')
    xtrain[:,1:] = (xtrain[:,1:]-means)/scales
    ytrain = np.array([r['terminal_success'] for r in rows['train']])
    w = fit_control(xtrain, ytrain)
    controls = {}; strata = {}
    for split, rs in rows.items():
        x=features(rs); x[:,1:] = (x[:,1:]-means)/scales
        p=sigmoid(x @ w)
        natural=[i for i,r in enumerate(rs) if r['termination']=='task_completed']
        controls[split] = dict(train_prevalence=metrics(rs, np.full(len(rs), ytrain.mean())),
                              length_only=metrics(rs, p),
                              length_only_excluding_budget_terminations=metrics([rs[i] for i in natural],p[natural]))
        strata[split] = {}
        for reason in sorted({r['termination'] for r in rs}):
            selected=[r for r in rs if r['termination']==reason]
            strata[split][reason]=dict(trajectories=len(selected),successes=sum(r['terminal_success'] for r in selected))
    report = dict(status='integrity_passed' if not failures else 'integrity_failed',
        manifest_sha256=digest(ROOT/'manifest.json'), integrity=dict(counts), integrity_failures=len(failures),
        control='Fixed L2=.01 logistic regression: log1p turns, response tokens, total context, maximum context. Train-only scaling/fitting; no outcome balancing.',
        controls=controls, terminal_reason_strata=strata,
        resources=dict(seconds=time.monotonic()-started,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,gpus=0,api_calls=0),
        limitations=['Image bytes checked against hashes, not image decoder or actor processor parity.',
            'Task-macro pair accuracy uses only tasks with both outcomes; not same-state action ranking.',
            'Later-actor test also has new task IDs: cannot isolate temporal drift from task generalization.',
            'Controls are diagnostics, not estimates of reward-model performance or causal action credit.',
            'Model inputs must whitelist archived prompt/tokens/images; metadata includes judge text/outcomes and post-action tool responses.'])
    (DEST/'controls-fit.json').write_text(json.dumps(dict(weights=w.tolist(),mean=means.tolist(),scale=scales.tolist(),features=['intercept','turns','response_tokens','total_context','max_context']),indent=2)+'\n')
    (DEST/'integrity-failures.json').write_text(json.dumps(failures,indent=2)+'\n')
    (DEST/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    PUBLIC.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
