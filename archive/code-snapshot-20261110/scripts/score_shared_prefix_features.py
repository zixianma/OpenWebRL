#!/usr/bin/env python3
"""Fit the old-cohort value head; freeze new-state picks before any new returns."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.linalg import solve
import torch


OLD = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/selection-action-chunk-outcome-20261017-v2')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def rows(path):
    with path.open() as stream:
        keep = ('state_id', 'task_group', 'actions', 'outcomes')
        return [{key: value[key] for key in keep if key in value}
                for line in stream if line.strip() for value in [json.loads(line)]]


def fit_rates(row):
    counts = defaultdict(lambda: [0, 0])
    for action, outcomes in zip(row['actions'], row['outcomes']):
        for repeat in range(5):
            result = outcomes[repeat]
            if result['valid']:
                counts[action['name']][0] += int(result['success'])
                counts[action['name']][1] += 1
    return {name: wins / valid for name, (wins, valid) in counts.items() if valid}


def collapse(options):
    by_name = {}
    for option in options:
        name = option['name']
        if name not in by_name or option['logprob'] > by_name[name]['logprob']:
            by_name[name] = option
    return by_name


def fit(output):
    panels = rows(OLD / 'panels.jsonl')
    shards = sorted((OLD / 'results').glob('features-*.pt'))
    features = [item for shard in shards
                for item in torch.load(shard, map_location='cpu', weights_only=False)]
    if len(panels) != len(features) or len(panels) != 280:
        raise ValueError('Frozen training panel identity changed')
    xs, ys, ws = [], [], []
    groups = set()
    for row, feature in zip(panels, features):
        if row['state_id'] != feature['state_id']:
            raise ValueError('Training features and outcomes differ')
        groups.add(row['task_group'])
        by_name = collapse(feature['options'])
        names = sorted(by_name)
        rates = fit_rates(row)
        if len(names) < 2 or set(names) != set(rates):
            raise ValueError('Training panel lost a scoreable tool name')
        h = np.stack([by_name[name]['h'].float().numpy() for name in names])
        h -= h.mean(0, keepdims=True)
        target = np.array([rates[name] for name in names]); target -= target.mean()
        xs.append(h); ys.append(target); ws.extend([1 / len(names)] * len(names))
    X, Y, weights = np.concatenate(xs), np.concatenate(ys), np.array(ws)
    mu, sd = X.mean(0), X.std(0) + 1e-5
    X = (X - mu) / sd
    root_weight = np.sqrt(weights)
    matrix = (X @ X.T) * root_weight[:, None] * root_weight[None, :]
    alpha = 1e6  # Four of five old-cohort outer folds selected this value.
    matrix.flat[::len(X) + 1] += alpha
    dual = solve(matrix, Y * root_weight, assume_a='pos', check_finite=False)
    head = X.T @ (root_weight * dual)
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / 'value-head.npz', head=head, mu=mu, sd=sd)
    manifest = dict(status='fit_before_new_outcomes', alpha=alpha,
                    training_states=280, training_task_groups=len(groups),
                    train_task_groups_sha256=hashlib.sha256('\n'.join(sorted(groups)).encode()).hexdigest(),
                    old_panels_sha256=sha(OLD / 'panels.jsonl'),
                    old_feature_shard_sha256={p.name: sha(p) for p in shards},
                    value_head_sha256=sha(output / 'value-head.npz'),
                    recipe='Within-state centered valid-only draw-0:4 rates; one weighted ridge fit on all 280 old states; fixed alpha 1e6')
    (output / 'head-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def score(root):
    root = Path(root)
    model = root / 'model'
    manifest = json.loads((model / 'head-manifest.json').read_text())
    if sha(model / 'value-head.npz') != manifest['value_head_sha256']:
        raise ValueError('Precommitted head changed')
    with np.load(model / 'value-head.npz') as z:
        head, mu, sd = z['head'], z['mu'], z['sd']
    new = rows(root / 'data/panels.jsonl')
    features = [item for shard in sorted((root / 'results/features').glob('features-*.pt'))
                for item in torch.load(shard, map_location='cpu', weights_only=False)]
    if len(new) != len(features) or len(new) != 8:
        raise ValueError('External features and states differ')
    old_groups = {row['task_group'] for row in rows(OLD / 'panels.jsonl')}
    if any(row['task_group'] in old_groups for row in new):
        raise ValueError('External task group overlaps training')
    picks = []
    for row, feature in zip(new, features):
        if row['state_id'] != feature['state_id']:
            raise ValueError('External feature identity mismatch')
        by_name = collapse(feature['options']); names = sorted(by_name)
        h = np.stack([by_name[name]['h'].float().numpy() for name in names])
        h -= h.mean(0, keepdims=True)
        value = ((h - mu) / sd) @ head
        actor = {name: by_name[name]['logprob'] for name in names}
        value = dict(zip(names, map(float, value)))
        actor_order = sorted(names, key=lambda name: (-actor[name], name))
        value_order = sorted(names, key=lambda name: (-value[name], name))
        picks.append(dict(state_id=row['state_id'], task_group=row['task_group'],
                          actor_name=actor_order[0], value_name=value_order[0],
                          actor_margin_nats=actor[actor_order[0]] - actor[actor_order[1]],
                          value_margin_rate=value[value_order[0]] - value[value_order[1]],
                          observed_name_count=len(names)))
    output = model / 'frozen-picks.json'
    if output.exists():
        if json.loads(output.read_text()) != picks:
            raise ValueError('Earlier outcome-blind picks differ')
    else:
        output.write_text(json.dumps(picks, indent=2) + '\n')
    return dict(states=len(picks), changed=sum(p['actor_name'] != p['value_name'] for p in picks),
                frozen_picks_sha256=sha(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--fit', action='store_true')
    parser.add_argument('--score', action='store_true')
    args = parser.parse_args()
    if args.fit == args.score:
        raise ValueError('Select exactly one mode')
    result = fit(args.root / 'model') if args.fit else score(args.root)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
