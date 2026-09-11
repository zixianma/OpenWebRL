"""Inspect saved DP1 rollout minibatch metadata without loading image payloads.

Reconstructs the logged owner-preserving epoch shuffle for fixed GBS, unbalanced
DP1 data. It describes batch composition, not per-sample gradient causality.
"""
import argparse
import ast
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import re
import statistics
from urllib.parse import urlparse

import torch


def describe(values):
    values = sorted(values)
    if not values:
        return None
    return dict(min=values[0], median=statistics.median(values),
                mean=statistics.mean(values), p95=values[int(.95 * (len(values) - 1))], max=values[-1])


def audit(root, rollout, epoch, spike_step, seed):
    log = (root / 'training.log').read_text(errors='replace')
    if not re.search(r'balance_data\s+\.+\s+False', log):
        raise ValueError('This audit requires balance_data=False.')
    if not re.search(r'grpo_std_normalization\s+\.+\s+True', log):
        raise ValueError('This audit requires GRPO standard-deviation normalization.')
    pattern = (rf'rollout_id={rollout} ppo_epoch={epoch} selection_seed={seed} .*?'
               r'local_num_samples=(\d+) source_local_counts=\[(\d+)\] '
               r'epoch_effective_global_batch_size=256')
    match = re.search(pattern, log)
    if not match:
        raise ValueError('Missing matching DP1 fixed-batch epoch selection record.')
    usable, source_count = map(int, match.groups())
    archive = root / 'rollout_recovery' / f'{rollout}.pt'
    data = torch.load(archive, map_location='cpu', weights_only=False, mmap=True)
    samples = data['samples']
    assert len(samples) == source_count and data['rollout_id'] == rollout
    order = list(range(len(samples)))
    random.Random(seed).shuffle(order)
    order = order[:usable]
    groups = defaultdict(dict)
    for s in samples:
        meta = s.get('metadata') or {}
        groups[s['group_index']].setdefault(meta.get('trajectory_id', s['index']), s['reward'])
    normalized = {}
    for group, rewards in groups.items():
        mean = statistics.mean(rewards.values())
        std = statistics.stdev(rewards.values()) if len(rewards) > 1 else 0
        for trajectory, reward in rewards.items():
            normalized[group, trajectory] = (reward - mean) / (std + 1e-6)
    batches, spike_samples = [], []
    for batch_id in range(usable // 256):
        selected = [samples[i] for i in order[batch_id * 256:(batch_id + 1) * 256]]
        rows = []
        for s in selected:
            meta = s.get('metadata') or {}
            lp = s.get('rollout_log_probs') or []
            mask = s.get('loss_mask')
            response_len = s['response_length']
            trajectory = meta.get('trajectory_id', s['index'])
            active = sum(mask) if mask is not None else response_len
            if s.get('remove_sample'):
                active = 0
            finite = [x for x in lp if math.isfinite(x)]
            rows.append(dict(sample_id=s['index'], group_id=s['group_index'],
                             trajectory_id=trajectory, task_id=meta.get('task_id'),
                             domain=urlparse(meta.get('start_url', '')).netloc,
                             turn=meta.get('turn_index'), trajectory_turns=meta.get('num_turns_in_trajectory'),
                             total_tokens=len(s['tokens']), response_tokens=response_len,
                             active_response_tokens=active, reward=s['reward'],
                             normalized_advantage=normalized[s['group_index'], trajectory],
                             status=s['status'], removed=bool(s.get('remove_sample')),
                             nonfinite_rollout_logprobs=len(lp)-len(finite),
                             mean_surprisal=-statistics.mean(finite) if finite else None,
                             max_surprisal=-min(finite) if finite else None))
        trajectories = Counter(x['trajectory_id'] for x in rows)
        grad = re.search(rf'train_one_step end rollout_id={rollout} ppo_epoch={epoch} '
                         rf'step_id={batch_id} grad_norm=([\d.eE+-]+)', log)
        # Independently check the reconstructed batch against recorded training
        # statistics rather than trusting the seed alone.
        label = rollout * (usable // 256) * 2 + epoch * (usable // 256) + batch_id
        logged = re.search(rf'model.py:\d+ - step {label}: (\{{[^\n]*\}})', log)
        recorded = ast.literal_eval(logged[1]) if logged else None
        # The training reducer contributes zero for a fully masked sample,
        # while retaining it in the minibatch denominator.
        reconstructed_mean = statistics.mean(
            abs(x['normalized_advantage']) if x['active_response_tokens'] > 0 else 0.0
            for x in rows)
        if recorded and not math.isclose(reconstructed_mean, recorded['train/abs_advantage'], abs_tol=1e-4):
            raise ValueError(f'Batch {batch_id} reconstructed advantages disagree with training; inspect masks/order.')
        batches.append(dict(batch=batch_id, grad_norm=float(grad[1]) if grad else None,
                            reconstructed_advantage_matches_log=bool(recorded),
                            reconstructed_logged_abs_advantage=reconstructed_mean,
                            sample_count=len(rows), unique_trajectories=len(trajectories),
                            max_turns_from_one_trajectory=max(trajectories.values()),
                            groups=len({x['group_id'] for x in rows}),
                            domains=dict(Counter(x['domain'] for x in rows)),
                            reward_mean=statistics.mean(x['reward'] for x in rows),
                            abs_advantage=describe([abs(x['normalized_advantage']) for x in rows]),
                            total_tokens=describe([x['total_tokens'] for x in rows]),
                            response_tokens=describe([x['response_tokens'] for x in rows]),
                            active_response_tokens=describe([x['active_response_tokens'] for x in rows]),
                            nonfinite_rollout_logprobs=sum(x['nonfinite_rollout_logprobs'] for x in rows),
                            removed_samples=sum(x['removed'] for x in rows),
                            status=dict(Counter(x['status'] for x in rows)),
                            mean_surprisal=describe([x['mean_surprisal'] for x in rows if x['mean_surprisal'] is not None])))
        if batch_id == spike_step:
            spike_samples = rows
    return dict(root=str(root), rollout_id=rollout, reward_iteration=rollout+1, ppo_epoch=epoch,
                seed=seed, source_samples=source_count, selected_samples=usable,
                spike_step=spike_step, batches=batches, spike_samples=spike_samples,
                limitation='Metadata comparison only; does not identify per-sample gradient contributions. Image tensor payloads are not accessed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--rollout', type=int, required=True)
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    report = audit(a.root, a.rollout, a.epoch, a.step, a.seed)
    a.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['spike_samples', 'batches']}, indent=2))
    for batch in report['batches']:
        print(json.dumps({k:batch[k] for k in ['batch','grad_norm','unique_trajectories','max_turns_from_one_trajectory','reward_mean','abs_advantage','response_tokens','nonfinite_rollout_logprobs','removed_samples']}))
