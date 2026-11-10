#!/usr/bin/env python3
"""CPU replay of sequentially stopping the frozen pointwise selection head.

The five archived proposals are kept in actor sampling order. A policy sees only
the scores of proposals already generated and stops once their best logit
exceeds a threshold. Thresholds target an average proposal count using scores
on other task groups; continuation outcomes never tune the stopping rule.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openwebrl.adaptive_selection import AdaptiveScorePolicy

ROOT = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/selection-head-20261008')


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def load_scores(root: Path):
    import torch
    from openwebrl.selection_head import CANDIDATE_POOLS, CandidateSetHead, HeadConfig

    torch.set_num_threads(1)
    head_path = root / 'heads/pointwise/head.pt'
    checkpoint = torch.load(head_path, map_location='cpu', weights_only=False, mmap=True)
    config = HeadConfig.from_json(checkpoint['config'])
    assert not config.listwise and tuple(config.layers) == (18, 27, 36)
    assert tuple(config.pools) == CANDIDATE_POOLS
    head = CandidateSetHead(config).eval()
    head.load_state_dict(checkpoint['state_dict'])
    config_json = json.loads(checkpoint['config'])
    del checkpoint
    gc.collect()

    panel_path = root / 'branch124/panels.jsonl'
    panels = [json.loads(line) for line in panel_path.open() if line.strip()]
    assert len(panels) == 124 and len({p['task_group'] for p in panels}) == 124
    wanted = {p['state_id']: p for p in panels}
    features = {}
    paths = sorted((root / 'features-branch124').glob('shard*-chunk*.pt'))
    assert paths
    for path in paths:
        shard = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        keep = [shard['layers'].index(layer) for layer in config.layers]
        for sid, tensor in shard['panels'].items():
            if sid not in wanted:
                continue
            row = wanted[sid]
            features[sid] = (tensor, shard['states'][row['base_state_id']], keep,
                             len(shard['layers']))
    assert len(features) == len(panels)
    scores = []
    max_prefix_error = 0.0
    with torch.inference_mode():
        for row in panels:
            tensor, state_tensor, keep, layer_count = features[row['state_id']]
            candidates = tensor.view(5, layer_count, len(CANDIDATE_POOLS), -1)[:, keep]
            candidates = candidates.reshape(1, 5, -1).float()
            state = state_tensor[keep].reshape(1, -1).float()
            complete = head(candidates, state)[0]
            scores.append(complete[0].tolist())
            # The deployed pointwise scorer must give the same early scores
            # whether called after k proposals or after all five.
            for k in range(1, 5):
                partial = head(candidates[:, :k], state)[0]
                max_prefix_error = max(max_prefix_error,
                                       (partial - complete[:, :k]).abs().max().item())
    assert max_prefix_error < 1e-4, max_prefix_error
    archived_path = root / 'heads/pointwise/metrics.json'
    archived = json.loads(archived_path.read_text())['branch_head_picks']
    disagreements = [(row, score) for row, score in zip(panels, scores)
                     if pick(score, 5) != archived[row['state_id']]]
    assert all(max(score) - min(score) < 1e-4 for _, score in disagreements)
    assert all(len({(o['successes'], o['trials'], o['valid']) for o in row['outcomes']}) == 1
               for row, _ in disagreements)
    return panels, scores, {'checkpoint_sha256': digest(head_path),
                             'panel_sha256': digest(panel_path),
                             'archived_metrics_sha256': digest(archived_path),
                             'feature_shards': [{'name': p.name, 'sha256': digest(p)} for p in paths],
                             'max_prefix_logit_error': max_prefix_error,
                             'archived_pick_matches': len(panels) - len(disagreements),
                             'archived_pick_near_ties': len(disagreements),
                             'head_config': config_json}


def pick(scores, k):
    return max(range(k), key=lambda i: (scores[i], -i))


def stop_at(scores, threshold):
    return AdaptiveScorePolicy(threshold).replay(scores)[0]


def threshold_for_target(scores, target):
    values = sorted({value for row in scores for value in row[:4]})
    thresholds = [values[0] - 1.] + values + [values[-1] + 1.]
    candidate = min(thresholds, key=lambda t: (abs(sum(stop_at(row, t) for row in scores) /
                                                    len(scores) - target),
                                                sum(stop_at(row, t) for row in scores), t))
    return candidate


def rate(row, choice, valid_only):
    observed = row['outcomes'][choice]
    denominator = observed['valid'] if valid_only else observed['trials']
    return observed['successes'] / denominator if denominator else None


def mean(values):
    return sum(values) / len(values)


def interval(deltas, seed=0, n=10000):
    rng = random.Random(seed)
    draws = sorted(mean([deltas[rng.randrange(len(deltas))] for _ in deltas])
                   for _ in range(n))
    return [100 * draws[int(.025 * n)], 100 * draws[int(.975 * n) - 1]]


def evaluate(panels, scores, picks, costs):
    strict_valid = [i for i, p in enumerate(panels)
                    if all(outcome['valid'] > 0 for outcome in p['outcomes'])]
    tokens = [sum(len(ids) for ids in row['candidate_token_ids'][:cost])
              for row, cost in zip(panels, costs)]
    full_tokens = [sum(len(ids) for ids in row['candidate_token_ids']) for row in panels]
    result = {'states': len(panels), 'task_groups': len({p['task_group'] for p in panels}),
              'avg_proposals': mean(costs),
              'avg_generated_response_tokens': mean(tokens),
              'response_tokens_vs_full_five_pct': 100 * sum(tokens) / sum(full_tokens),
              'overall_success_pct': 100 * mean([rate(p, j, False) for p, j in zip(panels, picks)]),
              'fully_valid_coverage_states': len(strict_valid),
              'fully_valid_selected_success_pct': 100 * mean([rate(panels[i], picks[i], True)
                                                              for i in strict_valid]) if strict_valid else None,
              'matches_full_head_states': sum(pick(s, 5) == j for s, j in zip(scores, picks))}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    panels, scores, provenance = load_scores(args.root)
    strict = [i for i, row in enumerate(panels) if row['prompt_verified']
              and not row['piotr_task_overlap']]
    assert len(strict) == 105
    rows = [panels[i] for i in strict]
    logits = [scores[i] for i in strict]
    folds = [int(hashlib.sha256(p['task_group'].encode()).hexdigest()[:8], 16) % 5 for p in rows]
    full = [pick(s, 5) for s in logits]
    fixed = {}
    adaptive = {}
    for k in range(1, 6):
        choices = [pick(s, k) for s in logits]
        result = evaluate(rows, logits, choices, [k] * len(rows))
        result['minus_full_pp'] = result['overall_success_pct'] - evaluate(
            rows, logits, full, [5] * len(rows))['overall_success_pct']
        result['task_bootstrap_95_minus_full_pp'] = interval([
            rate(row, j, False) - rate(row, fj, False)
            for row, j, fj in zip(rows, choices, full)], seed=100 + k)
        fixed[str(k)] = result
    for target in (2, 3, 4):
        thresholds = {}
        choices = [None] * len(rows)
        costs = [None] * len(rows)
        for fold in range(5):
            train = [logits[i] for i, f in enumerate(folds) if f != fold]
            threshold = threshold_for_target(train, target)
            thresholds[str(fold)] = threshold
            for i, f in enumerate(folds):
                if f == fold:
                    costs[i] = stop_at(logits[i], threshold)
                    choices[i] = pick(logits[i], costs[i])
        assert all(x is not None for x in choices + costs)
        result = evaluate(rows, logits, choices, costs)
        result['thresholds_by_fold'] = thresholds
        result['cost_histogram'] = {str(k): costs.count(k) for k in range(1, 6)}
        result['minus_fixed_target_pp'] = result['overall_success_pct'] - fixed[str(target)]['overall_success_pct']
        result['task_bootstrap_95_minus_fixed_target_pp'] = interval([
            rate(row, j, False) - rate(row, pick(score, target), False)
            for row, score, j in zip(rows, logits, choices)], seed=200 + target)
        common_valid = [i for i, row in enumerate(rows)
                        if all(outcome['valid'] > 0 for outcome in row['outcomes'])]
        result['fully_valid_minus_fixed_target_pp'] = (
            result['fully_valid_selected_success_pct'] - fixed[str(target)]['fully_valid_selected_success_pct'])
        result['task_bootstrap_95_fully_valid_minus_fixed_target_pp'] = interval([
            rate(rows[i], choices[i], True) - rate(rows[i], pick(logits[i], target), True)
            for i in common_valid], seed=400 + target)
        result['minus_full_pp'] = result['overall_success_pct'] - fixed['5']['overall_success_pct']
        result['task_bootstrap_95_minus_full_pp'] = interval([
            rate(row, j, False) - rate(row, fj, False)
            for row, j, fj in zip(rows, choices, full)], seed=300 + target)
        adaptive[str(target)] = result
    output = {'status': 'retrospective_cpu_replay',
              'design': 'Frozen pointwise head; five archived actor-order proposals; score-only '
                        'task-cross-fitted stopping thresholds; three historical continuation draws '
                        'per action; invalid outcomes count as failures',
              'provenance': provenance,
              'cohort': {'all_archived_states': len(panels),
                         'strict_no_piotr_overlap_states': len(rows),
                         'strict_task_groups': len({p['task_group'] for p in rows}),
                         'fully_valid_strict_states': fixed['5']['fully_valid_coverage_states'],
                         'fold_sizes': {str(f): folds.count(f) for f in range(5)}},
              'prospective_score_only_thresholds': {
                  str(target): threshold_for_target(logits, target) for target in (2, 3, 4)},
              'fixed_k': fixed, 'adaptive_target_k': adaptive,
              'limits': ['Historical three-draw outcomes have been reused in prior architecture work.',
                         'No new candidate generation, live browser run or measured wall-time saving.',
                         'Adaptive thresholds use only other tasks\' score distributions, not outcomes.',
                         'Candidate count is a compute proxy; serial stopping can alter latency.',
                         'The actor candidate order and pointwise head are fixed.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output': str(args.output), 'fixed': fixed,
                      'adaptive': adaptive}, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
