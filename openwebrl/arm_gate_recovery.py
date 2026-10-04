"""Staged recovery helpers for the deliberately higher-coverage B/C ablations.

The optional scale bound follows the pre-training coverage audit, not the
current batch's measured reward. Rewards and their coefficients stay unchanged.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

AUDIT_OLD_ADMITTED = 13471
AUDIT_MIN2_ADMITTED = 25181
SCALE_MAX = .10 * math.sqrt(AUDIT_MIN2_ADMITTED / AUDIT_OLD_ADMITTED)


def auxiliary_payload_limit(config):
    """Keep a bounded artifact while allowing min2's roughly doubled coverage."""
    from openwebrl.arm_turn_bonus import candidate_minimum
    return (8 if config.get('failure_scored_fraction') == .4 else
            4 if candidate_minimum(config) == 2 else 2) * 1024**3


def calibration_guard(report, decision):
    config = report.get('config', {})
    if config.get('gate_scale_guard') is None:
        return decision
    if (config['gate_scale_guard'] != 'pretraining_coverage_audit_v1'
            or config.get('candidate_gate') != 'min2'
            or config.get('credit_assignment') not in ('response_index', 'action_class')
            or config.get('beta') != .5 or config.get('scored_fraction') != .2):
        raise ValueError('Unsupported coverage-aware guard configuration')
    result = deepcopy(decision)
    ratio = report['variants']['0.5']['bonus_to_outcome_rms']
    result['checks']['scale'] = ratio is not None and math.isfinite(ratio) and .04 <= ratio <= SCALE_MAX
    result['passed'] = all(result['checks'].values())
    result['scale_guard'] = dict(mode=config['gate_scale_guard'], old_upper=.10,
        lower=.04, upper=SCALE_MAX, coverage_audit_old=13471, coverage_audit_min2=25181,
        above_old_upper=ratio is not None and math.isfinite(ratio) and ratio > .10)
    result['rule'] = ('Keep beta=0.5 and q=0.2. For min2 only, retain the 4% RMS floor '
        'and use 10% * sqrt(25181/13471) as the ceiling from the pre-training coverage audit. '
        'All label/task/coverage checks remain unchanged. This changes the stop guard, not rewards.')
    return result


def _canonical_action_histogram(histogram):
    """JSON stringifies integer Counter keys; preserve every bin and count."""
    result = {}
    for key, count in histogram.items():
        if type(key) is int and 2 <= key <= 5:
            canonical = str(key)
        elif type(key) is str and key in ('2', '3', '4', '5'):
            canonical = key
        else:
            raise ValueError('Invalid distinct-action histogram bin')
        if canonical in result or type(count) is not int or count < 0:
            raise ValueError('Ambiguous or invalid distinct-action histogram count')
        result[canonical] = count
    return result


def replay_auxiliary(args, samples, current, report):
    config = current['config']
    origin = config.get('gate_replay_origin')
    if not origin or config['rollout_id'] != 0:
        return False
    folder = Path(origin['source'])/'iterations/0000'
    manifest_path = folder/'failure_auxiliary.json'
    raw = manifest_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != origin['auxiliary_manifest_sha256']:
        raise ValueError('Saved auxiliary manifest changed')
    manifest = json.loads(raw)
    if manifest['tensor_file'] != 'failure_auxiliary.pt':
        raise ValueError('Unexpected auxiliary tensor path')
    for key in ('policy_id', 'checkpoint', 'candidate_gate', 'credit_assignment'):
        if manifest[key] != config[key]:
            raise ValueError('Saved auxiliary policy/recipe does not match replay')
    if (manifest['rollout_id'] != 0 or manifest['beta'] != .5 or manifest['q'] != .2
            or manifest['expected_windows'] != len(samples)//args.global_batch_size):
        raise ValueError('Saved auxiliary windows or coefficients changed')
    if len({s.group_index for s in samples}) != 48 or manifest['mixed_groups'] != 48:
        raise ValueError('Replay must preserve all 48 ordinary groups')
    previous = json.loads((folder/'calibration.json').read_text())
    for key in ('rows', 'batch_rows', 'admitted', 'admitted_distinct_tasks',
                'admitted_distinct_action_counts', 'task_counts', 'label_reasons'):
        observed, expected = report[key], previous[key]
        if key == 'admitted_distinct_action_counts':
            observed = _canonical_action_histogram(observed)
            expected = _canonical_action_histogram(expected)
        if observed != expected:
            raise ValueError('Replayed batch differs from the saved calibration: '+key)
    for key in ('outcome_rms', 'unit_bonus_rms'):
        if not math.isclose(report[key], previous[key], rel_tol=1e-7, abs_tol=1e-9):
            raise ValueError('Replayed reward scale differs: '+key)
    tensor = folder/manifest['tensor_file']
    if tensor.stat().st_size != origin['auxiliary_bytes']:
        raise ValueError('Saved auxiliary tensor size changed')
    # The native actor subsequently verifies the full tensor SHA256 before use.
    root = Path(config['output'])
    (root/manifest['tensor_file']).symlink_to(tensor)
    (root/'failure_auxiliary.json').write_bytes(raw)
    for key, value in previous.items():
        if key.startswith('additive_failure_') or key == 'outcome_groups_preserved':
            report[key] = value
    report['replayed_auxiliary_from'] = str(manifest_path)
    from openwebrl.arm_turn_bonus import write_json
    write_json(root/'calibration.json', report)
    return True
