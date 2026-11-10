#!/usr/bin/env python3
"""Audit a proposed weighting recipe on verified JSON journals; never train."""
import hashlib
import json
import math
from pathlib import Path
import resource
import statistics
import time

from audit_arm_mixed_alignment import reconstruct_group, reweighted

RUNTIME = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime')
REPO = Path(__file__).resolve().parents[1]


def main():
    resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
    started = time.monotonic()
    source = RUNTIME/'arm-turn-bonus-preparation/mixed-alignment-20260926/audit.json'
    audit = json.loads(source.read_text())
    counts = dict(groups=0, binary_groups=0, fallback_groups=0, trajectories=0,
                  variable_signal_trajectories=0, labeled_trajectories=0,
                  turns=0, fallback_turns=0)
    a2 = b2 = delta2 = 0.
    binary_a2 = binary_b2 = binary_delta2 = 0.
    weights = []
    ess_fractions = []
    max_mean_error = 0.
    examples = []
    for iteration, check in sorted(audit['checks'].items(), key=lambda item: int(item[0])):
        calibration = Path(check['folder'])/'calibration.json'
        assert hashlib.sha256(calibration.read_bytes()).hexdigest() == check['calibration_sha256']
        policy = json.loads(calibration.read_text())['policy_id']
        local_groups = 0
        for path in sorted((Path(check['journal_folder'])/'groups').glob('*/*.json')):
            header = json.loads(path.read_text())
            if not header['accepted']:
                continue
            assert header['policy_id'] == policy
            trajectories = reconstruct_group(header, policy)
            binary = all(t['outcome'] in (0, 1) for t in trajectories)
            counts['groups'] += 1
            counts['binary_groups' if binary else 'fallback_groups'] += 1
            local_groups += 1
            for tr in trajectories:
                old = [t['A'] + t['b'] for t in tr['turns']]
                changed = reweighted(tr, .5) if binary else [(None, x) for x in old]
                counts['turns'] += len(changed)
                if binary:
                    counts['trajectories'] += 1
                    counts['labeled_trajectories'] += any(t['labeled'] for t in tr['turns'])
                    counts['variable_signal_trajectories'] += len({t['u'] for t in tr['turns']}) > 1
                    ws = [w for w, _ in changed]
                    weights.extend(ws)
                    ess_fractions.append(sum(ws)**2 / (len(ws)*sum(w*w for w in ws)))
                    max_mean_error = max(max_mean_error, abs(statistics.mean(x for _, x in changed) - tr['A']))
                else:
                    counts['fallback_turns'] += len(changed)
                for t, (w, new) in zip(tr['turns'], changed):
                    a2 += t['A']**2
                    b2 += t['b']**2
                    delta2 += (new-t['A'])**2
                    if binary:
                        binary_a2 += t['A']**2
                        binary_b2 += t['b']**2
                        binary_delta2 += (new-t['A'])**2
                        assert new*t['A'] >= 0
                        assert math.exp(-.5)-1e-12 <= w <= math.exp(.5)+1e-12
        assert local_groups == 48, (iteration, local_groups)
    for A in (1., -1.):
        tr = dict(A=A, turns=[dict(u=.8), dict(u=-.2)]+[dict(u=0.)]*8)
        values = reweighted(tr, .5)
        examples.append(dict(A=A, turns=10, preferred=values[0], rejected=values[1], unlabeled=values[2]))
    weights.sort()
    result = dict(
        schema_version=1, experiment='arm-outcome-aware-reweighting-lambda0.5',
        status='CPU diagnostic only; no new training or GPU/API calls',
        source_audit_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_iterations=sorted(map(int, audit['checks'])), lambda_value=.5,
        error_policy='Groups containing native -1 sentinel keep the original Gate B objective.',
        counts=counts,
        all_groups=dict(parent_perturbation_rms=math.sqrt(b2/a2), proposed_perturbation_rms=math.sqrt(delta2/a2)),
        binary_groups=dict(parent_perturbation_rms=math.sqrt(binary_b2/binary_a2),
            proposed_perturbation_rms=math.sqrt(binary_delta2/binary_a2),
            minimum_weight=weights[0], maximum_weight=weights[-1],
            p05_weight=weights[int(.05*(len(weights)-1))],p95_weight=weights[int(.95*(len(weights)-1))],
            mean_trajectory_ess_fraction=statistics.mean(ess_fractions),
            minimum_trajectory_ess_fraction=min(ess_fractions),
            max_mean_advantage_error=max_mean_error),
        examples=examples,
        limits=['Before native epoch shuffling/trimming; not a gradient or policy-performance estimate.',
                'Weight ESS is descriptive concentration, not independent effective observations.',
                'Only the already audited B1-69 panel is used.'],
        resources=dict(wall_seconds=time.monotonic()-started,
                       maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                       gpus=0,api_calls=0,tensors_loaded=0))
    output=REPO/'openwebrl/docs/arm_results/rl_integration/reweighting-design-audit.json'
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
