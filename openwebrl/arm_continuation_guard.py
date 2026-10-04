"""Optional count-only tolerance for a continued, already calibrated ARM run.

No reward, label, scale, task-coverage or failure-supervision rule changes here.
The preceding batch must have passed the original >=100-label count on its own
and produced the checkpoint used to collect this batch. Two consecutive sparse
batches cannot borrow evidence from each other.
"""
import hashlib
import json
from copy import deepcopy
from pathlib import Path


def apply_adjacent_count_support(report, decision):
    config = report.get('config', {})
    mode = config.get('label_count_guard')
    if mode is None:
        return decision
    if mode != 'adjacent_batch_v1' or not report.get('all_failure_experiment'):
        raise ValueError('Unsupported continuation label-count guard')
    result = deepcopy(decision)
    result['count_guard'] = dict(mode=mode, used=False, current_labels=report['admitted'])
    # Only the original count check may be relaxed. Current task coverage,
    # advantage scale and usable all-failure supervision must pass independently.
    if result['checks']['labels'] or not all(v for k, v in result['checks'].items() if k != 'labels'):
        return result
    if report['admitted'] < 50:
        result['count_guard']['reason'] = 'current_batch_below_50_label_floor'
        return result
    rid = config['rollout_id']
    start = config['label_guard_start_rollout_id']
    if rid < start or rid < 1 or report['policy_id'] != f"{config['run_id']}:rollout{rid}":
        raise ValueError('Invalid continued collection identity')
    root = Path(config['label_guard_resume_from'] if rid == start else config['run_output'])
    folder = root/'iterations'/f'{rid-1:04d}'
    paths = {name: folder/(name+'.json') for name in ('calibration', 'training_gate', 'checkpoint-saved')}
    evidence = {name: json.loads(path.read_text()) for name, path in paths.items()}
    prior = evidence['calibration']
    saved = evidence['checkpoint-saved']
    previous_checkpoint = Path(saved['checkpoint'])
    if (saved['rollout_id'] != rid-1 or previous_checkpoint.name != f'iter_{rid-1:07d}'
            or previous_checkpoint.resolve() != Path(config['checkpoint']).resolve()
            or not (previous_checkpoint/'common.pt').is_file()
            or not evidence['training_gate']['ready']
            or not prior['decision']['passed'] or prior['applied_beta'] != .5
            or prior['policy_id'] != f"{config['run_id']}:rollout{rid-1}"
            or not prior.get('all_failure_experiment')):
        raise ValueError('Previous-batch support is not from the immediately preceding trained checkpoint')
    for key, default in [('beta', None), ('scored_fraction', None), ('k', None),
                         ('candidate_gate', 'distinct5'), ('credit_assignment', 'response_index')]:
        if config.get(key, default) != prior['config'].get(key, default):
            raise ValueError('Previous-batch support has a different reward recipe')
    if prior['admitted'] < 100:
        result['count_guard']['reason'] = 'previous_batch_did_not_pass_original_count'
        return result
    result['checks']['labels'] = True
    result['passed'] = all(result['checks'].values())
    result['count_guard'].update(used=True, previous_labels=prior['admitted'],
        previous_rollout_id=rid-1, previous_checkpoint=str(previous_checkpoint),
        evidence_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths.values()})
    result['rule'] += (' Count-only continuation exception: current >=50 labels and the immediately '
                       'preceding trained batch independently had >=100; all other current gates remain mandatory.')
    return result
