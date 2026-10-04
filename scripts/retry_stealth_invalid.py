#!/usr/bin/env python3
"""One isolated retry of invalid/missing stealth tasks; never submits compute."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from evaluate_baseline_checkpoint import build_plan, configure_evaluation_tracking, run, RUNTIME
from resume_baseline import validate_source, write_json


def is_valid(record):
    reward = record.get('reward') or {}
    return (not record.get('remove_sample') and not record.get('judge_invalid')
            and record.get('status') != 'aborted' and reward.get('judge') is not None)


def read_records(directory):
    records = {}
    for path in sorted((directory / 'completed_tasks').glob('*.json')):
        record = json.loads(path.read_text())
        task_id = record['task_id']
        if task_id in records:
            raise ValueError('Duplicate task identity in completed records')
        records[task_id] = record
    return records


def select_retry(tasks, records):
    ids = [task['metadata']['task_id'] for task in tasks]
    if len(ids) != len(set(ids)) or not set(records) <= set(ids):
        raise ValueError('Task identities are duplicated or outside the benchmark')
    return [task for task in tasks if task['metadata']['task_id'] not in records
            or not is_valid(records[task['metadata']['task_id']])]


def prepare(original, source, target):
    original, source, target = map(lambda p: p.resolve(), (original, source, target))
    validate_source(source)
    if target.exists() or not target.is_relative_to(RUNTIME) or target.is_relative_to(source):
        raise ValueError('Use a new isolated source under runtime storage')
    tasks = [json.loads(line) for line in (source / 'online_mind2web_monitor.jsonl').read_text().splitlines() if line]
    records = read_records(original)
    subset = select_retry(tasks, records)
    if len(tasks) != 300 or not 1 <= len(subset) <= 10:
        raise ValueError('This small retry profile expects at most ten unresolved tasks out of 300')
    origin_manifest = json.loads((original / 'evaluation_manifest.json').read_text())
    if Path(origin_manifest['source']).resolve() != source:
        raise ValueError('Retry must use the original frozen source')
    shutil.copytree(source, target, symlinks=True,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git', '.env*', '.browser_use_sessions'))
    dataset = target / 'online_mind2web_monitor.jsonl'
    if dataset.is_symlink():
        raise ValueError('Refuse to replace a symlinked benchmark dataset')
    dataset.write_text(''.join(json.dumps(task, ensure_ascii=False) + '\n' for task in subset))
    manifest = json.loads((target / 'reference_manifest.json').read_text())
    manifest['recipe_files_sha256']['online_mind2web_monitor.jsonl'] = hashlib.sha256(dataset.read_bytes()).hexdigest()
    manifest['invalid_task_retry'] = {
        'original_output': str(original), 'original_source': str(source),
        'checkpoint': origin_manifest['checkpoint'], 'original_planned': 300,
        'original_completed': len(records), 'original_valid': sum(map(is_valid, records.values())),
        'missing_ids': [t['metadata']['task_id'] for t in subset if t['metadata']['task_id'] not in records],
        'expected_task_ids': [t['metadata']['task_id'] for t in subset],
        'selection': 'Original invalid or missing only; one new attempt each; retain valid failures and successes',
        'changed_setting': 'Evaluation task subset only',
    }
    write_json(target / 'reference_manifest.json', manifest)
    validate_source(target)
    return manifest['invalid_task_retry']


def retry_plan(source, output, job_id):
    provenance = json.loads((source / 'reference_manifest.json').read_text())['invalid_task_retry']
    plan = build_plan(source, Path(provenance['checkpoint']), output, job_id,
                      browser_env='browser-use', gpus=2, protocol='benchmark')
    plan.update(expected_task_count=len(provenance['expected_task_ids']),
                expected_task_ids=provenance['expected_task_ids'], retry_provenance=provenance,
                wandb_run_id=f'qcq7i4ug-after58-invalid-retry-{job_id}')
    plan['environment']['WANDB_RUN_ID'] = plan['wandb_run_id']
    plan['protocol'] = 'After58 invalid/missing subset; one retry; unchanged stealth/o4-mini/AgentTrek, actor T=0.6'
    return configure_evaluation_tracking(plan)


def merge_records(original, retry, expected_ids, planned=300):
    if set(retry) != set(expected_ids):
        raise ValueError('Retry results do not match the exact selected tasks')
    if any(task_id in original and is_valid(original[task_id]) for task_id in retry):
        raise ValueError('Refuse to replace an originally valid task')
    merged = {**original, **retry}
    if len(merged) != planned:
        raise ValueError('Merged result does not cover the planned benchmark')
    valid = [r for r in merged.values() if is_valid(r)]
    successes = sum((r.get('reward') or {}).get('judge') == 1 for r in valid)
    return {'completed': len(merged), 'planned': planned, 'valid': len(valid),
            'invalid': len(merged) - len(valid), 'successes': successes,
            'overall_pct': successes / planned * 100,
            'valid_only_pct': successes / len(valid) * 100 if valid else None,
            'retry_tasks': len(retry), 'retry_valid': sum(map(is_valid, retry.values())),
            'retained_original_valid': sum(map(is_valid, original.values())),
            'protocol_label': 'Original valid attempts plus one retry of invalid/missing tasks'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--prepare-from', type=Path)
    p.add_argument('--job-id', default='PENDING_APPROVAL')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--env-file', type=Path, default=Path(__file__).resolve().parents[1] / '.env')
    a = p.parse_args()
    if a.prepare_from:
        if a.execute:
            p.error('Preparation cannot execute a job')
        print(json.dumps(prepare(a.prepare_from, a.source, a.output), indent=2))
    else:
        plan = retry_plan(a.source, a.output, a.job_id)
        if a.execute:
            run(plan, a.env_file)
            provenance = plan['retry_provenance']
            merged = merge_records(read_records(Path(provenance['original_output'])),
                                   read_records(a.output), plan['expected_task_ids'])
            write_json(a.output / 'merged_metrics.json', merged)
            print(json.dumps(merged, indent=2))
        else:
            print(json.dumps(plan, indent=2))
