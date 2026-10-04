#!/usr/bin/env python3
"""Prepare missing original-bonus evaluations; never submit compute implicitly."""
import sys
sys.dont_write_bytecode = True
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
import zipfile

import run_arm_iteration80_eval as base
from audit_arm_failure_terminations import load_metadata
from resume_baseline import allocation, validate_source, write_json

REPO, RUNTIME = base.REPO, base.RUNTIME
CONTROL = RUNTIME / 'arm-turn-bonus-preparation/original-backfill-20260927'
ROOTS = {20: 'arm-turn-bonus-fresh-295764', 30: 'arm-turn-bonus-fresh-295764',
         40: 'arm-turn-bonus-fresh-297008', 50: 'arm-turn-bonus-fresh-300543',
         60: 'arm-turn-bonus-fresh-300543'}
OLD40 = RUNTIME / 'evaluations/qcq7i4ug-record-299156-after40'
LINEAGE = 'arm-turn-bonus-fresh-294197'


def router_isolated_source():
    """Freeze a separate source; never edit files used by active evaluators."""
    parent = base.prepare_source('original')
    source = RUNTIME/'reference-arm-original-backfill-router-20260927-v2'
    relative = 'slime/ray/rollout.py'
    text = (parent/relative).read_text()
    old_port = 'find_available_port(random.randint(3000, 4000))'
    old_metrics = 'router_args.prometheus_port = find_available_port(random.randint(4000, 5000))'
    if text.count(old_port) != 2 or text.count(old_metrics) != 1:
        raise ValueError('Unexpected frozen router allocation implementation')
    # Reserve the upper part of the existing per-job block for this one-model
    # evaluator. find_available_port adds random jitter and cannot be used
    # within a lease; both HTTP and Prometheus require explicit positive ports.
    new_port = ('int(os.environ["OPENWEBRL_ROLLOUT_PORT_BASE"]) + 240 '
                'if "OPENWEBRL_ROLLOUT_PORT_BASE" in os.environ else ' + old_port)
    new_metrics = ('router_args.prometheus_port = '
                   'int(os.environ["OPENWEBRL_ROLLOUT_PORT_BASE"]) + 241')
    patched = text.replace(old_port, new_port).replace(old_metrics, new_metrics)
    if not source.exists():
        shutil.copytree(parent, source, symlinks=True, copy_function=base.copy_plain,
            ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git', '.browser_use_sessions'))
        (source/relative).write_text(patched)
        manifest = read(source/'reference_manifest.json')
        manifest['recipe_files_sha256'][relative] = hashlib.sha256(patched.encode()).hexdigest()
        manifest['router_port_isolation'] = dict(parent=str(parent), model_router_offset=240,
                                                prometheus_offset=241)
        write_json(source/'reference_manifest.json', manifest)
    validate_source(source)
    parent_hashes = read(parent/'reference_manifest.json')['recipe_files_sha256']
    hashes = read(source/'reference_manifest.json')['recipe_files_sha256']
    if ((source/relative).read_text() != patched or any(hashes.get(k) != v
            for k, v in parent_hashes.items() if k != relative)):
        raise ValueError('Router recovery changed outside the isolated port allocation')
    return parent, source


def replace_source_paths(value, parent, source):
    if isinstance(value, str):
        return value.replace(str(parent), str(source))
    if isinstance(value, list):
        return [replace_source_paths(v, parent, source) for v in value]
    if isinstance(value, dict):
        return {k: replace_source_paths(v, parent, source) for k,v in value.items()}
    return value


def read(path):
    return json.loads(Path(path).read_text())


def task_rows():
    source = base.prepare_source('original')
    return [json.loads(line) for line in (source/'online_mind2web_monitor.jsonl').read_text().splitlines()]


def recovered_record(payload, archive, metrics_function):
    """Recover embedded verdicts without judging again or reading tensor storage."""
    turns = payload['turns']
    if not turns or archive.stem != hashlib.sha256(str(payload['task_id']).encode()).hexdigest():
        raise ValueError('Empty or misidentified saved trajectory')
    samples = []
    for turn in turns:
        samples.append(SimpleNamespace(remove_sample=turn.get('remove_sample', False),
            status=turn['status'], metadata=turn.get('metadata', {}), reward=turn.get('reward'),
            get_reward_value=lambda args, reward=turn.get('reward'): reward))
    terminal = max(samples, key=lambda s: s.metadata.get('turn_index', 0))
    metrics = metrics_function(SimpleNamespace(reward_key=None), [samples])
    reward = terminal.metadata.get('reward', {})
    if metrics['valid_trajectories'] and (reward.get('judge_prompt_variant') != 'action_history'
            or reward.get('judge_timeout') or reward.get('combined') != terminal.reward
            or terminal.reward not in (0, 1)):
        raise ValueError('Valid outcome lacks its preserved judge evidence')
    return dict(task_id=str(payload['task_id']), rollout_file=archive.name, turns=len(turns),
        error_type=None, metrics=metrics, judge_model='gpt-4.1',
        judge_prompt_variant='action_history', terminal_status=terminal.status,
        reward_metadata=reward, provenance=dict(original_archive=str(archive),
        original_job='299156', recovered_from_embedded_metadata=True, judge_rerun=False))


def recover40():
    manifest, status = read(OLD40/'evaluation_manifest.json'), read(OLD40/'status.json')
    expected = RUNTIME/'evaluations'/ROOTS[40]/'runtime/iter_0000039'
    restore = read(OLD40/'checkpoint_restore_evidence.json')
    if (Path(manifest['checkpoint']) != expected or not status.get('complete')
            or status.get('saved_rollouts') != 100 or restore.get('checkpoint') != str(expected)
            or manifest['environment'].get('JUDGE_MODEL') != 'gpt-4.1'):
        raise ValueError('Historical iteration40 identity or completion changed')
    spec = importlib.util.spec_from_file_location('original_metrics',
        Path(manifest['source'])/'slime/utils/trajectory_metrics.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    records = []; metadata_bytes = 0
    target = CONTROL/'retained40/rollouts'; target.mkdir(parents=True, exist_ok=True)
    for archive in sorted((OLD40/'rollouts').glob('*.pt')):
        payload, size = load_metadata(archive); metadata_bytes += size
        record = recovered_record(payload, archive, module.trajectory_metrics)
        link = target/archive.name
        if link.is_symlink() and link.resolve() != archive.resolve():
            raise ValueError('Historical archive link changed')
        if not link.exists(): link.symlink_to(archive)
        write_json(target/archive.with_suffix('.json').name, record)
        records.append(record)
    fixed = read(REPO/'openwebrl/docs/arm_c2_scaling_100.json')['task_ids']
    ids = [r['task_id'] for r in records]
    successes = sum(r['metrics']['successes'] for r in records)
    valid = sum(r['metrics']['valid_trajectories'] for r in records)
    if len(ids) != 100 or len(set(ids)) != 100 or set(ids) != set(fixed) or (successes, valid) != (31, 69):
        raise ValueError('Recovered cohort or outcomes disagree with historical result')
    result = dict(tasks=100, successes=successes, valid=valid, invalid=100-valid,
        overall=successes/100, valid_only=successes/valid, saved_rollouts=100, saved_verdicts=100,
        metadata_bytes_read=metadata_bytes, tensor_storage_read=False, new_api_calls=0,
        checkpoint=str(expected), original_evaluation=str(OLD40), rollouts=str(target),
        task_ids=sorted(ids), preserved_invalid_tasks=True)
    write_json(CONTROL/'retained40/audit.json', result)
    return result


def plan(iteration, job, attempt=0):
    root = RUNTIME/'evaluations'/ROOTS[iteration]
    original = read(root/'launch_manifest.json')
    if (original.get('wandb_run_id') != LINEAGE or original['arm_config']['beta'] != 0.5
            or original['arm_config']['scored_fraction'] != 0.2):
        raise ValueError('Wrong original-bonus training lineage')
    label = f'{job}-retry{attempt}' if attempt else job
    p = base.plan('original', label, iteration, root)
    if attempt >= 2:
        parent, source = router_isolated_source()
        p = replace_source_paths(p, parent, source)
    p['job_id'] = job
    p['parent_training_run'] = LINEAGE
    p['wandb_run_id'] = f'arm-original-iter{iteration}-backfill-{label}'
    p['environment']['WANDB_RUN_ID'] = p['wandb_run_id']
    p['environment']['PYTHONDONTWRITEBYTECODE'] = '1'
    p['startup_recovery_attempt'] = attempt
    p['backfill'] = dict(completed_iteration=iteration, reuse_tasks=100 if iteration == 40 else 0)
    if iteration == 40:
        saved = read(CONTROL/'retained40/audit.json')
        rows = task_rows(); ids = [str(row['metadata']['task_id']) for row in rows
                                  if str(row['metadata']['task_id']) not in set(saved['task_ids'])]
        if len(ids) != 200: raise ValueError('Iteration40 complement is not200 tasks')
        p.update(expected_task_count=200, expected_rollout_task_ids=ids)
        p['command'][p['command'].index('--eval-config')+1] = str(CONTROL/'remaining40.yaml')
        p['command'][p['command'].index('--wandb-group')+1] = 'arm-original-iter40-remaining200'
        p['environment']['OPENWEBRL_RECORD_EVAL_EXPECTED_TASK_COUNT'] = '200'
        p['protocol'] = 'local browser; GPT-4.1/action_history; T0; disjoint200 completing historical100'
    return base.evaluation.configure_evaluation_tracking(p, 'openwebrl-evals')


def combined_audit(directories):
    records = []
    for directory in directories:
        for f in Path(directory).glob('*.json'):
            record = read(f); archive = f.parent/record['rollout_file']
            if (record['judge_model'], record['judge_prompt_variant']) != ('gpt-4.1', 'action_history'):
                raise ValueError('Judge protocol changed')
            with zipfile.ZipFile(archive) as z:
                if len([m for m in z.infolist() if m.filename.endswith('/data.pkl')]) != 1:
                    raise ValueError('Malformed saved archive')
            records.append(record)
    ids = [r['task_id'] for r in records]
    expected = {str(row['metadata']['task_id']) for row in task_rows()}
    if len(ids) != 300 or len(set(ids)) != 300 or set(ids) != expected:
        raise ValueError('Full300 union is incomplete or overlaps')
    s = sum(r['metrics']['successes'] for r in records)
    v = sum(r['metrics']['valid_trajectories'] for r in records)
    return dict(tasks=300, successes=s, valid=v, invalid=300-v, overall=s/300,
        valid_only=s/v if v else None, saved_rollouts=300, saved_verdicts=300,
        rollout_directories=[str(d) for d in directories], cohort_verified=True,
        historical100_plus_new200=len(directories) == 2)


def prepare():
    CONTROL.mkdir(parents=True, exist_ok=True)
    saved = recover40()
    rows = [row for row in task_rows() if str(row['metadata']['task_id']) not in set(saved['task_ids'])]
    tasks = CONTROL/'remaining40.jsonl'
    tasks.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    text = (base.source_for('original')/'openwebrl/online_mind2web_monitor.yaml').read_text()
    if text.count('path: online_mind2web_monitor.jsonl') != 1:
        raise ValueError('Unexpected frozen task config')
    (CONTROL/'remaining40.yaml').write_text(text.replace('path: online_mind2web_monitor.jsonl',f'path: {tasks}'))
    entries = []
    for iteration in ROOTS:
        p = plan(iteration, f'PREPARE-original{iteration}')
        write_json(CONTROL/f'original{iteration}-plan.json', p)
        entries.append(dict(iteration=iteration, checkpoint=p['checkpoint'], new_tasks=p['expected_task_count'],
            reused_tasks=p['backfill']['reuse_tasks'], gpus=2, hours=1, cpus=16, memory_gib=480,
            submission_command=['sbatch', '--parsable', f'--job-name=arm-original{iteration}-backfill',
                f'--export=ALL,ARM_ORIGINAL_BACKFILL_ITERATION={iteration}',
                str(REPO/'scripts/evaluate_arm_original_backfill_2gpu.sbatch')]))
    proposal = dict(status='prepared_awaiting_exact_resource_approval', new_jobs=entries,
        total_max_gpu_hours=10, new_browser_tasks=1400, reused_browser_tasks=100,
        existing_owned_evaluations={'331778':[80,90],'332003':[30,40],'332005':[50,60]},
        blocked_missing_checkpoints={'failure_beta1':[30,40,50,60],'original_bonus':[90]},
        no_jobs_submitted=True)
    write_json(CONTROL/'proposal.json', proposal)
    print(json.dumps(proposal, indent=2))


def execute(iteration, job):
    approval = read(CONTROL/'approval.json')
    if not approval.get('approved') or approval.get('resources') != dict(jobs=5,gpus_per_job=2,hours_per_job=1,total_gpu_hours=10):
        raise ValueError('Exact five-job resource approval required')
    if os.environ.get('SLURM_JOB_ID') != job: raise ValueError('Wrong allocation')
    allocation(subprocess.check_output(['scontrol','show','job',job,'-o'],text=True),job,
               requested_gpus=2,maximum_hours=1)
    attempt = int(os.environ.get('SLURM_RESTART_COUNT', '0'))
    if attempt:
        budget = read(CONTROL/f'{job}-requeue-budget.json')
        if (budget['previous_elapsed_seconds'] + budget['requested_seconds'] > 3600
                or budget['attempt'] != attempt):
            raise ValueError('Requeue exceeds the original one-hour approval')
    p = plan(iteration, job, attempt)
    os.environ.update(OPENWEBRL_EXPECTED_SCHEDULER_OFFSET='0',
        OPENWEBRL_RECORD_EVAL_EXPECTED_TASK_COUNT=str(p['expected_task_count']))
    base.evaluation.run(p, REPO/'.env')
    directories = [Path(p['output'])/'rollouts']
    if iteration == 40: directories.insert(0,CONTROL/'retained40/rollouts')
    write_json(Path(p['output'])/'full300-audit.json', combined_audit(directories))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--iteration',type=int,choices=ROOTS)
    parser.add_argument('--job-id',default='PREPARE')
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if args.prepare: prepare()
    elif args.iteration is None: parser.error('--iteration is required')
    elif args.execute: execute(args.iteration,args.job_id)
    else: print(json.dumps(plan(args.iteration,args.job_id),indent=2))
