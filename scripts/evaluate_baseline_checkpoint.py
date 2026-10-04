#!/usr/bin/env python3
"""Evaluate a selected baseline checkpoint in an existing, dedicated allocation.

Dry run is the default. Never submits compute or changes the training pointer.
Use the same preserved 300-task monitor as the online baseline, in a separate
W&B run. The baseline's eval-only mode executes zero optimizer updates.
"""
import argparse
import ast
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from resume_baseline import allocation, clean_environment, source_command, validate_source, write_json

REPO = Path(__file__).resolve().parents[1]
RUNTIME = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime')
DEFAULT_EVAL_PROJECT = 'openwebrl-evals'


def configure_evaluation_tracking(plan, project=DEFAULT_EVAL_PROJECT):
    """Keep separate evaluation runs out of the training-only W&B project.

    Some long-lived controllers still request the training project for evals
    inside their allocation. Normalize that legacy request here and again at
    the worker entry point, without interrupting the controller or training.
    """
    if not isinstance(project, str) or not project.strip() or '/' in project:
        raise ValueError('Specify a nonempty W&B project name without slashes')
    project = project.strip()
    if project.casefold() == 'openwebrl':
        plan['wandb_project_redirected_from'] = project
        project = DEFAULT_EVAL_PROJECT
    command = plan['command']
    command[command.index('--wandb-project') + 1] = project
    plan['environment']['WANDB_PROJECT'] = project
    plan['wandb_project'] = project
    entity = command[command.index('--wandb-team') + 1] if '--wandb-team' in command else 'zixianma'
    plan['wandb_url'] = f'https://wandb.ai/{entity}/{project}/runs/{plan["wandb_run_id"]}'
    return plan


def build_plan(source, checkpoint, output, job_id, attempt=0, browser_env='local_process', gpus=4,
               protocol='monitor'):
    if protocol not in ('monitor', 'benchmark'):
        raise ValueError('Unknown evaluation protocol')
    if gpus not in (1, 2, 4, 8):
        raise ValueError('Supported evaluation profiles use 1, 2, 4 or 8 H200 GPUs')
    if gpus == 1:
        manifest = json.loads((Path(source) / 'reference_manifest.json').read_text())
        profile = manifest.get('single_gpu_evaluation', {})
        if not isinstance(profile, dict) or profile.get('sequence_parallel') is not False:
            raise ValueError('TP1 requires an isolated source with sequence parallelism disabled')
    source, checkpoint, output = map(lambda p: Path(p).resolve(), (source, checkpoint, output))
    validate_source(source)
    match = re.fullmatch(r'iter_(\d{7})', checkpoint.name)
    if not match:
        raise ValueError('Specify an iter_NNNNNNN checkpoint directory')
    index = int(match[1])
    for path in [checkpoint / 'common.pt', checkpoint / '.metadata',
                 checkpoint.parent / 'rollout' / f'global_dataset_state_dict_{index}.pt']:
        if not path.is_file():
            raise ValueError(f'Missing checkpoint component: {path}')
    if not list(checkpoint.glob('*.distcp')):
        raise ValueError('Checkpoint has no distributed tensor shards')
    if output.exists():
        raise ValueError('Use a new output directory; existing evaluations are preserved')
    if not output.is_relative_to(RUNTIME / 'evaluations'):
        raise ValueError('Keep evaluations under the runtime evaluations directory')
    run_id = f'qcq7i4ug-eval-after{index+1}-{job_id}'
    if browser_env == 'browser-use':
        manifest = json.loads((source / 'reference_manifest.json').read_text())
        if not manifest.get('browser_use_evaluation'):
            raise ValueError('Prepare an isolated Browser Use evaluation source first')
        run_id = f'qcq7i4ug-eval-browseruse-after{index+1}-{job_id}'
    elif browser_env != 'local_process':
        raise ValueError('Unsupported evaluation browser backend')
    if attempt:
        run_id += f'-r{attempt}'
    env = {
        'NUM_GPUS': str(gpus), 'TP_SIZE': str(gpus), 'NUM_ROLLOUT': '0',
        'BROWSER_MAX_STEPS': '15', 'ROLLOUT_BATCH_SIZE': '48', 'N_SAMPLES': '5',
        'GLOBAL_BATCH_SIZE': '256', 'CONTEXT_LEN': '32768', 'RESPONSE_LEN': '1024',
        'BROWSER_CONCURRENCY': '32', 'SGLANG_CONCURRENCY': '48',
        'LEARNING_RATE': '1e-6', 'RECOMPUTE_ACTIVATIONS': '1', 'SAVE_INTERVAL': '1',
        'SAVE_DIR': str(output / 'runtime'), 'SLIME_LOAD_CHECKPOINT': str(output / 'checkpoint-view'),
        'SLIME_CKPT_STEP': str(index), 'WANDB_MODE': 'online', 'WANDB_RUN_ID': run_id,
        'JUDGE_MODEL': 'gpt-4.1', 'OMP_NUM_THREADS': '2',
        'RAY_ADDRESS': 'local',
        'RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES': str(8 * 1024**3),
        'OPENWEBRL_MULTIMODAL_STORAGE_DIR': f'/tmp/openwebrl-eval-{job_id}-{index}-multimodal',
        'OPENWEBRL_EVAL_ROLLOUT_DIR': str(output / 'rollouts'),
        'SLIME_ADAPTIVE_QUERY_BLACKLIST_PATH': str(source / 'reference_empty_blacklist.txt'),
        'SLIME_BROWSER_QUERY_BLACKLIST_PATH': str(source / 'reference_empty_blacklist.txt'),
    }
    if browser_env == 'browser-use':
        env.update(SLIME_BROWSER_ENV_MODE='browser-use',
                   OPENWEBRL_BROWSER_USE_SESSION_DIR=str(output / 'browser_sessions'),
                   PYTHONPATH=str(RUNTIME / 'browser-use-sdk-3.11.3'))
    eval_config = os.environ.get('OPENWEBRL_RECORD_EVAL_CONFIG', str(source / 'openwebrl/online_mind2web_monitor.yaml'))
    command = ['bash', str(source / 'scripts/run_h200_browser.sh'),
               '--use-wandb', '--wandb-mode', 'online', '--wandb-project', 'openwebrl',
               '--wandb-team', 'zixianma', '--wandb-group', 'qcq7i4ug-checkpoint-evaluation',
               '--disable-wandb-random-suffix', '--wandb-dir', str(output / 'wandb'),
               '--sglang-disable-cuda-graph', '--eval-interval', '10',
               '--eval-config', eval_config,
               '--rollout-health-check-first-wait', '180', '--use-fault-tolerance',
               '--router-balance-abs-threshold', '2', '--skip-eval-before-train',
               '--lr-decay-iters', '1', '--use-checkpoint-opt-param-scheduler',
               '--save-debug-rollout-data', str(output / 'runtime/rollout_recovery/{rollout_id}.pt')]
    if protocol == 'benchmark':
        manifest = json.loads((source / 'reference_manifest.json').read_text())
        if not manifest.get('paper_om2w_benchmark') or browser_env != 'browser-use':
            raise ValueError('Benchmark requires an isolated, prepared stealth-browser source')
        run_id = f'qcq7i4ug-benchmark-after{index+1}-{job_id}' + (f'-r{attempt}' if attempt else '')
        env.update(JUDGE_MODEL='o4-mini', WANDB_RUN_ID=run_id,
                   OPENWEBRL_BENCHMARK_RESULTS_DIR=str(output / 'completed_tasks'))
        command[command.index('--eval-config') + 1] = str(source / 'openwebrl/online_mind2web_benchmark.yaml')
        command[command.index('--wandb-group') + 1] = 'qcq7i4ug-paper-benchmark'
    return {'source': str(source), 'checkpoint': str(checkpoint), 'checkpoint_index': index,
            'completed_training_iterations': index+1, 'output': str(output), 'job_id': job_id,
            'attempt': attempt,
            'browser_env': browser_env,
            'gpus': gpus,
            'wandb_run_id': run_id, 'parent_training_run': 'qcq7i4ug',
            'metric_prefix': 'eval/online-mind2web-' + protocol,
            'protocol': ('paper o4-mini/AgentTrek OM2W; T=0.6 p=0.95 k=20; Browser Use stealth, 300 tasks'
                         if protocol == 'benchmark' else
                         'existing deterministic GPT-4.1 Online-Mind2Web monitor, 300 tasks'
                         + ('; browser=browser-use, proxy=disabled' if browser_env == 'browser-use' else '')),
            'optimizer_updates_requested': 0, 'environment': env, 'command': command,
            'note': 'Native eval/iteration is 1 in eval-only mode; use checkpoint identity in this manifest.'}


def restore_pattern(plan):
    view = Path(plan['output']) / 'checkpoint-view'
    return r'successfully loaded checkpoint from ' + re.escape(str(view)) + r' .* at iteration ' + str(plan['checkpoint_index']) + r'\b'


def record_restore_evidence(plan, log_filename='evaluation.log'):
    """Read this job's actor stdout directly; Ray may omit forwarded stdout lines."""
    output = Path(plan['output'])
    target = output / 'checkpoint_restore_evidence.json'
    if target.exists():
        return
    log = output / log_filename
    if not log.exists():
        return
    text = log.read_text(errors='replace')
    for pid in set(re.findall(r'MegatronTrainRayActor pid=(\d+)', text)):
        try:
            if f"/job_{plan['job_id']}/" not in Path(f'/proc/{pid}/cgroup').read_text():
                continue
            raw = Path(f'/proc/{pid}/fd/1').resolve()
            if not raw.is_file():
                continue
            match = re.search(restore_pattern(plan), raw.read_text(errors='replace'))
            if match:
                write_json(target, {'checkpoint': plan['checkpoint'], 'job_id': plan['job_id'],
                                    'actor_pid': int(pid), 'worker_log': str(raw), 'restore_line': match[0]})
                return
        except (OSError, ProcessLookupError):
            continue


def finalize_evaluation(plan, code):
    output = Path(plan['output'])
    text = (output / 'evaluation.log').read_text(errors='replace')
    rows = re.findall(r'rollout.py:\d+ - eval 0: (\{.*\})', text)
    if code != 0 or len(rows) != 1 or re.search(r'train_one_step start|\[TrainMetrics\]', text):
        write_json(output / 'status.json', {'complete': False, 'returncode': code, 'eval_rows': len(rows)})
        raise RuntimeError('Evaluation failed or incomplete; inspect its log before continuing')
    restored = re.search(restore_pattern(plan), text)
    receipt = output / 'checkpoint_restore_evidence.json'
    if not restored and receipt.exists():
        evidence = json.loads(receipt.read_text())
        if evidence.get('checkpoint') == plan['checkpoint'] and evidence.get('job_id') == plan['job_id']:
            restored = re.search(restore_pattern(plan), evidence.get('restore_line', ''))
    if not restored:
        raise ValueError('Missing evidence that the selected checkpoint was restored on GPU')
    metrics = ast.literal_eval(rows[0])
    expected = plan.get(
        'expected_task_count',
        int(os.environ.get('OPENWEBRL_RECORD_EVAL_EXPECTED_TASK_COUNT', '300')),
    )
    metric_count = metrics.get(plan.get('metric_prefix', 'eval/online-mind2web-monitor') + '/task/trajectories')
    saved_rollouts = len(list((output / 'rollouts').glob('*.pt')))
    if plan.get('require_task_rollouts'):
        records = [json.loads(path.read_text()) for path in (output / 'rollouts').glob('*.json')]
        ids = [record['task_id'] for record in records]
        expected_ids = plan['expected_rollout_task_ids']
        if (saved_rollouts != expected or len(ids) != expected or len(set(ids)) != expected
                or set(ids) != set(expected_ids)
                or any(not (output / 'rollouts' / record['rollout_file']).is_file() for record in records)):
            raise ValueError('Per-task rollout/verdict persistence is incomplete; preserve artifacts for recovery')
    # The task metric can be absent when the Ray actor exits during final
    # logging.  A complete set of durable per-task rollout files is stronger
    # evidence and allows judge-only replay, so accept it as the fallback.
    if metric_count != expected and saved_rollouts != expected:
        raise ValueError(f'Evaluation did not cover all {expected} tasks (metric={metric_count}, rollouts={saved_rollouts})')
    if plan.get('expected_task_ids') is not None:
        records = [json.loads(p.read_text()) for p in (output / 'completed_tasks').glob('*.json')]
        ids = [r['task_id'] for r in records]
        if len(ids) != expected or len(set(ids)) != expected or set(ids) != set(plan['expected_task_ids']):
            raise ValueError('Evaluation task identities differ from the planned subset')
    write_json(output / 'metrics.json', metrics)
    write_json(output / 'status.json', {'complete': True, 'returncode': code,
                                      'checkpoint': plan['checkpoint'], 'wandb_run_id': plan['wandb_run_id'],
                                      'expected_tasks': expected, 'saved_rollouts': saved_rollouts})
    return metrics


def wait_for_exclusive_step(job, step_id, timeout_seconds=0):
    """Let short observer steps finish without weakening allocation isolation."""
    timeout_seconds = float(timeout_seconds)
    if not 0 <= timeout_seconds <= 300:
        raise ValueError('Exclusive-step wait must be between 0 and 300 seconds')
    allowed = {f'{job}.{suffix}' for suffix in ['batch', 'extern', step_id]}
    deadline = time.monotonic() + timeout_seconds
    previous = None
    while True:
        steps = subprocess.check_output(
            ['squeue', '--steps', f'--jobs={job}', '--noheader', '--format=%i'], text=True)
        conflicts = sorted({x.strip() for x in steps.splitlines() if x.strip()} - allowed)
        if not conflicts:
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ValueError('Dedicated allocation required: active steps ' + ', '.join(conflicts))
        if conflicts != previous:
            print(f'Waiting for allocation steps to finish: {conflicts}', flush=True)
            previous = conflicts
        time.sleep(min(5, remaining))


def run(plan, env_file):
    configure_evaluation_tracking(plan, plan.get('wandb_project', DEFAULT_EVAL_PROJECT))
    job = plan['job_id']
    if os.getenv('SLURM_JOB_ID') != job or f'/job_{job}/' not in Path('/proc/self/cgroup').read_text():
        raise ValueError('Execute inside the explicitly authorized Slurm GPU step')
    wait_for_exclusive_step(job, os.getenv('SLURM_STEP_ID'),
                            plan.get('exclusive_step_wait_seconds', 0))
    # Query remaining allocation time after the wait so it consumes the budget.
    record = subprocess.check_output(['scontrol', 'show', 'job', job, '-o'], text=True)
    resources = allocation(record, job, requested_gpus=plan.get('gpus', 4),
                           maximum_hours=plan.get('allocation_hours_cap', 8))
    # A completed evaluation can be finalized after a bookkeeping failure.
    # On a supervised retry, reuse that verified result instead of rerunning tasks.
    if plan.get('attempt', 0):
        output = Path(plan['output'])
        previous = output.with_name(output.name.rsplit('-retry', 1)[0])
        status_path = previous / 'status.json'
        if status_path.exists() and json.loads(status_path.read_text()).get('complete'):
            prior = json.loads((previous / 'evaluation_manifest.json').read_text())
            if any(prior[k] != plan[k] for k in ['checkpoint', 'source', 'protocol', 'job_id']):
                raise ValueError('Completed evaluation does not match retry identity')
            metrics = finalize_evaluation(prior, 0)
            output.mkdir(parents=True, exist_ok=False)
            write_json(output / 'status.json', {'complete': True, 'reused_complete': str(previous),
                                              'wandb_run_id': prior['wandb_run_id'], 'checkpoint': prior['checkpoint']})
            print(json.dumps({'reused_complete': str(previous), 'metrics': metrics}))
            return
    # Keep a ten-minute finalization margin. A fixed-100 evaluation is
    # explicitly sized for one hour; the full-300 profile retains the longer
    # reserve because model startup and browser cleanup are more variable.
    expected_task_count = int(os.environ.get('OPENWEBRL_RECORD_EVAL_EXPECTED_TASK_COUNT', '300'))
    # The allocation-level wrapper may not preserve the task-count export on
    # supervised retries; use the conservative 20-minute floor for the
    # explicitly bounded evaluation profiles.
    minimum_seconds = 1200
    if resources['maximum_seconds'] < minimum_seconds:
        raise ValueError(f'Reserve at least {minimum_seconds // 60} usable minutes for this evaluation')
    from dotenv import dotenv_values
    env = clean_environment()
    # Explicit evaluation options take precedence over inherited training options.
    for key, value in dotenv_values(env_file).items():
        if value and key.startswith(('WANDB_', 'JUDGE_', 'OPENAI_', 'AZURE_', 'BROWSER_USE_')):
            env.setdefault(key, value)
    if not env.get('WANDB_API_KEY'):
        raise ValueError('WANDB_API_KEY is required')
    if plan.get('browser_env') == 'browser-use' and not env.get('BROWSER_USE_API_KEY'):
        raise ValueError('BROWSER_USE_API_KEY is required')
    if env.get('OPENAI_API_KEY') and not env.get('JUDGE_API_BASE'):
        env.update(JUDGE_API_MODE='served', JUDGE_API_BASE='https://api.openai.com/v1')
    env.update(plan['environment'])
    output, checkpoint, source = map(Path, [plan['output'], plan['checkpoint'], plan['source']])
    output.mkdir(parents=True, exist_ok=False)
    view = output / 'checkpoint-view'
    view.mkdir()
    (view / checkpoint.name).symlink_to(checkpoint, target_is_directory=True)
    (view / 'rollout').symlink_to(checkpoint.parent / 'rollout', target_is_directory=True)
    (view / 'latest_checkpointed_iteration.txt').write_text(str(plan['checkpoint_index'])+'\n')
    write_json(output / 'evaluation_manifest.json', plan)
    expected_scheduler_offset = os.environ.get('OPENWEBRL_EXPECTED_SCHEDULER_OFFSET', '1')
    if expected_scheduler_offset not in {'0', '1'}:
        raise ValueError('OPENWEBRL_EXPECTED_SCHEDULER_OFFSET must be 0 or 1')
    inspection = source_command(source, ['python', str(REPO / 'scripts/inspect_training_checkpoint.py'),
                                str(view), '--expected-scheduler-offset-updates', expected_scheduler_offset,
                                '--report', str(output / 'checkpoint_validation.json')])
    subprocess.run(inspection, env=env, check=True, stdout=subprocess.DEVNULL)
    command = source_command(source, ['timeout', '--signal=INT', '--kill-after=120',
                                      str(resources['maximum_seconds']), *plan['command']])
    port_lease = None
    if json.loads((source/'reference_manifest.json').read_text()).get('runtime_port_leases'):
        from runtime_ports import PORT_ENV, lease_ports
        port_lease, base = lease_ports(job)
        env[PORT_ENV] = str(base)
        write_json(output/'port-lease.json', dict(base=base, size=256, job_id=job))
    try:
        with (output / 'evaluation.log').open('w') as log:
            proc = subprocess.Popen(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT)
            while proc.poll() is None:
                record_restore_evidence(plan)
                time.sleep(5)
            code = proc.returncode
    finally:
        if port_lease is not None:
            port_lease.close()
    write_json(output / 'launcher_exit.json', {'returncode': code})
    metrics = finalize_evaluation(plan, code)
    print(json.dumps({'output': str(output), 'wandb_run_id': plan['wandb_run_id'], 'metrics': metrics}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--job-id', default='APPROVED_JOB')
    parser.add_argument('--env-file', type=Path, default=REPO / '.env')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--attempt', type=int, default=0)
    parser.add_argument('--browser-env', choices=['local_process', 'browser-use'], default='local_process')
    parser.add_argument('--gpus', choices=[1, 2, 4, 8], type=int, default=4)
    parser.add_argument('--protocol', choices=['monitor', 'benchmark'], default='monitor')
    parser.add_argument('--wandb-project', default=DEFAULT_EVAL_PROJECT)
    args = parser.parse_args()
    plan = build_plan(args.source, args.checkpoint, args.output, args.job_id, args.attempt, args.browser_env, args.gpus, args.protocol)
    configure_evaluation_tracking(plan, args.wandb_project)
    if args.execute:
        run(plan, args.env_file)
    else:
        print(json.dumps(plan, indent=2))


if __name__ == '__main__':
    main()
