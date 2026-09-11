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


def build_plan(source, checkpoint, output, job_id, attempt=0):
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
    if attempt:
        run_id += f'-r{attempt}'
    env = {
        'NUM_GPUS': '4', 'TP_SIZE': '4', 'NUM_ROLLOUT': '0',
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
        'SLIME_ADAPTIVE_QUERY_BLACKLIST_PATH': str(source / 'reference_empty_blacklist.txt'),
        'SLIME_BROWSER_QUERY_BLACKLIST_PATH': str(source / 'reference_empty_blacklist.txt'),
    }
    command = ['bash', str(source / 'scripts/run_h200_browser.sh'),
               '--use-wandb', '--wandb-mode', 'online', '--wandb-project', 'openwebrl',
               '--wandb-team', 'zixianma', '--wandb-group', 'qcq7i4ug-checkpoint-evaluation',
               '--disable-wandb-random-suffix', '--wandb-dir', str(output / 'wandb'),
               '--sglang-disable-cuda-graph', '--eval-interval', '10',
               '--eval-config', str(source / 'openwebrl/online_mind2web_monitor.yaml'),
               '--rollout-health-check-first-wait', '180', '--use-fault-tolerance',
               '--router-balance-abs-threshold', '2', '--skip-eval-before-train',
               '--lr-decay-iters', '1', '--use-checkpoint-opt-param-scheduler',
               '--save-debug-rollout-data', str(output / 'runtime/rollout_recovery/{rollout_id}.pt')]
    return {'source': str(source), 'checkpoint': str(checkpoint), 'checkpoint_index': index,
            'completed_training_iterations': index+1, 'output': str(output), 'job_id': job_id,
            'attempt': attempt,
            'wandb_run_id': run_id, 'parent_training_run': 'qcq7i4ug',
            'protocol': 'existing deterministic GPT-4.1 Online-Mind2Web monitor, 300 tasks',
            'optimizer_updates_requested': 0, 'environment': env, 'command': command,
            'note': 'Native eval/iteration is 1 in eval-only mode; use checkpoint identity in this manifest.'}


def restore_pattern(plan):
    view = Path(plan['output']) / 'checkpoint-view'
    return r'successfully loaded checkpoint from ' + re.escape(str(view)) + r' .* at iteration ' + str(plan['checkpoint_index']) + r'\b'


def record_restore_evidence(plan):
    """Read this job's actor stdout directly; Ray may omit forwarded stdout lines."""
    output = Path(plan['output'])
    target = output / 'checkpoint_restore_evidence.json'
    if target.exists():
        return
    log = output / 'evaluation.log'
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
    if metrics.get('eval/online-mind2web-monitor/task/trajectories') != 300:
        raise ValueError('Evaluation did not cover all 300 tasks')
    write_json(output / 'metrics.json', metrics)
    write_json(output / 'status.json', {'complete': True, 'returncode': code,
                                      'checkpoint': plan['checkpoint'], 'wandb_run_id': plan['wandb_run_id']})
    return metrics


def run(plan, env_file):
    job = plan['job_id']
    if os.getenv('SLURM_JOB_ID') != job or f'/job_{job}/' not in Path('/proc/self/cgroup').read_text():
        raise ValueError('Execute inside the explicitly authorized Slurm GPU step')
    record = subprocess.check_output(['scontrol', 'show', 'job', job, '-o'], text=True)
    resources = allocation(record, job, requested_gpus=4)
    step_id = os.getenv('SLURM_STEP_ID')
    steps = subprocess.check_output(['squeue', '--steps', f'--jobs={job}', '--noheader', '--format=%i'], text=True)
    allowed = {f'{job}.{suffix}' for suffix in ['batch', 'extern', step_id]}
    if any(x.strip() not in allowed for x in steps.splitlines() if x.strip()):
        raise ValueError('Dedicated allocation required: another step is active')
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
    if resources['maximum_seconds'] < 60 * 60:
        raise ValueError('Reserve at least one hour for a full monitor evaluation')
    from dotenv import dotenv_values
    env = clean_environment()
    # Explicit evaluation options take precedence over inherited training options.
    for key, value in dotenv_values(env_file).items():
        if value and key.startswith(('WANDB_', 'JUDGE_', 'OPENAI_', 'AZURE_')):
            env.setdefault(key, value)
    if not env.get('WANDB_API_KEY'):
        raise ValueError('WANDB_API_KEY is required')
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
    inspection = source_command(source, ['python', str(REPO / 'scripts/inspect_training_checkpoint.py'),
                                str(view), '--expected-scheduler-offset-updates', '1',
                                '--report', str(output / 'checkpoint_validation.json')])
    subprocess.run(inspection, env=env, check=True, stdout=subprocess.DEVNULL)
    command = source_command(source, ['timeout', '--signal=INT', '--kill-after=120',
                                      str(resources['maximum_seconds']), *plan['command']])
    with (output / 'evaluation.log').open('w') as log:
        proc = subprocess.Popen(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT)
        while proc.poll() is None:
            record_restore_evidence(plan)
            time.sleep(5)
        code = proc.returncode
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
    args = parser.parse_args()
    plan = build_plan(args.source, args.checkpoint, args.output, args.job_id, args.attempt)
    if args.execute:
        run(plan, args.env_file)
    else:
        print(json.dumps(plan, indent=2))


if __name__ == '__main__':
    main()
