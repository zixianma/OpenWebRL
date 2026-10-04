#!/usr/bin/env python3
"""Freeze a fresh native outcome-only baseline on the private4102-task union.

Preparation and --dry-run are CPU-only; neither submits or consumes compute.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time

from prepare_task_pool_baseline import CONTROL, REPO, RUNTIME
from prepare_arm_turn_bonus import copy_plain, replace_once
from resume_baseline import validate_source, write_json

PARENT = RUNTIME/'reference-stage1-to100-20260922-24h-v3'
BROWSER_PARENT = RUNTIME/'reference-arm-mixed-async-label-20261002-v3'
BROWSER_RUNTIME_FILES = (
    'openwebrl/env/local_process_env.py',
    'openwebrl/env/browser_runtime.py',
    'openwebrl/env/web_env.py',
)
SOURCE = RUNTIME/'reference-expanded-outcome-4102-20261003-v1'
RUN = RUNTIME/'runs/outcome-only-expanded-4102-20261003'
RUN_ID = 'outcome-only-expanded-4102-20261003'
PYTHON = RUNTIME/'venv/bin/python'
MODEL = Path('/gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT')
TARGET_ITERATION = 60
EVAL_ITERATIONS = tuple(range(10, TARGET_ITERATION + 1, 10))
TRAINING_STAGES = (1, *EVAL_ITERATIONS)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment(target, *, resume=False):
    if target not in TRAINING_STAGES:
        raise ValueError('Unknown training checkpoint boundary')
    # Native scheduler derives its horizon from num_rollout. A one-iteration
    # horizon rounds 48*5/256 down to zero and cannot initialize. Keep the full
    # scientific horizon for every stage, and stop after its durable checkpoint.
    env = dict(NUM_GPUS='8', TP_SIZE='2', NUM_ROLLOUT=str(TARGET_ITERATION),
        OPENWEBRL_STOP_AFTER_SAVED_ROLLOUT=str(target-1),
        TRAIN_DATA=str(CONTROL/'combined-tasks.parquet'), HF_CHECKPOINT=str(MODEL),
        BROWSER_TRAIN_CONFIG=str(SOURCE/'openwebrl/browser_training_config.yaml'),
        BROWSER_MAX_STEPS='15', ROLLOUT_BATCH_SIZE='48', N_SAMPLES='5',
        GLOBAL_BATCH_SIZE='256', CONTEXT_LEN='32768', RESPONSE_LEN='1024',
        BROWSER_CONCURRENCY='64', SGLANG_CONCURRENCY='64', LEARNING_RATE='1e-6',
        RECOMPUTE_ACTIVATIONS='1', SAVE_INTERVAL='1', SAVE_DIR=str(RUN),
        WANDB_MODE='online', WANDB_PROJECT='openwebrl', WANDB_RUN_ID=RUN_ID,
        JUDGE_MODEL='gpt-4.1', JUDGE_API_MODE='served', JUDGE_API_BASE='https://api.openai.com/v1',
        OPENWEBRL_CUDA_CACHE_LIMIT_GIB='48', OMP_NUM_THREADS='2',
        FLASHINFER_WORKSPACE_BASE=str(RUNTIME), RAY_ADDRESS='local',
        RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES=str(8*1024**3),
        OPENWEBRL_MULTIMODAL_STORAGE_DIR=str(RUN/'multimodal'),
        OPENWEBRL_EXPANDED_BASELINE_CONTROL=str(CONTROL),
        SLIME_ADAPTIVE_QUERY_BLACKLIST_PATH=str(SOURCE/'reference_empty_blacklist.txt'),
        SLIME_BROWSER_QUERY_BLACKLIST_PATH=str(SOURCE/'reference_empty_blacklist.txt'))
    if resume:
        env['SLIME_LOAD_CHECKPOINT'] = str(RUN)
    return env


def training_command(target, *, resume=False):
    # Target is an absolute one-based endpoint. Each stage owns its subprocess
    # and saves the native dataset cursor and optimizer before evaluation.
    command = ['bash', str(SOURCE/'scripts/run_h200_browser.sh'),
        '--use-wandb', '--wandb-mode', 'online', '--wandb-project', 'openwebrl',
        '--wandb-team', 'zixianma', '--wandb-group', RUN_ID,
        '--disable-wandb-random-suffix', '--wandb-dir', str(RUN/'wandb'),
        '--sglang-disable-cuda-graph', '--rollout-health-check-first-wait', '180',
        '--use-fault-tolerance', '--router-balance-abs-threshold', '2',
        '--skip-eval-before-train', '--seed', '42',
        '--rollout-top-p', '1', '--rollout-top-k', '-1',
        '--save-debug-rollout-data', str(RUN/'rollout_recovery/{rollout_id}.pt')]
    if resume:
        command += ['--use-checkpoint-opt-param-scheduler']
    return dict(command=command, environment=environment(target, resume=resume),
                target_iteration=target, fresh_iteration0=not resume, tp=2, dp=4,
                eval_owned_by_controller=True)


def clean_env():
    env = os.environ.copy()
    for k in list(env):
        if k.startswith(('OPENWEBRL_', 'ARM_', 'SLIME_ADAPTIVE_QUERY_')) or k in {
            'WANDB_RUN_ID', 'WANDB_SERVICE', 'DRY_RUN', 'SLIME_LOAD_CHECKPOINT',
            'SLIME_CKPT_STEP', 'OVERRIDE_OPT_PARAM_SCHEDULER', 'OPENWEBRL_VERIFY_RESUME_ONLY',
            'OPENWEBRL_STOP_AFTER_SAVED_ROLLOUT', 'OPENWEBRL_PENDING_EVAL_ITERATION',
            'OPENWEBRL_EVAL_ROLLOUT_DIR', 'OPENWEBRL_RECORD_EVAL_CONFIG',
            'OPENWEBRL_RECORD_EVAL_EXPECTED_TASK_COUNT'}:
            env.pop(k, None)
    return env


def align_browser_concurrency():
    # This config is loaded once when a worker starts. Updating it while the
    # first stage collects does not alter that stage; the next worker reads64.
    name = 'openwebrl/browser_training_config.yaml'
    parent = (PARENT/name).read_text()
    assert parent.count('browser_rollout_concurrency: 32') == 1
    desired = parent.replace('browser_rollout_concurrency: 32', 'browser_rollout_concurrency: 64', 1)
    current = (SOURCE/name).read_text()
    if current not in (parent, desired):
        raise ValueError('Unexpected scientific config change; do not overwrite')
    manifest = json.loads((SOURCE/'reference_manifest.json').read_text())
    if current != desired:
        history = CONTROL/'browser-concurrency-correction'
        history.mkdir(exist_ok=True)
        for path in [SOURCE/name, SOURCE/'reference_manifest.json']:
            old = history/path.name
            if not old.exists():
                copy_plain(path, old)
        (SOURCE/name).write_text(desired)
        manifest['recipe_files_sha256'][name] = digest(SOURCE/name)
        manifest['expanded4102']['changed_files'].append(name)
        manifest['expanded4102']['runtime_changes'].append('Explicit browser config gate64; matches approved64-slot pool')
        write_json(SOURCE/'reference_manifest.json', manifest)
    assert manifest['recipe_files_sha256'][name] == digest(SOURCE/name)


def freeze():
    validate_source(PARENT)
    if not SOURCE.exists():
        shutil.copytree(PARENT, SOURCE, symlinks=True, copy_function=copy_plain,
            ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git', '.browser_use_sessions'))
        changed = []
        for name in BROWSER_RUNTIME_FILES:
            copy_plain(BROWSER_PARENT/name, SOURCE/name)
            changed.append(name)
        name = 'openwebrl/expanded_baseline_budget.py'
        copy_plain(REPO/name, SOURCE/name)
        changed.append(name)
        name = 'openwebrl/reward_browser.py'
        text = (SOURCE/name).read_text()
        text = replace_once(text, '        return AsyncOpenAI(api_key=api_key, base_url=base_url)',
            '        from openwebrl.expanded_baseline_budget import wrap\n'
            '        return wrap(AsyncOpenAI(api_key=api_key, base_url=base_url, max_retries=0))')
        (SOURCE/name).write_text(text)
        changed.append(name)
        manifest = json.loads((SOURCE/'reference_manifest.json').read_text())
        manifest['recipe_files_sha256'].update({n:digest(SOURCE/n) for n in changed})
        manifest['expanded4102'] = dict(parent=str(PARENT), browser_parent=str(BROWSER_PARENT),
            changed_files=changed, initialization='fresh original SFT0',
            scientific_change='training pool only',
            runtime_changes=['nonblocking browser startup/log I/O', 'shared judge-spend ledger; native prompts/output settings/parser preserved'])
        write_json(SOURCE/'reference_manifest.json', manifest)
    align_browser_concurrency()
    validate_source(SOURCE)
    protected = ['slime/rollout/data_source.py', 'slime/rollout/filter_hub/dynamic_sampling_filters.py',
        'slime/ray/rollout.py', 'slime/backends/megatron_utils/loss.py', 'openwebrl/generate_browser.py']
    assert all(digest(PARENT/n)==digest(SOURCE/n) for n in protected)
    assert (SOURCE/'openwebrl/expanded_baseline_budget.py').read_bytes()==(REPO/'openwebrl/expanded_baseline_budget.py').read_bytes()
    for name in BROWSER_RUNTIME_FILES:
        assert (SOURCE/name).read_bytes()==(BROWSER_PARENT/name).read_bytes()
    for folder in ['openwebrl','slime','slime_plugins','scripts']:
        for p in (SOURCE/folder).rglob('*.py'):
            if not p.is_symlink() and 'data' not in p.relative_to(SOURCE).parts:
                compile(p.read_bytes(), str(p), 'exec')
    return protected


def prepare():
    protected = freeze()
    data = json.loads((CONTROL/'training-plan.json').read_text())
    assert digest(CONTROL/'combined-tasks.parquet') == data['data']['sha256']
    assert MODEL.is_dir()
    stages = [training_command(i, resume=i > 1) for i in TRAINING_STAGES]
    native = []
    for stage in stages:
        env = clean_env();env.update(stage['environment']);env['DRY_RUN']='1'
        command = subprocess.check_output(stage['command'], env=env, cwd=SOURCE, text=True)
        argv = shlex.split(command)
        def value(k):
            assert argv.count(k)==1,(k,argv.count(k))
            return argv[argv.index(k)+1]
        assert value('--tensor-model-parallel-size')=='2'
        assert value('--actor-num-gpus-per-node')=='8'
        assert value('--prompt-data')==str(CONTROL/'combined-tasks.parquet')
        assert value('--custom-config-path')==str(SOURCE/'openwebrl/browser_training_config.yaml')
        assert value('--num-rollout')==str(TARGET_ITERATION)
        assert stage['environment']['OPENWEBRL_STOP_AFTER_SAVED_ROLLOUT']==str(stage['target_iteration']-1)
        assert value('--load')==str(MODEL if stage['fresh_iteration0'] else RUN)
        assert value('--custom-rm-path')=='openwebrl.reward_browser.reward_func'
        assert value('--custom-generate-function-path')=='openwebrl.generate_browser.generate_turn_sample'
        assert value('--dynamic-sampling-filter-path').endswith('.check_reward_nonempty_nonzero_std')
        assert value('--global-batch-size')=='256' and value('--micro-batch-size')=='1'
        assert value('--lr')=='1e-6' and value('--ppo-epochs')=='2'
        assert value('--judge-prompt-variant')=='action_history'
        assert '--eval-interval' not in argv and not any('arm_' in x for x in argv)
        native.append(argv)
    # Require each scheduled evaluation to have its own immutable output root.
    evals = {i:str(RUNTIME/f'evaluations/{RUN_ID}-iter{i}') for i in EVAL_ITERATIONS}
    approval = json.loads((CONTROL/'approval-request.json').read_text())
    report = dict(epoch=time.time(),source=str(SOURCE),run_directory=str(RUN),
        wandb_id=RUN_ID,training_stages=stages,native_argv=native,evaluation_roots=evals,
        same_baseline_files=protected,data_sha256=data['data']['sha256'],
        target_iteration=TARGET_ITERATION,approval_required=not approval.get('approved',False),
        submitted=(CONTROL/'attempts.json').exists(),source_compilation_passed=True,
        native_launcher_dry_runs_passed=True,gpu_startup_validation_pending=True)
    report['frozen_file_hashes'] = {
        str(p.relative_to(SOURCE)):digest(p)
        for p in SOURCE.rglob('*')
        if p.is_file() and not p.is_symlink() and p.suffix in {'.py','.sh','.yaml','.jsonl'}
        and not any(part in {'data','docs','__pycache__'} for part in p.relative_to(SOURCE).parts)}
    write_json(CONTROL/'launcher-preparation.json', report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['training_stages','native_argv']},indent=2))


if __name__ == '__main__':
    prepare()
