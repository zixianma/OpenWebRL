#!/usr/bin/env python3
"""Step-0 pass@K on training tasks: 16 rollouts per task from the OpenWebRL SFT checkpoint.

Approved 2026-10-24: one 8 H200 / 64 CPU / 960 GiB job for at most 4 h including retries,
GPT-4.1 judge cap $40. Rollout-only (no optimizer): the G8 training recipe's sampling
(T0.8, 15 browser steps, 1024 response / 32K context tokens, GPT-4.1 action_history judge,
local browsers at 64 concurrency) on a fixed seed-0 sample of 128 tasks from the original
2,102-task pool. A keep-all group filter records every rollout, including invalid ones.

--prepare (CPU) builds the control; --execute runs in the allocation (holds on failure);
--analyze computes unbiased pass@k and the mixed-group probability for k = 1, 5, 8, 16.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/'scripts'))
from resume_baseline import source_command, write_json  # noqa: E402

RUNTIME = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime')
CONTROL = RUNTIME/'training-controls/passk-step0-20261024'
PARENT_SOURCE = RUNTIME/'training-controls/uniform8-original2102-20261006/to90-20261024/source'
SOURCE = CONTROL/'source'
POOL = RUNTIME/'training-controls/uniform8-original2102-20261006/combined-tasks.parquet'
MODEL = Path('/gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT')
TASKS, SAMPLES, SEED = 128, 16, 0
CAPS = dict(judge_cap_usd=40, judge_call_cap=2500, browser_dispatch_cap=2600, browser_concurrency=64)
OUT = CONTROL/'collection'
HOLD_IDLE_SECONDS, HOLD_TOTAL_IDLE_SECONDS = 3600, 7200
KEEP_ALL = '''from slime.rollout.filter_hub.base_types import DynamicFilterOutput


def keep_all(args, samples, **kwargs):
    """Step-0 pass@K: record every group, including all-invalid ones."""
    return DynamicFilterOutput(keep=True)
'''


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def prepare():
    CONTROL.mkdir(parents=True, exist_ok=True)
    if not SOURCE.exists():
        shutil.copytree(PARENT_SOURCE, SOURCE, symlinks=True, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git'))
        f = SOURCE/'openwebrl/rollout_sampling_budget.py'
        old = ('EXPECTED_CAPS = dict(judge_cap_usd=300, judge_call_cap=30000,\n'
               '                     browser_dispatch_cap=40000, browser_concurrency=64)')
        text = f.read_text()
        if text.count(old) != 1:
            raise ValueError('Unexpected budget module')
        f.write_text(text.replace(old, 'EXPECTED_CAPS = dict(judge_cap_usd=40, judge_call_cap=2500,\n'
                                       '                     browser_dispatch_cap=2600, browser_concurrency=64)'))
        (SOURCE/'slime/rollout/filter_hub/keep_all.py').write_text(KEEP_ALL)
        manifest = read(SOURCE/'reference_manifest.json')
        for rel in ('openwebrl/rollout_sampling_budget.py', 'slime/rollout/filter_hub/keep_all.py'):
            manifest['recipe_files_sha256'][rel] = sha(SOURCE/rel)
        manifest['passk_step0'] = dict(parent=str(PARENT_SOURCE), scope='pass@K caps and keep-all filter; no recipe change')
        write_json(SOURCE/'reference_manifest.json', manifest)
    import pandas as pd
    tasks = CONTROL/'tasks-128.parquet'
    if not tasks.exists():
        pool = pd.read_parquet(POOL)
        pool.sample(n=TASKS, random_state=SEED).to_parquet(tasks)
    if not (CONTROL/'approval-request.json').exists():
        write_json(CONTROL/'approval-request.json', dict(
            approved=True, approved_unix=time.time(), user_message='yes I approve',
            approved_in_response_to='Pass@K: 8 x H200 / 64 CPUs / 960 GiB for 4 hours plus a $40 judge cap',
            gpus=8, cpus=64, memory_gib=960, hours=4, **CAPS))
    from resume_baseline import validate_source
    validate_source(SOURCE)
    return dict(control=str(CONTROL), tasks=str(tasks), task_sha256=sha(tasks), pool_sha256=sha(POOL))


def command_and_env():
    env = dict(NUM_GPUS='8', TP_SIZE='2', NUM_ROLLOUT='1', TRAIN_DATA=str(CONTROL/'tasks-128.parquet'),
               HF_CHECKPOINT=str(MODEL), BROWSER_TRAIN_CONFIG=str(SOURCE/'openwebrl/browser_training_config.yaml'),
               BROWSER_MAX_STEPS='15', ROLLOUT_BATCH_SIZE=str(TASKS), N_SAMPLES=str(SAMPLES), GLOBAL_BATCH_SIZE='256',
               CONTEXT_LEN='32768', RESPONSE_LEN='1024', BROWSER_CONCURRENCY='64', SGLANG_CONCURRENCY='64',
               SAVE_DIR=str(OUT), JUDGE_MODEL='gpt-4.1', JUDGE_API_MODE='served', JUDGE_API_BASE='https://api.openai.com/v1',
               OPENWEBRL_CUDA_CACHE_LIMIT_GIB='48', FLASHINFER_WORKSPACE_BASE=str(RUNTIME), RAY_ADDRESS='local',
               OPENWEBRL_MULTIMODAL_STORAGE_DIR=str(OUT/'multimodal'), OPENWEBRL_SAMPLING_CONTROL=str(CONTROL),
               SLIME_BROWSER_LOCAL_PROCESS_LOG_DIR=str(OUT/'browser-server-logs'),
               SLIME_ADAPTIVE_QUERY_BLACKLIST_PATH=str(SOURCE/'reference_empty_blacklist.txt'),
               SLIME_BROWSER_QUERY_BLACKLIST_PATH=str(SOURCE/'reference_empty_blacklist.txt'), WANDB_MODE='disabled')
    command = ['bash', str(SOURCE/'scripts/run_h200_browser.sh'), '--debug-rollout-only', '--seed', '42',
               '--rollout-top-p', '1', '--rollout-top-k', '-1', '--sglang-disable-cuda-graph',
               '--rollout-health-check-first-wait', '180', '--skip-eval-before-train',
               '--dynamic-sampling-filter-path', 'slime.rollout.filter_hub.keep_all.keep_all',
               '--save-debug-rollout-data', str(OUT/'rollout_{rollout_id}.pt')]
    return command, env


def collect(job):
    approval = read(CONTROL/'approval-request.json')
    if approval.get('approved') is not True or approval.get('hours') != 4 or approval.get('judge_cap_usd') != 40:
        raise PermissionError('Exact approval required')
    if os.getenv('SLURM_JOB_ID') != job:
        raise ValueError('Run inside the approved allocation')
    if (OUT/'rollout_0.pt').exists():
        return 0
    from dotenv import dotenv_values
    from resume_baseline import clean_environment
    command, extra = command_and_env()
    env = clean_environment()
    for k, v in dotenv_values(REPO/'.env').items():
        if v and k.startswith(('JUDGE_', 'OPENAI_')):
            env.setdefault(k, v)
    env.update(extra)
    OUT.mkdir(parents=True, exist_ok=True)
    attempt = len(list(CONTROL.glob('collect-*.log')))+1
    write_json(CONTROL/'status.json', dict(stage='collection', job_id=job, attempt=attempt, started=time.time()))
    with (CONTROL/f'collect-{job}-a{attempt}.log').open('w') as log:
        code = subprocess.run(source_command(SOURCE, ['timeout', '--signal=INT', '--kill-after=120', str(4*3600-900), *command]),
                              cwd=SOURCE, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    if code or not (OUT/'rollout_0.pt').exists():
        write_json(CONTROL/'status.json', dict(stage='failed', job_id=job, attempt=attempt, exit_code=code))
        return code or 1
    write_json(CONTROL/'status.json', dict(stage='collected', job_id=job, attempt=attempt, finished=time.time()))
    return 0


def pass_at_k(n, c, k):
    from math import comb
    return 1.0 if n-c < k else 1-comb(n-c, k)/comb(n, k)


def mixed_at_k(n, c, k):
    """P(a random size-k subset contains both a success and a failure)."""
    from math import comb
    return 1-comb(c, k)/comb(n, k)-comb(n-c, k)/comb(n, k)


def analyze():
    import torch
    data = torch.load(OUT/'rollout_0.pt', weights_only=False, map_location='cpu')
    trajectories = {}
    for s in data['samples']:
        m = s.get('metadata') or {}
        task = str(m.get('task_id')).split('/')[-1]  # invalid trajectories can carry a 'webvoyager/' prefix
        key = (task, m.get('trajectory_id', s.get('index')))
        if m.get('is_last_turn') or key not in trajectories:
            reward = s.get('reward')
            value = reward.get('judge') if isinstance(reward, dict) else reward
            trajectories[key] = dict(task=task, reward=value, status=str(s.get('status')),
                                     reason=m.get('terminate_reason'))
    tasks = {}
    for t in trajectories.values():
        tasks.setdefault(t['task'], []).append(t)
    rows, ks = [], (1, 5, 8, 16)
    for task, ts in tasks.items():
        valid = [t for t in ts if t['reward'] is not None and t['reason'] != 'env_step_error' and 'ABORT' not in t['status'].upper()]
        c = sum(float(t['reward']) > 0 for t in valid)
        rows.append(dict(task=task, n=len(ts), valid=len(valid), successes=c))
    summary = dict(tasks=len(rows), trajectories=len(trajectories), samples_per_task=SAMPLES, seed=SEED,
                   invalid_trajectories=sum(r['n']-r['valid'] for r in rows))
    for label, subset in (('all_tasks_invalid_as_failure', [dict(r, n=r['n']) for r in rows]),
                          ('valid_draws_only', [dict(r, n=r['valid']) for r in rows])):
        res = {}
        for k in ks:
            use = [r for r in subset if r['n'] >= k]
            res[f'k{k}'] = dict(tasks=len(use),
                                pass_at_k=sum(pass_at_k(r['n'], r['successes'], k) for r in use)/max(1, len(use)),
                                mixed_group_probability=sum(mixed_at_k(r['n'], r['successes'], k) for r in use)/max(1, len(use)))
        summary[label] = res
    summary['task_success_count_histogram'] = {str(c): sum(r['successes'] == c for r in rows) for c in range(SAMPLES+1)}
    write_json(CONTROL/'passk-summary.json', summary)
    write_json(CONTROL/'passk-per-task.json', rows)
    print(json.dumps(summary, indent=1))


def hold(job, reason):
    request = CONTROL/f'{job}-hold-request.json'
    idle_used = 0.
    while True:
        limit = min(HOLD_IDLE_SECONDS, HOLD_TOTAL_IDLE_SECONDS-idle_used)
        write_json(CONTROL/f'hold-{job}.json', dict(holding=True, reason=str(reason)[:600], request_path=str(request),
                                                    idle_limit_seconds=int(limit), epoch=time.time()))
        start, r = time.time(), None
        while time.time()-start < limit:
            if request.is_file():
                r = read(request)
                request.rename(request.with_name(f'{request.name}.{int(time.time())}.consumed'))
                break
            time.sleep(15)
        idle_used += time.time()-start
        if r is None or r.get('action') == 'release':
            write_json(CONTROL/f'hold-{job}.json', dict(holding=False, released=True, idle_cap=r is None, epoch=time.time()))
            return 1
        argv = ([sys.executable, '-B', str(Path(__file__).resolve()), '--job-id', job, '--collect']
                if r.get('action') == 'retry' else list(map(str, r.get('argv', []))))
        code = subprocess.run(argv).returncode if argv else 2
        with (CONTROL/f'hold-{job}.jsonl').open('a') as log:
            log.write(json.dumps(dict(epoch=time.time(), argv=argv, exit_code=code))+'\n')
        if code == 0 and r.get('action') == 'retry':
            return 0
        reason = f'{r.get("action")} exited {code}'


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job-id')
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--collect', action='store_true')
    p.add_argument('--analyze', action='store_true')
    a = p.parse_args()
    if a.prepare:
        print(json.dumps(prepare(), indent=1))
    elif a.collect:
        sys.exit(collect(a.job_id))
    elif a.execute:
        code = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()), '--job-id', a.job_id, '--collect']).returncode
        code = code if code == 0 else hold(a.job_id, f'collection exited {code}; see {CONTROL}/status.json')
        if code == 0:
            analyze()
        sys.exit(code)
    elif a.analyze:
        analyze()
    else:
        print(json.dumps(command_and_env(), indent=1))
