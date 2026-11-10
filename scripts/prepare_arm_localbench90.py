#!/usr/bin/env python3
"""Freeze CPU-reviewed local benchmark inputs; never submit or execute jobs."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import shutil
import sys
from types import SimpleNamespace
from unittest.mock import patch

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluate_baseline_checkpoint as evaluator
from resume_baseline import validate_source, write_json
from browser_prompt_assets import freeze_browser_prompt_assets, validate_browser_prompt_assets

REPO = Path(__file__).resolve().parents[1]
RUNTIME = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime')
CONTROL = RUNTIME / 'arm-turn-bonus-preparation/original90-localbench-20261007'
PARENT = RUNTIME / 'arm-turn-bonus-preparation/webvoyager90-stealth-20260929/source-v2-chat-prompts'
SOURCE = CONTROL / 'source-local-v2-prompt-assets'
COUNTS = {'webvoyager': 595, 'deepshop': 150}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_source():
    validate_source(PARENT)
    if SOURCE.exists():
        return validate()
    shutil.copytree(PARENT, SOURCE, symlinks=True,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.env*', '.git'))
    changes = []
    for source, target in [
        ('scripts/arm_local_benchmark_generator.py', 'openwebrl/eval_local_benchmarks.py'),
        ('openwebrl/eval/reward_deepshop.py', 'openwebrl/eval/reward_deepshop.py'),
        ('openwebrl/eval/_shared.py', 'openwebrl/eval/_shared.py'),
        ('openwebrl/env/local_process_env.py', 'openwebrl/env/local_process_env.py'),
    ]:
        shutil.copy2(REPO / source, SOURCE / target)
        changes.append(target)
    env_name = 'openwebrl/env/config.yaml'
    env = yaml.safe_load((SOURCE / env_name).read_text())
    env['mode'] = 'local_process'
    env['local_process'].update(max_processes=8, startup_timeout_secs=120)
    (SOURCE / env_name).write_text(yaml.safe_dump(env, sort_keys=False))
    training_name = 'openwebrl/browser_training_config.yaml'
    config = yaml.safe_load((SOURCE / training_name).read_text())
    config.update(browser_rollout_concurrency=8, max_steps=30,
                  inference_step_timeout_secs=120, rollout_task_timeout_secs=600)
    (SOURCE / training_name).write_text(yaml.safe_dump(config, sort_keys=False))
    changes += [env_name, training_name]
    datasets = {}
    for benchmark, count in COUNTS.items():
        raw_path = (REPO / 'openwebrl/data/eval/deepshop.jsonl' if benchmark == 'deepshop'
                    else PARENT / 'webvoyager_released.jsonl')
        raw = [json.loads(line) for line in raw_path.read_text().splitlines()]
        lookup, prepared = [], []
        lookup_name = f'{benchmark}_local_lookup.jsonl'
        data_name = f'{benchmark}_local_eval.jsonl'
        for row in raw:
            if benchmark == 'deepshop':
                ident, intent, url = row['task_id'], row['task_name'], row['website']
                metadata = dict(benchmark_name='deepshop')
            else:
                ident, intent, url = row['id'], row['ques'], row['web']
                metadata = dict(row['metadata'])
                assert (metadata['intent'], metadata['start_url']) == (intent, url)
            ident = str(ident)
            if (SOURCE / 'openwebrl/env/tasks' / f'{ident}.json').exists():
                raise ValueError('Per-task file could override the pinned dataset')
            metadata.update(task_id=ident, intent=intent, start_url=url,
                            _browser_task_file=str(SOURCE / lookup_name))
            lookup.append(dict(id=ident, ques=intent, web=url, metadata=metadata))
            prepared.append(dict(prompt=[dict(role='user', content=intent)], metadata=metadata))
        assert len(prepared) == count == len({r['metadata']['task_id'] for r in prepared})
        for name, rows in [(lookup_name, lookup), (data_name, prepared)]:
            (SOURCE / name).write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in rows))
        yaml_name = f'openwebrl/{benchmark}_local_eval.yaml'
        (SOURCE / yaml_name).write_text(yaml.safe_dump(dict(eval=dict(
            defaults=dict(n_samples_per_eval_prompt=1, temperature=.6, top_p=.95,
                          top_k=20, max_response_len=4096),
            datasets=[dict(name=f'{benchmark}-local', path=data_name, input_key='prompt',
                           metadata_key='metadata', custom_generate_function_path=
                           f'openwebrl.eval_local_benchmarks.generate_{benchmark}')]))))
        changes += [lookup_name, data_name, yaml_name]
        datasets[benchmark] = dict(count=count, raw_path=str(raw_path), raw_sha256=sha(raw_path),
                                   eval_file=data_name, lookup_file=lookup_name, yaml_file=yaml_name)
    manifest = json.loads((SOURCE / 'reference_manifest.json').read_text())
    prompt_hashes = freeze_browser_prompt_assets(SOURCE, REPO)
    changes += list(prompt_hashes)
    manifest['recipe_files_sha256'].update({name: sha(SOURCE/name) for name in changes})
    manifest['single_gpu_evaluation']['browser_concurrency'] = 8
    manifest['local_benchmark_suite'] = dict(parent=str(PARENT), datasets=datasets,
        browser_env='local_process', actor_only=True, repeated_passes_require_new_approval=True,
        protocol='T0.6/p0.95/k20, 4096 tokens, 30 turns; native GPT-4o benchmark judges',
        changes=changes, gpu_validation_pending=True)
    write_json(SOURCE / 'reference_manifest.json', manifest)
    return validate()


def validate():
    validate_source(SOURCE)
    validate_browser_prompt_assets(SOURCE)
    manifest = json.loads((SOURCE / 'reference_manifest.json').read_text())
    module = runpy.run_path(str(SOURCE / 'openwebrl/eval_local_benchmarks.py'))
    env = yaml.safe_load((SOURCE / 'openwebrl/env/config.yaml').read_text())
    assert env['mode'] == 'local_process' and env['local_process']['max_processes'] == 8
    reports = {}
    for benchmark, count in COUNTS.items():
        dataset = manifest['local_benchmark_suite']['datasets'][benchmark]
        assert sha(dataset['raw_path']) == dataset['raw_sha256']
        rows = [json.loads(s) for s in (SOURCE/dataset['eval_file']).read_text().splitlines()]
        assert len(rows) == count == len({r['metadata']['task_id'] for r in rows})
        with patch.dict('os.environ', {'SLIME_BROWSER_ENV_MODE': 'local_process'}):
            configured, sampling = module['configure'](SimpleNamespace(), {}, benchmark)
        assert configured.judge_api_model == 'gpt-4o'
        assert configured.judge_prompt_variant == benchmark
        assert sampling == dict(temperature=.6, top_p=.95, top_k=20,
                                max_new_tokens=4096, repetition_penalty=1.)
        assert configured.max_steps == configured.judge_max_attached_imgs == 30
        reports[benchmark] = dict(tasks=count, dataset_sha256=sha(SOURCE/dataset['eval_file']),
                                  raw_sha256=dataset['raw_sha256'], sampling=sampling)
    return dict(source=str(SOURCE), manifest_sha256=sha(SOURCE/'reference_manifest.json'),
                datasets=reports, browser_env='local_process', gpu_validation_pending=True)


def preview(method, checkpoint, optimizer_updates, scheduler_offset, parent_run):
    """Build first-repeat previews only; an absent future90 stays a dependency."""
    validate()
    checkpoint = Path(checkpoint)
    if checkpoint.name != 'iter_0000089':
        raise ValueError('This matrix is specifically iteration90')
    plans = []
    for benchmark, count in COUNTS.items():
        output = RUNTIME / f'evaluations/arm90-local-{method}-{benchmark}-r1-PREPARE'
        p = evaluator.build_plan(SOURCE, checkpoint, output, 'PREPARE', gpus=1,
                                 browser_env='local_process')
        rows = [json.loads(s) for s in (SOURCE/f'{benchmark}_local_eval.jsonl').read_text().splitlines()]
        p.update(method=method, benchmark=benchmark, repeat=1, completed_iteration=90,
                 completed_optimizer_updates=optimizer_updates, expected_task_count=count,
                 expected_rollout_task_ids=[r['metadata']['task_id'] for r in rows],
                 require_task_rollouts=True, parent_training_run=parent_run,
                 wandb_run_id=output.name, browser_concurrency=8, allocation_hours_cap=12,
                 judge_model='gpt-4o', judge_prompt_variant=benchmark,
                 metric_prefix=f'eval/{benchmark}-local',
                 protocol=f'local_process/{benchmark}; GPT-4o; T0.6/p0.95/k20; 4096 tokens; 30 turns',
                 submission_enabled=False, requires_exact_compute_approval=True)
        p['environment'].update(SLIME_BROWSER_ENV_MODE='local_process', WANDB_RUN_ID=output.name,
            WANDB_RESUME='never', JUDGE_MODEL='gpt-4o', BROWSER_MAX_STEPS='30', RESPONSE_LEN='4096',
            BROWSER_CONCURRENCY='8', SGLANG_CONCURRENCY='8',
            BROWSER_TRAIN_CONFIG=str(SOURCE/'openwebrl/browser_training_config.yaml'),
            OPENWEBRL_EXPECTED_SCHEDULER_OFFSET=str(scheduler_offset),
            OPENWEBRL_RECORD_EVAL_EXPECTED_TASK_COUNT=str(count), PYTHONDONTWRITEBYTECODE='1',
            OPENWEBRL_CUDA_CACHE_LIMIT_GIB='48',
            WANDB_CACHE_DIR=str(output/'wandb-cache'), RAY_TMPDIR=f'/tmp/arm90-PREPARE-{method}-{benchmark}')
        for flag, value in {'--eval-config':str(SOURCE/f'openwebrl/{benchmark}_local_eval.yaml'),
                            '--wandb-group':'ARM iteration90 local WebVoyager DeepShop first repeat'}.items():
            p['command'][p['command'].index(flag)+1] = value
        p['command'].extend(['--seed', '1234'])
        evaluator.configure_evaluation_tracking(p)
        assert p['environment']['WANDB_PROJECT'] == 'openwebrl-evals'
        plans.append(p)
    return plans


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-source', action='store_true')
    args = parser.parse_args()
    print(json.dumps(prepare_source() if args.prepare_source else validate(), indent=2))
