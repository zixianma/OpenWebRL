#!/usr/bin/env python3
"""Prepare or run a frozen actor-temperature comparison; never submits jobs."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import shutil

from evaluate_baseline_checkpoint import (
    build_plan, run, REPO, RUNTIME, DEFAULT_EVAL_PROJECT, configure_evaluation_tracking,
)
from resume_baseline import validate_source, write_json


def prepare_greedy(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    validate_source(source)
    if output.exists() or not output.is_relative_to(RUNTIME) or output.is_relative_to(source):
        raise ValueError('Use a new isolated source directory under runtime storage')
    manifest = json.loads((source / 'reference_manifest.json').read_text())
    if manifest['paper_om2w_benchmark']['temperature'] != 0.6:
        raise ValueError('Expected the frozen temperature-0.6 benchmark source')
    changes = {
        'openwebrl/eval_benchmark.py': ('SAMPLING = dict(temperature=0.6,', 'SAMPLING = dict(temperature=0.0,'),
        'openwebrl/online_mind2web_benchmark.yaml': ('temperature: 0.6', 'temperature: 0.0'),
    }
    modified = {}
    for name, (old, new) in changes.items():
        text = (source / name).read_text()
        if text.count(old) != 1:
            raise ValueError(f'Unexpected temperature configuration in {name}')
        modified[name] = text.replace(old, new, 1)
    shutil.copytree(source, output, symlinks=True,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git', '.env*', '.browser_use_sessions'))
    for name, text in modified.items():
        (output / name).write_text(text)
        digest = hashlib.sha256((output / name).read_bytes()).hexdigest()
        manifest['recipe_files_sha256'][name] = digest
        manifest['paper_om2w_benchmark']['changed_files_sha256'][name] = digest
    manifest['paper_om2w_benchmark']['temperature'] = 0.0
    manifest['actor_temperature_comparison'] = {
        'parent_source': str(source), 'temperature': 0.0,
        'changed_settings': ['actor temperature'],
        'unchanged': 'checkpoint, tasks, o4-mini judge, top_p=.95, top_k=20, token and turn limits, stealth browser',
        'paper_sampling_match': False,
    }
    write_json(output / 'reference_manifest.json', manifest)
    validate_source(output)


def temperature_plan(source, checkpoint, output, job_id, temperature, wandb_project=DEFAULT_EVAL_PROJECT):
    if temperature not in (0.0, 0.6):
        raise ValueError('This comparison only supports temperature 0 or 0.6')
    plan = build_plan(source, checkpoint, output, job_id, gpus=2,
                      browser_env='browser-use', protocol='benchmark')
    source = Path(source)
    manifest = json.loads((source / 'reference_manifest.json').read_text())
    sampling = runpy.run_path(str(source / 'openwebrl/eval_benchmark.py'))['SAMPLING']
    if (manifest['paper_om2w_benchmark']['temperature'] != temperature
            or sampling['temperature'] != temperature):
        raise ValueError('Requested temperature differs from the frozen source')
    if sampling['top_p'] != .95 or sampling['top_k'] != 20:
        raise ValueError('Unexpected change to the other actor sampling settings')
    tag = f't{temperature:g}'
    run_id = f'qcq7i4ug-stealth-o4-after{plan["completed_training_iterations"]}-{tag}-{job_id}'
    plan.update(wandb_run_id=run_id, actor_temperature=temperature,
                protocol=f'o4-mini/AgentTrek OM2W; actor T={temperature:g}, p=.95, k=20; Browser Use stealth, 300 tasks')
    plan['environment']['WANDB_RUN_ID'] = run_id
    command = plan['command']
    command[command.index('--wandb-group') + 1] = 'qcq7i4ug-stealth-o4-temperature-comparison'
    return configure_evaluation_tracking(plan, wandb_project)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--prepare-greedy', action='store_true')
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--job-id')
    p.add_argument('--temperature', type=float, choices=[0.0, 0.6])
    p.add_argument('--execute', action='store_true')
    p.add_argument('--env-file', type=Path, default=REPO / '.env')
    p.add_argument('--wandb-project', default=DEFAULT_EVAL_PROJECT)
    a = p.parse_args()
    if a.prepare_greedy:
        if a.execute:
            p.error('Preparation does not execute evaluations')
        prepare_greedy(a.source, a.output)
        print(a.output)
    else:
        if a.checkpoint is None or a.job_id is None or a.temperature is None:
            p.error('Evaluation needs checkpoint, job ID and temperature')
        plan = temperature_plan(a.source, a.checkpoint, a.output, a.job_id, a.temperature, a.wandb_project)
        if a.execute:
            run(plan, a.env_file)
        else:
            print(json.dumps(plan, indent=2))
