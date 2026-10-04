#!/usr/bin/env python3
"""Prepare an isolated paper-protocol source; never submits compute."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from evaluate_baseline_checkpoint import REPO, RUNTIME
from resume_baseline import validate_source, write_json


def prepare(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    validate_source(source)
    parent = json.loads((source / 'reference_manifest.json').read_text())
    if not parent.get('browser_use_evaluation'):
        raise ValueError('Start from the validated Browser Use evaluation source')
    if output.exists() or not output.is_relative_to(RUNTIME):
        raise ValueError('Use a new runtime directory')
    shutil.copytree(source, output, symlinks=True,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.browser_use_sessions'))
    changed = ['openwebrl/eval_benchmark.py', 'openwebrl/online_mind2web_benchmark.yaml',
               'openwebrl/eval/reward_online_mind2web.py', 'openwebrl/eval/_shared.py',
               'openwebrl/eval_monitor.py', 'slime/utils/rollout_transport.py',
               'slime/utils/trajectory_metrics.py', 'slime/utils/types.py']
    for name in changed:
        (output / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / name, output / name)
    hashes = {name: hashlib.sha256((output/name).read_bytes()).hexdigest() for name in changed}
    parent['recipe_files_sha256'].update(hashes)
    parent['paper_om2w_benchmark'] = {
        'parent_source': str(source), 'changed_files_sha256': hashes,
        'paper': 'https://arxiv.org/html/2606.02031v1#A6',
        'judge': 'o4-mini', 'judge_protocol': 'Online-Mind2Web/AgentTrek',
        'judge_images': 'last screenshot, as implemented by the benchmark judge',
        'judge_seed': 42, 'judge_temperature': 'API default (not sent)',
        'judge_max_tokens': 'API default (not sent)',
        'temperature': .6, 'top_p': .95, 'top_k': 20, 'max_response_tokens': 4096,
        'max_context_tokens': 32768, 'repetition_penalty': 1., 'max_steps': 30,
        'browser_backend': 'browser-use', 'browser_concurrency': 8,
        'proxy_country_code': None, 'seed': 'paper does not specify actor seed; runtime sampling default',
        'training_recipe_changed': False,
        'gpu_validation_passed': False,
    }
    write_json(output / 'reference_manifest.json', parent)
    validate_source(output)
    return parent['paper_om2w_benchmark']


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(prepare(a.source, a.output), indent=2))
