#!/usr/bin/env python3
"""Create an isolated Browser Use evaluation source without changing the baseline."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from evaluate_baseline_checkpoint import RUNTIME, REPO
from resume_baseline import validate_source, write_json


def prepare(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    validate_source(source)
    if output.exists() or not output.is_relative_to(RUNTIME):
        raise ValueError('Use a new source directory inside the project runtime')
    shutil.copytree(source, output, symlinks=True, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.browser_use_sessions'))
    launcher = output / 'scripts/run_h200_browser.sh'
    text = launcher.read_text()
    old = 'export SLIME_BROWSER_ENV_MODE=local_process'
    if text.count(old) != 1:
        raise ValueError('Unexpected baseline launcher browser-mode assignment')
    launcher.write_text(text.replace(old, 'export SLIME_BROWSER_ENV_MODE="${SLIME_BROWSER_ENV_MODE:-local_process}"'))
    adapter = 'openwebrl/env/browser_use_env.py'
    shutil.copyfile(REPO / adapter, output / adapter)
    config = output / 'openwebrl/env/config.yaml'
    text = config.read_text()
    if '  timeout: 10 ' not in text:
        raise ValueError('Unexpected Browser Use timeout configuration')
    config.write_text(text.replace('  timeout: 10 ', '  timeout: 12 ', 1))
    manifest_path = output / 'reference_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    changed = ['scripts/run_h200_browser.sh', adapter, 'openwebrl/env/config.yaml']
    hashes = {name: hashlib.sha256((output/name).read_bytes()).hexdigest() for name in changed}
    manifest['recipe_files_sha256'].update(hashes)
    manifest['browser_use_evaluation'] = {
        'parent_source': str(source), 'changed_files_sha256': hashes,
        'backend': 'Browser Use Cloud stealth Chromium via CDP', 'proxy_country_code': None,
        'session_timeout_minutes': 12, 'sdk_version': '3.11.3',
        'sdk_dependency_policy': 'Isolated SDK-only overlay; existing httpx/pydantic/idna retained and smoke-tested.',
        'purpose': 'Evaluation only; model, task cohort, judge, prompts and decoding unchanged.',
    }
    write_json(manifest_path, manifest)
    validate_source(output)
    return manifest['browser_use_evaluation']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output), indent=2))
