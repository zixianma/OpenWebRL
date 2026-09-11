#!/usr/bin/env python3
"""Create an isolated Browser Use evaluation source without changing the baseline."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
from evaluate_baseline_checkpoint import RUNTIME, REPO
from resume_baseline import validate_source, write_json


def patch_browser_billing_models(text):
    """Accept finite scientific notation returned by the browser billing API."""
    for model in ('BrowserSessionView', 'BrowserSessionItemView'):
        start = text.index(f'class {model}(BaseModel):')
        end = text.index('\nclass ', start + 1)
        block = text[start:end]
        pattern = "        pattern=" + repr(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\Z") + ",\n"
        block, count = re.subn(r'        pattern=.*,\n', lambda _: pattern, block)
        if count != 3:
            raise ValueError(f'Unexpected SDK billing fields in {model}')
        text = text[:start] + block + text[end:]
    return text


def prepare(source, output, browser_concurrency=8):
    if not 1 <= browser_concurrency <= 10:
        raise ValueError('Browser Use account allows at most 10 concurrent sessions')
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
    training_config = output / 'openwebrl/browser_training_config.yaml'
    text = training_config.read_text()
    if text.count('browser_rollout_concurrency: 16') != 1:
        raise ValueError('Unexpected baseline browser concurrency configuration')
    training_config.write_text(text.replace('browser_rollout_concurrency: 16',
                                            f'browser_rollout_concurrency: {browser_concurrency}'))
    sdk_parent = RUNTIME / 'browser-use-sdk-3.11.3/browser_use_sdk'
    shutil.copytree(sdk_parent, output / 'browser_use_sdk',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    sdk_models = 'browser_use_sdk/generated/v2/models.py'
    model_text = (output / sdk_models).read_text()
    (output / sdk_models).write_text(patch_browser_billing_models(model_text))
    manifest_path = output / 'reference_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    changed = ['scripts/run_h200_browser.sh', adapter, 'openwebrl/env/config.yaml',
               'openwebrl/browser_training_config.yaml', sdk_models]
    hashes = {name: hashlib.sha256((output/name).read_bytes()).hexdigest() for name in changed}
    manifest['recipe_files_sha256'].update(hashes)
    manifest['browser_use_evaluation'] = {
        'parent_source': str(source), 'changed_files_sha256': hashes,
        'backend': 'Browser Use Cloud stealth Chromium via CDP', 'proxy_country_code': None,
        'session_timeout_minutes': 12, 'sdk_version': '3.11.3',
        'browser_task_concurrency': browser_concurrency,
        'sdk_package': str(output / 'browser_use_sdk'),
        'sdk_parent': str(sdk_parent),
        'sdk_parent_models_sha256': hashlib.sha256(model_text.encode()).hexdigest(),
        'sdk_billing_fix': 'Accept finite decimal/scientific notation in the six browser-session billing fields; source-local SDK precedes the original overlay.',
        'concurrency_reason': 'Sustained evaluation previously hit the account limit of 10 sessions; default 8 leaves headroom.',
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
    parser.add_argument('--browser-concurrency', type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output, args.browser_concurrency), indent=2))
