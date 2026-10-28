#!/usr/bin/env python3
"""Freeze pilot code and install pinned browser dependencies; no GPU/API calls."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
FILES = [
    'openwebrl/__init__.py', 'openwebrl/jev_eval.py',
    'openwebrl/webprm_baselines.py', 'openwebrl/webprm_source.py',
    'openwebrl/webprm_worker.py', 'openwebrl/webprm_arbiter_harness.py',
    'openwebrl/webprm_shepherd_harness.py',
    'openwebrl/webprm_runtime.py',
    'openwebrl/webprm_lifecycle.py',
    'openwebrl/eval/reward_online_mind2web.py',
    'scripts/run_webprm_pilot.py', 'scripts/launch_webprm_pilot.py',
    'scripts/analyze_webprm_baselines.py',
    'scripts/preflight_webprm_models.py',
    'scripts/evaluate_webprm_pilot_1gpu.sbatch',
]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--version', default='original')
    parser.add_argument('--reuse-browser-env', type=Path,
                        help='Use an already pinned environment without reinstalling packages')
    args = parser.parse_args()
    root = args.root.resolve()
    plan = json.loads((root/'plan.json').read_text())
    assert plan['proposal']['api_usd_cap'] == 30
    assert plan['proposal']['total_gpu_hours'] > 0
    assert not (root/'submission.json').exists(), 'Do not change a submitted freeze'
    if not args.version.replace('-', '').replace('_', '').isalnum():
        parser.error('Version must be a simple directory suffix')
    code = root/('code' if args.version == 'original' else f'code-{args.version}')
    hashes = {}
    for name in FILES:
        source, destination = REPO/name, code/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.read_bytes() != source.read_bytes():
            raise ValueError(f'Existing freeze differs: {name}')
        shutil.copyfile(source, destination)
        hashes[name] = hashlib.sha256(destination.read_bytes()).hexdigest()
    browser = root/'browser-env'
    if args.reuse_browser_env:
        if browser.exists() and browser.resolve() != args.reuse_browser_env.resolve():
            raise ValueError('Existing browser environment differs')
        if not browser.exists():
            browser.symlink_to(args.reuse_browser_env.resolve(), target_is_directory=True)
    elif not (browser/'bin/python').exists():
        subprocess.run([sys.executable, '-m', 'venv', str(browser)], check=True)
    if not args.reuse_browser_env:
        subprocess.run([str(browser/'bin/pip'), 'install', '--disable-pip-version-check',
                        '--cache-dir', str(root/'pip-cache'), '-r', str(root/'requirements-browser.txt')], check=True)
    subprocess.run([str(browser/'bin/python'), '-m', 'compileall', '-q', str(code)], check=True)
    checks = subprocess.check_output([str(browser/'bin/python'), '-c',
        "import importlib.metadata,json; print(json.dumps({n:importlib.metadata.version(n) for n in ['browsergym-core','playwright','gymnasium','numpy','pillow','pyparsing','python-dotenv']}))"], text=True)
    expected = dict(line.strip().split('==') for line in (root/'requirements-browser.txt').read_text().splitlines() if line.strip())
    if json.loads(checks) != expected:
        raise ValueError('Browser environment dependencies differ from pinned requirements')
    freeze = root/('code-freeze.json' if args.version == 'original' else f'code-freeze-{args.version}.json')
    freeze.write_text(json.dumps(dict(code_directory=str(code),
        code_sha256=hashes, browser_dependencies=json.loads(checks),
        engine_dependencies={n:importlib.metadata.version(n) for n in ['torch','transformers','sglang']},
        paid_api_calls=0, gpu_seconds=0), indent=2))
    print(json.dumps(dict(status='code_and_dependencies_staged', files=len(hashes), root=str(root))))

if __name__ == '__main__':
    main()
