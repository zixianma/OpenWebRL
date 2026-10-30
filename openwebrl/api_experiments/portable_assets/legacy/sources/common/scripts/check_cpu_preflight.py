#!/usr/bin/env python3
"""Run lightweight offline checks; never launch models, browsers, or training."""
import argparse
import ast
from datetime import datetime, timezone
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tomllib
import unittest
import warnings

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', default='outputs/preflight/cpu_report.json')
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    import os
    os.chdir(ROOT)
    files = subprocess.check_output(['rg', '--files', '-g', '*.py', '-g', '*.sh', '-g', '*.json',
                                     '-g', '*.jsonl', '-g', '*.yaml', '-g', '*.yml'], text=True).splitlines()
    counts = dict(python=0, shell=0, json=0, jsonl_records=0, yaml=0, toml=0)
    errors, notices = [], []
    for filename in files:
        path = Path(filename)
        try:
            if path.suffix == '.py':
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    ast.parse(path.read_text(), filename=filename)
                    notices.extend(f'{filename}: {w.message}' for w in caught)
                counts['python'] += 1
            elif path.suffix == '.sh':
                subprocess.run(['bash', '-n', filename], check=True, capture_output=True, text=True)
                counts['shell'] += 1
            elif path.suffix == '.json':
                json.loads(path.read_text())
                counts['json'] += 1
            elif path.suffix == '.jsonl':
                for line in path.read_text().splitlines():
                    if line.strip():
                        json.loads(line)
                        counts['jsonl_records'] += 1
        except Exception as error:
            errors.append(f'{filename}: {error}')
    # The system Python may have PyYAML even when the training Python does not.
    yaml_files = [p for p in files if Path(p).suffix in ('.yaml', '.yml')]
    yaml_code = '''import json, sys, yaml
for filename in json.loads(sys.stdin.read()):
    with open(filename) as handle:
        list(yaml.safe_load_all(handle))
'''
    yaml_checked = False
    for python in dict.fromkeys([sys.executable, 'python3']):
        result = subprocess.run([python, '-c', 'import yaml'], capture_output=True, text=True)
        if result.returncode == 0:
            result = subprocess.run([python, '-c', yaml_code], input=json.dumps(yaml_files),
                                    capture_output=True, text=True)
            if result.returncode:
                errors.append('YAML parsing: ' + result.stderr)
            else:
                counts['yaml'] = len(yaml_files)
            yaml_checked = True
            break
    if not yaml_checked:
        notices.append('YAML parsing skipped: PyYAML unavailable.')
    try:
        tomllib.loads((ROOT/'pyproject.toml').read_text())
        counts['toml'] = 1
    except Exception as error:
        errors.append(f'pyproject.toml: {error}')
    output = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(ROOT/'tests'))
    result = unittest.TextTestRunner(stream=output, verbosity=2).run(suite)
    print(output.getvalue(), end='')
    diff = subprocess.run(['git', 'diff', '--check'], text=True, capture_output=True)
    if diff.returncode:
        errors.append('git diff --check: ' + diff.stdout + diff.stderr)
    dependencies = {name: importlib.util.find_spec(name) is not None for name in
                    ['torch','ray','transformers','openai','PIL','yaml','playwright','sglang']}
    report = {
        'checked_at_utc': datetime.now(timezone.utc).isoformat(),
        'python': sys.executable,
        'lightweight_checks_passed': result.wasSuccessful() and not errors,
        'syntax_and_data_counts': counts,
        'tests_run': result.testsRun,
        'test_failures': len(result.failures), 'test_errors': len(result.errors),
        'test_log': output.getvalue(), 'errors': errors, 'notices': notices,
        'runtime_dependencies_present': dependencies,
        'not_validated': ['CUDA/NCCL and GPU memory', 'Megatron/SGLang/model compatibility',
                          'real tokenizer and multimodal processor', 'live browsers and judge endpoints',
                          'checkpoint tensor save/load', 'Parquet contents (reader unavailable)'],
    }
    target = Path(args.report)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'test_log'}, indent=2))
    print(f'Report: {target.resolve()}')
    raise SystemExit(0 if report['lightweight_checks_passed'] else 1)


if __name__ == '__main__':
    main()
