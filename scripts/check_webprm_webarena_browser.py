#!/usr/bin/env python3
"""Exercise pinned native WebArena scoring on controlled Chromium pages, no API."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openwebrl.webprm_runtime import prepare_browser_runtime
from openwebrl.webprm_webarena import EvaluationNotReady, load_native_evaluator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--chromium', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    protected = Path('/gpfs/home/zixianma')
    if output == protected or protected in output.parents or output.exists():
        parser.error('Choose a new output file outside the protected home directory')
    output.parent.mkdir(parents=True, exist_ok=True)
    runtime = prepare_browser_runtime()
    native = load_native_evaluator(args.source)
    from playwright.sync_api import sync_playwright

    checks = []
    state = {'observation': {'text': 'Controlled fixture'}, 'info': {}}
    task = {'intent': 'Controlled fixture; no benchmark task', 'eval': {
        'eval_types': ['program_html'],
        'program_html': [{'url': 'last', 'locator': 'document.querySelector("#result").textContent',
                          'required_contents': {'exact_match': '42'}}]}}
    with sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path=str(args.chromium), headless=True)
        try:
            context = browser.new_context()
            context.route('**/*', lambda route: route.fulfill(
                status=200, content_type='text/html', body='<html><body><p id="result">42</p></body></html>'))
            page = context.new_page()
            page.goto('http://webarena-fixture.invalid/goal', wait_until='domcontentloaded', timeout=5000)
            page.set_default_timeout(5000)

            def check(name, config, expected):
                result = native.evaluate(config, [state], page, termination='turn_limit')
                assert result['valid'] and result['success'] is expected, (name, result)
                checks.append({'name': name, 'passed': True,
                               'native_score': result['score'],
                               'native_javascript_errors': len(result['native_javascript_errors'])})

            check('real_html_at_turn_limit_positive', task, True)
            negative = deepcopy(task)
            negative['eval']['program_html'][0]['required_contents']['exact_match'] = '43'
            check('real_html_at_turn_limit_negative', negative, False)
            combined = deepcopy(task)
            combined['eval'].update(eval_types=['url_match', 'program_html'],
                                    reference_url='http://webarena-fixture.invalid/goal')
            check('combined_url_html', combined, True)
            redirected = deepcopy(task)
            redirected['eval']['program_html'][0]['url'] = 'http://webarena-fixture.invalid/evaluator-target'
            check('native_evaluator_navigation', redirected, True)
            missing = deepcopy(task)
            missing['eval']['program_html'][0]['locator'] = 'document.querySelector("#missing").textContent'
            check('healthy_browser_missing_dom_native_failure', missing, False)
            page.close()
            try:
                native.evaluate(task, [state], page, termination='turn_limit')
            except EvaluationNotReady:
                checks.append({'name': 'closed_browser_rejected', 'passed': True})
            else:
                raise AssertionError('Closed browser received a valid verdict')
        finally:
            browser.close()
    report = {'status': 'passed', 'scope': 'controlled real-browser evaluator contracts, not benchmark success',
              'checks': checks, 'model_provider_calls': 0, 'external_website_requests': 0,
              'gpu_allocation_seconds': 0, 'runtime': str(runtime), 'source': native.manifest,
              'adapter_sha256': hashlib.sha256((Path(__file__).resolve().parents[1] /
                                                   'openwebrl/webprm_webarena.py').read_bytes()).hexdigest()}
    with output.open('x') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    print(json.dumps({'status': 'passed', 'checks': len(checks), 'output': str(output)}))


if __name__ == '__main__':
    main()
