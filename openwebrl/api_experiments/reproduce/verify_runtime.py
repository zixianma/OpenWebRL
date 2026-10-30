"""Import real runtime modules and optionally run a zero-model browser preflight.

This checks dependencies and the selected policy constructors without importing
SDK clients, calling model APIs, registering a campaign or touching old ledgers.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile


def verify(runtime, *, browser_preflight=False):
    runtime = Path(runtime).resolve()
    config = json.loads((runtime / 'runtime-config.json').read_text())
    plan = json.loads((runtime / 'campaign-plan.json').read_text())
    sys.dont_write_bytecode = True
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.path.insert(0, str(runtime))
    worker = importlib.import_module('worker')
    environments = importlib.import_module('environments')
    results = []
    with tempfile.TemporaryDirectory(prefix='openwebrl-offline-import-') as temporary:
        root = Path(temporary)
        os.environ['OPENWEBRL_REPRO_OUTPUT'] = str(root)
        budget = getattr(worker, 'ReservationBudget', worker.Budget)(root / 'model-budget', 1, 1)
        for spec in plan['arms']:
            kwargs = dict(harness='browsercode', model_name=spec['model'], reasoning_effort=spec.get('reasoning_effort', 'high'), max_output_tokens=spec['output_tokens'], response_mode='run_command_tool', action_field='bash_command', openai_api_key='offline-no-network-client', attach_observation_screenshot=False, observation_template=config['observation_template'])
            if config.get('portable_campaign') == 'browsercode_luna_effort':
                from request_budget import SharedHTTPGate
                kwargs['http_gate'] = SharedHTTPGate(root / 'http-gate', 8)
            elif config.get('portable_campaign') == 'browsercode_claude':
                from claude_budget import SharedHTTPGate
                kwargs['http_gate'] = SharedHTTPGate(root / 'http-gate', 8)
            model = worker.PairedModel(spec['arm'], root, object(), budget, config.get('model_pricing', {}).get(spec['model'], config['pricing']), **kwargs)
            results.append({'arm': spec['arm'], 'class': type(model).__name__, 'constructed': True})
        if browser_preflight:
            browser_budget = worker.Budget(root / 'browser-budget', 1, 1)
            try:
                worker.preflight(root, browser_budget)
            finally:
                environments.cleanup_registered_groups(root)
            results.append({'browser_preflight': True, 'model_requests': 0})
    return {'runtime': str(runtime), 'verified': results, 'paid_requests': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--browser-preflight', action='store_true', help='Start one local Chromium and the pinned BrowserCode bridge against a data URL; zero model calls')
    args = parser.parse_args()
    print(json.dumps(verify(args.runtime, browser_preflight=args.browser_preflight), indent=2))


if __name__ == '__main__':
    main()
