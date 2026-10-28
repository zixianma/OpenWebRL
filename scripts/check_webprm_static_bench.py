#!/usr/bin/env python3
"""Validate WEBPRMBench data identity and released metrics without model calls."""
import argparse
import json
from pathlib import Path

from openwebrl.webprm_native_release import benchmark_inventory, reference
from openwebrl.webprm_static_bench import metric_fixture


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', required=True)
    ap.add_argument('--evaluator', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    result = {'inventory': benchmark_inventory(args.data), 'evaluator': reference(args.evaluator),
              'fixture': metric_fixture(args.evaluator, output / 'metric-fixture'),
              'new_api_calls': 0, 'new_allocations': 0, 'benchmark_predictions': 0}
    with (output / 'validation.json').open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps({'verified': True, 'pairs': result['inventory']['pairs'],
                      'states': result['inventory']['states'], 'benchmark_predictions': 0}))


if __name__ == '__main__':
    main()
