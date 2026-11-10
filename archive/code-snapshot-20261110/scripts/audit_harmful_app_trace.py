#!/usr/bin/env python3
"""Audit a captured application campaign or five-clone immediate-effect panel.

Offline only; performs no application, browser or model calls. Live captures
require externally retained seals. Neither unit fixtures nor this consistency
audit alone establish deployment readiness or native-collector completeness.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openwebrl.harmful_app_trace import evaluate_trace, panel_summary, InvalidTrace


def strict_json(path):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError('Duplicate JSON key: '+key)
            value[key] = item
        return value

    def reject(value):
        raise ValueError('Non-finite JSON number: '+value)

    return json.loads(Path(path).read_text(), object_pairs_hook=unique, parse_constant=reject)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--trace', type=Path)
    group.add_argument('--panel', type=Path, help='Manifest with five {path,seal_sha256} traces and selected_index')
    parser.add_argument('--expected-seal', help='Independently retained campaign seal; required for live evidence')
    args = parser.parse_args()
    try:
        if args.trace:
            result = evaluate_trace(strict_json(args.trace), expected_seal=args.expected_seal)
        else:
            manifest = strict_json(args.panel)
            entries = manifest['traces']
            traces = [strict_json(args.panel.parent / entry['path']) for entry in entries]
            result = panel_summary(traces, manifest['selected_index'],
                                   expected_seals=[entry.get('seal_sha256') for entry in entries])
    except (InvalidTrace, ValueError, KeyError, TypeError, OSError) as exc:
        result = dict(valid=False, invalid_reasons=[str(exc)])
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result['valid'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
