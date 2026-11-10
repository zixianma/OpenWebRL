#!/usr/bin/env python3
"""Prepare private pinned task references, optionally resolving a native snapshot."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openwebrl.harmful_shopping_references import (
    CASES, prepare_manifest, private_path, reference_plan, resolve_development,
    resolve_task, write_new)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('shortlist', 'source', 'admission', 'fixture', 'output'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--snapshot', help='Complete independently captured native snapshot wrapper')
    args = parser.parse_args()
    root = private_path(args.output)
    root.mkdir(mode=0o700)
    manifest = prepare_manifest(args.shortlist, args.source, args.admission, args.fixture)
    write_new(root / 'manifest.json', manifest)
    if args.snapshot:
        snapshot = json.loads(Path(args.snapshot).read_text())
        resolved = [resolve_task(task, snapshot) for task in manifest['tasks']]
        resolved += resolve_development(manifest, snapshot)
        write_new(root / 'resolved-private.json', resolved)
        plans = [reference_plan(task, case) for task in resolved for case in CASES]
        write_new(root / 'plans-private.json', plans)
    print(json.dumps({'status': manifest['status'], 'source_tasks': len(manifest['tasks']),
        'development_tasks': len(manifest['development']), 'reference_cases': manifest['reference_case_count'],
        'admitted_tasks': 0, 'model_calls': 0, 'output': str(root)}))


if __name__ == '__main__':
    main()
