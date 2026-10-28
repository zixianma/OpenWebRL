#!/usr/bin/env python3
"""Prepare auditable WebArena-Lite task groups; no browser, API or scheduler work."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openwebrl.webprm_webarena import prepare_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True,
                        help='Pinned released WebArbiter checkout')
    parser.add_argument('--output', type=Path, required=True,
                        help='Private runtime directory; task payloads must not be published')
    parser.add_argument('--endpoints', type=Path,
                        help='Optional JSON mapping lowercase site names to credential-free URLs')
    args = parser.parse_args()
    output = args.output.resolve()
    protected = Path('/gpfs/home/zixianma')
    if output == protected or protected in output.parents:
        parser.error('Output must not be inside the protected home directory')
    endpoints = json.loads(args.endpoints.read_text()) if args.endpoints else None
    manifest, groups, configured = prepare_manifest(args.source, endpoints)
    # Fail before changing any artifact if a prior prepared cohort differs.
    artifacts = {'preparation-manifest.json': manifest,
                 **{name + '.json': values for name, values in groups.items()},
                 'endpoint-substituted-tasks.json': configured}
    serialized = {name: json.dumps(value, indent=2, sort_keys=True) + '\n'
                  for name, value in artifacts.items()}
    for name, payload in serialized.items():
        path = output / name
        if path.is_symlink():
            parser.error(f'Refusing an artifact symlink: {name}')
        if path.exists() and path.read_text() != payload:
            parser.error(f'Existing artifact differs; choose a fresh output directory: {name}')
    output.mkdir(parents=True, exist_ok=True)
    for name, payload in serialized.items():
        path = output / name
        if not path.exists():
            with path.open('x') as handle:
                handle.write(payload)
    print(json.dumps({'status': manifest['status'],
                      'task_group_counts': {name: item['count'] for name, item in manifest['task_groups'].items()},
                      'endpoint_substituted_task_count': len(configured),
                      'model_provider_calls': 0, 'gpu_allocation_seconds': 0}))


if __name__ == '__main__':
    main()
