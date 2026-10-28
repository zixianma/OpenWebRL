"""Load released WebArbiter tools without importing its optional UI services.

No agent policy, parser, filter, deduplication or tournament is reimplemented.
The namespace setup only replaces the release's eager package initializers.
Live model transport is deliberately not supplied by this preparation module.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
import sys
import types
from collections import Counter, defaultdict
from pathlib import Path

ARBITER_REVISION = '764ec2670750c50388bac56efb1a3989fd5c1b54'
BENCHMARK_SHA256 = '9cc5a462fcf9b6c5b45cbe76425111dd47105b5cabae240feceba9b6571e8cc3'


def reference(path):
    path = Path(path).resolve()
    checksum = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024**2), b''):
            checksum.update(block)
    return {'path': str(path), 'sha256': checksum.hexdigest()}


def namespace(name, path):
    path = Path(path).resolve()
    existing = sys.modules.get(name)
    if existing is not None:
        if [str(Path(p).resolve()) for p in getattr(existing, '__path__', [])] != [str(path)]:
            raise RuntimeError(f'Conflicting native package namespace: {name}')
        return
    package = types.ModuleType(name)
    package.__path__ = [str(path)]
    sys.modules[name] = package


def load_tools(source):
    source = Path(source).resolve()
    revision = subprocess.check_output(
        ['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != ARBITER_REVISION:
        raise ValueError('Unexpected released source revision')
    viewer = source / 'viewer'
    # The release mixes webarbiter.src.* and prompts.* absolute imports.
    namespace('webarbiter', viewer)
    namespace('webarbiter.src', viewer / 'src')
    namespace('prompts', viewer / 'src/prompts')
    names = {
        'engine': 'server.engine',
        'sampling': 'server.core.tools.sampling',
        'policy': 'server.core.tools.policy',
        'reward': 'server.core.tools.reward',
        'tournament': 'server.core.tools.tournament',
        'terminal': 'server.core.tools.terminal',
        'history': 'server.service.utils.trajectory_utils',
    }
    # Avoid service/__init__.py, which eagerly creates UI/DB dependencies.
    namespace('webarbiter.src.server.service', viewer / 'src/server/service')
    loaded = {key: importlib.import_module('webarbiter.src.' + name)
              for key, name in names.items()}
    refs = []
    for name, module in sorted(sys.modules.items()):
        if name.startswith(('webarbiter.', 'prompts.')) and getattr(module, '__file__', None):
            path = Path(module.__file__).resolve()
            if not path.is_relative_to(source):
                raise RuntimeError(f'Native module resolved outside pinned source: {name}')
            refs.append(dict(module=name, **reference(path)))
    return loaded, refs


def benchmark_inventory(path):
    path = Path(path)
    groups = defaultdict(list)
    fields = Counter()
    checksum = hashlib.sha256()
    byte_count = row_count = 0
    with path.open('rb') as handle:
        for line in handle:
            checksum.update(line)
            byte_count += len(line)
            if not line.strip():
                continue
            row = json.loads(line)
            if set(row) != {'state_idx', 'pair_idx', 'expected', 'prompt', 'source_name'}:
                raise ValueError('Unexpected benchmark schema')
            if row['expected'] not in (1, 2) or not isinstance(row['prompt'], str) or not row['prompt']:
                raise ValueError('Malformed preference pair')
            groups[(row['source_name'], row['state_idx'])].append(row['pair_idx'])
            fields.update(row.keys())
            row_count += 1
    if checksum.hexdigest() != BENCHMARK_SHA256:
        raise ValueError('Official benchmark byte identity changed')
    if row_count != 4600 or len(groups) != 1150:
        raise ValueError('Official benchmark denominator mismatch')
    if any(len(panel) != 4 or set(panel) != set(range(4))
           for panel in groups.values()):
        raise ValueError('Incomplete or duplicate preference panel')
    return {
        'dataset': {'path': str(path.resolve()), 'sha256': checksum.hexdigest()},
        'bytes': byte_count, 'pairs': row_count,
        'states': len(groups), 'state_identity': ['source_name', 'state_idx'],
        'fields': dict(fields),
        'states_by_source': dict(Counter(key[0] for key in groups)),
        'screenshots_in_release': False,
        'evaluation': 'Official rendered text prompts and official pairwise/all-four-correct metrics',
        'piotr_protocol': 'Text-only checkpoint evaluation; not native screenshot SelectionARM inference',
    }
