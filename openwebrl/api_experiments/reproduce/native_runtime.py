"""Assemble an isolated native runtime from hash-verified public sources.

The archived files remain byte-for-byte unchanged. Adaptations here replace
machine paths and campaign/Slurm source registration with a local hash inventory;
they retain the native agent, prompts, tools, transport and budget gateway.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
SUPPORT = Path(__file__).resolve().parent / 'portable_native'
CHROMIUM_SHA256 = '481fea1516a1f2b76454664272f12cd9dd1f20117b21e1f1498e08bc7f872c00'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_browser(chromium, browser_cache):
    """Require the exact Playwright 1.58.0 Linux Chromium 1208 executable."""
    chromium, browser_cache = Path(chromium), Path(browser_cache)
    if not chromium.is_file() or not os.access(chromium, os.X_OK) or not browser_cache.is_dir():
        raise ValueError('Install pinned Playwright Chromium before running')
    if digest(chromium) != CHROMIUM_SHA256:
        raise ValueError('Chromium binary differs from frozen Playwright1208 build')


def verify_source(source, name):
    source = Path(source).resolve()
    manifest = json.loads((SUPPORT / (name + '-source.json')).read_text())
    for rel, expected in manifest['hashes'].items():
        path = source / rel
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f'{name} pinned source mismatch: {rel}')
    return manifest


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise ValueError('Archived adaptation anchor changed: ' + before[:90])
    return source.replace(before, after, 1)


# No scheduler, historical approval record or host path is needed for a new run.
PORTABLE_BINDING = '''def bind_runtime_inventory(root,output):
 root=Path(root).resolve();output=Path(output).resolve()
 parts=output.relative_to(root).parts
 assert len(parts)==3 and parts[:2]==('shard0','episodes'),'Unexpected isolated output scope'
 plan=json.loads((root/'plan.json').read_text())
 assert output.name in plan['shards'][0]['task_ids'],'Unknown isolated task'
 inventory_path=root/'runtime-inventory.json'
 inventory_raw=inventory_path.read_bytes();inventory=json.loads(inventory_raw)
 assert inventory['passed'] is True
 assert inventory['frozen_plan_sha256']==hashlib.sha256((root/'plan.json').read_bytes()).hexdigest()
 for rel,expected in inventory['source_hashes'].items():
  p=(root/rel).resolve()
  assert p.is_relative_to(root) and hashlib.sha256(p.read_bytes()).hexdigest()==expected,'Runtime source changed: '+rel
 source_hashes=inventory['source_hashes']
 snapshot=output/'runtime-execution-inventory.json'
 with snapshot.open('xb') as f:f.write(inventory_raw)
 binding={'schema':'cuawright-portable-child-source-v1','shard':0,'pid':os.getpid(),'parent_pid':os.getppid(),'source_hashes':source_hashes,'original_plan_sha256':inventory['frozen_plan_sha256'],'registered_inventory_path':str(inventory_path),'registered_inventory_sha256':hashlib.sha256(inventory_raw).hexdigest(),'snapshot_path':str(snapshot),'snapshot_sha256':hashlib.sha256(inventory_raw).hexdigest()}
 (output/'runtime-child-source.json').write_text(json.dumps(binding,indent=2)+'\\n')
 return binding
'''


def adapt(name, text, chromium, browser_cache):
    if name in ('native_episode.py', 'diagnostic_browser.py'):
        text = replace_once(text, "'PLAYWRIGHT_BROWSERS_PATH':'/gpfs/scrubbed/zixianma/openwebrl-runtime/browsers'", "'PLAYWRIGHT_BROWSERS_PATH':" + repr(str(browser_cache)))
    elif name == 'browser_broker_support.py':
        text = replace_once(text, "'physical_chromium':'/gpfs/scrubbed/zixianma/openwebrl-runtime/browsers/chromium-1208/chrome-linux64/chrome'", "'physical_chromium':" + repr(str(chromium)))
    elif name == 'confinement.py':
        # venv --copies places Python outside the original system-bin allowlist.
        # Grant only that interpreter file, never its containing writable tree.
        before = "    allow(Path(playwright.__file__).parent / 'driver/node', 1)"
        text = replace_once(text, before, before + "\n    allow(Path(__import__('sys').executable).resolve(), 1)")
    elif name == 'browser_launcher.py':
        text = '#!' + sys.executable + '\n' + text.split('\n', 1)[1]
    elif name == 'external_judge.py':
        begin = text.index("PRIOR = Path(")
        end = text.index('\n', begin)
        text = text[:begin] + "PRIOR = ROOT / 'webjudge'" + text[end:]
    elif name == 'command_timeout_guard.py':
        text = text[:text.index('\ndef registered_shard_scope(')] + '\n' + PORTABLE_BINDING
    return text


def build_runtime(output, cuawright_source, webjudge_source, chromium, browser_cache, plan):
    """Write only a new run directory; never mutate either supplied checkout."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite or retry existing run: {output}')
    native = verify_source(cuawright_source, 'cuawright')
    judge = verify_source(webjudge_source, 'webjudge')
    chromium = Path(chromium).resolve()
    browser_cache = Path(browser_cache).resolve()
    verify_browser(chromium, browser_cache)
    spec = importlib.util.spec_from_file_location('_native_materialize', ROOT / 'materialize.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.materialize('cuawright_native', output)
    adaptations = []
    for path in sorted(output.glob('*.py')):
        original = digest(path)
        text = path.read_text()
        adapted = adapt(path.name, text, chromium, browser_cache)
        if text != adapted:
            path.write_text(adapted)
            adaptations.append({'file': path.name, 'archived_sha256': original, 'portable_sha256': digest(path)})
    (output / 'browser_launcher.py').chmod(0o755)
    for rel in native['hashes']:
        target = output / 'upstream' / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(cuawright_source) / rel, target)
    for rel in judge['hashes']:
        target = output / 'webjudge/official-source' / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(webjudge_source) / rel, target)
    (output / 'webjudge/deps').mkdir()
    (output / 'webjudge/plan.json').write_text(json.dumps({'source_hashes': {'official-source/' + k: v for k, v in judge['hashes'].items()}}, indent=2) + '\n')
    (output / 'confinement_support').mkdir()
    shutil.copyfile(SUPPORT / 'sitecustomize.py', output / 'confinement_support/sitecustomize.py')
    shutil.copyfile(SUPPORT / 'local_override.yaml', output / 'local_override.yaml')
    import yaml
    config = {}
    def merge(target, value):
        for key, child in value.items():
            if isinstance(child, dict) and isinstance(target.get(key), dict):
                merge(target[key], child)
            else:
                target[key] = child
    for path in [output / 'upstream/src/cuawright/webwright/config/best_default_judge_json_persistent_cli.yaml', output / 'upstream/src/cuawright/webwright/config/model_openai.yaml', output / 'local_override.yaml']:
        merge(config, yaml.safe_load(path.read_text()))
    (output / 'resolved-native-config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    (output / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    rates = {'gpt-6-luna': {'input': .1, 'cached_input': .01, 'cache_write_input': .125, 'output': .5}, 'gpt-6-sol': {'input': 2., 'cached_input': .2, 'cache_write_input': 2.5, 'output': 10.}}
    (output / 'pricing-snapshot.json').write_text(json.dumps({'usd_per_million_tokens': rates[plan['model']]}, indent=2) + '\n')
    (output / 'portable-adaptations.json').write_text(json.dumps({'adaptations': adaptations, 'cuawright_commit': native['source_commit'], 'webjudge_commit': judge['source_commit'], 'source_registration': 'isolated-run-inventory', 'failure_policy': 'one physical provider HTTP attempt; retain pinned native model defaults'}, indent=2) + '\n')
    sources = {}
    for path in sorted(output.rglob('*')):
        if path.is_file() and (path.suffix in ('.py', '.yaml') or 'upstream' in path.relative_to(output).parts):
            sources[str(path.relative_to(output))] = digest(path)
    inventory = {'schema': 'cuawright-portable-source-v1', 'passed': True, 'frozen_plan_sha256': digest(output / 'plan.json'), 'source_hashes': sources}
    (output / 'runtime-inventory.json').write_text(json.dumps(inventory, indent=2) + '\n')
    return output
