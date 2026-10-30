"""CPU import bridge for an exact frozen coordinate-browser study worker."""
from __future__ import annotations

import argparse
import ast
import asyncio
from contextlib import contextmanager
import hashlib
import importlib
import json
from pathlib import Path
from types import ModuleType
import sys
from argparse import Namespace


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def install_generation_state(source):
    """Execute the exact state classes without loading unrelated GPU router code.

    The browser evaluator only imports GenerateState. Its tokenizer, processor,
    singleton behavior and initialization are the unchanged frozen class bodies.
    No model-serving or reward-model functions are needed by hosted API policies.
    """
    import numpy as np
    from slime.utils.processing_utils import load_processor, load_tokenizer
    from slime.utils.types import Sample
    namespace = dict(asyncio=asyncio, np=np, contextmanager=contextmanager, Namespace=Namespace,
                     load_processor=load_processor, load_tokenizer=load_tokenizer, Sample=Sample)
    for filename, classname in [('slime/utils/misc.py', 'SingletonMeta'), ('slime/rollout/sglang_rollout.py', 'GenerateState')]:
        path = source / filename
        node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == classname)
        tree = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
        exec(compile(tree, str(path), 'exec'), namespace)
    module = ModuleType('slime.rollout.sglang_rollout')
    module.GenerateState = namespace['GenerateState']
    sys.modules[module.__name__] = module



async def browser_preflight(source):
    import os
    import tempfile
    import yaml
    from openwebrl.env.local_process_env import create_local_process_env
    with tempfile.TemporaryDirectory(prefix='legacy-browser-preflight-') as temporary:
        os.environ['SLIME_BROWSER_LOCAL_PROCESS_LOG_DIR'] = temporary
        os.environ['SLIME_BROWSER_LOCAL_PROCESS_PORT_LOCK_DIR'] = temporary
        local = await create_local_process_env({'port_start': 27000, 'port_end': 27100, 'max_processes': 1})
        try:
            cfg = yaml.safe_load((source / 'openwebrl/env/config.yaml').read_text())
            tools = json.loads((source / 'openwebrl/env/prompts/tool_list.json').read_text())
            policy = (source / 'openwebrl/env/prompts/system_prompt_browser_env.md').read_text()
            cfg['resize_output_coords'] = False
            await local.initialize('offline-preflight', {'intent': 'Return ready', 'start_url': 'data:text/html,<h1>ready</h1>'}, cfg, tools, policy)
            observation, info = await local.reset()
            if not observation.get('screenshot', b'').startswith(b'\x89PNG'):
                raise ValueError('Browser screenshot missing')
            result = await local.step([{'name': 'done', 'arguments': {'response': 'ready'}}])
            if not result[2]:
                raise ValueError('Native done action did not terminate')
        finally:
            await local.exit()
        if local.proc.returncode is None:
            raise ValueError('Owned browser server did not exit')
        return {'browser_preflight': True, 'paid_requests': 0, 'owned_server_exited': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--verify-imports', action='store_true')
    parser.add_argument('--browser-preflight', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    runtime = Path(config['runtime'])
    source = runtime / 'source'
    sys.path.insert(0, str(source))
    identity = json.loads((runtime / 'portable-runtime.json').read_text())
    for filename, entry in identity['source_manifest']['files'].items():
        if filename != 'openwebrl/env/config.yaml' and hashlib.sha256((source / filename).read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError('Frozen source changed: ' + filename)
    if identity.get('tokenizer_source'):
        from legacy import validate_tokenizer_source
        validate_tokenizer_source(identity['tokenizer_source'])
    install_generation_state(source)
    from openwebrl import run_evaluate  # Match frozen import order for browser hooks.
    worker = importlib.import_module('openwebrl.sol_serial_parallel_eval')
    if args.verify_imports:
        if args.browser_preflight:
            print(json.dumps(asyncio.run(browser_preflight(source))))
        print(json.dumps({'imports_verified': True, 'worker': str(Path(worker.__file__).resolve()), 'hosted_api_only': True}))
        return
    spec = json.loads((runtime / 'scientific-plan.json').read_text())
    identity = json.loads((runtime / 'portable-runtime.json').read_text())
    if digest(spec) != identity['scientific_plan_sha256']:
        raise ValueError('Scientific plan changed')
    for filename, entry in identity['source_manifest']['files'].items():
        if filename == 'openwebrl/env/config.yaml':
            continue  # Explicit task-file path overlay recorded during preparation.
        if hashlib.sha256((source / filename).read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError('Frozen source changed: ' + filename)
    output = Path(config['output'])
    tokenizer = config.get('tokenizer_source')
    if not tokenizer:
        from huggingface_hub import snapshot_download
        tokenizer = snapshot_download(spec['tokenizer_model'], revision=spec['actor_revision'],
            allow_patterns=['*.json', '*.txt', '*.model', '*.jinja'], cache_dir=output / 'huggingface-cache')
    from legacy import validate_tokenizer_source
    validate_tokenizer_source(tokenizer)
    ids = [row['task_id'] for row in spec['schedule'][:config['limit']]]
    schedule = [{**row, 'modes': [config['arm']] if config.get('arm') else row['modes']}
                for row in spec['schedule'][:config['limit']]]
    all_tasks = [json.loads(line) for line in (runtime / 'public-tasks.jsonl').read_text().splitlines()]
    taskfile = output / 'tasks.jsonl'
    taskfile.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in all_tasks if row['id'] in ids))
    caps = dict(actor_usd=config['model_usd'], actor_calls=config['model_calls'], judge_usd=config['judge_usd'],
                judge_calls=config['judge_calls'], browser_sessions=config['browser_starts'])
    plan = dict(spec, source=str(source), task_file=str(taskfile), task_file_sha256=hashlib.sha256(taskfile.read_bytes()).hexdigest(),
                task_ids=ids, schedule=schedule, output=str(output), budget_root=str(output / 'budgets'),
                judge_budget_root=str(output / 'budgets/judge'), limits=caps, actor_path=str(tokenizer),
                wandb_group='portable-' + spec['campaign'], wandb_prefix='portable-' + str(output.name))
    (output / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    worker_config = dict(plan=str(output / 'plan.json'), output=str(output), plan_sha256=digest(plan),
                         worker_id='portable', browser_port=config['browser_port'])
    original_protocol = spec['protocol']
    def validate_plan(candidate, worker_config):
        if candidate['protocol'] != original_protocol or worker_config['plan_sha256'] != digest(candidate):
            raise ValueError('Frozen episode protocol changed')
        worker.validate_browser_harness_protocol(candidate['protocol'])
        if candidate['family'] != 'luna' or candidate['limits'] != caps:
            raise ValueError('API family or explicit caps changed')
        return 'luna'
    worker.validate_plan = validate_plan
    worker.LIMITS = caps
    asyncio.run(worker.run(worker_config))
    expected = {(row['task_id'], mode) for row in schedule for mode in row['modes']}
    records = [json.loads(p.read_text()) for p in (output / 'records').glob('*.json')]
    if {(r['task_id'], r['mode']) for r in records} != expected or len(records) != len(expected):
        raise ValueError('Worker returned without every planned disposition')
    report = dict(complete=True, dispositions=len(records), valid=sum(r['valid'] for r in records),
                  successes=sum(r['valid'] and r['reward'] == 1 for r in records),
                  protocol_sha256=digest(original_protocol), original_campaign=spec['campaign'])
    (output / 'completion.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
