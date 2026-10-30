"""Saved-state calibration and matched selector-effort probes, using frozen source.

Inputs and outputs stay local. These experiments execute no browser actions and
make no task-success claims. Prepare from a fresh coordinate-browser run, or
supply a manifest matching portable_assets/snapshots/calibration/input-schema.json.
"""
from __future__ import annotations

import argparse
import ast
import asyncio
import concurrent.futures
from copy import deepcopy
from decimal import Decimal
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import threading

ASSETS = Path(__file__).resolve().parents[1] / 'portable_assets'
SEED = 'sol-selector-low-medium-probe-20261014-v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def new_directory(path):
    path = Path(path).resolve()
    if path.is_relative_to(Path('/gpfs/home/zixianma')):
        raise ValueError('Choose a project or temporary output directory')
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.mkdir(parents=True)
    return path


def frozen_module(kind, *, state_count=None):
    filename = 'run_pilot.py' if kind == 'calibration' else 'run_probe.py'
    source = ASSETS / 'snapshots' / kind / filename
    provenance = json.loads((ASSETS / 'provenance.json').read_text())
    if sha(source) != provenance['files'][str(source.relative_to(ASSETS))]['sha256']:
        raise ValueError('Frozen saved-state runner hash differs')
    tree = ast.parse(source.read_text())
    # The original calibration imports a host repository path. Only that path
    # changes; context builders, schemas, requests, response parsing and metering
    # execute the exact archived function bodies.
    if kind == 'calibration':
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'REPO' for t in node.targets):
                node.value = ast.parse('Path(' + repr(str(ASSETS / 'support')) + ')', mode='eval').body
    if kind == 'selector_effort' and state_count is not None:
        load = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'load_inputs')
        # Replace the original fixed100 input cardinality with the requested
        # bounded manifest size; every identity/schema/control check remains.
        for node in ast.walk(load):
            if isinstance(node, ast.Constant) and node.value == 100:
                node.value = state_count
    name = '_portable_snapshot_' + kind
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(ast.fix_missing_locations(tree), str(source), 'exec'), module.__dict__)
    return module


def prepare_calibration(states_file, output):
    import jsonschema
    states_file = Path(states_file).resolve()
    payload = json.loads(states_file.read_text())
    jsonschema.validate(payload, json.loads((ASSETS / 'snapshots/calibration/input-schema.json').read_text()))
    states = deepcopy(payload['states'])
    if len({s['state_id'] for s in states}) != len(states):
        raise ValueError('Duplicate state IDs')
    checked = []
    for state in states:
        image = state['images'][0]
        path = Path(image['path'])
        if not path.is_absolute():
            path = states_file.parent / path
        if sha(path) != image['sha256']:
            raise ValueError('Saved screenshot hash differs')
        checked.append(path)
    root = new_directory(output)
    module = frozen_module('calibration')
    module.ROOT = root
    (root / 'system-policy.txt').write_bytes((ASSETS / 'snapshots/calibration/system-policy.txt').read_bytes())
    for state, image in zip(states, checked):
        target = root / 'images' / (state['state_id'] + image.suffix)
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(image.read_bytes())
        state['images'][0]['path'] = str(target)
        module.context(state)
    write(root / 'states.json', {'states': states})
    write(root / 'portable-snapshot.json', {'kind': 'calibration', 'states': len(states), 'states_sha256': sha(root / 'states.json'),
        'system_policy_sha256': sha(root / 'system-policy.txt'), 'runner_sha256': sha(ASSETS / 'snapshots/calibration/run_pilot.py'),
        'wrapper_sha256': sha(Path(__file__)), 'cohort': 'collaborator-provided states; original private30 states are not redistributed'})
    return {'states': len(states), 'runtime': str(root), 'paid_requests': 0}


def extract_calibration(episodes, output, limit):
    if not 1 <= limit <= 30:
        raise ValueError('Calibration limit must be between1 and30')
    eligible = []
    for sample in sorted(Path(episodes).rglob('samples/**/*.json')):
        value = json.loads(sample.read_text())
        prompt = value.get('llm_input_texts')
        if not isinstance(prompt, str) or '<tools>' not in prompt:
            continue
        episode = next((p for p in sample.parents if (p / 'decisions').is_dir()), None)
        if episode is None:
            continue
        turn = int(value.get('total_steps', 0)) - 1
        screenshot = episode / 'screenshots' / f'{turn:03d}.png'
        if turn < 0 or not screenshot.is_file():
            continue
        sid = hashlib.sha256(canonical([str(sample.relative_to(episodes)), sha(screenshot), prompt]).encode()).hexdigest()[:24]
        eligible.append({'state_id': sid, 'prompt': prompt, 'images': [{'path': str(screenshot.resolve()), 'sha256': sha(screenshot), 'format': 'png'}]})
    # Outcome-independent deterministic order on local artifact identities.
    states = sorted(eligible, key=lambda s: s['state_id'])[:limit]
    if not states:
        raise ValueError('No saved coordinate-browser prompts/screenshots found; run legacy.py first or supply a state manifest')
    import tempfile
    with tempfile.TemporaryDirectory(prefix='snapshot-manifest-') as temporary:
        manifest = Path(temporary) / 'states.json'
        write(manifest, {'states': states})
        return prepare_calibration(manifest, output)


def safe_caps(args, calls):
    if args.model_usd is None or not math.isfinite(args.model_usd) or args.model_usd <= 0:
        raise ValueError('Execution requires a positive finite --model-usd cap')
    if args.model_calls is not None and args.model_calls < calls:
        raise ValueError('The explicit call cap is smaller than the prepared complete design')
    if not os.environ.get('OPENAI_API_KEY'):
        raise ValueError('OPENAI_API_KEY is required')


def run_calibration(args):
    root = args.runtime.resolve()
    record = json.loads((root / 'portable-snapshot.json').read_text())
    if record['wrapper_sha256'] != sha(Path(__file__)):
        raise ValueError('Snapshot wrapper changed; prepare a new immutable runtime')
    if sha(root / 'states.json') != record['states_sha256'] or sha(root / 'system-policy.txt') != record['system_policy_sha256']:
        raise ValueError('Prepared calibration inputs changed')
    states = json.loads((root / 'states.json').read_text())['states']
    module = frozen_module('calibration');module.ROOT = root
    for state in states:
        module.context(state)
    calls = 5 * len(states)
    if not args.execute:
        return {'states': len(states), 'maximum_calls': calls, 'execute': False, 'protocol': json.loads((ASSETS / 'snapshots/calibration/protocol.json').read_text())}
    safe_caps(args, calls)
    if (root / 'ledger.json').exists() or (root / 'started.json').exists():
        raise ValueError('Prior attempt exists; preserved without an automatic retry')
    module.CAP = Decimal(str(args.model_usd));module.HALT = threading.Event()
    with (root / 'started.json').open('x') as stream:
        json.dump({'model_usd': args.model_usd, 'maximum_calls': calls, 'wrapper_sha256': sha(Path(__file__))}, stream)
    key = os.environ['OPENAI_API_KEY']
    module.run_state(states[0], key)
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(module.run_state, state, key) for state in states[1:]]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    report = {'complete': True, 'states': len(states), 'ledger': json.loads((root / 'ledger.json').read_text()), 'task_success_measured': False}
    write(root / 'completion.json', report)
    return report


def prepare_medium(calibration, output):
    source = Path(calibration).resolve()
    states = json.loads((source / 'states.json').read_text())['states']
    rows = []
    for state in states:
        high = source / 'requests' / (state['state_id'] + '--serial_high.json')
        receipt = json.loads(high.read_text())
        if receipt.get('status') != 'received' or 'action' not in receipt:
            raise ValueError('Medium control requires each original high action receipt')
        body = dict(receipt['request'], reasoning={'effort': 'medium'}, max_output_tokens=4096)
        rows.append({'state_id': state['state_id'], 'request': body, 'high_receipt': str(high), 'high_receipt_sha256': sha(high)})
    root = new_directory(output)
    write(root / 'requests.json', rows)
    write(root / 'portable-snapshot.json', {'kind': 'medium_actor', 'requests_sha256': sha(root / 'requests.json'), 'source_calibration': str(source), 'wrapper_sha256': sha(Path(__file__))})
    return {'runtime': str(root), 'states': len(rows), 'paid_requests': 0}


def run_medium(args):
    root = args.runtime.resolve();record = json.loads((root / 'portable-snapshot.json').read_text())
    if record['wrapper_sha256'] != sha(Path(__file__)):
        raise ValueError('Snapshot wrapper changed; prepare a new immutable runtime')
    if sha(root / 'requests.json') != record['requests_sha256']:
        raise ValueError('Prepared requests changed')
    rows = json.loads((root / 'requests.json').read_text())
    for row in rows:
        if sha(row['high_receipt']) != row['high_receipt_sha256']:
            raise ValueError('Original high receipt changed')
        prior = json.loads(Path(row['high_receipt']).read_text())['request']
        if row['request'] != dict(prior, reasoning={'effort': 'medium'}, max_output_tokens=4096):
            raise ValueError('Medium control differs from its high input')
    if not args.execute:
        return {'states': len(rows), 'maximum_calls': len(rows), 'model': 'gpt-6.1-sol', 'effort': 'medium', 'output_tokens': 4096, 'execute': False}
    safe_caps(args, len(rows))
    if (root / 'started.json').exists():
        raise ValueError('Prior attempt exists; no automatic retry')
    write(root / 'started.json', {'model_usd': args.model_usd, 'maximum_calls': len(rows)})
    module = frozen_module('calibration');module.ROOT = root;module.CAP = Decimal(str(args.model_usd));module.HALT = threading.Event()
    for row in rows:
        receipt = module.call(row['state_id'], 'actor_medium_probe', row['request'], os.environ['OPENAI_API_KEY'])
        if 'action' not in receipt:
            raise ValueError('Invalid medium action; preserve it without retry')
    report = {'complete': True, 'states': len(rows), 'ledger': json.loads((root / 'ledger.json').read_text()), 'task_success_measured': False}
    write(root / 'completion.json', report)
    return report


def probe_module(root, count):
    module = frozen_module('selector_effort', state_count=count)
    module.SOURCE = root / 'support'
    return module


def prepare_selector(episodes, output, limit):
    if not 1 <= limit <= 100:
        raise ValueError('Selector probe limit must be between1 and100')
    module = frozen_module('selector_effort')
    eligible = []
    for path in sorted(Path(episodes).rglob('decisions/*.json')):
        decision = json.loads(path.read_text())
        if decision.get('status') != 'accepted' or len(decision.get('candidates', [])) != 3:
            continue
        receipts = []
        for p in path.parent.parent.glob('requests/*.json'):
            row = json.loads(p.read_text())
            if (row.get('physical_role') == 'selector' and row.get('request', {}).get('reasoning') == {'effort': 'medium'}
                    and canonical(row.get('response')) == canonical(decision.get('selector_response'))):
                receipts.append((p, row))
        if len(receipts) != 1:
            continue
        receipt, original = receipts[0]
        if module.parse_response(original['response']) != decision['selected_index']:
            raise ValueError('Saved selector receipt/decision identity differs')
        sid = hashlib.sha256(canonical([decision['task_id'], decision['turn'], sha(path)]).encode()).hexdigest()[:24]
        eligible.append(dict(state_id=sid, task_id=decision['task_id'], turn=decision['turn'], decision=str(path.resolve()),
            decision_sha256=sha(path), original_selector_receipt=str(receipt.resolve()), original_selector_receipt_sha256=sha(receipt),
            candidate_actions=decision['candidates'], original_selected_index=decision['selected_index'], original_request=original['request']))
    # Original sampling strategy: one hashed decision per task, then remaining
    # hashed decisions. The new sample is explicitly a fresh collaborator cohort.
    ranked = sorted(eligible, key=lambda x: hashlib.sha256((SEED + '/' + x['state_id']).encode()).hexdigest())
    chosen=[];seen=set()
    for row in ranked:
        if row['task_id'] not in seen:
            chosen.append(row);seen.add(row['task_id'])
    chosen.extend(row for row in ranked if row not in chosen)
    chosen=chosen[:limit]
    if not chosen:
        raise ValueError('No accepted three-candidate medium-selector states; collect a legacy medium-selector run first')
    root = new_directory(output)
    manifests=json.loads((ASSETS / 'legacy/sol61_om2w100_medium/source-manifest.json').read_text())['files']
    helpers={}
    for name in ['responses_actor_history.py', 'luna_qwen_metrics.py']:
        entry=manifests['openwebrl/'+name];target=root/'support'/name;target.parent.mkdir(exist_ok=True);target.write_bytes((ASSETS/entry['published_path']).read_bytes());helpers[name]=sha(target)
    entries=[]
    for index, row in enumerate(chosen):
        request=row.pop('original_request');bindings={}
        for effort in ['low','medium']:
            body=dict(request,reasoning={'effort':effort});module.validate_request(body,effort)
            relative=f'prepared-requests/{row["state_id"]}-{effort}.json';write(root/relative,body);bindings[effort]={'path':relative,'sha256':sha(root/relative)}
        row.update(order=['low','medium']if index%2==0 else ['medium','low'],requests=bindings);entries.append(row)
    write(root/'request-manifest-private.json',entries)
    caps=dict(module.CAPS,sol_calls=2*len(entries),sol_usd=0.)
    write(root/'plan.json',{'schema':'portable-selector-effort-probe-v1','caps':caps,'pricing':module.PRICING,'audited_sources':helpers,
        'protocol':json.loads((ASSETS/'snapshots/selector_effort/protocol.json').read_text()),'cohort':'fresh collaborator states; historical private100 states are not redistributed'})
    write(root/'portable-snapshot.json',{'kind':'selector_effort','states':len(entries),'manifest_sha256':sha(root/'request-manifest-private.json'),'wrapper_sha256':sha(Path(__file__))})
    return {'runtime':str(root),'states':len(entries),'maximum_calls':2*len(entries),'paid_requests':0}


def run_selector(args):
    root=args.runtime.resolve();record=json.loads((root/'portable-snapshot.json').read_text())
    if record['wrapper_sha256'] != sha(Path(__file__)):
        raise ValueError('Snapshot wrapper changed; prepare a new immutable runtime')
    if sha(root/'request-manifest-private.json')!=record['manifest_sha256']:
        raise ValueError('Prepared selector manifest changed')
    module=probe_module(root,record['states']);plan=json.loads((root/'plan.json').read_text());module.CAPS=plan['caps']
    if not args.execute:
        proof,_=module.preflight(root)
        proof['status']='validated_dry_run'
        return {'states':record['states'],'execute':False,'maximum_calls':2*record['states'],'preflight':proof}
    safe_caps(args,2*record['states'])
    if (root/'run/started.json').exists():
        raise ValueError('Prior attempt exists; no automatic retry')
    module.CAPS=dict(module.CAPS,sol_usd=args.model_usd);plan['caps']=module.CAPS;write(root/'plan.json',plan)
    original_preflight=module.preflight
    def portable_preflight(root, require_approval=False):
        proof, loaded = original_preflight(root, require_approval=False)
        proof['status'] = 'validated_with_explicit_run_caps'
        return proof, loaded
    module.preflight = portable_preflight
    module.credentials=lambda:(os.environ['OPENAI_API_KEY'],os.environ.get('OPENAI_BASE_URL'))
    asyncio.run(module.run(root))
    result=json.loads((root/'run/completion.json').read_text())
    if result['pairs']!=record['states']:
        raise ValueError('Missing paired probe outputs')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    for command in ['extract-calibration','prepare-selector']:
        p=sub.add_parser(command);p.add_argument('--episodes',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--limit',type=int,default=30 if command=='extract-calibration'else 100)
    p=sub.add_parser('prepare-calibration');p.add_argument('--states',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p=sub.add_parser('prepare-medium');p.add_argument('--calibration-output',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    for command in ['run-calibration','run-medium','run-selector']:
        p=sub.add_parser(command);p.add_argument('--runtime',required=True,type=Path);p.add_argument('--execute',action='store_true');p.add_argument('--model-usd',type=float);p.add_argument('--model-calls',type=int)
    args=parser.parse_args()
    try:
        if args.command=='extract-calibration':result=extract_calibration(args.episodes,args.output,args.limit)
        elif args.command=='prepare-calibration':result=prepare_calibration(args.states,args.output)
        elif args.command=='prepare-medium':result=prepare_medium(args.calibration_output,args.output)
        elif args.command=='prepare-selector':result=prepare_selector(args.episodes,args.output,args.limit)
        elif args.command=='run-calibration':result=run_calibration(args)
        elif args.command=='run-medium':result=run_medium(args)
        else:result=run_selector(args)
        print(json.dumps(result,indent=2))
    except (ValueError,FileExistsError)as error:parser.error(str(error))


if __name__=='__main__':main()
