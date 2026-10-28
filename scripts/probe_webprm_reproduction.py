#!/usr/bin/env python3
"""Read-only saved-run diagnosis and separately approved, bounded API probes.

Preparation and audit never call a provider or start a browser. Live probes reuse
saved inputs; their accuracy is diagnostic and is not benchmark task success.
Every live attempt is claimed durably before dispatch and is never retried.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.parse import urlsplit

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
RUNTIME = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime')
WA = RUNTIME / 'webprm-webarena-20261011'
CAMPAIGN = WA / 'next-stage-preparation/reddit-table5-190-v3/campaign'
PREPARATION = WA / 'next-stage-preparation/reddit-table5-190-v13'
SOURCE = RUNTIME / 'webprm-baselines-20261009/sources/arbiter'
BROWSER_PYTHON = RUNTIME / 'webprm-baselines-20261009/browser-env/bin/python'
DEFAULT_ROOT = WA / 'reproduction-diagnostics-20261016-v3'
MINI = 'gpt-4o-mini-2024-07-18'
FOUR_O = 'gpt-4o-2024-05-13'
PRICES = {MINI: [0.15e-6, 0.60e-6], FOUR_O: [5e-6, 15e-6]}
BATCHES = ('terminal_model', 'state_diff_model', 'terminal_write_rule', 'terminal_temperature')
WRITE_RULE = ('\n\nWRITE-TASK CHECK: Finding an edit link, opening a form, or filling fields '
              'does not complete a requested change. Require visible evidence that the change '
              'has actually been saved or submitted. Never hand the remaining UI work to the '
              'user. If only a pending form is visible, return action=null. '
              'For READ tasks, retain the original completion rules.\n')
# These are deliberately selected development failures, plus a positive control.
# They are not a held-out sample and cannot estimate general benchmark accuracy.
CASE_SPECS = ((97, FOUR_O, False), (131, FOUR_O, False),
              (141, MINI, False), (141, FOUR_O, True))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def checked(path):
    path = Path(path).resolve()
    if not any(path == root or root in path.parents for root in (RUNTIME, Path('/tmp'))):
        raise ValueError('Output must stay in approved runtime or /tmp')
    return path


def write_once(path, value):
    path = checked(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def reference(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def verified_reference(ref):
    path = Path(ref['path'])
    if sha(path) != ref['sha256']:
        raise ValueError('Saved evidence hash changed: ' + str(path))
    return path


def journal_pairs(path):
    pending, pairs = {}, []
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        if row.get('event') == 'request':
            ident = row['reservation']
            if ident in pending:
                raise ValueError('Repeated saved reservation')
            pending[ident] = row
        elif row.get('event') == 'response':
            request = pending.pop(row['reservation'])
            pairs.append(dict(request=request, response=row['response']))
    return pairs


def inventory(campaign=CAMPAIGN):
    episodes = {}
    for path in sorted(Path(campaign).glob('attempt-*/cells/*/episodes/*/*__no_search/result.json')):
        task = int(path.parent.parent.name.rsplit('_', 1)[1])
        actor = path.parent.name.split('__')[0]
        key = (task, actor)
        if key in episodes:
            raise ValueError('Ambiguous duplicate primary episode: ' + repr(key))
        episodes[key] = path
    return episodes


def upstream_defaults(source=SOURCE):
    path = Path(source) / 'viewer/src/server/engine.py'
    tree = ast.parse(path.read_text())
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'LMEngine')
    method = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == '_to_lm_config')
    call = next(x for x in ast.walk(method) if isinstance(x, ast.Call)
                and isinstance(x.func, ast.Name) and x.func.id == 'ModelSpec')
    defaults = {x.arg: ast.literal_eval(x.value) for x in call.keywords}
    files = ['viewer/src/server/core/tools/terminal.py',
             'viewer/src/server/core/tools/state_diff.py']
    for filename, factory in zip(files, ('create_terminal_detector', 'create_state_diff_generator')):
        calls = [x for x in ast.walk(ast.parse((Path(source) / filename).read_text()))
                 if isinstance(x, ast.Call) and isinstance(x.func, ast.Name) and x.func.id == factory]
        if not any(not x.args and not x.keywords for x in calls):
            raise ValueError('Released auxiliary default-model route changed')
    return dict(defaults=defaults, source=[reference(path)] + [reference(Path(source) / p) for p in files])


def offline_analysis(episodes, cohort, source=SOURCE):
    from openwebrl.webprm_arbiter_harness import load_arbiter_harness
    harness = load_arbiter_harness(source)
    cells = cohort.get('cohort', cohort)['cells']
    expected = {(x['task_id'], x['actor']) for x in cells
                if x['selector'] == 'no_search' and x['domain'] == 'reddit'}
    if set(episodes) != expected:
        raise ValueError('Saved no-search inventory differs from the fixed cohort')
    records, counts, sampling, routes = [], Counter(), Counter(), Counter()
    for (task, actor), path in sorted(episodes.items()):
        result = read(path)
        rows = read(path.parent / 'rollout.json')
        last = rows[-1]['proposal'] if rows else {}
        stop = bool(last.get('terminal'))
        valid, success = bool(result['valid']), bool(result.get('success'))
        native = result.get('native_success')
        url = last.get('context', {}).get('current_url', '')
        form = any(x in urlsplit(url).path for x in ('/submit', '/create_forum', '/edit_biography', '/-/edit'))
        errors = Counter(row.get('execution_error') for row in rows if row.get('execution_error'))
        parse_errors = []
        for pair in journal_pairs(path.parent / 'api.jsonl'):
            req = pair['request']; payload = req['payload']
            if req.get('purpose') in ('policy', 'terminal', 'state_diff'):
                routes[(actor, req['purpose'], payload['model'])] += 1
                sampling[(req['purpose'], payload.get('temperature'), payload.get('top_p'), payload.get('max_tokens'))] += 1
            if req.get('purpose') == 'terminal':
                text = pair['response']['choices'][0]['message']['content']
                detection = harness.parse_terminal_response(text).terminal_action
                if detection is not None and detection.action:
                    try:
                        harness.parse_action(detection.action)
                    except Exception as error:
                        parse_errors.append(dict(type=type(error).__name__, message=str(error)[:500]))
        record = dict(task_id=task, actor=actor, valid=valid, success=success,
                      native_success=native, turns=len(rows), detector_override_at_last_turn=stop,
                      final_saved_url=url, detector_failure_at_write_form=bool(valid and stop and not success and form),
                      execution_errors=dict(errors), terminal_parser_errors=parse_errors,
                      protocol_fallback=result.get('protocol_fallback'), error=result.get('error'),
                      result=reference(path), api=reference(path.parent / 'api.jsonl'),
                      rollout=reference(path.parent / 'rollout.json'))
        records.append(record)
        counts['saved_episodes'] += 1
        counts['valid' if valid else 'invalid'] += 1
        counts['successes'] += int(success)
        counts['native_positives'] += int(bool(native))
        counts['detector_stop'] += int(valid and stop)
        counts['detector_stop_failures'] += int(valid and stop and not success)
        counts['detector_failure_at_write_form'] += int(record['detector_failure_at_write_form'])
        counts['turn_limit'] += int(bool(result.get('protocol_fallback')))
        counts['terminal_action_parse_errors'] += len(parse_errors)
    return dict(schema_version=1, status='offline_analysis_completed', counts=dict(counts), records=records,
                upstream=upstream_defaults(source),
                request_routes=[dict(actor=a, purpose=p, model=m, count=n) for (a, p, m), n in sorted(routes.items())],
                sampling=[dict(purpose=p, temperature=t, top_p=q, max_tokens=k, count=n)
                          for (p, t, q, k), n in sorted(sampling.items())],
                scope='Saved requests/actions/native outcomes only; no new inference or task successes.',
                api_calls=0, browser_launches=0)


def frozen_replay(path):
    folder = path.parents[3]
    plan = read(folder / 'worker-plan.json')
    roots = [Path(p).parents[1] for p in plan['code_sha256']
             if p.endswith('/scripts/audit_webprm_table5_cell.py')]
    if len(roots) != 1:
        raise ValueError('Frozen auditor root is ambiguous')
    root = roots[0]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
               PYTHONPATH=str(WA / 'nltk-runtime/venv/lib/python3.12/site-packages') + ':' + str(root),
               NLTK_DATA=str(WA / 'nltk-runtime/nltk_data'))
    supplement = CAMPAIGN / 'audit-replacements' / (folder.name + '.json')
    correction = None
    if supplement.exists():
        pointer = read(supplement)
        if pointer.get('protocol') == 'table5_empty_dispatch_audit_erratum_v1':
            audit_path = verified_reference(pointer['effective_audit'])
            audit = read(audit_path)
            if audit.get('audit_verified') is not True:
                raise ValueError('Effective erratum audit is not verified')
            correction = dict(pointer=reference(supplement), audit=reference(audit_path),
                              implementation=audit['erratum']['implementation_reference'],
                              source=audit['erratum']['corrected_source_reference'])
            verified_reference(correction['implementation'])
            verified_reference(correction['source'])
    code = '''import importlib.util,json,socket,sys
socket.socket.connect=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('Offline network forbidden'))
import scripts.audit_webprm_table5_cell as auditor
inputs=json.load(open(sys.argv[1]))
correction=json.loads(sys.argv[2])
if correction is not None:
    # Reproduce the preserved original rejection, then exactly its recorded erratum.
    try: auditor.verify(inputs)
    except ValueError as error:
        if str(error)!='Native dispatch differs from selected action': raise
    else: raise ValueError('Original rejection no longer reproduced')
    spec=importlib.util.spec_from_file_location('_saved_erratum',correction['implementation']['path'])
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    exec(module.replacement_functions(correction['source']['path']),auditor.__dict__)
v=auditor.verify(inputs)
if correction is not None and v['record']!=json.load(open(correction['audit']['path']))['record']:
    raise ValueError('Independent replay differs from the effective supplemental record')
print(json.dumps({'record':v['record'],'native_replay_verified':v['native_replay']['verified']}))
'''
    result = subprocess.run([str(BROWSER_PYTHON), '-B', '-c', code, str(folder / 'audit-input.json'), json.dumps(correction)],
                            cwd=root, env=env, text=True, capture_output=True, timeout=90)
    if result.returncode:
        return dict(verified=False, exit_code=result.returncode, error=result.stderr[-1800:])
    value = json.loads(result.stdout)
    return dict(verified=value['native_replay_verified'], record=value['record'], recorded_erratum=correction,
                audit_input=reference(folder / 'audit-input.json'), frozen_source=str(root))


def change_payload(payload, batch, variant):
    changed = deepcopy(payload)
    if variant == 'control':
        return changed
    if variant != 'treatment':
        raise ValueError('Unknown paired variant')
    if batch in ('terminal_model', 'state_diff_model'):
        changed['model'] = MINI
    elif batch == 'terminal_temperature':
        changed['temperature'] = 0.0
    elif batch == 'terminal_write_rule':
        if changed['messages'][0]['role'] != 'system':
            raise ValueError('System prompt must be explicit')
        changed['messages'][0]['content'] += WRITE_RULE
    else:
        raise ValueError('Unknown diagnostic factor')
    return changed


def prepare(root, *, replays=False):
    from openwebrl.webprm_arbiter_harness import load_arbiter_harness
    root = checked(root)
    if root.exists():
        raise ValueError('Preserve the existing preparation; use a new root or execute it')
    root.mkdir(parents=True, mode=0o700)
    episodes = inventory()
    analysis = offline_analysis(episodes, read(PREPARATION / 'cohort.json'))
    write_once(root / 'offline-analysis.json', analysis)
    if replays:
        results = []
        for task, actor in [(t, a) for t, a, _ in CASE_SPECS] + [(133, FOUR_O), (5, MINI)]:
            proof = frozen_replay(episodes[task, actor])
            results.append(dict(task_id=task, actor=actor, **proof))
            print(json.dumps(dict(stage='offline_native_replay', task_id=task, actor=actor, verified=proof['verified'])), flush=True)
        write_once(root / 'native-replays.json', dict(api_calls=0, cases=results, verified=all(x['verified'] for x in results)))
    cases, calls = [], []
    for task, actor, expected in CASE_SPECS:
        path = episodes[task, actor]
        case_id = str(task) + '__' + actor
        pairs = journal_pairs(path.parent / 'api.jsonl')
        selected = {p: [x for x in pairs if x['request'].get('purpose') == p][-1]
                    for p in ('terminal', 'state_diff')}
        cases.append(dict(case_id=case_id, task_id=task, actor=actor, expected_complete=expected,
                          label_basis='Pending write form versus visibly saved comment; development-only, not held-out.',
                          result=reference(path), api=reference(path.parent / 'api.jsonl'),
                          context=read(path.parent / 'rollout.json')[-1]['proposal']['context'], saved_pairs=selected))
    for batch in BATCHES:
        for ci, case in enumerate(cases):
            if batch in ('terminal_model', 'state_diff_model') and case['actor'] != FOUR_O:
                continue  # A mini-to-mini comparison has no changed variable.
            purpose = 'state_diff' if batch == 'state_diff_model' else 'terminal'
            base = case['saved_pairs'][purpose]['request']['payload']
            for draw in range(2):
                # Balance pair order across cases/draws; requests stay serial.
                variants = ('control', 'treatment') if (ci + draw) % 2 == 0 else ('treatment', 'control')
                for variant in variants:
                    payload = change_payload(base, batch, variant)
                    call_id = '__'.join((batch, case['case_id'], str(draw), variant))
                    calls.append(dict(call_id=call_id, batch=batch, case_id=case['case_id'], draw=draw,
                                      variant=variant, purpose=purpose, payload=payload,
                                      payload_sha256=digest(payload), expected_complete=case['expected_complete']))
    case_file = root / 'cases.json'
    write_once(case_file, dict(cases=cases))
    parent = CAMPAIGN / 'shared-budget.json'
    balance = read(parent)
    harness = load_arbiter_harness(SOURCE)
    source_refs = [reference(SOURCE / name) for name in harness.manifest['files']]
    implementation = [reference(REPO / name) for name in (
        'openwebrl/webprm_worker.py', 'openwebrl/webprm_runtime.py',
        'openwebrl/webprm_arbiter_harness.py', 'openwebrl/webprm_source.py',
        'openwebrl/webprm_lifecycle.py', 'openwebrl/jev_eval.py')]
    plan = dict(schema_version=1, protocol='saved_state_one_factor_webprm_probe_v1',
                status='prepared_not_executed_approval_required', created_utc=datetime.now(timezone.utc).isoformat(),
                scientific_scope='Four development states; paired fresh responses, two draws per condition. No browser rollouts or success-rate claims.',
                exact_reproduction_verified=False, cases=reference(case_file), source=source_refs,
                implementation=implementation,
                script=reference(__file__), batches=list(BATCHES), calls=calls,
                resources=dict(new_slurm_allocations=0, gpus=0, live_browsers=0, cpu_allocation_seconds=0,
                               cumulative_wall_seconds=1800, api_usd_cap=5.0, max_provider_calls=len(calls),
                               automatic_retries=0, max_output_tokens=2048),
                accounting=dict(parent_budget=reference(parent), parent_cap_usd=balance['cap'],
                                parent_charged_usd=balance['charged'], scope_transfer_usd=5.0,
                                combined_api_cap_usd=balance['cap'], no_added_api_budget=True),
                api_base='https://api.openai.com/v1', prices_usd_per_token=PRICES,
                env_file=read(PREPARATION / 'proposal.json')['env_file'],
                followup='Analyze each batch before the next; browser rollouts need a separately specified existing or newly approved allocation.')
    write_once(root / 'plan.json', plan)
    write_once(root / 'approval-template.json', dict(status='pending', plan_sha256=digest(plan), exact_user_approval=None))
    print(json.dumps(dict(status=plan['status'], root=str(root), plan_sha256=digest(plan),
                          counts=analysis['counts'], batch_calls=dict(Counter(c['batch'] for c in calls)),
                          resources=plan['resources'])))


def validate_approval(plan, approval):
    if (approval.get('status') != 'approved' or approval.get('plan_sha256') != digest(plan)
            or not isinstance(approval.get('exact_user_approval'), str) or not approval['exact_user_approval'].strip()):
        raise ValueError('Explicit approval of this exact diagnostic cap/scope is required')
    if plan.get('api_base') != 'https://api.openai.com/v1':
        raise ValueError('Only the explicitly disclosed OpenAI destination is allowed')
    verified_reference(plan['script'])
    verified_reference(plan['cases'])
    for ref in plan['source'] + plan['implementation']:
        verified_reference(ref)
    parent = read(verified_reference(plan['accounting']['parent_budget']))
    if (parent.get('active_epoch_id') is not None or parent['cap'] != plan['accounting']['combined_api_cap_usd']
            or parent['charged'] + plan['resources']['api_usd_cap'] > parent['cap']):
        raise ValueError('Closed parent campaign and remaining original API cap are required')
    resources = plan['resources']
    if (resources['new_slurm_allocations'] != 0 or resources['gpus'] != 0 or resources['live_browsers'] != 0
            or resources['automatic_retries'] != 0 or resources['max_provider_calls'] != len(plan['calls'])
            or len({c['call_id'] for c in plan['calls']}) != len(plan['calls'])):
        raise ValueError('Diagnostic resource/call identity changed')
    if (not math.isfinite(resources['api_usd_cap']) or resources['api_usd_cap'] <= 0
            or resources['cumulative_wall_seconds'] != 1800):
        raise ValueError('Explicit finite diagnostic caps required')
    for c in plan['calls']:
        if digest(c['payload']) != c['payload_sha256'] or c['payload']['model'] not in PRICES:
            raise ValueError('Unapproved diagnostic payload/model')


def parse_result(payload, response, *, purpose, source=SOURCE):
    from openwebrl.webprm_arbiter_harness import load_arbiter_harness
    from openwebrl.webprm_worker import single_response_text
    text = single_response_text(response)
    if purpose != 'terminal':
        return dict(parse_valid=True, diff_text=text, completion_claim=None)
    harness = load_arbiter_harness(source)
    terminal = harness.parse_terminal_response(text).terminal_action
    if terminal is None:
        return dict(parse_valid=False, completion_claim=None, error='Released terminal parser rejected response')
    if not terminal.action:
        return dict(parse_valid=True, completion_claim=False, action=terminal.action, thought=terminal.thought)
    try:
        action = harness.parse_action(terminal.action)
    except Exception as error:
        return dict(parse_valid=False, completion_claim=None, action=terminal.action,
                    error=type(error).__name__ + ': ' + str(error))
    return dict(parse_valid=True, completion_claim=action['kind'] == 'send_message_to_user',
                action=terminal.action, thought=terminal.thought)


def execute(root, approval_path, batch):
    root = checked(root)
    plan = read(root / 'plan.json')
    validate_approval(plan, read(approval_path))  # Before credentials, claims or provider work.
    from openwebrl.webprm_runtime import load_api_credentials
    from openwebrl.webprm_worker import completion
    load_api_credentials(plan['env_file'])
    lock_path = root / 'owner.lock'
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        batch_index = plan['batches'].index(batch)
        for earlier in plan['batches'][:batch_index]:
            report = read(root / (earlier + '-summary.json'))
            if not report['all_responses_recorded']:
                raise ValueError('Earlier batch has unresolved attempts; review before advancing')
        start_file = root / 'start.json'
        if not start_file.exists():
            write_once(start_file, dict(started_epoch=time.time(), plan_sha256=digest(plan)))
        start = read(start_file)
        if start['plan_sha256'] != digest(plan):
            raise ValueError('Resume must retain the frozen plan')
        deadline = start['started_epoch'] + plan['resources']['cumulative_wall_seconds']
        config = dict(api_usd_cap=plan['resources']['api_usd_cap'], budget_path=str(root / 'budget.json'),
                      deadline=deadline, env_file=plan['env_file'], api_base=plan['api_base'],
                      prices_usd_per_token=plan['prices_usd_per_token'])
        records = []
        for call in (c for c in plan['calls'] if c['batch'] == batch):
            folder = checked(root / 'calls' / call['call_id'])
            claim, result_path = folder / 'claim.json', folder / 'result.json'
            if result_path.exists():
                saved = read(result_path)
                if saved['call'] != call:
                    raise ValueError('Saved probe identity changed')
                records.append(saved)
                continue
            if claim.exists():
                raise ValueError('Unknown previous dispatch stays charged and must not be repeated')
            if time.time() >= deadline:
                raise TimeoutError('Cumulative diagnostic deadline exhausted; no reset on restart')
            write_once(claim, dict(call=call, plan_sha256=digest(plan), started_epoch=time.time()))
            saved = dict(call=call, response_recorded=False)
            try:
                response = completion(config, deepcopy(call['payload']), folder / 'api.jsonl', purpose=call['purpose'])
                write_once(folder / 'response.json', response)
                saved.update(response_recorded=True, response=reference(folder / 'response.json'))
                saved['parsed'] = parse_result(call['payload'], response, purpose=call['purpose'])
                if call['purpose'] == 'terminal':
                    saved['correct_on_development_state'] = bool(saved['parsed']['parse_valid'] and
                        saved['parsed']['completion_claim'] == call['expected_complete'])
            except Exception as error:
                saved['error'] = dict(type=type(error).__name__, message=str(error)[:600])
            write_once(result_path, saved)
            records.append(saved)
            print(json.dumps(dict(batch=batch, variant=call['variant'], case=call['case_id'],
                                  response_recorded=saved['response_recorded'], error=saved.get('error'))), flush=True)
            if not saved['response_recorded']:
                break
        expected = sum(c['batch'] == batch for c in plan['calls'])
        groups = {}
        for variant in ('control', 'treatment'):
            rows = [r for r in records if r['call']['variant'] == variant]
            groups[variant] = dict(attempts=len(rows), responses=sum(r['response_recorded'] for r in rows),
                                   parsed=sum(r.get('parsed', {}).get('parse_valid', False) for r in rows),
                                   correct_development_states=(sum(r.get('correct_on_development_state', False) for r in rows)
                                       if rows and rows[0]['call']['purpose'] == 'terminal' else None))
        summary = dict(batch=batch, all_responses_recorded=len(records) == expected and all(r['response_recorded'] for r in records),
                       expected_calls=expected, records=len(records), groups=groups,
                       budget=read(root / 'budget.json') if (root / 'budget.json').exists() else None,
                       benchmark_episodes=0, benchmark_successes_added=0,
                       caveat='Selected development states with two draws; neither held-out accuracy nor rollout success.')
        report_path = root / (batch + '-summary.json')
        if report_path.exists():
            if read(report_path) != summary:
                raise ValueError('Preserve existing summary; unresolved batches need explicit recovery review')
        else:
            write_once(report_path, summary)
        print(json.dumps(dict(batch=batch, all_responses_recorded=summary['all_responses_recorded'], groups=groups)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--execute', action='store_true')
    parser.add_argument('--verify-saved', action='store_true')
    parser.add_argument('--approval', type=Path)
    parser.add_argument('--batch', choices=BATCHES)
    args = parser.parse_args()
    if args.prepare:
        prepare(args.root, replays=args.verify_saved)
    elif args.approval and args.batch:
        execute(args.root, args.approval, args.batch)
    else:
        parser.error('--execute requires --approval and --batch')


if __name__ == '__main__':
    main()
