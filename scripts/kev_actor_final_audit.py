#!/usr/bin/env python3
"""Offline audit of direct Kev actor artifacts; never calls models or browsers.

Writes only explicitly requested report/HTML paths. A passing evidence audit is
not permission to set a completion pointer: scheduler, W&B and publication must
also be closed out by the owning supervisor. Invalid diagnoses never relabel a
canonical verdict or waive request/trajectory consistency checks.
"""
import argparse
import ast
import base64
from collections import Counter
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import time

REPO = Path(__file__).resolve().parents[1]
TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED'}
KINDS = {'kev': 'local_kev_requests', 'text': 'text_http_attempts', 'judge': 'judge_http_attempts'}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def require(condition, message):
    if not condition:
        raise ValueError(message)


def image_bytes(value):
    from PIL import Image
    raw = base64.b64decode(value, validate=True)
    with Image.open(io.BytesIO(raw)) as image:
        require(image.format == 'JPEG' and min(image.size) > 0, 'Invalid JPEG evidence')
        image.verify()
    return raw


def canonical_messages(task, history, screenshot, terminal, source):
    values = {}
    for node in ast.parse(Path(source).read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {'SYSTEM_PROMPT', 'USER_PROMPT'}:
                    values[target.id] = ast.literal_eval(node.value)
    actions = []
    for entry in history:
        action = entry['action']
        if entry.get('text') is not None:
            action += ' ' + json.dumps({'text': entry['text']}, ensure_ascii=False)
        actions.append(action.replace('\n\n', ' '))
    actions.append(f'Agent stopped: {terminal} (this is not a success verdict)')
    trajectory = '\n\n'.join(f'Thought {i}: \nAction {i}: {a}' for i, a in enumerate(actions, 1))
    return [{'role': 'system', 'content': values['SYSTEM_PROMPT']},
            {'role': 'user', 'content': [
                {'type': 'text', 'text': values['USER_PROMPT'].format(task=task['intent'], thoughts_and_actions=trajectory)},
                {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + screenshot, 'detail': 'high'}}]}]


def choice(answer, criteria):
    probabilities = answer['probabilities']
    require(set(probabilities) == set(criteria) and answer['choice'] in criteria, 'Choice criteria mismatch')
    require(all(type(p) in (int, float) and math.isfinite(p) and 0 <= p <= 1
                for p in [*probabilities.values(), answer['confidence']]), 'Invalid probability/confidence')
    # Match the pinned actor validator, including its serialized rounding tolerance.
    require(abs(sum(probabilities.values()) - 1) < .02, 'Probability mass mismatch')
    require(probabilities[answer['choice']] >= max(probabilities.values()) - 1e-6, 'Non-argmax choice')


def validate_decision(decision, request, response):
    require(decision['request'] == request and decision['raw_answers'] == response['answers'], 'Decision/API mismatch')
    require(decision['model'] == response['model'] == 'kev-latest' and not response.get('truncated'), 'Kev identity/truncation mismatch')
    require(set(request['state']) == {'page', 'elements', 'recent_actions'}, 'Unexpected actor input fields')
    require(set(request['state']['page']) == {'url', 'title', 'text'}, 'Actor page contains non-text input')
    answer = response['answers']['operation']
    choice(answer, request['questions']['operation']['criteria'])
    require(decision['operation'] == answer['choice'] and decision['operation_probabilities'] == answer['probabilities']
            and decision['confidence'] == answer['confidence'], 'Selected operation mismatch')
    require(decision['usage'] == response.get('usage', {}), 'Decision usage mismatch')
    if decision['target'] is not None:
        name = decision['operation'].lower() + '_target'
        target = response['answers'][name]
        choice(target, request['questions'][name]['criteria'])
        require(decision['target'] == target['choice'] and decision['target_probabilities'] == target['probabilities']
                and decision['target_confidence'] == target['confidence'], 'Selected target mismatch')
        require(math.isclose(decision['probabilities'][decision['choice']], target['probabilities'][target['choice']]),
                'Selected target probability mismatch')
    else:
        require(decision['operation'] not in {'CLICK', 'TYPE_TEXT', 'SELECT'}, 'Target operation lacks target')


def invalid_diagnosis(root, directory, result, diagnosis):
    require(not result['valid'] and result['score'] is None, 'Invalid diagnosis cannot relabel scored evidence')
    require(diagnosis and diagnosis.get('task_id') == result['task_id'], 'Invalid result needs explicit diagnosis')
    require(diagnosis.get('result_sha256') == sha(directory/'result.json'), 'Invalid diagnosis result hash mismatch')
    require(bool(diagnosis.get('reason')) and diagnosis.get('expected_actor_error') == result.get('actor_error')
            and diagnosis.get('expected_judge_error') == result.get('judge_error'), 'Invalid diagnosis error mismatch')
    evidence = diagnosis.get('evidence_sha256', {})
    required = {str((directory/name).relative_to(root)) for name in ('result.json', 'trajectory.json', 'worker.log')}
    require(required <= set(evidence), 'Invalid diagnosis lacks immutable result/trajectory/log evidence')
    for name, expected in evidence.items():
        path = (root/name).resolve()
        require(path.is_relative_to(root.resolve()) and sha(path) == expected, 'Invalid diagnosis evidence hash mismatch: ' + name)
    require(bool(result.get('actor_error') or result.get('judge_error') or result.get('cleanup_errors')),
            'Invalid diagnosis lacks recorded failure')


def saved_actor_metadata(task, trajectory, config, attempts):
    """Result-shaped actor facts only; no invented response or verdict."""
    counts = Counter(a['provider'] for a in attempts)
    return dict(task_id=task['task_id'], completed=True, valid=True, decision_provider='kev',
                decision_checkpoint=config['kev']['run'], requested_jev_model=config['jev_model'],
                actions=len(trajectory['history']), decisions=len(trajectory['decisions']),
                terminal=trajectory['terminal'], actor_error=trajectory['error'], cleanup_errors=trajectory['cleanup_errors'],
                provider_blocked=False, api_attempts={p: counts[p] for p in KINDS},
                resolved_jev_models=sorted({d['model'] for d in trajectory['decisions']}))


def audit_task(root, task, config, judge_source, diagnosis=None, cohort_name='actor', *, _pending_judge=False):
    directory = root/cohort_name/'tasks'/digest(task['task_id'])
    trajectory = read(directory/'trajectory.json')
    if _pending_judge:
        require(not (directory/'result.json').exists() and not (directory/'judge-response.json').exists(),
                'Saved judge recovery cannot replace a result or verdict')
        require(trajectory['terminal'] in {'done', 'blocked'} and trajectory['error'] is None
                and not trajectory['cleanup_errors'], 'Saved judge recovery requires a clean terminal actor')
        result = saved_actor_metadata(task, trajectory, config, lines(directory/'api-attempts.jsonl'))
    else:
        result = read(directory/'result.json')
    require(read(directory/'task.json') == trajectory['task'] == task, 'Task identity mismatch')
    require(result['task_id'] == task['task_id'] and result['completed'], 'Result identity/completion mismatch')
    require(result['decision_provider'] == 'kev' and result['decision_checkpoint'] == config['kev']['run']
            and result['requested_jev_model'] == 'kev-latest', 'Result actor identity mismatch')
    history, decisions = trajectory['history'], trajectory['decisions']
    require(result['actions'] == len(history) <= config['max_steps'], 'Action count/cap mismatch')
    require(result['decisions'] == len(decisions) <= config['max_decisions'], 'Decision count/cap mismatch')
    require(result['terminal'] == trajectory['terminal'] and result['actor_error'] == trajectory['error']
            and result['cleanup_errors'] == trajectory['cleanup_errors'], 'Trajectory termination mismatch')
    marker = read(directory/'browser-session.json') if (directory/'browser-session.json').exists() else None
    require(marker is None or marker.get('stopped') is True, 'Browser session remains active')
    if result['valid']:
        require(marker and not result['provider_blocked'] and not result['cleanup_errors'], 'Valid result has provider/cleanup failure')
    attempts, responses = lines(directory/'api-attempts.jsonl'), lines(directory/'api-responses.jsonl')
    indexed = {r['call_id']: r for r in responses}
    require(len(indexed) == len(responses), 'Duplicate response IDs')
    require(len({r['call_id'] for r in attempts}) == len(attempts), 'Duplicate attempt IDs')
    counts = Counter(a['provider'] for a in attempts)
    require({p: counts[p] for p in KINDS} == result['api_attempts'], 'Result HTTP accounting mismatch')
    require(set(indexed) <= {a['call_id'] for a in attempts}, 'Response without charged attempt')
    successful = {p: [] for p in KINDS}
    for attempt in attempts:
        provider, request = attempt['provider'], attempt['request']
        require(provider in KINDS and attempt['model'] == request['model'], 'API request identity mismatch')
        if provider == 'text':
            require(request['model'] == config['text_model'] and request['temperature'] == .6 and request['top_p'] == .95
                    and request['max_tokens'] == config['text_max_tokens'], 'Text helper protocol mismatch')
            require(request['response_format'] == {'type': 'json_object'}, 'Text helper output schema mismatch')
        elif provider == 'kev':
            require(request['model'] == 'kev-latest', 'Wrong Kev alias')
        response = indexed.get(attempt['call_id'])
        if response:
            require(response['provider'] == provider, 'API response provider mismatch')
            if response.get('http_status') == 200:
                body = response['response']
                expected = 'kev-latest' if provider == 'kev' else config[provider + '_model']
                require(body['model'] == expected or (provider == 'judge' and body['model'].startswith(expected + '-')),
                        'Resolved model identity mismatch')
                successful[provider].append((request, body))
    # A successful HTTP call may fail schema validation before a decision is saved.
    # Consume matches in order; trailing calls remain charged and visible.
    cursor = 0
    for decision in decisions:
        require(cursor < len(successful['kev']), 'Decision lacks successful HTTP response')
        request, response = successful['kev'][cursor]
        validate_decision(decision, request, response)
        cursor += 1
    require(result['resolved_jev_models'] == sorted({d['model'] for d in decisions}), 'Resolved decision models mismatch')
    cursor = 0
    for i, entry in enumerate(history, 1):
        require(entry['step'] == i, 'History step sequence mismatch')
        match = next((j for j in range(cursor, len(decisions)) if all(entry[k] == decisions[j][k]
                     for k in ('choice', 'operation', 'target', 'confidence', 'latency_ms', 'usage'))), None)
        require(match is not None, 'Executed action has no matching preceding decision')
        decision = decisions[match]
        require(entry['probability'] == decision['probabilities'][entry['choice']], 'History probability mismatch')
        if decision['target'] is not None:
            criterion = decision['request']['questions'][decision['operation'].lower() + '_target']['criteria'][decision['target']]
            require(criterion['element'] == f"[{decision['target']}] {entry['action']}", 'Executed label differs from selected request target')
        else:
            require(entry['action'] == decision['request']['questions']['operation']['criteria'][decision['operation']],
                    'Executed control label differs from selected operation')
        cursor = match + 1
        if entry['kind'] == 'fill':
            require(entry['text_helper'] == config['text_model'] and any(t['value'] == entry['text'] and t['field'] == entry['action']
                    for t in trajectory['text_calls']), 'Executed fill lacks helper output')
    for call in trajectory['text_calls']:
        require(call['model'] == config['text_model'] and any(
            json.loads(body['choices'][0]['message']['content']).get('text') == call['value']
            and body.get('usage', {}) == call['usage'] for _, body in successful['text']), 'Helper output/API mismatch')
    states = []
    for path in sorted(directory.glob('state-*.json.gz')):
        with gzip.open(path, 'rt') as handle:
            state = json.load(handle)
        require(state['goal'] == task['intent'], 'Snapshot goal mismatch')
        require(state['history'] == history[:len(state['history'])] and state['decisions'] == decisions[:len(state['decisions'])]
                and state['text_calls'] == trajectory['text_calls'][:len(state['text_calls'])], 'Snapshot/trajectory prefix mismatch')
        if state.get('page', {}).get('screenshot'):
            image_bytes(state['page']['screenshot'])
        states.append((path, state))
    final_name = f"state-{config['max_decisions'] + 1:04d}.json.gz"
    final = next((s for p, s in states if p.name == final_name), None)
    if final:
        require(final['history'] == history and final['decisions'] == decisions and final['text_calls'] == trajectory['text_calls'],
                'Final archive differs from trajectory')
    fresh = bool(final and final.get('final_screenshot_fresh') is True and final.get('page', {}).get('screenshot'))
    if result['valid']:
        require(fresh, 'Missing fresh terminal screenshot')
        require(any(p.name == 'state-0000.json.gz' for p, _ in states), 'Missing initial observation archive')
    expected, response, score = None, None, None
    if fresh:
        screenshot = final['page']['screenshot']
        require(image_bytes(screenshot) == (directory/'final.jpg').read_bytes(), 'Terminal JPEG differs from archive')
        expected = dict(model=config['judge_model'], seed=config['judge_seed'], max_completion_tokens=config['judge_max_completion_tokens'],
                        messages=canonical_messages(task, history, screenshot, result['terminal'], judge_source))
    if (directory/'judge-request.json').exists():
        require(expected is not None, 'Judge was given missing or stale final evidence')
        require(read(directory/'judge-request.json') == expected, 'Canonical judge request mismatch')
        require(all(a['request'] == expected for a in attempts if a['provider'] == 'judge'), 'Judge retry changed canonical request')
    if _pending_judge:
        require(expected is not None and (directory/'judge-request.json').exists(), 'Missing saved canonical judge input')
        missing = [a for a in attempts if a['call_id'] not in indexed]
        require(len(missing) == 1 and missing[0]['provider'] == 'judge' and counts['judge'] == 1,
                'Recovery requires exactly one unanswered original judge and no other unresolved calls')
        return dict(task_id=task['task_id'], status='saved_judge_pending_verified', api_attempts=result['api_attempts'],
                    missing_call_id=missing[0]['call_id'])
    if (directory/'judge-response.json').exists():
        response = read(directory/'judge-response.json')
        require(any(request == expected and body == response for request, body in successful['judge']), 'Judge evidence/API mismatch')
        text = response['choices'][0]['message']['content'] or ''
        if 'status:' in text.lower():
            score = 1.0 if 'success' in text.lower().split('status:', 1)[1] else 0.0
        require(result['judge_text'] == text and result['score'] == score, 'Canonical verdict/result mismatch')
    if result['valid']:
        require(response is not None and score is not None, 'Unparseable or missing judge verdict marked valid')
        require(result['judge_text'] == text and result['score'] == score and result['judge_model'] == config['judge_model']
                and result['judge_prompt_variant'] == 'agenttrek' and result['judge_error'] is None, 'Canonical verdict/result mismatch')
        if (result.get('saved_judge_recovery') or (directory/'saved-judge-recovery').exists()
                or len(responses) != len(attempts)):
            validate_saved_judge_recovery(root, directory, result, attempts, responses, expected)
    else:
        error = result['judge_error']
        if error == 'unparseable_verdict':
            require(response is not None and score is None, 'Invalid verdict diagnosis contradicts saved judge')
        elif error == 'missing_browser_evidence':
            require(not fresh and not successful['judge'] and not counts['judge'], 'Missing evidence diagnosis contradicts artifacts')
        elif error == 'provider_error':
            require(result['provider_blocked'] and not counts['judge'] and bool(result['actor_error'] or result['cleanup_errors']),
                    'Provider diagnosis lacks recorded actor/cleanup failure')
        elif result['provider_blocked'] and isinstance(error, str) and error.startswith('judge HTTP '):
            require(expected is not None and counts['judge'] and response is None, 'Judge API diagnosis lacks attempted request')
            require(any(a['provider'] == 'judge' and (a['call_id'] not in indexed or indexed[a['call_id']].get('http_status') != 200)
                        for a in attempts), 'Judge API diagnosis lacks failed attempt')
        else:
            raise ValueError('Unrecognized invalid failure; add a targeted verified diagnosis check before acceptance')
        invalid_diagnosis(root, directory, result, diagnosis)
    return dict(task_id=task['task_id'], status='evidence_verified' if result['valid'] else 'diagnosed_invalid',
                valid=result['valid'], score=result['score'], terminal=result['terminal'], actions=len(history),
                decisions=len(decisions), screenshots=len(states), api_attempts=dict(counts),
                diagnosis=diagnosis if not result['valid'] else None,
                artifact_sha256={str(p.relative_to(root)): sha(p) for p in directory.rglob('*') if p.is_file() and p.suffix != '.tmp'})


def validate_saved_judge_recovery(root, directory, result, attempts, responses, expected):
    """Exempt only the proven original interrupted judge call, never actor calls."""
    try:
        from scripts.recover_kev_actor_saved_judge import validate_intent
    except ModuleNotFoundError as exc:
        if exc.name != 'scripts':
            raise
        from recover_kev_actor_saved_judge import validate_intent
    folder = directory/'saved-judge-recovery'
    require((folder/'intent.json').exists() and (folder/'completion.json').exists(),
            'Valid task has unresolved API attempts without completed saved-judge recovery')
    intent = read(folder/'intent.json')
    original = validate_intent(root, directory, intent)
    completion = read(folder/'completion.json')
    require(completion['intent_sha256'] == sha(folder/'intent.json')
            and completion['result_digest'] == digest(result)
            and completion['judge_response_sha256'] == sha(directory/'judge-response.json'),
            'Saved judge recovery completion hash mismatch')
    require(result.get('saved_judge_recovery') == dict(intent_sha256=sha(folder/'intent.json'),
            original_unanswered_call_id=intent['missing_call_id'], original_judge_latency_unknown=True),
            'Missing saved judge recovery result provenance')
    indexed = {r['call_id']: r for r in responses}
    require({a['call_id'] for a in attempts} - set(indexed) == {intent['missing_call_id']},
            'Recovery cannot waive additional unanswered calls')
    added = attempts[len(original['attempts']):]
    require(0 < len(added) <= intent['max_additional_calls'] and all(
        a['provider'] == 'judge' and a['request'] == expected and a['started_unix'] >= intent['prepared_unix']
        for a in added), 'Recovery changed actor calls, judge input, or call allowance')
    judges = [a for a in attempts if a['provider'] == 'judge']
    require(len(judges) <= 4 and [a['call_id'] for a in judges] == [f'judge-{i:04d}' for i in range(1, len(judges)+1)],
            'Recovery judge counter/cap mismatch')
    require(result.get('actor_seconds') == read(directory/'trajectory.json')['actor_seconds'],
            'Recovery changed original actor duration')


def audit(root):
    root = Path(root).resolve()
    plan, approval, cohort = read(root/'plan.json'), read(root/'approval.json'), read(root/'actor/plan.json')
    require(approval['approved'] and approval['plan_sha256'] == sha(root/'plan.json'), 'Approval/plan identity mismatch')
    require(approval['resources'] == plan['resources'] and approval['limits'] == plan['limits'], 'Approval caps mismatch')
    require(digest(cohort) == plan['cohort_plan_sha256'] and cohort['tasks'] == plan['tasks'], 'Cohort plan identity mismatch')
    require(len(plan['tasks']) == len({t['task_id'] for t in plan['tasks']}) == 300, 'Expected exactly 300 unique tasks')
    require(cohort['config'] == plan['protocol'] and plan['wandb_project'] == 'openwebrl-evals', 'Protocol or W&B project mismatch')
    for name, expected in plan['source_sha256'].items():
        require(sha(Path(plan['source'])/name) == expected, 'Frozen source changed: ' + name)
    # Saved identity checks cost no GPU/API calls and cover each owned attempt.
    validation_source = Path(plan['source'])/'openwebrl/kev_eval.py'
    kev_validation = {'__name__': 'pinned_kev_audit_validation', '__file__': str(validation_source)}
    # Compile the already hash-verified pure validation module without creating
    # bytecode files inside the frozen worker source.
    exec(compile(validation_source.read_bytes(), str(validation_source), 'exec'), kev_validation)
    for directory in (root/'attempts').iterdir():
        if not directory.is_dir():
            continue
        for name in ('server-identity.json', 'server-warm.json', 'server-final.json'):
            if (directory/name).exists():
                kev_validation['validate_card']({'models': [read(directory/name)]}, plan['kev'])
        warmup_requests = lines(directory/'warmup-attempts.jsonl')
        warmup_responses = lines(directory/'warmup-responses.jsonl')
        require(len(warmup_responses) <= len(warmup_requests), 'Warmup response lacks attempt')
        require([r['request'] for r in warmup_requests] == read(root/'warmup-requests.json')[:len(warmup_requests)],
                'Warmup request schedule changed')
        for request, response in zip(warmup_requests, warmup_responses):
            if response['status'] == 200:
                kev_validation['validate_answers'](request['request'], response['response'])
    require(sha(root/'launch.sbatch') == plan['launch_sha256'], 'Launch resource manifest changed')
    require(digest(read(root/'warmup-requests.json')) == plan['warmup_sha256'], 'Warmup identity mismatch')
    diagnoses_path = root/'diagnosed-invalids.json'
    diagnoses = read(diagnoses_path) if diagnoses_path.exists() else {}
    require(isinstance(diagnoses, dict), 'Diagnoses must map task IDs to hash-bound records')
    rows, missing, issues = [], [], []
    # Freeze the completed task set before reading large archives during collection.
    tasks = [t for t in plan['tasks'] if (root/'actor/tasks'/digest(t['task_id'])/'result.json').exists()]
    for task in plan['tasks']:
        if task not in tasks:
            missing.append(task['task_id'])
    for task in tasks:
        try:
            rows.append(audit_task(root, task, plan['protocol'], Path(plan['source'])/'openwebrl/eval/reward_online_mind2web.py',
                                   diagnoses.get(task['task_id'])))
        except (ValueError, KeyError, OSError, TypeError, IndexError) as exc:
            issues.append(dict(task_id=task['task_id'], error_type=type(exc).__name__, error=str(exc)))
    # Include preserved task attempts, not just the canonical task directories.
    actual = Counter()
    for path in root.rglob('api-attempts.jsonl'):
        for record in lines(path):
            actual[KINDS[record['provider']]] += 1
    warmups = sum(len(lines(p)) for p in (root/'attempts').glob('*/warmup-attempts.jsonl'))
    actual['local_kev_requests'] += warmups
    markers = [read(p) for p in root.rglob('browser-session.json')]
    # Read reservations after attempt records so active workers cannot make a new
    # attempt appear uncharged. Retry only the tiny state-before-journal interval.
    for _ in range(3):
        reservations = lines(root/'budget/reservations.jsonl')
        usage = read(root/'budget/usage.json') if (root/'budget/usage.json').exists() else {'reserved': {}, 'sequence': 0, 'limits': plan['budget']['limits']}
        if usage['sequence'] == len(reservations):
            break
    charged = Counter()
    for i, reservation in enumerate(reservations, 1):
        require(reservation['sequence'] == i and type(reservation['count']) is int and reservation['count'] > 0, 'Reservation sequence/count mismatch')
        charged[reservation['kind']] += reservation['count']
    require(usage['limits'] == plan['budget']['limits'], 'Reservation cap identity mismatch')
    require(dict(charged) == usage['reserved'] and usage['sequence'] == len(reservations), 'Reservation ledger/state mismatch')
    require(all(n <= plan['limits'][kind] for kind, n in charged.items()), 'Approved API/browser cap exceeded')
    require(all(n <= charged[k] for k, n in actual.items()), 'API attempts exceed durable reservations')
    session_ids = [m['id'] for m in markers]
    require(len(session_ids) == len(set(session_ids)), 'Browser session reused across task attempts')
    require(len(markers) <= charged['browser_sessions'], 'Browser markers exceed reservations')
    scheduler_path = root/'scheduler-ledger.json'
    scheduler = read(scheduler_path) if scheduler_path.exists() else {}
    attempts = scheduler.get('attempts', [])
    scheduler_closed = bool(attempts) and {r['job_id'] for r in attempts} == {r['job_id'] for r in approval['attempts']} and all(r['state'] in TERMINAL for r in attempts)
    elapsed = sum(r['scheduler_seconds'] for r in attempts)
    require(elapsed <= plan['resources']['total_seconds'], 'Approved scheduler cap exceeded')
    require(not scheduler or scheduler['charged_seconds'] == elapsed, 'Scheduler ledger total mismatch')
    valid = sum(row['valid'] for row in rows)
    successes = sum(row['valid'] and row['score'] == 1 for row in rows)
    closed = all(m.get('stopped') is True for m in markers)
    return dict(audit_version=1, audited_unix=time.time(), plan_sha256=sha(root/'plan.json'),
                audit_source_sha256=sha(__file__), planned=300, completed=len(tasks), audited=len(rows), valid=valid,
                invalid=sum(not r['valid'] for r in rows), successes=successes,
                overall_success_rate=successes/300, valid_only_success_rate=successes/valid if valid else None,
                all_results_audited=len(rows) == 300 and not issues, missing_task_ids=missing, issues=issues,
                all_browsers_closed=closed, browser_sessions=len(markers), reservations=dict(charged),
                recorded_http_attempts=dict(actual), warmup_attempts=warmups, scheduler_closed=scheduler_closed,
                scheduler=scheduler, remaining_approved_seconds=plan['resources']['total_seconds']-elapsed,
                evidence_ready_for_owner_closeout=len(rows) == 300 and not issues and closed and scheduler_closed,
                verified_complete=False, completion_note='Owner must independently verify remote W&B, docs/private review and scheduler freshness.',
                per_task=rows)


def write_output(path, value):
    path = Path(path).resolve()
    require(not path.is_relative_to(REPO), 'Private audit payloads must stay outside the repository')
    require(not path.exists(), 'Preserve prior audits: choose a new output filename')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.root)
    write_output(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k not in {'per_task', 'missing_task_ids'}}, indent=2))
