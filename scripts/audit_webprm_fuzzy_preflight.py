"""Read-only saved native fuzzy startup proof; no provider or browser access."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from openwebrl.webprm_broker import payload_sha256
from openwebrl.webprm_reddit_pilot_preflight import (NATIVE_FUZZY_FIXTURE, NATIVE_FUZZY_PREFLIGHT,
    GUARDED_FUZZY_PREFLIGHT, GUARDED_FUZZY_FIXTURES)
from openwebrl.webprm_webarena import digest
from openwebrl.webprm_webarena_fuzzy import NativeFuzzyJudge, validate_fuzzy_configuration


def _read(path):
    return json.loads(Path(path).read_text())


def _events(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()] if Path(path).exists() else []


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_native_fuzzy_preflight(attempt, plan, proposal, broker_events, *, primary_launched):
    """Audit a passed gate strictly; keep incomplete startup attempts auditable."""
    if proposal.get('native_fuzzy_preflight') == GUARDED_FUZZY_PREFLIGHT:
        return audit_guarded_fuzzy_preflight(attempt, plan, proposal, broker_events,
            primary_launched=primary_launched)
    attempt = Path(attempt)
    if proposal.get('native_fuzzy_preflight') != NATIVE_FUZZY_PREFLIGHT:
        raise ValueError('Corrected pilot lacks its frozen native fuzzy preflight contract')
    folder = attempt / 'native-fuzzy-preflight'
    journal = _events(attempt / 'native-fuzzy-preflight-api.jsonl')
    requests = [row for row in journal if row.get('event') == 'request']
    responses = [row for row in journal if row.get('event') == 'response']
    starts = [row for row in broker_events if row.get('event') == 'dispatch_started'
              and row.get('cell_id') == NATIVE_FUZZY_PREFLIGHT['cell_id']]
    if len(requests) > 1 or len(responses) > 1 or len(starts) > 1:
        raise ValueError('Native fuzzy startup made more than one attempt')
    for request in requests:
        if (request.get('local') is not False or request.get('purpose') != 'native_fuzzy' or
                request.get('payload', {}).get('model') != NATIVE_FUZZY_PREFLIGHT['model']):
            raise ValueError('Native fuzzy startup used a different route')
    passed = folder / 'passed.json'
    if not passed.exists():
        if primary_launched:
            raise ValueError('Launched corrected primary lacks passed native fuzzy preflight')
        return dict(status='failed_or_incomplete' if folder.exists() or starts else 'not_reached',
                    paid_dispatches=len(starts), primary_admitted=False)
    if (folder / 'failed.json').exists():
        raise ValueError('Native fuzzy startup has contradictory pass/failure evidence')
    proof = _read(passed); inputs = _read(folder / 'input.json')
    native_events = _events(folder / 'native-fuzzy.jsonl')
    native_responses = [row['response'] for row in native_events if row.get('event') == 'response']
    verdicts = [row for row in native_events if row.get('event') == 'native_verdict']
    expected_proof = dict(status='passed', synthetic=True, primary_admitted=False, score=1,
                          expected_score=1, provider_calls=1,
                          budget_owner='webprm_worker.completion', automatic_retry=False)
    if (any(proof.get(k) != v for k, v in expected_proof.items()) or
            proof.get('input_sha256') != _sha(folder / 'input.json') or
            proof.get('native_journal_sha256') != _sha(folder / 'native-fuzzy.jsonl') or
            inputs.get('protocol') != NATIVE_FUZZY_PREFLIGHT or
            inputs.get('fixture') != NATIVE_FUZZY_FIXTURE or inputs.get('synthetic') is not True or
            inputs.get('identity_sha256') != digest(plan) or inputs.get('primary_admitted') is not False or
            len(requests) != 1 or len(responses) != 1 or len(native_responses) != 1 or
            len(verdicts) != 1 or verdicts[0].get('score') != 1 or len(starts) != 1):
        raise ValueError('Native fuzzy startup proof is incomplete or changed')
    response = native_responses[0]; receipt = response.get('_webprm_broker', {})
    reservation = receipt.get('reservation_id'); payload = requests[0]['payload']
    projected = {key: response.get(key) for key in ('model', 'choices', 'usage')}
    expected = dict(identity_sha256=digest(plan), cell_id=NATIVE_FUZZY_PREFLIGHT['cell_id'],
        purpose='native_fuzzy', model=NATIVE_FUZZY_PREFLIGHT['model'], local=False,
        payload_sha256=payload_sha256(payload))
    matches = [row for row in broker_events if row.get('event') == 'dispatch_completed'
               and row.get('request_id') == receipt.get('request_id')]
    if (type(reservation) is not int or receipt.get('reservation_state') != 'settled' or
            receipt.get('schema_version') != 1 or
            receipt.get('journal_sha256') != _sha(attempt / 'native-fuzzy-preflight-api.jsonl') or
            receipt.get('budget_owner') != 'webprm_worker.completion' or
            requests[0].get('reservation') != reservation or responses[0].get('reservation') != reservation or
            {key: responses[0]['response'].get(key) for key in projected} != projected or
            any(receipt.get(k) != v or starts[0].get(k) != v for k, v in expected.items()) or
            starts[0].get('request_id') != receipt.get('request_id') or len(matches) != 1 or
            any(matches[0].get(k) != v for k, v in receipt.items()) or
            receipt.get('usage') != response.get('usage')):
        raise ValueError('Native fuzzy startup lacks matching settled broker evidence')
    choices = response.get('choices', []); usage = response.get('usage', {})
    if (response.get('model') != NATIVE_FUZZY_PREFLIGHT['model'] or len(choices) != 1 or
            choices[0].get('finish_reason') != 'stop' or
            not isinstance(choices[0].get('message', {}).get('content'), str) or
            any(type(usage.get(k)) is not int or usage[k] <= 0
                for k in ('prompt_tokens', 'completion_tokens'))):
        raise ValueError('Native fuzzy startup response is incomplete')
    selected = validate_fuzzy_configuration(plan)
    native = NativeFuzzyJudge(plan['arbiter_source'], enabled=False, model=selected['model'],
        protocol=selected['protocol'], model_change_approved=selected['model_change_approved'],
        max_input_tokens=plan['fuzzy_max_input_tokens'], max_calls=1, deadline=plan['deadline'])
    if proof.get('manifest') != dict(native.manifest, enabled=True, budget_owner='webprm_worker.completion'):
        raise ValueError('Native fuzzy startup model/source manifest changed')
    seen = []
    def saved_response(**request):
        if request.pop('context_length', None) != 0 or request.pop('model', None) != 'gpt-4-1106-preview':
            raise ValueError('Native fuzzy startup request differs from pinned source')
        request['model'] = selected['model']
        if request != payload:
            raise ValueError('Native fuzzy startup prompt differs from the synthetic known-answer fixture')
        seen.append(request)
        return choices[0]['message']['content'], usage
    native._native['generate_from_openai_chat_completion'] = saved_response
    try:
        score = native._native['llm_fuzzy_match'](NATIVE_FUZZY_FIXTURE['prediction'],
            NATIVE_FUZZY_FIXTURE['reference'], NATIVE_FUZZY_FIXTURE['question'])
    except (AssertionError, KeyError, TypeError) as error:
        raise ValueError('Native fuzzy startup parser replay failed') from error
    if score != 1 or len(seen) != 1:
        raise ValueError('Native fuzzy startup known-answer verdict failed')
    return dict(status='passed', paid_dispatches=1, primary_admitted=False,
                model=selected['model'], reservation_id=reservation, proof_sha256=_sha(passed),
                scope='Saved synthetic native-prompt/parser replay only; no benchmark-quality claim')


def audit_guarded_fuzzy_preflight(attempt, plan, proposal, broker_events, *, primary_launched):
    """Replay the fixed v4 controls and both native/effective saved scores."""
    from openwebrl.webprm_webarena import load_native_evaluator, ABSTENTION_EVALUATOR_PROTOCOL
    attempt = Path(attempt); folder = attempt / 'native-fuzzy-preflight'
    if (proposal.get('native_fuzzy_preflight') != GUARDED_FUZZY_PREFLIGHT
            or plan.get('evaluator_protocol') != ABSTENTION_EVALUATOR_PROTOCOL):
        raise ValueError('Guarded startup lacks explicit frozen scoring/control identities')
    starts = [row for row in broker_events if row.get('event') == 'dispatch_started'
        and row.get('cell_id') == GUARDED_FUZZY_PREFLIGHT['cell_id']]
    api_path = attempt / 'native-fuzzy-preflight-api.jsonl'
    api = _events(api_path)
    requests = [row for row in api if row.get('event') == 'request']
    responses = [row for row in api if row.get('event') == 'response']
    if max(len(starts), len(requests), len(responses)) > 8:
        raise ValueError('Guarded startup exceeded the eight fixed requests')
    passed = folder / 'passed.json'
    if not passed.exists():
        if primary_launched:
            raise ValueError('Primary launch lacks passed guarded startup controls')
        return dict(status='failed_or_incomplete' if folder.exists() or starts else 'not_reached',
            paid_dispatches=len(starts), primary_admitted=False)
    if (folder / 'failed.json').exists():
        raise ValueError('Guarded startup has contradictory pass/failure evidence')
    inputs = _read(folder / 'input.json'); proof = _read(passed)
    native_events = _events(folder / 'native-fuzzy.jsonl')
    native_responses = [row['response'] for row in native_events if row.get('event') == 'response']
    verdicts = [row for row in native_events if row.get('event') == 'native_verdict']
    if (inputs != dict(synthetic=True, protocol=GUARDED_FUZZY_PREFLIGHT,
            fixtures=list(GUARDED_FUZZY_FIXTURES), identity_sha256=digest(plan), primary_admitted=False)
            or proof.get('input_sha256') != _sha(folder / 'input.json')
            or proof.get('native_journal_sha256') != _sha(folder / 'native-fuzzy.jsonl')
            or any(len(items) != 8 for items in (starts, requests, responses, native_responses, verdicts))):
        raise ValueError('Guarded startup fixed inputs/evidence are incomplete or changed')
    selected = validate_fuzzy_configuration(plan)
    native = NativeFuzzyJudge(plan['arbiter_source'], enabled=False, model=selected['model'],
        protocol=selected['protocol'], model_change_approved=selected['model_change_approved'],
        max_input_tokens=plan['fuzzy_max_input_tokens'], max_calls=8, deadline=plan['deadline'])
    manifest = dict(native.manifest, enabled=True, budget_owner='webprm_worker.completion')
    proxy = SimpleNamespace(enabled=True, provider_calls=0, manifest=manifest)
    proxy.fuzzy_match = native._native['llm_fuzzy_match']; proxy.ua_match = native._native['llm_ua_match']
    evaluator = load_native_evaluator(plan['arbiter_source'], fuzzy_judge=proxy,
        scoring_protocol=ABSTENTION_EVALUATOR_PROTOCOL)
    prefixes = {}; prefix = b''
    for line in api_path.read_bytes().splitlines(keepends=True):
        prefix += line
        row = json.loads(line)
        if row.get('event') == 'response':
            prefixes[row['reservation']] = hashlib.sha256(prefix).hexdigest()
    replayed = []; reservations = set(); request_ids = set()
    for index, (fixture, response) in enumerate(zip(GUARDED_FUZZY_FIXTURES, native_responses)):
        receipt = response.get('_webprm_broker', {}); reservation = receipt.get('reservation_id')
        request = requests[index]; received = responses[index]; payload = request.get('payload', {})
        expected = dict(identity_sha256=digest(plan), cell_id=GUARDED_FUZZY_PREFLIGHT['cell_id'],
            purpose='native_fuzzy', model=GUARDED_FUZZY_PREFLIGHT['model'], local=False,
            payload_sha256=payload_sha256(payload))
        own = [row for row in broker_events if row.get('request_id') == receipt.get('request_id')]
        completed = [row for row in own if row.get('event') == 'dispatch_completed']
        start = [row for row in own if row.get('event') == 'dispatch_started']
        projected = {key: response.get(key) for key in ('model', 'choices', 'usage')}
        if (type(reservation) is not int or reservation in reservations or receipt.get('request_id') in request_ids
                or receipt.get('reservation_state') != 'settled' or receipt.get('schema_version') != 1
                or receipt.get('budget_owner') != 'webprm_worker.completion'
                or receipt.get('journal_sha256') != prefixes.get(reservation)
                or request.get('reservation') != reservation or received.get('reservation') != reservation
                or request.get('purpose') != 'native_fuzzy' or request.get('local') is not False
                or payload.get('model') != selected['model'] or response.get('model') != selected['model']
                or {key: received.get('response', {}).get(key) for key in projected} != projected
                or any(receipt.get(key) != value for key, value in expected.items())
                or len(start) != 1 or len(completed) != 1
                or any(start[0].get(key) != value for key, value in expected.items())
                or any(completed[0].get(key) != value for key, value in receipt.items())
                or receipt.get('usage') != response.get('usage')):
            raise ValueError('Guarded startup lacks matching settled broker evidence')
        reservations.add(reservation); request_ids.add(receipt['request_id'])
        choices = response.get('choices', []); usage = response.get('usage', {})
        if (len(choices) != 1 or choices[0].get('finish_reason') != 'stop'
                or not isinstance(choices[0].get('message', {}).get('content'), str)
                or any(type(usage.get(k)) is not int or usage[k] <= 0 for k in ('prompt_tokens', 'completion_tokens'))):
            raise ValueError('Guarded startup response incomplete')
        def saved_response(**actual):
            if actual.pop('context_length', None) != 0 or actual.pop('model', None) != 'gpt-4-1106-preview':
                raise ValueError('Native guarded startup request identity changed')
            actual['model'] = selected['model']
            if actual != payload:
                raise ValueError('Guarded startup prompt differs from the fixed evaluator-routed fixture')
            proxy.provider_calls += 1
            return choices[0]['message']['content'], usage
        native._native['generate_from_openai_chat_completion'] = saved_response
        task = dict(intent=fixture['question'], eval=dict(eval_types=['string_match'],
            reference_answers=dict(fuzzy_match=[fixture['reference']])))
        trajectory = [dict(observation=dict(text='Synthetic fixture'), info={}),
            dict(action_type='send_msg_to_user', answer=fixture['prediction'])]
        result = evaluator.evaluate(task, trajectory)
        if (_read(folder / f'control-{index}.json') != dict(fixture=fixture, result=result)
                or result['score'] != fixture['expected_score']
                or verdicts[index].get('score') != result['native_score'] or proxy.provider_calls != index + 1):
            raise ValueError('Guarded startup native/effective score replay differs')
        replayed.append(result)
    expected = dict(status='passed', synthetic=True, primary_admitted=False,
        protocol=GUARDED_FUZZY_PREFLIGHT, provider_calls=8,
        native_positive_count=sum(row['native_success'] for row in replayed),
        effective_positive_count=sum(row['success'] for row in replayed),
        corrected_count=sum(bool(row['scoring_amendments']) for row in replayed),
        manifest=manifest, budget_owner='webprm_worker.completion', automatic_retry=False)
    if any(proof.get(key) != value for key, value in expected.items()):
        raise ValueError('Guarded startup proof summary differs')
    return dict(status='passed', paid_dispatches=8, primary_admitted=False,
        model=selected['model'], proof_sha256=_sha(passed),
        native_positive_count=expected['native_positive_count'],
        effective_positive_count=expected['effective_positive_count'],
        scope='Eight saved fixed positive/negative/abstention controls; not global judge accuracy')
