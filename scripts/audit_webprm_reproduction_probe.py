#!/usr/bin/env python3
"""Independently reconcile probe payloads, responses, labels and spending offline."""
from __future__ import annotations

import argparse
from decimal import Decimal
import math
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.probe_webprm_reproduction import (DEFAULT_ROOT, BATCHES, MINI,
    WRITE_RULE, digest, read, reference, verified_reference, write_once, parse_result)


def paired_treatment(base, factor):
    """Independent comparison specification; does not call the experiment mutator."""
    import copy
    value = copy.deepcopy(base)
    if factor in ('terminal_model', 'state_diff_model'):
        value['model'] = MINI
    elif factor == 'terminal_write_rule':
        value['messages'][0]['content'] += WRITE_RULE
    elif factor == 'terminal_temperature':
        value['temperature'] = 0.0
    else:
        raise ValueError('Unknown approved factor')
    return value


def audit(root):
    root = Path(root).resolve()
    plan = read(root / 'plan.json')
    cases = read(verified_reference(plan['cases']))['cases']
    case_map = {c['case_id']: c for c in cases}
    authorization = read(root / 'authorization.json')
    if authorization.get('status') != 'approved' or authorization['plan_sha256'] != digest(plan):
        raise ValueError('Exact plan authorization missing')
    for ref in plan['source'] + plan['implementation'] + [plan['script']]:
        verified_reference(ref)
    parent = read(verified_reference(plan['accounting']['parent_budget']))
    if parent.get('active_epoch_id') is not None:
        raise ValueError('Original benchmark campaign must remain closed')
    budget = read(root / 'budget.json')
    if budget['cap'] != plan['resources']['api_usd_cap'] or budget.get('halt_reason'):
        raise ValueError('Diagnostic cap changed or budget halted')
    expected = {c['call_id']: c for c in plan['calls']}
    if len(expected) != plan['resources']['max_provider_calls']:
        raise ValueError('Duplicate or changed dispatch cap')
    seen_reservations, records, artifact_refs = set(), [], []
    for folder in sorted((root / 'calls').iterdir()):
        if folder.name not in expected:
            raise ValueError('Unapproved diagnostic attempt directory')
        call = expected[folder.name]
        if read(folder / 'claim.json')['call'] != call:
            raise ValueError('Claim differs from frozen call')
        result = read(folder / 'result.json')
        if result['call'] != call or not result.get('response_recorded'):
            raise ValueError('Unknown or incomplete diagnostic dispatch')
        case = case_map[call['case_id']]
        if call['expected_complete'] != case['expected_complete']:
            raise ValueError('Development label differs from frozen case')
        base = case['saved_pairs'][call['purpose']]['request']['payload']
        correct_payload = base if call['variant'] == 'control' else paired_treatment(base, call['batch'])
        if call['payload'] != correct_payload or digest(correct_payload) != call['payload_sha256']:
            raise ValueError('A condition changes more than its declared variable')
        journal = [__import__('json').loads(line) for line in (folder / 'api.jsonl').read_text().splitlines()]
        requests = [x for x in journal if x['event'] == 'request']
        responses = [x for x in journal if x['event'] == 'response']
        if len(requests) != 1 or len(responses) != 1 or len(journal) != 2:
            raise ValueError('Exactly one dispatch and one response per probe required')
        req, reply = requests[0], responses[0]
        if req['payload'] != correct_payload or req['purpose'] != call['purpose'] or req.get('local'):
            raise ValueError('Actual paid payload or purpose differs')
        ident = req['reservation']
        if type(ident) is not int or ident in seen_reservations or reply['reservation'] != ident:
            raise ValueError('Repeated or mismatched financial reservation')
        seen_reservations.add(ident)
        response_path = verified_reference(result['response'])
        response = read(response_path)
        if reply['response'] != response or response.get('model') != correct_payload['model']:
            raise ValueError('Saved response identity differs')
        financial = budget['calls'][ident]
        if financial['id'] != ident or financial['state'] != 'settled' or financial['model'] != correct_payload['model']:
            raise ValueError('Probe lacks exactly one settled paid reservation')
        usage = response['usage']
        if financial['usage'] != usage:
            raise ValueError('Metered token usage differs')
        rates = [Decimal(str(x)) for x in plan['prices_usd_per_token'][correct_payload['model']]]
        charge = usage['prompt_tokens'] * rates[0] + usage['completion_tokens'] * rates[1]
        if not math.isclose(financial['charged'], float(charge), rel_tol=0, abs_tol=1e-10):
            raise ValueError('Token charges differ from frozen prices')
        parsed = parse_result(correct_payload, response, purpose=call['purpose'])
        if parsed != result.get('parsed'):
            raise ValueError('Fresh native parsing differs from saved probe result')
        correct = None
        if call['purpose'] == 'terminal':
            correct = bool(parsed['parse_valid'] and parsed['completion_claim'] == case['expected_complete'])
            if correct != result.get('correct_on_development_state'):
                raise ValueError('Development correctness differs')
        records.append(dict(call_id=folder.name, batch=call['batch'], case_id=call['case_id'],
                            variant=call['variant'], draw=call['draw'], expected_complete=case['expected_complete'],
                            parse_valid=parsed['parse_valid'], completion_claim=parsed['completion_claim'],
                            correct=correct, charged_usd=float(charge)))
        artifact_refs.extend(reference(folder / filename) for filename in
                             ('claim.json', 'result.json', 'api.jsonl', 'response.json'))
    if seen_reservations != set(range(len(budget['calls']))):
        raise ValueError('Extra or missing paid attempts in the budget')
    total = sum(Decimal(str(x['charged'])) for x in budget['calls'])
    if not math.isclose(float(total), budget['charged'], rel_tol=0, abs_tol=1e-9):
        raise ValueError('Budget aggregate differs from settled calls')
    if total > Decimal(str(budget['cap'])) or total + Decimal(str(parent['charged'])) > Decimal(str(parent['cap'])):
        raise ValueError('Diagnostic or combined original API cap exceeded')
    groups = []
    for batch in BATCHES:
        for variant in ('control', 'treatment'):
            rows = [x for x in records if x['batch'] == batch and x['variant'] == variant]
            if rows:
                groups.append(dict(batch=batch, variant=variant, responses=len(rows),
                                   parsed=sum(x['parse_valid'] for x in rows),
                                   correct=None if rows[0]['correct'] is None else sum(x['correct'] for x in rows),
                                   false_completion_claims=sum(x['completion_claim'] is True and not x['expected_complete'] for x in rows),
                                   invalid_outputs=sum(not x['parse_valid'] for x in rows)))
    complete = len(records) == len(expected)
    return dict(schema_version=1, audit_verified=True, diagnostic_sequence_complete=complete,
                auditor=reference(__file__),
                plan=reference(root / 'plan.json'), authorization=reference(root / 'authorization.json'),
                budget=reference(root / 'budget.json'), artifact_refs=artifact_refs,
                paid_attempts=len(records), settled_attempts=len(budget['calls']),
                charged_usd=float(total), combined_charged_usd=float(total + Decimal(str(parent['charged']))),
                combined_cap_usd=parent['cap'], groups=groups, records=records,
                benchmark_episodes=0, benchmark_successes_added=0,
                caveat='Selected development states with repeated draws; these are probe counts, not a task success rate or held-out accuracy.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root)
    write_once(args.output, result)
    print(__import__('json').dumps({k: v for k, v in result.items() if k not in ('records', 'artifact_refs')}))


if __name__ == '__main__':
    main()
