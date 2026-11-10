#!/usr/bin/env python3
"""Approval-gated runner for prepared strict audits; --execute is never implicit.

No browser or scheduler work. One HTTP attempt per prepared item, no
automatic retries. Unknown outcomes retain their full reservation and halt the
campaign. One acceptance call precedes concurrency four; the coordinator owns
every reservation and drains all in-flight responses before exiting on failure.
Canonical cohorts and immutable prepared inputs are read-only.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

from prepare_selector_strict_audit import (
    FROZEN_INPUT_USD_PER_MILLION, FROZEN_OUTPUT_USD_PER_MILLION, MODEL_CONFIG,
    sha, validate_verdict,
)


def strict_loads(value):
    def pairs(items):
        out = {}
        for key, item in items:
            if key in out:
                raise ValueError('Duplicate JSON key: ' + key)
            out[key] = item
        return out
    def constant(value):
        raise ValueError('Nonfinite JSON constant is forbidden')
    return json.loads(value, object_pairs_hook=pairs, parse_constant=constant)


def read_json(path: Path):
    return strict_loads(path.read_bytes())


def persist_bytes(path: Path, value: bytes):
    with path.open('xb') as f:
        f.write(value); f.flush(); os.fsync(f.fileno())


def persist(path: Path, value: dict):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def preflight(prepared: Path, approval: dict) -> tuple[list[dict], dict]:
    manifest_path = prepared/'private-manifest.json'
    readiness_path = prepared/'readiness.json'
    if (approval.get('authorized') is not True or
            approval.get('scope') != 'strict-selector-secondary-audit-v1'):
        raise ValueError('Explicit secondary-audit authorization is required')
    for field, path in [('manifest_sha256', manifest_path), ('readiness_sha256', readiness_path)]:
        if approval.get(field) != sha(path.read_bytes()):
            raise ValueError('Approval does not bind the exact immutable inputs')
    readiness = read_json(readiness_path)
    proposal = readiness['bounded_proposal']
    if approval.get('max_calls', 0) > 575 or approval.get('max_calls', 0) < proposal['eligible_initial_calls']:
        raise ValueError('Approval must cover prepared eligible calls, with an absolute maximum of 575')
    if approval.get('max_usd', 0) < proposal['frozen_price_upper_bound_usd']:
        raise ValueError('Approved dollar cap is below the conservative whole-audit reservation')
    if approval.get('max_calls_per_item') != 1 or approval.get('max_output_tokens') != 4096:
        raise ValueError('Only one attempt per item and 4096 output tokens are supported')
    if approval.get('max_concurrency') != 4:
        raise ValueError('Approval must explicitly bind maximum concurrency four, after one acceptance call')
    if approval.get('browser_sessions') != 0 or approval.get('gpus') != 0:
        raise ValueError('This runner does not authorize browsers or GPUs')
    if approval.get('input_usd_per_million') != FROZEN_INPUT_USD_PER_MILLION or approval.get('output_usd_per_million') != FROZEN_OUTPUT_USD_PER_MILLION:
        raise ValueError('Rates must match the explicitly approved frozen accounting')
    if readiness['model_config'] != MODEL_CONFIG:
        raise ValueError('Prepared model configuration is not this reviewed protocol')
    manifest = read_json(manifest_path)
    rows = [r for r in manifest['records'] if r['status'] == 'ready']
    if len(rows) != proposal['eligible_initial_calls'] or len({r['blind_id'] for r in rows}) != len(rows):
        raise ValueError('Eligible input inventory mismatch')
    for row in rows:
        path = (prepared/row['input_path']).resolve()
        if prepared.resolve() not in path.parents or sha(path.read_bytes()) != row['input_sha256']:
            raise ValueError('Immutable input changed')
        payload = read_json(path)
        if {k: payload.get(k) for k in MODEL_CONFIG} != MODEL_CONFIG or set(payload) != set(MODEL_CONFIG)|{'messages'}:
            raise ValueError('Payload model/schema/sampling differs')
    return rows, readiness


def validate_receipts_target(prepared: Path, receipts: Path, approval: dict):
    receipts = receipts.resolve()
    approved_target = approval.get('receipts_root')
    if not isinstance(approved_target, str) or not Path(approved_target).is_absolute() or Path(approved_target).resolve() != receipts:
        raise ValueError('Approval must bind this exact resolved receipts_root; budget cannot move to a fresh directory')
    manifest = read_json(prepared/'private-manifest.json')
    protected = {prepared.resolve()}
    for row in manifest['records']:
        protected.add(Path(row['cohort_root']).resolve())
    if any(receipts == root or root in receipts.parents or receipts in root.parents for root in protected):
        raise ValueError('Receipts must be disjoint from canonical cohorts and immutable prepared inputs')


def validate_response(response: dict, row: dict) -> tuple[dict, float]:
    # Import only when validating a response, not when displaying dry-run plans.
    import jsonschema
    if response.get('model') != MODEL_CONFIG['model']:
        raise ValueError('Returned judge model identity mismatch')
    choices = response.get('choices', [])
    if len(choices) != 1 or choices[0].get('finish_reason') != 'stop':
        raise ValueError('Missing, ambiguous or incomplete judge output')
    message = choices[0].get('message', {})
    if message.get('refusal'):
        raise ValueError('Judge refusal')
    usage = response.get('usage', {})
    it, ot = usage.get('prompt_tokens'), usage.get('completion_tokens')
    if (not isinstance(it, int) or isinstance(it, bool) or not isinstance(ot, int) or isinstance(ot, bool)
            or it < 0 or ot < 0 or it > row['budget_bound']['input_token_bound'] or ot > 4096):
        raise ValueError('Missing usage or usage beyond reserved bounds')
    if usage.get('total_tokens') != it + ot:
        raise ValueError('Inconsistent returned token usage')
    verdict = strict_loads(message['content'])
    jsonschema.validate(verdict, MODEL_CONFIG['response_format']['json_schema']['schema'])
    validate_verdict(verdict)
    cost = (it * FROZEN_INPUT_USD_PER_MILLION + ot * FROZEN_OUTPUT_USD_PER_MILLION) / 1_000_000
    return verdict, cost


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, 'Redirect disabled', headers, fp)


def run(prepared: Path, approval_path: Path, receipts: Path, execute: bool = False):
    approval = read_json(approval_path)
    rows, readiness = preflight(prepared, approval)
    validate_receipts_target(prepared, receipts, approval)
    if not execute:
        return {'dry_run': True, 'paid_calls': 0, 'bounded_proposal': readiness['bounded_proposal']}
    import jsonschema  # Fail before dispatch if the offline validator is absent.
    jsonschema.Draft202012Validator.check_schema(MODEL_CONFIG['response_format']['json_schema']['schema'])
    # Binding the endpoint in the approval prevents sending private evidence to
    # a substituted service through environment variables.
    endpoint = approval.get('endpoint', '')
    url = urlsplit(endpoint)
    if url.scheme != 'https' or url.netloc != 'api.openai.com' or url.path != '/v1/chat/completions' or url.query or url.fragment:
        raise ValueError('Only explicitly approved official OpenAI chat endpoint is supported')
    key = os.environ.get('OPENAI_API_KEY')
    if not key:
        raise ValueError('OPENAI_API_KEY is required; credentials are never saved')
    receipts = receipts.resolve()
    receipts.mkdir(parents=True, exist_ok=True)
    with (receipts/'writer.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (receipts/'halt.json').exists():
            raise ValueError('Previous halt requires independent review; no automatic restart')
        approval_hash = sha(approval_path.read_bytes())
        if (receipts/'campaign.json').exists():
            if read_json(receipts/'campaign.json')['approval_sha256'] != approval_hash:
                raise ValueError('Cannot replace approval within an existing campaign')
        else:
            persist(receipts/'campaign.json', {'approval_sha256': approval_hash,
                    'manifest_sha256': approval['manifest_sha256'], 'created_unix': time.time()})
        spent = 0.0
        calls = 0
        known_ids = {r['blind_id'] for r in rows}
        if any(p.is_dir() and p.name not in known_ids for p in receipts.iterdir()):
            raise ValueError('Unexpected receipt directory; cannot establish all-attempt accounting')
        remaining = []
        for row in rows:
            item = receipts/row['blind_id']
            reserve = (row['budget_bound']['input_token_bound'] * FROZEN_INPUT_USD_PER_MILLION + 4096 * FROZEN_OUTPUT_USD_PER_MILLION) / 1_000_000
            if item.exists():
                if not (item/'validated.json').exists():
                    raise ValueError('Existing request has unknown/unvalidated outcome; never replay automatically')
                saved = read_json(item/'validated.json')
                if saved['input_sha256'] != row['input_sha256']:
                    raise ValueError('Prior receipt input mismatch')
                reservation = read_json(item/'reservation.json')
                if (reservation.get('input_sha256') != row['input_sha256'] or reservation.get('attempt') != 1
                        or reservation.get('reserved_usd') != reserve):
                    raise ValueError('Prior reservation mismatch')
                if sha((item/'request.json').read_bytes()) != row['input_sha256']:
                    raise ValueError('Preserved request bytes changed')
                raw_bytes = (item/'response.raw').read_bytes()
                if (sha(raw_bytes) != saved.get('response_raw_sha256') or
                        sha((item/'response.json').read_bytes()) != saved.get('response_sha256')):
                    raise ValueError('Preserved response bytes changed')
                previous_response = strict_loads(raw_bytes)
                if previous_response != read_json(item/'response.json'):
                    raise ValueError('Raw and parsed receipts disagree')
                previous_verdict, previous_cost = validate_response(previous_response, row)
                if previous_verdict != saved['verdict'] or previous_cost != saved['charged_usd']:
                    raise ValueError('Saved verdict or charge does not match raw response')
                spent += saved['charged_usd']; calls += 1
                continue
            remaining.append((row, reserve))
        failed = threading.Event()
        dispatch_lock = threading.Lock()

        def mark_failed():
            # Same lock protects the final failure check + submission below.
            # Once a worker records failure, no later submission can cross it.
            with dispatch_lock:
                failed.set()

        def perform(row, item, payload_bytes):
            try:
                req = urllib.request.Request(endpoint, data=payload_bytes, headers={
                    'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'}, method='POST')
                opener = urllib.request.build_opener(NoRedirect())
                with opener.open(req, timeout=180) as response:
                    response_bytes = response.read(8 * 1024 * 1024 + 1)
                    if len(response_bytes) > 8 * 1024 * 1024:
                        raise ValueError('Response body exceeded bounded receipt size')
                    persist_bytes(item/'response.raw', response_bytes)
                    raw_response = strict_loads(response_bytes)
                persist(item/'response.json', raw_response)
                verdict, cost = validate_response(raw_response, row)
                persist(item/'validated.json', {'input_sha256': row['input_sha256'], 'charged_usd': cost,
                        'verdict': verdict, 'response_sha256': sha((item/'response.json').read_bytes()),
                        'response_raw_sha256': sha(response_bytes)})
                return {'ok': True, 'cost': cost}
            except Exception as exc:
                mark_failed()
                if isinstance(exc, urllib.error.HTTPError):
                    persist(item/'http-error.json', {'status': exc.code, 'body': exc.read(8192).decode('utf-8', 'replace')})
                error = {'reason': type(exc).__name__ + ': ' + str(exc),
                         'blind_id': row['blind_id'], 'unknown_or_unvalidated_outcome': True}
                persist(item/'failure.json', error)
                return {'ok': False, 'error': error}

        # Only this coordinator changes calls/spent or creates reservations.
        accepted = calls > 0  # All existing calls were independently revalidated.
        failures, futures, cursor = [], {}, 0
        with ThreadPoolExecutor(max_workers=4) as pool:
            while cursor < len(remaining) or futures:
                limit = 4 if accepted else 1
                while (not failed.is_set() and cursor < len(remaining) and len(futures) < limit):
                    row, reserve = remaining[cursor]
                    held = sum(x[1] for x in futures.values())
                    if calls + 1 > approval['max_calls'] or spent + held + reserve > approval['max_usd']:
                        failures.append({'reason': 'Approved budget exhausted before next request'})
                        mark_failed(); break
                    item = receipts/row['blind_id']
                    try:
                        payload_bytes = (prepared/row['input_path']).read_bytes()
                        if sha(payload_bytes) != row['input_sha256']:
                            raise ValueError('Immutable input changed after preflight')
                        item.mkdir()
                        persist(item/'reservation.json', {'input_sha256': row['input_sha256'], 'reserved_usd': reserve,
                                'input_token_bound': row['budget_bound']['input_token_bound'],
                                'output_token_bound': 4096, 'attempt': 1, 'unix': time.time()})
                        persist_bytes(item/'request.json', payload_bytes)
                        # Dispatch only after durable reservation and raw request.
                        with dispatch_lock:
                            if failed.is_set():
                                persist(item/'not-dispatched.json', {
                                    'reason': 'Another in-flight request failed during request preparation',
                                    'input_sha256': row['input_sha256'],
                                    'reservation_released_usd': reserve,
                                    'charged_or_reserved_usd': 0,
                                    'http_attempts': 0, 'automatic_replay_allowed': False})
                                break
                            future = pool.submit(perform, row, item, payload_bytes)
                            futures[future] = (row, reserve)
                            calls += 1; cursor += 1
                    except Exception as exc:
                        failures.append({'reason': 'Pre-dispatch failure: '+str(exc), 'blind_id': row['blind_id']})
                        mark_failed(); break
                if not futures:
                    break
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                # Process every available outcome before filling more slots.
                done |= {f for f in futures if f.done()}
                for future in done:
                    row, reserve = futures.pop(future)
                    try:
                        outcome = future.result()
                    except Exception as exc:
                        outcome = {'ok': False, 'error': {'reason': str(exc), 'blind_id': row['blind_id']}}
                    if outcome['ok']:
                        spent += outcome['cost']; accepted = True
                    else:
                        spent += reserve
                        failures.append(outcome['error']); mark_failed()
                # A failure prevents future dispatches; existing futures are
                # still awaited and their receipts/charges are retained.
        if failed.is_set():
            persist(receipts/'halt.json', {'reason': 'Strict audit halted; all in-flight requests drained',
                    'failures': failures, 'calls': calls, 'charged_or_reserved_usd': spent,
                    'unknown_or_unvalidated_outcome': True, 'in_flight_remaining': 0})
            raise RuntimeError('Strict audit halted; inspect preserved halt and per-item receipts')
        summary = {'completed': True, 'calls': calls, 'charged_or_reserved_usd': spent,
                   'canonical_scores_unchanged': True, 'strict_verdicts_are_secondary': True,
                   'max_concurrency': 4, 'first_call_acceptance_required': True}
        if not (receipts/'completion.json').exists(): persist(receipts/'completion.json', summary)
        return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--approval', type=Path, required=True)
    p.add_argument('--receipts', type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    print(json.dumps(run(a.prepared, a.approval, a.receipts, a.execute), indent=2))


if __name__ == '__main__': main()
