import importlib.util
import json
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
import run_selector_strict_audit as runner
from prepare_selector_strict_audit import MODEL_CONFIG, sha


@pytest.fixture
def prepared(tmp_path):
    root = tmp_path/'prepared'
    (root/'inputs').mkdir(parents=True)
    payload = dict(MODEL_CONFIG, messages=[{'role': 'system', 'content': 'offline fixture'}])
    p = root/'inputs/episode-0001.json'
    p.write_text(json.dumps(payload))
    row = {'status': 'ready', 'blind_id': 'episode-0001', 'input_path': 'inputs/episode-0001.json',
           'cohort_root': str(tmp_path/'canonical'),
           'input_sha256': sha(p.read_bytes()), 'budget_bound': {'input_token_bound': 10000}}
    manifest = root/'private-manifest.json'
    manifest.write_text(json.dumps({'records': [row]}))
    readiness = root/'readiness.json'
    readiness.write_text(json.dumps({'model_config': MODEL_CONFIG,
                         'bounded_proposal': {'eligible_initial_calls': 1, 'frozen_price_upper_bound_usd': .03}}))
    approval = {'authorized': True, 'scope': 'strict-selector-secondary-audit-v1',
                'manifest_sha256': sha(manifest.read_bytes()), 'readiness_sha256': sha(readiness.read_bytes()),
                'max_calls': 1, 'max_usd': .03, 'max_calls_per_item': 1, 'max_output_tokens': 4096,
                'max_concurrency': 4,
                'browser_sessions': 0, 'gpus': 0, 'input_usd_per_million': 1.1,
                'output_usd_per_million': 4.4, 'endpoint': 'https://api.openai.com/v1/chat/completions',
                'receipts_root': str(tmp_path/'receipts')}
    a = tmp_path/'approval.json'; a.write_text(json.dumps(approval))
    return root, approval, a, row


def test_dry_run_never_dispatches_or_requires_secret(prepared, tmp_path, monkeypatch):
    root, _, path, _ = prepared
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    monkeypatch.setattr(runner.urllib.request, 'build_opener', lambda *a: pytest.fail('network dispatch'))
    result = runner.run(root, path, tmp_path/'receipts')
    assert result['paid_calls'] == 0 and result['dry_run']
    assert not (tmp_path/'receipts').exists()


@pytest.mark.parametrize('field,value', [('authorized', False), ('manifest_sha256', 'wrong'),
    ('readiness_sha256', 'wrong'), ('max_calls', 576), ('max_calls', 0), ('max_usd', .001),
    ('max_calls_per_item', 2), ('max_output_tokens', 8192), ('gpus', 1), ('browser_sessions', 1),
    ('input_usd_per_million', 0), ('max_concurrency', 8)])
def test_authorization_limits_fail_closed(prepared, field, value):
    root, approval, _, _ = prepared
    approval[field] = value
    with pytest.raises(ValueError): runner.preflight(root, approval)


def valid_response():
    return {'model': 'o4-mini-2025-04-16', 'choices': [{'finish_reason': 'stop', 'message': {
        'content': json.dumps({'status': 'evidence_insufficient', 'strict_success': None,
                              'requirements': [{'requirement': 'Opening hours', 'finding': 'not_verifiable',
                                                'evidence': 'Saved screenshot does not show hours.'}],
                              'explanation': 'Cannot verify terminal claim.'})}}],
        'usage': {'prompt_tokens': 1000, 'completion_tokens': 100, 'total_tokens': 1100}}


def test_response_validation_keeps_evidence_insufficient_distinct(prepared):
    _, _, _, row = prepared
    verdict, cost = runner.validate_response(valid_response(), row)
    assert verdict['strict_success'] is None
    assert cost == pytest.approx(.00154)


@pytest.mark.parametrize('mutation', ['model', 'usage', 'over_bound', 'truncated', 'refusal', 'schema'])
def test_bad_response_halts(prepared, mutation):
    _, _, _, row = prepared
    r = valid_response()
    if mutation == 'model': r['model'] = 'other-model'
    if mutation == 'usage': r['usage'] = {}
    if mutation == 'over_bound': r['usage'].update(prompt_tokens=11000, total_tokens=11100)
    if mutation == 'truncated': r['choices'][0]['finish_reason'] = 'length'
    if mutation == 'refusal': r['choices'][0]['message']['refusal'] = 'refused'
    if mutation == 'schema': r['choices'][0]['message']['content'] = '{}'
    with pytest.raises(Exception): runner.validate_response(r, row)


def test_unknown_http_outcome_reserves_and_halts_without_retry(prepared, tmp_path, monkeypatch):
    root, _, path, _ = prepared
    monkeypatch.setenv('OPENAI_API_KEY', 'synthetic-key-not-sent')
    calls = []
    class OfflineTransport:
        def open(self, request, timeout):
            calls.append(request)
            raise TimeoutError('offline synthetic timeout')
    monkeypatch.setattr(runner.urllib.request, 'build_opener', lambda *a: OfflineTransport())
    receipts = tmp_path/'receipts'
    with pytest.raises(RuntimeError): runner.run(root, path, receipts, execute=True)
    assert len(calls) == 1
    halt = json.loads((receipts/'halt.json').read_text())
    assert halt['unknown_or_unvalidated_outcome'] is True
    assert halt['charged_or_reserved_usd'] > 0
    assert (receipts/'episode-0001/reservation.json').exists()
    assert 'synthetic-key-not-sent' not in '\n'.join(p.read_text() for p in receipts.rglob('*.json'))
    with pytest.raises(ValueError): runner.run(root, path, receipts, execute=True)
    assert len(calls) == 1


def test_success_receipt_resumes_without_second_call(prepared, tmp_path, monkeypatch):
    root, _, path, _ = prepared
    monkeypatch.setenv('OPENAI_API_KEY', 'synthetic-key-not-sent')
    calls = []
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, n): return json.dumps(valid_response()).encode()
    class OfflineTransport:
        def open(self, request, timeout): calls.append(request); return Response()
    monkeypatch.setattr(runner.urllib.request, 'build_opener', lambda *a: OfflineTransport())
    receipts = tmp_path/'receipts'
    first = runner.run(root, path, receipts, execute=True)
    second = runner.run(root, path, receipts, execute=True)
    assert first == second
    assert len(calls) == 1
    raw = receipts/'episode-0001/response.raw'
    raw.write_bytes(raw.read_bytes() + b' ')
    with pytest.raises(ValueError, match='response bytes changed'):
        runner.run(root, path, receipts, execute=True)
    assert len(calls) == 1


@pytest.mark.parametrize('value', ['{"x":1,"x":2}', '{"x":NaN}', '{"a":{"x":1,"x":1}}'])
def test_duplicate_or_nonfinite_json_is_rejected(value):
    with pytest.raises(ValueError): runner.strict_loads(value)


def test_same_approval_cannot_reset_budget_in_new_receipts_directory(prepared, tmp_path):
    root, _, path, _ = prepared
    with pytest.raises(ValueError, match='exact resolved receipts_root'):
        runner.run(root, path, tmp_path/'fresh-receipts', execute=False)


@pytest.mark.parametrize('target', ['canonical', 'canonical/child', '', 'prepared', 'prepared/child'])
def test_receipts_cannot_touch_canonical_or_prepared_trees(prepared, tmp_path, target):
    root, approval, _, _ = prepared
    receipt_target = tmp_path/target
    approval['receipts_root'] = str(receipt_target)
    with pytest.raises(ValueError, match='disjoint'):
        runner.validate_receipts_target(root, receipt_target, approval)


def multi_prepared(prepared, count=8):
    root, approval, path, row = prepared
    rows = []
    for i in range(count):
        rows.append(dict(row, blind_id=f'episode-{i+1:04d}'))
    manifest = root/'private-manifest.json'
    manifest.write_text(json.dumps({'records': rows}))
    readiness = root/'readiness.json'
    contents = json.loads(readiness.read_text())
    contents['bounded_proposal'].update(eligible_initial_calls=count, frozen_price_upper_bound_usd=.03*count)
    readiness.write_text(json.dumps(contents))
    approval.update(manifest_sha256=sha(manifest.read_bytes()), readiness_sha256=sha(readiness.read_bytes()),
                    max_calls=count, max_usd=.03*count)
    path.write_text(json.dumps(approval))
    return root, path


@pytest.mark.parametrize('fail_parallel', [False, True])
def test_acceptance_then_four_workers_and_drain_on_failure(prepared, tmp_path, monkeypatch, fail_parallel):
    root, path = multi_prepared(prepared)
    receipts = tmp_path/'receipts'
    monkeypatch.setenv('OPENAI_API_KEY', 'synthetic-key-not-sent')
    mutex = threading.Lock()
    barrier = threading.Barrier(4)
    state = {'calls': 0, 'active': 0, 'peak': 0, 'closed': 0}
    class Response:
        def __init__(self, index): self.index = index
        def __enter__(self): return self
        def __exit__(self, *args):
            with mutex: state['active'] -= 1; state['closed'] += 1
        def read(self, n):
            if 2 <= self.index <= 5:
                barrier.wait(timeout=5)
                if fail_parallel and self.index == 2:
                    raise TimeoutError('synthetic parallel failure')
            time.sleep(.03 if self.index != 1 else .005)
            return json.dumps(valid_response()).encode()
    class OfflineTransport:
        def open(self, request, timeout):
            with mutex:
                state['calls'] += 1
                index = state['calls']
                if index > 1:
                    assert (receipts/'episode-0001/validated.json').exists()
                state['active'] += 1
                state['peak'] = max(state['peak'], state['active'])
            assert (receipts/f'episode-{index:04d}'/'reservation.json').exists()
            return Response(index)
    monkeypatch.setattr(runner.urllib.request, 'build_opener', lambda *a: OfflineTransport())
    if fail_parallel:
        with pytest.raises(RuntimeError): runner.run(root, path, receipts, execute=True)
        assert state['calls'] == 5
        halt = json.loads((receipts/'halt.json').read_text())
        assert halt['calls'] == 5 and halt['in_flight_remaining'] == 0
        assert halt['charged_or_reserved_usd'] == pytest.approx(.0290224 + 4*.00154)
        assert len(list(receipts.glob('*/validated.json'))) == 4
        assert len(list(receipts.glob('*/failure.json'))) == 1
    else:
        result = runner.run(root, path, receipts, execute=True)
        assert result['completed'] and result['calls'] == 8
    assert state['peak'] == 4
    assert state['closed'] == state['calls'] and state['active'] == 0


def test_failure_during_request_persistence_cancels_unsent_item(prepared, tmp_path, monkeypatch):
    root, path = multi_prepared(prepared, count=6)
    receipts = tmp_path/'receipts'
    monkeypatch.setenv('OPENAI_API_KEY', 'synthetic-key-not-sent')
    third_request_persisting = threading.Event()
    second_failure_persisted = threading.Event()
    real_persist_bytes = runner.persist_bytes
    real_persist = runner.persist
    calls = []

    def slowed_persist_bytes(path, value):
        real_persist_bytes(path, value)
        if path.parent.name == 'episode-0003' and path.name == 'request.json':
            third_request_persisting.set()
            assert second_failure_persisted.wait(5)

    def record_persist(path, value):
        real_persist(path, value)
        if path.parent.name == 'episode-0002' and path.name == 'failure.json':
            second_failure_persisted.set()

    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, n): return json.dumps(valid_response()).encode()

    class OfflineTransport:
        def open(self, request, timeout):
            calls.append(request)
            if len(calls) == 2:
                assert third_request_persisting.wait(5)
                raise TimeoutError('Sibling fails while next request is being persisted')
            return Response()

    monkeypatch.setattr(runner, 'persist_bytes', slowed_persist_bytes)
    monkeypatch.setattr(runner, 'persist', record_persist)
    monkeypatch.setattr(runner.urllib.request, 'build_opener', lambda *a: OfflineTransport())
    with pytest.raises(RuntimeError): runner.run(root, path, receipts, execute=True)
    assert len(calls) == 2
    canceled = json.loads((receipts/'episode-0003/not-dispatched.json').read_text())
    assert canceled['http_attempts'] == 0
    assert canceled['charged_or_reserved_usd'] == 0
    assert canceled['automatic_replay_allowed'] is False
    assert (receipts/'episode-0003/request.json').exists()
    assert (receipts/'episode-0003/reservation.json').exists()
    assert not (receipts/'episode-0004').exists()
    halt = json.loads((receipts/'halt.json').read_text())
    assert halt['calls'] == 2 and halt['in_flight_remaining'] == 0
    assert halt['charged_or_reserved_usd'] == pytest.approx(.00154 + .0290224)
