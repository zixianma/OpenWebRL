"""Real Unix-socket/process tests; transport is scripted and never contacts a provider."""
import json
import io
import multiprocessing
import os
from pathlib import Path
import socket
import struct
import tempfile
import time
import unittest
import uuid

from openwebrl.webprm_broker import (
    BrokerError, BrokerProcess, MAX_FRAME_BYTES, _receive, _send,
    make_broker_client, payload_sha256,
)


MODEL = 'scripted-fixed-model'
IDENTITY = 'a' * 64


def scripted_completion(config, payload, journal, *, local=False, endpoint=None, purpose=None):
    from openwebrl.jev_eval import append_json
    from openwebrl.webprm_worker import SharedBudget
    # A fresh process has no monkeypatch inheritance: block provider/network here.
    import urllib.request
    urllib.request.urlopen = lambda *_a, **_k: (_ for _ in ()).throw(AssertionError('No network allowed'))
    assert endpoint == ('http://127.0.0.1:8123/v1' if local else None)
    budget = SharedBudget(config['budget_path'], config['api_usd_cap'])
    ident = None if local else budget.reserve(.1, payload['model'])
    append_json(journal, dict(event='request', payload=payload, purpose=purpose, local=local, reservation=ident))
    if config.get('scripted_behavior') == 'hang':
        time.sleep(30)
    if config.get('scripted_behavior') == 'error':
        raise RuntimeError('SECRET_KEY=do-not-return https://credential.example/?token=hidden')
    result = dict(model=payload['model'], choices=[dict(finish_reason='stop', message=dict(content=config.get('scripted_text', 'Yes')))],
                  usage=dict(prompt_tokens=12, completion_tokens=2))
    if config.get('scripted_behavior') == 'token_overrun':
        result['usage']['completion_tokens'] = 999
    append_json(journal, dict(event='response', reservation=ident, response=result))
    if not local:
        budget.settle(ident, .01, result['usage'])
    return dict(result, secret_internal_field='must not cross boundary')


def unrelated_wait():
    time.sleep(30)


def existing_completion_with_scripted_http(config, payload, journal, **kwargs):
    """Exercise actual reservation/journal/settlement with the HTTP seam blocked."""
    import openwebrl.webprm_worker as worker
    from unittest.mock import patch
    def respond(request, timeout):
        assert request.full_url == 'https://api.openai.com/v1/chat/completions'
        assert json.loads(request.data) == payload
        assert request.get_header('Authorization') == 'Bearer fake-test-key'
        assert 0 < timeout <= 90
        body = dict(model=payload['model'], choices=[dict(finish_reason='stop', message=dict(content='exact v6 path'))],
                    usage=dict(prompt_tokens=12, completion_tokens=2))
        if 'scripted_usage' in config:
            body['usage'] = config['scripted_usage']
        if 'scripted_usage_case' in config:
            cases = {'-1': -1, "'12'": '12', 'nan': float('nan'), 'inf': float('inf'), 'True': True, 'None': None}
            body['usage']['prompt_tokens'] = cases[config['scripted_usage_case']]
        if config.get('scripted_wrong_model'):
            body['model'] = 'untrusted-returned-model'
        return io.BytesIO(json.dumps(body).encode())
    with patch.object(worker, 'load_api_credentials', return_value='fake-test-key'), patch.object(worker, 'urlopen', side_effect=respond):
        return worker.completion(config, payload, journal, **kwargs)


class BrokerTests(unittest.TestCase):
    def setUp(self):
        # Intentionally retain these small test artifacts; never touch protected home.
        self.root = Path(tempfile.mkdtemp(prefix='wbkr-', dir='/tmp'))
        self.deadline = time.time() + 20
        self.config = dict(identity_sha256=IDENTITY, deadline=self.deadline,
                           budget_path=str(self.root/'budget.json'), api_usd_cap=1.,
                           prices_usd_per_token={MODEL: [1e-6, 1e-6]})
        self.payload = dict(model=MODEL, messages=[dict(role='user', content='fixed request')],
                            max_tokens=20, temperature=0)
        self.kwargs = dict(socket_path=str(self.root/'s'), deadline=self.deadline,
            cells={'cell': dict(config=self.config, journal=str(self.root/'cell.jsonl'),
                                allowed_routes=[[MODEL, 'policy']])},
            routes={(MODEL, 'policy'): dict(local=False, endpoint=None, max_input_tokens=10000, max_output_tokens=20)},
            max_dispatches=5, audit_path=str(self.root/'broker.jsonl'), completion_fn=scripted_completion)

    def client(self, **overrides):
        return make_broker_client(self.kwargs['socket_path'], cell_id='cell',
                                  identity_sha256=IDENTITY, deadline=self.deadline, **overrides)

    def call(self, **overrides):
        return self.client()(self.config, overrides.pop('payload', self.payload), '/irrelevant/client/journal',
                             purpose=overrides.pop('purpose', 'policy'), **overrides)

    def rows(self):
        path = self.root/'broker.jsonl'
        return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []

    def test_paid_result_has_exact_single_settled_reservation_and_safe_receipt(self):
        with BrokerProcess(**self.kwargs) as process:
            result = self.call(endpoint='https://client-cannot-select.example')
            self.assertEqual(result['choices'][0]['message']['content'], 'Yes')
            self.assertNotIn('secret_internal_field', result)
            proof = result['_webprm_broker']
            self.assertEqual(proof['payload_sha256'], payload_sha256(self.payload))
            self.assertEqual(proof['reservation_state'], 'settled')
            self.assertEqual(proof['reservation_id'], 0)
            self.assertEqual(len(json.loads((self.root/'budget.json').read_text())['calls']), 1)
            self.assertNotIn('fixed request', (self.root/'broker.jsonl').read_text())
        self.assertTrue(process.closure['reaped'])
        self.assertFalse(Path('/proc', str(process.process.pid)).exists())

    def test_existing_v6_completion_ledger_journal_and_receipt_without_network(self):
        self.kwargs['completion_fn'] = existing_completion_with_scripted_http
        with BrokerProcess(**self.kwargs):
            result = self.call()
        self.assertEqual(result['choices'][0]['message']['content'], 'exact v6 path')
        ledger = json.loads((self.root/'budget.json').read_text())
        self.assertEqual(len(ledger['calls']), 1)
        self.assertEqual(ledger['calls'][0]['state'], 'settled')
        self.assertAlmostEqual(ledger['charged'], 14e-6)
        self.assertEqual(result['_webprm_broker']['usage'], ledger['calls'][0]['usage'])
        self.assertNotIn('fake-test-key', (self.root/'cell.jsonl').read_text())

    def test_invalid_v6_usage_preserves_full_reservation_without_corrupting_ledger(self):
        for invalid in (-1, '12', float('nan'), float('inf'), True, None):
            self.setUp()
            self.config['scripted_usage'] = dict(prompt_tokens=invalid, completion_tokens=2)
            # Nonfinite responses are scripted HTTP data, never host policy values.
            self.kwargs['completion_fn'] = existing_completion_with_scripted_http
            # Keep invalid values out of the immutable broker policy JSON hash.
            self.config.pop('scripted_usage')
            self.config['scripted_usage_case'] = repr(invalid)
            with BrokerProcess(**self.kwargs):
                with self.assertRaises(BrokerError):
                    self.call()
            data = json.loads((self.root/'budget.json').read_text())
            self.assertEqual(len(data['calls']), 1)
            self.assertEqual(data['calls'][0]['state'], 'reserved')
            self.assertEqual(data['charged'], data['calls'][0]['reserved'])
            self.assertGreater(data['charged'], 0)

    def test_returned_model_mismatch_keeps_full_original_reservation(self):
        self.kwargs['completion_fn'] = existing_completion_with_scripted_http
        self.config['scripted_wrong_model'] = True
        with BrokerProcess(**self.kwargs):
            with self.assertRaises(BrokerError):
                self.call()
        data = json.loads((self.root/'budget.json').read_text())
        self.assertEqual(len(data['calls']), 1)
        self.assertEqual(data['calls'][0]['state'], 'reserved')
        self.assertEqual(data['charged'], data['calls'][0]['reserved'])

    def test_route_and_cell_identity_rejections_do_not_dispatch(self):
        self.kwargs['routes'][('other-model', 'policy')] = dict(
            local=False, endpoint=None, max_input_tokens=10000, max_output_tokens=20)
        with BrokerProcess(**self.kwargs):
            for override in (dict(purpose='native_fuzzy'), dict(local=True),
                             dict(payload=dict(self.payload, model='other-model')),
                             dict(payload=dict(self.payload, stream=True)),
                             dict(payload=dict(self.payload, n=2)),
                             dict(payload=dict(self.payload, max_tokens=21))):
                with self.assertRaises(BrokerError):
                    self.call(**override)
            with self.assertRaises(BrokerError):
                self.client()(dict(self.config, identity_sha256='b'*64), self.payload, 'x', purpose='policy')
        self.assertEqual(self.rows(), [])
        self.assertFalse((self.root/'budget.json').exists())

    def test_local_route_uses_fixed_host_endpoint_and_no_paid_reservation(self):
        self.kwargs['routes'][(MODEL, 'policy')].update(local=True, endpoint='http://127.0.0.1:8123/v1')
        with BrokerProcess(**self.kwargs):
            result = self.call(local=True, endpoint='https://ignored-client-choice.example')
        self.assertEqual(result['_webprm_broker']['reservation_state'], 'local')
        self.assertIsNone(result['_webprm_broker']['reservation_id'])
        self.assertFalse((self.root/'budget.json').exists())

    def test_error_has_no_credentials_and_unknown_reservation_is_preserved(self):
        self.config['scripted_behavior'] = 'error'
        with BrokerProcess(**self.kwargs):
            with self.assertRaises(BrokerError) as error:
                self.call()
        self.assertNotIn('SECRET_KEY', str(error.exception))
        data = json.loads((self.root/'budget.json').read_text())
        self.assertEqual(data['calls'][0]['state'], 'reserved')
        self.assertEqual(data['charged'], .1)
        self.assertEqual(len(data['calls']), 1)
        self.assertNotIn('SECRET_KEY', (self.root/'broker.jsonl').read_text())

    def test_durable_cap_and_duplicate_id_survive_broker_restart(self):
        request = dict(protocol=1, cell_id='cell', identity_sha256=IDENTITY,
            request_id=uuid.uuid4().hex, purpose='policy', local=False, payload=self.payload, expires_at=self.deadline)
        def raw_call():
            with socket.socket(socket.AF_UNIX) as connection:
                connection.connect(self.kwargs['socket_path'])
                _send(connection, request, MAX_FRAME_BYTES, self.deadline)
                return _receive(connection, MAX_FRAME_BYTES, self.deadline)
        self.kwargs['max_dispatches'] = 1
        with BrokerProcess(**self.kwargs):
            self.assertTrue(raw_call()['ok'])
            self.assertFalse(raw_call()['ok'])
        self.kwargs['socket_path'] = str(self.root/'s2')
        with BrokerProcess(**self.kwargs):
            self.assertFalse(raw_call()['ok'])
            with self.assertRaises(BrokerError):
                self.call()
        self.assertEqual(len(json.loads((self.root/'budget.json').read_text())['calls']), 1)

    def test_paid_and_local_caps_are_separate_and_durable(self):
        self.kwargs.update(max_dispatches=3, max_paid_dispatches=1, max_local_dispatches=1)
        self.kwargs['routes'][(MODEL, 'critic')] = dict(local=True, endpoint='http://127.0.0.1:8123/v1',
                                                      max_input_tokens=10000, max_output_tokens=20)
        self.kwargs['cells']['cell']['allowed_routes'].append([MODEL, 'critic'])
        with BrokerProcess(**self.kwargs):
            self.call()
            with self.assertRaises(BrokerError): self.call()
            self.call(local=True, purpose='critic')
        self.kwargs['socket_path'] = str(self.root/'s2')
        with BrokerProcess(**self.kwargs):
            with self.assertRaises(BrokerError): self.call()
            with self.assertRaises(BrokerError): self.call(local=True, purpose='critic')
        started = [row for row in self.rows() if row['event']=='dispatch_started']
        self.assertEqual([row['local'] for row in started], [False, True])
        self.assertEqual(len(json.loads((self.root/'budget.json').read_text())['calls']), 1)

    def test_wrong_peer_uid_and_oversized_frame_rejected_before_dispatch(self):
        self.kwargs['expected_uid'] = os.getuid()+1
        with BrokerProcess(**self.kwargs):
            with self.assertRaises((BrokerError, OSError)):
                self.call()
        self.kwargs['expected_uid'] = os.getuid()
        self.kwargs['socket_path'] = str(self.root/'s2')
        with BrokerProcess(**self.kwargs):
            with socket.socket(socket.AF_UNIX) as connection:
                connection.connect(self.kwargs['socket_path'])
                connection.sendall(struct.pack('!I', MAX_FRAME_BYTES + 1))
                self.assertFalse(_receive(connection, MAX_FRAME_BYTES, self.deadline)['ok'])
        self.assertEqual(self.rows(), [])

    def test_token_overrun_halts_shared_paid_budget(self):
        self.config['scripted_behavior'] = 'token_overrun'
        with BrokerProcess(**self.kwargs):
            with self.assertRaises(BrokerError):
                self.call()
            with self.assertRaises(BrokerError):
                self.call()
        data = json.loads((self.root/'budget.json').read_text())
        self.assertEqual(data['halt_reason'], 'broker_observed_token_cap_overrun_or_invalid_usage')
        self.assertEqual(len(data['calls']), 1)

    def test_parent_deadline_kills_hung_dispatch_and_keeps_unrelated_process(self):
        unrelated = multiprocessing.get_context('spawn').Process(target=unrelated_wait)
        unrelated.start()
        self.deadline = time.time()+2
        self.config.update(deadline=self.deadline, scripted_behavior='hang')
        self.kwargs['deadline'] = self.deadline
        process = BrokerProcess(shutdown_seconds=.1, **self.kwargs)
        try:
            with process:
                with self.assertRaises((BrokerError, OSError, TimeoutError)):
                    self.call()
            self.assertTrue(process.closure['reaped'])
            self.assertTrue(unrelated.is_alive())
            self.assertEqual(json.loads((self.root/'budget.json').read_text())['calls'][0]['state'], 'reserved')
        finally:
            unrelated.terminate(); unrelated.join(timeout=2)
        self.assertFalse(unrelated.is_alive())

    def test_expired_deadline_does_not_spawn(self):
        self.kwargs['deadline'] = time.time()-1
        process = BrokerProcess(**self.kwargs)
        with self.assertRaises(TimeoutError):
            process.start()
        self.assertIsNone(process.process)
        self.assertEqual(self.rows(), [])

    def test_concurrent_broker_cannot_share_dispatch_ledger(self):
        with BrokerProcess(**self.kwargs):
            other = BrokerProcess(**dict(self.kwargs, socket_path=str(self.root/'other')))
            with self.assertRaises(BrokerError):
                other.start()
            self.assertTrue(other.closure['reaped'])
            self.call()
        self.assertEqual(len(json.loads((self.root/'budget.json').read_text())['calls']), 1)

    def test_native_fuzzy_uses_real_broker_once_without_double_reservation(self):
        from openwebrl.webprm_webarena_fuzzy import MODEL as FUZZY, BrokerBudgetedTransport, NativeFuzzyJudge
        from openwebrl.webprm_worker import input_token_upper_bound
        source = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-baselines-20261009/sources/arbiter')
        if not source.exists():
            self.skipTest('Pinned native source absent')
        self.config.update(prices_usd_per_token={FUZZY: [1e-6, 1e-6]}, scripted_text='correct')
        self.kwargs['cells']['cell']['allowed_routes'] = [[FUZZY, 'native_fuzzy']]
        self.kwargs['routes'] = {(FUZZY, 'native_fuzzy'): dict(local=False, endpoint=None,
                                                           max_input_tokens=10000, max_output_tokens=768)}
        with BrokerProcess(**self.kwargs):
            transport = BrokerBudgetedTransport(self.client(), self.config, self.root/'ignored.jsonl')
            judge = NativeFuzzyJudge(source, enabled=True, transport=transport, budget=None,
                journal_path=self.root/'native.jsonl', count_input_tokens=lambda request: input_token_upper_bound(request['messages']),
                max_input_tokens=10000, max_calls=1, deadline=self.deadline)
            self.assertEqual(judge.fuzzy_match('café', 'café', 'Unicode question'), 1)
            self.assertEqual(judge.provider_calls, 1)
        ledger = json.loads((self.root/'budget.json').read_text())
        self.assertEqual(len(ledger['calls']), 1)
        self.assertEqual(ledger['calls'][0]['state'], 'settled')
        self.assertFalse((self.root/'ignored.jsonl').exists())

    def test_unsafe_path_nonfinite_cap_prices_and_policy_restart_fail_closed(self):
        for change in ({'budget_path': '/gpfs/home/zixianma/must-not-write'},
                       {'api_usd_cap': float('nan')},
                       {'prices_usd_per_token': {MODEL:[float('inf'), 1e-6]}}):
            saved = dict(self.config); self.config.update(change)
            process = BrokerProcess(**self.kwargs)
            with self.assertRaises(BrokerError):
                process.start()
            self.assertTrue(process.closure['reaped'])
            self.config.clear(); self.config.update(saved)
        with BrokerProcess(**self.kwargs):
            self.call()
        self.kwargs['socket_path'] = str(self.root/'s2')
        self.kwargs['max_dispatches'] = 100
        process = BrokerProcess(**self.kwargs)
        with self.assertRaises(BrokerError):
            process.start()
        self.assertTrue(process.closure['reaped'])

    def test_derived_write_path_symlinks_rejected_without_touching_targets(self):
        for filename in ('budget.lock', 'budget.json.tmp', 'broker.jsonl.lock'):
            self.setUp()
            target = self.root/'preserved-target'
            target.write_text('untouched')
            (self.root/filename).symlink_to(target)
            process = BrokerProcess(**self.kwargs)
            with self.assertRaises(BrokerError):
                process.start()
            self.assertTrue(process.closure['reaped'])
            self.assertEqual(target.read_text(), 'untouched')
            self.assertFalse((self.root/'budget.json').exists())


if __name__ == '__main__':
    unittest.main()
