"""Offline native-payload tests: real reservations; fake provider boundary."""
import ast
from contextlib import contextmanager
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
API = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


budget = module('_cached_real_budget', API / 'browsercode/budgets/request_budget.py')


class BaseModel:
    def __init__(self, **kwargs):
        self.options = kwargs

    def _build_payload(self, messages):
        items = []
        for message in messages:
            extra = message.get('extra') or {}
            if message['role'] == 'assistant':
                items.extend(deepcopy(extra['response_items']))
            elif extra.get('tool_call_id'):
                items.append(dict(type='function_call_output',
                                  call_id=extra['tool_call_id'], output=message['content']))
            else:
                items.append(dict(type='message', role=message['role'], content=message['content']))
        return dict(input=items, include=['reasoning.encrypted_content'])

    def _response_extra(self, response):
        return dict(response_items=deepcopy(response['output']))

    def format_message(self, **kwargs):
        return kwargs


def load_actor():
    mocks = {
        'request_budget': budget,
        'cuawright.webwright.models.openai_model': types.SimpleNamespace(OpenAIModel=BaseModel),
        'openwebrl.responses_actor_history': types.SimpleNamespace(response_input_bound=lambda *args: 1000),
        'openwebrl.sol_serial_parallel_policy': types.SimpleNamespace(selector_input=lambda *args: []),
    }
    with patch.dict(sys.modules, mocks):
        base = module('_cached_original_luna', API / 'browsercode/models/luna_model.py')
    with patch.dict(sys.modules, dict(mocks, luna_model=base)):
        return module('_portable_fixed_cached', API / 'browsercode/models/cached_luna_model.py')


class CachedTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir='/tmp', prefix='fixed-cached-offline-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.actor = load_actor()

    def create(self, arm, selection=None):
        count = self.actor.MODES[arm][0]
        barrier = threading.Barrier(count, timeout=5)
        lock = threading.Lock()
        calls = []
        def client(payload, role):
            with lock:
                calls.append((role, deepcopy(payload)))
            if role == 'selector':
                output = [dict(type='message', role='assistant', content=[dict(
                    type='output_text', text=json.dumps(dict(selected_index=count-1) if selection is None else selection))])]
            else:
                barrier.wait()
                index = int(role.split('_')[-1])
                output = [dict(type='reasoning', id='reason-'+str(index), encrypted_content='encrypted-'+str(index)),
                          dict(type='function_call', name='browser_execute', call_id='call-'+str(index),
                               arguments=json.dumps(dict(code='console.log('+str(index)+')', timeout=30000,
                                                         description='candidate '+str(index))))]
            return dict(id='response-'+role+'-'+str(len(calls)), model='gpt-6-luna', status='completed',
                        reasoning=dict(effort='high'), output=output,
                        usage=dict(input_tokens=100, output_tokens=20,
                                   input_tokens_details=dict(cached_tokens=80, cache_write_tokens=0)))
        pool = budget.ReservationBudget(self.root / (arm+'-pool'), '1', 100)
        gate = budget.SharedHTTPGate(self.root / (arm+'-gate'), 8)
        actor = self.actor.CachedLunaModel(arm, self.root / (arm+'-episode'), client, pool,
                                          http_gate=gate, model_name='gpt-6-luna',
                                          reasoning_effort='high', max_output_tokens=12288)
        return actor, calls, pool

    def test_both_widths_parallel_and_exact_full_context_selection_payload(self):
        for arm in self.actor.MODES:
            actor, calls, pool = self.create(arm)
            messages = [dict(role='user', content='task and actual observation')]
            before = deepcopy(messages)
            result = actor.query(messages)
            self.assertEqual(messages, before)
            n = actor.count
            self.assertEqual(len(calls), n+1)
            self.assertEqual(calls[-1][0], 'selector')
            candidates = [p for role,p in calls if role.startswith('candidate_')]
            self.assertTrue(all(p == candidates[0] for p in candidates))
            native = candidates[0]
            selection = calls[-1][1]
            self.assertEqual(selection['input'][:-1], native['input'])
            for key in ('model','reasoning','max_output_tokens','store','service_tier','tools','include','parallel_tool_calls'):
                self.assertEqual(selection[key], native[key])
            self.assertEqual(selection['tool_choice'], 'none')
            self.assertEqual(selection['input'][-1]['role'], 'developer')
            listing = json.loads(selection['input'][-1]['content'][1]['text'])
            self.assertEqual([x['index'] for x in listing['candidate_actions']], list(range(n)))
            self.assertEqual(selection['text']['format']['schema']['properties']['selected_index']['enum'], list(range(n)))
            self.assertEqual(result['extra']['response_items'][0]['encrypted_content'], 'encrypted-'+str(n-1))
            self.assertEqual(pool.snapshot()['calls'], n+1)
            self.assertTrue(all(r['status'] == 'known' for r in pool.snapshot()['requests'].values()))

    def test_only_selected_native_history_carries_forward_and_actual_cap_enforced(self):
        for arm in self.actor.MODES:
            actor, calls, pool = self.create(arm)
            messages = [dict(role='user', content='original task')]
            chosen = actor.query(messages)
            call_id = chosen['extra']['response_items'][-1]['call_id']
            messages += [chosen, dict(role='user', content='actual execution observation', extra=dict(tool_call_id=call_id))]
            actor.query(messages)
            native = [p for role,p in calls if role == 'candidate_0'][-1]
            reasoning = [x for x in native['input'] if x['type'] == 'reasoning']
            self.assertEqual([x['encrypted_content'] for x in reasoning], ['encrypted-'+str(actor.count-1)])
            self.assertEqual(native['input'][-1], dict(type='function_call_output',call_id=call_id,output='actual execution observation'))
            self.assertFalse(any(x.get('role') == 'developer' for x in native['input']))
            actor.turn = actor.decision_cap
            before = len(calls)
            with self.assertRaisesRegex(ValueError, 'decision cap'):
                actor.query(messages)
            self.assertEqual(len(calls), before)

    def test_malformed_selector_preserves_paid_calls_and_halts(self):
        actor, calls, pool = self.create('luna_n3_cached_high30', dict(selected_index=True))
        with self.assertRaisesRegex(ValueError, 'Malformed selected index'):
            actor.query([dict(role='user', content='actual observation')])
        self.assertEqual(len(calls), 4)
        self.assertEqual(pool.snapshot()['calls'], 4)
        self.assertTrue(pool.halted())
        self.assertFalse(list((actor.directory/'decisions').glob('*.json')))

    def test_explicit_gate_and_supported_protocol_required(self):
        pool = budget.ReservationBudget(self.root/'constructor-pool','1',100)
        with self.assertRaisesRegex(ValueError, 'shared HTTP gate'):
            self.actor.CachedLunaModel('luna_n3_cached_high30',self.root/'bad',lambda *_:None,pool)
        with self.assertRaisesRegex(ValueError, 'fixed cached'):
            self.actor.CachedLunaModel('luna_n5_high',self.root/'bad',lambda *_:None,pool)


if __name__ == '__main__':
    unittest.main(verbosity=2)
