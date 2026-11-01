"""Offline checks of every adaptive arm with a scripted provider; no network.

Run from a prepared campaign runtime:
  PYTHONPATH=<runtime>:<cuawright src>:<support src> python -B test_adaptive_model.py <runtime>
"""
import json
import sys
import tempfile
import threading
from pathlib import Path

sys.path.insert(0, sys.argv[1])
from bootstrap import CONFIG  # noqa: E402
import adaptive_model as am  # noqa: E402
from request_budget import ReservationBudget, SharedHTTPGate  # noqa: E402
from action_signature import agree, signature  # noqa: E402

CLICK_A = json.dumps({'code': "document.querySelector('#buy').click()", 'timeout': None, 'description': 'buy'})
CLICK_A2 = json.dumps({'code': "  document.querySelector('#buy').click();\n", 'timeout': 5000, 'description': 'click buy'})
CLICK_B = json.dumps({'code': "document.querySelector('#cart').click()", 'timeout': None, 'description': 'cart'})


def action(arguments):
    return [{'type': 'function_call', 'name': 'browser_execute', 'call_id': 'c', 'arguments': arguments}]


def deliberate(reason='ambiguous'):
    return [{'type': 'function_call', 'name': 'request_deliberation', 'call_id': 'd', 'arguments': json.dumps({'reason': reason})}]


def final(text='answer'):
    return [{'type': 'message', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': text}]}]


class Scripted:
    """Return scripted outputs by role; record every payload."""
    def __init__(self, script, fail_role=None):
        self.script, self.fail_role, self.calls, self.lock = script, fail_role, [], threading.Lock()

    def __call__(self, payload, role):
        with self.lock:
            self.calls.append((role, payload))
        if role == self.fail_role:
            raise TimeoutError('scripted provider timeout')
        output = self.script(role)
        if role == 'selector':
            output = [{'type': 'message', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': json.dumps({'selected_index': output})}]}]
        return {'model': am.MODEL, 'status': 'completed', 'reasoning': {'effort': 'high'}, 'output': output,
                'usage': {'input_tokens': 1000, 'output_tokens': 100, 'input_tokens_details': {'cached_tokens': 0}}}


def run(arm, script, fail_role=None):
    work = Path(tempfile.mkdtemp(prefix='adaptive-test-'))
    budget = ReservationBudget(work / 'budget', '5', 1000)
    client = Scripted(script, fail_role)
    model = am.AdaptiveLunaModel(arm, work / 'episode', client, budget, CONFIG['model_pricing']['gpt-6-luna'],
                                 harness='browsercode', model_name=am.MODEL, reasoning_effort='high', max_output_tokens=12288,
                                 response_mode='run_command_tool', action_field='bash_command', openai_api_key='x',
                                 attach_observation_screenshot=False, observation_template=CONFIG['observation_template'],
                                 http_gate=SharedHTTPGate(work / 'gate', 8))
    messages = [{'role': 'system', 'content': 'system'}, {'role': 'user', 'content': 'Task: buy the item'}]
    error = None
    try:
        message = model.query(messages)
    except BaseException as exc:
        message, error = None, exc
    record = None
    files = sorted((work / 'episode' / 'decisions').glob('*.json'))
    if files:
        record = json.loads(files[0].read_text())
    return client, record, message, error, budget, work


def roles(client):
    return sorted(role for role, _ in client.calls)


def check(condition, label):
    print(('PASS ' if condition else 'FAIL ') + label)
    if not condition:
        check.failed = True


check.failed = False

# Signatures: whitespace/timeout variants of one click agree; different targets and termination do not.
check(agree([{'command': CLICK_A, 'done': False}, {'command': CLICK_A2, 'done': False}]), 'same click agrees')
check(not agree([{'command': CLICK_A, 'done': False}, {'command': CLICK_B, 'done': False}]), 'different targets disagree')
check(not agree([{'done': True, 'final_response': 'x'}, {'done': True, 'final_response': 'x'}]), 'termination never agrees')
check(signature({'command': json.dumps({'code': 'console.log(document.title)'}), 'done': False})[1] == ('read',), 'read-only probe')

c, r, m, e, b, _ = run('luna_single_high', lambda role: action(CLICK_A))
check(e is None and roles(c) == ['candidate_0'] and not r['escalated'] and r['selector_response'] is None, 'single: one call, no selector')
check(c.calls[0][1]['tools'] == [am.TOOL], 'single: browser tool only')

c, r, m, e, b, _ = run('luna_n5_high', lambda role: 3 if role == 'selector' else action(CLICK_B if role == 'candidate_3' else CLICK_A))
check(e is None and roles(c) == sorted([f'candidate_{i}' for i in range(5)] + ['selector']) and r['selected_index'] == 3, 'n5: five + selector, selection honored')
check(json.loads(m['extra']['actions'][0]['command']) == json.loads(CLICK_B), 'n5: selected command executed')

c, r, m, e, b, _ = run('adaptive_agree', lambda role: action(CLICK_A2 if role == 'candidate_1' else CLICK_A))
check(e is None and roles(c) == ['candidate_0', 'candidate_1'] and not r['escalated'], 'agree: agreeing pair stops at two calls')

c, r, m, e, b, _ = run('adaptive_agree', lambda role: 0 if role == 'selector' else action(CLICK_B if role == 'candidate_1' else CLICK_A))
check(e is None and len(c.calls) == 6 and r['escalation_trigger'] == 'disagreement' and r['candidate_count'] == 5, 'agree: disagreement escalates to five + selector')

c, r, m, e, b, _ = run('adaptive_agree', lambda role: 1 if role == 'selector' else final() if role == 'candidate_0' else action(CLICK_A))
check(e is None and r['escalation_trigger'] == 'termination' and len(c.calls) == 6, 'agree: a final answer escalates')

c, r, m, e, b, _ = run('adaptive_self', lambda role: action(CLICK_A))
check(e is None and roles(c) == ['probe_0'] and not r['escalated'], 'self: acting probe executes directly')
probe = c.calls[0][1]
check(probe['tools'] == [am.TOOL, am.DELIBERATION_TOOL] and probe['input'][-1] == am.DELIBERATION_MESSAGE, 'self: probe sees instruction + tool')
check('response_items' in m['extra'], 'self: executed probe keeps native history')

c, r, m, e, b, _ = run('adaptive_self', lambda role: deliberate('two similar buttons') if role == 'probe_0' else 2 if role == 'selector' else action(CLICK_A))
check(e is None and len(c.calls) == 7 and r['escalation_trigger'] == 'self_request' and r['deliberation_reasons'] == ['two similar buttons'], 'self: request triggers five fresh + selector')
fresh = [p for role, p in c.calls if role.startswith('candidate_')]
probe = next(p for role, p in c.calls if role == 'probe_0')
check(all(p['tools'] == [am.TOOL] and p['input'] == probe['input'][:-1] for p in fresh), 'self: escalation candidates identical to n5 candidates')
check(r['selected_role'].startswith('candidate_'), 'self: a deliberation probe is never executed')

c, r, m, e, b, _ = run('adaptive_self_agree', lambda role: deliberate() if role == 'probe_1' else 4 if role == 'selector' else action(CLICK_A))
check(e is None and len(c.calls) == 7 and r['candidate_count'] == 5 and r['candidate_roles'][0] == 'probe_0', 'self_agree: acting probe kept, four fresh + selector')

c, r, m, e, b, _ = run('adaptive_self_agree', lambda role: action(CLICK_A2 if role == 'probe_1' else CLICK_A))
check(e is None and len(c.calls) == 2 and not r['escalated'], 'self_agree: agreeing probes act at two calls')


c, r, m, e, b, _ = run('luna_n5_cached', lambda role: 2 if role == 'selector' else action(CLICK_B if role == 'candidate_2' else CLICK_A))
sel = next(p for role, p in c.calls if role == 'selector')
cand = next(p for role, p in c.calls if role == 'candidate_0')
check(e is None and len(c.calls) == 6 and r['selected_index'] == 2 and r['selector_mode'] == 'cached_native', 'n5 cached: five + cached selector, selection honored')
check(sel['input'][:-1] == cand['input'] and sel['tools'] == cand['tools'] and sel['tool_choice'] == 'none'
      and sel['reasoning'] == cand['reasoning'] and sel['text']['format']['type'] == 'json_schema', 'cached selector: candidate prefix, tools and effort; text-only answer')
check('candidate_actions' in sel['input'][-1]['content'][1]['text'], 'cached selector: proposals listed last')

c, r, m, e, b, _ = run('adaptive_agree_cached', lambda role: action(CLICK_A2 if role == 'candidate_1' else CLICK_A))
check(e is None and len(c.calls) == 2 and not r['escalated'], 'agree cached: agreeing pair stops at two calls')
c, r, m, e, b, _ = run('adaptive_agree_cached', lambda role: 4 if role == 'selector' else action(CLICK_B if role == 'candidate_1' else CLICK_A))
check(e is None and len(c.calls) == 6 and r['candidate_count'] == 5 and r['selector_mode'] == 'cached_native', 'agree cached: disagreement escalates to five + cached selector')
c, r, m, e, b, _ = run('adaptive_agree3_cached', lambda role: 2 if role == 'selector' else action(CLICK_B if role == 'candidate_1' else CLICK_A))
check(e is None and roles(c) == ['candidate_0', 'candidate_1', 'candidate_2', 'selector'] and r['candidate_count'] == 3, 'agree3 cached: disagreement escalates to three + cached selector')
c, r, m, e, b, _ = run('luna_n5_high', lambda role: 0 if role == 'selector' else action(CLICK_A))
check(r['selector_mode'] == 'projected', 'phase-1 n5 keeps the projected selector')

c, r, m, e, b, work = run('adaptive_agree', lambda role: action(CLICK_A), fail_role='candidate_1')
snapshot = b.snapshot()
check(isinstance(e, TimeoutError) and not b.halted() and (work / 'episode' / 'episode-model-failure.json').exists(), 'provider error: episode fails, shared pool not halted')
check(snapshot['unknown_calls'] == 1 and snapshot['charged_or_reserved_usd'] > 0, 'provider error: unknown usage keeps its full reserve')

sys.exit(1 if check.failed else 0)
