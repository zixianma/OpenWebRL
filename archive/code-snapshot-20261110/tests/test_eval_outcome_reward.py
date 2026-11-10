import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import eval_outcome_reward as E  # noqa: E402


def test_binary_metrics_counts_and_kappa():
    m = E.binary_metrics([(1, 1), (1, 0), (0, 0), (0, 0)])
    assert (m['tp'], m['fp'], m['fn'], m['tn']) == (1, 1, 0, 2)
    assert m['accuracy'] == 0.75 and m['false_positive_rate'] == 1/3
    assert abs(m['kappa'] - 0.5) < 1e-9


def test_gate_requires_grounded_claims_and_completion():
    ok = {'claims': [{'status': 'supported'}, {'status': 'partial'}], 'answer_completes_task': True}
    assert E.gate(ok) == 1.0
    assert E.gate(dict(ok, answer_completes_task=False)) == 0.0
    assert E.gate(dict(ok, claims=[{'status': 'not_found'}])) == 0.0
    assert E.gate(dict(ok, claims=[{'status': 'made_up'}])) == 0.0
    assert E.gate(None) is None


def test_fara_steps_maps_terminate_to_done_and_screens_to_steps():
    log = '\n'.join(json.dumps({'arguments': a}) for a in [
        {'action': 'left_click', 'coordinate': [1, 2], 'thoughts': 'open'},
        {'action': 'terminate', 'status': 'success', 'thoughts': 'answer text'}])
    row = dict(web_surfer_log=log, final_answer='answer text', screenshots=[{'bytes': b'\x89PNG'}] * 3)
    steps, answer = E.fara_steps(row)
    assert steps == [('open', 'left_click', {'coordinate': [1, 2]}), ('answer text', 'done', {'response': 'answer text'})]
    assert answer == 'answer text'
    assert [k for k, _ in E.fara_screens(row, 'fara7b_om2w_browserbase')] == [1, 2, 3]
    assert [k for k, _ in E.fara_screens(row, 'internal')] == [None, None, None]
    assert E.fara_steps(dict(row, final_answer='<no_answer>'))[1] == ''


def test_checklist_gate_needs_every_requirement_satisfied():
    cons = [{'id': 'C1', 'requirement': 'red'}, {'id': 'C2', 'requirement': 'size 10'}]
    ok = {'requirements': [{'id': 'C1', 'status': 'satisfied'}, {'id': 'C2', 'status': 'satisfied'}]}
    assert E.checklist_gate(ok, cons) == 1.0
    assert E.checklist_gate({'requirements': [{'id': 'C1', 'status': 'satisfied'}, {'id': 'C2', 'status': 'not_verifiable'}]}, cons) == 0.0
    assert E.checklist_gate({'requirements': [{'id': 'C1', 'status': 'satisfied'}]}, cons) == 0.0  # missing id
    assert E.checklist_gate(None, cons) is None


def test_checklist_messages_insert_requirements_before_answer():
    msgs = [{'role': 'system', 'content': 'x'},
            {'role': 'user', 'content': [{'type': 'text', 'text': 'TASK: t\n\nFINAL ANSWER: a\n\nTRAJECTORY:\nR1: r'}]}]
    out = E.checklist_messages(msgs, [{'id': 'C1', 'requirement': 'red'}])
    assert out[0]['content'] == E.CHECKLIST_CHECK
    assert out[1]['content'][0]['text'] == 'TASK: t\n\nREQUIREMENTS:\nC1: red\n\nFINAL ANSWER: a\n\nTRAJECTORY:\nR1: r'
    assert msgs[1]['content'][0]['text'].startswith('TASK: t\n\nFINAL')  # input not mutated
