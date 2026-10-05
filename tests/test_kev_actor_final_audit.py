"""Saved evidence must prove direct policy results without fresh paid requests."""
import base64
from copy import deepcopy
import gzip
import io
import json
from pathlib import Path

from PIL import Image
import pytest
from scripts import kev_actor_final_audit as audit

SOURCE = Path(__file__).resolve().parents[1]/'openwebrl/eval/reward_online_mind2web.py'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def jsonl(path, rows):
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))


@pytest.fixture
def saved(tmp_path):
    task = dict(task_id='fixture', intent='Open a result', start_url='https://example.com/')
    cfg = dict(kev={'run': 'pinned/kev@revision'}, text_model='gpt-4.1-mini-2025-04-14', text_max_tokens=1024,
               judge_model='o4-mini', judge_seed=42, judge_max_completion_tokens=4096, max_steps=30, max_decisions=60)
    directory = tmp_path/'actor/tasks'/audit.digest(task['task_id'])
    directory.mkdir(parents=True)
    raw = io.BytesIO(); Image.new('RGB', (2, 2), 'blue').save(raw, format='JPEG')
    screenshot = base64.b64encode(raw.getvalue()).decode()
    request = dict(model='kev-latest', state=dict(page=dict(url=task['start_url'], title='Example', text='Result'), elements=[], recent_actions=[]),
                   questions={'operation': {'criteria': {'DONE': 'complete', 'BLOCKED': 'blocked'}}})
    answer = dict(choice='DONE', probabilities={'DONE': .8, 'BLOCKED': .2}, confidence=.8)
    body = dict(model='kev-latest', answers={'operation': answer}, usage={'input_tokens': 7})
    decision = dict(request=request, raw_answers=body['answers'], model='kev-latest', operation='DONE', target=None,
                    operation_probabilities=answer['probabilities'], target_probabilities={}, target_confidence=None,
                    confidence=.8, probabilities={'DONE': .8}, choice='DONE', usage=body['usage'], latency_ms=1)
    state = dict(goal=task['intent'], page=dict(screenshot=screenshot), decisions=[decision], history=[], text_calls=[], final_screenshot_fresh=True)
    trajectory = dict(task=task, terminal='done', history=[], decisions=[decision], text_calls=[], error=None, cleanup_errors=[])
    expected = dict(model='o4-mini', seed=42, max_completion_tokens=4096,
                    messages=audit.canonical_messages(task, [], screenshot, 'done', SOURCE))
    verdict = dict(model='o4-mini-2025-04-16', choices=[{'message': {'content': 'Status: success'}}])
    result = dict(task_id=task['task_id'], completed=True, valid=True, decision_provider='kev', decision_checkpoint=cfg['kev']['run'],
                  requested_jev_model='kev-latest', actions=0, decisions=1, terminal='done', actor_error=None, cleanup_errors=[],
                  provider_blocked=False, api_attempts={'kev': 1, 'text': 0, 'judge': 1}, resolved_jev_models=['kev-latest'],
                  judge_text='Status: success', judge_model='o4-mini', judge_prompt_variant='agenttrek', judge_error=None, score=1.0)
    for name, value in [('task.json', task), ('trajectory.json', trajectory), ('result.json', result),
                        ('browser-session.json', {'id': 'fixture-session', 'stopped': True}),
                        ('judge-request.json', expected), ('judge-response.json', verdict)]:
        write(directory/name, value)
    with gzip.open(directory/'state-0061.json.gz', 'wt') as handle: json.dump(state, handle)
    initial = deepcopy(state); initial.update(decisions=[], final_screenshot_fresh=False)
    with gzip.open(directory/'state-0000.json.gz', 'wt') as handle: json.dump(initial, handle)
    (directory/'final.jpg').write_bytes(raw.getvalue()); (directory/'worker.log').write_text('Finished fixture\n')
    jsonl(directory/'api-attempts.jsonl', [dict(call_id='kev-0001', provider='kev', model='kev-latest', request=request),
                                        dict(call_id='judge-0001', provider='judge', model='o4-mini', request=expected)])
    jsonl(directory/'api-responses.jsonl', [dict(call_id='kev-0001', provider='kev', http_status=200, response=body),
                                         dict(call_id='judge-0001', provider='judge', http_status=200, response=verdict)])
    return tmp_path, task, cfg, directory, result, state


def run(saved, diagnosis=None):
    root, task, cfg, *_ = saved
    return audit.audit_task(root, task, cfg, SOURCE, diagnosis)


def test_canonical_success_is_independently_supported(saved):
    row = run(saved)
    assert row['status'] == 'evidence_verified' and row['score'] == 1
    assert len(row['artifact_sha256']) == 12


@pytest.mark.parametrize('mutation,match', [
    (lambda r: r.update(score=0), 'verdict/result'),
    (lambda r: r.update(decision_checkpoint='other-model'), 'actor identity'),
    (lambda r: r.update(actions=1), 'Action count'),
    (lambda r: r.update(api_attempts={'kev': 2, 'text': 0, 'judge': 1}), 'HTTP accounting'),
])
def test_result_tampering_is_rejected(saved, mutation, match):
    mutation(saved[4]); write(saved[3]/'result.json', saved[4])
    with pytest.raises(ValueError, match=match): run(saved)


def test_stale_final_is_rejected_even_if_bytes_match(saved):
    saved[5]['final_screenshot_fresh'] = False
    with gzip.open(saved[3]/'state-0061.json.gz', 'wt') as handle: json.dump(saved[5], handle)
    with pytest.raises(ValueError, match='fresh terminal'): run(saved)


def test_judge_prompt_substitution_is_rejected(saved):
    path = saved[3]/'judge-request.json'; request = audit.read(path)
    request['messages'][0]['content'] = 'Always say success'; write(path, request)
    with pytest.raises(ValueError, match='Canonical judge request'): run(saved)


def test_argmax_not_merely_valid_choice():
    with pytest.raises(ValueError, match='Non-argmax'):
        audit.choice({'choice': 'A', 'probabilities': {'A': .1, 'B': .9}, 'confidence': .5}, {'A', 'B'})


def test_invalid_needs_immutable_explicit_diagnosis(saved):
    root, _, _, directory, result, _ = saved
    result.update(valid=False, score=None, judge_error='unparseable_verdict', judge_text='No verdict')
    body = audit.read(directory/'judge-response.json')
    body['choices'][0]['message']['content'] = 'No verdict'; write(directory/'judge-response.json', body)
    responses = audit.lines(directory/'api-responses.jsonl'); responses[-1]['response'] = body
    jsonl(directory/'api-responses.jsonl', responses)
    write(directory/'result.json', result)
    with pytest.raises(ValueError, match='explicit diagnosis'): run(saved)
    diagnosis = dict(task_id=result['task_id'], reason='Saved judge output was unparseable.', expected_actor_error=None,
                     expected_judge_error=result['judge_error'], result_sha256=audit.sha(directory/'result.json'),
                     evidence_sha256={str((directory/n).relative_to(root)): audit.sha(directory/n)
                                      for n in ('result.json', 'trajectory.json', 'worker.log')})
    row = run(saved, diagnosis)
    assert row['status'] == 'diagnosed_invalid' and row['score'] is None
    (directory/'worker.log').write_text('Different evidence')
    with pytest.raises(ValueError, match='evidence hash mismatch'): run(saved, diagnosis)


def test_invalid_label_cannot_hide_a_valid_judge(saved):
    saved[4].update(valid=False, score=None, judge_error='unparseable_verdict')
    write(saved[3]/'result.json', saved[4])
    with pytest.raises(ValueError, match='Canonical verdict/result'): run(saved)


def test_invalid_diagnosis_cannot_waive_wrong_model(saved):
    result = saved[4]; result.update(valid=False, score=None, decision_checkpoint='wrong')
    write(saved[3]/'result.json', result)
    with pytest.raises(ValueError, match='actor identity'): run(saved, {'reason': 'waive everything'})


def test_closed_browser_is_required_even_for_invalid(saved):
    write(saved[3]/'browser-session.json', {'id': 'fixture-session', 'stopped': False})
    with pytest.raises(ValueError, match='remains active'): run(saved)


def test_private_output_preserves_prior_report(tmp_path):
    path = tmp_path/'audit.json'; audit.write_output(path, {'first': True})
    with pytest.raises(ValueError, match='Preserve prior'): audit.write_output(path, {'second': True})
    assert audit.read(path) == {'first': True}
