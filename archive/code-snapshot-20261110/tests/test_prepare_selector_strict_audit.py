import base64
import copy
import importlib.util
import io
import json
from pathlib import Path

import pytest
from PIL import Image

SPEC = importlib.util.spec_from_file_location(
    "prepare_selector_strict_audit", Path(__file__).parents[1] / "scripts/prepare_selector_strict_audit.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


@pytest.fixture
def episode():
    f = io.BytesIO()
    Image.new('RGB', (64, 64), 'white').save(f, format='PNG')
    png = f.getvalue()
    encoded = base64.b64encode(png).decode()
    response = '<think>PRIVATE_ACTOR_THOUGHT</think><tool_call>{"name":"done","arguments":{"response":"The store is open 9–5."}}</tool_call>'
    result = {
        'task_id': 'task-a', 'valid': True, 'intent': 'Find the opening hours.',
        'total_steps': 1, 'prompt': 'PRIVATE_PROMPT', 'response': response,
        'reward': 1, 'mode': 'SECRET_SELECTOR_ID',
        'metadata': {
            'full_image_list': [encoded], 'reward': {'judge_text': 'OLD_JUDGE_VERDICT'},
            'messages': [
                {'role': 'system', 'content': 'SECRET_MODEL_ID'},
                {'role': 'user', 'content': [{'type': 'text', 'text': '<user_request>Find the opening hours.</user_request><observation>Tab 0: https://example.org/ Hours<|vision_start|><|image_pad|><|vision_end|></observation>'},
                                           {'type': 'image_url', 'image_url': 'data:image/png;base64,' + encoded}]},
                {'role': 'assistant', 'content': response}],
            'step_tool_responses': [{'tool_name': 'done', 'tool_response': 'Succeed: task marked as done.'}]}}
    row = {'task_id': 'task-a', 'mode': 'SECRET_SELECTOR_ID', 'valid': True,
           'reward': 1, 'issues': [], 'evidence_verified': True, 'final_sha256': audit.sha(png)}
    return result, row, png


def test_allowlist_excludes_reasoning_identity_old_verdict_and_images(episode):
    result, row, png = episode
    original = copy.deepcopy(result)
    evidence = audit.build_evidence(result, row, png)
    dumped = json.dumps(evidence)
    for secret in ['PRIVATE_ACTOR_THOUGHT', 'PRIVATE_PROMPT', 'OLD_JUDGE_VERDICT',
                   'SECRET_SELECTOR_ID', 'SECRET_MODEL_ID', 'data:image', '<think>']:
        assert secret not in dumped
    assert evidence['terminal_answer'] == 'The store is open 9–5.'
    assert result == original


@pytest.mark.parametrize('change', ['invalid', 'missing_image', 'stale_image', 'missing_feedback', 'audit_failed'])
def test_missing_or_invalid_evidence_rejected(episode, change):
    result, row, png = episode
    if change == 'invalid': result['valid'] = False
    if change == 'missing_image': result['metadata']['full_image_list'] = []
    if change == 'stale_image': row['final_sha256'] = 'bad'
    if change == 'missing_feedback': result['metadata']['step_tool_responses'] = []
    if change == 'audit_failed': row['issues'] = ['incomplete']
    with pytest.raises(audit.EvidenceError): audit.build_evidence(result, row, png)


def test_action_feedback_pairing_and_no_reasoning(episode):
    result, row, png = episode
    result['total_steps'] = 2
    messages = result['metadata']['messages']
    messages[1:1] = []
    messages.insert(2, {'role': 'assistant', 'content': '<think>SECRET</think><tool_call>{"name":"click","arguments":{"point_2d":[20,20]}}</tool_call>'})
    messages.insert(3, {'role': 'user', 'content': [{'type': 'text', 'text': '<tool_response>Succeed: click executed.<observation>Hours: 9–5</observation></tool_response>'}]})
    evidence = audit.build_evidence(result, row, png)
    assert evidence['executed_steps'][0]['function_calls'][0]['name'] == 'click'
    assert 'Hours: 9–5' in evidence['executed_steps'][0]['tool_feedback_and_observation']
    assert 'SECRET' not in json.dumps(evidence)


def test_malformed_action_preserved_as_failure_without_thoughts(episode):
    result, row, png = episode
    result['metadata']['messages'][-1]['content'] = '<think>SECRET</think><tool_call>bad JSON</tool_call>'
    result['metadata']['step_tool_responses'] = [{'tool_name': 'parser', 'tool_response': 'No executable function found.'}]
    evidence = audit.build_evidence(result, row, png)
    assert evidence['executed_steps'][0]['action_parse_failed'] is True
    assert evidence['executed_steps'][0]['function_calls'] == []
    assert evidence['terminal_answer'] is None


def test_independently_verified_parser_termination_needs_no_browser_feedback(episode):
    result, row, png = episode
    result['metadata']['messages'][-1]['content'] = '<think>SECRET</think><tool_call>bad</tool_call>'
    result['metadata']['step_tool_responses'] = None
    result['terminate_reason'] = 'format_error_failed'
    row['deterministic_format_failure_verified'] = True
    evidence = audit.build_evidence(result, row, png)
    assert evidence['executed_steps'][-1]['terminal_tool_feedback'][0]['tool_name'] == 'framework_parser'


@pytest.mark.parametrize('status,success,finding', [
    ('success', True, 'satisfied'), ('failure', False, 'not_satisfied'),
    ('evidence_insufficient', None, 'not_verifiable')])
def test_verdict_invariants_accept_distinct_outcomes(status, success, finding):
    audit.validate_verdict({'status': status, 'strict_success': success,
                           'requirements': [{'finding': finding}]})


@pytest.mark.parametrize('status,success,finding', [
    ('success', True, 'not_verifiable'), ('failure', False, 'not_verifiable'),
    ('evidence_insufficient', None, 'not_satisfied'), ('success', False, 'satisfied')])
def test_verdict_invariants_reject_unsupported_scoring(status, success, finding):
    with pytest.raises(ValueError):
        audit.validate_verdict({'status': status, 'strict_success': success,
                               'requirements': [{'finding': finding}]})


def save_cohort(root, episode, invalid=False):
    result, row, png = copy.deepcopy(episode)
    if invalid: result['valid'] = row['valid'] = False
    folder = root / row['mode']
    key = audit.sha(row['task_id'].encode())
    (folder / 'results').mkdir(parents=True)
    (folder / 'final').mkdir()
    rollout = folder / 'rollout.json'
    rollout.write_text(json.dumps({'sample_id': result['task_id'], 'total_steps': result['total_steps'],
                                  'llm_input_texts': result['prompt'], 'llm_response': result['response']}))
    path = folder / 'results' / (key + '.json')
    path.write_text(json.dumps(result))
    (folder / 'final' / (key + '.png')).write_bytes(png)
    row.update(rollout=str(rollout), result_sha256=audit.sha(path.read_bytes()))
    (root / 'independent-audit.json').write_text(json.dumps({'rows': [row], 'all_results_audited': True}))


def test_full_preparation_blinds_and_preserves_sources(tmp_path, episode):
    root, out = tmp_path/'cohort', tmp_path/'prepared'
    save_cohort(root, episode)
    before = {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    summary = audit.prepare([root], out, expected_per_arm=1)
    assert summary['counts'] == {'ready': 1}
    assert summary['bounded_proposal']['eligible_initial_calls'] == 1
    assert summary['bounded_proposal']['automatic_retries'] == 0
    assert summary['paid_calls'] == 0
    assert before == {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    payload = next((out/'inputs').glob('*.json')).read_text()
    assert all(s not in payload for s in ['SECRET_SELECTOR_ID', 'PRIVATE_ACTOR_THOUGHT', 'OLD_JUDGE_VERDICT'])
    assert all(not (p.stat().st_mode & 0o222) for p in out.rglob('*') if p.is_file())
    with pytest.raises(FileExistsError): audit.prepare([root], out, expected_per_arm=1)


def test_invalid_records_have_no_judge_payload(tmp_path, episode):
    root, out = tmp_path/'cohort', tmp_path/'prepared'
    save_cohort(root, episode, invalid=True)
    summary = audit.prepare([root], out, expected_per_arm=1)
    assert summary['counts'] == {'excluded_original_invalid': 1}
    assert not (out/'inputs').exists()


def test_input_overflow_excluded_without_truncation(tmp_path, episode, monkeypatch):
    root, out = tmp_path/'cohort', tmp_path/'prepared'
    save_cohort(root, episode)
    monkeypatch.setattr(audit, 'MAX_INPUT_TOKEN_BOUND', 10)
    summary = audit.prepare([root], out, expected_per_arm=1)
    assert summary['counts'] == {'excluded_input_bound_overflow': 1}
    assert not (out/'inputs').exists()


def test_preparation_rejects_output_inside_canonical_tree(tmp_path):
    with pytest.raises(ValueError): audit.prepare([tmp_path/'source'], tmp_path/'source'/'out')


def test_derivation_hashes_parent_preserves_evidence_adds_temporal_rule(tmp_path, episode):
    root, out, child = tmp_path/'cohort', tmp_path/'prepared', tmp_path/'derived'
    save_cohort(root, episode)
    audit.prepare([root], out, expected_per_arm=1)
    result = audit.derive_with_current_rubric(out, child)
    before = json.loads(next((out/'inputs').glob('*.json')).read_text())
    after = json.loads(next((child/'inputs').glob('*.json')).read_text())
    assert before['messages'][1] == after['messages'][1]
    assert 'do not use the current' in after['messages'][0]['content']
    assert result['derivation']['all_parent_prepared_file_hashes_verified']
    assert result['counts'] == {'ready': 1}
