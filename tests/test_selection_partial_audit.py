"""A timeout may save the next observation before any corresponding selection."""
import importlib.util
import base64
import io
import json
from pathlib import Path

import pytest
from PIL import Image

spec = importlib.util.spec_from_file_location(
    'selection_audit', Path(__file__).resolve().parents[1] / 'scripts/audit_sft_decision_selection.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def png():
    out = io.BytesIO()
    Image.new('RGB', (2, 2)).save(out, format='PNG')
    return out.getvalue()


def test_diagnosed_next_observation_is_separate_from_selected_states():
    raw = png()
    states, hashes = audit.align_partial_screenshots(
        [raw, raw], 1, {'unselected_observation_sha256': [audit.digest(raw)]})
    assert states == [raw] and hashes == [audit.digest(raw)]
    assert audit.align_partial_screenshots([raw], 1, {}) == ([raw], [])


@pytest.mark.parametrize('count,extra,diagnosis', [
    (1, 1, {}),
    (1, 1, {'unselected_observation_sha256': ['wrong']}),
    (1, 2, {}),
    (2, 0, {}),
])
def test_unexplained_extra_or_missing_states_fail(count, extra, diagnosis):
    with pytest.raises(AssertionError):
        audit.align_partial_screenshots([png()] * (1 + extra), count, diagnosis)


def test_pre_action_abort_requires_empty_response_and_matching_initial_image():
    raw = png()
    sample = dict(sample_id='task', status='aborted', total_steps=0, llm_response='',
        terminate_reason='generation_error: Selector halted or request cap reached',
        images=[[base64.b64encode(raw).decode()]])
    assert audit.verify_pre_action_abort(sample, [], 'task', raw) == [2, 2]
    for changed, decisions, image in [
        (dict(sample, llm_response='action'), [], raw),
        (dict(sample, total_steps=1), [], raw),
        (sample, [{}], raw),
        (sample, [], b'different'),
    ]:
        with pytest.raises(AssertionError):
            audit.verify_pre_action_abort(changed, decisions, 'task', image)


@pytest.fixture
def setup_failure(tmp_path):
    folder = tmp_path / 'kev-27b'
    task_id, url = 'task', 'https://dblp.org/'
    key = audit.digest(task_id.encode())
    session = '4089dd32-216c-423f-baaa-baac4b46da16'
    error = (f'Failed to navigate to {url} after 3 attempts: '
        f'Page.goto: net::ERR_CONNECTION_CLOSED at {url}\nCall log:\n'
        f'  - navigating to "{url}", waiting until "domcontentloaded"\n')
    result = dict(task_id=task_id, start_url=url, status='Status.ABORTED', valid=False,
        reward=None, total_steps=0, response='', terminate_reason='generation_error: ' + error,
        metadata=dict(task_id=task_id, start_url=url, terminate_reason='generation_error: ' + error))
    result_path = folder / 'results' / (key + '.json')
    started = folder / 'started' / (key + '.json')
    stopped = folder / 'browser_sessions' / 'stopped' / session
    for path in (result_path, started, stopped):
        path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result))
    started.write_text(json.dumps(dict(task_id=task_id, started_unix=1)))
    stopped.touch()
    excerpt = tmp_path / 'setup-failure.log'
    excerpt.write_text(f'Task {task_id}: generate_turn_sample failed: {error}\n'
        f'OSError: {error}\n[Stopped Browser-Use session {session} after failure]\n')
    diagnosis = dict(initial_navigation_failure=True,
        classification='initial browser navigation failed before any observation or inference',
        error_type='OSError', original_result_sha256=audit.digest(result_path.read_bytes()),
        saved_reward=None, saved_final_image=False, terminate_reason=result['terminate_reason'],
        start_receipt=str(started), start_receipt_sha256=audit.digest(started.read_bytes()),
        browser_session_id=session, browser_stop_receipt=str(stopped), log_excerpt=str(excerpt),
        log_excerpt_sha256=audit.digest(excerpt.read_bytes()))
    return folder, key, result_path, diagnosis, task_id, url, []


def test_initial_navigation_failure_needs_no_fabricated_rollout(setup_failure):
    audit.verify_initial_navigation_failure(*setup_failure)
    folder, key, *_ = setup_failure
    assert not (folder / 'samples' / key).exists()


@pytest.mark.parametrize('artifact', ['trace', 'archive', 'final', 'judge', 'active_browser'])
def test_initial_navigation_failure_rejects_any_inference_or_terminal_artifacts(setup_failure, artifact):
    folder, key, _, diagnosis, _, _, decisions = setup_failure
    if artifact == 'trace':
        decisions.append(dict(status='selected'))
    else:
        paths = {'archive': folder / 'samples' / key / 'rollout.json',
            'final': folder / 'final' / (key + '.png'),
            'judge': folder / 'judge' / key / 'request-01.json',
            'active_browser': folder / 'browser_sessions' / diagnosis['browser_session_id']}
        path = paths[artifact]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'evidence')
    with pytest.raises(AssertionError):
        audit.verify_initial_navigation_failure(*setup_failure)


@pytest.mark.parametrize('change', ['task', 'url', 'response', 'steps', 'reward', 'error', 'messages', 'image'])
def test_initial_navigation_failure_rejects_misclassified_results_even_with_matching_hash(setup_failure, change):
    _, _, path, diagnosis, *_ = setup_failure
    result = json.loads(path.read_text())
    updates = {'task': ('task_id', 'other'), 'url': ('start_url', 'https://other.test/'),
        'response': ('response', 'action'), 'steps': ('total_steps', 1),
        'reward': ('reward', 0), 'error': ('terminate_reason', 'generation_error: unrelated')}
    if change in ('messages', 'image'):
        result['metadata']['messages' if change == 'messages' else 'full_image_list'] = ['evidence']
    else:
        name, value = updates[change]
        result[name] = value
    path.write_text(json.dumps(result))
    diagnosis['original_result_sha256'] = audit.digest(path.read_bytes())
    with pytest.raises(AssertionError):
        audit.verify_initial_navigation_failure(*setup_failure)


@pytest.mark.parametrize('change', ['result_hash', 'start_hash', 'log_hash', 'missing_stop_evidence'])
def test_initial_navigation_failure_requires_immutable_setup_and_shutdown_evidence(setup_failure, change):
    diagnosis = setup_failure[3]
    if change == 'missing_stop_evidence':
        path = Path(diagnosis['log_excerpt'])
        path.write_text(path.read_text().split('[Stopped Browser-Use session')[0])
        diagnosis['log_excerpt_sha256'] = audit.digest(path.read_bytes())
    else:
        name = {'result_hash': 'original_result_sha256', 'start_hash': 'start_receipt_sha256',
            'log_hash': 'log_excerpt_sha256'}[change]
        diagnosis[name] = 'wrong'
    with pytest.raises(AssertionError):
        audit.verify_initial_navigation_failure(*setup_failure)


@pytest.fixture
def observation_failure(setup_failure):
    folder, key, path, diagnosis, task_id, url, decisions = setup_failure
    result = json.loads(path.read_text())
    error = ('Page.evaluate: Error: eval is disabled\n'
        '    at monkeypatchedEval (https://example.test/app.js:2:1)\n'
        '    at UtilityScript.evaluate (<anonymous>:290:30)')
    result['terminate_reason'] = result['metadata']['terminate_reason'] = 'generation_error: ' + error
    path.write_text(json.dumps(result))
    diagnosis.update(initial_observation_failure=True, terminate_reason=result['terminate_reason'],
        original_result_sha256=audit.digest(path.read_bytes()))
    excerpt = Path(diagnosis['log_excerpt'])
    excerpt.write_text(f'Task {task_id}: generate_turn_sample failed: {error}\n'
        'in wrapped_reset\nawait enrich(self, observation)\nin enrich\n'
        'data = await env.page.evaluate\n'
        f'playwright._impl._errors.Error: {error}\n'
        f"[Stopped Browser-Use session {diagnosis['browser_session_id']}]\n")
    diagnosis['log_excerpt_sha256'] = audit.digest(excerpt.read_bytes())
    ledger = folder.parent / 'budget/reservations.jsonl'
    ledger.parent.mkdir()
    ledger.write_text(json.dumps(dict(task_id=task_id, kind='browser_sessions', count=1)) + '\n')
    return setup_failure


def test_initial_observation_failure_verified_without_fabricated_image(observation_failure):
    audit.verify_initial_observation_failure(*observation_failure)


@pytest.mark.parametrize('change', ['inference', 'trace', 'archive', 'final', 'judge', 'active',
    'messages', 'image', 'reward', 'error', 'log', 'stop', 'result_hash', 'start_hash'])
def test_initial_observation_failure_is_not_a_generic_missing_evidence_bypass(observation_failure, change):
    folder, key, path, diagnosis, task_id, _, decisions = observation_failure
    if change == 'inference':
        with (folder.parent / 'budget/reservations.jsonl').open('a') as out:
            out.write(json.dumps(dict(task_id=task_id, kind='actor_proposals', count=5)) + '\n')
    elif change == 'trace':
        decisions.append({})
    elif change in ('archive', 'final', 'judge', 'active'):
        artifact = {'archive': folder / 'samples' / key / 'rollout.json',
            'final': folder / 'final' / (key + '.png'),
            'judge': folder / 'judge' / key / 'request-1.json',
            'active': folder / 'browser_sessions' / diagnosis['browser_session_id']}[change]
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text('evidence')
    elif change in ('messages', 'image', 'reward', 'error'):
        result = json.loads(path.read_text())
        if change in ('messages', 'image'):
            result['metadata']['messages' if change == 'messages' else 'full_image_list'] = ['evidence']
        else:
            result['reward' if change == 'reward' else 'terminate_reason'] = 0 if change == 'reward' else 'other'
        path.write_text(json.dumps(result))
        diagnosis['original_result_sha256'] = audit.digest(path.read_bytes())
    elif change in ('log', 'stop'):
        excerpt = Path(diagnosis['log_excerpt'])
        value = excerpt.read_text().replace('in enrich', 'elsewhere') if change == 'log' else excerpt.read_text().split('[Stopped')[0]
        excerpt.write_text(value)
        diagnosis['log_excerpt_sha256'] = audit.digest(excerpt.read_bytes())
    else:
        diagnosis['original_result_sha256' if change == 'result_hash' else 'start_receipt_sha256'] = 'wrong'
    with pytest.raises(AssertionError):
        audit.verify_initial_observation_failure(*observation_failure)


@pytest.fixture
def format_failure():
    result = dict(valid=True, status='Status.FAILED', reward=0, terminate_reason='format_error_failed',
        metadata=dict(terminate_reason='format_error_failed', reward=dict(judge=0., combined=0.,
            judge_text='Judge not run for status=Status.FAILED', judge_timeout=False, protocol='online_mind2web')))
    assistants = ['<think>reason</think><tool_call>{"name":"click","arguments":{"point_200, 787]}}</tool_call>'] * 3
    sample = dict(status='failed', terminate_reason='format_error_failed')
    return result, assistants, sample, []


def test_deterministic_format_failure_preserves_zero_without_judge(format_failure):
    audit.verify_deterministic_format_failure(*format_failure)


@pytest.mark.parametrize('change', ['valid_json', 'missing_call', 'two_errors', 'success', 'status', 'judge', 'sample', 'metadata'])
def test_deterministic_format_failure_requires_three_malformed_outputs(format_failure, change):
    result, assistants, sample, requests = format_failure
    if change == 'valid_json':
        assistants[-1] = '<tool_call>{"name":"click","arguments":{"point_2d":[200,787]}}</tool_call>'
    elif change == 'missing_call':
        assistants[-1] = 'text'
    elif change == 'two_errors':
        assistants.pop()
    elif change == 'success':
        result['reward'] = 1
    elif change == 'status':
        result['status'] = 'Status.COMPLETED'
    elif change == 'judge':
        requests.append('request.json')
    elif change == 'sample':
        sample['status'] = 'completed'
    else:
        result['metadata']['reward']['judge_text'] = 'success'
    with pytest.raises(AssertionError):
        audit.verify_deterministic_format_failure(*format_failure)
