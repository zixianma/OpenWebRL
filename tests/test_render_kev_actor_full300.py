"""Private reviews must not upgrade changed evidence or invent clicked targets."""
import base64
from copy import deepcopy
import gzip
import io
import json

from PIL import Image
import pytest
from scripts import render_kev_actor_full300 as review


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def snapshot(path, value):
    path.write_bytes(gzip.compress(json.dumps(value).encode()))


def sign(root, directory, row):
    row['artifact_sha256'] = {str(p.relative_to(root)): review.sha(p.read_bytes())
                              for p in directory.iterdir() if p.is_file()}


@pytest.fixture
def saved(tmp_path):
    root = tmp_path / 'runtime'
    tasks = [dict(task_id=f'task-{i}', intent='Open result', start_url='https://example.org/') for i in range(300)]
    plan = dict(tasks=tasks, protocol={'max_decisions': 60}, resources={}, limits={})
    write(root / 'plan.json', plan)
    directory = root / 'actor/tasks' / review.task_key(tasks[0]['task_id'])
    directory.mkdir(parents=True)
    image = io.BytesIO()
    Image.new('RGB', (4, 4), 'blue').save(image, 'JPEG')
    picture = base64.b64encode(image.getvalue()).decode()
    decision = dict(choice='e1', operation='CLICK', target='1', fingerprint='before',
                    confidence=.9, latency_ms=123, usage={'input_tokens': 20},
                    operation_probabilities={'CLICK': .9, 'DONE': .1}, target_probabilities={'1': 1.0},
                    request={'questions': {'click_target': {'criteria': {'1': {'element': 'Open result'}}}}})
    history = [{k: decision[k] for k in ('choice', 'operation', 'target', 'confidence', 'latency_ms', 'usage')}]
    terminal = dict(choice='done', operation='DONE', target=None, confidence=.8, latency_ms=100,
                    usage={'input_tokens': 15}, operation_probabilities={'DONE': .8, 'CLICK': .2},
                    target_probabilities={}, request={'questions': {}})
    trajectory = dict(task=tasks[0], terminal='done', decisions=[decision, terminal], history=history, text_calls=[])
    result = dict(task_id=tasks[0]['task_id'], score=0, valid=True, judge_text='Status: failure')
    page = dict(url=tasks[0]['start_url'], title='Example', text='Open result', screenshot=picture,
                fingerprint='before', w=100, h=80,
                actions=[dict(id='e1', node=17, label='Open result', rect=dict(x=1, y=2, w=10, h=12))])
    initial = dict(page=page, decisions=[], history=[], text_calls=[], status='running')
    after = deepcopy(initial)
    after.update(decisions=[decision], history=history)
    after['page'].update(fingerprint='after', actions=[dict(id='e1', node=99, label='Different control', rect=dict(x=50, y=20, w=10, h=12))])
    final = deepcopy(after)
    final.update(decisions=[decision, terminal], status='done', final_screenshot_fresh=True)
    for name, value in [('task.json', tasks[0]), ('result.json', result), ('trajectory.json', trajectory)]:
        write(directory / name, value)
    for name, value in [('state-0000.json.gz', initial), ('state-0001.json.gz', after), ('state-0061.json.gz', final)]:
        snapshot(directory / name, value)
    (directory / 'final.jpg').write_bytes(image.getvalue())
    row = dict(task_id=tasks[0]['task_id'], status='evidence_verified', score=0, valid=True)
    sign(root, directory, row)
    audit = dict(audit_version=1, planned=300, plan_sha256=review.sha((root / 'plan.json').read_bytes()),
                 per_task=[row], issues=[], scheduler_closed=False)
    audit_path = root / 'audit.json'
    write(audit_path, audit)
    write(root / 'actor/tasks' / review.task_key(tasks[1]['task_id']) / 'result.json', {'score': 1})
    (root / 'actor/tasks' / review.task_key(tasks[2]['task_id'])).mkdir()
    assets = tmp_path / 'review.assets'
    assets.mkdir()
    return root, audit_path, assets, directory, audit, result, final


def build(saved):
    return review.build(*saved[:3])


def test_full_plan_separates_audited_pending_and_unaudited(saved):
    data = build(saved)
    assert len(data['tasks']) == 300
    assert data['statuses'] == {'failure': 1, 'unaudited': 1, 'running': 1, 'pending': 297}
    assert data['tasks'][1]['result'] is None  # An unaudited positive cannot inflate the score.


def test_target_uses_exact_predecision_state_and_not_reused_dom_id(saved):
    data = build(saved)
    event = data['tasks'][0]['frames'][1]['decisions'][0]
    mapping = event['target_mapping']
    assert (mapping['model_target'], mapping['harness_action_id'], mapping['cached_dom_node']) == ('1', 'e1', 17)
    assert mapping['observed_rect']['x'] == 1
    assert mapping['actual_execution_coordinates_recorded'] is False
    assert event['execution_status'] == 'recorded_executed'
    assert data['tasks'][0]['frames'][-1]['decisions'][0]['execution_status'] == 'terminal_signal'
    assert len(data['images']) == 1
    assert all((saved[2].parent / path).is_file() for path in data['images'].values())


def test_missing_matching_state_does_not_infer_geometry(saved):
    root, audit_path, _, directory, audit, _, _ = saved
    key = str((directory / 'state-0000.json.gz').relative_to(root))
    del audit['per_task'][0]['artifact_sha256'][key]
    write(audit_path, audit)
    mapping = build(saved)['tasks'][0]['frames'][0]['decisions'][0]['target_mapping']
    assert mapping['harness_action_id'] == 'e1' and 'cached_dom_node' not in mapping
    assert 'No matching' in mapping['geometry_status']


def test_changed_artifact_is_unaudited_and_cannot_contribute_a_score(saved):
    saved[3].joinpath('result.json').write_text('{"score":1}')
    task = build(saved)['tasks'][0]
    assert task['status'] == 'unaudited' and task['result'] is None and task['frames'] == []
    assert 'changed' in task['audit_issues'][0]['error']


def test_artifact_cannot_escape_task_directory(saved):
    audit = saved[4]
    audit['per_task'][0]['artifact_sha256']['plan.json'] = review.sha((saved[0] / 'plan.json').read_bytes())
    write(saved[1], audit)
    task = build(saved)['tasks'][0]
    assert task['status'] == 'unaudited'
    assert 'escapes' in task['audit_issues'][0]['error']


def test_explicit_audit_issue_prevents_verified_badge(saved):
    saved[4]['issues'] = [{'task_id': 'task-0', 'error_type': 'ValueError', 'error': 'Unresolved judge'}]
    write(saved[1], saved[4])
    assert build(saved)['tasks'][0]['status'] == 'unaudited'


def test_diagnosed_invalid_is_unscored_and_caveat_does_not_relabel(saved):
    root, audit_path, _, directory, audit, result, _ = saved
    result.update(valid=False, score=None, judge_text='Unparseable')
    write(directory / 'result.json', result)
    row = audit['per_task'][0]
    row.update(status='diagnosed_invalid', valid=False, score=None, diagnosis='Saved unparseable verdict')
    sign(root, directory, row)
    write(audit_path, audit)
    note = dict(task_id='task-0', note='Looks plausible, but score stays unchanged.', canonical_result_unchanged=True)
    write(root / 'judge-review-notes.json', [note])
    task = build(saved)['tasks'][0]
    assert task['status'] == 'invalid' and task['result']['score'] is None
    assert task['review_notes'] == [note]


def test_final_mismatch_cannot_be_rendered_as_a_verified_fresh_image(saved):
    root, audit_path, _, directory, audit, _, _ = saved
    (directory / 'final.jpg').write_bytes(b'changed final image')
    sign(root, directory, audit['per_task'][0])
    write(audit_path, audit)
    task = build(saved)['tasks'][0]
    assert task['status'] == 'unaudited' and 'Final JPEG differs' in task['audit_issues'][0]['error']


def test_safe_json_cannot_close_script_or_change_its_value():
    value = {'page': '</script><img src=x onerror=alert(1)>\u2028\u2029 & text'}
    encoded = review.safe_json(value)
    assert '<' not in encoded and '\u2028' not in encoded and '\u2029' not in encoded
    assert json.loads(encoded) == value


def test_plan_and_audit_identity_must_match(saved):
    saved[4]['plan_sha256'] = 'different'
    write(saved[1], saved[4])
    with pytest.raises(ValueError, match='plan identity'):
        build(saved)


def test_private_render_uses_external_assets_and_refuses_repository_output(saved, tmp_path):
    with pytest.raises(ValueError, match='outside the repository'):
        review.render(saved[0], saved[1], review.REPO / 'review.html')
    output = tmp_path / 'review.html'
    receipt = review.render(saved[0], saved[1], output)
    assert receipt['tasks'] == 300 and receipt['unique_screenshots'] == 1 and receipt['public'] is False
    html = output.read_text()
    assert '__REVIEW_DATA__' not in html and 'data:image/' not in html
    data = json.loads(html.split('<script id="data" type="application/json">')[1].split('</script>')[0])
    assert all((output.parent / url).exists() for url in data['images'].values())
