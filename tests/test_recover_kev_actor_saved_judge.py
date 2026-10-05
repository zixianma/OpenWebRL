"""Judge-only recovery must not reroll actors or erase ambiguous paid calls."""
from copy import deepcopy
import json
from pathlib import Path

import httpx
import pytest
from test_kev_actor_final_audit import saved, SOURCE, write, jsonl
from scripts import kev_actor_final_audit as audit
from scripts import recover_kev_actor_saved_judge as recovery
from openwebrl.jev_eval import ModelTransport


@pytest.fixture
def pending(saved):
    root, task, cfg, directory, result, state = saved
    (directory/'result.json').rename(root/'old-result.json')
    (directory/'judge-response.json').rename(root/'old-judge.json')
    cfg.update(jev_model='kev-latest', decision_provider='kev', judge_base_url='https://example.com/v1')
    cfg['budget'] = dict(root=str(root/'budget'), limits={'judge_http_attempts': 1320})
    trajectory = audit.read(directory/'trajectory.json'); trajectory['actor_seconds'] = 8.5
    write(directory/'trajectory.json', trajectory)
    attempts = audit.lines(directory/'api-attempts.jsonl')
    for row in attempts: row.update(started_unix=100.0, attempt=1)
    jsonl(directory/'api-attempts.jsonl', attempts)
    jsonl(directory/'api-responses.jsonl', audit.lines(directory/'api-responses.jsonl')[:1])
    source = root/'frozen'; judge = source/'openwebrl/eval/reward_online_mind2web.py'
    judge.parent.mkdir(parents=True); judge.write_bytes(SOURCE.read_bytes())
    plan = dict(tasks=[task], source=str(source), source_sha256={'openwebrl/eval/reward_online_mind2web.py': audit.sha(judge)},
                resources={'total_seconds': 14400}, limits={'judge_http_attempts': 1320}, budget=cfg['budget'], protocol=cfg)
    write(root/'plan.json', plan)
    approval = dict(approved=True, plan_sha256=audit.sha(root/'plan.json'), resources=plan['resources'], limits=plan['limits'],
                    attempts=[dict(job_id='123', submitted_unix=90.0)])
    write(root/'approval.json', approval)
    write(root/'budget/usage.json', dict(limits=cfg['budget']['limits'], reserved={'judge_http_attempts': 1}, sequence=1))
    return saved


def scheduler(job):
    return dict(job_id=job, state='FAILED', scheduler_seconds=20, start_unix=90.0, end_unix=110.0)


def prepare(pending, **kw):
    root, task, *_ = pending
    return recovery.prepare(root, task['task_id'], '123', scheduler_fn=scheduler, **kw)


def transport(directory, cfg):
    def respond(request):
        assert json.loads(request.content) == audit.read(directory/'judge-request.json')
        assert request.headers['authorization'] == 'Bearer fixture-key'
        return httpx.Response(200, json=dict(model='o4-mini-2025-04-16', choices=[{'message': {'content': 'Status: failure'}}]))
    return ModelTransport(directory, cfg, client=httpx.Client(transport=httpx.MockTransport(respond)))


def execute(pending):
    directory, intent = prepare(pending)
    return recovery.execute_prepared(pending[0], directory, intent, 'fixture-key', transport_factory=transport)


def audit_result(pending):
    root, task, cfg, *_ = pending
    return audit.audit_task(root, task, cfg, Path(audit.read(root/'plan.json')['source'])/'openwebrl/eval/reward_online_mind2web.py')


def test_dry_run_is_read_only_and_retains_original_ambiguous_call(pending):
    root, _, _, directory, *_ = pending
    before = {str(p): audit.sha(p) for p in root.rglob('*') if p.is_file()}
    _, intent = prepare(pending)
    assert intent['original_counts'] == {'kev': 1, 'text': 0, 'judge': 1}
    assert intent['max_additional_calls'] == 1 and intent['missing_call_id'] == 'judge-0001'
    assert before == {str(p): audit.sha(p) for p in root.rglob('*') if p.is_file()}


def test_exact_saved_judge_recovery_is_charged_auditable_and_preserves_actor(pending):
    root, _, _, directory, *_ = pending
    preserved = {name: audit.sha(directory/name) for name in ('trajectory.json', 'final.jpg', 'judge-request.json', 'state-0061.json.gz')}
    result = execute(pending)
    assert result['score'] == 0 and result['valid'] and result['actor_seconds'] == 8.5
    assert result['api_attempts'] == {'kev': 1, 'text': 0, 'judge': 2}
    assert audit.read(root/'budget/usage.json')['reserved']['judge_http_attempts'] == 2
    assert [r['call_id'] for r in audit.lines(directory/'api-responses.jsonl')] == ['kev-0001', 'judge-0002']
    assert preserved == {name: audit.sha(directory/name) for name in preserved}
    row = audit_result(pending)
    assert row['status'] == 'evidence_verified'
    assert any(n.endswith('saved-judge-recovery/intent.json') for n in row['artifact_sha256'])
    assert any(n.endswith('saved-judge-recovery/completion.json') for n in row['artifact_sha256'])


@pytest.mark.parametrize('what,match', [('active', 'not terminal'), ('completed', 'failed job'), ('global', 'Global judge'),
                                       ('browser', 'remains active'), ('request', 'Canonical judge'), ('actor_missing', 'Decision lacks'),
                                       ('result', 'replace a result'), ('verdict', 'replace a result')])
def test_preflight_rejects_unsafe_recovery(pending, what, match):
    root, task, cfg, directory, *_ = pending
    custom = scheduler
    if what in {'active', 'completed'}:
        custom = lambda job: {**scheduler(job), 'state': 'RUNNING' if what == 'active' else 'COMPLETED'}
    elif what == 'global':
        usage = audit.read(root/'budget/usage.json'); usage['reserved']['judge_http_attempts'] = 1320; write(root/'budget/usage.json', usage)
    elif what == 'browser': write(directory/'browser-session.json', {'id': 'fixture', 'stopped': False})
    elif what == 'request':
        request = audit.read(directory/'judge-request.json'); request['seed'] = 9; write(directory/'judge-request.json', request)
    elif what == 'actor_missing': jsonl(directory/'api-responses.jsonl', [])
    elif what == 'result': write(directory/'result.json', {})
    elif what == 'verdict': write(directory/'judge-response.json', {})
    with pytest.raises(ValueError, match=match):
        recovery.prepare(root, task['task_id'], '123', scheduler_fn=custom)


@pytest.mark.parametrize('calls', [0, 4, -1])
def test_per_task_cap_cannot_be_reset(pending, calls):
    with pytest.raises(ValueError, match='remaining per-task'):
        prepare(pending, max_additional_calls=calls)


def test_successful_result_cannot_be_rerolled(pending):
    execute(pending)
    with pytest.raises(ValueError, match='Preserve existing recovery'): prepare(pending)


@pytest.mark.parametrize('tamper,match', [('prefix', 'prefix changed'), ('proof', 'evidence changed'),
                                        ('source', 'frozen source changed'), ('extra', 'additional unanswered'),
                                        ('completion', 'completion hash'), ('intent', 'completion hash')])
def test_audit_exemption_is_exact_and_hash_bound(pending, tamper, match):
    root, _, _, directory, *_ = pending
    execute(pending)
    if tamper == 'prefix':
        p = directory/'api-attempts.jsonl'; raw = p.read_text(); p.write_text(raw.replace('100.0', '101.0', 1))
    elif tamper == 'proof': (directory/'worker.log').write_text('changed')
    elif tamper == 'source':
        source = Path(audit.read(root/'plan.json')['source'])/'openwebrl/eval/reward_online_mind2web.py'
        source.write_text(source.read_text()+'\n# changed\n')
    elif tamper == 'extra':
        rows = audit.lines(directory/'api-attempts.jsonl'); extra = deepcopy(rows[-1]); extra['call_id'] = 'judge-0003'; rows.append(extra)
        jsonl(directory/'api-attempts.jsonl', rows)
        result = audit.read(directory/'result.json'); result['api_attempts']['judge'] = 3; write(directory/'result.json', result)
        completion = audit.read(directory/'saved-judge-recovery/completion.json'); completion['result_digest'] = audit.digest(result)
        write(directory/'saved-judge-recovery/completion.json', completion)
    elif tamper == 'completion':
        p = directory/'saved-judge-recovery/completion.json'; value = audit.read(p); value['result_digest'] = 'wrong'; write(p, value)
    elif tamper == 'intent':
        p = directory/'saved-judge-recovery/intent.json'; value = audit.read(p); value['max_additional_calls'] = 2; write(p, value)
    with pytest.raises(ValueError, match=match): audit_result(pending)


def test_transport_failure_preserves_intent_and_never_creates_result(pending):
    def failing(directory, cfg):
        def fail(request): raise httpx.ReadTimeout('fixture')
        return ModelTransport(directory, cfg, client=httpx.Client(transport=httpx.MockTransport(fail)))
    directory, intent = prepare(pending)
    with pytest.raises(Exception, match='judge HTTP transport'):
        recovery.execute_prepared(pending[0], directory, intent, 'fixture-key', transport_factory=failing)
    assert (directory/'saved-judge-recovery/intent.json').exists()
    assert (directory/'saved-judge-recovery/failure.json').exists()
    assert not (directory/'result.json').exists()
    assert len(audit.lines(directory/'api-attempts.jsonl')) == 3
    assert audit.read(pending[0]/'budget/usage.json')['reserved']['judge_http_attempts'] == 2
    with pytest.raises(ValueError, match='Preserve existing recovery'): prepare(pending)


def test_recovered_result_cannot_backfill_original_ambiguous_response(pending):
    execute(pending)
    directory = pending[3]
    responses = audit.lines(directory/'api-responses.jsonl')
    fabricated = deepcopy(responses[-1]); fabricated['call_id'] = 'judge-0001'; responses.append(fabricated)
    jsonl(directory/'api-responses.jsonl', responses)
    with pytest.raises(ValueError, match='additional unanswered'):
        audit_result(pending)


def test_api_prefix_cannot_be_redirected_to_other_file(pending):
    execute(pending)
    directory = pending[3]
    path = directory/'saved-judge-recovery/intent.json'; intent = audit.read(path)
    intent['original_api_attempts']['name'] = '../api-attempts.jsonl'; write(path, intent)
    with pytest.raises(ValueError, match='prefix names'):
        audit_result(pending)
