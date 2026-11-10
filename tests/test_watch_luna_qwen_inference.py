"""Read-only scheduler observation must survive expired squeue job IDs."""
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import watch_luna_qwen_inference as watcher


JOB = '344655'


def reply(stdout='', returncode=0, stderr=''):
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


def mock_queries(monkeypatch, *responses):
    run = Mock(side_effect=responses)
    monkeypatch.setattr(watcher.subprocess, 'run', run)
    return run


def test_running_job_uses_squeue_without_accounting(monkeypatch):
    raw = f'{JOB}|RUNNING|03:12|g020'
    run = mock_queries(monkeypatch, reply(raw + '\n'))
    assert watcher.scheduler(JOB) == dict(job_id=JOB, state='RUNNING', raw=raw)
    run.assert_called_once()
    assert run.call_args.args[0][0] == 'squeue'


@pytest.mark.parametrize('live', [
    reply(),
    reply(returncode=1, stderr='slurm_load_jobs error: Invalid job id specified'),
    reply(f'{JOB}|RUNNING|03:12|g020', returncode=1, stderr='query failed'),
    subprocess.TimeoutExpired('squeue', 20),
    FileNotFoundError('squeue unavailable'),
])
def test_missing_or_failed_squeue_falls_back_to_terminal_accounting(monkeypatch, live):
    raw = f'{JOB}|FAILED|00:44:06|1:0'
    run = mock_queries(monkeypatch, live, reply(raw))
    assert watcher.scheduler(JOB) == dict(job_id=JOB, state='FAILED', raw=raw)
    assert [call.args[0][0] for call in run.call_args_list] == ['squeue', 'sacct']
    assert all(call.kwargs['timeout'] == 20 for call in run.call_args_list)


def test_accounting_cancelled_by_user_and_trailing_delimiter(monkeypatch):
    raw = f'{JOB}|CANCELLED by 1385309|00:02:30|0:0|'
    mock_queries(monkeypatch, reply(), reply(raw))
    assert watcher.scheduler(JOB)['state'] == 'CANCELLED'


@pytest.mark.parametrize('live', [
    reply(f'{JOB}|UNKNOWN|00:01|None'),
    reply(f'{JOB}|RUNNING'),
    reply('999999|RUNNING|00:01|g020'),
])
def test_unusable_live_row_can_be_resolved_by_accounting(monkeypatch, live):
    mock_queries(monkeypatch, live, reply(f'{JOB}|COMPLETED|00:44:06|0:0'))
    assert watcher.scheduler(JOB)['state'] == 'COMPLETED'


@pytest.mark.parametrize('accounting', [
    reply(),
    reply(f'{JOB}|UNKNOWN|00:00:00|0:0'),
    reply(f'{JOB}|MYSTERY|00:00:00|0:0'),
    reply(f'{JOB}'),
    reply(f'{JOB}|FAILED'),
    reply(f'{JOB}||00:44:06|1:0'),
    reply(f'{JOB}|FAILED||1:0'),
    reply('999999|FAILED|00:44:06|1:0'),
    reply(f'{JOB}|FAILED|00:44:06|1:0\n{JOB}|RUNNING|00:01|g020'),
    reply(returncode=1, stderr='accounting database unavailable'),
    subprocess.TimeoutExpired('sacct', 20),
    FileNotFoundError('sacct unavailable'),
])
def test_both_queries_unable_to_establish_state_fail_visibly(monkeypatch, accounting):
    mock_queries(monkeypatch, reply(returncode=1, stderr='Invalid job id'), accounting)
    with pytest.raises(RuntimeError, match=f'Cannot establish scheduler state for registered job {JOB}; squeue:.*sacct:'):
        watcher.scheduler(JOB)


def test_invalid_job_refused_before_query(monkeypatch):
    run = mock_queries(monkeypatch)
    with pytest.raises(ValueError, match='Invalid registered'):
        watcher.scheduler('344655; other-command')
    run.assert_not_called()


def test_snapshot_records_terminal_state_after_squeue_expiry(tmp_path, monkeypatch):
    monkeypatch.setattr(watcher, 'ROOT', tmp_path)
    (tmp_path / 'current.json').write_text(json.dumps(dict(job_id=JOB)))
    (tmp_path / 'approval.json').write_text(json.dumps(dict(attempts=[dict(job_id=JOB)])))
    raw = f'{JOB}|FAILED|00:44:06|1:0'
    mock_queries(monkeypatch, reply(returncode=1, stderr='Invalid job id'), reply(raw))
    snapshot = watcher.snapshot()
    assert snapshot['job_id'] == JOB and snapshot['state'] == 'FAILED'
    assert snapshot['scheduler'] == raw
    assert snapshot['completed'] == 0 and snapshot['expected_episodes'] == 40


def blocked_snapshot():
    return dict(job_id=JOB, state='FAILED', api_halts={}, sft_extension={},
                controller=dict(stage='awaiting_recovery_approval_and_luna_repair'),
                context_repair={}, recovery_approval_sha256=None)


@pytest.mark.parametrize('state,stage,expected', [
    ('FAILED', 'awaiting_recovery_approval_and_luna_repair', 3600),
    ('FAILED', 'awaiting_resource_profile_approval', 3600),
    ('RUNNING', 'awaiting_recovery_approval_and_luna_repair', 900),
    ('FAILED', 'failed_or_interrupted', 900),
    ('UNKNOWN', 'awaiting_resource_profile_approval', 900),
])
def test_only_known_terminal_approval_blocks_get_hourly_reviews(state, stage, expected):
    latest = blocked_snapshot()
    latest.update(state=state, controller=dict(stage=stage))
    assert watcher.review_interval(latest) == expected


def test_unchanged_block_waits_an_hour_but_new_approval_dispatches_immediately():
    latest = blocked_snapshot()
    state = dict(last_dispatched_state=watcher.dispatch_state(latest), queued_epoch=50)
    review = dict(epoch=100)
    assert not watcher.continuation_due(latest, state, review, 1000)
    assert watcher.continuation_due(latest, state, review, 3700)
    latest['recovery_approval_sha256'] = 'changed-approval-content'
    assert watcher.continuation_due(latest, state, review, 101)


def test_resumed_worker_and_new_api_failure_dispatch_without_waiting():
    latest = blocked_snapshot()
    state = dict(last_dispatched_state=watcher.dispatch_state(latest))
    review = dict(epoch=100)
    latest['state'] = 'RUNNING'
    assert watcher.continuation_due(latest, state, review, 101)
    latest['state'] = 'FAILED'
    latest['api_halts'] = {'luna': {'reason': 'new halt'}}
    assert watcher.continuation_due(latest, state, review, 101)
