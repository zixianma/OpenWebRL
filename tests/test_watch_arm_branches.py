from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from watch_arm_branches import dispatch_due


def signature(**changes):
    return dict(dict(job='1',state='RUNNING',stage='discovering_prefixes',halt=None,stale=False,gate={}),**changes)


def test_ordinary_stages_wait_fifteen_minutes_after_ack():
    old=dict(epoch=100,signature=signature(stage='model_startup'))
    for stage in ('discovering_prefixes','reconstructing_and_continuing','continuations','teacher_closeout'):
        assert not dispatch_due(signature(stage=stage),old,110,1009,False)
        assert dispatch_due(signature(stage=stage),old,110,1010,False)


def test_failures_and_stalls_bypass_pending_routine_callback_without_duplicates():
    old=dict(epoch=100,signature=signature())
    for change in (dict(state='FAILED'),dict(stale=True),dict(halt={'reason':'usage'})):
        failure=signature(**change)
        assert dispatch_due(failure,old,90,101,True)
        queued=dict(epoch=101,signature=failure)
        assert not dispatch_due(failure,queued,90,10000,True)
        assert not dispatch_due(failure,queued,102,103,True)


def test_new_job_and_closeout_trigger_after_ack():
    old=dict(epoch=100,signature=signature())
    assert dispatch_due(signature(job='2'),old,110,111,False)
    assert dispatch_due(signature(stage='collected_pending_audit'),old,110,111,False)
    assert not dispatch_due(signature(),old,90,10000,False)


def test_callback_uses_approved_cohort_size(tmp_path, monkeypatch):
    import types
    import watch_arm_branches as watcher
    messages = []
    monkeypatch.setattr(watcher, 'scheduler_records', lambda jobs: {'1': {'state': 'PENDING'}})
    monkeypatch.setattr(watcher, 'database_snapshot', lambda root: {})
    monkeypatch.setattr(watcher, 'write', lambda path, value: None)
    def queue(command, **kwargs):
        messages.append(command[command.index('--message') + 1])
        return types.SimpleNamespace(returncode=0, stdout='queued', stderr='')
    monkeypatch.setattr(watcher.subprocess, 'run', queue)
    for target in (100, 149):
        plan = dict(states_target=target, candidate_count=5, continuation_repetitions=3,
                    resources={}, limits={})
        approval = dict(approved=True, plan_sha256=watcher.digest(plan), resources={},
                        limits={}, attempts=[{'job_id': '1'}])
        files = {'plan.json': plan, 'approval.json': approval, 'current.json': {'job_id': '1'}}
        monkeypatch.setattr(watcher, 'read', lambda path: files.get(path.name, {}))
        assert watcher.poll(tmp_path, 'owner-thread', 'test-codex')
        assert f'Verify{target * 15}continuations/{target}states' in messages[-1]
